"""
PyTorch side of the memory-compact loader (config "dataframe_type": "numpy").
See np_reader.py for the array layout and why it exists.

The dataset returns exactly the same 5-tuple per event as Normalized_Polars_Dataset_with_cache:
    (vector [max_num_prongs, n_vector] float32, scalar [n_scalar] float32,
     mask [max_num_prongs] bool (True = padding), target [n_target] float32, weight [1] float32)
so the trainer, the noise collate function and the models are unchanged.

There is no per-sample cache: building one event from the arrays takes microseconds, and a cache
of all events (~3 kB each) would cost tens of GB.
"""

import gc
import os

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from transformer_ee.dataloader.noise import normalized_noise
from transformer_ee.dataloader.np_reader import (
    load_or_read_arrays,
    normalize_in_place,
    split_indices,
    train_statistics,
)
from transformer_ee.utils.weights import create_weighter


class Normalized_Numpy_Dataset(Dataset):  # pylint: disable=C0103
    """
    One split (train/valid/test) of the shared, already normalised arrays.

    arrays:  dict from np_reader.read_compact_arrays (shared by all splits, not copied)
    rows:    event indices of this split
    weights: float32 [n_events] per-event loss weights (shared by all splits)
    """

    def __init__(self, arrays, rows, weights, max_num_prongs):
        self.arrays = arrays
        self.rows = np.asarray(rows, dtype=np.int64)
        self.weights = weights
        self.maxpronglen = max_num_prongs
        self.n_vector = arrays["vec_values"].shape[1]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        start, end = self.arrays["offsets"][r], self.arrays["offsets"][r + 1]
        n = end - start
        vector = np.zeros((self.maxpronglen, self.n_vector), dtype=np.float32)
        vector[:n] = self.arrays["vec_values"][start:end]
        mask = np.ones(self.maxpronglen, dtype=bool)
        mask[:n] = False
        return (
            torch.from_numpy(vector),
            torch.from_numpy(self.arrays["scalars"][r].copy()),
            torch.from_numpy(mask),
            torch.from_numpy(self.arrays["targets"][r].copy()),
            torch.tensor([self.weights[r]], dtype=torch.float32),
        )

    @property
    def extra(self):
        """
        The raw config "eval_columns" of this split, in dataset order, as a polars DataFrame
        (the same attribute name and type as the polars dataset).
        """
        names = self.arrays["names"]["extra"]
        return pl.DataFrame(self.arrays["extra"][self.rows], schema=names, orient="row")


def _event_weights(config, targets, train_rows):
    """Per-event weights from a weighter built on the training targets (as the polars path)."""
    n = len(targets)
    if not len(config["target"]):
        return np.ones(n, dtype=np.float32)
    weighter = create_weighter(config, targets[train_rows, 0].astype(np.float64))
    print("Created weighter from config and data. Type: ", type(weighter))
    w = weighter.getweight(targets[:, 0].astype(np.float64))
    return np.broadcast_to(np.asarray(w, dtype=np.float32), (n,)).copy()


def _freeze_inherited_objects(_worker_id):
    """
    DataLoader worker_init_fn. A forked worker inherits the parent's whole Python heap
    copy-on-write, but the garbage collector writes to every object it scans, so each worker
    would gradually duplicate that heap (hundreds of MiB with torch loaded). gc.freeze() moves
    the inherited objects out of the collector's reach; objects the worker creates itself are
    still collected.
    """
    gc.freeze()


def get_numpy_dataloaders(config: dict):
    """
    The "numpy" counterpart of load.get_train_valid_test_dataloader: same return value
    (trainloader, validloader, testloader, train statistics).

    Optional config keys:
        "array_cache":    directory to save/load the raw arrays as .npy (skips re-reading the
                          CSV on later runs). Put it on large storage, not /exp/nova/app.
        "read_block_mib": block size for reading the CSV (default 64).
        "num_workers_valid": workers for the validation loader (default min(num_workers, 2)).
                          The trainer validates in the middle of each epoch while the training
                          workers are still alive, so this keeps the number of processes (and
                          memory) down; building events from the arrays is cheap and validation
                          is limited by the GPU.
    """
    arrays = load_or_read_arrays(config, cache_dir=config.get("array_cache"))
    n_events = len(arrays["offsets"]) - 1
    train_rows, valid_rows, test_rows = split_indices(n_events, config)

    stat = train_statistics(arrays, train_rows)
    weights = _event_weights(config, arrays["targets"], train_rows)
    normalize_in_place(arrays, stat)

    maxp = config["max_num_prongs"]
    train_set = Normalized_Numpy_Dataset(arrays, train_rows, weights, maxp)
    valid_set = Normalized_Numpy_Dataset(arrays, valid_rows, weights, maxp)
    test_set = Normalized_Numpy_Dataset(arrays, test_rows, weights, maxp)

    # Workers share the arrays copy-on-write through fork (the arrays are only read), so do
    # not let the start method fall back to spawn/forkserver, which would copy them per worker.
    # Workers are not persistent: train, valid and test workers would otherwise all stay alive
    # at once; re-forking each epoch is cheap.
    requested = config.get("num_workers", 10)
    available = len(os.sched_getaffinity(0))
    num_workers = min(requested, available)
    if num_workers != requested:
        print(f"num_workers: {requested} requested, {available} CPUs available -> {num_workers}")
    num_workers_valid = min(int(config.get("num_workers_valid", 2)), num_workers)
    print(f"DataLoader workers: train/test {num_workers}, valid {num_workers_valid}")

    def worker_kwargs(n):
        if n == 0:
            return {}
        return {"multiprocessing_context": "fork", "worker_init_fn": _freeze_inherited_objects}

    train_collate_fn = normalized_noise(config, stat) if "noise" in config else None
    trainloader = DataLoader(
        train_set,
        batch_size=config["batch_size_train"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
        **worker_kwargs(num_workers),
    )
    validloader = DataLoader(
        valid_set,
        batch_size=config["batch_size_valid"],
        shuffle=False,
        num_workers=num_workers_valid,
        pin_memory=torch.cuda.is_available(),
        **worker_kwargs(num_workers_valid),
    )
    testloader = DataLoader(
        test_set,
        batch_size=config["batch_size_test"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **worker_kwargs(num_workers),
    )
    return trainloader, validloader, testloader, stat
