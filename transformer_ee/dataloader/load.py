"""
Load the data
"""

import numpy as np
import torch

from transformer_ee.dataloader.noise import normalized_noise


def get_sample_indices(sample_size: int, config) -> tuple:
    """
    USED BY PANDAS.
    A function to get the indices of the sample rows from a whole Pandas DataFrame.

    sample_size:    the number of samples
    config:         the configuration dictionary
    return:         the indices of train, validation and test sets
    """

    seed = config["seed"]

    _indices = np.arange(sample_size)
    np.random.seed(seed)
    np.random.shuffle(_indices)

    test_size = config["test_size"]
    valid_size = config["valid_size"]

    # test_size and valid_size can be either int or float
    if isinstance(test_size, float):
        test_size = int(sample_size * test_size)
    if isinstance(valid_size, float):
        valid_size = int(sample_size * valid_size)

    train_indicies = _indices[: sample_size - valid_size - test_size]
    valid_indicies = _indices[
        sample_size - valid_size - test_size : sample_size - test_size
    ]
    test_indicies = _indices[sample_size - test_size :]

    print("train indicies size:\t", len(train_indicies))
    print("valid indicies size:\t", len(valid_indicies))
    print("test  indicies size:\t", len(test_indicies))

    return train_indicies, valid_indicies, test_indicies


def get_sample_sizes(sample_size: int, config) -> tuple:
    """
    USED BY POLARS.
    A function to get the sizes of the samples.

    sample_size:    the number of samples
    config:         the configuration dictionary
    return:         a tuple of three integers:
                        the number of training samples,
                        the number of validation samples,
                        the number of test samples
    """

    test_size = config["test_size"]
    valid_size = config["valid_size"]

    # test_size and valid_size can be either int or float
    if isinstance(test_size, float):
        test_size = int(sample_size * test_size)
    if isinstance(valid_size, float):
        valid_size = int(sample_size * valid_size)

    print("train indicies size:\t", sample_size - test_size - valid_size)
    print("valid indicies size:\t", valid_size)
    print("test  indicies size:\t", test_size)

    return sample_size - test_size - valid_size, valid_size, test_size


def save_indices(indices, filename):
    """
    Save the indices to a text file.
    
    indices: The indices to save.
    filename: The file to save the indices in.
    """
    with open(filename, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")


def get_train_valid_test_dataloader(config: dict):  # pylint: disable=R0914
    """
    A function to get the train, validation and test datasets
    Use the statistic of the training set to normalize the validation and test sets
    """

    dataframe_type = config.get("dataframe_type", "pandas")

    # Per-sample cache of the pandas/polars datasets. With num_workers > 0 every DataLoader
    # worker fills its own private copy of the cache, and the workers are recreated at each
    # epoch (and each validation pass), so the cache is never hit: it only grows, to ~48 GiB
    # for the 14.6 M-event mag-moment training set. At the half-epoch validation pass the
    # training workers are still alive holding ~half of it while the validation workers are
    # forked, and the job is OOM-killed. Off by default when workers are used; set
    # "sample_cache": true in the config to force it on.
    num_workers = config.get("num_workers", 10)
    use_cache = bool(config.get("sample_cache", num_workers == 0))
    print("per-sample dataset cache:", "on" if use_cache else "off")

    if dataframe_type == "numpy":
        # Memory-compact path: streams the CSV into a few float32 arrays, no per-sample cache.
        # See np_reader.py / np_dataset.py.
        from transformer_ee.dataloader.np_dataset import (  # pylint: disable=C0415
            get_numpy_dataloaders,
        )

        return get_numpy_dataloaders(config)

    if dataframe_type == "pandas":
        import pandas as pd  # pylint: disable=C0415

        from transformer_ee.dataloader.pd_dataset import (  # pylint: disable=C0415
            Normalized_pandas_Dataset_with_cache,
        )

        df = pd.read_csv(config["data_path"])
        train_idx, valid_idx, test_idx = get_sample_indices(len(df), config)
        train_set = Normalized_pandas_Dataset_with_cache(
            config,
            df.iloc[train_idx].reset_index(drop=True, inplace=False),
            use_cache=use_cache,
        )
        valid_set = Normalized_pandas_Dataset_with_cache(
            config,
            df.iloc[valid_idx].reset_index(drop=True, inplace=False),
            weighter=train_set.weighter,
            use_cache=use_cache,
        )
        test_set = Normalized_pandas_Dataset_with_cache(
            config,
            df.iloc[test_idx].reset_index(drop=True, inplace=False),
            weighter=train_set.weighter,
            use_cache=use_cache,
        )
    elif dataframe_type == "polars":
        from transformer_ee.dataloader.pl_dataset import (  # pylint: disable=C0415
            Normalized_Polars_Dataset_with_cache,
            get_polars_df_from_file,
        )

        df = get_polars_df_from_file(
            config["data_path"], string_columns=config["vector"]
        )
        randomdf = df.sample(fraction=1.0, seed=config["seed"], shuffle=True)
        del df
        sizes = get_sample_sizes(randomdf.height, config)
        train_set = Normalized_Polars_Dataset_with_cache(
            config, randomdf.slice(offset=0, length=sizes[0]), use_cache=use_cache
        )
        valid_set = Normalized_Polars_Dataset_with_cache(
            config,
            randomdf.slice(offset=sizes[0], length=sizes[1]),
            weighter=train_set.weighter,
            use_cache=use_cache,
        )
        test_set = Normalized_Polars_Dataset_with_cache(
            config,
            randomdf.slice(offset=sizes[0] + sizes[1], length=sizes[2]),
            weighter=train_set.weighter,
            use_cache=use_cache,
        )

    else:
        raise ValueError(
            f"Unknown dataframe_type: {dataframe_type}. Supported types: ['pandas', 'polars', 'numpy']"
        )

    train_set.statistic()
    train_set.normalize()
    valid_set.normalize(train_set.stat)
    test_set.normalize(train_set.stat)

    batch_size_train = config["batch_size_train"]
    batch_size_valid = config["batch_size_valid"]
    batch_size_test = config["batch_size_test"]

    train_collate_fn = None
    if "noise" in config:
        train_collate_fn = normalized_noise(config, train_set.stat)

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
    )

    validloader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
    )

    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
    )

    return trainloader, validloader, testloader, train_set.stat
