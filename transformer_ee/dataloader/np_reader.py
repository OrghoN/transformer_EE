"""
Memory-compact CSV reader for transformer_ee (config "dataframe_type": "numpy"). No torch dependency.

Why: the polars path holds the whole CSV as a DataFrame (all columns, prong lists as strings and
then as List[Float32] with one 8-byte offset per event per column), the decompressed .xz text in
memory, a full shuffled copy of the frame, and a per-sample tensor cache in every DataLoader
worker. On the NOvA magnetic-moment sample (19 M events, 12.7 GiB of CSV) that is ~30 GiB to
load and ~65 GiB by the end of the first epoch.

This reader streams the file (plain .csv or .csv.xz) in blocks, keeps only the columns the config
uses, and stores them as a few contiguous float32 arrays:

    vec_values  float32 [n_prongs_total, n_vector]   all prongs of all events, event by event
    offsets     int64   [n_events + 1]               prongs of event i: vec_values[offsets[i]:offsets[i+1]]
    scalars     float32 [n_events, n_scalar]
    targets     float32 [n_events, n_target]         (empty if no targets)
    extra       float32 [n_events, n_extra]          config "eval_columns", raw (never normalised)

For the sample above this is ~4.8 GB. Splitting, statistics and normalisation work on these
arrays without copying them.
"""

import io
import json
import lzma
import os

import numpy as np
import polars as pl

from transformer_ee.dataloader.csv_schema import _schema_overrides, _split_csv_line

# Integers above 2**24 cannot all be represented exactly in float32.
_FLOAT32_EXACT_INT = 2**24

ARRAY_NAMES = ("vec_values", "offsets", "scalars", "targets", "extra")


def _unique(names):
    """Remove duplicates, keep order."""
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def _open_binary(file_path):
    if file_path.endswith(".xz"):
        return lzma.open(file_path, "rb")
    if file_path.endswith(".csv"):
        return open(file_path, "rb")  # pylint: disable=R1732
    raise ValueError("File format not supported (expected .csv or .csv.xz): " + file_path)


def _iter_blocks(f, block_bytes):
    """Yield chunks of complete lines (bytes) of about block_bytes from an open binary file."""
    rest = b""
    while True:
        chunk = f.read(block_bytes)
        if not chunk:
            if rest.strip():
                yield rest if rest.endswith(b"\n") else rest + b"\n"
            return
        buf = rest + chunk
        cut = buf.rfind(b"\n")
        if cut < 0:
            rest = buf
            continue
        yield buf[: cut + 1]
        rest = buf[cut + 1 :]


def _concat_freeing(chunks, width):
    """
    Concatenate a list of float32 [n_i, width] arrays into one, emptying the list as it goes.
    The output is allocated with np.empty, whose pages only become resident as they are
    written, while each copied chunk is freed at once; the peak therefore stays near the size
    of the result instead of twice it (as with np.concatenate).
    """
    out = np.empty((sum(len(c) for c in chunks), width), dtype=np.float32)
    pos = 0
    while chunks:
        c = chunks.pop(0)
        out[pos : pos + len(c)] = c
        pos += len(c)
        del c
    return out


def _vector_block(df, vector, max_prongs):
    """
    Convert the comma-separated prong columns of one block to (lengths, values[n_prongs, n_vec]).
    As in the polars dataset, an empty field is read as one prong with value 0, and events with
    more than max_prongs prongs keep their first max_prongs prongs.
    """
    lengths = None
    columns = []
    for v in vector:
        lists = (
            df.get_column(v)
            .fill_null("0")
            .replace("", "0")
            .str.split(",")
            .list.head(max_prongs)
        )
        n = lists.list.len().to_numpy().astype(np.int64)
        if lengths is None:
            lengths = n
        elif not np.array_equal(lengths, n):
            bad = int(np.flatnonzero(lengths != n)[0])
            raise ValueError(
                f"Prong columns disagree in length: '{vector[0]}' vs '{v}' "
                f"(first mismatch at row {bad} of the block)"
            )
        columns.append(lists.explode().cast(pl.Float32).to_numpy())
    values = np.empty((int(lengths.sum()), len(vector)), dtype=np.float32)
    for j, col in enumerate(columns):
        values[:, j] = col
    return lengths, values


def _check_float32_exact(block, names):
    """Raise if an integer-valued column holds values float32 cannot represent exactly."""
    for j, name in enumerate(names):
        col = block[:, j]
        finite = col[np.isfinite(col)]
        if finite.size and np.all(finite == np.round(finite)):
            if np.max(np.abs(finite)) >= _FLOAT32_EXACT_INT:
                raise ValueError(
                    f"eval column '{name}' holds integers >= 2**24, which float32 cannot store "
                    "exactly. Remove it from eval_columns or extend np_reader to keep float64."
                )


def read_compact_arrays(  # pylint: disable=R0913,R0914
    file_path,
    vector,
    scalar,
    target=(),
    extra=(),
    max_prongs=10,
    block_bytes=64 << 20,
    verbose=True,
):
    """
    Read a transformer_ee CSV (.csv or .csv.xz) into compact numpy arrays (see module docstring).

    Only the listed columns are parsed. Column dtypes are fixed from the header (no inference):
    prong columns as strings, everything else as float. Raises ValueError for missing columns,
    null values, prong columns of unequal length, or eval columns float32 cannot hold exactly.
    Returns a dict with the arrays in ARRAY_NAMES and "names" (the column lists).
    """
    vector, scalar, target, extra = (list(x) for x in (vector, scalar, target, extra))
    needed = _unique(vector + scalar + target + extra)
    numeric = _unique(scalar + target + extra)

    parts = {k: [] for k in ("lengths", "vec_values", "scalars", "targets", "extra")}
    n_rows = 0
    with _open_binary(file_path) as f:
        header = f.readline()
        first = f.readline()
        header_line = header.decode("utf-8")
        columns = _split_csv_line(header_line)
        missing = [c for c in needed if c not in columns]
        if missing:
            raise ValueError(f"Columns not found in CSV header of {file_path}: {missing}")
        schema = _schema_overrides(header_line, first.decode("utf-8"), vector)
        # Prong columns stay strings; the numeric columns used here are parsed as Float64
        # (exact, as in the polars path) and stored as float32 below.
        for name in numeric:
            if schema[name] != pl.Float64:
                raise ValueError(f"Column '{name}' looks like a prong (list) column, not a number")

        if verbose:
            print("Reading", file_path, "in blocks of", block_bytes >> 20, "MiB ...", flush=True)

        def blocks():
            yield first
            yield from _iter_blocks(f, block_bytes)

        for block in blocks():
            if not block.strip():
                continue
            df = pl.read_csv(
                io.BytesIO(header + block),
                columns=needed,
                schema_overrides=schema,
            )
            nulls = {c: n for c, n in zip(df.columns, df.null_count().row(0)) if n and c in numeric}
            if nulls:
                raise ValueError(f"Null/empty values in numeric columns: {nulls}")

            lengths, values = _vector_block(df, vector, max_prongs)
            parts["lengths"].append(lengths)
            parts["vec_values"].append(values)
            parts["scalars"].append(df.select(scalar).to_numpy().astype(np.float32))
            parts["targets"].append(
                df.select(target).to_numpy().astype(np.float32)
                if target
                else np.empty((df.height, 0), np.float32)
            )
            if extra:
                ex = df.select(extra).to_numpy()
                _check_float32_exact(ex, extra)
                parts["extra"].append(ex.astype(np.float32))
            else:
                parts["extra"].append(np.empty((df.height, 0), np.float32))
            n_rows += df.height
            del df
            if verbose and len(parts["lengths"]) % 10 == 0:
                print(f"  {n_rows:,} events read", flush=True)

    # Concatenate one array at a time and free its parts, to keep the peak low.
    out = {}
    lengths = np.concatenate(parts.pop("lengths")) if n_rows else np.empty(0, np.int64)
    out["offsets"] = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(lengths, out=out["offsets"][1:])
    del lengths
    widths = {"vec_values": vector, "scalars": scalar, "targets": target, "extra": extra}
    for key in ("vec_values", "scalars", "targets", "extra"):
        out[key] = _concat_freeing(parts.pop(key), len(widths[key]))
    out["names"] = {"vector": vector, "scalar": scalar, "target": target, "extra": extra}
    if verbose:
        size = sum(out[k].nbytes for k in ARRAY_NAMES) / 2**30
        print(
            f"Read {n_rows:,} events, {len(out['vec_values']):,} prongs; arrays use {size:.2f} GiB",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------------------------
# Optional on-disk cache of the raw (un-normalised) arrays
# --------------------------------------------------------------------------------------------
def _cache_key(file_path, names, max_prongs):
    st = os.stat(file_path)
    return {
        "source": os.path.abspath(file_path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "names": names,
        "max_prongs": max_prongs,
    }


def load_or_read_arrays(config, cache_dir=None, verbose=True):
    """
    read_compact_arrays() for a transformer_ee config. If cache_dir is given, the raw arrays are
    saved there as .npy files after the first read and loaded from there on later runs, as long
    as the source file, the column lists and max_num_prongs are unchanged.
    """
    names = {
        "vector": list(config["vector"]),
        "scalar": list(config["scalar"]),
        "target": list(config.get("target", [])),
        "extra": list(config.get("eval_columns", [])),
    }
    max_prongs = config["max_num_prongs"]
    key = _cache_key(config["data_path"], names, max_prongs)
    meta_path = os.path.join(cache_dir, "meta.json") if cache_dir else None

    if meta_path and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fm:
            meta = json.load(fm)
        if meta == key:
            if verbose:
                print("Loading cached arrays from", cache_dir, flush=True)
            out = {k: np.load(os.path.join(cache_dir, k + ".npy")) for k in ARRAY_NAMES}
            out["names"] = names
            return out
        if verbose:
            print("Array cache in", cache_dir, "is out of date; re-reading the CSV", flush=True)

    out = read_compact_arrays(
        config["data_path"],
        names["vector"],
        names["scalar"],
        names["target"],
        names["extra"],
        max_prongs=max_prongs,
        block_bytes=config.get("read_block_mib", 64) << 20,
        verbose=verbose,
    )
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(meta_path):
            os.remove(meta_path)  # invalidate while rewriting
        for k in ARRAY_NAMES:
            np.save(os.path.join(cache_dir, k + ".npy"), out[k])
        with open(meta_path, "w", encoding="utf-8") as fm:
            json.dump(key, fm, indent=4)
        if verbose:
            print("Saved array cache to", cache_dir, flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Split, statistics, normalisation (all in place / chunked; no full copies)
# --------------------------------------------------------------------------------------------
def split_indices(n_events, config):
    """
    Random train/valid/test row indices. Sizes follow load.get_sample_sizes (int or fraction).
    The order is a seeded permutation, as in the polars path (but not the same permutation).
    """
    test_size, valid_size = config["test_size"], config["valid_size"]
    if isinstance(test_size, float):
        test_size = int(n_events * test_size)
    if isinstance(valid_size, float):
        valid_size = int(n_events * valid_size)
    n_train = n_events - test_size - valid_size
    perm = np.random.default_rng(config["seed"]).permutation(n_events)
    print("train indicies size:\t", n_train)
    print("valid indicies size:\t", valid_size)
    print("test  indicies size:\t", test_size)
    return perm[:n_train], perm[n_train : n_train + valid_size], perm[n_train + valid_size :]


def _chunked_mean_std(array, row_mask, chunk=1 << 20):
    """Column mean and std (ddof=1) of array[row_mask], computed in float64 chunks."""
    n = int(row_mask.sum())
    total = np.zeros(array.shape[1])
    for s in range(0, len(array), chunk):
        total += array[s : s + chunk][row_mask[s : s + chunk]].sum(axis=0, dtype=np.float64)
    mean = total / n
    sq = np.zeros(array.shape[1])
    for s in range(0, len(array), chunk):
        d = array[s : s + chunk][row_mask[s : s + chunk]].astype(np.float64) - mean
        sq += (d * d).sum(axis=0)
    std = np.sqrt(sq / (n - 1)) if n > 1 else np.zeros_like(mean)
    return mean, std


def train_statistics(arrays, train_rows):
    """
    Mean and std of every vector and scalar input over the training set, in the format of the
    other datasets: {name: [mean, std + 1e-10]}. Vector statistics run over all prongs of the
    training events.
    """
    n_events = len(arrays["offsets"]) - 1
    is_train = np.zeros(n_events, dtype=bool)
    is_train[train_rows] = True
    lengths = np.diff(arrays["offsets"])
    prong_is_train = np.repeat(is_train, lengths)
    stat = {}
    for names, key, mask in (
        (arrays["names"]["vector"], "vec_values", prong_is_train),
        (arrays["names"]["scalar"], "scalars", is_train),
    ):
        mean, std = _chunked_mean_std(arrays[key], mask)
        for j, name in enumerate(names):
            stat[name] = [float(mean[j]), float(std[j]) + 1e-10]
    return stat


def normalize_in_place(arrays, stat, chunk=1 << 20):
    """(x - mean) / std for the vector and scalar arrays, in place. Targets and extras untouched."""
    for names, key in (
        (arrays["names"]["vector"], "vec_values"),
        (arrays["names"]["scalar"], "scalars"),
    ):
        mean = np.array([stat[n][0] for n in names], dtype=np.float32)
        std = np.array([stat[n][1] for n in names], dtype=np.float32)
        a = arrays[key]
        for s in range(0, len(a), chunk):
            a[s : s + chunk] -= mean
            a[s : s + chunk] /= std
