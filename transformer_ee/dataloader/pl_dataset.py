import io
import lzma

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformer_ee.utils.weights import create_weighter


def _split_csv_line(line):
    """
    Split one CSV line (str) into raw fields, keeping the surrounding quotes, so that it can be
    seen which fields were quoted. Commas inside quotes do not split.
    """
    fields, cur, in_quotes = [], [], False
    for ch in line.rstrip("\r\n"):
        if ch == '"':
            in_quotes = not in_quotes
        if ch == "," and not in_quotes:
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields


def _read_head_lines(raw):
    """
    Return (header line, first data line) as str from a CSV held in memory (bytes).
    Uses find() so that no copy of the (possibly multi-GB) buffer is made.
    """
    end0 = raw.find(b"\n")
    if end0 < 0:
        return raw.decode("utf-8"), ""
    end1 = raw.find(b"\n", end0 + 1)
    if end1 < 0:
        end1 = len(raw)
    return raw[:end0].decode("utf-8"), raw[end0 + 1 : end1].decode("utf-8")


def _schema_overrides(header_line, first_data_line, string_columns):
    """
    Fix the dtype of every column instead of letting polars infer it.

    polars infers dtypes from the first 100 rows only. An integer-looking column such as
    leadPngTruePdg was therefore read as i64, and the parse aborted when a later row held a value
    that CSVMaker::setPrecision(6) had written in scientific notation (the nuclear PDG code
    1.00001e+09). So no column is left to inference:

    - pl.Utf8 for the columns in string_columns (the config "vector" columns), and for every
      column that is quoted in the first data row. CSVMaker always quotes MultiVar (per-prong)
      columns, even with one prong, so this catches the prong columns not used by the config.
      They hold comma-separated lists; the vector ones are split later with .str.split.
    - pl.Float64 for every other column. Float64 holds the integer columns of these files
      (run, evt, counts, flags, PDG codes) exactly; the dataset casts to float32 anyway.

    header_line:      the CSV header line (str)
    first_data_line:  the first data row (str), used only to see which fields are quoted
    string_columns:   iterable of column names to read as strings (e.g. config["vector"])
    NOTE: schema_overrides needs polars >= 0.20.31 (older versions call it dtypes).
    """
    header = _split_csv_line(header_line)
    string_columns = set(string_columns)
    missing = string_columns.difference(header)
    if missing:
        raise ValueError(f"Columns not found in CSV header: {sorted(missing)}")
    first = _split_csv_line(first_data_line) if first_data_line else []
    quoted = {
        name for name, field in zip(header, first) if field.startswith('"')
    }
    return {
        c: (pl.Utf8 if c in string_columns or c in quoted else pl.Float64)
        for c in header
    }


def get_polars_df_from_xz_file(file_path, string_columns=()):
    """
    Reads an xz file and returns a Polars DataFrame.
    Columns in string_columns are read as strings.
    """

    with open(file_path, "rb") as f:
        xz_data = f.read()
    memory_file = io.BytesIO(xz_data)
    print("Decompressing data from xz file", file_path, "to memory...")
    decompressed_data = lzma.decompress(memory_file.read())
    header_line, first_data_line = _read_head_lines(decompressed_data)
    decompressed_memory_file = io.BytesIO(decompressed_data)
    return pl.read_csv(
        decompressed_memory_file,
        schema_overrides=_schema_overrides(header_line, first_data_line, string_columns),
    )


def get_polars_df_from_file(file_path, string_columns=()):
    """
    Reads a file and returns a Polars DataFrame.

    Args:
        file_path (str): The path to the file.
        string_columns (iterable of str): columns to read as strings (e.g. config["vector"]).

    Returns:
        pl.DataFrame: A Polars DataFrame containing the data from the file.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.endswith(".csv"):
        with open(file_path, "rb") as f:
            header_line = f.readline().decode("utf-8")
            first_data_line = f.readline().decode("utf-8")
        return pl.read_csv(
            file_path,
            schema_overrides=_schema_overrides(header_line, first_data_line, string_columns),
        )
    elif file_path.endswith(".xz"):
        return get_polars_df_from_xz_file(file_path, string_columns=string_columns)
    else:
        raise ValueError("File format not supported")


class Polars_Dataset(Dataset): # pylint: disable=C0103, W0223
    """
    A base PyTorch dataset for polars dataframe
    """

    def __init__(self, config: dict, dtframe: pl.DataFrame, weighter=None, eval=False): # pylint: disable=W0622
        self.eval = eval
        self.config = config.copy()

        self.df = dtframe

        # Raw copy of the evaluation columns (config key "eval_columns"), taken BEFORE any
        # normalisation: normalize() rewrites scalar columns in place, and a column may be
        # both a scalar input and an evaluation column (e.g. shwCalE).
        # With no "eval_columns" key this is an empty frame and nothing extra is saved.
        self.extra = self.df.select(config.get("eval_columns", []))

        self.maxpronglen = config["max_num_prongs"]
        self.vectornames = config["vector"]
        self.scalarnames = config["scalar"]
        self.targetname = config["target"] if not self.eval else None
        # If weighter is not provided, create a new one from config and dataframe.
        # For prediction, the weighter should be set to NullWeights() manually.
        self.weighter = None
        if self.eval:
            print("In evaluation mode, target = 0 and weighter = 1.")
        elif weighter is None:
            self.weighter = create_weighter(
                self.config, self.df[self.targetname[0]].to_numpy()
            )
            print("Created weighter from config and data. Type: ", type(self.weighter))
        else:
            self.weighter = weighter
            print("Using provided weighter. Type: ", type(self.weighter))

        # convert string to list of float
        for sequence_name in self.config["vector"]:
            self.df = self.df.with_columns(
                self.df.get_column(sequence_name).replace("", "0")
            )
            self.df = self.df.with_columns(
                self.df.get_column(sequence_name)
                .str.split(by=",")
                .cast(pl.List(pl.Float32))
            )

    def __len__(self):
        return len(self.df)


class Normalized_Polars_Dataset_with_cache(Polars_Dataset): # pylint: disable=C0103
    """
    A base PyTorch dataset for polars dataframe with normalization and caching.
    NOTE: Normalization is done in place.
    """

    def __init__(
        self,
        config: dict,
        dtframe: pl.DataFrame,
        weighter=None,
        eval=False,
        use_cache=True,
    ):
        super().__init__(config, dtframe, weighter=weighter, eval=eval)
        self.use_cache = use_cache
        self.cached = {}
        self.stat = {}
        self.normalized = False

    def statistic(self):
        """
        Calculate the mean and standard deviation with respect to each column.
        """
        if self.eval:
            raise ValueError("In evaluation mode, do not call statistic()!")
        if self.normalized:
            raise ValueError("Already normalized! Do not call statistic() again!")

        # calculate mean and std for sequence features
        for sequence_name in self.vectornames:
            _ = self.df.get_column(sequence_name).explode()
            self.stat[sequence_name] = [_.mean(), _.std() + 1e-10]
            del _

        # calculate mean and std for scalar features
        for scalar_name in self.scalarnames:
            self.stat[scalar_name] = [
                self.df.get_column(scalar_name).mean(),
                self.df.get_column(scalar_name).std() + 1e-10,
            ]

    def normalize(self, stat=None):
        """
        Normalize the dataset with respect to the provided statistics.
        If no statistics is provided, use the statistics calculated by statistic().
        """

        _stat = stat
        if _stat is None:  # by default, use the statistics calculated by statistic()
            if self.eval:
                raise ValueError("In evaluation mode, stat cannot be None!")
            print("Using statistics calculated by statistic()!")
            if not self.stat:
                raise ValueError("Please call statistics() first!")
            _stat = self.stat

        if self.normalized:
            raise ValueError("Already normalized! Do not call normalize() again!")

        for sc_name in self.scalarnames:
            self.df = self.df.with_columns(
                (self.df.select(sc_name) - _stat[sc_name][0]) / _stat[sc_name][1]
            )

        for sequence_name in self.vectornames:
            self.df = self.df.with_columns(
                self.df.get_column(sequence_name).list.eval(
                    (pl.element() - _stat[sequence_name][0]) / _stat[sequence_name][1]
                )
            )

        self.normalized = True

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index]

        if not self.normalized:
            raise ValueError("Please call normalize() first!")

        _vectorsize = len(self.df[index, self.vectornames[0]])
        _vector = torch.Tensor(
            [self.df[index, v].to_list() for v in self.vectornames]
        ).T
        _scalar = torch.Tensor([self.df[index, s] for s in self.scalarnames])

        _mask = torch.Tensor([0] * _vectorsize)

        if _vectorsize < self.maxpronglen:
            _vector = F.pad(
                _vector, (0, 0, 0, self.maxpronglen - _vectorsize), "constant", 0
            )
            _mask = F.pad(_mask, (0, self.maxpronglen - _vectorsize), "constant", 1)
        else:
            _vector = _vector[: self.maxpronglen, :]
            _mask = _mask[: self.maxpronglen]

        if not _vectorsize:
            _vector = torch.zeros((self.maxpronglen, len(self.vectornames)))
            _mask = torch.ones(self.maxpronglen)

        _target = 0.0
        _weight = 1.0
        if not self.eval:
            _target = torch.Tensor([self.df[index, t] for t in self.targetname])
            _weight = torch.Tensor(
                [self.weighter.getweight(self.df[index, self.targetname[0]])]
            )

        return_tuple = (
            _vector,  # shape: (max_seq_len, vector_dim)
            _scalar,  # shape: (scalar_dim)
            _mask.to(torch.bool),  # shape: (max_seq_len)
            _target,  # shape: (target_dim)
            _weight,  # shape: (1)
        )
        if self.use_cache:
            self.cached[index] = return_tuple
        return return_tuple
