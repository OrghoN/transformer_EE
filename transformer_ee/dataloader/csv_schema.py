"""
CSV header helpers shared by the polars loader (pl_dataset.py) and the compact numpy loader
(np_reader.py). No torch dependency.
"""

import polars as pl


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
