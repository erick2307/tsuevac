# -*- coding: utf-8 -*-
"""Reading the case tables (nodes, links, agents, actions, transitions, next-node).

The tables are plain comma-separated numbers. Files written by different tools differ in small ways that
should not matter to the model, so the reader accepts all of them:

* no header, or a header line (starting with `#` or not): leading lines that name columns are skipped
* `#` comment lines anywhere
* integers written as floats (`116.0`), which `numpy.loadtxt(..., dtype=int)` rejects

A line is a header when one of its fields is text (`number`, `Coord_x`). A data line with an empty field
(`0,0,,5`) is not a header: it is an error, reported by `numpy.loadtxt`, not a row to be dropped quietly.
"""
import numpy as np

_ENCODING = "utf-8-sig"  # a byte order mark, if the file has one, is not part of the first value


def _is_text(field):
    """True for a field that is neither a number (`nan` and `inf` are numbers) nor empty."""
    field = field.strip()
    if not field:
        return False
    try:
        float(field)
        return False
    except ValueError:
        return True


def _first_data_line(path):
    """Index (0-based) of the first line that is neither blank, a `#` comment, nor a header."""
    with open(path, "r", encoding=_ENCODING) as f:
        for i, line in enumerate(f):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if not any(_is_text(v) for v in text.split(",")):
                return i
    return 0


def load_table(path, dtype=float):
    """Load a comma-separated numeric table as a 2-D array of `dtype`.

    For an integer `dtype` the values may be written as floats, but only when they are whole numbers;
    anything else raises `ValueError` rather than being silently truncated.
    """
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=_first_data_line(path), ndmin=2,
                      encoding=_ENCODING)
    if np.issubdtype(np.dtype(dtype), np.integer):
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{path}: expected whole numbers, found nan or inf")
        rounded = np.rint(data)
        if not np.array_equal(rounded, data):
            raise ValueError(f"{path}: expected whole numbers, found fractional values")
        return rounded.astype(dtype)
    return data.astype(dtype, copy=False)
