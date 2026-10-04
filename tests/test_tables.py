#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evacrl.tables.load_table: the table formats the two code bases write must all be readable.

Run: python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evacrl.tables import load_table  # noqa: E402


class LoadTable(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text, name="t.csv"):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_no_header(self):
        np.testing.assert_array_equal(load_table(self.write("0,1\n2,3\n"), dtype=int), [[0, 1], [2, 3]])

    def test_hash_header_and_comments(self):
        path = self.write("# number,x\n0,1.5\n# a comment\n2,3.5\n")
        np.testing.assert_array_equal(load_table(path), [[0, 1.5], [2, 3.5]])

    def test_header_without_hash_is_skipped(self):
        # the format of the 2024 Kochi study: "#number,..." or "number,..." (any single header line)
        path = self.write("number,Coord_x,Coord_y\n0,1.5,2.5\n1,3.5,4.5\n")
        np.testing.assert_array_equal(load_table(path), [[0, 1.5, 2.5], [1, 3.5, 4.5]])

    def test_integers_written_as_floats(self):
        path = self.write("#number,Node1,Node2,Length,Width\n0,16,17,116.0,3\n1,16,18,220.0,3\n")
        data = load_table(path, dtype=int)
        self.assertTrue(np.issubdtype(data.dtype, np.integer))
        np.testing.assert_array_equal(data, [[0, 16, 17, 116, 3], [1, 16, 18, 220, 3]])

    def test_fractional_values_are_not_silently_truncated(self):
        path = self.write("0,1.5\n")
        with self.assertRaises(ValueError):
            load_table(path, dtype=int)

    def test_one_row_stays_two_dimensional(self):
        self.assertEqual(load_table(self.write("0,1,2\n"), dtype=int).shape, (1, 3))

    def test_byte_order_mark_and_blank_lines(self):
        path = self.write("\ufeff# a,b\n\n1,2\n\n3,4\n")
        np.testing.assert_array_equal(load_table(path, dtype=int), [[1, 2], [3, 4]])

    def test_byte_order_mark_before_a_numeric_first_row(self):
        path = self.write("\ufeff1,2\n3,4\n")
        np.testing.assert_array_equal(load_table(path, dtype=int), [[1, 2], [3, 4]])

    def test_a_malformed_first_data_row_is_an_error_not_a_header(self):
        # an empty field is not text: the row must not be dropped as if it were a header
        with self.assertRaises(ValueError):
            load_table(self.write("0,0,0,,5\n1,0,0,2,5\n"), dtype=int)

    def test_nan_in_an_integer_table_is_named_as_such(self):
        with self.assertRaisesRegex(ValueError, "nan or inf"):
            load_table(self.write("0,nan\n"), dtype=int)

    def test_the_tables_of_the_repository_load(self):
        for area in ("kochi", "new_kochi"):
            data = REPO / "cases" / area / "data"
            nodes = load_table(str(data / "nodesdb.csv"))
            links = load_table(str(data / "linksdb.csv"), dtype=int)
            self.assertEqual(nodes.shape[1], 5)
            self.assertEqual(links.shape[1], 5)
            self.assertEqual(load_table(str(data / "actionsdb.csv"), dtype=int).shape[0], nodes.shape[0])


if __name__ == "__main__":
    unittest.main()
