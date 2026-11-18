"""Data subset invariants that make the incremental comparison meaningful."""
import csv
import json
import unittest

import _bootstrap  # noqa: F401
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "data" / "main"
STABILITY = REPO / "data" / "stability"


def titles(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [row["title"] for row in csv.DictReader(stream)]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MainSubsetTest(unittest.TestCase):
    def test_base_and_delta_titles_are_disjoint(self):
        self.assertFalse(set(titles(MAIN / "base.csv")) & set(titles(MAIN / "delta.csv")))

    def test_fixed_sizes(self):
        self.assertEqual(len(titles(MAIN / "base.csv")), 300)
        self.assertEqual(len(titles(MAIN / "delta.csv")), 100)
        self.assertEqual(len(jsonl(MAIN / "queries_base.jsonl")), 30)
        self.assertEqual(len(jsonl(MAIN / "queries_delta.jsonl")), 10)

    def test_edit_and_delete_probes(self):
        edited = set(titles(MAIN / "edit.csv"))
        deleted = set(titles(MAIN / "delete.csv"))
        self.assertEqual(len(edited), 10)
        self.assertEqual(len(deleted), 5)
        self.assertFalse(edited & deleted)
        self.assertTrue(edited <= set(titles(MAIN / "base.csv")))
        self.assertTrue(deleted <= set(titles(MAIN / "base.csv")))
        self.assertEqual(len(jsonl(MAIN / "edit_probes.jsonl")), 10)
        self.assertEqual(len(jsonl(MAIN / "delete_probes.jsonl")), 5)


class StabilitySubsetTest(unittest.TestCase):
    def test_sizes_and_folds(self):
        self.assertEqual(len(titles(STABILITY / "base.csv")), 900)
        self.assertEqual(len(titles(STABILITY / "delta.csv")), 300)
        folds = json.loads((STABILITY / "folds.json").read_text(encoding="utf-8"))
        self.assertEqual(len(folds), 120)
        self.assertEqual({item["fold"] for item in folds}, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
