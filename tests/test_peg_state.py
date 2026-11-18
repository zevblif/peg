"""PEG channel mechanics: version mask, tombstone and RRF fusion budget."""
import csv
import tempfile
import unittest

import _bootstrap  # noqa: F401
from pathlib import Path

import numpy as np

from peg_bench import peg_channel


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(rows)


class PegStateTest(unittest.TestCase):
    def setUp(self):
        self.vectors = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ], dtype=np.float32)
        self.state = peg_channel.PegState(["A", "B", "C"], ["a", "b", "c"], self.vectors)
        self._embed = peg_channel.embed_texts
        peg_channel.embed_texts = lambda texts, batch_size=16: np.asarray(
            [[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        peg_channel.embed_texts = self._embed
        self.tmp.cleanup()

    def test_search_returns_nearest_active_paragraph(self):
        ranking, _ = self.state.search("q", top_k=3)
        self.assertEqual(ranking[0], "A")

    def test_tombstone_removes_paragraph_from_results(self):
        path = self.dir / "delete.csv"
        write_csv(path, [{"title": "A", "text": "a"}])
        self.state.delete(path)
        ranking, _ = self.state.search("q", top_k=3)
        self.assertNotIn("A", ranking)

    def test_replace_updates_text_and_version(self):
        path = self.dir / "edit.csv"
        write_csv(path, [{"title": "A", "text": "a2"}])
        before = self.state.version[self.state.index["A"]]
        self.state.replace(path)
        self.assertEqual(self.state.version[self.state.index["A"]], before + 1)
        self.assertEqual(self.state.texts[self.state.index["A"]], "a2")

    def test_append_ignores_existing_titles(self):
        path = self.dir / "delta.csv"
        write_csv(path, [{"title": "A", "text": "a"}, {"title": "D", "text": "d"}])
        added = self.state.add(path)
        self.assertEqual(added, 1)
        self.assertEqual(len(self.state.titles), 4)


class RrfTest(unittest.TestCase):
    def test_shared_hit_outranks_single_list_hits(self):
        fused = peg_channel.rrf([["B", "A"], ["A", "C"]])
        self.assertEqual(fused[0], "A")

    def test_budget_is_bounded_by_top_k(self):
        fused = peg_channel.rrf([["A", "B", "C", "D"]], top_k=2)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused, ["A", "B"])

    def test_empty_input_yields_empty_ranking(self):
        self.assertEqual(peg_channel.rrf([], top_k=5), [])


if __name__ == "__main__":
    unittest.main()
