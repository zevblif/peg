"""Last-writer-wins per title: the final state does not depend on interleaving."""
import csv
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from peg_bench import peg_channel


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(rows)


def stub_embed(texts, batch_size=16):
    """Deterministic per-text vector so an operation is identifiable in the state."""
    return np.asarray(
        [[len(text) % 7 + 1.0, ord(text[0]) % 5 + 1.0, 1.0] for text in texts],
        dtype=np.float32,
    )


def interleavings(sequences):
    """Every merge of the per-title operation sequences that preserves each order."""
    if all(not sequence for sequence in sequences):
        yield []
        return
    for index, sequence in enumerate(sequences):
        if not sequence:
            continue
        rest = sequences[:index] + [sequence[1:]] + sequences[index + 1:]
        for tail in interleavings(rest):
            yield [sequence[0]] + tail


def per_title(state):
    return {
        title: (state.texts[state.index[title]], state.status[state.index[title]], state.version[state.index[title]])
        for title in state.titles
    }


class LwwTest(unittest.TestCase):
    def setUp(self):
        self._embed = peg_channel.embed_texts
        peg_channel.embed_texts = stub_embed
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.count = 0

    def tearDown(self):
        peg_channel.embed_texts = self._embed
        self.tmp.cleanup()

    def apply(self, state, operations):
        for kind, title, text in operations:
            self.count += 1
            path = self.dir / ("op-%d.csv" % self.count)
            write_csv(path, [{"title": title, "text": text}])
            if kind == "replace":
                state.replace(path)
            elif kind == "delete":
                state.delete(path)
            else:
                state.add(path)

    def test_final_state_is_independent_of_interleaving(self):
        sequences = [
            [("replace", "A", "a1"), ("replace", "A", "a2")],
            [("replace", "B", "b1"), ("delete", "B", "b1")],
            [("replace", "C", "c1"), ("replace", "C", "c2")],
        ]
        results = set()
        for order in interleavings(sequences):
            state = peg_channel.PegState(["A", "B", "C"], ["a", "b", "c"], np.eye(3, dtype=np.float32))
            self.apply(state, order)
            results.add(tuple(sorted(per_title(state).items())))
        self.assertEqual(len(results), 1, results)
        final = dict(next(iter(results)))
        self.assertEqual(final["A"], ("a2", "active", 3))
        self.assertEqual(final["C"], ("c2", "active", 3))
        self.assertEqual(final["B"][1], "deleted")

    def test_delete_then_replace_reactivates_with_later_text(self):
        state = peg_channel.PegState(["A"], ["a"], np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
        self.apply(state, [("delete", "A", "a"), ("replace", "A", "a3")])
        self.assertEqual(state.status[0], "active")
        self.assertEqual(state.texts[0], "a3")
        ranking, _ = state.search("q", top_k=1)
        self.assertEqual(ranking, ["A"])

    def test_replace_then_delete_is_terminal(self):
        state = peg_channel.PegState(["A"], ["a"], np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
        self.apply(state, [("replace", "A", "a3"), ("delete", "A", "a3")])
        self.assertEqual(state.status[0], "deleted")
        ranking, _ = state.search("q", top_k=1)
        self.assertEqual(ranking, [])

    def test_append_is_idempotent_and_creates_one_slot(self):
        state = peg_channel.PegState(["A"], ["a"], np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
        self.apply(state, [("append", "B", "b1"), ("append", "B", "b2")])
        self.assertEqual(state.titles.count("B"), 1)
        self.assertEqual(state.texts[state.index["B"]], "b1")


if __name__ == "__main__":
    unittest.main()
