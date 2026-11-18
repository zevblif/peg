"""Host delta behaviour that motivates the version channel.

GraphRAG v3.3.0 compares document *titles*: a same-title rewrite is not new input,
and a removed title is reported as a deletion the index does not consume.
"""
import asyncio
import unittest

import _bootstrap  # noqa: F401

import pandas as pd

try:
    from graphrag.index.update.incremental_index import get_delta_docs

    HAVE_GRAPHRAG = True
except Exception:  # pragma: no cover - dependency is installed for the benchmark
    HAVE_GRAPHRAG = False


class FakeDocuments:
    """Minimal stand-in for the table provider used by get_delta_docs."""

    def __init__(self, documents):
        self.documents = documents

    async def read_dataframe(self, name):
        if name != "documents":
            raise KeyError(name)
        return self.documents


def delta(previous, incoming):
    provider = FakeDocuments(pd.DataFrame(previous))
    return asyncio.run(get_delta_docs(pd.DataFrame(incoming), provider))


@unittest.skipUnless(HAVE_GRAPHRAG, "graphrag is not installed")
class HostDeltaTest(unittest.TestCase):
    def test_same_title_rewrite_is_not_new_input(self):
        previous = [{"id": "1", "title": "A", "text": "old"}]
        incoming = [{"title": "A", "text": "rewritten"}]
        result = delta(previous, incoming)
        self.assertEqual(len(result.new_inputs), 0)

    def test_new_title_is_new_input(self):
        previous = [{"id": "1", "title": "A", "text": "a"}]
        incoming = [{"title": "A", "text": "a"}, {"title": "B", "text": "b"}]
        result = delta(previous, incoming)
        self.assertEqual(list(result.new_inputs["title"]), ["B"])

    def test_removed_title_is_reported_as_deleted(self):
        previous = [{"id": "1", "title": "A", "text": "a"}, {"id": "2", "title": "B", "text": "b"}]
        incoming = [{"title": "A", "text": "a"}]
        result = delta(previous, incoming)
        self.assertEqual(list(result.deleted_inputs["title"]), ["B"])


if __name__ == "__main__":
    unittest.main()
