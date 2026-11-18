"""Smoke test for the local embedding endpoint.

Skipped when the endpoint is unreachable, so the suite stays offline-safe.
"""
import unittest

import _bootstrap  # noqa: F401

import httpx

from peg_bench import config


def endpoint_ready():
    try:
        client = httpx.Client(base_url=config.OLLAMA_BASE_URL, timeout=3.0)
        return client.get("/api/tags").status_code == 200
    except Exception:
        return False


@unittest.skipUnless(endpoint_ready(), f"no embedding endpoint at {config.OLLAMA_BASE_URL}")
class EmbedEndpointTest(unittest.TestCase):
    def test_embeddings_return_expected_dimension(self):
        from peg_bench import peg_channel

        vectors = peg_channel.embed_texts(["version channel smoke test"])
        self.assertEqual(vectors.shape[0], 1)
        self.assertGreaterEqual(vectors.shape[1], 256)


if __name__ == "__main__":
    unittest.main()
