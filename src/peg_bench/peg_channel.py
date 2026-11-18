"""PEG paragraph-vector channel.

Real embeddings from a local OpenAI-compatible embedding endpoint; append,
replace and tombstone with a version mask. No training, no potential-field
iteration and no graph reasoning.
"""
import csv
import json
import time
from pathlib import Path

import httpx
import numpy as np

from . import config

_CLIENT = httpx.Client(base_url=config.OLLAMA_BASE_URL, timeout=180.0)


def embed_texts(texts, batch_size=16):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {"model": config.EMBED_MODEL, "input": batch}
        for attempt in range(3):
            try:
                response = _CLIENT.post("/v1/embeddings", json=payload)
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in data)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.asarray(vectors, dtype=np.float32)


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class PegState:
    def __init__(self, titles, texts, vectors, status=None, version=None):
        self.titles = list(titles)
        self.texts = list(texts)
        self.vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self.status = list(status) if status else ["active"] * len(titles)
        self.version = list(version) if version else [1] * len(titles)
        self.index = {title: i for i, title in enumerate(self.titles)}
        self.append_cost_ms = 0.0
        self.embed_cost_ms = 0.0

    @classmethod
    def from_csvs(cls, *paths):
        rows = []
        for path in paths:
            rows.extend(read_csv(path))
        started = time.perf_counter()
        vectors = embed_texts([row["text"] for row in rows])
        state = cls([row["title"] for row in rows], [row["text"] for row in rows], vectors)
        state.embed_cost_ms = (time.perf_counter() - started) * 1000
        return state

    def add(self, path):
        rows = [row for row in read_csv(path) if row["title"] not in self.index]
        if not rows:
            self.append_cost_ms = 0.0
            return 0
        started = time.perf_counter()
        vectors = embed_texts([row["text"] for row in rows])
        self.titles.extend(row["title"] for row in rows)
        self.texts.extend(row["text"] for row in rows)
        self.vectors = normalize(np.vstack([self.vectors, vectors]))
        self.status.extend(["active"] * len(rows))
        self.version.extend([1] * len(rows))
        self.index = {title: i for i, title in enumerate(self.titles)}
        self.append_cost_ms = (time.perf_counter() - started) * 1000
        return len(rows)

    def replace(self, path):
        rows = read_csv(path)
        started = time.perf_counter()
        vectors = embed_texts([row["text"] for row in rows])
        for row, vector in zip(rows, vectors):
            position = self.index[row["title"]]
            self.vectors[position] = normalize(vector.reshape(1, -1))[0]
            self.texts[position] = row["text"]
            self.version[position] += 1
            self.status[position] = "active"
        self.embed_cost_ms = (time.perf_counter() - started) * 1000
        return len(rows)

    def delete(self, path):
        rows = read_csv(path)
        for row in rows:
            self.status[self.index[row["title"]]] = "deleted"
        return len(rows)

    def search(self, question, top_k=10):
        started = time.perf_counter()
        query = normalize(embed_texts([question]))[0]
        scores = self.vectors @ query
        active = np.asarray(self.status) == "active"
        active_positions = np.flatnonzero(active)
        order = active_positions[np.argsort(-scores[active])[:top_k]]
        latency = (time.perf_counter() - started) * 1000
        return [self.titles[i] for i in order], latency

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            vectors=self.vectors,
            status=np.asarray(self.status),
            version=np.asarray(self.version),
        )
        path.with_suffix(".json").write_text(json.dumps({
            "titles": self.titles,
            "texts": self.texts,
            "append_cost_ms": self.append_cost_ms,
            "embed_cost_ms": self.embed_cost_ms,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        data = np.load(path.with_suffix(".npz"))
        state = cls(meta["titles"], meta["texts"], data["vectors"], list(data["status"]), list(data["version"]))
        state.append_cost_ms = meta.get("append_cost_ms", 0.0)
        state.embed_cost_ms = meta.get("embed_cost_ms", 0.0)
        return state


def rrf(rankings, k=config.RRF_K, top_k=config.TOP_K):
    scores = {}
    for ranking in rankings:
        for rank, title in enumerate(ranking, 1):
            scores[title] = scores.get(title, 0.0) + 1.0 / (k + rank)
    return [title for title, _ in sorted(scores.items(), key=lambda item: -item[1])[:top_k]]
