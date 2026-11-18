"""Integration cost: build a PEG state from an existing host index.

The measurement covers reading the host document table, embedding the paragraph
texts, and building the state. It produces no generation call and does not
rebuild the host.

Protocol: one warm-up embedding pass is discarded, then every repeat re-reads the
table and re-embeds from scratch; the report carries the median together with the
minimum and maximum so the spread stays visible. Model and batch size are
recorded because absolute seconds depend on both.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd

from . import config, peg_channel

EMBED_BATCH = 16


def display_path(path):
    """Report repository-relative paths; fall back to absolute for outside roots."""
    try:
        return str(Path(path).resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def read_table(run_root):
    documents = pd.read_parquet(Path(run_root) / "output" / "documents.parquet")
    return documents["title"].tolist(), documents["text"].fillna("").tolist()


def measure(run_root, repeats):
    titles, texts = read_table(run_root)
    warm_texts = texts[:EMBED_BATCH]
    peg_channel.embed_texts(warm_texts, batch_size=EMBED_BATCH)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        titles, texts = read_table(run_root)
        read_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        vectors = peg_channel.normalize(peg_channel.embed_texts(texts, batch_size=EMBED_BATCH))
        embed_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        peg_channel.PegState(titles, texts, vectors)
        state_ms = (time.perf_counter() - started) * 1000

        samples.append({
            "read_ms": round(read_ms, 1),
            "embed_ms": round(embed_ms, 1),
            "state_ms": round(state_ms, 1),
            "total_ms": round(read_ms + embed_ms + state_ms, 1),
        })

    def summary(key):
        values = [sample[key] for sample in samples]
        return {
            f"{key}_median": round(statistics.median(values), 1),
            f"{key}_min": round(min(values), 1),
            f"{key}_max": round(max(values), 1),
        }

    return {
        "run_root": display_path(run_root),
        "paragraphs": len(titles),
        "embed_dim": int(vectors.shape[1]),
        "generation_calls": 0,
        **summary("read_ms"),
        **summary("embed_ms"),
        **summary("state_ms"),
        **summary("total_ms"),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", nargs="+", default=[str(config.ARTIFACTS_DIR / "runs" / "base"),
                                                           str(config.ARTIFACTS_DIR / "runs" / "edit")])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "integration.json"))
    args = parser.parse_args()

    runs = [measure(Path(root), args.repeats) for root in args.run_roots]
    report = {
        "config": {
            "embed_model": config.EMBED_MODEL,
            "embed_batch": EMBED_BATCH,
            "repeats": args.repeats,
            "warmup": True,
        },
        "runs": runs,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
