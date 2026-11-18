"""Integration cost: build a PEG state from an existing host index.

The measurement covers reading the host document table and embedding the paragraph
texts. It produces no generation call and does not rebuild the host.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import config, peg_channel


def measure(run_root):
    started = time.perf_counter()
    documents = pd.read_parquet(Path(run_root) / "output" / "documents.parquet")
    titles = documents["title"].tolist()
    texts = documents["text"].fillna("").tolist()
    read_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    vectors = peg_channel.normalize(peg_channel.embed_texts(texts))
    embed_ms = (time.perf_counter() - started) * 1000

    state = peg_channel.PegState(titles, texts, vectors)
    return {
        "run_root": str(run_root),
        "paragraphs": len(titles),
        "read_ms": round(read_ms, 1),
        "embed_ms": round(embed_ms, 1),
        "total_ms": round(read_ms + embed_ms, 1),
        "generation_calls": 0,
        "host_documents": len(documents),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", nargs="+", default=[str(config.ARTIFACTS_DIR / "runs" / "base"),
                                                           str(config.ARTIFACTS_DIR / "runs" / "edit")])
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "integration.json"))
    args = parser.parse_args()

    runs = [measure(Path(root)) for root in args.run_roots]
    report = {"config": {"embed_model": config.EMBED_MODEL}, "runs": runs}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
