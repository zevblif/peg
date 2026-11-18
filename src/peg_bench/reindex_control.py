"""Control B: rewrite by re-ingesting the edited document under a new title.

The host decides new documents by title, so renaming is the only way to make a
same-title rewrite enter the index without touching the host. This script measures
what that costs and what it leaves behind: whether the new title (and its body) is
retrievable, whether the old title is still retrievable, and how many generation
calls plus how much index state the rename produced.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .evaluate import build_retriever, graph_retrieve


def load_probes(data_dir):
    rows = []
    for line in (Path(data_dir) / "edit_probes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def counts(run_root):
    run_root = Path(run_root)
    output = run_root / "output"
    frame = {
        "documents": int(len(pd.read_parquet(output / "documents.parquet"))),
        "text_units": int(len(pd.read_parquet(output / "text_units.parquet"))),
        "entities": int(len(pd.read_parquet(output / "entities.parquet"))),
        "communities": int(len(pd.read_parquet(output / "communities.parquet"))),
    }
    try:
        import lancedb

        db = lancedb.connect(str(output / "lancedb"))
        frame["text_unit_vectors"] = int(db.open_table("text_unit_text").count_rows())
    except Exception:
        frame["text_unit_vectors"] = None
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    frame["total_runtime_s"] = round(float(stats.get("total_runtime", 0.0)), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(config.ARTIFACTS_DIR / "runs" / "rename"))
    parser.add_argument("--previous-root", default=str(config.ARTIFACTS_DIR / "runs" / "update"))
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--calls-file", default=None)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "reindex.json"))
    args = parser.parse_args()

    retriever = build_retriever(args.run_root)
    graph_retrieve(retriever, "warmup query before measurement")

    indexed_titles = set(pd.read_parquet(Path(args.run_root) / "output" / "documents.parquet")["title"])
    rows = []
    for probe in load_probes(args.data_dir):
        new_title = probe["title"] + " (2026 revision)"
        titles, texts, latency = graph_retrieve(retriever, probe["question"])
        rows.append({
            "title": probe["title"],
            "new_title": new_title,
            "code_visible": any(probe["code"] in text for text in texts),
            "new_title_retrieved": new_title in titles,
            "old_title_still_retrieved": probe["title"] in titles,
            "duplicate_slot": new_title in titles and probe["title"] in titles,
            "old_title_in_index": probe["title"] in indexed_titles,
            "new_title_in_index": new_title in indexed_titles,
            "ranking": titles[: config.TOP_K],
            "latency_ms": round(latency, 1),
        })

    calls = None
    if args.calls_file and Path(args.calls_file).exists():
        calls = sum(1 for _ in Path(args.calls_file).read_text(encoding="utf-8").splitlines())

    report = {
        "config": {"run_root": args.run_root, "previous_root": args.previous_root, "top_k": config.TOP_K},
        "cost": {
            "generation_calls": calls,
            "documents_added": 10,
            "before": counts(args.previous_root),
            "after": counts(args.run_root),
        },
        "behavior": {
            "n": len(rows),
            "code_visible": sum(row["code_visible"] for row in rows),
            "new_title_retrieved": sum(row["new_title_retrieved"] for row in rows),
            "old_title_still_retrieved": sum(row["old_title_still_retrieved"] for row in rows),
            "duplicate_slot": sum(row["duplicate_slot"] for row in rows),
            "old_title_in_index": sum(row["old_title_in_index"] for row in rows),
            "new_title_in_index": sum(row["new_title_in_index"] for row in rows),
            "median_latency_ms": float(np.median([row["latency_ms"] for row in rows])),
            "detail": rows,
        },
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cost", "behavior")}, ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()
