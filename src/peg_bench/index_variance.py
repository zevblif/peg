"""Index variance: rebuild the same corpus several times and compare host retrieval.

The host's extraction and community steps are stochastic (sampling, parallel calls),
so its index differs between rebuilds of identical input. PEG's paragraph channel
depends only on the embeddings of fixed texts, so its numbers are identical across
rebuilds; this script reports both sides on the same question set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .evaluate import build_retriever, graph_retrieve, load_queries, metrics


def index_counts(run_root, calls_file=None):
    output = Path(run_root) / "output"
    frame = {
        "documents": int(len(pd.read_parquet(output / "documents.parquet"))),
        "text_units": int(len(pd.read_parquet(output / "text_units.parquet"))),
        "entities": int(len(pd.read_parquet(output / "entities.parquet"))),
        "communities": int(len(pd.read_parquet(output / "communities.parquet"))),
    }
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    frame["index_runtime_s"] = round(float(stats.get("total_runtime", 0.0)), 1)
    if calls_file and Path(calls_file).exists():
        frame["generation_calls"] = sum(1 for _ in Path(calls_file).read_text(encoding="utf-8").splitlines())
    return frame


def evaluate_host(run_root, queries):
    retriever = build_retriever(run_root)
    graph_retrieve(retriever, "warmup query before measurement")
    rows = []
    for query in queries:
        titles, _, latency = graph_retrieve(retriever, query["question"])
        row = {"qid": query["qid"], "group": query["group"], "ranking": titles[: config.TOP_K]}
        row.update(metrics(titles, query["supporting_titles"]))
        row["latency_ms"] = round(latency, 1)
        rows.append(row)
    summary = {}
    for group in ("base", "delta", "all"):
        part = rows if group == "all" else [row for row in rows if row["group"] == group]
        summary[group] = {
            "n": len(part),
            **{key: float(np.mean([row[key] for row in part])) for key in ("Hit@1", "Hit@5", "MRR")},
        }
    return summary, rows


def spread(values):
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--calls-files", nargs="*", default=None)
    parser.add_argument("--peg-results", default=str(config.RESULTS_DIR / "peg_gr_001.json"))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "rebuild_variance.json"))
    args = parser.parse_args()

    queries = []
    for group in ("base", "delta"):
        queries.extend(dict(query, group=group) for query in load_queries(group, args.data_dir))

    runs = []
    calls_files = args.calls_files or [str(Path(root) / "calls.txt") for root in args.roots]
    for root, calls in zip(args.roots, calls_files):
        summary, rows = evaluate_host(root, queries)
        runs.append({
            "root": root,
            "counts": index_counts(root, calls),
            "summary": summary,
            "per_query": rows,
        })

    groups = ("base", "delta", "all")
    metrics_spread = {
        group: {
            key: spread([run["summary"][group][key] for run in runs])
            for key in ("Hit@1", "Hit@5", "MRR")
        }
        for group in groups
    }

    peg = json.loads(Path(args.peg_results).read_text(encoding="utf-8"))
    peg_summary = next(item for item in peg["summary"] if item["label"] == "PEG_only")

    report = {
        "config": {"roots": args.roots, "top_k": config.TOP_K, "corpus": "data/main/base.csv"},
        "runs": [{key: run[key] for key in ("root", "counts", "summary")} for run in runs],
        "host_spread": metrics_spread,
        "index_counts_spread": {
            key: spread([run["counts"][key] for run in runs])
            for key in ("entities", "communities", "text_units", "generation_calls", "index_runtime_s")
        },
        "peg_reference": {
            "note": "段落层只依赖固定文本的嵌入，重建不改变其指标；此处为同题集上的既有结果。",
            **{key: peg_summary[key] for key in ("Hit@1", "Hit@5", "Hit@10", "MRR")},
        },
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "runs": report["runs"],
        "host_spread": report["host_spread"],
        "index_counts_spread": report["index_counts_spread"],
        "peg_reference": report["peg_reference"],
    }, ensure_ascii=False, indent=2)[:3500])


if __name__ == "__main__":
    main()
