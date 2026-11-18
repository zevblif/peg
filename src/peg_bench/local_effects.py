"""Local effects of replace and delete, plus search latency across state sizes.

Three measurements, all reuse existing PEG states and need no generation call:
- neighborhood: for each rewritten paragraph, Jaccard of its top-10 neighbours
  before and after the rewrite;
- delete side effects: how the other 40 query rankings change when five paragraphs
  are tombstoned;
- latency: query embedding plus inner-product ranking at 300, 400 and 1,200
  paragraphs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config, peg_channel


def neighbours(state, title, k=10):
    position = state.index[title]
    scores = state.vectors @ state.vectors[position]
    order = np.argsort(-scores)
    return [state.titles[i] for i in order if i != position][:k]


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left | right else 1.0


def load_queries(data_dir):
    rows = []
    for group in ("base", "delta"):
        for line in (Path(data_dir) / ("queries_%s.jsonl" % group)).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                rows.append({"qid": item["qid"], "group": group, "question": item["question"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "local_effects.json"))
    args = parser.parse_args()

    before = peg_channel.PegState.load(config.ARTIFACTS_DIR / "states" / "peg_base_delta")
    after = peg_channel.PegState.load(config.ARTIFACTS_DIR / "states" / "peg_edit_delete")
    edits = peg_channel.read_csv(Path(args.data_dir) / "edit.csv")
    deletes = peg_channel.read_csv(Path(args.data_dir) / "delete.csv")

    neighbourhood = []
    for row in edits:
        title = row["title"]
        before_neighbours = neighbours(before, title)
        after_neighbours = neighbours(after, title)
        neighbourhood.append({
            "title": title,
            "overlap": jaccard(before_neighbours, after_neighbours),
            "kept": sorted(set(before_neighbours) & set(after_neighbours)),
            "added": sorted(set(after_neighbours) - set(before_neighbours)),
            "removed": sorted(set(before_neighbours) - set(after_neighbours)),
        })

    deleted_titles = {row["title"] for row in deletes}
    queries = load_queries(args.data_dir)
    side_effects = []
    for query in queries:
        before_rank, latency_before = before.search(query["question"], top_k=config.TOP_K)
        after_rank, latency_after = after.search(query["question"], top_k=config.TOP_K)
        before_visible = [title for title in before_rank if title not in deleted_titles]
        side_effects.append({
            "qid": query["qid"],
            "group": query["group"],
            "slots_freed": sum(1 for title in before_rank if title in deleted_titles),
            "overlap_on_common": jaccard(before_visible, after_rank),
            "before": before_rank,
            "after": after_rank,
            "latency_before_ms": round(latency_before, 1),
            "latency_after_ms": round(latency_after, 1),
        })

    latency = []
    for name in ("peg_base", "peg_base_delta", "peg_stab_base_delta"):
        state = peg_channel.PegState.load(config.ARTIFACTS_DIR / "states" / name)
        timings = [state.search(query["question"], top_k=config.TOP_K)[1] for query in queries]
        latency.append({
            "state": name,
            "paragraphs": len(state.titles),
            "embed_plus_search_p50_ms": round(float(np.median(timings)), 1),
            "embed_plus_search_p95_ms": round(float(np.percentile(timings, 95)), 1),
        })

    report = {
        "config": {"edited": len(edits), "deleted": len(deletes), "top_k": config.TOP_K},
        "neighbourhood": {
            "mean_overlap": float(np.mean([item["overlap"] for item in neighbourhood])),
            "detail": neighbourhood,
        },
        "delete_side_effects": {
            "queries": len(side_effects),
            "mean_overlap_on_common": float(np.mean([item["overlap_on_common"] for item in side_effects])),
            "queries_with_freed_slots": sum(1 for item in side_effects if item["slots_freed"] > 0),
            "detail": side_effects,
        },
        "latency": latency,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "neighbourhood_mean_overlap": report["neighbourhood"]["mean_overlap"],
        "delete_mean_overlap": report["delete_side_effects"]["mean_overlap_on_common"],
        "queries_with_freed_slots": report["delete_side_effects"]["queries_with_freed_slots"],
        "latency": latency,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
