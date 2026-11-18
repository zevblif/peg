"""3-fold stability evaluation for PEG-GR-001."""
import argparse
import json
from pathlib import Path

from . import config, evaluate as ev


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rrf_peg_first(graph, peg, k=config.RRF_K, top=config.TOP_K):
    """Sensitivity variant: on equal RRF scores, PEG hits rank ahead of graph hits."""
    graph, peg = graph[:top], peg[:top]
    scores = {}
    for rank, title in enumerate(graph, 1):
        scores[title] = scores.get(title, 0.0) + 1.0 / (k + rank)
    for rank, title in enumerate(peg, 1):
        scores[title] = scores.get(title, 0.0) + 1.0 / (k + rank)

    def key(title):
        peg_rank = peg.index(title) if title in peg else 999
        graph_rank = graph.index(title) if title in graph else 999
        return (-scores[title], 0 if title in peg else 1, peg_rank, graph_rank)

    return sorted(scores, key=key)[:top]


def metrics(rank, ground_truth):
    return {
        "Hit@1": 1.0 if any(t in rank[:1] for t in ground_truth) else 0.0,
        "Hit@5": 1.0 if any(t in rank[:5] for t in ground_truth) else 0.0,
        "Hit@10": 1.0 if any(t in rank[:10] for t in ground_truth) else 0.0,
        "MRR": next((1.0 / (i + 1) for i, t in enumerate(rank) if t in ground_truth), 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--update-root", required=True)
    parser.add_argument("--peg-state", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "stability_3fold.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base = ev.build_retriever(args.base_root)
    update = ev.build_retriever(args.update_root)
    peg = ev.peg_channel.PegState.load(args.peg_state)
    for retriever in (base, update):
        ev.graph_retrieve(retriever, "warmup query before measurement")
    peg.search("warmup query before measurement", top_k=config.TOP_K)

    base_queries = load_jsonl(data_dir / "queries_base.jsonl")
    delta_queries = load_jsonl(data_dir / "queries_delta.jsonl")
    queries = base_queries + delta_queries
    folds = json.loads((data_dir / "folds.json").read_text(encoding="utf-8"))
    fold_of = {item["qid"]: item["fold"] for item in folds}

    arms = {
        "A0_graph_base": (base, False, True),
        "A1_graph_update": (update, False, True),
        "A2_graph_base_plus_peg": (base, True, True),
        "A3_graph_update_plus_peg": (update, True, True),
        "PEG_only": (None, True, False),
    }
    rows, summary = [], []
    for name, (retriever, use_peg, use_graph) in arms.items():
        arm_rows = ev.evaluate_arm(name, retriever, peg, queries, use_graph, use_peg)
        rows.extend(arm_rows)
        summary.append(ev.summarize(arm_rows, name))
        for fold in range(config.FOLDS):
            for group, group_queries in (("base", base_queries), ("delta", delta_queries)):
                ids = {q["qid"] for q in group_queries if fold_of[q["qid"]] == fold}
                selected = [row for row in arm_rows if row["qid"] in ids]
                if selected:
                    summary.append(ev.summarize(selected, f"{name}|fold{fold}|{group}"))

    def select(arm, fold=None, group=None):
        selected = [row for row in rows if row["arm"] == arm]
        if fold is not None:
            selected = [row for row in selected if fold_of[row["qid"]] == fold]
        if group == "base":
            ids = {q["qid"] for q in base_queries}
            selected = [row for row in selected if row["qid"] in ids]
        elif group == "delta":
            ids = {q["qid"] for q in delta_queries}
            selected = [row for row in selected if row["qid"] in ids]
        return selected

    bootstrap = {}
    for label, arm_a, arm_b in (
        ("A3-A1", "A1_graph_update", "A3_graph_update_plus_peg"),
        ("A3-PEG", "PEG_only", "A3_graph_update_plus_peg"),
        ("A1-A0", "A0_graph_base", "A1_graph_update"),
    ):
        bootstrap[label] = {}
        for fold in range(config.FOLDS):
            for group in ("base", "delta"):
                a, b = select(arm_a, fold, group), select(arm_b, fold, group)
                if a and b:
                    bootstrap[label][f"fold{fold}|{group}"] = {
                        key: ev.bootstrap_diff(a, b, key) for key in ("Hit@1", "Hit@5", "MRR")
                    }

    # Sensitivity: PEG-priority tie-break for fusion arms.
    sensitivity = {}
    for arm in ("A2_graph_base_plus_peg", "A3_graph_update_plus_peg"):
        for fold in range(config.FOLDS):
            for group in ("base", "delta"):
                selected = select(arm, fold, group)
                if not selected:
                    continue
                scored = []
                for row in selected:
                    ranked = rrf_peg_first(row["graph_rank"], row["peg_rank"])
                    scored.append(metrics(ranked, row["ground_truth"]))
                sensitivity[f"{arm}|fold{fold}|{group}"] = {
                    key: sum(item[key] for item in scored) / len(scored) for key in ("Hit@1", "Hit@5", "MRR")
                }

    result = {
        "config": {
            "folds": config.FOLDS,
            "base_questions": len(base_queries),
            "delta_questions": len(delta_queries),
            "community_level": config.COMMUNITY_LEVEL,
            "rrf_k": config.RRF_K,
            "top_k": config.TOP_K,
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
        },
        "summary": summary,
        "bootstrap": bootstrap,
        "sensitivity_peg_first": sensitivity,
        "per_query": rows,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {"summary": [item for item in summary if "|" not in item["label"]], "bootstrap": bootstrap},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
