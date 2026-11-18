"""Control designs, coverage, confidence intervals and entity-mapping analysis.

Three zero-generation controls ask whether a simpler side table reproduces PEG's
visibility:

- delete_filter_only: keep the stored paragraphs, drop deleted titles at query time.
- revision_filter_text: same filter, plus an external revision log that supplies the
  latest text; ranking still uses the pre-edit vector.
- revision_filter_reeembed: same log, and the edited paragraphs are re-embedded.

The script also reports multi-hop coverage from the stored per-query rankings and
bootstrap confidence intervals for each arm.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config, peg_channel


def load_queries(data_dir):
    rows = []
    for group in ("base", "delta"):
        path = Path(data_dir) / ("queries_%s.jsonl" % group)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                rows.append({
                    "qid": item["qid"],
                    "group": group,
                    "question": item["question"],
                    "supporting_titles": item["supporting_titles"],
                })
    return rows


def run_control(rows, probes, deleted, revision_texts, label, note):
    """One control arm: rank over the stored vectors, override text per revision."""
    probe_rows = []
    for probe in probes:
        kind = "delete" if "code" not in probe else "edit"
        scores = rows["vectors"] @ probe["query_vector"]
        order = np.argsort(-scores)
        ranking, texts = [], []
        for position in order:
            title = rows["titles"][position]
            if title in deleted:
                continue
            ranking.append(title)
            texts.append(revision_texts.get(title, rows["texts"][position]))
            if len(ranking) >= config.TOP_K:
                break
        rank = ranking.index(probe["title"]) + 1 if probe["title"] in ranking else None
        if kind == "edit":
            probe_rows.append({
                "kind": kind,
                "title": probe["title"],
                "rank": rank,
                "code_visible": any(probe["code"] in text for text in texts),
            })
        else:
            probe_rows.append({
                "kind": kind,
                "title": probe["title"],
                "rank": rank,
                "stale_retained": probe["title"] in ranking,
            })
    edit = [row for row in probe_rows if row["kind"] == "edit"]
    delete = [row for row in probe_rows if row["kind"] == "delete"]
    return {
        "label": label,
        "edit_code_visible": sum(row["code_visible"] for row in edit),
        "edit_n": len(edit),
        "delete_stale_retained": sum(row["stale_retained"] for row in delete),
        "delete_n": len(delete),
        "edit_rank": [row["rank"] for row in edit],
        "note": note,
    }


def coverage(per_query):
    """Share of queries whose full supporting set is inside top-k, plus mean recall."""
    out = {}
    for arm in sorted({row["arm"] for row in per_query}):
        selected = [row for row in per_query if row["arm"] == arm]
        entry = {"n": len(selected)}
        for k in (1, 5, 10):
            inside = [len(set(row["ground_truth"]) & set(row["ranking"][:k])) for row in selected]
            total = [len(set(row["ground_truth"])) for row in selected]
            entry["all@%d" % k] = float(np.mean([i == t for i, t in zip(inside, total)]))
            entry["recall@%d" % k] = float(np.mean([i / t for i, t in zip(inside, total)]))
        out[arm] = entry
    return out


def bootstrap_ci(values, rounds=config.BOOTSTRAP_ROUNDS, seed=config.BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    samples = values[rng.integers(0, len(values), (rounds, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def bootstrap_arms(per_query, group_of):
    out = {}
    for arm in sorted({row["arm"] for row in per_query}):
        selected = [row for row in per_query if row["arm"] == arm]
        out[arm] = {}
        for group in ("base", "delta", "all"):
            rows = selected if group == "all" else [row for row in selected if group_of[row["qid"]] == group]
            if not rows:
                continue
            out[arm][group] = {
                key: bootstrap_ci([row[key] for row in rows]) for key in ("Hit@1", "Hit@5", "MRR")
            }
            out[arm][group]["n"] = len(rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(config.ARTIFACTS_DIR / "runs" / "update"))
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--peg-state", default=str(config.ARTIFACTS_DIR / "states" / "peg_base_delta"))
    parser.add_argument("--paragraph-results", default=str(config.RESULTS_DIR / "peg_gr_001.json"))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "controls.json"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    corpus = peg_channel.read_csv(data_dir / "input_base_delta.csv")
    state = peg_channel.PegState.load(args.peg_state)
    titles = [row["title"] for row in corpus]
    rows = {
        "titles": titles,
        "texts": [row["text"] for row in corpus],
        "vectors": np.stack([state.vectors[state.index[title]] for title in titles]),
    }

    edits = peg_channel.read_csv(data_dir / "edit.csv")
    deleted = {row["title"] for row in peg_channel.read_csv(data_dir / "delete.csv")}
    revision_texts = {row["title"]: row["text"] for row in edits}

    probes = []
    for item in map(json.loads, (data_dir / "edit_probes.jsonl").read_text(encoding="utf-8").splitlines()):
        probes.append({**item, "query_vector": peg_channel.normalize(peg_channel.embed_texts([item["question"]]))[0]})
    for item in map(json.loads, (data_dir / "delete_probes.jsonl").read_text(encoding="utf-8").splitlines()):
        probes.append({**item, "query_vector": peg_channel.normalize(peg_channel.embed_texts([item["question"]]))[0]})

    controls = [
        run_control(
            rows, probes, deleted, {}, "delete_filter_only",
            "保留旧正文与旧向量，仅在查询时排除已删除标题；不改写文档（对照组 C）。",
        ),
        run_control(
            rows, probes, deleted, revision_texts, "revision_filter_text",
            "外部修订表提供最新正文，排序仍用改写前向量；不重新嵌入（对照组 A，无重嵌入）。",
        ),
    ]

    edit_vectors = peg_channel.normalize(peg_channel.embed_texts([row["text"] for row in edits]))
    reeembed = {
        "titles": titles,
        "texts": [revision_texts.get(title, rows["texts"][position]) for position, title in enumerate(titles)],
        "vectors": rows["vectors"].copy(),
    }
    for row, vector in zip(edits, edit_vectors):
        reeembed["vectors"][titles.index(row["title"])] = vector
    controls.append(run_control(
        reeembed, probes, deleted, revision_texts, "revision_filter_reeembed",
        "外部修订表 + 逐篇重嵌入被改写段落；可见性与 PEG 相同，但无标题唯一、墓碑与版本计数等结构。",
    ))

    paragraph = json.loads(Path(args.paragraph_results).read_text(encoding="utf-8"))
    group_of = {query["qid"]: query["group"] for query in load_queries(data_dir)}
    report = {
        "config": {
            "corpus_docs": len(titles),
            "top_k": config.TOP_K,
            "embed_model": config.EMBED_MODEL,
            "edited_docs": len(edits),
            "deleted_docs": len(deleted),
            "generation_calls": 0,
        },
        "controls": controls,
        "coverage": coverage(paragraph["per_query"]),
        "bootstrap": bootstrap_arms(paragraph["per_query"], group_of),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {key: report[key] for key in ("controls", "coverage")},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
