"""Mechanism ablation: is the gain from version awareness, or just an extra retriever?

Three paragraph-channel variants are compared on the same host rankings:

- ``version_aware``: replace a rewritten paragraph in place and tombstone deletes
  (the PEG channel used in the main experiment).
- ``append_only``: re-embed rewrites as extra entries and keep duplicates, which
  is what a naive "add another vector retriever" setup does.
- ``append_no_tombstone``: ``append_only`` plus deletes left active.

Host rankings are read from the published result file, so no index rebuild and no
extra generation calls are needed; only the rewritten paragraphs are embedded.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config, peg_channel


class VariantState:
    """Paragraph vectors with duplicate titles allowed."""

    def __init__(self, titles, texts, vectors, status):
        self.titles = list(titles)
        self.texts = list(texts)
        self.vectors = peg_channel.normalize(np.asarray(vectors, dtype=np.float32))
        self.status = list(status)

    def search(self, question, top_k=config.TOP_K):
        """Return the ranked titles and the position of each title's best entry."""
        query = peg_channel.normalize(peg_channel.embed_texts([question]))[0]
        scores = self.vectors @ query
        active = np.asarray(self.status) == "active"
        order = np.flatnonzero(active)[np.argsort(-scores[active])[:top_k]]
        ranking, position_of, seen = [], {}, set()
        for position in order:
            title = self.titles[position]
            if title in seen:
                continue
            seen.add(title)
            ranking.append(title)
            position_of[title] = position
        return ranking, position_of

    def search_raw(self, question, top_k=config.TOP_K):
        """Top-k positions without collapsing duplicate titles."""
        query = peg_channel.normalize(peg_channel.embed_texts([question]))[0]
        scores = self.vectors @ query
        active = np.asarray(self.status) == "active"
        return list(np.flatnonzero(active)[np.argsort(-scores[active])[:top_k]])


def duplicate_slot_rate(state, queries, top_k=config.TOP_K):
    """Share of retrieval slots wasted on a title already returned in the same list."""
    wasted, slots = 0, 0
    for query in queries:
        titles = [state.titles[i] for i in state.search_raw(query["question"], top_k)]
        slots += len(titles)
        wasted += len(titles) - len(set(titles))
    return wasted / slots if slots else 0.0


def version_aware_state(state, edit_rows, delete_rows, edits):
    """In-place replacement plus tombstones, mirroring the main experiment."""
    titles, texts = list(state.titles), list(state.texts)
    vectors = state.vectors.copy()
    status = list(state.status)
    for row, vector in zip(edit_rows, edits):
        position = titles.index(row["title"])
        texts[position] = row["text"]
        vectors[position] = peg_channel.normalize(vector.reshape(1, -1))[0]
        status[position] = "active"
    for row in delete_rows:
        status[titles.index(row["title"])] = "deleted"
    return VariantState(titles, texts, vectors, status)


def append_only_state(state, edit_rows, delete_rows, edits, tombstone):
    titles, texts = list(state.titles), list(state.texts)
    vectors = state.vectors.copy()
    status = list(state.status)
    for row, vector in zip(edit_rows, edits):
        titles.append(row["title"])
        texts.append(row["text"])
        vectors = np.vstack([vectors, peg_channel.normalize(vector.reshape(1, -1))])
        status.append("active")
    if not tombstone:
        for row in delete_rows:
            status[titles.index(row["title"])] = "active"
    else:
        for row in delete_rows:
            status[titles.index(row["title"])] = "deleted"
    return VariantState(titles, texts, vectors, status)


def metrics(ranking, ground_truth):
    out = {}
    for k in (1, 5, 10):
        out["Hit@%d" % k] = 1.0 if any(t in ranking[:k] for t in ground_truth) else 0.0
    out["MRR"] = next((1.0 / (i + 1) for i, t in enumerate(ranking) if t in ground_truth), 0.0)
    return out


def fusion(graph_rank, peg_rank):
    return peg_channel.rrf([graph_rank[:config.TOP_K], peg_rank[:config.TOP_K]])


def evaluate_variant(label, state, queries, graph_ranks, edit_probes, delete_probes):
    retrieval = []
    for query in queries:
        peg_rank, _ = state.search(query["question"])
        ranking = fusion(graph_ranks[query["qid"]], peg_rank)
        row = {"qid": query["qid"], "group": query["group"], "ranking": ranking}
        row.update(metrics(ranking, query["supporting_titles"]))
        retrieval.append(row)

    edit_hits = 0
    for probe in edit_probes:
        ranking, position_of = state.search(probe["question"])
        position = position_of.get(probe["title"])
        edit_hits += int(position is not None and probe["code"] in state.texts[position])
    delete_stale = 0
    for probe in delete_probes:
        ranking, _ = state.search(probe["question"])
        delete_stale += int(probe["title"] in ranking)

    summary = {}
    for group in ("base", "delta"):
        rows = [row for row in retrieval if row["group"] == group]
        summary[group] = {
            "n": len(rows),
            **{key: sum(row[key] for row in rows) / len(rows) for key in ("Hit@1", "Hit@5", "MRR")},
        }
    return {
        "label": label,
        "retrieval": summary,
        "edit_visible": edit_hits,
        "edit_n": len(edit_probes),
        "delete_stale": delete_stale,
        "delete_n": len(delete_probes),
        "duplicate_slot_rate": duplicate_slot_rate(state, queries),
        "per_query": retrieval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(config.RESULTS_DIR / "peg_gr_001.json"))
    parser.add_argument("--peg-state", default=str(config.ARTIFACTS_DIR / "states" / "peg_base_delta"))
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "ablation.json"))
    args = parser.parse_args()

    data = Path(args.data_dir)
    published = json.loads(Path(args.results).read_text(encoding="utf-8"))
    graph_ranks = {row["qid"]: row["graph_rank"] for row in published["per_query"] if row["arm"] == "A1_graph_update"}

    queries = []
    for group in ("base", "delta"):
        for line in (data / ("queries_%s.jsonl" % group)).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                queries.append({"qid": item["qid"], "question": item["question"],
                                "supporting_titles": item["supporting_titles"], "group": group})

    edit_rows = peg_channel.read_csv(data / "edit.csv")
    delete_rows = peg_channel.read_csv(data / "delete.csv")
    edits = peg_channel.embed_texts([row["text"] for row in edit_rows])

    state = peg_channel.PegState.load(args.peg_state)
    variants = {
        "version_aware": version_aware_state(state, edit_rows, delete_rows, edits),
        "append_only": append_only_state(state, edit_rows, delete_rows, edits, tombstone=True),
        "append_no_tombstone": append_only_state(state, edit_rows, delete_rows, edits, tombstone=False),
    }

    probe_edit = [json.loads(line) for line in (data / "edit_probes.jsonl").read_text(encoding="utf-8").splitlines()]
    probe_delete = [json.loads(line) for line in (data / "delete_probes.jsonl").read_text(encoding="utf-8").splitlines()]

    report = []
    for label, variant in variants.items():
        report.append(evaluate_variant(label, variant, queries, graph_ranks, probe_edit, probe_delete))
        print("%-24s base H1=%.3f delta H1=%.3f edit=%d/%d stale=%d/%d" % (
            label,
            report[-1]["retrieval"]["base"]["Hit@1"],
            report[-1]["retrieval"]["delta"]["Hit@1"],
            report[-1]["edit_visible"], report[-1]["edit_n"],
            report[-1]["delete_stale"], report[-1]["delete_n"],
        ))

    payload = {
        "config": {**published["config"], "host_rankings_from": Path(args.results).name},
        "variants": report,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
