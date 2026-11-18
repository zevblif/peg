"""Evaluate the PEG-GR-001 arms on real GraphRAG v3.3.0 indexes.

Retrieval-only: builds the real local-search context without answer generation.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, peg_channel

DATA = config.DATA_MAIN
COMMUNITY_LEVEL = config.COMMUNITY_LEVEL


def load_frames(graph_config):
    from graphrag.data_model.data_reader import DataReader
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider

    storage = create_storage(graph_config.output_storage)
    provider = create_table_provider(graph_config.table_provider, storage=storage)
    reader = DataReader(provider)
    names = ["entities", "communities", "community_reports", "text_units", "relationships", "documents"]
    return {name: asyncio.run(getattr(reader, name)()) for name in names}


def build_retriever(root):
    from graphrag.config.embeddings import entity_description_embedding
    from graphrag.config.load_config import load_config
    from graphrag.query.factory import get_local_search_engine
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_reports,
        read_indexer_relationships,
        read_indexer_text_units,
    )
    from graphrag.utils.api import get_embedding_store, load_search_prompt

    previous = os.getcwd()
    try:
        graph_config = load_config(root_dir=Path(root))
        frames = load_frames(graph_config)
        store = get_embedding_store(graph_config.vector_store, entity_description_embedding)
        engine = get_local_search_engine(
            config=graph_config,
            reports=read_indexer_reports(frames["community_reports"], frames["communities"], COMMUNITY_LEVEL),
            text_units=read_indexer_text_units(frames["text_units"]),
            entities=read_indexer_entities(frames["entities"], frames["communities"], COMMUNITY_LEVEL),
            relationships=read_indexer_relationships(frames["relationships"]),
            covariates={"claims": []},
            response_type="single sentence",
            description_embedding_store=store,
            system_prompt=load_search_prompt(graph_config.local_search.prompt),
            callbacks=[],
        )
        text_map = {
            str(short): text
            for short, text in zip(frames["text_units"]["human_readable_id"], frames["text_units"]["text"])
        }
        doc_map = {
            str(short): doc
            for short, doc in zip(frames["text_units"]["human_readable_id"], frames["text_units"]["document_id"])
        }
        title_map = dict(zip(frames["documents"]["id"], frames["documents"]["title"]))
        return {
            "engine": engine,
            "text_map": text_map,
            "title_map": {short: title_map.get(doc) for short, doc in doc_map.items()},
        }
    finally:
        os.chdir(previous)


def graph_retrieve(retriever, question):
    started = time.perf_counter()
    context = retriever["engine"].context_builder.build_context(
        query=question, **retriever["engine"].context_builder_params
    )
    latency = (time.perf_counter() - started) * 1000
    sources = context.context_records.get("sources")
    titles, texts = [], []
    if sources is not None and len(sources):
        for short_id in sources["id"].tolist():
            title = retriever["title_map"].get(short_id)
            if title and title not in titles:
                titles.append(title)
                texts.append(retriever["text_map"].get(short_id, ""))
    return titles, texts, latency


def metrics(ranking, ground_truth):
    out = {}
    for k in (1, 5, 10):
        out[f"Hit@{k}"] = 1.0 if any(t in ranking[:k] for t in ground_truth) else 0.0
    reciprocal = 0.0
    for rank, title in enumerate(ranking, 1):
        if title in ground_truth:
            reciprocal = 1.0 / rank
            break
    out["MRR"] = reciprocal
    return out


def load_queries(name, data_dir=None):
    path = Path(data_dir or DATA) / f"queries_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_arm(name, retriever, peg_state, queries, use_graph, use_peg):
    rows = []
    for query in queries:
        graph_rank, graph_texts, graph_ms = ([], [], 0.0) if not use_graph else graph_retrieve(retriever, query["question"])
        peg_rank, peg_ms = ([], 0.0) if not use_peg else peg_state.search(query["question"], top_k=config.TOP_K)
        if use_graph and use_peg:
            ranking = peg_channel.rrf([graph_rank[:config.TOP_K], peg_rank[:config.TOP_K]])
        elif use_graph:
            ranking = graph_rank
        else:
            ranking = peg_rank
        row = {
            "arm": name,
            "qid": query["qid"],
            "question": query["question"],
            "ground_truth": query["supporting_titles"],
            "ranking": ranking[:config.TOP_K],
            "graph_rank": graph_rank[:config.TOP_K],
            "peg_rank": peg_rank[:config.TOP_K],
            "graph_ms": graph_ms,
            "peg_ms": peg_ms,
        }
        row.update(metrics(ranking, query["supporting_titles"]))
        rows.append(row)
    return rows


def summarize(rows, label):
    frame = pd.DataFrame(rows)
    return {
        "label": label,
        "n": int(len(frame)),
        **{key: float(frame[key].mean()) for key in ("Hit@1", "Hit@5", "Hit@10", "MRR")},
        "graph_ms_median": float(frame["graph_ms"].median()) if frame["graph_ms"].max() > 0 else 0.0,
        "graph_ms_p95": float(frame["graph_ms"].quantile(0.95)) if frame["graph_ms"].max() > 0 else 0.0,
        "peg_ms_median": float(frame["peg_ms"].median()) if frame["peg_ms"].max() > 0 else 0.0,
        "peg_ms_p95": float(frame["peg_ms"].quantile(0.95)) if frame["peg_ms"].max() > 0 else 0.0,
    }


def bootstrap_diff(rows_a, rows_b, key, n_boot=config.BOOTSTRAP_ROUNDS, seed=config.BOOTSTRAP_SEED):
    """Paired bootstrap over questions for B - A."""
    rng = np.random.default_rng(seed)
    a = np.asarray([row[key] for row in rows_a])
    b = np.asarray([row[key] for row in rows_b])
    diff = b - a
    samples = np.asarray([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {
        "mean_diff": float(diff.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--update-root", required=True)
    parser.add_argument("--peg-state", required=True)
    parser.add_argument("--edit-root", required=True)
    parser.add_argument("--peg-behavior-state", required=True)
    parser.add_argument("--data-dir", default=str(DATA))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "peg_gr_001.json"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    base = build_retriever(args.base_root)
    update = build_retriever(args.update_root)
    edit = build_retriever(args.edit_root)
    peg = peg_channel.PegState.load(args.peg_state)
    peg_behavior = peg_channel.PegState.load(args.peg_behavior_state)

    for retriever in (base, update, edit):
        graph_retrieve(retriever, "warmup query before measurement")
    peg.search("warmup query before measurement", top_k=config.TOP_K)
    peg_behavior.search("warmup query before measurement", top_k=config.TOP_K)

    base_queries = load_queries("base", data_dir)
    delta_queries = load_queries("delta", data_dir)
    all_queries = base_queries + delta_queries

    arms = {
        "A0_graph_base": (base, False, True),
        "A1_graph_update": (update, False, True),
        "A2_graph_base_plus_peg": (base, True, True),
        "A3_graph_update_plus_peg": (update, True, True),
        "PEG_only": (None, True, False),
    }
    rows, summary = [], []
    for name, (retriever, use_peg, use_graph) in arms.items():
        arm_rows = evaluate_arm(name, retriever, peg, all_queries, use_graph, use_peg)
        rows.extend(arm_rows)
        summary.append(summarize(arm_rows, name))

    # S2/S3 behavioral probes on the post-update index and the PEG state.
    edit_probes = [json.loads(line) for line in (data_dir / "edit_probes.jsonl").read_text(encoding="utf-8").splitlines()]
    delete_probes = [json.loads(line) for line in (data_dir / "delete_probes.jsonl").read_text(encoding="utf-8").splitlines()]
    behavior = {"edit": [], "delete": []}
    for probe in edit_probes:
        graph_rank, graph_texts, graph_ms = graph_retrieve(edit, probe["question"])
        peg_rank, peg_ms = peg_behavior.search(probe["question"], top_k=config.TOP_K)
        behavior["edit"].append({
            "title": probe["title"],
            "code": probe["code"],
            "graph_rank": graph_rank,
            "graph_code_visible": any(probe["code"] in text for text in graph_texts),
            "peg_rank": peg_rank,
            "peg_code_visible": any(
                probe["code"] in peg_behavior.texts[peg_behavior.index[t]] for t in peg_rank if t in peg_behavior.index
            ),
            "graph_ms": graph_ms,
            "peg_ms": peg_ms,
        })
    for probe in delete_probes:
        graph_rank, graph_texts, graph_ms = graph_retrieve(edit, probe["question"])
        peg_rank, peg_ms = peg_behavior.search(probe["question"], top_k=config.TOP_K)
        behavior["delete"].append({
            "title": probe["title"],
            "graph_stale_retained": probe["title"] in graph_rank,
            "peg_stale_retained": probe["title"] in peg_rank,
            "graph_ms": graph_ms,
            "peg_ms": peg_ms,
        })

    result = {
        "config": {
            "community_level": COMMUNITY_LEVEL,
            "rrf_k": config.RRF_K,
            "top_k": config.TOP_K,
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
        },
        "summary": summary,
        "paired_bootstrap": {
            "A3_vs_A1_Hit@5": bootstrap_diff(
                [row for row in rows if row["arm"] == "A1_graph_update"],
                [row for row in rows if row["arm"] == "A3_graph_update_plus_peg"],
                "Hit@5",
            ),
            "A3_vs_A1_MRR": bootstrap_diff(
                [row for row in rows if row["arm"] == "A1_graph_update"],
                [row for row in rows if row["arm"] == "A3_graph_update_plus_peg"],
                "MRR",
            ),
        },
        "behavior": {
            "edit_graph_visible": sum(item["graph_code_visible"] for item in behavior["edit"]),
            "edit_peg_visible": sum(item["peg_code_visible"] for item in behavior["edit"]),
            "delete_graph_stale": sum(item["graph_stale_retained"] for item in behavior["delete"]),
            "delete_peg_stale": sum(item["peg_stale_retained"] for item in behavior["delete"]),
            "edit_n": len(behavior["edit"]),
            "delete_n": len(behavior["delete"]),
            "detail": behavior,
        },
        "per_query": rows,
        "peg_cost": {"embed_ms": peg.embed_cost_ms, "append_ms": peg.append_cost_ms},
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {key: result[key] for key in ("summary", "paired_bootstrap", "behavior", "peg_cost")},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
