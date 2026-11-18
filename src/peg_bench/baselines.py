"""Strong baselines for the retrieval comparison.

Arms:
- bm25: lexical ranking over paragraph texts (base + delta corpus)
- vector_rag: paragraph cosine ranking over the same corpus, no version semantics
- basic_search: GraphRAG basic search context builder (text units), mapped to titles
- timestamp_filter: feasibility only; the host stores one index-time timestamp per
  document, not a per-document revision time, so a revision filter cannot be built
  from host metadata alone.
"""
import argparse
import asyncio
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, peg_channel


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.docs = [tokenize(text) for text in documents]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in self.docs) / max(len(self.docs), 1)
        self.freqs = [Counter(doc) for doc in self.docs]
        self.doc_freq = Counter()
        for doc in self.docs:
            self.doc_freq.update(set(doc))
        self.n = len(self.docs)

    def scores(self, query):
        terms = tokenize(query)
        out = np.zeros(self.n, dtype=np.float32)
        for index, freq in enumerate(self.freqs):
            length = len(self.docs[index])
            for term in terms:
                if term not in freq:
                    continue
                idf = math.log(1 + (self.n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                tf = freq[term]
                out[index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
        return out


def metrics(ranking, ground_truth):
    out = {}
    for k in (1, 5, 10):
        out["Hit@%d" % k] = 1.0 if any(t in ranking[:k] for t in ground_truth) else 0.0
    out["MRR"] = next((1.0 / (i + 1) for i, t in enumerate(ranking) if t in ground_truth), 0.0)
    return out


def load_queries(data_dir):
    rows = []
    for group in ("base", "delta"):
        for line in (Path(data_dir) / ("queries_%s.jsonl" % group)).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                rows.append({"qid": item["qid"], "group": group, "question": item["question"], "supporting_titles": item["supporting_titles"]})
    return rows


def corpus(data_dir):
    rows = []
    for name in ("base.csv", "delta.csv"):
        rows.extend(peg_channel.read_csv(Path(data_dir) / name))
    return rows


def summarize(rows, label):
    result = {"label": label}
    for group in ("base", "delta", "all"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        result[group] = {
            "n": len(selected),
            **{key: sum(row[key] for row in selected) / len(selected) for key in ("Hit@1", "Hit@5", "Hit@10", "MRR")},
        }
    return result


def build_basic_search(run_root):
    from graphrag.config.embeddings import text_unit_text_embedding
    from graphrag.config.load_config import load_config
    from graphrag.query.factory import get_basic_search_engine
    from graphrag.query.indexer_adapters import read_indexer_text_units
    from graphrag.utils.api import get_embedding_store, load_search_prompt

    import os

    previous = os.getcwd()
    try:
        graph_config = load_config(root_dir=Path(run_root))
        from graphrag.data_model.data_reader import DataReader
        from graphrag_storage import create_storage
        from graphrag_storage.tables.table_provider_factory import create_table_provider

        storage = create_storage(graph_config.output_storage)
        provider = create_table_provider(graph_config.table_provider, storage=storage)
        reader = DataReader(provider)
        text_units_frame = asyncio.run(reader.text_units())
        documents_frame = asyncio.run(reader.documents())
        store = get_embedding_store(graph_config.vector_store, text_unit_text_embedding)
        engine = get_basic_search_engine(
            text_units=read_indexer_text_units(text_units_frame),
            text_unit_embeddings=store,
            config=graph_config,
            response_type="single sentence",
            system_prompt=load_search_prompt(graph_config.basic_search.prompt),
            callbacks=[],
        )
        title_map = dict(zip(documents_frame["id"], documents_frame["title"]))
        unit_to_title = {
            str(unit): title_map.get(document)
            for unit, document in zip(text_units_frame["human_readable_id"], text_units_frame["document_id"])
        }
        # The vector store keys text units by their document id, not by the short id
        # the context builder prints, so both keys are needed to map results back.
        for unit, document in zip(text_units_frame["id"], text_units_frame["document_id"]):
            unit_to_title.setdefault(str(unit), title_map.get(document))
        return engine, unit_to_title
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(config.ARTIFACTS_DIR / "runs" / "update"))
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--peg-state", default=str(config.ARTIFACTS_DIR / "states" / "peg_base_delta"))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "baselines.json"))
    args = parser.parse_args()

    queries = load_queries(args.data_dir)
    rows_corpus = corpus(args.data_dir)
    texts = [row["text"] for row in rows_corpus]
    titles = [row["title"] for row in rows_corpus]

    report = {"config": {"run_root": args.run_root, "corpus_docs": len(texts), "top_k": config.TOP_K}, "arms": [], "timestamp_filter": {}}

    bm25 = BM25(texts)
    started = time.perf_counter()
    bm25_rows = []
    for query in queries:
        scores = bm25.scores(query["question"])
        order = np.argsort(-scores)[: config.TOP_K]
        ranking = [titles[i] for i in order]
        row = {"qid": query["qid"], "group": query["group"], "ranking": ranking}
        row.update(metrics(ranking, query["supporting_titles"]))
        bm25_rows.append(row)
    bm25_ms = (time.perf_counter() - started) * 1000 / len(queries)
    summary = summarize(bm25_rows, "bm25")
    summary["latency_ms_per_query"] = round(bm25_ms, 2)
    report["arms"].append(summary)

    state = peg_channel.PegState.load(args.peg_state)
    vector_rows = []
    latencies = []
    for query in queries:
        started = time.perf_counter()
        vector, latency = state.search(query["question"], top_k=config.TOP_K)
        latencies.append(latency)
        row = {"qid": query["qid"], "group": query["group"], "ranking": vector}
        row.update(metrics(vector, query["supporting_titles"]))
        vector_rows.append(row)
    summary = summarize(vector_rows, "vector_rag")
    summary["latency_ms_per_query"] = round(float(np.median(latencies)), 2)
    report["arms"].append(summary)

    try:
        engine, unit_to_title = build_basic_search(args.run_root)
        engine.context_builder.build_context(query="warmup", **engine.context_builder_params)
        basic_rows = []
        latencies = []
        for query in queries:
            started = time.perf_counter()
            context = engine.context_builder.build_context(query=query["question"], **engine.context_builder_params)
            latencies.append((time.perf_counter() - started) * 1000)
            sources = context.context_records.get("sources")
            ranking = []
            if sources is not None and len(sources):
                for unit in sources["id"].tolist():
                    title = unit_to_title.get(str(unit))
                    if title and title not in ranking:
                        ranking.append(title)
            row = {"qid": query["qid"], "group": query["group"], "ranking": ranking[: config.TOP_K]}
            row.update(metrics(ranking[: config.TOP_K], query["supporting_titles"]))
            basic_rows.append(row)
        summary = summarize(basic_rows, "basic_search")
        summary["latency_ms_per_query"] = round(float(np.median(latencies)), 2)
        summary["note"] = (
            "上下文构建器把命中的文本单元按文本单元表顺序输出，相似度顺序在返回的 DataFrame 中丢失；"
            "保留相似度顺序的同一引擎见 basic_search_ordered 臂。"
        )
        report["arms"].append(summary)

        # Same engine and same vector store, but the ranking is read from the
        # similarity search results instead of the re-sorted context DataFrame.
        store = engine.context_builder.text_unit_embeddings
        embed_one = lambda text: engine.context_builder.text_embedder.embedding(input=[text]).first_embedding

        ordered_rows = []
        latencies = []
        for query in queries:
            started = time.perf_counter()
            results = store.similarity_search_by_text(text=query["question"], text_embedder=embed_one, k=30)
            latencies.append((time.perf_counter() - started) * 1000)
            ranking = []
            for result in results:
                title = unit_to_title.get(str(result.document.id))
                if title and title not in ranking:
                    ranking.append(title)
                if len(ranking) >= config.TOP_K:
                    break
            row = {"qid": query["qid"], "group": query["group"], "ranking": ranking}
            row.update(metrics(ranking, query["supporting_titles"]))
            ordered_rows.append(row)
        summary = summarize(ordered_rows, "basic_search_ordered")
        summary["latency_ms_per_query"] = round(float(np.median(latencies)), 2)
        summary["note"] = "同一 basic search 引擎与向量库，按相似度结果顺序映射标题。"
        report["arms"].append(summary)
    except Exception as exc:  # keep the failure visible instead of hiding it
        report["arms"].append({"label": "basic_search", "error": str(exc)[:400]})

    # Direct text-unit vector retrieval: the same retrieval unit GraphRAG's basic
    # search uses, built here so the comparison does not depend on that engine.
    import pandas as pd

    units_frame = pd.read_parquet(Path(args.run_root) / "output" / "text_units.parquet")
    documents_frame = pd.read_parquet(Path(args.run_root) / "output" / "documents.parquet")
    unit_texts = units_frame["text"].fillna("").tolist()
    cache = config.ARTIFACTS_DIR / "text_unit_vectors_update.npz"
    if cache.exists():
        unit_vectors = np.load(cache)["vectors"]
    else:
        unit_vectors = peg_channel.normalize(peg_channel.embed_texts(unit_texts))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, vectors=unit_vectors)
    title_of_document = dict(zip(documents_frame["id"], documents_frame["title"]))
    unit_titles = [title_of_document.get(document) for document in units_frame["document_id"]]

    text_unit_rows = []
    latencies = []
    for query in queries:
        started = time.perf_counter()
        q = peg_channel.normalize(peg_channel.embed_texts([query["question"]]))[0]
        scores = unit_vectors @ q
        order = np.argsort(-scores)[:30]
        latencies.append((time.perf_counter() - started) * 1000)
        ranking = []
        for position in order:
            title = unit_titles[position]
            if title and title not in ranking:
                ranking.append(title)
            if len(ranking) >= config.TOP_K:
                break
        row = {"qid": query["qid"], "group": query["group"], "ranking": ranking}
        row.update(metrics(ranking, query["supporting_titles"]))
        text_unit_rows.append(row)
    summary = summarize(text_unit_rows, "text_unit_vector")
    summary["latency_ms_per_query"] = round(float(np.median(latencies)), 2)
    summary["note"] = "与宿主 basic search 相同的检索单位（文本单元），在此直接实现，避免依赖该引擎的向量库状态。"
    report["arms"].append(summary)

    documents = pd.read_parquet(Path(args.run_root) / "output" / "documents.parquet")
    report["timestamp_filter"] = {
        "feasible": False,
        "distinct_creation_dates": int(documents["creation_date"].nunique()),
        "reason": "宿主只记录索引时刻，不记录每篇文档的修订时间；同标题改写不更新该字段，删除的文档不保留记录。无法仅用宿主元数据构造修订时间过滤。",
    }

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
