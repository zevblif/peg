"""Generation-level check: answer the same questions from each retrieval arm.

Retrieval metrics say whether the right paragraph is in the list; this script asks
whether the answer can be produced from that list. Three arms are compared under
the same prompt and model: PEG paragraphs, host local-search context, and their
reciprocal-rank fusion. Edit probes additionally check whether the rewritten
identifier appears in the answer.
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import litellm
import numpy as np

from . import config, peg_channel
from .evaluate import build_retriever, graph_retrieve

SYSTEM_PROMPT = (
    "You answer questions from the supplied context only. "
    "Reply with a short phrase or entity name, no explanation. "
    "If the context does not contain the answer, reply exactly: unknown"
)


def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    predicted = normalize(prediction).split()
    gold = normalize(reference).split()
    if not predicted or not gold:
        return 0.0
    common = 0
    remaining = list(gold)
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def build_context(titles, texts):
    blocks = []
    for index, (title, text) in enumerate(zip(titles, texts), 1):
        blocks.append(f"[{index}] {title}\n{text}")
    return "\n\n".join(blocks)


async def complete(question, context, semaphore):
    async with semaphore:
        response = await litellm.acompletion(
            model=f"openai/{config.CHAT_MODEL}",
            api_base=config.CHAT_API_BASE,
            api_key=os.environ["PEG_CHAT_API_KEY"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
            max_tokens=64,
            extra_body={"enable_thinking": False},
        )
    return (response.choices[0].message.content or "").strip()


def peg_context(state, question):
    titles, _ = state.search(question, top_k=config.TOP_K)
    texts = [state.texts[state.index[title]] for title in titles if title in state.index]
    return titles, texts


def fusion_context(host_titles, host_texts, peg_titles, peg_texts, state):
    ranking = peg_channel.rrf([host_titles[: config.TOP_K], peg_titles[: config.TOP_K]])
    host_map = dict(zip(host_titles, host_texts))
    titles, texts = [], []
    for title in ranking:
        if title in state.index:
            titles.append(title)
            texts.append(state.texts[state.index[title]])
        elif title in host_map:
            titles.append(title)
            texts.append(host_map[title])
    return titles, texts


async def run(queries, armpayloads, concurrency=8):
    semaphore = asyncio.Semaphore(concurrency)
    jobs = []
    for query, arms in zip(queries, armpayloads):
        for arm, (titles, texts) in arms.items():
            jobs.append((query, arm, complete(query["question"], build_context(titles, texts), semaphore)))
    results = await asyncio.gather(*(job[2] for job in jobs))
    rows = []
    for (query, arm, _), answer in zip(jobs, results):
        rows.append({
            "qid": query["qid"],
            "group": query["group"],
            "arm": arm,
            "answer": answer,
            "reference": query.get("answer", ""),
            "EM": exact_match(answer, query.get("answer", "")),
            "F1": token_f1(answer, query.get("answer", "")),
            "unknown": normalize(answer) == "unknown",
        })
    return rows


def summarize(rows):
    out = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        entry = {}
        for group in ("base", "delta", "all"):
            part = selected if group == "all" else [row for row in selected if row["group"] == group]
            entry[group] = {
                "n": len(part),
                "EM": float(np.mean([row["EM"] for row in part])),
                "F1": float(np.mean([row["F1"] for row in part])),
                "unknown_rate": float(np.mean([row["unknown"] for row in part])),
            }
        out[arm] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(config.ARTIFACTS_DIR / "runs" / "edit"))
    parser.add_argument("--peg-state", default=str(config.ARTIFACTS_DIR / "states" / "peg_base_delta"))
    parser.add_argument("--probe-state", default=str(config.ARTIFACTS_DIR / "states" / "peg_edit_delete"))
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "generation.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    queries = []
    for group in ("base", "delta"):
        for line in (data_dir / ("queries_%s.jsonl" % group)).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                queries.append({
                    "qid": item["qid"],
                    "group": group,
                    "question": item["question"],
                    "answer": item["answer"],
                })

    state = peg_channel.PegState.load(args.peg_state)
    probe_state = peg_channel.PegState.load(args.probe_state)
    retriever = build_retriever(args.run_root)
    graph_retrieve(retriever, "warmup query before measurement")

    payloads = []
    for query in queries:
        host_titles, host_texts, _ = graph_retrieve(retriever, query["question"])
        peg_titles, peg_texts = peg_context(state, query["question"])
        fused_titles, fused_texts = fusion_context(host_titles, host_texts, peg_titles, peg_texts, state)
        payloads.append({
            "PEG": (peg_titles, peg_texts),
            "HOST": (host_titles, host_texts),
            "FUSION": (fused_titles, fused_texts),
        })

    started = time.perf_counter()
    rows = asyncio.run(run(queries, payloads, args.concurrency))
    elapsed = time.perf_counter() - started

    probes = [json.loads(line) for line in (data_dir / "edit_probes.jsonl").read_text(encoding="utf-8").splitlines()]
    probe_payloads = []
    for probe in probes:
        host_titles, host_texts, _ = graph_retrieve(retriever, probe["question"])
        peg_titles, peg_texts = peg_context(probe_state, probe["question"])
        probe_payloads.append({
            "question": probe["question"],
            "code": probe["code"],
            "PEG": (peg_titles, peg_texts),
            "HOST": (host_titles, host_texts),
        })
    probe_jobs = []
    semaphore = asyncio.Semaphore(args.concurrency)
    for probe, arms in zip(probes, probe_payloads):
        for arm in ("PEG", "HOST"):
            titles, texts = arms[arm]
            probe_jobs.append((probe, arm, complete(probe["question"], build_context(titles, texts), semaphore)))
    probe_answers = asyncio.run(_gather(probe_jobs))

    probe_rows = []
    for (probe, arm, _), answer in zip(probe_jobs, probe_answers):
        probe_rows.append({
            "title": probe["title"],
            "code": probe["code"],
            "arm": arm,
            "answer": answer,
            "code_in_answer": probe["code"].lower() in answer.lower(),
        })

    report = {
        "config": {
            "run_root": args.run_root,
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
            "top_k": config.TOP_K,
            "concurrency": args.concurrency,
        },
        "cost": {
            "calls": len(rows) + len(probe_rows),
            "wall_s": round(elapsed, 1),
        },
        "summary": summarize(rows),
        "probes": {
            "n": len(probes),
            "by_arm": {
                arm: {
                    "code_in_answer": sum(row["code_in_answer"] for row in probe_rows if row["arm"] == arm),
                    "n": sum(1 for row in probe_rows if row["arm"] == arm),
                }
                for arm in ("PEG", "HOST")
            },
            "detail": probe_rows,
        },
        "per_query": rows,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cost", "summary", "probes")}, ensure_ascii=False, indent=2)[:3000])


async def _gather(jobs):
    return await asyncio.gather(*(job[2] for job in jobs))


if __name__ == "__main__":
    main()
