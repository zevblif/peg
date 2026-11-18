"""Build the fixed HotpotQA dev subset used by the main experiment.

Deterministic, no network. Base and delta paragraph titles are disjoint so that
GraphRAG's title-based delta detection is meaningful.
"""
import argparse
import csv
import json
from pathlib import Path

from . import config

BASE_N = 30
DELTA_N = 10


def titles_of(question):
    return [title for title, _ in question["context"]]


def text_of(question):
    return {title: " ".join(sentences).strip() for title, sentences in question["context"]}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot", default=str(config.HOTPOT_DEV))
    parser.add_argument("--out", default=str(config.DATA_MAIN))
    parser.add_argument("--base-n", type=int, default=BASE_N)
    parser.add_argument("--delta-n", type=int, default=DELTA_N)
    args = parser.parse_args()

    source = config.require(args.hotpot, "set HOTPOT_DEV or pass --hotpot to the HotpotQA dev_distractor file")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(source.read_text(encoding="utf-8"))

    base, delta, used = [], [], set()
    for question in questions:
        names = titles_of(question)
        if len(names) != 10 or set(names) & used:
            continue
        support = [title for title, _ in question["supporting_facts"]]
        if not support or not set(support).issubset(names):
            continue
        if len(base) < args.base_n:
            base.append(question)
            used.update(names)
        elif len(delta) < args.delta_n:
            delta.append(question)
            used.update(names)
        else:
            break
    assert len(base) == args.base_n and len(delta) == args.delta_n

    base_rows, delta_rows = [], []
    for question in base:
        for title, text in text_of(question).items():
            base_rows.append({"title": title, "text": text})
    for question in delta:
        for title, text in text_of(question).items():
            delta_rows.append({"title": title, "text": text})
    assert len({row["title"] for row in base_rows}) == len(base_rows) == args.base_n * 10
    assert len({row["title"] for row in delta_rows}) == len(delta_rows) == args.delta_n * 10

    write_csv(out / "base.csv", base_rows)
    write_csv(out / "delta.csv", delta_rows)

    for name, group in (("base", base), ("delta", delta)):
        with (out / f"queries_{name}.jsonl").open("w", encoding="utf-8") as stream:
            for question in group:
                stream.write(json.dumps({
                    "qid": question["_id"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "supporting_titles": [title for title, _ in question["supporting_facts"]],
                }, ensure_ascii=False) + "\n")

    # S2: append a synthetic, uniquely identifiable fact to 10 base paragraphs.
    edit_rows, edit_probes = [], []
    for index, row in enumerate(base_rows[:10]):
        code = f"ZX-{index:02d}"
        edited = row["text"] + f" In 2026, the archive verification code for this entry is {code}."
        edit_rows.append({"title": row["title"], "text": edited})
        edit_probes.append({
            "title": row["title"],
            "code": code,
            "question": f"What is the archive verification code for {row['title']}?",
        })
    write_csv(out / "edit.csv", edit_rows)
    with (out / "edit_probes.jsonl").open("w", encoding="utf-8") as stream:
        for probe in edit_probes:
            stream.write(json.dumps(probe, ensure_ascii=False) + "\n")

    # S3: delete 5 base paragraphs (disjoint from the 10 edited rows) and keep probes.
    delete_rows = base_rows[10:15]
    write_csv(out / "delete.csv", delete_rows)
    with (out / "delete_probes.jsonl").open("w", encoding="utf-8") as stream:
        for row in delete_rows:
            probe = {"title": row["title"], "question": f"Summarize the entry about {row['title']}."}
            stream.write(json.dumps(probe, ensure_ascii=False) + "\n")

    selection = {
        "source": str(source),
        "source_sha256": config.HOTPOT_DEV_SHA256,
        "base_questions": [q["_id"] for q in base],
        "delta_questions": [q["_id"] for q in delta],
        "base_docs": len(base_rows),
        "delta_docs": len(delta_rows),
        "edit_docs": len(edit_rows),
        "delete_docs": len(delete_rows),
        "title_overlap_base_delta": len({row["title"] for row in base_rows} & {row["title"] for row in delta_rows}),
    }
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
