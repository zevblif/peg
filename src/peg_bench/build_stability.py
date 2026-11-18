"""Build the 3-fold stability subset: 90 base questions + 30 delta questions."""
import argparse
import csv
import json
from pathlib import Path

from . import config

BASE_N = 90
DELTA_N = 30


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot", default=str(config.HOTPOT_DEV))
    parser.add_argument("--out", default=str(config.DATA_STABILITY))
    parser.add_argument("--base-n", type=int, default=BASE_N)
    parser.add_argument("--delta-n", type=int, default=DELTA_N)
    args = parser.parse_args()

    source = config.require(args.hotpot, "set HOTPOT_DEV or pass --hotpot to the HotpotQA dev_distractor file")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(source.read_text(encoding="utf-8"))

    base, delta, used = [], [], set()
    for question in questions:
        names = [title for title, _ in question["context"]]
        if len(names) != 10 or set(names) & used:
            continue
        support = [title for title, _ in question["supporting_facts"]]
        if not support or not set(support).issubset(names):
            continue
        if len(base) < args.base_n:
            base.append(question)
        elif len(delta) < args.delta_n:
            delta.append(question)
        else:
            break
        used.update(names)
    assert len(base) == args.base_n and len(delta) == args.delta_n

    def rows(group):
        return [
            {"title": title, "text": " ".join(sentences).strip()}
            for question in group
            for title, sentences in question["context"]
        ]

    base_rows, delta_rows = rows(base), rows(delta)
    assert len({row["title"] for row in base_rows}) == len(base_rows) == args.base_n * 10
    assert len({row["title"] for row in delta_rows}) == len(delta_rows) == args.delta_n * 10
    assert not ({row["title"] for row in base_rows} & {row["title"] for row in delta_rows})

    write_csv(out / "base.csv", base_rows)
    write_csv(out / "delta.csv", delta_rows)
    write_csv(out / "input_base_delta.csv", base_rows + delta_rows)

    folds = []
    for group_name, group in (("base", base), ("delta", delta)):
        with (out / f"queries_{group_name}.jsonl").open("w", encoding="utf-8") as stream:
            for index, question in enumerate(group):
                stream.write(json.dumps({
                    "qid": question["_id"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "supporting_titles": [title for title, _ in question["supporting_facts"]],
                }, ensure_ascii=False) + "\n")
                folds.append({"qid": question["_id"], "group": group_name, "fold": index % config.FOLDS})
    (out / "folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "base_questions": len(base),
        "delta_questions": len(delta),
        "base_docs": len(base_rows),
        "delta_docs": len(delta_rows),
        "fold_sizes": {
            str(fold): {
                "base": sum(1 for item in folds if item["fold"] == fold and item["group"] == "base"),
                "delta": sum(1 for item in folds if item["fold"] == fold and item["group"] == "delta"),
            } for fold in range(config.FOLDS)
        },
    }
    (out / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
