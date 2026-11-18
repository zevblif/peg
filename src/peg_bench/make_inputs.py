"""Assemble the GraphRAG input CSVs for the update and edit/delete scenarios."""
import argparse
import csv
from pathlib import Path

from . import config


def read(directory, name):
    with (directory / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write(directory, name, rows):
    with (directory / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(config.DATA_MAIN))
    args = parser.parse_args()
    data = Path(args.data_dir)

    base = read(data, "base.csv")
    delta = read(data, "delta.csv")
    edit_rows = read(data, "edit.csv")
    edit = {row["title"]: row["text"] for row in edit_rows}
    delete = {row["title"] for row in read(data, "delete.csv")}

    write(data, "input_base_delta.csv", base + delta)

    after = []
    for row in base + delta:
        if row["title"] in delete:
            continue
        after.append({"title": row["title"], "text": edit.get(row["title"], row["text"])})
    write(data, "input_after_edit.csv", after)

    # Control B: the same rewrite re-ingested under a new title, while the old
    # title stays in the corpus (the host cannot retire it).
    rename = [{"title": row["title"] + " (2026 revision)", "text": row["text"]} for row in edit_rows]
    write(data, "rename.csv", rename)
    write(data, "input_rename.csv", base + delta + rename)

    print({
        "base": len(base),
        "delta": len(delta),
        "base_delta": len(base) + len(delta),
        "after_edit_delete": len(after),
        "renamed": len(rename),
        "input_rename": len(base) + len(delta) + len(rename),
        "edited": len(edit),
        "deleted": len(delete),
    })


if __name__ == "__main__":
    main()
