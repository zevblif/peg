"""Render the paper for GitHub: convert LaTeX-style delimiters to GitHub math.

GitHub only renders `$...$` (inline) and `$$...$$` (display); `\\( ... \\)` is shown
literally. The working copy keeps the LaTeX-style delimiters, and this script
produces the published snapshot under docs/.
"""
import argparse
import re
from pathlib import Path

INLINE = re.compile(r"\\\(([^\n]*?)\\\)")
DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.S)


def convert(text):
    text = DISPLAY.sub(lambda match: "\n$$\n" + match.group(1).strip() + "\n$$\n", text)
    text = INLINE.sub(
        lambda match: "$" + re.sub(r"[ \t]+", " ", match.group(1).strip()) + "$",
        text,
    )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="working copy with \\( ... \\)")
    parser.add_argument("--output", required=True, help="GitHub-rendered snapshot")
    args = parser.parse_args()

    source = Path(args.source)
    text = source.read_text(encoding="utf-8")
    converted = convert(text)
    Path(args.output).write_text(converted, encoding="utf-8")
    print({
        "inline_converted": len(INLINE.findall(text)),
        "display_converted": len(DISPLAY.findall(text)),
        "remaining_latex_delimiters": len(re.findall(r"\\[()\[\]]", converted)),
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
