"""Render the Chinese paper snapshot for GitHub.

GitHub only renders `$...$` (inline) and `$$...$$` (display); `\\( ... \\)` is shown
literally. The working copy keeps the LaTeX-style delimiters, and this script
produces `docs/PEG-visibility-zh.md`. The English paper at
`docs/PEG-visibility.md` is maintained separately and is never overwritten by a
conversion; `--normalize-only` runs the spacing pass on it in place.

Inline math is padded with a space on both sides when it touches other text:
without the gap GitHub leaves some formulas unrendered when they sit directly
against CJK characters.
"""
import argparse
import re
from pathlib import Path

INLINE = re.compile(r"\\\(([^\n]*?)\\\)")
DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.S)
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)")


def space_inline_math(text):
    """Put a space before and after every inline `$...$` that touches text."""
    out, position = [], 0
    for match in INLINE_MATH.finditer(text):
        start, end = match.start(), match.end()
        out.append(text[position:start])
        if start > 0 and not text[start - 1].isspace():
            out.append(" ")
        out.append(match.group(0))
        if end < len(text) and not text[end].isspace():
            out.append(" ")
        position = end
    out.append(text[position:])
    return "".join(out)


def convert(text):
    text = DISPLAY.sub(lambda match: "\n$$\n" + match.group(1).strip() + "\n$$\n", text)
    text = INLINE.sub(
        lambda match: "$" + re.sub(r"[ \t]+", " ", match.group(1).strip()) + "$",
        text,
    )
    return space_inline_math(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="working copy with \\( ... \\)")
    parser.add_argument("--output", required=True, help="GitHub-rendered snapshot")
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="only add inline-math spacing; do not convert LaTeX delimiters",
    )
    args = parser.parse_args()

    source = Path(args.source)
    text = source.read_text(encoding="utf-8")
    converted = space_inline_math(text) if args.normalize_only else convert(text)
    Path(args.output).write_text(converted, encoding="utf-8")
    print({
        "inline_converted": 0 if args.normalize_only else len(INLINE.findall(text)),
        "display_converted": 0 if args.normalize_only else len(DISPLAY.findall(text)),
        "remaining_latex_delimiters": len(re.findall(r"\\[()\[\]]", converted)),
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
