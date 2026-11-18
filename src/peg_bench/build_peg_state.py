"""Build the PEG paragraph-vector states used by the evaluation arms.

States are derived artifacts: rebuild them with this script instead of
committing them.
"""
import argparse
from pathlib import Path

from . import config, peg_channel


def build(name, out_dir):
    main, stability = config.DATA_MAIN, config.DATA_STABILITY
    if name == "base":
        state = peg_channel.PegState.from_csvs(main / "base.csv")
    elif name == "base_delta":
        state = peg_channel.PegState.from_csvs(main / "base.csv")
        state.add(main / "delta.csv")
    elif name == "edit_delete":
        state = peg_channel.PegState.from_csvs(main / "base.csv")
        state.add(main / "delta.csv")
        state.replace(main / "edit.csv")
        state.delete(main / "delete.csv")
    elif name == "stab_base":
        state = peg_channel.PegState.from_csvs(stability / "base.csv")
    elif name == "stab_base_delta":
        state = peg_channel.PegState.from_csvs(stability / "base.csv")
        state.add(stability / "delta.csv")
    else:
        raise SystemExit(f"unknown state: {name}")
    target = out_dir / f"peg_{name}"
    state.save(target)
    print(f"{name}: {len(state.titles)} paragraphs -> {target}.npz")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(config.ARTIFACTS_DIR / "states"))
    parser.add_argument("--states", nargs="+", default=["base", "base_delta", "edit_delete", "stab_base", "stab_base_delta"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    for name in args.states:
        build(name, out_dir)


if __name__ == "__main__":
    main()
