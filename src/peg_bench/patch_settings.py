"""Generate a GraphRAG settings.yaml for a benchmark run and install its input CSV.

The template keeps the pinned GraphRAG configuration; this script only fills in
paths, the chat model endpoint and the input file so a run is reproducible
without hand-editing YAML.
"""
import argparse
import shutil
from pathlib import Path

import yaml

from . import config


def patch(root, csv_path, template=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    template = Path(template or config.CONFIGS_DIR / "settings.template.yaml")
    text = template.read_text(encoding="utf-8")
    fills = {
        "__CHAT_PROVIDER__": config.CHAT_PROVIDER,
        "__CHAT_MODEL__": config.CHAT_MODEL,
        "__CHAT_API_BASE__": config.CHAT_API_BASE,
        "__EMBED_MODEL__": config.EMBED_MODEL,
        "__EMBED_API_BASE__": config.OLLAMA_BASE_URL,
    }
    for key, value in fills.items():
        text = text.replace(key, value)
    graph_config = yaml.safe_load(text)
    graph_config["concurrent_requests"] = config.CONCURRENT_REQUESTS
    graph_config.setdefault("embed_text", {})["batch_size"] = config.EMBED_BATCH
    if config.CHAT_DISABLE_THINKING:
        graph_config["completion_models"]["default_completion_model"]["call_args"]["enable_thinking"] = False
    text = yaml.safe_dump(graph_config, allow_unicode=True, sort_keys=False)
    settings_path = root / "settings.yaml"
    settings_path.write_text(text, encoding="utf-8")

    prompts = root / "prompts"
    if not prompts.exists():
        shutil.copytree(config.CONFIGS_DIR / "prompts", prompts)

    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("*"):
        if old.is_file():
            old.unlink()
    shutil.copy2(csv_path, input_dir / csv_path.name)
    print(f"patched {settings_path} with {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="GraphRAG project root for this run")
    parser.add_argument("--csv", required=True, help="input CSV to install")
    parser.add_argument("--template", default=None)
    args = parser.parse_args()
    patch(args.root, Path(args.csv), args.template)
