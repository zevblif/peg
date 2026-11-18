"""Paths, endpoints and protocol constants in one place.

Every path is relative to the repository root and every endpoint can be
overridden through an environment variable, so a checkout runs on a new
machine without editing code.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_MAIN = Path(os.environ.get("PEG_DATA_MAIN", REPO_ROOT / "data" / "main"))
DATA_STABILITY = Path(os.environ.get("PEG_DATA_STABILITY", REPO_ROOT / "data" / "stability"))
RESULTS_DIR = Path(os.environ.get("PEG_RESULTS_DIR", REPO_ROOT / "results"))
ARTIFACTS_DIR = Path(os.environ.get("PEG_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CONFIGS_DIR = REPO_ROOT / "configs"

HOTPOT_DEV = Path(
    os.environ.get(
        "HOTPOT_DEV",
        REPO_ROOT / "data" / "hotpot" / "hotpot_dev_distractor_v1.json",
    )
)
HOTPOT_DEV_SHA256 = "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("PEG_EMBED_MODEL", "bge-m3:latest")
EMBED_BATCH = int(os.environ.get("PEG_EMBED_BATCH", "16"))

CHAT_PROVIDER = os.environ.get("PEG_CHAT_PROVIDER", "openai")
CHAT_MODEL = os.environ.get("PEG_CHAT_MODEL", "deepseek-chat")
CHAT_API_BASE = os.environ.get("PEG_CHAT_API_BASE", "https://api.deepseek.com/v1")
CONCURRENT_REQUESTS = int(os.environ.get("PEG_CONCURRENCY", "8"))
# Provider-specific switch: some reasoning models accept enable_thinking=false,
# which removes reasoning tokens and cuts per-call latency. Off by default so
# the template stays portable across OpenAI-compatible endpoints.
CHAT_DISABLE_THINKING = os.environ.get("PEG_CHAT_DISABLE_THINKING") == "1"

COMMUNITY_LEVEL = 2
RRF_K = 60
TOP_K = 10
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 17
FOLDS = 3


def require(path, hint):
    """Fail early with an actionable message instead of a bare FileNotFoundError."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing {path}\n{hint}")
    return path
