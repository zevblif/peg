"""GraphRAG CLI entry point with an opt-in structured-output compatibility layer.

GraphRAG sends JSON-schema response formats for community reports. Some chat
providers reject that request shape. Setting PEG_DOWNGRADE_JSON_SCHEMA=1
downgrades those calls to {"type": "json_object"}; the outer completion layer
still parses the reply into the same pydantic model. Off by default, so the
stock host is unchanged.
"""
import json
import os
import sys
import time

import litellm

_sync = litellm.completion
_async = litellm.acompletion
_DOWNGRADE = os.environ.get("PEG_DOWNGRADE_JSON_SCHEMA") == "1"
_BLOCKED_DIR = os.environ.get("PEG_BLOCKED_DIR")
_COUNT_FILE = os.environ.get("PEG_COUNT_FILE")


def _count_call():
    """Opt-in call counter so a run's generation cost can be reported directly."""
    if not _COUNT_FILE:
        return
    with open(_COUNT_FILE, "a", encoding="utf-8") as stream:
        stream.write("1\n")


def _is_content_block(exc):
    text = str(exc).lower()
    return "prohibited or sensitive content" in text or "content_policy" in text


def _dump_blocked(kwargs):
    if not _BLOCKED_DIR:
        return None
    directory = os.path.abspath(_BLOCKED_DIR)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "blocked_%d.json" % int(time.time() * 1000))
    payload = {key: value for key, value in kwargs.items() if isinstance(value, (str, int, float, list, dict, type(None)))}
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
    return path


def _adapt(kwargs):
    if not _DOWNGRADE:
        return
    response_format = kwargs.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        kwargs["response_format"] = {"type": "json_object"}


def completion(*args, **kwargs):
    _adapt(kwargs)
    _count_call()
    try:
        return _sync(*args, **kwargs)
    except Exception as exc:
        if _is_content_block(exc):
            _dump_blocked(kwargs)
        raise


async def acompletion(*args, **kwargs):
    _adapt(kwargs)
    _count_call()
    try:
        return await _async(*args, **kwargs)
    except Exception as exc:
        if _is_content_block(exc):
            _dump_blocked(kwargs)
        raise


litellm.completion = completion
litellm.acompletion = acompletion

from graphrag.cli.main import app  # noqa: E402

if __name__ == "__main__":
    sys.exit(app())
