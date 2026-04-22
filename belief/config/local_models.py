"""Per-model Ollama configuration defaults.

Different local models need different context windows, prediction
budgets, and sampling settings for good results on the Belief Engine's
coding workloads.  This module keeps that knowledge in one table
instead of hard-coded in :class:`~belief.llm.AsyncOllamaClient`.

When the user sets ``BELIEF_LOCAL_MODEL=<name>`` and the name matches
a key here, :class:`AsyncOllamaClient` picks up the corresponding
``num_ctx`` / ``num_predict`` / ``keep_alive`` / ``temperature`` /
``repeat_penalty`` values as its defaults.  Explicit kwargs on the
client still win — this is only the *default* layer.

Model selection guide
~~~~~~~~~~~~~~~~~~~~~

+-------------------------+--------+------------+---------------------+
| Model                   | VRAM   | Throughput | When to pick it     |
+=========================+========+============+=====================+
| qwen2.5-coder:14b       | ~10 GB | slow       | Best overall on 16G |
+-------------------------+--------+------------+---------------------+
| qwen2.5-coder:7b        |  ~5 GB | fast       | Tier 1-2 builds,    |
|                         |        |            | speed > quality     |
+-------------------------+--------+------------+---------------------+
| qwen3:8b                |  ~6 GB | medium     | Newer arch, still   |
|                         |        |            | being validated     |
+-------------------------+--------+------------+---------------------+
"""

from __future__ import annotations

from typing import Any


LOCAL_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen2.5-coder:14b": {
        "num_ctx": 8192,
        "num_predict": 4096,
        "keep_alive": "30m",
        "temperature": 0.0,
        "repeat_penalty": 1.1,
        "notes": "Best quality for 16GB. Q4_K_M quantization.",
    },
    "qwen2.5-coder:7b": {
        "num_ctx": 16384,
        "num_predict": 4096,
        "keep_alive": "30m",
        "temperature": 0.0,
        "notes": "Faster, lower quality. Good for Tier 1-2.",
    },
    "qwen3:8b": {
        "num_ctx": 8192,
        "num_predict": 4096,
        "keep_alive": "30m",
        "temperature": 0.0,
        "notes": "Newer architecture. Test against 14b.",
    },
}


# Fallback applied when the caller's model isn't in the table above.
# Conservative defaults that work on a typical 16 GB M-series Mac.
DEFAULT_LOCAL_MODEL_CONFIG: dict[str, Any] = {
    "num_ctx": 8192,
    "num_predict": 4096,
    "keep_alive": "30m",
    "temperature": 0.0,
    "notes": "Fallback — unknown model. Override per-call if needed.",
}


def get_model_config(model: str) -> dict[str, Any]:
    """Return the config dict for ``model``, or the default fallback.

    The returned dict is a copy — callers may mutate it freely without
    affecting the module-level constant.
    """
    if model in LOCAL_MODEL_CONFIGS:
        return dict(LOCAL_MODEL_CONFIGS[model])
    return dict(DEFAULT_LOCAL_MODEL_CONFIG)
