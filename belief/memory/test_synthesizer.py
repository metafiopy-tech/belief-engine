"""Haiku-backed property-test synthesizer for belief competition.

Given a fresh build's code + its natural-language goal, ask Claude
Haiku (cheap) to emit 3-5 test inputs that exercise edge cases,
boundary conditions, and adversarial paths. The output format is the
same dict shape that :func:`belief.memory.trophic.compete` consumes:

    [{"description": str, "kwargs": str->Any, "expected": Any}, ...]

Design:
  - ``client`` is injectable. In production it's an Anthropic client
    wrapper (Session 5's BreakerAnthropic). In tests, callers pass a
    fake so nothing hits the network.
  - The client must expose a method ``generate_text(system, prompt,
    max_tokens)`` returning raw text. The synthesizer parses the first
    JSON array out of the response and drops malformed entries.
  - Cost guard: we ask for at most 5 tests and cap output at 1200
    tokens. Typical Haiku cost: ~\$0.01-\$0.02 per build.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol


logger = logging.getLogger("belief.memory.test_synthesizer")


HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TESTS = 5
DEFAULT_MAX_TOKENS = 1200


SYSTEM_PROMPT = """\
You are a property-test generator. Given a code artifact and its goal,
you produce a short JSON array of test inputs. Each entry:

    {"description": str,
     "kwargs": { ... },
     "expected": any}

- kwargs maps to positional-or-keyword args of the primary function.
- expected is the value the function should return.
- Favor edge cases and boundaries: zeros, negatives, empty collections,
  very long inputs, unicode, Off-By-One, domain-specific corners.

Return ONLY the JSON array. No markdown fences, no prose.
"""

_USER_TEMPLATE = """\
GOAL: {goal}

CODE (truncated to 4k chars):
---
{code}
---

Produce between 1 and {n_tests} test inputs. Return strict JSON array.
"""


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


class _LLMClient(Protocol):
    def generate_text(
        self, *, system: str, prompt: str, max_tokens: int
    ) -> str: ...  # noqa: E704


@dataclass
class SynthesisResult:
    tests: list[dict[str, Any]]
    raw: str = ""
    dropped: int = 0

    def __len__(self) -> int:
        return len(self.tests)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_tests(
    code_files: dict[str, str],
    goal: str,
    *,
    n_tests: int = DEFAULT_MAX_TESTS,
    client: Optional[_LLMClient] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SynthesisResult:
    """Ask the LLM for property-test inputs.

    Returns a :class:`SynthesisResult`; ``tests`` is the filtered list
    of well-formed entries, ``dropped`` counts malformed rows rejected
    by the schema validator. When ``client`` is None the function
    returns an empty result (no network call made) — useful in tests.
    """
    if client is None:
        return SynthesisResult(tests=[], raw="", dropped=0)

    code_blob = _pack_code(code_files)
    prompt = _USER_TEMPLATE.format(
        goal=str(goal or ""),
        code=code_blob,
        n_tests=max(1, int(n_tests)),
    )

    try:
        raw = client.generate_text(
            system=SYSTEM_PROMPT, prompt=prompt, max_tokens=max_tokens
        )
    except Exception as exc:
        logger.warning("synthesize_tests client call failed: %s", exc)
        return SynthesisResult(tests=[], raw="", dropped=0)

    parsed = _parse_tests_array(raw)
    if parsed is None:
        return SynthesisResult(tests=[], raw=raw, dropped=0)

    tests, dropped = _validate_entries(parsed, limit=int(n_tests))
    return SynthesisResult(tests=tests, raw=raw, dropped=dropped)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _pack_code(code_files: dict[str, str]) -> str:
    """Concatenate code files with separators, truncated to 4000 chars."""
    chunks: list[str] = []
    for name, body in (code_files or {}).items():
        chunks.append(f"### {name}\n{body}")
    blob = "\n\n".join(chunks)
    if len(blob) > 4000:
        blob = blob[:4000] + "\n...<truncated>"
    return blob


def _parse_tests_array(raw: str) -> Optional[list[Any]]:
    if not raw:
        return None
    # Try strict first
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list):
        return None
    return data


def _validate_entries(
    data: list[Any], *, limit: int
) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    dropped = 0
    for row in data:
        if not isinstance(row, dict):
            dropped += 1
            continue
        kwargs = row.get("kwargs")
        if not isinstance(kwargs, dict):
            dropped += 1
            continue
        if "expected" not in row:
            dropped += 1
            continue
        desc = str(row.get("description", "") or "")
        valid.append(
            {
                "description": desc[:160],
                "kwargs": kwargs,
                "expected": row["expected"],
            }
        )
        if len(valid) >= limit:
            break
    return valid, dropped


__all__ = [
    "HAIKU_MODEL",
    "SynthesisResult",
    "synthesize_tests",
]
