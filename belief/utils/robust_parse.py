"""
Robust JSON / code-block parsing for local-model outputs (Session 17).

Local models (qwen, llama, mistral, ...) routinely wrap JSON with
prose, drop closing braces, leave trailing commas, or dress the
whole thing up in markdown code fences.  The cloud client in
``belief/llm.py`` has its own strict parse + repair path; this
module exposes a **never-raising** counterpart that any caller can
drop into a build step without wrapping in try/except.

Guarantees:

* Functions here never raise on malformed input.  They return the
  supplied ``default`` and log at debug level.
* Extraction is best-effort — we try multiple strategies in order:
  markdown-fence strip → brace matching → trailing-comma repair →
  quote-repair → fallback to the default.
* No network, no LLM.  Pure string manipulation + ``json``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger("belief.utils.robust_parse")


# ── Strip markdown fences ─────────────────────────────────────────────────


_FENCE_OPEN_RE = re.compile(r"^```(?:json|JSON|python|\w+)?\s*\n?", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences, if any.

    Matches ```` ```json ```` / ```` ```python ```` / bare ```` ``` ````
    at the start, and any closing ```` ``` ```` at the end.  Leaves
    text inside unaffected.
    """
    if not text:
        return text
    cleaned = _FENCE_OPEN_RE.sub("", text.strip(), count=1)
    cleaned = _FENCE_CLOSE_RE.sub("", cleaned)
    return cleaned.strip()


# ── Brace extraction ──────────────────────────────────────────────────────


def _extract_first_json_blob(text: str) -> Optional[str]:
    """Find the first balanced ``{...}`` or ``[...]`` substring.

    Tolerates leading prose and trailing commentary.  String-literal
    awareness means braces inside JSON strings don't throw the
    counter off.  Returns ``None`` when no balanced blob is found.
    """
    if not text:
        return None
    candidates = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    candidates.append((start, text[start:i + 1]))
                    break
    if not candidates:
        return None
    # Return the earliest-starting balanced blob.
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# ── Small repairs ─────────────────────────────────────────────────────────


_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")
_SINGLE_QUOTE_KEY_RE = re.compile(r"([\{,]\s*)'([A-Za-z_][A-Za-z_0-9]*)'\s*:")
_BARE_KEY_RE = re.compile(r"([\{,]\s*)([A-Za-z_][A-Za-z_0-9]*)\s*:")


def _repair_small(s: str) -> str:
    """Apply the usual small fixes (trailing commas, quote styles)."""
    # Trailing commas before ] or }
    repaired = _TRAILING_COMMA_RE.sub(r"\1", s)
    # Single-quoted keys → double-quoted
    repaired = _SINGLE_QUOTE_KEY_RE.sub(r'\1"\2":', repaired)
    # Bare keys (identifier followed by colon inside an object) → "key":
    # Only applied if the above didn't already turn it into "key":
    repaired = _BARE_KEY_RE.sub(
        lambda m: f'{m.group(1)}"{m.group(2)}":'
                   if not m.group(0).lstrip("{, ").startswith('"')
                   else m.group(0),
        repaired,
    )
    return repaired


def _close_dangling(s: str) -> str:
    """Best-effort close for truncated structures.

    Walks the text tracking a stack of ``{``/``[`` and appends the
    matching closers in reverse order.  Ignores text inside strings.
    This is the last-ditch attempt — callers already tried balanced
    extraction.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for c in s:
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c == "}" and stack and stack[-1] == "}":
            stack.pop()
        elif c == "]" and stack and stack[-1] == "]":
            stack.pop()
    if in_string:
        s += '"'
    return s + "".join(reversed(stack))


# ── Public API ────────────────────────────────────────────────────────────


def try_parse_json(
    raw: Any, default: Any = None,
) -> Any:
    """Parse ``raw`` as JSON in the most forgiving way possible.

    Strategy (stops at the first success):

      1. ``raw`` is already a ``dict`` / ``list`` / ``int`` / ``float``
         / ``bool`` / ``None``  → return it as-is.
      2. Strip markdown code fences.
      3. ``json.loads`` the stripped text.
      4. Extract the first balanced ``{...}`` / ``[...]`` blob.
      5. Small repairs (trailing commas, unquoted keys) + retry.
      6. Close dangling brackets + retry.
      7. Return ``default``.

    Never raises — every exception is caught and logged at debug
    level.  Callers pass whatever "empty result" sentinel they want
    as ``default`` (often ``{}`` or ``[]``).
    """
    if raw is None:
        return default
    if isinstance(raw, (dict, list, bool, int, float)):
        return raw
    if not isinstance(raw, (str, bytes)):
        raw = str(raw)
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            return default

    text = strip_code_fences(raw)
    if not text:
        return default

    # Fast path — the whole thing is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Balanced-blob extraction.
    blob = _extract_first_json_blob(text)
    if blob is not None:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_small(blob))
            except json.JSONDecodeError:
                try:
                    return json.loads(_close_dangling(_repair_small(blob)))
                except json.JSONDecodeError as exc:
                    logger.debug(
                        f"robust_parse: every repair failed ({exc}); "
                        f"returning default"
                    )
                    return default

    # Nothing balanced — try repairs on the raw text.
    try:
        return json.loads(_close_dangling(_repair_small(text)))
    except json.JSONDecodeError as exc:
        logger.debug(
            f"robust_parse: no balanced blob and final repair failed "
            f"({exc}); returning default"
        )
        return default


def extract_code_block(
    raw: str, *, language: Optional[str] = None,
) -> Optional[str]:
    """Pull the first fenced code block out of ``raw``.

    Matches ```` ```python\n...\n``` ```` (or bare ```` ``` ```` when
    ``language`` is None).  Returns the block content (without the
    fences) or ``None`` when no match is found.

    Safe on arbitrary string inputs; never raises.
    """
    if not raw:
        return None
    if language:
        pattern = re.compile(
            rf"```{re.escape(language)}\s*\n(.*?)\n```",
            re.DOTALL | re.IGNORECASE,
        )
    else:
        pattern = re.compile(r"```\w*\s*\n(.*?)\n```", re.DOTALL)
    m = pattern.search(raw)
    if m is None:
        return None
    return m.group(1)
