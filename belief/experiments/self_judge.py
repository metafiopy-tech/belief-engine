"""STARVED-arm self-judge — the build model rating its own output.

The STARVED admission gate never runs an external test. Instead the *same model
that produced the build* scores its own work through one frozen LLM-as-judge
call, returning a structured ``{self_score, self_confidence, rationale}``. The
experiment admits top-K by ``self_score`` and logs ``self_confidence`` alongside
(see ``docs/experiments/starved_arm_design.md`` §1).

Design constraints:

- **Frozen prompt + seed + temperature.** The judge is itself a model call, so a
  drifting prompt would measure judge drift, not soil decay. The prompt template
  is fixed here and exposes a ``SELF_JUDGE_PROMPT_FINGERPRINT`` (sha256 prefix)
  that the driver asserts is stable across pilot and full run — same discipline
  as the encoder fingerprint. The caller sets ``temperature=0`` / ``seed=42``
  (already the Ollama option defaults).

- **Injectable completion.** ``judge_build`` takes a ``generate`` callable
  (prompt -> raw text), so the prompt construction, parsing, and scoring are
  pure and unit-testable without a live model. The Session-4 driver supplies the
  real callable bound to the build's local model.

- **No external signal leaks in.** The judge sees only the goal and the produced
  code — never test results, covenant verdicts, or the hidden external grade.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping

# Frozen judge prompt. Edit ONLY between experiment runs, never mid-run — the
# fingerprint below changes if a single character moves, which the driver uses to
# refuse a run whose judge prompt drifted from the pilot's.
SELF_JUDGE_PROMPT = """\
You are reviewing code you just wrote, to decide whether it correctly and \
completely solves the task. You do NOT have the ability to run tests — judge \
from the code alone.

TASK:
{goal}

CODE YOU PRODUCED:
{code}

Rate your own work. Respond with ONLY a JSON object, no prose, no code fences:
{{"self_score": <float 0..1, how correct/complete you believe this is>, \
"self_confidence": <float 0..1, how sure you are of that judgment>, \
"rationale": "<one sentence>"}}"""

# Cap per-file and total code shown to the judge so a huge build doesn't blow the
# context window or make the call nondeterministic in length. The cap is part of
# the frozen contract (it affects the prompt), so it is fingerprinted with it.
_MAX_CHARS_PER_FILE = 4000
_MAX_TOTAL_CHARS = 24000


def _prompt_signature_source() -> str:
    """The exact text whose hash defines the frozen-prompt fingerprint."""
    return f"{SELF_JUDGE_PROMPT}||{_MAX_CHARS_PER_FILE}||{_MAX_TOTAL_CHARS}"


SELF_JUDGE_PROMPT_FINGERPRINT = hashlib.sha256(_prompt_signature_source().encode()).hexdigest()[:12]


@dataclass(frozen=True)
class JudgeResult:
    """Parsed self-judge output.

    ``self_score`` is the admission key; ``self_confidence`` is logged alongside.
    ``ok`` is False when the model returned unparseable output — callers decide
    how to treat a failed judge (the driver scores it 0 so a broken self-judge
    can never win a top-K slot).
    """

    self_score: float
    self_confidence: float
    rationale: str
    ok: bool = True
    raw: str = ""

    @property
    def prompt_fingerprint(self) -> str:
        return SELF_JUDGE_PROMPT_FINGERPRINT


def _render_code(code_files: Mapping[str, str]) -> str:
    """Render code files into the prompt under per-file and total caps."""
    parts: list[str] = []
    total = 0
    for fname in sorted(code_files):
        body = code_files[fname] or ""
        if len(body) > _MAX_CHARS_PER_FILE:
            body = body[:_MAX_CHARS_PER_FILE] + "\n# ... [truncated]"
        block = f"# === {fname} ===\n{body}"
        if total + len(block) > _MAX_TOTAL_CHARS:
            parts.append("# ... [remaining files omitted]")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "(no files produced)"


def build_judge_prompt(goal: str, code_files: Mapping[str, str]) -> str:
    """Construct the frozen self-judge prompt for one build."""
    return SELF_JUDGE_PROMPT.format(goal=goal.strip(), code=_render_code(code_files))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def parse_judge_response(raw: str) -> JudgeResult:
    """Robustly parse the judge's JSON, tolerating fences / surrounding prose.

    A failed parse (or out-of-range / missing fields) yields ``ok=False`` with a
    zero score, so a malformed self-judge cannot be admitted over a valid one.
    """
    text = (raw or "").strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # Otherwise grab the first balanced-looking {...} span.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = m.group(0) if m else text
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return JudgeResult(0.0, 0.0, "", ok=False, raw=raw)
    if not isinstance(data, dict) or "self_score" not in data:
        return JudgeResult(0.0, 0.0, "", ok=False, raw=raw)
    try:
        score = _clamp01(float(data["self_score"]))
        conf = _clamp01(float(data.get("self_confidence", 0.0)))
    except (TypeError, ValueError):
        return JudgeResult(0.0, 0.0, "", ok=False, raw=raw)
    rationale = str(data.get("rationale", ""))[:500]
    return JudgeResult(score, conf, rationale, ok=True, raw=raw)


def judge_build(
    goal: str,
    code_files: Mapping[str, str],
    generate: Callable[[str], str],
) -> JudgeResult:
    """Run the frozen self-judge on one build.

    ``generate`` maps the frozen prompt to raw model text (the driver binds it to
    the build's local model at temperature 0 / seed 42). Any exception from the
    model is treated as a failed judgment (``ok=False``, score 0) so the run
    never crashes on a flaky judge call.
    """
    prompt = build_judge_prompt(goal, code_files)
    try:
        raw = generate(prompt)
    except Exception as e:  # pragma: no cover - defensive
        return JudgeResult(0.0, 0.0, f"judge call failed: {e}", ok=False, raw="")
    return parse_judge_response(raw)
