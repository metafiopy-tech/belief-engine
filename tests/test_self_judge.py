"""Tests for the STARVED self-judge (frozen prompt + robust parser).

Gate-safe: no live model. The injectable ``generate`` callable lets us drive
``judge_build`` with canned responses.
"""

from __future__ import annotations

from belief.experiments.self_judge import (
    SELF_JUDGE_PROMPT_FINGERPRINT,
    JudgeResult,
    build_judge_prompt,
    judge_build,
    parse_judge_response,
)


def test_prompt_includes_goal_and_code():
    p = build_judge_prompt("Build a CLI", {"main.py": "print('hi')"})
    assert "Build a CLI" in p
    assert "main.py" in p
    assert "print('hi')" in p


def test_prompt_caps_huge_files():
    big = "x" * 10000
    p = build_judge_prompt("g", {"big.py": big})
    assert "[truncated]" in p
    assert len(p) < 10000  # capped well below the raw file size


def test_prompt_fingerprint_is_stable_and_short():
    assert isinstance(SELF_JUDGE_PROMPT_FINGERPRINT, str)
    assert len(SELF_JUDGE_PROMPT_FINGERPRINT) == 12
    # Same prompt for same inputs -> the fingerprint is independent of inputs.
    assert build_judge_prompt("a", {}) != build_judge_prompt("b", {})


def test_parse_plain_json():
    r = parse_judge_response('{"self_score": 0.8, "self_confidence": 0.6, "rationale": "ok"}')
    assert r.ok
    assert r.self_score == 0.8
    assert r.self_confidence == 0.6
    assert r.rationale == "ok"


def test_parse_json_in_code_fence():
    raw = 'Here:\n```json\n{"self_score": 0.5, "self_confidence": 0.9}\n```\nthanks'
    r = parse_judge_response(raw)
    assert r.ok
    assert r.self_score == 0.5
    assert r.self_confidence == 0.9


def test_parse_json_with_surrounding_prose():
    r = parse_judge_response('I think {"self_score": 0.3} overall.')
    assert r.ok
    assert r.self_score == 0.3
    assert r.self_confidence == 0.0  # missing -> default


def test_parse_clamps_out_of_range():
    r = parse_judge_response('{"self_score": 1.7, "self_confidence": -0.4}')
    assert r.self_score == 1.0
    assert r.self_confidence == 0.0


def test_parse_garbage_is_not_ok_and_zero():
    r = parse_judge_response("the model rambled with no json")
    assert not r.ok
    assert r.self_score == 0.0


def test_parse_missing_self_score_is_not_ok():
    r = parse_judge_response('{"confidence": 0.9}')
    assert not r.ok


def test_parse_non_numeric_score_is_not_ok():
    r = parse_judge_response('{"self_score": "high"}')
    assert not r.ok


def test_judge_build_composes_generate():
    def fake_generate(prompt: str) -> str:
        assert "Build X" in prompt  # received the real prompt
        return '{"self_score": 0.7, "self_confidence": 0.5, "rationale": "looks fine"}'

    r = judge_build("Build X", {"a.py": "pass"}, fake_generate)
    assert isinstance(r, JudgeResult)
    assert r.self_score == 0.7
    assert r.prompt_fingerprint == SELF_JUDGE_PROMPT_FINGERPRINT


def test_judge_build_survives_generate_exception():
    def boom(_):
        raise RuntimeError("model down")

    r = judge_build("g", {}, boom)
    assert not r.ok
    assert r.self_score == 0.0
    assert "model down" in r.rationale
