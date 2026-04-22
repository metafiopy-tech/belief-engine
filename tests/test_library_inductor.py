"""belief.memory.library_inductor — apex-predator promotion tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from belief.memory.library_inductor import (
    Candidate,
    MAX_PROMOTIONS_PER_CYCLE,
    MIN_SUCCESSFUL_USES,
    MIN_TROPHIC_LEVEL,
    NamingResult,
    promote_apex_predator,
    promote_eligible,
)
from belief.memory.tool_registry import SelfAuthoredTool


VALID_CODE = (
    "def normalize_whitespace(text: str) -> str:\n"
    "    return ' '.join(text.split())\n"
)

VALID_NAMING = json.dumps(
    {
        "name": "normalize_whitespace",
        "description": (
            "Collapse all runs of whitespace in a string to a single space. "
            "Useful when ingesting user-typed free-text into a normalized form."
        ),
        "type_hints": "def normalize_whitespace(text: str) -> str:",
        "usage_examples": [
            "normalize_whitespace('a   b') == 'a b'",
            "normalize_whitespace('\\t\\nfoo  bar') == 'foo bar'",
            "normalize_whitespace('') == ''",
        ],
    }
)

INVALID_NAMING_GENERIC_NAME = json.dumps(
    {
        "name": "helper",  # too generic
        "description": "A helper function.",
        "type_hints": "def helper(x) -> str:",
        "usage_examples": ["helper('x')"],
    }
)

INVALID_NAMING_BAD_IDENT = json.dumps(
    {
        "name": "my function",  # not a valid identifier
        "description": "x",
        "type_hints": "def f() -> None:",
        "usage_examples": ["f()"],
    }
)


@dataclass
class _FakeClient:
    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def generate_text(self, *, system: str, prompt: str, max_tokens: int) -> str:
        self.calls.append((system, prompt, max_tokens))
        return self.responses.pop(0) if self.responses else ""


@dataclass
class _FakeToolRegistry:
    registered: list[SelfAuthoredTool] = field(default_factory=list)
    raise_on_register: bool = False

    def register_tool(self, tool: SelfAuthoredTool) -> str:
        if self.raise_on_register:
            raise RuntimeError("simulated registry failure")
        self.registered.append(tool)
        return tool.id


def _eligible_candidate(**overrides: Any) -> Candidate:
    kwargs = dict(
        nutrient_id="n-42",
        code=VALID_CODE,
        trophic_level=3,
        use_count=5,
        tags=["text", "normalization"],
    )
    kwargs.update(overrides)
    return Candidate(**kwargs)


# ---------------------------------------------------------------------------
# Candidate eligibility
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_eligible_meets_thresholds(self) -> None:
        c = _eligible_candidate()
        assert c.eligible() is True

    def test_not_eligible_below_trophic_level(self) -> None:
        assert _eligible_candidate(trophic_level=2).eligible() is False

    def test_not_eligible_below_use_count(self) -> None:
        assert _eligible_candidate(use_count=4).eligible() is False

    def test_not_eligible_empty_code(self) -> None:
        assert _eligible_candidate(code="").eligible() is False

    def test_thresholds_match_spec(self) -> None:
        assert MIN_TROPHIC_LEVEL == 3
        assert MIN_SUCCESSFUL_USES == 5


# ---------------------------------------------------------------------------
# NamingResult schema
# ---------------------------------------------------------------------------


class TestNamingResult:
    def test_valid_roundtrip(self) -> None:
        data = json.loads(VALID_NAMING)
        result = NamingResult.model_validate(data)
        assert result.name == "normalize_whitespace"
        assert len(result.usage_examples) == 3

    def test_rejects_generic_name(self) -> None:
        with pytest.raises(Exception):
            NamingResult.model_validate(json.loads(INVALID_NAMING_GENERIC_NAME))

    def test_rejects_non_identifier_name(self) -> None:
        with pytest.raises(Exception):
            NamingResult.model_validate(json.loads(INVALID_NAMING_BAD_IDENT))


# ---------------------------------------------------------------------------
# promote_apex_predator — single shot
# ---------------------------------------------------------------------------


def _permissive_validator(tool: SelfAuthoredTool) -> bool:
    return True


def _rejecting_validator(tool: SelfAuthoredTool) -> Any:
    class _Result:
        valid = False
        errors = ["banned call: os.remove"]
    return _Result()


class TestPromoteApexPredator:
    def test_ineligible_returns_reason(self) -> None:
        outcome = promote_apex_predator(
            _eligible_candidate(trophic_level=1),
            tool_registry=_FakeToolRegistry(),
            naming_client=_FakeClient([VALID_NAMING]),
            validator=_permissive_validator,
        )
        assert outcome.success is False
        assert "not eligible" in outcome.reason

    def test_no_client_returns_reason(self) -> None:
        outcome = promote_apex_predator(
            _eligible_candidate(),
            tool_registry=_FakeToolRegistry(),
            naming_client=None,
            validator=_permissive_validator,
        )
        assert outcome.success is False
        assert "no naming client" in outcome.reason

    def test_success_path(self) -> None:
        reg = _FakeToolRegistry()
        outcome = promote_apex_predator(
            _eligible_candidate(),
            tool_registry=reg,
            naming_client=_FakeClient([VALID_NAMING]),
            validator=_permissive_validator,
        )
        assert outcome.success is True
        assert outcome.tool_id == reg.registered[0].id
        # Carries parent_id back to the nutrient
        assert reg.registered[0].parent_id == "n-42"
        assert reg.registered[0].created_by == "jitterbug"
        assert reg.registered[0].name == "normalize_whitespace"

    def test_invalid_naming_retries_once_then_fails(self) -> None:
        # Two invalid responses — retry budget exhausted
        outcome = promote_apex_predator(
            _eligible_candidate(),
            tool_registry=_FakeToolRegistry(),
            naming_client=_FakeClient(
                [INVALID_NAMING_GENERIC_NAME, INVALID_NAMING_BAD_IDENT]
            ),
            validator=_permissive_validator,
        )
        assert outcome.success is False
        assert "naming rejected" in outcome.reason

    def test_naming_retry_succeeds_on_second_try(self) -> None:
        reg = _FakeToolRegistry()
        outcome = promote_apex_predator(
            _eligible_candidate(),
            tool_registry=reg,
            naming_client=_FakeClient([INVALID_NAMING_GENERIC_NAME, VALID_NAMING]),
            validator=_permissive_validator,
        )
        assert outcome.success is True
        assert len(reg.registered) == 1

    def test_validator_rejection_prevents_registration(self) -> None:
        reg = _FakeToolRegistry()
        outcome = promote_apex_predator(
            _eligible_candidate(),
            tool_registry=reg,
            naming_client=_FakeClient([VALID_NAMING]),
            validator=_rejecting_validator,
        )
        assert outcome.success is False
        assert "tool_validator rejected" in outcome.reason
        assert reg.registered == []

    def test_register_failure_surfaced(self) -> None:
        reg = _FakeToolRegistry(raise_on_register=True)
        outcome = promote_apex_predator(
            _eligible_candidate(),
            tool_registry=reg,
            naming_client=_FakeClient([VALID_NAMING]),
            validator=_permissive_validator,
        )
        assert outcome.success is False
        assert "register_tool failed" in outcome.reason


# ---------------------------------------------------------------------------
# promote_eligible — cap + stream handling
# ---------------------------------------------------------------------------


class TestPromoteEligible:
    def test_cap_respected(self) -> None:
        reg = _FakeToolRegistry()
        candidates = [
            _eligible_candidate(nutrient_id=f"n-{i}") for i in range(6)
        ]
        # Every call returns the same VALID_NAMING
        client = _FakeClient([VALID_NAMING] * 6)
        outcomes = promote_eligible(
            candidates,
            tool_registry=reg,
            naming_client=client,
            validator=_permissive_validator,
            max_promotions=2,
        )
        # Only 2 successes, capped
        assert len([o for o in outcomes if o.success]) == 2
        assert len(reg.registered) == 2

    def test_failures_dont_consume_budget(self) -> None:
        reg = _FakeToolRegistry()
        candidates = [
            _eligible_candidate(nutrient_id=f"n-{i}") for i in range(4)
        ]
        # First response fails validation, rest succeed
        client = _FakeClient(
            [
                INVALID_NAMING_GENERIC_NAME, INVALID_NAMING_BAD_IDENT,  # first candidate retries
                VALID_NAMING,  # second candidate
                VALID_NAMING,  # third candidate
                VALID_NAMING,  # fourth (shouldn't be hit; budget=2)
            ]
        )
        outcomes = promote_eligible(
            candidates,
            tool_registry=reg,
            naming_client=client,
            validator=_permissive_validator,
            max_promotions=2,
        )
        # First failed, next two succeed, stop.
        assert len([o for o in outcomes if o.success]) == 2
        assert len([o for o in outcomes if not o.success]) == 1
        # Fourth candidate never attempted
        assert len(outcomes) == 3

    def test_default_cap_matches_spec(self) -> None:
        assert MAX_PROMOTIONS_PER_CYCLE == 3
