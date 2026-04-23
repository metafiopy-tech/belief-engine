"""Tests for the local-mode classifier-skip behavior (Session 2).

Before Session 2, ``cli.run()`` always called the LLM multi-service
classifier.  In local mode that routed to Ollama and burned ~25s on
every build for a binary decision.  Session 2 replaces that with the
keyword classifier in local mode unless ``BELIEF_FORCE_LLM_CLASSIFY=1``
is set.

These tests are module-level (no CLI invocation) — they verify the
keyword classifier's outputs for the shapes of goals we care about.
The actual wiring is a one-branch change in cli.run() that is
impossible to unit-test without spinning up a LangGraph run.
"""

from __future__ import annotations


def test_fizzbuzz_is_single_service():
    from belief.tools.multi_service import _classify_by_keywords

    result = _classify_by_keywords("Build a Python FizzBuzz script")
    assert result.is_multi_service is False


def test_cli_is_single_service():
    from belief.tools.multi_service import _classify_by_keywords

    result = _classify_by_keywords("build a click CLI that reverses a string")
    assert result.is_multi_service is False


def test_two_separate_services_flagged():
    """The keyword classifier uses structural patterns, not loose
    mentions.  'two separate services' is one of the strong patterns."""
    from belief.tools.multi_service import _classify_by_keywords

    result = _classify_by_keywords(
        "build two separate services: an API on port 8000 and a worker on port 8001"
    )
    assert result.is_multi_service is True


def test_gateway_routes_to_pattern_flagged():
    from belief.tools.multi_service import _classify_by_keywords

    result = _classify_by_keywords("build an API gateway that routes to three backend services")
    assert result.is_multi_service is True


def test_loose_microservice_mention_not_flagged():
    """Regression guard: the classifier is deliberately conservative.
    A single mention of 'microservice' without structural detail is
    a Tier-1 single-service build, not a multi-service orchestration."""
    from belief.tools.multi_service import _classify_by_keywords

    result = _classify_by_keywords("build a microservice that computes primes up to N")
    assert result.is_multi_service is False


def test_force_env_var_name_is_documented():
    """The escape hatch name is intentionally ``BELIEF_FORCE_LLM_CLASSIFY``
    — check it hasn't drifted.  Anyone hitting a classifier-bug
    will search for this exact name."""
    import pathlib

    cli_src = pathlib.Path(__file__).resolve().parents[1] / "belief" / "cli.py"
    src = cli_src.read_text()
    assert "BELIEF_FORCE_LLM_CLASSIFY" in src, (
        "the escape-hatch env var was renamed; update the test or docs"
    )
