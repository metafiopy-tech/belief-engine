"""Tests for the substrate-transfer experiment condition toggle.

Covers belief.experiments.conditions plus the bypass/skip behavior wired
into Soil.retrieve and Soil.retrieve_profile.

The covenant_enforce node short-circuit and the recomposer covenant-strip
both consume the same conditions helper, so testing the helper plus the
soil-level bypass is enough to verify the wiring at this layer. Full
end-to-end (LangGraph) testing happens at the shakedown stage.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Conditions helper
# ---------------------------------------------------------------------------


def _reload_conditions():
    """Force a reload so the new env var value is picked up cleanly."""
    import belief.experiments.conditions as mod

    return importlib.reload(mod)


def test_default_condition_is_full_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BELIEF_EXPERIMENT_CONDITION", raising=False)
    mod = _reload_conditions()
    assert mod.current_condition() == "full"
    assert mod.covenants_enabled() is True
    assert mod.fsrs_decay_enabled() is True
    assert mod.soil_retrieval_enabled() is True


def test_condition_full_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "full")
    mod = _reload_conditions()
    assert mod.current_condition() == "full"
    assert mod.covenants_enabled() is True
    assert mod.fsrs_decay_enabled() is True


def test_condition_soil_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "soil_only")
    mod = _reload_conditions()
    assert mod.current_condition() == "soil_only"
    assert mod.covenants_enabled() is False
    assert mod.fsrs_decay_enabled() is False
    # Soil retrieval itself still ON — that's the whole point of this condition
    assert mod.soil_retrieval_enabled() is True


def test_condition_raw_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "raw_local")
    mod = _reload_conditions()
    assert mod.current_condition() == "raw_local"
    assert mod.covenants_enabled() is False
    assert mod.fsrs_decay_enabled() is False
    assert mod.soil_retrieval_enabled() is False


def test_unrecognized_condition_falls_back_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "magic_unicorn")
    mod = _reload_conditions()
    # Defensive: any unknown value must NOT silently disable safety features
    assert mod.current_condition() == "full"
    assert mod.covenants_enabled() is True
    assert mod.fsrs_decay_enabled() is True


def test_empty_condition_falls_back_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "")
    mod = _reload_conditions()
    assert mod.current_condition() == "full"


def test_condition_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "SOIL_ONLY")
    mod = _reload_conditions()
    assert mod.current_condition() == "soil_only"


def test_condition_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELIEF_EXPERIMENT_CONDITION", "  full  ")
    mod = _reload_conditions()
    assert mod.current_condition() == "full"


# ---------------------------------------------------------------------------
# Soil retrieve bypass_fsrs_decay parameter
# ---------------------------------------------------------------------------


def test_soil_retrieve_signature_accepts_bypass_kwarg() -> None:
    """Smoke check: the public Soil.retrieve API exposes the experiment kwarg."""
    import inspect

    try:
        from belief.memory.soil import Soil
    except ImportError as exc:
        pytest.skip(f"Soil module unimportable in this env: {exc}")

    retrieve_sig = inspect.signature(Soil.retrieve)
    assert "bypass_fsrs_decay" in retrieve_sig.parameters
    assert retrieve_sig.parameters["bypass_fsrs_decay"].default is False


def test_soil_retrieve_profile_signature_accepts_bypass_kwarg() -> None:
    """Same for retrieve_profile — recomposer relies on this."""
    import inspect

    try:
        from belief.memory.soil import Soil
    except ImportError as exc:
        pytest.skip(f"Soil module unimportable in this env: {exc}")

    sig = inspect.signature(Soil.retrieve_profile)
    assert "bypass_fsrs_decay" in sig.parameters
    assert sig.parameters["bypass_fsrs_decay"].default is False


# ---------------------------------------------------------------------------
# Conditions module is importable independently of belief stack
# ---------------------------------------------------------------------------


def test_conditions_module_has_no_heavy_imports() -> None:
    """The conditions module must be ultra-lightweight so it can be imported
    from inside the LangGraph nodes without pulling in chromadb / langgraph /
    anthropic deps. If this test fails, the recomposer/graph short-circuits
    may stop working in stripped-down environments.
    """
    import belief.experiments.conditions as mod

    # The module should only depend on the stdlib
    source = open(mod.__file__).read()
    assert "import os" in source
    # No heavy deps that would break on lean installs
    for forbidden in ("chromadb", "langgraph", "anthropic", "pydantic"):
        assert forbidden not in source, (
            f"conditions.py must not import {forbidden} — it is consulted "
            f"from hot paths and must stay dependency-light"
        )
