"""Session 0 (v3.2) — BELIEF_OFFLINE=1 hermeticity tests.

Three audit findings converged on the same surface: cascade stage 3
(the MiniLM embedding) was the only stage that could reach out to
Hugging Face during a 'cheap path' test run, and the failure mode
was an opaque HTTP timeout chain rather than a clear 'no network'
signal.

These tests pin two invariants:

1. Stages 0–2 (blocklist, keyword, TF-IDF) run unchanged under
   ``BELIEF_OFFLINE=1`` — no network, no model download, no error.
2. Stage 3, when actually reached, raises ``OfflineModeError`` with
   a clear message, *not* whatever sentence-transformers would have
   raised after failing to fetch the checkpoint.

The tests do not require sentence-transformers to be installed —
the offline guard fires before any import attempt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.photosynthesis.filter.cascade import (
    CascadingRelevanceFilter,
    OfflineModeError,
    Stage,
    _offline_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def keywords_file(tmp_path: Path) -> Path:
    p = tmp_path / "keywords.yaml"
    p.write_text(
        "keywords:\n  - fastapi\n  - langgraph\n  - mcp\n  - pydantic\n"
    )
    return p


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module runs under BELIEF_OFFLINE=1.  Autouse
    so no test body has to remember to set it."""
    monkeypatch.setenv("BELIEF_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Env var parser
# ---------------------------------------------------------------------------


class TestOfflineEnvVar:
    """The env-var parser is tiny but load-bearing — one typo
    flips the whole guard off."""

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
    def test_truthy_values_enable_offline(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        monkeypatch.setenv("BELIEF_OFFLINE", val)
        assert _offline_mode() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "maybe"])
    def test_falsy_values_leave_offline_off(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        monkeypatch.setenv("BELIEF_OFFLINE", val)
        assert _offline_mode() is False


# ---------------------------------------------------------------------------
# Stages 0–2 remain fully usable offline
# ---------------------------------------------------------------------------


class TestCheapStagesOffline:
    """The audit's core complaint was that 'cheap path' tests
    reached Hugging Face. These tests prove the cheap path is
    genuinely network-free under BELIEF_OFFLINE=1 — *for signals
    that don't reach stage 3*. If a signal survives to stage 3
    under offline mode, raising is the correct behaviour and is
    covered by TestStage3OfflineRaise below.
    """

    def test_stage0_blocklist_drop_runs_offline(self, keywords_file: Path) -> None:
        """A signal that hits the blocklist is dropped at stage 0 and
        never reaches the embed model — must not raise offline."""
        f = CascadingRelevanceFilter(
            keywords_path=keywords_file,
            blocklist=["spam-domain.example"],
        )
        [res] = f.score(["visit https://spam-domain.example/cool-fastapi-post"])
        assert res.stage_reached == Stage.BLOCKED
        assert not res.kept

    def test_stage1_keyword_miss_runs_offline(self, keywords_file: Path) -> None:
        """A signal with no keyword hit is dropped at stage 1 and
        never reaches the embed model — must not raise offline."""
        f = CascadingRelevanceFilter(keywords_path=keywords_file)
        [res] = f.score(["Breaking news from the world of competitive yodeling"])
        assert res.stage_reached == Stage.BLOCKED
        assert not res.kept
        assert "no keyword" in res.reason

    def test_stage2_high_confidence_bypass_runs_offline(
        self, keywords_file: Path
    ) -> None:
        """Requires sklearn — without it, stage 2 is inert (tfidf
        returns -1) and every keyword survivor falls through to
        stage 3, where offline mode correctly raises."""
        pytest.importorskip("sklearn")
        """A signal that clears stage-2's high-confidence bypass is
        kept without invoking stage 3 — must not raise offline.
        (This is the production hot-path for strongly on-topic
        signals: keyword hit + high TF-IDF → kept before stage 3.)"""
        f = CascadingRelevanceFilter(
            keywords_path=keywords_file,
            corpus=[
                "fastapi background tasks with redis",
                "fastapi websocket tutorial",
                "pydantic v2 settings migration",
            ],
            stage2_coarse=0.0,   # let everything pass the low bar
            stage2_high=0.0,     # force stage-2 high-confidence bypass
        )
        [res] = f.score(["fastapi background jobs"])
        assert res.kept is True
        assert res.stage_reached == Stage.TFIDF

    def test_stage2_low_drop_runs_offline(self, keywords_file: Path) -> None:
        """A signal that fails stage 2's coarse threshold is dropped
        at stage 2 and never reaches stage 3 — must not raise offline.
        """
        pytest.importorskip("sklearn")
        f = CascadingRelevanceFilter(
            keywords_path=keywords_file,
            corpus=[
                "pydantic v2 settings migration",
                "langgraph state machine tutorial",
            ],
            stage2_coarse=0.99,  # impossibly high bar → everything drops
        )
        [res] = f.score(["fastapi plugin for background jobs"])
        # Keyword matched (stage 1 passed), stage 2 rejected.
        # stage_reached stays at KEYWORD per the drop path.
        assert res.kept is False
        assert "low tfidf" in res.reason


# ---------------------------------------------------------------------------
# Stage 3 raises a clear, actionable error
# ---------------------------------------------------------------------------


class TestStage3OfflineRaise:
    """When a test or production code path genuinely needs stage 3,
    offline mode must tell you so clearly — not fail silently, not
    return -1, not leak an opaque HuggingFace HTTPError."""

    def test_ensure_embed_model_raises(self, keywords_file: Path) -> None:
        import numpy as _np  # noqa: F401 — presence check; ok to import
        f = CascadingRelevanceFilter(
            keywords_path=keywords_file,
            centroids=_np.array([[1.0, 0.0]]),
        )
        with pytest.raises(OfflineModeError) as excinfo:
            f._ensure_embed_model()
        msg = str(excinfo.value)
        # The message must name BELIEF_OFFLINE so an operator reading a
        # traceback can trace the cause in under a minute.
        assert "BELIEF_OFFLINE" in msg
        assert "offline" in msg.lower()

    def test_score_raises_when_stage3_is_reached(
        self, keywords_file: Path
    ) -> None:
        """Full-flow assertion: if a signal survives to stage 3 and we
        are offline, .score() raises OfflineModeError rather than
        quietly dropping to -1 (the pre-Session-0 behaviour)."""
        import numpy as np

        f = CascadingRelevanceFilter(
            keywords_path=keywords_file,
            # No corpus → stage 2 is inert → survivors fall through to
            # the stage-3 batch block.
            centroids=np.array([[1.0, 0.0]]),
        )
        with pytest.raises(OfflineModeError):
            f.score(["New fastapi plugin for background jobs"])


# ---------------------------------------------------------------------------
# Belt and braces — the guard fires *before* we touch the HF network
# ---------------------------------------------------------------------------


class TestNoNetworkUnderOffline:
    """If this test fails, the guard is in the wrong place: either
    _ensure_embed_model is calling SentenceTransformer(...) before
    the offline check, or some other code path is fetching from HF
    outside the guard.  Treat a failure here as a correctness bug,
    not a test hygiene issue."""

    def test_sentence_transformers_is_never_imported(
        self, monkeypatch: pytest.MonkeyPatch, keywords_file: Path
    ) -> None:
        import numpy as np
        import sys

        # Remove any cached import — if something imports
        # sentence_transformers during .score(), sys.modules will
        # carry a fresh entry after the call.
        sys.modules.pop("sentence_transformers", None)

        f = CascadingRelevanceFilter(
            keywords_path=keywords_file,
            centroids=np.array([[1.0, 0.0]]),
        )
        with pytest.raises(OfflineModeError):
            f.score(["fastapi plugin for background jobs"])

        assert "sentence_transformers" not in sys.modules, (
            "sentence_transformers was imported under BELIEF_OFFLINE=1 — "
            "the guard fired too late."
        )
