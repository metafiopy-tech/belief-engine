"""Tests for the word-set source adapter (Synthesis Engine Session 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from belief.photosynthesis.config import PhotoConfig
from belief.photosynthesis.sources.word_set import (
    SOURCE_NAME,
    emit,
    make_bundle_id,
    parse_words,
)
from belief.photosynthesis.state import PhotosynthesisState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state(tmp_path: Path) -> PhotosynthesisState:
    """Fresh sqlite-backed state per test."""
    db = tmp_path / "signals.sqlite"
    return PhotosynthesisState(db_path=str(db))


@pytest.fixture
def config(tmp_path: Path) -> PhotoConfig:
    """PhotoConfig pinned to the per-test tmp_path so nothing pollutes ~."""
    return PhotoConfig(
        state_dir=tmp_path,
        log_dir=tmp_path / "logs",
        config_dir=tmp_path / "cfg",
    )


def _run(coro):
    return asyncio.run(coro)


def _all_word_set_rows(state: PhotosynthesisState) -> list[dict]:
    with state.conn() as c:
        rows = c.execute(
            "SELECT * FROM raw_signals WHERE source = ? ORDER BY id;",
            (SOURCE_NAME,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# parse_words
# ---------------------------------------------------------------------------


class TestParseWords:
    def test_simple_two_words(self) -> None:
        assert parse_words("mantis_shrimp,camera") == ["mantis_shrimp", "camera"]

    def test_strips_whitespace(self) -> None:
        assert parse_words("  mantis_shrimp ,  camera  ") == ["mantis_shrimp", "camera"]

    def test_single_word_allowed(self) -> None:
        # Session 2 doesn't enforce the >=2 cross-domain requirement;
        # Session 3's cross_domain_generator will gate on len(words) >= 2.
        assert parse_words("mantis_shrimp") == ["mantis_shrimp"]

    def test_dedups_duplicates(self) -> None:
        assert parse_words("x,x,y") == ["x", "y"]

    def test_drops_empty_tokens(self) -> None:
        assert parse_words("x,,y,") == ["x", "y"]

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            parse_words("")

    def test_only_separators_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            parse_words(",,, ,")

    def test_invalid_characters_rejected(self) -> None:
        # Spaces, slashes, parens, etc. are out -- the cascade filter
        # later does its own normalization but signals shouldn't carry
        # arbitrary punctuation.
        with pytest.raises(ValueError, match="invalid word"):
            parse_words("hello world,camera")

    def test_hyphenated_words_accepted(self) -> None:
        assert parse_words("slime-mold,routing") == ["slime-mold", "routing"]

    def test_preserves_input_order(self) -> None:
        assert parse_words("c,a,b") == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# make_bundle_id
# ---------------------------------------------------------------------------


class TestMakeBundleId:
    def test_returns_hex_default_length(self) -> None:
        bid = make_bundle_id()
        assert len(bid) == 12
        assert all(c in "0123456789abcdef" for c in bid)

    def test_unique_per_call(self) -> None:
        ids = {make_bundle_id() for _ in range(20)}
        assert len(ids) == 20

    def test_prefix_preserved(self) -> None:
        bid = make_bundle_id(prefix="mantis")
        assert bid.startswith("mantis-")
        assert len(bid) == len("mantis-") + 8


# ---------------------------------------------------------------------------
# emit -- the load-bearing source adapter
# ---------------------------------------------------------------------------


class TestEmit:
    def test_inserts_per_word_plus_bundle(self, state, config) -> None:
        inserted = _run(emit(state, config, words=["mantis_shrimp", "camera"], bundle_id="b1"))
        # 2 words -> 2 per-word + 1 bundle = 3 rows
        assert len(inserted) == 3

        rows = _all_word_set_rows(state)
        assert len(rows) == 3

        ids = {row["source_id"] for row in rows}
        assert "b1:bundle" in ids
        assert "b1:word:0:mantis_shrimp" in ids
        assert "b1:word:1:camera" in ids

    def test_all_signals_carry_word_set_source(self, state, config) -> None:
        _run(emit(state, config, words=["mantis_shrimp", "camera"], bundle_id="b1"))
        rows = _all_word_set_rows(state)
        assert {row["source"] for row in rows} == {SOURCE_NAME}

    def test_signals_enter_in_raw_status_for_cascade(self, state, config) -> None:
        """The cascade filter picks up status='raw' rows. Inserted
        signals must enter in that status -- never in a special-cased
        status that would short-circuit the filter."""
        _run(emit(state, config, words=["x", "y"], bundle_id="b1"))
        rows = _all_word_set_rows(state)
        assert {row["status"] for row in rows} == {"raw"}
        assert {row["stage_reached"] for row in rows} == {0}

    def test_per_word_titles_match_words(self, state, config) -> None:
        _run(
            emit(
                state,
                config,
                words=["mantis_shrimp", "camera", "slime-mold"],
                bundle_id="b1",
            )
        )
        rows = _all_word_set_rows(state)
        per_word_titles = {row["title"] for row in rows if not row["source_id"].endswith(":bundle")}
        assert per_word_titles == {"mantis_shrimp", "camera", "slime-mold"}

    def test_bundle_signal_carries_all_words_in_excerpt(self, state, config) -> None:
        _run(
            emit(
                state,
                config,
                words=["mantis_shrimp", "camera"],
                bundle_id="b1",
            )
        )
        rows = _all_word_set_rows(state)
        bundle = next(r for r in rows if r["source_id"] == "b1:bundle")
        assert "mantis_shrimp" in bundle["raw_excerpt"]
        assert "camera" in bundle["raw_excerpt"]

    def test_per_word_summary_lists_peer_words(self, state, config) -> None:
        _run(emit(state, config, words=["a", "b", "c"], bundle_id="b1"))
        rows = _all_word_set_rows(state)
        a_row = next(r for r in rows if r["source_id"] == "b1:word:0:a")
        # Peers should be listed in the summary so cross-word context
        # survives even on the per-word signal.
        assert "b" in a_row["summary"]
        assert "c" in a_row["summary"]

    def test_redundant_bundle_id_dedups(self, state, config) -> None:
        first = _run(emit(state, config, words=["x", "y"], bundle_id="b1"))
        second = _run(emit(state, config, words=["x", "y"], bundle_id="b1"))
        assert len(first) == 3
        assert len(second) == 0  # all source_ids already seen
        rows = _all_word_set_rows(state)
        assert len(rows) == 3  # nothing duplicated

    def test_fresh_bundle_id_inserts_new_rows(self, state, config) -> None:
        """Re-submitting the same words with a new bundle_id produces
        fresh rows -- this is the retry path when prior signals were
        rejected by the cascade."""
        _run(emit(state, config, words=["x", "y"], bundle_id="b1"))
        _run(emit(state, config, words=["x", "y"], bundle_id="b2"))
        rows = _all_word_set_rows(state)
        # 2 bundles * 3 rows = 6
        assert len(rows) == 6
        assert {r["source_id"] for r in rows} == {
            "b1:word:0:x",
            "b1:word:1:y",
            "b1:bundle",
            "b2:word:0:x",
            "b2:word:1:y",
            "b2:bundle",
        }

    def test_auto_bundle_id_when_none_supplied(self, state, config) -> None:
        first = _run(emit(state, config, words=["x", "y"]))
        second = _run(emit(state, config, words=["x", "y"]))
        # Different auto-generated bundle ids -> both runs succeed.
        assert len(first) == 3
        assert len(second) == 3
        assert len(_all_word_set_rows(state)) == 6

    def test_single_word_emits_one_per_word_plus_bundle(self, state, config) -> None:
        inserted = _run(emit(state, config, words=["mantis_shrimp"], bundle_id="b1"))
        assert len(inserted) == 2  # 1 per-word + 1 bundle
        rows = _all_word_set_rows(state)
        assert {r["source_id"] for r in rows} == {
            "b1:word:0:mantis_shrimp",
            "b1:bundle",
        }

    def test_empty_words_raises(self, state, config) -> None:
        with pytest.raises(ValueError, match="at least one word"):
            _run(emit(state, config, words=[]))

    def test_captured_at_propagates(self, state, config) -> None:
        _run(
            emit(
                state,
                config,
                words=["x", "y"],
                bundle_id="b1",
                captured_at=1234567890,
            )
        )
        rows = _all_word_set_rows(state)
        assert {r["captured_at"] for r in rows} == {1234567890}

    def test_inserted_rows_visible_to_pending_signals(self, state, config) -> None:
        """Sanity: word-set signals are picked up by the same query the
        cascade filter uses (status='raw'), without any source-name
        special-casing."""
        _run(emit(state, config, words=["mantis_shrimp", "camera"], bundle_id="b1"))
        pending = state.pending_signals()
        # All three rows are in 'raw' status, so the cascade filter
        # would see them on its next pass.
        word_set_pending = [row for row in pending if row["source"] == SOURCE_NAME]
        assert len(word_set_pending) == 3


# ---------------------------------------------------------------------------
# Pipeline-uniformity guard -- the spec is explicit that the cascade
# filter / novelty gate / ranker treat word_set signals just like the
# rest. Pin the absence of source-name special-casing so refactors can't
# silently introduce it.
# ---------------------------------------------------------------------------


class TestPipelineUniformity:
    def test_no_source_name_branches_in_filter(self) -> None:
        """The cascade filter must not branch on source name -- if it
        does, word_set signals are getting differential treatment, which
        the SE plan forbids."""
        from belief.photosynthesis.filter import cascade as cascade_mod

        src = Path(cascade_mod.__file__).read_text(encoding="utf-8")
        # If anyone adds a literal "word_set" check anywhere in the
        # cascade module, the test fails so the deviation gets reviewed.
        assert "word_set" not in src

    def test_no_source_name_branches_in_synthesis_cycle(self) -> None:
        from belief.photosynthesis.synthesis import cycle as cycle_mod

        src = Path(cycle_mod.__file__).read_text(encoding="utf-8")
        assert "word_set" not in src

    def test_no_source_name_branches_in_generator(self) -> None:
        from belief.photosynthesis.synthesis import generator as gen_mod

        src = Path(gen_mod.__file__).read_text(encoding="utf-8")
        assert "word_set" not in src

    def test_word_set_not_in_daemon_harvesters(self) -> None:
        """The SE plan is explicit: word-set is intentional / manual,
        never scheduled. Guard against accidental registration."""
        from belief.photosynthesis import daemon as daemon_mod

        # HARVESTERS may be a tuple/list of (name, callable, cadence) or
        # a dict -- check for the source name anywhere it could appear.
        src = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        # The daemon module body should not import word_set as a harvester.
        assert "word_set" not in src
