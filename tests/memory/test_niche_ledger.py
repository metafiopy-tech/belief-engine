"""Hermetic tests for belief.memory.niche_ledger (mycorrhizal Stage 2).

All tests use tmp_path so nothing touches the real ~/.belief-engine/ tree.
Cross-ledger integration (niche reference → reciprocity contribution) is
exercised with explicit fixture injection, not the singleton accessors.

Run with:

    python3 -m pytest tests/memory/test_niche_ledger.py -q --timeout=60
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.memory.niche_ledger import (
    DEFAULT_REFERENCE_CREDIT,
    NICHE_KINDS,
    NicheLedger,
    cli_format_by_agent,
    cli_format_query,
    cli_format_top,
)
from belief.memory.reciprocity import ReciprocityLedger


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def recip(tmp_path: Path) -> ReciprocityLedger:
    r = ReciprocityLedger(db_path=tmp_path / "reciprocity.db")
    yield r
    r.close()


@pytest.fixture
def niches(tmp_path: Path, recip: ReciprocityLedger) -> NicheLedger:
    n = NicheLedger(
        db_path=tmp_path / "niches.db",
        reciprocity_ledger=recip,
    )
    yield n
    n.close()


# ── 1. Round-trip ───────────────────────────────────────────────────────────


def test_record_modification_returns_stable_id(niches: NicheLedger) -> None:
    nid1 = niches.record_modification(
        constructing_agent_id="architect",
        kind="tool",
        soil_reference="tool-001",
        pre_state_description="no fastapi validator",
        post_state_description="fastapi_route_validator available",
    )
    # Same (kind, soil_reference) — must return the same id, not a new one.
    nid2 = niches.record_modification(
        constructing_agent_id="someone-else",
        kind="tool",
        soil_reference="tool-001",
        pre_state_description="(ignored on dup)",
        post_state_description="(ignored on dup)",
    )
    assert nid1 == nid2

    rec = niches.get_niche(nid1)
    assert rec is not None
    assert rec.kind == "tool"
    assert rec.soil_reference == "tool-001"
    assert rec.constructing_agent_id == "architect"  # first-write-wins
    assert rec.pre_state_description == "no fastapi validator"
    assert rec.reference_count == 0
    assert rec.last_referenced_at is None


def test_record_modification_rejects_bad_kind(niches: NicheLedger) -> None:
    with pytest.raises(ValueError):
        niches.record_modification(
            constructing_agent_id="x",
            kind="not-a-kind",
            soil_reference="ref",
        )


def test_record_modification_rejects_empty_fields(niches: NicheLedger) -> None:
    with pytest.raises(ValueError):
        niches.record_modification(
            constructing_agent_id="",
            kind="tool",
            soil_reference="ref",
        )
    with pytest.raises(ValueError):
        niches.record_modification(
            constructing_agent_id="x",
            kind="tool",
            soil_reference="",
        )


def test_each_kind_is_independent(niches: NicheLedger) -> None:
    """Same soil_reference across kinds is allowed — they're distinct
    capability additions (a tool and a primitive both named 'foo')."""
    for k in NICHE_KINDS:
        niches.record_modification(
            constructing_agent_id="agent",
            kind=k,
            soil_reference="same-ref",
        )
    assert niches.count_niches() == len(NICHE_KINDS)
    for k in NICHE_KINDS:
        assert niches.count_niches(kind=k) == 1


def test_lookup_by_soil_reference(niches: NicheLedger) -> None:
    nid = niches.record_modification(constructing_agent_id="a", kind="tool", soil_reference="t1")
    rec = niches.lookup_by_soil_reference(kind="tool", soil_reference="t1")
    assert rec is not None and rec.niche_id == nid
    assert niches.lookup_by_soil_reference(kind="tool", soil_reference="missing") is None


# ── 2. Reference counting ──────────────────────────────────────────────────


def test_record_reference_increments_counter_and_timestamp(
    niches: NicheLedger,
) -> None:
    nid = niches.record_modification(
        constructing_agent_id="alice", kind="tool", soil_reference="t-1"
    )
    before = niches.get_niche(nid)
    assert before is not None and before.reference_count == 0

    assert niches.record_reference(nid, referring_build_id="build-1") is True
    after = niches.get_niche(nid)
    assert after is not None
    assert after.reference_count == 1
    assert after.last_referenced_at is not None

    # Second distinct build → counter ticks again.
    niches.record_reference(nid, referring_build_id="build-2")
    again = niches.get_niche(nid)
    assert again is not None
    assert again.reference_count == 2


def test_record_reference_idempotent_per_build(niches: NicheLedger) -> None:
    """The decomposer/recomposer may replay; counters must not inflate."""
    nid = niches.record_modification(constructing_agent_id="a", kind="tool", soil_reference="t")
    for _ in range(5):
        niches.record_reference(nid, referring_build_id="build-1")
    rec = niches.get_niche(nid)
    assert rec is not None and rec.reference_count == 1


def test_record_reference_unknown_niche_is_noop(niches: NicheLedger) -> None:
    assert niches.record_reference("does-not-exist", referring_build_id="b") is False


def test_record_reference_rejects_empty_inputs(niches: NicheLedger) -> None:
    with pytest.raises(ValueError):
        niches.record_reference("", referring_build_id="b")
    with pytest.raises(ValueError):
        niches.record_reference("nid", referring_build_id="")


# ── 3. Downstream credit propagation (the load-bearing test) ───────────────


def test_record_reference_credits_constructor_in_reciprocity(
    niches: NicheLedger, recip: ReciprocityLedger
) -> None:
    """The single most important assertion in Stage 2: a downstream build
    that consumes alice's tool causes alice's reciprocity score to rise.
    Without this, the niche ledger is observational only."""
    nid = niches.record_modification(
        constructing_agent_id="alice", kind="tool", soil_reference="t-x"
    )
    # Three different builds consume alice's tool.
    niches.record_reference(nid, referring_build_id="build-1")
    niches.record_reference(nid, referring_build_id="build-2")
    niches.record_reference(nid, referring_build_id="build-3")

    stats = recip.stats("alice")
    # Three references × 0.1 per reference = 0.3 nutrients_returned
    assert stats.nutrients_returned == pytest.approx(0.3)
    assert stats.contribution_count == 3


def test_downstream_credit_idempotent(niches: NicheLedger, recip: ReciprocityLedger) -> None:
    """Replays must not double-credit the constructor."""
    nid = niches.record_modification(
        constructing_agent_id="alice", kind="tool", soil_reference="t-y"
    )
    for _ in range(10):
        niches.record_reference(nid, referring_build_id="build-1")
    assert recip.stats("alice").nutrients_returned == pytest.approx(DEFAULT_REFERENCE_CREDIT)


def test_constructor_credited_not_referrer(niches: NicheLedger, recip: ReciprocityLedger) -> None:
    """The CRITICAL semantic: alice constructed, bob consumed — alice gets
    credit, bob does not. Confusing these directions would invert the
    incentive."""
    nid = niches.record_modification(constructing_agent_id="alice", kind="tool", soil_reference="t")
    # Pretend bob is the agent_id of the consuming build's environment —
    # the niche ledger doesn't know that. Even so, the credit must go to
    # alice (the original constructor stored on the niche).
    niches.record_reference(nid, referring_build_id="build-by-bob")
    assert recip.stats("alice").nutrients_returned > 0
    assert recip.stats("bob").nutrients_returned == 0


def test_widely_used_niche_outranks_unused_pile(
    niches: NicheLedger, recip: ReciprocityLedger
) -> None:
    """The whole point of downstream credit: an agent who built one
    heavily-referenced niche outranks one who built ten unreferenced
    ones. Validates the credit-propagation logic at the reciprocity
    ranking level, not just the raw counter.

    Stronger-than-ordering claim: bob built ten niches but nobody
    consumed any of them, so the reciprocity ledger never even sees
    bob (record_modification alone is not a reciprocity event;
    only record_reference is). Alice, by contrast, accumulates
    15 × 0.1 = 1.5 credit and lands in the ranking. The system's
    incentive is "build things others actually use."
    """
    pop_nid = niches.record_modification(
        constructing_agent_id="alice", kind="tool", soil_reference="popular"
    )
    for i in range(15):
        niches.record_reference(pop_nid, referring_build_id=f"build-{i}")
    for i in range(10):
        niches.record_modification(
            constructing_agent_id="bob",
            kind="tool",
            soil_reference=f"dead-{i}",
        )
    rows = recip.rank_agents()
    ids = [s.agent_id for s in rows]
    # Alice received downstream credit and appears in the ranking.
    assert "alice" in ids
    # Bob's unused construction did not move the reciprocity needle.
    assert "bob" not in ids
    # And the credit amount is exactly what the contract promises.
    assert recip.stats("alice").nutrients_returned == pytest.approx(15 * DEFAULT_REFERENCE_CREDIT)


# ── 4. Discovery API ───────────────────────────────────────────────────────


def test_query_niches_substring(niches: NicheLedger) -> None:
    niches.record_modification(
        constructing_agent_id="a",
        kind="tool",
        soil_reference="tool-fastapi",
        post_state_description="fastapi_route_validator available",
    )
    niches.record_modification(
        constructing_agent_id="a",
        kind="tool",
        soil_reference="tool-redis",
        post_state_description="redis_pubsub_emitter available",
    )
    hits = niches.query_niches("fastapi")
    assert len(hits) == 1 and hits[0].soil_reference == "tool-fastapi"

    # Case-insensitive.
    hits_upper = niches.query_niches("FASTAPI")
    assert len(hits_upper) == 1


def test_query_niches_kind_filter(niches: NicheLedger) -> None:
    niches.record_modification(
        constructing_agent_id="a",
        kind="tool",
        soil_reference="tool-x",
        post_state_description="x available",
    )
    niches.record_modification(
        constructing_agent_id="a",
        kind="primitive",
        soil_reference="prim-x",
        post_state_description="x available",
    )
    assert len(niches.query_niches("x")) == 2
    assert len(niches.query_niches("x", kind="tool")) == 1
    assert len(niches.query_niches("x", kind="primitive")) == 1


def test_query_niches_rejects_bad_kind(niches: NicheLedger) -> None:
    with pytest.raises(ValueError):
        niches.query_niches("x", kind="bogus")


def test_niches_by_agent_orders_by_references(niches: NicheLedger) -> None:
    a_pop = niches.record_modification("agent-a", "tool", "pop")
    a_dead = niches.record_modification("agent-a", "tool", "dead")
    niches.record_reference(a_pop, referring_build_id="b1")
    niches.record_reference(a_pop, referring_build_id="b2")

    rows = niches.niches_by_agent("agent-a")
    assert [r.niche_id for r in rows] == [a_pop, a_dead]


# ── 5. top_constructors ranking ────────────────────────────────────────────


def test_top_constructors_lifetime(niches: NicheLedger) -> None:
    a = niches.record_modification("alice", "tool", "a-tool")
    b = niches.record_modification("bob", "tool", "b-tool")
    niches.record_modification("bob", "tool", "b-other")

    for i in range(5):
        niches.record_reference(a, referring_build_id=f"r{i}")
    niches.record_reference(b, referring_build_id="r0")

    rows = niches.top_constructors()
    assert rows[0].agent_id == "alice"
    assert rows[0].total_references == 5
    assert rows[1].agent_id == "bob"
    assert rows[1].niche_count == 2


def test_top_constructors_windowed(niches: NicheLedger) -> None:
    """Windowed ranking only counts references inside the window — used by
    Session 5's router to prefer agents whose niches are *currently* hot,
    not those who shipped one heavily-used tool a year ago."""
    a = niches.record_modification("alice", "tool", "a-tool")
    b = niches.record_modification("bob", "tool", "b-tool")
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=60)
    recent = now - timedelta(hours=1)

    # Alice: 10 old references, 0 recent.
    for i in range(10):
        niches.record_reference(a, referring_build_id=f"old-{i}", ts=old)
    # Bob: 2 recent references.
    for i in range(2):
        niches.record_reference(b, referring_build_id=f"rec-{i}", ts=recent)

    rows = niches.top_constructors(window="30d")
    # In a 30-day window, bob has 2 references, alice has 0 — only bob
    # appears (HAVING total_refs > 0 in the SQL).
    ids = [r.agent_id for r in rows]
    assert "bob" in ids
    assert "alice" not in ids


# ── 6. CLI rendering ───────────────────────────────────────────────────────


def test_cli_top_empty_ledger(niches: NicheLedger) -> None:
    out = cli_format_top(niches)
    assert "Niche ledger" in out
    assert "0 total niche" in out


def test_cli_top_with_data(niches: NicheLedger) -> None:
    nid = niches.record_modification(
        constructing_agent_id="alice",
        kind="tool",
        soil_reference="popular-tool",
    )
    niches.record_reference(nid, referring_build_id="b1")
    out = cli_format_top(niches)
    assert "alice" in out
    assert "popular-tool" in out
    assert "tool" in out


def test_cli_by_agent_no_matches(niches: NicheLedger) -> None:
    out = cli_format_by_agent(niches, "ghost")
    assert "0 niches" in out
    assert "not been credited" in out


def test_cli_by_agent_renders_post_state(niches: NicheLedger) -> None:
    niches.record_modification(
        constructing_agent_id="alice",
        kind="tool",
        soil_reference="t-1",
        post_state_description="fastapi_route_validator now available",
    )
    out = cli_format_by_agent(niches, "alice")
    assert "alice" in out
    assert "fastapi" in out


def test_cli_query_no_matches(niches: NicheLedger) -> None:
    out = cli_format_query(niches, "missing")
    assert "0 matches" in out


# ── 7. Singleton accessor ──────────────────────────────────────────────────


def test_singleton_returns_same_instance(tmp_path: Path, monkeypatch) -> None:
    """The hook sites rely on a shared singleton — exercise it once to
    ensure the lazy construction path doesn't crash."""
    import belief.memory.niche_ledger as nl

    nl._reset_default_ledger_for_tests()
    monkeypatch.setattr(nl, "_DEFAULT_DB_PATH", tmp_path / "ledger.db")
    a = nl.get_default_ledger()
    b = nl.get_default_ledger()
    assert a is b
    nl._reset_default_ledger_for_tests()
