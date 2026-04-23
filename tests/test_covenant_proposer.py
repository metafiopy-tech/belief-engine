"""Hermetic tests for Session 8 (v3.2) — covenant proposer + gate.

No LLM, no network.  Clustering is deterministic (error-signature
hash); proposer is a stub; gate evaluates regex-based proposals
against in-memory ArchivedBuild fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.covenants.policy import DEFAULT_POLICY, GatePolicy
from belief.covenants.precision_gate import (
    ArchivedBuild,
    default_regex_applier,
    evaluate_gate,
    measure_precision,
)
from belief.covenants.proposer import (
    CovenantProposal,
    FailureTrace,
    cluster_failures,
    error_signature,
    load_proposals,
    propose_covenants_from_failures,
    save_proposals,
)


# ---------------------------------------------------------------------------
# Error signature + clustering
# ---------------------------------------------------------------------------


class TestClustering:
    def test_addresses_scrubbed(self) -> None:
        sig = error_signature(
            "AttributeError: 'module' object at 0x7f0abc12 has no attribute 'foo'"
        )
        assert "0x7f0abc12" not in sig
        assert "0xADDR" in sig

    def test_line_numbers_scrubbed(self) -> None:
        sig = error_signature("SyntaxError: invalid syntax, line 432")
        assert "432" not in sig
        assert "line N" in sig

    def test_groups_similar_failures(self) -> None:
        failures = [
            FailureTrace("r1", "g", "ImportError: No module named 'foo'"),
            FailureTrace("r2", "g", "ImportError: No module named 'bar'"),
            FailureTrace("r3", "g", "SyntaxError: invalid syntax"),
            FailureTrace("r4", "g", "ImportError: No module named 'baz'"),
        ]
        # After scrubbing, all three ImportErrors may collapse OR stay
        # distinct depending on scrub rules — we just assert the
        # cluster structure is deterministic and partitions cleanly.
        clusters = cluster_failures(failures)
        total = sum(len(v) for v in clusters.values())
        assert total == len(failures)
        # At least SyntaxError is its own cluster.
        assert any("SyntaxError" in k for k in clusters)


# ---------------------------------------------------------------------------
# End-to-end proposer with stub LLM
# ---------------------------------------------------------------------------


class TestProposerPipeline:
    def test_skips_small_clusters(self) -> None:
        failures = [
            FailureTrace(f"r{i}", "g", "AttributeError: fizz") for i in range(3)
        ]
        proposed = propose_covenants_from_failures(
            failures, min_cluster_size=5,
            proposer=lambda sig, cluster: {"pattern": sig, "replacement": "", "rationale": ""},
        )
        assert proposed == []

    def test_proposes_one_per_large_cluster(self) -> None:
        failures = [
            FailureTrace(f"r{i}", "g", "AttributeError: NoneType has no attribute 'split'")
            for i in range(6)
        ]
        proposed = propose_covenants_from_failures(
            failures, min_cluster_size=5,
            proposer=lambda sig, cluster: {
                "pattern": r"\.split\(",
                "replacement": "",
                "rationale": "forbid bare .split() — add a None guard first",
            },
        )
        assert len(proposed) == 1
        p = proposed[0]
        assert p.cluster_size == 6
        assert p.proposed_pattern == r"\.split\("


# ---------------------------------------------------------------------------
# Precision gate
# ---------------------------------------------------------------------------


def _proposal(pattern: str = r"from pydantic\.v1", replacement: str = "") -> CovenantProposal:
    return CovenantProposal(
        proposal_id="p-1",
        cluster_size=6,
        error_signature="ImportError: from pydantic.v1",
        representative_error="ImportError: No module named 'pydantic.v1'",
        proposed_pattern=pattern,
        proposed_replacement=replacement,
    )


def _archive(n_fail_fixable_with_match: int, n_pass_with_match: int,
             n_pass_clean: int) -> list[ArchivedBuild]:
    out: list[ArchivedBuild] = []
    for i in range(n_fail_fixable_with_match):
        out.append(ArchivedBuild(
            run_id=f"ff-{i}", goal="g", verdict="fail_fixable",
            code_files={"a.py": "from pydantic.v1 import BaseModel\n"},
        ))
    for i in range(n_pass_with_match):
        out.append(ArchivedBuild(
            run_id=f"pw-{i}", goal="g", verdict="pass",
            code_files={"a.py": "from pydantic.v1 import BaseModel\n"},
        ))
    for i in range(n_pass_clean):
        out.append(ArchivedBuild(
            run_id=f"pc-{i}", goal="g", verdict="pass",
            code_files={"a.py": "from pydantic import BaseModel\n"},
        ))
    return out


class TestGate:
    def test_would_have_prevented_counts_fail_fixable_matches(self) -> None:
        m = measure_precision(_proposal(), _archive(10, 0, 5))
        assert m.would_have_prevented == 10
        assert m.would_have_broken == 0
        assert m.precision == 1.0

    def test_would_have_broken_counts_pass_matches(self) -> None:
        m = measure_precision(_proposal(), _archive(5, 3, 5))
        assert m.would_have_prevented == 5
        assert m.would_have_broken == 3

    def test_auto_pass_when_all_thresholds_met(self) -> None:
        p = evaluate_gate(_proposal(), _archive(10, 0, 5))
        assert p.status == "auto_pass"

    def test_auto_fail_when_any_pass_build_breaks(self) -> None:
        p = evaluate_gate(_proposal(), _archive(10, 1, 5))
        assert p.status == "auto_fail"
        assert p.metrics["would_have_broken"] == 1

    def test_auto_fail_below_prevented_threshold(self) -> None:
        p = evaluate_gate(_proposal(), _archive(3, 0, 5))
        assert p.status == "auto_fail"

    def test_auto_fail_below_cluster_size(self) -> None:
        prop = _proposal()
        prop.cluster_size = 3
        p = evaluate_gate(prop, _archive(10, 0, 5))
        assert p.status == "auto_fail"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        p = _proposal()
        target = tmp_path / "proposals.json"
        save_proposals([p], path=target)
        assert target.exists()
        restored = load_proposals(path=target)
        assert len(restored) == 1
        assert restored[0].proposal_id == p.proposal_id
