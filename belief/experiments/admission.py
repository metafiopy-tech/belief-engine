"""K-matched top-K admission for the STARVED-arm experiment.

Per generation, both arms see the *same* candidate builds and admit the *same
number K*; only the ranking key differs (design doc §1):

- **FED** admits top-K by the external grader (real test/covenant score).
- **STARVED** admits top-K by the build model's own ``self_score``.

Volume is held identical across arms — that is the whole point, so this module
admits exactly ``min(K, n_candidates)`` per arm and never lets one arm admit
more than the other. Selection is pure and deterministic (ties broken by
``build_id``) so a run is reproducible and unit-testable without any model.

This module only *decides* admissions. Actually decomposing the admitted builds
into each arm's isolated soil (Session 2's ``BELIEF_SOIL_PATH``) is the driver's
job in Session 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One build considered for admission in a generation.

    ``external_score`` ranks the FED arm (e.g. weighted score / fraction of
    tests passed); ``external_pass`` is the boolean external verdict used later
    to count "fictions" (STARVED-admitted builds that actually fail the test).
    ``self_score`` ranks the STARVED arm; ``self_confidence`` is logged only.
    """

    build_id: str
    external_score: float
    external_pass: bool
    self_score: float
    self_confidence: float = 0.0


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of one generation's K-matched selection."""

    k: int
    fed_admitted: list[str]
    starved_admitted: list[str]

    def admitted_for(self, arm: str) -> list[str]:
        a = arm.upper()
        if a == "FED":
            return self.fed_admitted
        if a == "STARVED":
            return self.starved_admitted
        raise ValueError(f"unknown arm: {arm!r}")

    def is_admitted(self, arm: str, build_id: str) -> bool:
        return build_id in self.admitted_for(arm)


def _top_k(candidates: list[Candidate], key, k: int) -> list[str]:
    """Return the build_ids of the top-k candidates by ``key``.

    Deterministic: sort by descending key, then ascending build_id so ties never
    depend on input order. Returns ids in admission order (best first).
    """
    ranked = sorted(candidates, key=lambda c: (-float(key(c)), c.build_id))
    return [c.build_id for c in ranked[:k]]


def select_admissions(candidates: list[Candidate], k: int) -> AdmissionResult:
    """Pick top-K per arm under the K-matched rule.

    ``k`` is clamped to the number of candidates so a thin generation admits all
    of them in BOTH arms (still volume-matched). FED ranks by ``external_score``,
    STARVED by ``self_score``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0; got {k}")
    cands = list(candidates)
    k_eff = min(k, len(cands))
    fed = _top_k(cands, lambda c: c.external_score, k_eff)
    starved = _top_k(cands, lambda c: c.self_score, k_eff)
    return AdmissionResult(k=k_eff, fed_admitted=fed, starved_admitted=starved)


def count_fictions(candidates: list[Candidate], result: AdmissionResult) -> int:
    """How many STARVED-admitted builds actually fail the external test.

    The direct count of "elegant wrong physics" entering STARVED soil — the
    mechanism under test. Computed from the same candidate records the arms read.
    """
    by_id = {c.build_id: c for c in candidates}
    return sum(
        1 for bid in result.starved_admitted if bid in by_id and not by_id[bid].external_pass
    )
