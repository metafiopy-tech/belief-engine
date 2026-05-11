"""End-to-end synthesis cycle orchestrator.

Runs the full pipeline over the top-k stage-3 survivors from
raw_signals:

    for each surviving seed:
        novelty -> difficulty -> ranker -> heap.push

    pop_top -> generator -> renderer -> goal_archive upsert
    mark each processed raw_signal as 'promoted' or 'rejected'

Idempotent: only 'kept' stage-3 rows are processed, and each is
transitioned to a terminal status at cycle end. Re-running on the same
raw_signals set won't produce duplicate pending_sessions.

For Session 4 the cycle relies on injectable dependencies (embedder,
LLM clients, archive manager). When any is missing, the cycle returns
a degraded summary rather than raising — the daemon keeps running.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


logger = logging.getLogger("belief.photosynthesis.synthesis.cycle")


@dataclass
class CycleSummary:
    surveyed: int = 0
    pushed_to_heap: int = 0
    promoted: int = 0
    rejected: int = 0
    saturated: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "surveyed": self.surveyed,
            "pushed_to_heap": self.pushed_to_heap,
            "promoted": self.promoted,
            "rejected": self.rejected,
            "saturated": self.saturated,
            "errors": self.errors,
        }

    def __str__(self) -> str:
        return json.dumps(self.as_dict())


# ---------------------------------------------------------------------------
# Sync entrypoint (APScheduler callable)
# ---------------------------------------------------------------------------


def run_synthesis_cycle_sync(state: Any, config: Any, **kwargs: Any) -> CycleSummary:
    """Blocking wrapper around run_synthesis_cycle.

    The daemon scheduler's ThreadPoolExecutor can't directly invoke
    async callables; we spin up a one-shot loop per cycle.
    """
    return asyncio.run(run_synthesis_cycle(state, config, **kwargs))


# ---------------------------------------------------------------------------
# Async entrypoint
# ---------------------------------------------------------------------------


async def run_synthesis_cycle(
    state: Any,
    config: Any,
    *,
    archive: Any = None,
    embedder: Optional[Callable[[str], Any]] = None,
    skill_library_query: Optional[Callable[[str], list[tuple[str, float]]]] = None,
    novelty_judge: Optional[Callable[[str], Awaitable[str]]] = None,
    time_estimator: Optional[Callable[[str], Awaitable[str]]] = None,
    generator_client: Optional[Callable[..., Awaitable[str]]] = None,
    critic_client: Optional[Callable[..., Awaitable[str]]] = None,
    successful_builds: int = 0,
    pending_dir: Optional[Path] = None,
) -> CycleSummary:
    """Run one synthesis cycle. Returns a summary dict.

    All expensive collaborators are injected — this is what makes the
    cycle unit-testable. Dependencies default to "not available", in
    which case the cycle survey-and-track but doesn't call LLMs.

    Two phases now run per cycle:

      1. Cross-domain (SE Session 3): user-submitted ``word_set``
         bundles that bypass the cascade gate (which can't reach
         stage 3 in the current engine state without external corpus
         /centroids). Routed to the cross-domain generator + critic.
         Source-name branching is intentional here -- the SE plan
         allows cycle.py to route by source while cascade / novelty /
         ranker remain uniform.
      2. Single-domain: the existing photosynthesis path -- score-
         filtered survivors get scored, ranked, generated, archived.

    Cross-domain runs first so its output appears at the top of the
    summary. Both contribute to the same ``CycleSummary``.
    """
    from belief.photosynthesis.synthesis.heap import (
        BoundedPriorityHeap,
        NoveltySaturation,
    )
    from belief.photosynthesis.synthesis.novelty import async_score_novelty
    from belief.photosynthesis.synthesis.difficulty import (
        async_estimate_difficulty,
    )
    from belief.photosynthesis.synthesis.ranker import (
        combined_value,
        coverage_gain,
        source_quality,
    )
    from belief.photosynthesis.synthesis.generator import synthesize
    from belief.photosynthesis.synthesis.renderer import write_session

    summary = CycleSummary()

    # ------------------------------------------------------------------
    # Phase 0 (SE Session 3): cross-domain word_set bundles
    # ------------------------------------------------------------------
    # Bundles with >=2 words route through cross_domain_generator and
    # the CoVe critic. Bypasses survivors_for_synthesis since the
    # cascade is currently unable to reach stage 3 (engine-wide
    # finding from S2.5; harvested signals also stuck at <stage 3).
    if generator_client is not None and hasattr(state, "word_set_pending_bundles"):
        # Lazy-init the bio store (SE Session 4) so each cycle gets a
        # consistent view across all bundles processed in this pass.
        # Failure is non-fatal -- if chromadb is broken, the
        # cross-domain phase still runs without priming.
        bio_store_instance: Any = None
        try:
            from belief.memory.biological_primitives import (
                BiologicalPrimitiveStore,
                DEFAULT_PERSIST_DIR,
            )

            bio_store_instance = BiologicalPrimitiveStore(
                persist_dir=DEFAULT_PERSIST_DIR,
            )
        except Exception as exc:
            logger.warning("bio_store init failed (skipping priming): %s", exc)

        await _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=generator_client,
            critic_client=critic_client,
            embedder=embedder,
            archive=archive,
            pending_dir=pending_dir,
            bio_store=bio_store_instance,
        )

    heap = BoundedPriorityHeap(state)
    rows = state.survivors_for_synthesis(limit=20)
    summary.surveyed += len(rows)

    if not rows:
        return summary

    # Archive may be None in tests — fall through cleanly
    top_tags: list[str] = []
    if archive is not None:
        try:
            top_tags = archive.top_tags("goal_archive", top_n=20)
        except Exception as exc:
            logger.warning("top_tags failed: %s", exc)

    # --- Phase 1: score each seed and push the keepers onto the heap ---
    pushed = 0
    processed_ids: list[int] = []
    for row in rows:
        seed = _row_to_seed(row)
        processed_ids.append(int(row["id"]))

        try:
            novelty = await async_score_novelty(
                seed,
                archive=archive,
                embedder=embedder or _fallback_embedder,
                llm_judge=novelty_judge,
            )
        except Exception as exc:
            summary.errors.append(f"novelty:{exc}")
            state.set_signal_status(int(row["id"]), "rejected")
            summary.rejected += 1
            continue

        if not novelty.accepted:
            state.set_signal_status(int(row["id"]), "rejected")
            summary.rejected += 1
            continue

        try:
            diff = await async_estimate_difficulty(
                seed,
                skill_library_query=skill_library_query or (lambda _t: []),
                successful_builds=successful_builds,
                llm_time_estimator=time_estimator,
            )
        except Exception as exc:
            summary.errors.append(f"difficulty:{exc}")
            state.set_signal_status(int(row["id"]), "rejected")
            summary.rejected += 1
            continue

        if not diff.accepted:
            state.set_signal_status(int(row["id"]), "rejected")
            summary.rejected += 1
            continue

        cg = coverage_gain(seed.get("domain_tags", []), top_tags)
        sq = source_quality(seed)
        r = combined_value(
            novelty=novelty.novelty,
            zpd_fit=diff.zpd_fit,
            coverage_gain=cg,
            source_quality=sq,
        )
        if not r.accepted:
            state.set_signal_status(int(row["id"]), "rejected")
            summary.rejected += 1
            continue

        # Package everything the generator will need when we pop.
        payload = {
            "seed": seed,
            "novelty": novelty.novelty,
            "zpd_fit": diff.zpd_fit,
            "pred_time_min": diff.pred_time_min,
            "neighbors": [n.to_prompt_dict() for n in novelty.neighbors[:5]],
            "source_signal_id": int(row["id"]),
        }
        if heap.push(payload, r.value):
            pushed += 1

    summary.pushed_to_heap = pushed
    heap.record_cycle(pushed_count=pushed)

    # Saturation check — non-fatal; we just record and return.
    try:
        heap.raise_if_saturated()
    except NoveltySaturation:
        summary.saturated = True
        return summary

    # --- Phase 2: pop top and generate (if we have a generator client) ---
    if generator_client is None or archive is None:
        return summary

    top = heap.pop_top()
    if top is None:
        return summary

    payload = top.seed
    seed = payload["seed"]
    try:
        result = await synthesize(
            seed,
            novelty_score=float(payload.get("novelty", 0.0)),
            zpd_fit=float(payload.get("zpd_fit", 0.0)),
            pred_time_min=int(payload.get("pred_time_min", 25)),
            neighbors=[_dict_to_neighbor(n) for n in payload.get("neighbors", [])],
            archive=archive,
            embedder=embedder or _fallback_embedder,
            generator_client=generator_client,
        )
    except Exception as exc:
        summary.errors.append(f"generator:{exc}")
        return summary

    if result.spec is None:
        signal_id = payload.get("source_signal_id")
        if isinstance(signal_id, int):
            state.set_signal_status(signal_id, "rejected")
            summary.rejected += 1
        return summary

    # Write the session files
    target_dir = pending_dir or getattr(
        config, "pending_sessions_dir", Path("/var/lib/photosynthesis/pending_sessions")
    )
    write_session(result.spec, pending_dir=Path(target_dir))

    # Upsert into goal_archive
    spec_text = f"{result.spec.title}. {result.spec.one_paragraph_description}"
    try:
        embedding = (embedder or _fallback_embedder)(spec_text)
    except Exception as exc:
        summary.errors.append(f"embed:{exc}")
        embedding = [0.0] * 8  # no-op sentinel

    archive.upsert_goal(
        "goal_archive",
        goal_id=result.spec.goal_id,
        embedding=embedding,
        document=spec_text,
        metadata={
            "title": result.spec.title,
            "artifact_type": result.spec.artifact_type,
            "domain_tags": seed.get("domain_tags", []),
            "status": "pending_build",
            "source_citation": result.spec.source_citation,
        },
    )

    signal_id = payload.get("source_signal_id")
    if isinstance(signal_id, int):
        state.set_signal_status(signal_id, "promoted")
    summary.promoted += 1
    return summary


# ---------------------------------------------------------------------------
# Cross-domain phase (SE Session 3)
# ---------------------------------------------------------------------------


async def _run_cross_domain_phase(
    *,
    state: Any,
    config: Any,
    summary: "CycleSummary",
    generator_client: Callable[..., Awaitable[str]],
    embedder: Optional[Callable[[str], Any]],
    archive: Any,
    pending_dir: Optional[Path],
    critic_client: Optional[Callable[..., Awaitable[str]]] = None,
    bio_store: Any = None,
) -> None:
    """Process pending word_set bundles through the cross-domain generator.

    Mutates ``summary`` in place: promoted/rejected/errors counters
    reflect cross-domain outcomes alongside the single-domain phase.
    Failures degrade silently (logged, error appended) so the
    single-domain phase still runs.
    """
    from belief.photosynthesis.synthesis.cross_domain_generator import (
        synthesize_cross_domain,
    )
    from belief.photosynthesis.synthesis.renderer import write_session

    try:
        bundles = state.word_set_pending_bundles(limit=10)
    except Exception as exc:
        summary.errors.append(f"cross_domain_query:{exc}")
        return
    summary.surveyed += len(bundles)

    if not bundles:
        return

    target_dir = pending_dir or getattr(
        config, "pending_sessions_dir", Path("/var/lib/photosynthesis/pending_sessions")
    )

    for bundle in bundles:
        words = list(bundle.get("words") or [])
        bundle_id = str(bundle.get("bundle_id") or "")
        row_ids = [int(rid) for rid in bundle.get("row_ids") or []]

        if len(words) < 2:
            # Single-word bundles can't form a cross-domain pair --
            # mark rejected so they don't reappear next cycle.
            for rid in row_ids:
                state.set_signal_status(rid, "rejected")
            summary.rejected += 1
            continue

        try:
            xd_result = await synthesize_cross_domain(
                words=words,
                bundle_id=bundle_id,
                generator_client=generator_client,
                critic_client=critic_client,
                bio_store=bio_store,
            )
        except Exception as exc:
            logger.warning("cross_domain_generator raised: %s", exc)
            summary.errors.append(f"cross_domain_generator:{exc}")
            continue

        if xd_result.spec is None:
            for rid in row_ids:
                state.set_signal_status(rid, "rejected")
            summary.rejected += 1
            continue

        # SE Session 6: novelty gate. Skip mechanisms that are too
        # similar to ones already in the bio_store -- 0.30 threshold
        # means anything >= 70% similar to its nearest neighbor is
        # rejected as cross_domain_redundant. Failures degrade
        # permissively (accepted by default) so a broken bio_store
        # doesn't block synthesis.
        if bio_store is not None and xd_result.mechanism is not None:
            try:
                from belief.photosynthesis.synthesis.cross_domain_novelty import gate

                verdict = gate(xd_result.mechanism, bio_store=bio_store)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("novelty gate failed (accepting): %s", exc)
                verdict = None

            if verdict is not None and verdict.rejected:
                logger.info(
                    "cross_domain novelty gate rejected bundle=%s score=%.3f",
                    bundle_id,
                    verdict.novelty_score,
                )
                summary.errors.append(
                    f"cross_domain_redundant:bundle={bundle_id}:score={verdict.novelty_score:.3f}"
                )
                for rid in row_ids:
                    state.set_signal_status(rid, "rejected")
                summary.rejected += 1
                continue

        # Write the rendered session markdown / json to pending_sessions/
        try:
            write_session(xd_result.spec, pending_dir=Path(target_dir))
        except Exception as exc:
            summary.errors.append(f"cross_domain_render:{exc}")
            # Fall through; we still mark promoted because the spec is valid.

        # Optional archive upsert mirrors the single-domain path.
        if archive is not None:
            try:
                spec_text = f"{xd_result.spec.title}. {xd_result.spec.one_paragraph_description}"
                emb = (embedder or _fallback_embedder)(spec_text)
                archive.upsert_goal(
                    "goal_archive",
                    goal_id=xd_result.spec.goal_id,
                    embedding=emb,
                    document=spec_text,
                    metadata={
                        "title": xd_result.spec.title,
                        "artifact_type": xd_result.spec.artifact_type,
                        "domain_tags": [
                            xd_result.mechanism.source_domain,
                            xd_result.mechanism.target_domain,
                        ]
                        if xd_result.mechanism is not None
                        else [],
                        "status": "pending_build",
                        "source_citation": xd_result.spec.source_citation,
                        "structural_mechanism_present": True,
                    },
                )
            except Exception as exc:
                summary.errors.append(f"cross_domain_archive:{exc}")

        for rid in row_ids:
            state.set_signal_status(rid, "promoted")
        summary.promoted += 1


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _row_to_seed(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source": row["source"],
        "source_id": row["source_id"],
        "title": row["title"],
        "summary": row["summary"],
        "raw_excerpt": row["raw_excerpt"],
        "captured_at": row["captured_at"],
        # domain_tags not currently stored; later sessions may add a join
        "domain_tags": [],
    }


def _dict_to_neighbor(d: dict[str, Any]) -> Any:
    """Lightweight adapter — generator only reads .to_prompt_dict() later."""
    from belief.photosynthesis.synthesis.archives import Neighbor

    return Neighbor(
        goal_id=str(d.get("goal_id", "")),
        title=str(d.get("title", "")),
        cosine=float(d.get("cosine", 0.0)),
    )


def _fallback_embedder(text: str) -> list[float]:
    """A tiny, deterministic 8-dim embedder for tests and degraded paths.

    Maps `text` to an 8-dim vector via hash buckets; NOT a real
    semantic embedding. Production wires MiniLM through
    filter/cascade.py's embed model.
    """
    vec = [0.0] * 8
    for i, ch in enumerate(text):
        vec[ord(ch) % 8] += 1.0
    # Normalize
    s = sum(v * v for v in vec) ** 0.5
    return [v / s for v in vec] if s else vec


__all__ = [
    "CycleSummary",
    "run_synthesis_cycle",
    "run_synthesis_cycle_sync",
]
