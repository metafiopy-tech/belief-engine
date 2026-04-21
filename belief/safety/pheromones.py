"""
Pheromone trails — stigmergic coordination for self-modification (Session 16).

When a self-modification fires it drops a **pheromone**: a small
JSON record describing what changed, when, and whether the change
held up.  Pheromones decay over time (24-hour half-life) so the
trail naturally fades as the system stabilises.

The density of pheromones around a module is a coordination signal
that every self-modifier can read without coupling to every other:

* **Hot zone** — many recent, successful pheromones → this module
  is under active attention and further changes are cheap
  (incremental refinement).
* **Cold zone** — no pheromones or only ancient ones → require
  stronger evidence (full danger-theory gate, confirmed regression,
  etc.) before touching the module.

Everything is plain JSONL on disk (the Session-16 constraint: "no
ChromaDB").  One file per module so readers can mmap-scan a single
trail without loading everything.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("belief.safety.pheromones")


# ── Defaults ───────────────────────────────────────────────────────────────


# Default on-disk root for pheromone files.  Kept under
# ``~/.belief-engine`` to match the rest of the persistent state.
DEFAULT_PHEROMONE_DIR = Path("~/.belief-engine/pheromones").expanduser()

# Half-life of a single pheromone, in seconds.  Spec default: 24h.
DEFAULT_HALF_LIFE_SECONDS = 24 * 60 * 60

# Density threshold above which a zone counts as "hot".  Roughly
# equivalent to three successful modifications within the last 24
# hours, given a half-life weighting.
DEFAULT_HOT_ZONE_THRESHOLD = 1.5


# ── Data model ─────────────────────────────────────────────────────────────


@dataclass
class PheromoneTrail:
    """One modification event left on disk.

    Fields:
        module:      Canonical module path (``belief/memory/soil.py``).
        timestamp:   UTC epoch seconds of the modification.
        description: Short free-text summary ("crystallized
                     covenant-7", "refactor to use async DB pool").
        outcome:     ``"success"`` | ``"failure"`` | ``"deferred"`` —
                     callers decide their own vocabulary but these
                     three play nicely with :meth:`density`.
        source:      Optional label for the originator
                     (``"sica"``, ``"new_tool"``, ``"jitterbug"``).
        weight:      Multiplier on the decay-weighted contribution.
                     Defaults to 1.0; safety-oriented callers can
                     give successful modifications a larger weight
                     and failed ones a smaller (or negative) one so
                     the density naturally down-weights bad zones.
    """

    module: str
    timestamp: float
    description: str = ""
    outcome: str = "success"
    source: str = ""
    weight: float = 1.0

    def decay_weight(
        self,
        now: Optional[float] = None,
        half_life: float = DEFAULT_HALF_LIFE_SECONDS,
    ) -> float:
        """Multiplicative decay factor at time *now*.

        Uses ``(1/2)^(age / half_life)`` — equivalent to exponential
        decay with λ = ln(2)/half_life.  A pheromone at exactly one
        half-life of age contributes 0.5 of its weight; at 2× half-
        life, 0.25; and so on.
        """
        if now is None:
            now = time.time()
        age = max(0.0, float(now) - float(self.timestamp))
        if half_life <= 0:
            return 0.0 if age > 0 else 1.0
        return math.pow(0.5, age / float(half_life))


# ── Filesystem helpers ────────────────────────────────────────────────────


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _module_filename(module: str) -> str:
    """Slugified on-disk name for a module's trail file.

    ``belief/memory/soil.py`` → ``belief_memory_soil.py.jsonl``.  We
    keep the ``.py`` suffix inside the slug so human-eyeing the
    directory hints at which module each trail is for.
    """
    slug = _SAFE_NAME_RE.sub("_", module.strip().strip("/").strip("\\"))
    if not slug:
        slug = "_unknown_"
    return f"{slug}.jsonl"


def _trail_path(module: str, base_dir: Path) -> Path:
    return Path(base_dir) / _module_filename(module)


def _ensure_dir(base_dir: Path) -> Path:
    p = Path(base_dir).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Public API ────────────────────────────────────────────────────────────


def deposit_pheromone(
    module: str,
    description: str,
    outcome: str = "success",
    *,
    source: str = "",
    weight: float = 1.0,
    timestamp: Optional[float] = None,
    base_dir: Path = DEFAULT_PHEROMONE_DIR,
) -> PheromoneTrail:
    """Append a new pheromone to the module's on-disk trail.

    Returns the :class:`PheromoneTrail` that was written so the
    caller can include the record in its own audit log.  Never
    raises on filesystem errors — logs them and returns a detached
    in-memory trail so the self-modification pipeline keeps moving.
    """
    ts = float(time.time()) if timestamp is None else float(timestamp)
    trail = PheromoneTrail(
        module=module,
        timestamp=ts,
        description=description,
        outcome=outcome,
        source=source,
        weight=float(weight),
    )
    try:
        base = _ensure_dir(Path(base_dir))
        path = _trail_path(module, base)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(trail), ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning(
            f"pheromone deposit failed for {module!r}: {exc}; "
            f"continuing without persistent trail"
        )
    return trail


def read_pheromones(
    module: str,
    base_dir: Path = DEFAULT_PHEROMONE_DIR,
) -> list[PheromoneTrail]:
    """Return every stored pheromone for ``module`` in insertion order.

    Malformed or partially-written lines are skipped (logged at
    debug level) rather than aborting the read.  Returns an empty
    list when the trail file is absent.
    """
    path = _trail_path(module, Path(base_dir).expanduser())
    if not path.exists():
        return []
    trails: list[PheromoneTrail] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    trails.append(PheromoneTrail(
                        module=str(data.get("module", module)),
                        timestamp=float(data.get("timestamp", 0.0)),
                        description=str(data.get("description", "")),
                        outcome=str(data.get("outcome", "success")),
                        source=str(data.get("source", "")),
                        weight=float(data.get("weight", 1.0)),
                    ))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.debug(
                        f"pheromone read: skipping {path}:{lineno}: {exc}"
                    )
    except OSError as exc:
        logger.warning(f"pheromone read failed for {module!r}: {exc}")
    return trails


def pheromone_density(
    module: str,
    *,
    base_dir: Path = DEFAULT_PHEROMONE_DIR,
    now: Optional[float] = None,
    half_life: float = DEFAULT_HALF_LIFE_SECONDS,
    outcome_filter: Optional[Iterable[str]] = None,
) -> float:
    """Decay-weighted sum of pheromones on ``module``.

    Args:
        module:         Module path / slug.
        base_dir:       On-disk root.
        now:            Override "current time" (epoch seconds) for
                        deterministic tests.
        half_life:      Decay half-life in seconds (default 24h).
        outcome_filter: Optional set of outcomes to include.  When
                        None, every outcome counts; pass e.g.
                        ``{"success"}`` to ignore deferred/failed
                        attempts.

    Returns a non-negative float.  Callers compare against
    :data:`DEFAULT_HOT_ZONE_THRESHOLD` (or their own threshold) to
    classify hot vs. cold.
    """
    trails = read_pheromones(module, base_dir=base_dir)
    if not trails:
        return 0.0
    t_now = float(time.time()) if now is None else float(now)
    accepted = set(outcome_filter) if outcome_filter else None
    total = 0.0
    for t in trails:
        if accepted is not None and t.outcome not in accepted:
            continue
        total += t.weight * t.decay_weight(now=t_now, half_life=half_life)
    return max(0.0, total)


def is_hot_zone(
    module: str,
    *,
    threshold: float = DEFAULT_HOT_ZONE_THRESHOLD,
    base_dir: Path = DEFAULT_PHEROMONE_DIR,
    now: Optional[float] = None,
    half_life: float = DEFAULT_HALF_LIFE_SECONDS,
    outcome_filter: Optional[Iterable[str]] = None,
) -> bool:
    """Whether ``module``'s pheromone density exceeds ``threshold``.

    Thin wrapper over :func:`pheromone_density` — kept separate so
    callers can express intent at the call site.
    """
    return pheromone_density(
        module,
        base_dir=base_dir, now=now, half_life=half_life,
        outcome_filter=outcome_filter,
    ) >= threshold


def clear_pheromones(
    module: str,
    base_dir: Path = DEFAULT_PHEROMONE_DIR,
) -> int:
    """Delete the trail file for ``module``; returns bytes removed.

    Used by maintenance jobs that want to prune a module's trail
    (e.g. after a big refactor that rendered old pheromones
    irrelevant).  Missing files return 0 rather than raising.
    """
    path = _trail_path(module, Path(base_dir).expanduser())
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    try:
        path.unlink()
    except OSError as exc:
        logger.warning(f"pheromone clear failed for {module!r}: {exc}")
        return 0
    return int(size)
