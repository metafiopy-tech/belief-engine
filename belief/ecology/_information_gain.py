"""Gap identification + info-gain heuristics for Curiosity (v3.3 §3.5).

v1 ships simple, defensible heuristics. The spec explicitly notes the
sophisticated learned info-gain estimator is a v3.4 follow-up.

**Gap categories shipped here:**

1. ``file_ext_gap`` — common file extensions absent from soil's
   tool/skeleton corpus. E.g., zero .toml parsers means TOML config
   loading is an unexplored capability.
2. ``framework_gap`` — well-known framework tags absent or thinly
   represented (e.g., zero "fastmcp" or "starlette" patterns when
   "fastapi" is heavy means the engine has only one slice of the
   web-framework space).
3. ``covenant_sparse`` — soft signal: very few crystallized covenants
   in soil overall means the engine hasn't yet condensed many cross-
   build invariants. Surfaces as one gap regardless of count.

**What's deferred to v3.4** (per spec §3.5):
- Build-history embedding clusters (sparsity in semantic space)
- Per-covenant fire frequency from cross-build counters
  (covenant_registry's ``_fire_counts`` is per-process today)
- Learned info-gain model trained on observed builds
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger("belief.ecology.info_gain")

# ── Heuristic catalogs ────────────────────────────────────────────────────

# File extensions Curiosity will check coverage for. The list is
# intentionally short and biased toward "things the engine probably
# *should* know how to handle" — config formats, common scripting
# targets, important data interchange formats.
TRACKED_EXTENSIONS: tuple[str, ...] = (
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".env",
    ".sql",
    ".sh",
    ".dockerfile",
    ".graphql",
    ".proto",
)

# Frameworks Curiosity will check for tag-coverage. Same bias: things
# a serious Python codegen system should be able to produce.
TRACKED_FRAMEWORKS: tuple[str, ...] = (
    "fastapi",
    "fastmcp",
    "starlette",
    "flask",
    "click",
    "typer",
    "pydantic",
    "sqlalchemy",
    "celery",
    "pytest",
)

# Below this number of crystallized covenants total, Curiosity flags
# "soil has thin invariant coverage" as a gap. Tuned to match the
# typical post-Sleep state of Joe's 200+ nutrient soil today.
COVENANT_SPARSE_THRESHOLD = 5


# ── Data ──────────────────────────────────────────────────────────────────


@dataclass
class Gap:
    """One identified knowledge-substrate gap.

    ``signal_strength`` is in [0, 1] — higher means "more glaringly
    absent" (e.g., a heavily-used framework with zero coverage scores
    higher than a niche format with zero coverage). Used as a weight
    when ranking goals that address it.
    """

    category: str  # "file_ext" | "framework" | "covenant_sparse"
    name: str  # the missing thing — ".toml", "fastmcp", "covenants"
    signal_strength: float = 1.0
    rationale: str = ""


# ── Gap identifier ────────────────────────────────────────────────────────


def identify_gaps(soil: Any) -> list[Gap]:
    """Walk the soil and return identified gaps. Tolerates missing collections.

    Reads from the ``belief_tools``, ``belief_episodes``, and
    ``belief_covenants`` collections. If any are absent (e.g., a fresh
    install with no episodes yet), the corresponding detector
    contributes nothing rather than crashing.
    """
    gaps: list[Gap] = []

    tool_metas = _collection_metadatas(soil, "belief_tools")
    episode_metas = _collection_metadatas(soil, "belief_episodes")
    covenant_metas = _collection_metadatas(soil, "belief_covenants")

    # File extension coverage — drawn from tool dependencies + episode
    # code_files keys (when available). Episodes store flat metadata
    # only, so we look at any *_count or *_path style fields plus
    # raw filename hints if present.
    ext_counts = Counter()
    for meta in tool_metas:
        deps = _split_csv(meta.get("dependencies", ""))
        # Tool deps are package names; rough proxy: count files like
        # "Dockerfile" / "requirements.txt" patterns from episode metas
        # rather than tools. Tools rarely emit file extensions directly.
        for dep in deps:
            ext = _ext_for_dep(dep)
            if ext:
                ext_counts[ext] += 1
    for meta in episode_metas:
        if meta.get("has_dockerfile"):
            ext_counts[".dockerfile"] += 1
    for ext in TRACKED_EXTENSIONS:
        if ext_counts.get(ext, 0) == 0:
            gaps.append(
                Gap(
                    category="file_ext",
                    name=ext,
                    signal_strength=0.6,
                    rationale=f"Zero coverage of {ext} format across soil",
                )
            )

    # Framework coverage — check tag presence across tool nutrients.
    framework_counts = Counter()
    for meta in tool_metas:
        framework = (meta.get("framework") or "").lower()
        if framework:
            framework_counts[framework] += 1
        for tag in _split_csv(meta.get("tags", "")):
            tag_lower = tag.strip().lower()
            if tag_lower in TRACKED_FRAMEWORKS:
                framework_counts[tag_lower] += 1
    total_framework_mentions = max(1, sum(framework_counts.values()))
    for fw in TRACKED_FRAMEWORKS:
        if framework_counts.get(fw, 0) == 0:
            # Stronger signal when other frameworks are well-represented.
            density = sum(framework_counts.values()) / max(1, len(TRACKED_FRAMEWORKS))
            strength = min(1.0, 0.5 + density / max(1.0, total_framework_mentions))
            gaps.append(
                Gap(
                    category="framework",
                    name=fw,
                    signal_strength=strength,
                    rationale=f"Zero {fw} patterns/tools in soil",
                )
            )

    # Sparse-covenant signal — global, only one Gap object.
    crystallized = [
        m for m in covenant_metas if "crystallized" in (_split_csv(m.get("tags", "")) or [])
    ]
    if len(crystallized) < COVENANT_SPARSE_THRESHOLD:
        gaps.append(
            Gap(
                category="covenant_sparse",
                name="covenants",
                signal_strength=0.4,
                rationale=(
                    f"Only {len(crystallized)} crystallized covenants in soil "
                    f"(<{COVENANT_SPARSE_THRESHOLD} threshold) — wide invariant "
                    "discovery space remains"
                ),
            )
        )

    return gaps


# ── Info gain estimate ────────────────────────────────────────────────────


def estimate_info_gain(goal_text: str, gaps: list[Gap]) -> tuple[float, list[Gap]]:
    """Score how well a goal addresses identified gaps. Returns (score, addressed).

    v1 heuristic: a goal "addresses" a gap if the gap's name (lowercased)
    appears as a token in the goal text. Score is the sum of addressed
    gaps' ``signal_strength`` values, normalized to [0, 1] by the total
    available signal.
    """
    if not gaps:
        return 0.0, []
    text_tokens = set(re.findall(r"[a-z0-9.]+", goal_text.lower()))
    addressed: list[Gap] = []
    score_raw = 0.0
    for gap in gaps:
        # Match name with and without leading dot for file extensions.
        candidates = {gap.name.lower(), gap.name.lower().lstrip(".")}
        if candidates & text_tokens:
            addressed.append(gap)
            score_raw += gap.signal_strength
    total = sum(g.signal_strength for g in gaps) or 1.0
    return min(1.0, score_raw / total), addressed


# ── Helpers ───────────────────────────────────────────────────────────────


def _collection_metadatas(soil: Any, name: str) -> list[dict]:
    """Best-effort fetch of all metadata rows from a named collection."""
    try:
        col = soil._collections.get(name)
    except AttributeError:
        return []
    if col is None:
        return []
    try:
        n = col.count()
    except Exception:
        return []
    if n == 0:
        return []
    try:
        data = col.get(include=["metadatas"], limit=n)
    except Exception:
        return []
    metas = data.get("metadatas") or []
    return [m for m in metas if isinstance(m, dict)]


def _split_csv(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _ext_for_dep(dep: str) -> str | None:
    """Map a package name to an extension Curiosity tracks, if any.

    Heuristic; covers the common Python-ecosystem cases. Returns None
    for deps that don't imply one of TRACKED_EXTENSIONS.
    """
    d = dep.lower().strip()
    mapping = {
        "tomli": ".toml",
        "tomli-w": ".toml",
        "tomllib": ".toml",
        "pyyaml": ".yaml",
        "yaml": ".yaml",
        "configparser": ".ini",
        "python-dotenv": ".env",
        "dotenv": ".env",
        "sqlalchemy": ".sql",
        "psycopg2": ".sql",
        "psycopg": ".sql",
        "graphql-core": ".graphql",
        "ariadne": ".graphql",
        "strawberry-graphql": ".graphql",
        "grpcio": ".proto",
        "protobuf": ".proto",
    }
    return mapping.get(d)


def gaps_summary(gaps: Iterable[Gap]) -> str:
    """Human-readable one-liner describing identified gaps."""
    by_cat: dict[str, list[str]] = {}
    for g in gaps:
        by_cat.setdefault(g.category, []).append(g.name)
    parts = [
        f"{cat}: {len(names)} ({', '.join(names[:3])}{'…' if len(names) > 3 else ''})"
        for cat, names in by_cat.items()
    ]
    return "; ".join(parts) if parts else "no gaps"
