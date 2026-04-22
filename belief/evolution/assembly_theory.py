"""
Assembly-Theory promotion heuristic (Session 15 Task 2).

Assembly theory asks: how many compositional steps would it take to
build this object from scratch, given that any previously-formed
substructure can be reused?  Applied to code:

* The **assembly index** (``AI``) of a tool is the fraction of its
  AST substructures that also appear in at least one other tool in
  the library.  AI → 1.0 means the tool is entirely composed of
  familiar building blocks; AI → 0.0 means its structure is unique.
* The **copy number** of an AST substructure is how many tools in the
  library contain it.  High copy number is the signature of
  selection — common building blocks survive because they work.

Combined with usage counts, AI gives the engine a cheap way to
auto-flag tools for promotion:

    AI  usage  interpretation
    ─── ────── ──────────────────────────────────────────────────────
    hi  hi     fundamental building block → promote widely
    hi  lo     creative novelty built on familiar parts → preserve
    lo  hi     unique pattern with proven value → promote + study
    lo  lo     trivial / one-off → leave alone

No LLM calls — pure ``ast`` module walks and stdlib hashing.  The
module plays nicely with :mod:`belief.memory.tool_registry` (which
supplies the usage counter).
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

logger = logging.getLogger("belief.evolution.assembly_theory")


# Minimum substructure size (in AST nodes) to count toward the index.
# Tiny subtrees like bare ``Name`` or single-literal ``Constant`` are
# ubiquitous and would push AI toward 1.0 for every tool; they also
# carry no real structural information.  The default excludes 1-2
# node subtrees.
DEFAULT_MIN_NODES = 3

# Default promotion classifier thresholds.
DEFAULT_AI_HIGH = 0.6       # "lots of shared substructure"
DEFAULT_USAGE_HIGH = 5      # "reused >= 5 times"


# ── AST canonicalisation ───────────────────────────────────────────────────


def _node_label(node: ast.AST) -> str:
    """Return a canonical label for a single AST node.

    Preserves type plus the handful of string/int attributes that
    distinguish semantically-different nodes (function names, attribute
    names, operator types).  Drops things that would make signatures
    brittle (line numbers, source offsets, constant values).
    """
    label = type(node).__name__
    if isinstance(node, ast.Name):
        return f"Name({node.id})"
    if isinstance(node, ast.Attribute):
        return f"Attribute(.{node.attr})"
    if isinstance(node, ast.FunctionDef):
        return f"FunctionDef({node.name})"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"AsyncFunctionDef({node.name})"
    if isinstance(node, ast.ClassDef):
        return f"ClassDef({node.name})"
    if isinstance(node, ast.arg):
        return f"arg({node.arg})"
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
        # The op attribute is what distinguishes a + b from a * b.
        op_types = ", ".join(type(o).__name__ for o in
                             (getattr(node, "ops", None) or [getattr(node, "op", None)])
                             if o is not None)
        return f"{label}({op_types})"
    return label


def _count_nodes(node: ast.AST) -> int:
    """Total nodes in a subtree — cheap AST size measure."""
    return sum(1 for _ in ast.walk(node))


def _hash_subtree(node: ast.AST) -> str:
    """Stable 16-hex-char hash of a canonical subtree signature.

    Walks the subtree in a deterministic order (parent label then
    sorted child labels).  The signature depends only on structure +
    canonical names, so syntactically-equivalent snippets from
    different source files collide as intended.
    """
    def _sig(n: ast.AST) -> str:
        label = _node_label(n)
        children = [_sig(c) for c in ast.iter_child_nodes(n)]
        # Sorting makes argument order irrelevant within commutative
        # structures (e.g. Call args), at the cost of some precision.
        # That's the right trade-off for "did any library tool use
        # this structure?".
        children.sort()
        return f"{label}[{','.join(children)}]"

    sig = _sig(node)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def extract_signatures(
    code: str,
    *,
    min_nodes: int = DEFAULT_MIN_NODES,
) -> set[str]:
    """Return every substructure hash found in *code*.

    Uses the module AST — a syntax error means zero signatures (no
    fake matches leak through).  Each subtree with ``>= min_nodes``
    contributes one hash to the returned set; the root module itself
    is included so whole-tool matches count too.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.debug("assembly_theory: skipping unparseable tool (%d chars)", len(code))
        return set()

    signatures: set[str] = set()
    for node in ast.walk(tree):
        # Skip containers whose only content is their position; we
        # want structurally-meaningful nodes.
        if isinstance(node, (ast.Load, ast.Store, ast.Del, ast.Pass)):
            continue
        size = _count_nodes(node)
        if size < min_nodes:
            continue
        signatures.add(_hash_subtree(node))
    return signatures


# ── Core scoring ───────────────────────────────────────────────────────────


def assembly_index(
    tool_code: str,
    library: Sequence[str],
    *,
    min_nodes: int = DEFAULT_MIN_NODES,
) -> float:
    """Fraction of the tool's substructures that also appear in *library*.

    The tool itself is excluded from the library for the purpose of
    scoring (a tool never reinforces itself — that would trivially
    give every tool AI=1.0 if it were its own library entry).  Tools
    the caller wants counted as part of the library should come in
    via ``library``; we compare set membership.

    Args:
        tool_code: Source text of the tool being scored.
        library:   Iterable of other tools' source text.
        min_nodes: Minimum AST-node count for a substructure to count.

    Returns:
        Float in ``[0.0, 1.0]``.  ``0.0`` when the tool has no parseable
        substructures.
    """
    tool_sigs = extract_signatures(tool_code, min_nodes=min_nodes)
    if not tool_sigs:
        return 0.0

    library_sigs: set[str] = set()
    for source in library:
        if source == tool_code:
            # Tool must not reinforce itself.
            continue
        library_sigs.update(extract_signatures(source, min_nodes=min_nodes))

    if not library_sigs:
        return 0.0

    shared = tool_sigs & library_sigs
    return len(shared) / len(tool_sigs)


def copy_numbers(
    tool_code: str,
    library: Sequence[str],
    *,
    min_nodes: int = DEFAULT_MIN_NODES,
) -> dict[str, int]:
    """Per-substructure count of how many library tools contain it.

    Useful for diagnostics or deeper promotion heuristics: a tool
    whose most-copied substructure appears in 10 library tools is a
    stronger building-block candidate than one whose signatures each
    appear only once.
    """
    tool_sigs = extract_signatures(tool_code, min_nodes=min_nodes)
    if not tool_sigs:
        return {}

    counts: dict[str, int] = {sig: 0 for sig in tool_sigs}
    for source in library:
        if source == tool_code:
            continue
        lib_sigs = extract_signatures(source, min_nodes=min_nodes)
        for sig in tool_sigs & lib_sigs:
            counts[sig] += 1
    return counts


# ── Promotion classifier ──────────────────────────────────────────────────


@dataclass
class PromotionVerdict:
    """Human-readable outcome of :func:`should_promote`."""

    should_promote: bool
    category: str               # 'building_block' | 'unique_workhorse' |
                                # 'creative_novelty' | 'trivial'
    assembly_index: float
    usage_count: int
    rationale: str = ""


def should_promote(
    tool_code: str,
    library: Sequence[str],
    usage_count: int,
    *,
    ai_high: float = DEFAULT_AI_HIGH,
    usage_high: int = DEFAULT_USAGE_HIGH,
    min_nodes: int = DEFAULT_MIN_NODES,
) -> PromotionVerdict:
    """Classify a tool by assembly index × usage and recommend an action.

    Decision matrix (from the Session 15 spec):

        high AI + high usage → fundamental building block: promote
        high AI + low usage  → creative novelty: preserve but don't
                               promote (over-promoting hurts diversity)
        low AI  + high usage → unique workhorse: promote + study
        low AI  + low usage  → trivial, not informative

    Returns a :class:`PromotionVerdict` with the decision, category
    label, and raw inputs for logging.
    """
    ai = assembly_index(tool_code, library, min_nodes=min_nodes)
    high_ai = ai >= ai_high
    high_usage = usage_count >= usage_high

    if high_ai and high_usage:
        return PromotionVerdict(
            should_promote=True,
            category="building_block",
            assembly_index=ai,
            usage_count=usage_count,
            rationale=(
                f"AI={ai:.2f} ≥ {ai_high} and usage={usage_count} ≥ "
                f"{usage_high}: widely-shared, well-used building block"
            ),
        )
    if high_ai and not high_usage:
        return PromotionVerdict(
            should_promote=False,
            category="creative_novelty",
            assembly_index=ai,
            usage_count=usage_count,
            rationale=(
                f"AI={ai:.2f} ≥ {ai_high} but usage={usage_count} < "
                f"{usage_high}: creative novelty — preserve, don't promote"
            ),
        )
    if not high_ai and high_usage:
        return PromotionVerdict(
            should_promote=True,
            category="unique_workhorse",
            assembly_index=ai,
            usage_count=usage_count,
            rationale=(
                f"AI={ai:.2f} < {ai_high} but usage={usage_count} ≥ "
                f"{usage_high}: unique pattern with proven value — promote"
            ),
        )
    return PromotionVerdict(
        should_promote=False,
        category="trivial",
        assembly_index=ai,
        usage_count=usage_count,
        rationale=(
            f"AI={ai:.2f} < {ai_high} and usage={usage_count} < "
            f"{usage_high}: trivial or unproven"
        ),
    )


def scan_library_for_promotions(
    tools: Iterable[tuple[str, str, int]],
    *,
    ai_high: float = DEFAULT_AI_HIGH,
    usage_high: int = DEFAULT_USAGE_HIGH,
    min_nodes: int = DEFAULT_MIN_NODES,
) -> list[tuple[str, PromotionVerdict]]:
    """Batch scan: yield a ``(tool_id, verdict)`` list for an iterable of tools.

    Args:
        tools: iterable of ``(tool_id, source_code, usage_count)`` triples.
        ai_high / usage_high / min_nodes: passed through to
            :func:`should_promote`.

    Returns the full list (not a generator) so callers can easily
    filter (``[v for v in results if v[1].should_promote]``).
    """
    materialised = list(tools)
    library = [src for (_, src, _) in materialised]
    results: list[tuple[str, PromotionVerdict]] = []
    for tool_id, source, usage in materialised:
        other_library = [s for s in library if s is not source]
        verdict = should_promote(
            source,
            other_library,
            usage,
            ai_high=ai_high,
            usage_high=usage_high,
            min_nodes=min_nodes,
        )
        results.append((tool_id, verdict))
    return results
