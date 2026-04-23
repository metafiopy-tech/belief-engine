"""Collapsed local-mode pipeline — validation-phase Session 1.

The cloud pipeline in :mod:`belief.graph` hits the LLM 8-10 times per
build.  Each hop on a 14B local model costs 30-60 seconds, so the
same build that takes 90s on Claude takes 500-600s on qwen2.5-coder.
This module exposes a shorter pipeline that the CLI selects when
``BELIEF_MODEL_MODE=local`` (single-service builds only — multi-service
builds keep using :mod:`belief.graph_multi`).

## Four logical stages

The spec from *Validation Claude Code Sessions / Session 1* groups the
cloud pipeline into four stages:

    Stage 1 — PLAN    : intake → (research) → planner
    Stage 2 — BUILD   : architect → skeleton → builder + deterministic
                        covenant_enforce + import_fix
    Stage 3 — TEST    : executor with a bounded debugger loop
    Stage 4 — POLISH  : synthesizer → validator → refinement → decomposer

In the collapsed local pipeline we keep those four stages but drop the
agents that are either redundant (research / tester — local models do
this work poorly) or that add LLM cost without changing the outcome
on a local backend (gap_analyst / polarity_check / latios).

## What this module does NOT do

- It does NOT touch :mod:`belief.graph`.  Cloud builds still run the
  full 9-agent pipeline unchanged.  This preserves every existing test
  and the statistical comparability of the engine+cloud numbers in
  the handoff doc.
- It does NOT replace multi-service builds.  ``build_multi_pipeline``
  in :mod:`belief.graph_multi` is still the right choice when the goal
  decomposes into multiple services.  The CLI is responsible for
  picking the right pipeline per-build; see
  :func:`belief.cli.run`.
- It does NOT re-implement covenant enforcement, import fixing, or soil
  deposit.  Those nodes are imported from :mod:`belief.graph` verbatim
  so the self-learned-covenants and autocatalytic-tool stories stay
  identical across cloud and local.

## LLM-call budget

==================  =========================  =================
Stage               Agent                       LLM call?
------------------  -------------------------  -----------------
recomposer          (retrieval)                 no
PLAN                intake                      yes (haiku)
PLAN                planner                     yes (sonnet)
BUILD               architect                   yes (sonnet)
BUILD               skeleton_pass1              no (cached)
BUILD               builder                     yes (sonnet)
BUILD               covenant_enforce            no
BUILD               import_fix                  no
TEST                executor                    no (runs pytest)
TEST                debugger ×N (max 2)         yes (sonnet)
POLISH              synthesizer                 yes (haiku)
POLISH              validator                   no (deterministic)
POLISH              refinement (conditional)    yes (haiku) cycles
POLISH              decomposer                  no
==================  =========================  =================

Happy path: 5 LLM calls.  Worst case with 2 debug rounds + 1 refinement
cycle: 8 calls.  Cloud equivalent is 9-12.

This module deliberately does NOT do the "single merged PLAN agent"
rewrite the spec describes.  Writing a new merged agent means
reproducing every state-mutation side effect the three original agents
do today — a risky refactor that we can revisit once the other speed
wins (keep_alive, compressed context, skeleton cache) are verified in
isolation.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from belief.agents.architect import ArchitectAgent
from belief.agents.builder import BuilderAgent
from belief.agents.debugger import DebuggerAgent
from belief.agents.executor import ExecutorAgent
from belief.agents.intake import IntakeAgent
from belief.agents.planner import PlannerAgent
from belief.agents.skeleton_pass1 import skeleton_pass1_node
from belief.agents.synthesizer import SynthesizerAgent
from belief.agents.validator import ValidatorAgent
from belief.config.models import ModelRouter
from belief.memory.recomposer import recomposer_node
from belief.models.state import Phase

# Reuse the deterministic, zero-LLM helper nodes from the cloud pipeline.
# Importing here (rather than copy-pasting) means covenant / import-fix /
# refinement / trace-wrapper improvements flow through to local mode
# automatically.
from belief.graph import (
    _covenant_enforce_node,
    _exec_error_is_refinable,
    _import_fix_node,
    _increment_iteration,
    _make_refinement_node,
    _traced,
)

logger = logging.getLogger("belief.graph_local")


# Debug-loop budget.  Cloud allows 3 iterations; local halves that both
# to save time (each debug call is ~30s on a 14B model) and because the
# refinement water-cycle catches most fail-fixable issues afterwards.
LOCAL_MAX_DEBUG_ITERATIONS = 2


# ── Routing functions ─────────────────────────────────────────────────────


def _route_after_executor(
    state: dict[str, Any],
) -> Literal["debugger", "synthesizer", "validator"]:
    """After executor: debug on failure (bounded), else polish OR skip to validator.

    Session 4 (v3.2) adds a third route.  Historically a successful
    execution always went to the synthesizer for a ~180-300s polish
    pass; per the research report, no mainstream agentic coder does
    this because post-success polish rarely changes test outcomes.

    The new rule: on success, call :func:`belief.synthesizer_router.should_polish`.
    If any trigger fires (tests_failed, ruff_errors > 3, cyclomatic > 12,
    lines > 150) AND we're still under the wallclock budget, go to
    synthesizer.  Otherwise skip directly to validator.  The env var
    ``SYNTHESIZER_ROUTE_ENABLED=0`` restores pre-session-4 behaviour
    (always polish), which the ablation harness uses.

    Failure path is unchanged: bounded debug loop, then give up into
    the synthesizer (which may also emit a minimal run.sh / deploy
    artifacts that the validator needs).  We keep that behaviour so
    Session 4's change is net-zero on the failure path.
    """
    exec_r = state.get("execution_result")
    exec_success = False
    if exec_r:
        exec_success = (
            exec_r.get("success") if isinstance(exec_r, dict)
            else getattr(exec_r, "success", False)
        )
    if exec_success:
        # Session 4: short-circuit polish when there's nothing to polish.
        try:
            from belief.synthesizer_router import should_polish
            go_polish, reason = should_polish(state)
            logger.info("synthesizer router: %s → %s",
                        reason, "synthesizer" if go_polish else "validator")
            return "synthesizer" if go_polish else "validator"
        except Exception as e:  # pragma: no cover
            logger.debug("synthesizer router errored (%s); defaulting to polish", e)
            return "synthesizer"

    iteration = int(state.get("iteration", 0) or 0)
    if iteration >= LOCAL_MAX_DEBUG_ITERATIONS:
        logger.info(
            "graph_local: debug budget exhausted (%d) — proceeding to synthesizer",
            iteration,
        )
        return "synthesizer"

    return "debugger"


def _route_after_validator(
    state: dict[str, Any],
) -> Literal["refinement", "__end__"]:
    """Local validator routing — a simplified version of the cloud rule.

    Cloud's :func:`belief.graph._route_after_validation` can loop back
    to the builder or research nodes.  In local mode we flatten that:
    if the verdict is fixable-and-ran we hand off to refinement
    (water-cycle patching), otherwise we terminate the graph
    immediately.

    The decomposer used to live at the end of this pipeline and took
    ~60s on a 14B local model (Sonnet-scale abstraction work running
    on Ollama).  It doesn't gate anything the user is waiting for —
    it only deposits nutrients into soil.  The CLI now fires it after
    printing ``BUILD COMPLETE``, which gets the user's result back
    to them ~60s sooner.  See :func:`belief.cli.run`.
    """
    exec_r = state.get("execution_result")
    exec_ok = False
    if exec_r:
        exec_ok = (
            exec_r.get("success") if isinstance(exec_r, dict)
            else getattr(exec_r, "success", False)
        )

    validation = state.get("validation_result")
    if validation is None:
        return "__end__"

    verdict = (
        validation.get("verdict") if isinstance(validation, dict)
        else getattr(validation, "verdict", "pass")
    )
    if hasattr(verdict, "value"):
        verdict = verdict.value
    verdict = str(verdict)

    # Send fixable, runnable builds to refinement — same contract as cloud.
    if verdict == "fail_fixable" and exec_ok:
        return "refinement"

    # Fixable but didn't run: only refine when the error class is one
    # the water-cycle fixer actually handles (mirrors the cloud rule).
    if (
        verdict == "fail_fixable"
        and not exec_ok
        and _exec_error_is_refinable(exec_r)
    ):
        return "refinement"

    return "__end__"


# ── Pipeline construction ─────────────────────────────────────────────────


def build_local_pipeline(router: ModelRouter | None = None) -> StateGraph:
    """Compile the collapsed local-mode pipeline.

    Returns a LangGraph :class:`StateGraph` whose shape matches
    :func:`belief.graph.build_pipeline` so the CLI can swap pipelines
    without touching the build driver.

    The router is optional; passing None builds a fresh
    :class:`ModelRouter` which picks up ``BELIEF_MODEL_MODE`` and
    friends from the environment.
    """
    if router is None:
        router = ModelRouter()

    # Instantiate only the agents this pipeline actually uses. Skipping
    # ResearchAgent, TesterAgent, GapAnalystAgent here keeps their
    # (relatively cheap) construction cost out of the local cold start.
    intake = IntakeAgent(router)
    planner = PlannerAgent(router)
    architect = ArchitectAgent(router)
    builder = BuilderAgent(router)
    executor = ExecutorAgent(router)
    debugger = DebuggerAgent(router)
    synthesizer = SynthesizerAgent(router)
    validator = ValidatorAgent(router)

    graph = StateGraph(dict)

    # ── Stage 0: entry (nutrient retrieval, no LLM) ─────────────────
    graph.add_node("recomposer", _traced(recomposer_node, "recomposer"))

    # ── Stage 1: PLAN ───────────────────────────────────────────────
    graph.add_node("intake", _traced(intake, "intake"))
    graph.add_node("planner", _traced(planner, "planner"))

    # ── Stage 2: BUILD ──────────────────────────────────────────────
    graph.add_node("architect", _traced(architect, "architect"))
    graph.add_node(
        "skeleton_pass1", _traced(skeleton_pass1_node, "skeleton_pass1")
    )
    graph.add_node("builder", _traced(builder, "builder"))
    graph.add_node(
        "covenant_enforce",
        _traced(_covenant_enforce_node, "covenant_enforce"),
    )
    graph.add_node("import_fix", _traced(_import_fix_node, "import_fix"))

    # ── Stage 3: TEST ──────────────────────────────────────────────
    graph.add_node("executor", _traced(executor, "executor"))
    graph.add_node("debugger", _traced(debugger, "debugger"))
    graph.add_node(
        "increment_iteration",
        _traced(_increment_iteration, "increment_iteration"),
    )

    # ── Stage 4: POLISH ────────────────────────────────────────────
    # Note: decomposer is intentionally absent from the local graph.
    # The CLI runs it after printing BUILD COMPLETE so the user's result
    # isn't gated on a 60s Sonnet-scale extraction job on Ollama.
    graph.add_node("synthesizer", _traced(synthesizer, "synthesizer"))
    graph.add_node("validator", _traced(validator, "validator"))
    graph.add_node(
        "refinement", _traced(_make_refinement_node(router), "refinement")
    )

    # ── Edges ──────────────────────────────────────────────────────
    graph.set_entry_point("recomposer")
    graph.add_edge("recomposer", "intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "skeleton_pass1")
    graph.add_edge("skeleton_pass1", "builder")
    graph.add_edge("builder", "covenant_enforce")
    graph.add_edge("covenant_enforce", "import_fix")
    graph.add_edge("import_fix", "executor")

    # Executor → (debug loop, synthesizer, or validator directly)
    # Session 4: "validator" is a new destination — the synthesizer router
    # may decide there's nothing to polish and skip the pass entirely.
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "debugger": "increment_iteration",
            "synthesizer": "synthesizer",
            "validator": "validator",
        },
    )
    # Increment → debugger → executor (one lap; budget checked on return)
    graph.add_edge("increment_iteration", "debugger")
    graph.add_edge("debugger", "executor")

    # Synthesizer → validator
    graph.add_edge("synthesizer", "validator")

    # Validator → refinement? → END
    # (Decomposer is fired post-print by the CLI; see module docstring.)
    graph.add_conditional_edges(
        "validator",
        _route_after_validator,
        {"refinement": "refinement", "__end__": END},
    )
    graph.add_edge("refinement", END)

    return graph.compile()
