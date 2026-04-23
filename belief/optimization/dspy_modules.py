"""
DSPy module wrappers for Belief Engine agents.

Each major agent is wrapped as a DSPy Module so that DSPy/GEPA can
optimize its prompts (instructions + few-shot demos) without modifying
the core agent code.

These are *wrappers*, not replacements.  The core agents in
belief/agents/ continue to work exactly as before.  The optimized
instructions extracted here are saved to PromptStore and optionally
injected back via the recomposer.

DSPy is an optional dependency — this module raises ImportError if
dspy is not installed.
"""

from __future__ import annotations

try:
    import dspy

    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


def _require_dspy():
    if not _DSPY_AVAILABLE:
        raise ImportError(
            "dspy is required for prompt optimization. "
            "Install with: pip install 'belief-engine[optimize]'"
        )


class BeliefPlanner:
    """Planner agent: goal + research -> structured plan."""

    def __init__(self):
        _require_dspy()
        self.plan = dspy.ChainOfThought("goal, research_context, tier -> plan_json")

    def forward(self, goal: str, research_context: str = "", tier: str = "3"):
        return self.plan(goal=goal, research_context=research_context, tier=str(tier))

    def named_predictors(self):
        yield "plan", self.plan


class BeliefArchitect:
    """Architect agent: goal + plan -> file manifest + architecture."""

    def __init__(self):
        _require_dspy()
        self.design = dspy.ChainOfThought("goal, plan, principles, covenants -> architecture_json")

    def forward(self, goal: str, plan: str, principles: str = "", covenants: str = ""):
        return self.design(goal=goal, plan=plan, principles=principles, covenants=covenants)

    def named_predictors(self):
        yield "design", self.design


class BeliefBuilder:
    """Builder agent: architecture + skeleton -> code files."""

    def __init__(self):
        _require_dspy()
        self.build = dspy.ChainOfThought(
            "goal, architecture, skeleton, principles -> code_files_json"
        )

    def forward(self, goal: str, architecture: str, skeleton: str = "", principles: str = ""):
        return self.build(
            goal=goal,
            architecture=architecture,
            skeleton=skeleton,
            principles=principles,
        )

    def named_predictors(self):
        yield "build", self.build


class BeliefTester:
    """Tester agent: code files + architecture -> test files."""

    def __init__(self):
        _require_dspy()
        self.test = dspy.ChainOfThought("goal, code_files_summary, architecture -> test_files_json")

    def forward(self, goal: str, code_files_summary: str, architecture: str):
        return self.test(
            goal=goal,
            code_files_summary=code_files_summary,
            architecture=architecture,
        )

    def named_predictors(self):
        yield "test", self.test


class BeliefDebugger:
    """Debugger agent: error + code context -> fix."""

    def __init__(self):
        _require_dspy()
        self.debug = dspy.ChainOfThought(
            "error_summary, code_context, previous_attempts -> diagnosis, fix_edits_json"
        )

    def forward(self, error_summary: str, code_context: str, previous_attempts: str = ""):
        return self.debug(
            error_summary=error_summary,
            code_context=code_context,
            previous_attempts=previous_attempts,
        )

    def named_predictors(self):
        yield "debug", self.debug


# ── Module registry ─────────────────────────────────────────────────────────

AGENT_MODULES = {
    "planner": BeliefPlanner,
    "architect": BeliefArchitect,
    "builder": BeliefBuilder,
    "tester": BeliefTester,
    "debugger": BeliefDebugger,
}


def get_all_modules() -> dict[str, object]:
    """Instantiate all DSPy agent modules. Raises ImportError if dspy missing."""
    _require_dspy()
    return {name: cls() for name, cls in AGENT_MODULES.items()}


def is_dspy_available() -> bool:
    """Check if dspy is installed."""
    return _DSPY_AVAILABLE
