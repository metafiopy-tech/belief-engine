"""
Production hardening utilities for the Belief Engine.

Three components:
1. BuildBudget — per-build cost tracking with hard ceiling enforcement
2. RateLimiter — token bucket with exponential backoff for API calls
3. SecurityScanner — AST-based scan for banned function calls in generated code

Source: TIER_4_5_SCALING_PLAN.md Phase 4, research report
"""

from __future__ import annotations

import ast
import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

logger = logging.getLogger("belief.hardening")


# ── 1. Build Budget ─────────────────────────────────────────────────────────

# Cost per million tokens (input, output) by model
_MODEL_COSTS = {
    # Haiku
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-haiku-3-20240307": (0.25, 1.25),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # Sonnet
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    # Opus
    "claude-opus-4-6": (5.00, 25.00),
}


@dataclass
class BuildBudget:
    """Per-build cost tracking with hard ceiling enforcement.

    Usage:
        budget = BuildBudget(max_usd=5.00)
        budget.record_call("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        if budget.would_exceed(estimated_cost=0.50):
            raise BudgetExceeded(budget)
        budget.check()  # Raises if over budget
    """
    max_usd: float = 10.00
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _breakdown: list[dict] = field(default_factory=list)

    def record_call(
        self, model: str, input_tokens: int, output_tokens: int,
        role: str = "",
    ) -> float:
        """Record an API call's cost. Returns the cost of this call."""
        costs = _MODEL_COSTS.get(model, (3.0, 15.0))  # Default to Sonnet pricing
        cost = (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000
        self.spent_usd += cost
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self._breakdown.append({
            "model": model, "role": role,
            "input": input_tokens, "output": output_tokens,
            "cost": cost,
        })
        return cost

    def would_exceed(self, estimated_cost: float) -> bool:
        """Check if adding this cost would exceed the budget."""
        return (self.spent_usd + estimated_cost) > self.max_usd

    def check(self) -> None:
        """Raise BudgetExceeded if over budget."""
        if self.spent_usd > self.max_usd:
            raise BudgetExceeded(self)

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd)

    @property
    def utilization(self) -> float:
        return self.spent_usd / self.max_usd if self.max_usd > 0 else 0.0

    def summary(self) -> str:
        return (
            f"${self.spent_usd:.4f} / ${self.max_usd:.2f} "
            f"({self.utilization:.0%}) — "
            f"{self.calls} calls, "
            f"{self.input_tokens:,} in + {self.output_tokens:,} out"
        )


class BudgetExceeded(Exception):
    def __init__(self, budget: BuildBudget):
        self.budget = budget
        super().__init__(f"Build budget exceeded: {budget.summary()}")


# ── 2. Rate Limiter ─────────────────────────────────────────────────────────

class AsyncTokenBucket:
    """Token bucket rate limiter for async API calls.

    Prevents 429 errors during parallel file generation by limiting
    requests per second with burst capacity.

    Usage:
        limiter = AsyncTokenBucket(rate=0.8, burst=5)  # ~48 RPM
        async with limiter:
            result = await api_call()
    """

    def __init__(self, rate: float = 0.8, burst: int = 5):
        """
        Args:
            rate: Tokens added per second (0.8 = ~48 requests/minute)
            burst: Maximum burst size (requests that can fire immediately)
        """
        self._rate = rate
        self._max = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._max, self._tokens + elapsed * self._rate)
            self._last = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Need to wait for a token
            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0

        await asyncio.sleep(wait)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


async def retry_with_backoff(
    coro_fn,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """Retry an async function with exponential backoff and jitter.

    Critical for handling Anthropic 429 rate limit errors during
    parallel file generation.

    Args:
        coro_fn: Async callable (no args) to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Add random jitter to prevent thundering herd
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Only retry on rate limits and transient errors
            if not any(kw in error_str for kw in ("429", "rate", "overloaded", "timeout", "503")):
                raise

            if attempt >= max_retries:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= (0.5 + random.random())  # 50-150% of calculated delay

            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise last_error  # Should never reach here


# ── 3. Security Scanner ─────────────────────────────────────────────────────

# Functions that should never appear in generated code
_BANNED_CALLS = {
    "eval", "exec", "compile",
    "os.system", "os.popen", "os.exec", "os.execv", "os.execve",
    "subprocess.call", "subprocess.Popen", "subprocess.run",
    "shutil.rmtree",
    "__import__",
    "importlib.import_module",
    "ctypes.cdll", "ctypes.CDLL",
}

# Import patterns that indicate suspicious behavior
_BANNED_IMPORTS = {
    "ctypes",
    "pickle",  # Deserialization attacks
}

# Patterns in string literals that suggest data exfiltration
_SUSPICIOUS_STRINGS = [
    "os.environ",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET",
    "OPENAI_API_KEY",
    "requests.post",  # Potential data exfiltration
]


@dataclass
class SecurityViolation:
    """A security issue found in generated code."""
    file: str
    line: int
    severity: str  # "critical", "warning"
    message: str


def _get_call_name(node: ast.Call) -> str:
    """Extract the full dotted name from a Call node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


class _SecurityVisitor(ast.NodeVisitor):
    """AST visitor that flags banned function calls and suspicious patterns."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[SecurityViolation] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        if name in _BANNED_CALLS:
            self.violations.append(SecurityViolation(
                file=self.filename,
                line=node.lineno,
                severity="critical",
                message=f"Banned function call: {name}()",
            ))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in _BANNED_IMPORTS:
                self.violations.append(SecurityViolation(
                    file=self.filename,
                    line=node.lineno,
                    severity="warning",
                    message=f"Suspicious import: {alias.name}",
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in _BANNED_IMPORTS:
            self.violations.append(SecurityViolation(
                file=self.filename,
                line=node.lineno,
                severity="warning",
                message=f"Suspicious import from: {node.module}",
            ))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            for pattern in _SUSPICIOUS_STRINGS:
                if pattern.lower() in node.value.lower():
                    self.violations.append(SecurityViolation(
                        file=self.filename,
                        line=node.lineno,
                        severity="warning",
                        message=f"Suspicious string containing '{pattern}'",
                    ))
        self.generic_visit(node)


def scan_code(code: str, filename: str = "<generated>") -> list[SecurityViolation]:
    """Scan generated Python code for security issues.

    Returns a list of violations. Critical violations should block execution.
    Warnings are informational.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Can't scan invalid code — syntax errors are caught elsewhere

    visitor = _SecurityVisitor(filename)
    visitor.visit(tree)
    return visitor.violations


def scan_all_files(code_files: dict[str, str]) -> list[SecurityViolation]:
    """Scan all generated files for security issues."""
    all_violations = []
    for fname, code in code_files.items():
        if fname.endswith(".py"):
            violations = scan_code(code, fname)
            all_violations.extend(violations)
    return all_violations


def has_critical_violations(violations: list[SecurityViolation]) -> bool:
    """Check if any violations are critical (should block execution)."""
    return any(v.severity == "critical" for v in violations)


# ── 4. Resource Limits (Move 7) ──────────────────────────────────────────────

@dataclass
class AgentLimits:
    """Hard resource limits per agent to prevent runaway costs.

    Each agent gets a max token budget and max execution time.
    If exceeded, the agent is killed and the build continues with a warning.
    """
    max_input_tokens: int = 50_000
    max_output_tokens: int = 10_000
    max_time_seconds: float = 120.0
    max_files_modified: int = 10

    # Track current usage
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    start_time: float = 0.0
    files_modified: int = 0

    def start(self) -> None:
        self.start_time = time.time()

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens

    def record_file_modify(self) -> None:
        self.files_modified += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    def check(self) -> str | None:
        """Check if any limits are exceeded. Returns violation message or None."""
        if self.input_tokens_used > self.max_input_tokens:
            return f"Input token limit exceeded: {self.input_tokens_used}/{self.max_input_tokens}"
        if self.output_tokens_used > self.max_output_tokens:
            return f"Output token limit exceeded: {self.output_tokens_used}/{self.max_output_tokens}"
        if self.elapsed > self.max_time_seconds:
            return f"Time limit exceeded: {self.elapsed:.0f}s/{self.max_time_seconds:.0f}s"
        if self.files_modified > self.max_files_modified:
            return f"File modification limit exceeded: {self.files_modified}/{self.max_files_modified}"
        return None


# Default limits per agent role
AGENT_LIMITS = {
    "intake": AgentLimits(max_input_tokens=10_000, max_output_tokens=3_000, max_time_seconds=30),
    "research": AgentLimits(max_input_tokens=50_000, max_output_tokens=10_000, max_time_seconds=120),
    "planner": AgentLimits(max_input_tokens=30_000, max_output_tokens=8_000, max_time_seconds=60),
    "architect": AgentLimits(max_input_tokens=30_000, max_output_tokens=10_000, max_time_seconds=60),
    "builder": AgentLimits(max_input_tokens=100_000, max_output_tokens=50_000, max_time_seconds=180, max_files_modified=30),
    "tester": AgentLimits(max_input_tokens=50_000, max_output_tokens=15_000, max_time_seconds=60),
    "debugger": AgentLimits(max_input_tokens=50_000, max_output_tokens=10_000, max_time_seconds=60, max_files_modified=5),
    "gap_analyst": AgentLimits(max_input_tokens=30_000, max_output_tokens=5_000, max_time_seconds=60),
    "synthesizer": AgentLimits(max_input_tokens=80_000, max_output_tokens=30_000, max_time_seconds=120),
    "validator": AgentLimits(max_input_tokens=10_000, max_output_tokens=1_000, max_time_seconds=120),  # Mostly deterministic
}


# ── 5. Audit Logger (Move 7) ─────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path
from datetime import datetime as _datetime


class AuditLogger:
    """Append-only audit log of every significant action in a build.

    Records: LLM calls, file writes, deployments, SEED proposals.
    Stored as JSONL (one JSON object per line) for easy analysis.
    """

    def __init__(self, build_id: str, log_dir: _Path | None = None) -> None:
        self.build_id = build_id
        if log_dir is None:
            log_dir = _Path.home() / ".belief-engine" / "audit"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._file = log_dir / f"{build_id}.jsonl"

    def log(self, event_type: str, **data) -> None:
        """Log an event to the audit trail."""
        entry = {
            "ts": _datetime.now().isoformat(),
            "build": self.build_id,
            "event": event_type,
            **data,
        }
        try:
            with open(self._file, "a") as f:
                f.write(_json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Audit logging must never crash the build

    def log_llm_call(self, role: str, model: str, input_tokens: int,
                     output_tokens: int, cost: float) -> None:
        self.log("llm_call", role=role, model=model,
                 input_tokens=input_tokens, output_tokens=output_tokens,
                 cost=round(cost, 6))

    def log_file_write(self, filename: str, chars: int) -> None:
        self.log("file_write", filename=filename, chars=chars)

    def log_deploy(self, target: str, url: str, success: bool) -> None:
        self.log("deploy", target=target, url=url, success=success)

    def log_seed_proposal(self, title: str, target_file: str,
                          confidence: str, status: str) -> None:
        self.log("seed_proposal", title=title, target_file=target_file,
                 confidence=confidence, status=status)

    def log_covenant_enforcement(self, covenant: str, file: str,
                                  action: str) -> None:
        self.log("covenant_enforce", covenant=covenant, file=file, action=action)


# ── 6. SEED Approval Gate (Move 7) ──────────────────────────────────────────

def seed_requires_approval(proposal_title: str, target_file: str,
                           confidence: str) -> bool:
    """Determine if a SEED proposal requires human approval.

    FunSearch-inspired: the codebase is split into a FIXED SCAFFOLD
    (never auto-modified) and EVOLVABLE PRIORITY FUNCTIONS (safe to modify).

    AUTO-APPLY (no human needed):
    - HIGH confidence proposals
    - Targeting evolvable files only (prompts, thresholds, classification)
    - With syntax validation + snapshot rollback on failure

    REQUIRE APPROVAL:
    - LOW or MEDIUM confidence proposals
    - Any proposal targeting critical/scaffold files
    """
    # Critical scaffold files — NEVER auto-modify
    if is_critical_file(target_file):
        return True

    # Only HIGH confidence proposals can be auto-applied
    if confidence.upper() != "HIGH":
        return True

    # Evolvable priority functions — safe for SEED to modify
    evolvable_patterns = [
        "prompts/",           # Agent system/user prompts
        "prompts/__init__",   # Main prompt file
        "config/models.py",   # Model routing decisions
    ]
    is_evolvable = any(p in target_file for p in evolvable_patterns)

    if is_evolvable:
        return False  # Auto-apply allowed

    # Everything else requires approval
    return True


def is_critical_file(filepath: str) -> bool:
    """Check if a file is critical infrastructure that SEED should never modify alone."""
    critical_patterns = [
        "graph.py",       # Pipeline routing
        "llm.py",         # API client
        "hardening.py",   # Security infrastructure
        "cli.py",         # Entry point
        "__init__.py",    # Package initialization
        "config/",        # Configuration
        "validators/",    # Covenant enforcers
    ]
    return any(p in filepath for p in critical_patterns)

