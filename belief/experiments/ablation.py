"""Self-ablation instrument — toggle one mechanism, attribute the effect.

The reusable generalization of the STARVED arm machinery (see
``docs/experiments/agent_harness_program.md`` stage #3). Where ``starved_runner``
hard-coded two arms (FED vs STARVED) and one variable (the admission key), this
harness takes an arbitrary set of arms, each defined by a set of **mechanism
toggles** (env-condition overrides), runs the same task stream through each, and
reports each arm's outcome plus its **delta from a baseline arm** — so a single
toggled mechanism's contribution is cleanly attributable.

Design, carried over from what worked:

- **Substrate-agnostic.** All external work is an injected seam (``build_fn``,
  ``metric_fn``); the harness never imports the engine. The compute substrate —
  cheap proxy, tiny real builds, smaller model — is the caller's choice per run.
  Unit tests drive it with deterministic fakes; the real seams live in the CLI
  layer (future session).
- **Per-arm isolation.** Each arm gets its own soil dir under the run directory
  (via ``BELIEF_SOIL_PATH``) so arms never cross-contaminate, exactly like the
  STARVED arms.
- **Run-id guardrail + manifest.** Refuse to start on a non-empty run dir unless
  ``resume``; record the arm definitions + seed so a run is reproducible and a
  silent re-use can't masquerade as fresh data.
- **Load-bearing test.** A mechanism is load-bearing iff its arm's metric differs
  from baseline beyond the baseline's own noise (the caller supplies the noise
  band; the harness reports the delta and whether it clears the band).

This module only *orchestrates and attributes*. It runs no builds itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class AblationArm:
    """One arm: a name plus the env-condition overrides that define it.

    ``env`` is the set of environment toggles that make this arm differ from the
    baseline (e.g. ``{"BELIEF_SUPPRESS_DECOMPOSE": "1"}`` or
    ``{"BELIEF_EXPERIMENT_CONDITION": "soil_only"}``). The baseline arm typically
    has ``env={}`` (engine defaults). Keeping arms as plain env dicts means any
    mechanism reachable by a flag is ablatable without re-plumbing.
    """

    name: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ArmResult:
    """Aggregated outcome for one arm."""

    name: str
    n_tasks: int
    metric_mean: float
    metric_values: list[float] = field(default_factory=list)


@dataclass
class AblationConfig:
    experiment_id: str
    base_dir: Path
    arms: list[AblationArm]
    tasks: list[tuple[str, str]]  # (task_id, goal)
    baseline: str  # name of the baseline arm (deltas are computed against it)
    seed: int = 42
    resume: bool = False

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).expanduser()
        names = [a.name for a in self.arms]
        if len(names) != len(set(names)):
            raise ValueError(f"arm names must be unique: {names}")
        if self.baseline not in names:
            raise ValueError(f"baseline {self.baseline!r} not among arms {names}")

    @property
    def run_dir(self) -> Path:
        return self.base_dir / self.experiment_id

    def arm_soil(self, arm_name: str) -> Path:
        return self.run_dir / arm_name / "soil"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"


@dataclass
class AblationReport:
    experiment_id: str
    baseline: str
    results: dict[str, ArmResult]
    # delta[arm] = arm.metric_mean - baseline.metric_mean
    deltas: dict[str, float]

    def load_bearing(self, arm_name: str, noise_band: float) -> bool:
        """True if the arm's delta from baseline exceeds the noise band (abs)."""
        return abs(self.deltas.get(arm_name, 0.0)) > noise_band


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunDirNotEmptyError(RuntimeError):
    """Target run dir already has data and resume was not requested."""


class ManifestDriftError(RuntimeError):
    """Resumed run's arm definitions / seed differ from the manifest."""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# build_fn(goal, arm_name, soil_dir, env) -> outcome object (opaque to harness)
BuildFn = Callable[[str, str, Path, dict], object]
# metric_fn(outcome) -> float (the scalar the arms are compared on)
MetricFn = Callable[[object], float]


class AblationRunner:
    """Runs each arm over the task stream and attributes effects to baseline."""

    def __init__(
        self,
        config: AblationConfig,
        *,
        build_fn: BuildFn,
        metric_fn: MetricFn,
    ) -> None:
        self.config = config
        self.build_fn = build_fn
        self.metric_fn = metric_fn

    # -- guardrail + manifest ------------------------------------------------

    def _manifest(self) -> dict:
        cfg = self.config
        return {
            "experiment_id": cfg.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": cfg.seed,
            "baseline": cfg.baseline,
            "arms": {a.name: a.env for a in cfg.arms},
            "n_tasks": len(cfg.tasks),
        }

    def prepare(self) -> None:
        cfg = self.config
        manifest = self._manifest()
        if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
            if not cfg.resume:
                raise RunDirNotEmptyError(
                    f"{cfg.run_dir} is non-empty; pass resume=True to continue"
                )
            self._assert_manifest_matches(manifest)
        else:
            for arm in cfg.arms:
                cfg.arm_soil(arm.name).mkdir(parents=True, exist_ok=True)
            cfg.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _assert_manifest_matches(self, manifest: dict) -> None:
        try:
            existing = json.loads(self.config.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ManifestDriftError(f"cannot read manifest to resume: {e}") from e
        for key in ("seed", "baseline", "arms"):
            if existing.get(key) != manifest.get(key):
                raise ManifestDriftError(
                    f"{key} drifted: manifest={existing.get(key)!r} now={manifest.get(key)!r}"
                )

    # -- run -----------------------------------------------------------------

    def run_arm(self, arm: AblationArm) -> ArmResult:
        """Run one arm over all tasks; return its aggregated metric."""
        soil_dir = self.config.arm_soil(arm.name)
        env = dict(arm.env)
        env["BELIEF_SOIL_PATH"] = str(soil_dir)
        values: list[float] = []
        for _task_id, goal in self.config.tasks:
            outcome = self.build_fn(goal, arm.name, soil_dir, env)
            values.append(float(self.metric_fn(outcome)))
        mean = sum(values) / len(values) if values else 0.0
        return ArmResult(name=arm.name, n_tasks=len(values), metric_mean=mean, metric_values=values)

    def run(self) -> AblationReport:
        self.prepare()
        results: dict[str, ArmResult] = {}
        for arm in self.config.arms:
            results[arm.name] = self.run_arm(arm)
        base_mean = results[self.config.baseline].metric_mean
        deltas = {name: r.metric_mean - base_mean for name, r in results.items()}
        return AblationReport(
            experiment_id=self.config.experiment_id,
            baseline=self.config.baseline,
            results=results,
            deltas=deltas,
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_report(report: AblationReport, *, noise_band: float = 0.0) -> str:
    """Human-readable arm table with baseline-relative deltas."""
    lines = [f"Ablation — {report.experiment_id}  (baseline: {report.baseline})", "=" * 60]
    for name, r in report.results.items():
        delta = report.deltas[name]
        tag = ""
        if name != report.baseline:
            lb = abs(delta) > noise_band
            tag = f"  delta={delta:+.4f} {'[LOAD-BEARING]' if lb else '[within noise]'}"
        lines.append(f"  {name:<24} metric={r.metric_mean:.4f} (n={r.n_tasks}){tag}")
    if noise_band:
        lines.append(
            f"\n  noise band: +/-{noise_band:.4f} (mechanism load-bearing if |delta| exceeds)"
        )
    return "\n".join(lines)
