"""Generation-loop driver for the STARVED-arm experiment.

Runs the K-matched FED-vs-STARVED loop described in
``docs/experiments/starved_arm_design.md``:

  for each generation:
    for each arm (FED, STARVED):
      - run N_TASKS builds against THIS arm's evolving soil (BELIEF_SOIL_PATH),
        with auto-decompose suppressed (BELIEF_SUPPRESS_DECOMPOSE) so candidate
        builds read soil but do not write it;
      - self-judge each build (frozen prompt, same model as the builder);
      - admit the arm's top-K (FED by external grader, STARVED by self_score);
      - controlled influx: decompose ONLY the admitted builds into the arm soil
        (this is what gen N+1 eats);
      - snapshot the arm soil and log admission events;
      - at configured checkpoints, probe held-out generalization.

Per-arm builds (16/gen at 8 tasks) are deliberate: each arm builds against its
own soil, so soil divergence feeds back into builds — the only *exogenous*
difference is the admission key; the soil divergence is its downstream
consequence (design decision, 2026-06-02).

**Everything external is an injectable seam** (``build_fn``, ``judge_fn``,
``decompose_fn``, ``snapshot_fn``, ``probe_fn``) so the orchestration, guardrail,
and bookkeeping are unit-testable without Ollama, ChromaDB, or a real model. The
CLI wires the real seams via :func:`make_default_runner` (lazy imports). The live
run is verified on the Mac per a manual checklist — it cannot run in CI.

**Run isolation guardrail (hard-coded):** every run writes under a directory
keyed by ``experiment_id``. The runner refuses to start if that directory is
non-empty unless ``resume=True``; on resume it refuses if the encoder / judge /
k fingerprints differ from the manifest. Silent state leak between runs would
show up as a suspiciously healthy STARVED curve — a false negative on our own
thesis — so this is a refuse-to-start condition, not a convention.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from belief.experiments import admission_log
from belief.experiments.admission import Candidate, select_admissions
from belief.experiments.self_judge import SELF_JUDGE_PROMPT_FINGERPRINT, JudgeResult

logger = logging.getLogger("belief.experiments.starved")

ARMS = ("FED", "STARVED")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunDirNotEmptyError(RuntimeError):
    """Target run directory already has data and resume was not requested."""


class FingerprintDriftError(RuntimeError):
    """Encoder / judge / k fingerprint differs from the resumed run's manifest."""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class BuildArtifact:
    """One candidate build's result, enough to judge and admit it."""

    run_id: str
    goal: str
    code_files: dict[str, str]
    external_score: float  # FED ranking key (weighted score / fraction passed)
    external_pass: bool  # boolean external verdict (for fiction counting)
    cost_usd: float = 0.0
    wallclock_s: float = 0.0
    error: Optional[str] = None


@dataclass
class ProbeResult:
    """Held-out generalization probe outcome for one (gen, arm)."""

    gen: int
    arm: str
    n_instances: int
    n_resolved: int

    @property
    def resolve_rate(self) -> float:
        return self.n_resolved / self.n_instances if self.n_instances else 0.0


@dataclass
class StarvedConfig:
    """Configuration + pre-registration fingerprints for one experiment run."""

    experiment_id: str
    base_dir: Path
    n_generations: int
    n_tasks: int
    k: int
    kmeans_k: int = 8
    seed: int = 42
    local_model: str = "qwen2.5-coder:14b"
    encoder_fingerprint: str = ""
    judge_prompt_fingerprint: str = SELF_JUDGE_PROMPT_FINGERPRINT
    minilm_revision: str = "main"
    probe_at: tuple[int, ...] = ()
    resume: bool = False
    # Optional explicit task list [(task_id, goal)]; defaults to benchmark tiers.
    tasks: Optional[list[tuple[str, str]]] = None

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).expanduser()

    @property
    def run_dir(self) -> Path:
        return self.base_dir / self.experiment_id

    def arm_soil(self, arm: str) -> Path:
        return self.run_dir / arm.upper() / "soil"

    @property
    def snapshots_dir(self) -> Path:
        return self.run_dir / "snapshots"

    @property
    def admissions_db(self) -> Path:
        return self.run_dir / "admissions.db"

    @property
    def probe_db(self) -> Path:
        return self.run_dir / "probe.db"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"


# Seam type aliases (documentation only).
BuildFn = Callable[[str, str, Path], BuildArtifact]  # (goal, arm, soil_dir) -> artifact
JudgeFn = Callable[[str, dict], JudgeResult]  # (goal, code_files) -> JudgeResult
DecomposeFn = Callable[[BuildArtifact, Path], None]  # (artifact, soil_dir) -> None
SnapshotFn = Callable[[int, str, Path], object]  # (gen, arm, soil_dir) -> snapshot
ProbeFn = Callable[[int, str, Path], Optional[ProbeResult]]  # (gen, arm, soil_dir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class StarvedRunner:
    """Drives the generation loop given injected external seams."""

    def __init__(
        self,
        config: StarvedConfig,
        *,
        build_fn: BuildFn,
        judge_fn: JudgeFn,
        decompose_fn: DecomposeFn,
        snapshot_fn: SnapshotFn,
        probe_fn: Optional[ProbeFn] = None,
    ) -> None:
        self.config = config
        self.build_fn = build_fn
        self.judge_fn = judge_fn
        self.decompose_fn = decompose_fn
        self.snapshot_fn = snapshot_fn
        self.probe_fn = probe_fn
        self._task_order_cache: Optional[list[tuple[str, str]]] = None

    # -- guardrail + manifest ------------------------------------------------

    def prepare(self) -> None:
        """Create/validate the run dir and manifest. Enforces the guardrail."""
        cfg = self.config
        run_dir = cfg.run_dir
        manifest = self._build_manifest()

        if run_dir.exists() and any(run_dir.iterdir()):
            if not cfg.resume:
                raise RunDirNotEmptyError(
                    f"{run_dir} is non-empty; pass resume=True to continue an existing run"
                )
            self._assert_manifest_matches(manifest)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg.snapshots_dir.mkdir(parents=True, exist_ok=True)
            for arm in ARMS:
                cfg.arm_soil(arm).mkdir(parents=True, exist_ok=True)
            cfg.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

        if cfg.minilm_revision == "main":
            logger.warning(
                "STARVED: encoder revision is the 'main' placeholder — pin an exact "
                "HF commit SHA before the pilot (see starved_arm_design.md §2.3)."
            )

    def _build_manifest(self) -> dict:
        cfg = self.config
        return {
            "experiment_id": cfg.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "encoder_fingerprint": cfg.encoder_fingerprint,
            "judge_prompt_fingerprint": cfg.judge_prompt_fingerprint,
            "kmeans_k": cfg.kmeans_k,
            "n_generations": cfg.n_generations,
            "n_tasks": cfg.n_tasks,
            "k": cfg.k,
            "seed": cfg.seed,
            "local_model": cfg.local_model,
            "minilm_revision": cfg.minilm_revision,
            "design_doc": "docs/experiments/starved_arm_design.md",
        }

    def _assert_manifest_matches(self, manifest: dict) -> None:
        """On resume, refuse if any frozen fingerprint drifted from the manifest."""
        try:
            existing = json.loads(self.config.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise FingerprintDriftError(f"cannot read manifest to resume: {e}") from e
        for key in ("encoder_fingerprint", "judge_prompt_fingerprint", "kmeans_k"):
            if existing.get(key) != manifest.get(key):
                raise FingerprintDriftError(
                    f"{key} drifted: manifest={existing.get(key)!r} now={manifest.get(key)!r}. "
                    "A drifted encoder/judge/k voids the pre-registration."
                )

    # -- task stream ---------------------------------------------------------

    def _task_order(self) -> list[tuple[str, str]]:
        """Deterministic shuffled task order (cached). Same for both arms."""
        if self._task_order_cache is None:
            if self.config.tasks is not None:
                order = list(self.config.tasks)
            else:
                from belief.benchmark import CHALLENGES

                order = sorted(((c.id, c.goal) for c in CHALLENGES), key=lambda t: t[0])
            random.Random(self.config.seed).shuffle(order)
            self._task_order_cache = order
        return self._task_order_cache

    def task_stream(self, gen: int) -> list[tuple[str, str]]:
        """The n_tasks tasks for a generation, rotating with wraparound."""
        order = self._task_order()
        if not order:
            return []
        n = self.config.n_tasks
        start = (gen * n) % len(order)
        return [order[(start + i) % len(order)] for i in range(n)]

    # -- main loop -----------------------------------------------------------

    def run(self) -> dict:
        """Run all generations for both arms. Returns a summary dict."""
        self.prepare()
        for gen in range(self.config.n_generations):
            for arm in ARMS:
                self._run_arm_generation(gen, arm)
        return {
            "experiment_id": self.config.experiment_id,
            "run_dir": str(self.config.run_dir),
            "generations": self.config.n_generations,
            "fictions": admission_log.count_fictions(
                self.config.experiment_id, db_path=self.config.admissions_db
            ),
        }

    def _run_arm_generation(self, gen: int, arm: str) -> None:
        cfg = self.config
        soil_dir = cfg.arm_soil(arm)

        candidates: list[Candidate] = []
        artifacts: dict[str, BuildArtifact] = {}
        for _task_id, goal in self.task_stream(gen):
            art = self.build_fn(goal, arm, soil_dir)
            jr = self.judge_fn(goal, art.code_files)
            candidates.append(
                Candidate(
                    build_id=art.run_id,
                    external_score=art.external_score,
                    external_pass=art.external_pass,
                    # A failed/garbage judge scores 0 so it can never win a slot.
                    self_score=jr.self_score if jr.ok else 0.0,
                    self_confidence=jr.self_confidence,
                )
            )
            artifacts[art.run_id] = art

        result = select_admissions(candidates, cfg.k)
        admitted = result.admitted_for(arm)

        # Controlled influx: only the admitted builds feed THIS arm's soil.
        for bid in admitted:
            self.decompose_fn(artifacts[bid], soil_dir)

        # Snapshot AFTER influx — this is the soil gen N+1 will build from.
        self.snapshot_fn(gen, arm, soil_dir)

        admission_log.log_arm_generation(
            cfg.experiment_id, gen, arm, candidates, admitted, db_path=cfg.admissions_db
        )

        if gen in cfg.probe_at and self.probe_fn is not None:
            pr = self.probe_fn(gen, arm, soil_dir)
            if pr is not None:
                from belief.experiments.swebench_probe import record_probe

                record_probe(cfg.experiment_id, pr, db_path=cfg.probe_db)


# ---------------------------------------------------------------------------
# Default real seams (lazy imports; used by the CLI, not by unit tests)
# ---------------------------------------------------------------------------


def default_build_fn(model: str) -> BuildFn:
    """Real build seam: run ``belief build`` as a subprocess against arm soil.

    Sets ``BELIEF_SOIL_PATH`` (arm isolation), ``BELIEF_SUPPRESS_DECOMPOSE=1``
    (no auto-deposit), and local mode. Parses ``--json-output`` for run_id +
    scores, then reads the produced files from ``output/<run_id>/``.
    """

    def _run(goal: str, arm: str, soil_dir: Path) -> BuildArtifact:
        from belief.config.settings import settings

        env = os.environ.copy()
        env["BELIEF_SOIL_PATH"] = str(soil_dir)
        env["BELIEF_SUPPRESS_DECOMPOSE"] = "1"
        env["BELIEF_MODEL_MODE"] = "local"
        cmd = [
            "belief",
            "--mode",
            "local",
            "--local-model",
            model,
            "build",
            "--goal",
            goal,
            "--json-output",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        data = {}
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{") and "verdict" in line:
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
        run_id = data.get("run_id", "")
        code_files: dict[str, str] = {}
        if run_id:
            out_dir = Path(settings.output_path) / run_id
            if out_dir.exists():
                for p in sorted(out_dir.rglob("*")):
                    if p.is_file():
                        try:
                            code_files[str(p.relative_to(out_dir))] = p.read_text()
                        except (OSError, UnicodeDecodeError):
                            pass
        return BuildArtifact(
            run_id=run_id or f"norun-{arm}-{abs(hash(goal)) % 10**8}",
            goal=goal,
            code_files=code_files,
            external_score=float(data.get("weighted_score", 0.0)),
            external_pass=data.get("verdict") == "pass",
            cost_usd=float(data.get("cost_usd", data.get("cost", 0.0))),
            error=None if data else (proc.stderr or "no json output")[:300],
        )

    return _run


def default_judge_fn(model: str, *, seed: int = 42) -> JudgeFn:
    """Real judge seam: the build's own local model rates its output.

    Binds :func:`belief.experiments.self_judge.judge_build`'s ``generate`` to an
    Ollama call at temperature 0 / fixed seed so the judgment is reproducible.
    """

    def _generate(prompt: str) -> str:
        import ollama

        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.0, "seed": seed},
        )
        return resp.get("response", "")

    def _judge(goal: str, code_files: dict) -> JudgeResult:
        from belief.experiments.self_judge import judge_build

        return judge_build(goal, code_files, _generate)

    return _judge


def default_decompose_fn() -> DecomposeFn:
    """Real influx seam: deposit each admitted build as a nutrient in arm soil.

    Deterministic, in-process deposition via an explicit ``Soil(soil_dir)`` (no
    reliance on the cached global soil, which would not switch between arms). The
    nutrient's ``embedding_text`` is derived from the produced CODE — not just
    the goal — so FED-admitted (correct) and STARVED-admitted (fiction) builds
    yield divergent soil clouds even though both arms see the same task stream.
    """

    def _decompose(artifact: BuildArtifact, soil_dir: Path) -> None:
        from belief.memory.nutrients import Nutrient, NutrientType
        from belief.memory.soil import Soil

        code_blob = "\n".join(
            f"# {f}\n{artifact.code_files[f]}" for f in sorted(artifact.code_files)
        )
        embedding_text = (f"{artifact.goal}\n{code_blob}")[:8000]
        nutrient = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content=(f"Build for: {artifact.goal}\n{code_blob}")[:8000],
            embedding_text=embedding_text or artifact.goal,
        )
        Soil(soil_dir).deposit(nutrient)

    return _decompose


def default_snapshot_fn(config: StarvedConfig, encoder) -> SnapshotFn:
    """Real snapshot seam: re-embed the arm soil with the frozen encoder."""

    def _snapshot(gen: int, arm: str, soil_dir: Path):
        from belief.experiments.soil_snapshot import snapshot_soil
        from belief.memory.soil import Soil

        return snapshot_soil(
            Soil(soil_dir),
            encoder,
            gen=gen,
            arm=arm,
            kmeans_k=config.kmeans_k,
            dest_dir=config.snapshots_dir,
        )

    return _snapshot


def make_default_runner(config: StarvedConfig) -> StarvedRunner:
    """Wire a runner with the real (Ollama + ChromaDB + MiniLM) seams.

    Lazy: heavy deps are imported only here, so unit tests that inject fakes
    never trigger them. The SWE-bench probe is attached only if ``probe_at`` is
    non-empty (see :mod:`belief.experiments.swebench_probe`).
    """
    from belief.experiments.soil_snapshot import MiniLMEncoder

    encoder = MiniLMEncoder(revision=config.minilm_revision)
    config.encoder_fingerprint = encoder.fingerprint

    probe_fn = None
    if config.probe_at:
        from belief.experiments.swebench_probe import make_probe_fn

        probe_fn = make_probe_fn(config)

    return StarvedRunner(
        config,
        build_fn=default_build_fn(config.local_model),
        judge_fn=default_judge_fn(config.local_model, seed=config.seed),
        decompose_fn=default_decompose_fn(),
        snapshot_fn=default_snapshot_fn(config, encoder),
        probe_fn=probe_fn,
    )
