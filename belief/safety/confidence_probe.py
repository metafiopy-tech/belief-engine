"""Confidence probe — metacognitive sidecar for the Belief Engine.

Predicts build-success probability from per-step features so the graph
can circuit-break or escalate local→cloud on low-confidence signals.

Pipeline:

  TraceCollector (Session 9) writes StepTrace rows
                                      |
                                      v
  ConfidenceProbe.train() -> GradientBoostingClassifier
                                      |
                                      v
  CalibratedClassifierCV (Platt/isotonic) — honest probabilities
                                      |
                                      v
  predict_confidence(features) -> float in [0, 1]
                                      |
                                      v
  should_escalate(p) -> 'proceed' | 'escalate' | 'abort'

Design constraints (spec verbatim):
  - sklearn is the only ML dependency; lazy-imported so the module
    still loads without it installed.
  - Trainable on <1000 samples (small-data regime). Uses only the
    features listed in the spec plus a couple of derived ones
    (step_position, error_keywords).
  - Prediction is <1ms — just a few decision trees + logistic
    calibrator.
  - Untrained probe returns 1.0 (always "proceed") so routing falls
    back to normal behavior when no model has been trained.

Features per step (fed to the classifier):
  - agent_name         one-hot over the known set
  - iteration          int (debugger loop counter)
  - cost_ratio         cost_so_far / max_budget
  - output_length      chars in output_summary
  - error_keywords     binary: contains 'error' / 'failed' / 'warning'
  - step_position      step_index / expected_total_steps
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


logger = logging.getLogger("belief.safety.confidence_probe")


DEFAULT_PROBE_PATH = Path("~/.belief-engine/probe.pkl").expanduser()

# Stable one-hot agent vocabulary (every agent that emits a trace).
# Adding a new agent is a non-breaking change — unknown agents map to
# the all-zero vector (caught by the sparse-representation handler).
KNOWN_AGENTS: tuple[str, ...] = (
    "intake",
    "research",
    "planner",
    "architect",
    "skeleton_pass1",
    "builder",
    "tester",
    "executor",
    "debugger",
    "gap_analyst",
    "increment_iteration",
    "synthesizer",
    "validator",
    "polarity_check",
    "decomposer",
    "recomposer",
    "refinement",
    "import_fix",
    "covenant_enforce",
)

ERROR_KEYWORDS = ("error", "failed", "traceback", "warning", "exception")

# Spec: expected_total_steps heuristic for step_position normalization.
# v3.0 pipelines commonly run ~15-25 total steps including the debugger
# loop; 20 is a reasonable divisor.
EXPECTED_TOTAL_STEPS = 20


# Classifier hyperparameters — picked for small-data stability.
_GBDT_N_ESTIMATORS = 100
_GBDT_MAX_DEPTH = 3
_GBDT_LEARNING_RATE = 0.1
_CALIBRATION_CV = 3
_CALIBRATION_METHOD = "isotonic"


# Spec thresholds
THRESH_PROCEED = 0.8
THRESH_ESCALATE = 0.4


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass
class StepFeatures:
    """Flat feature vector expected by the probe."""

    agent_one_hot: list[int]           # len == len(KNOWN_AGENTS)
    iteration: int
    cost_ratio: float
    output_length: int
    error_keywords: int                # 0 or 1
    step_position: float               # [0, 1+]

    def to_vector(self) -> list[float]:
        return [
            *[float(x) for x in self.agent_one_hot],
            float(self.iteration),
            float(self.cost_ratio),
            float(self.output_length),
            float(self.error_keywords),
            float(self.step_position),
        ]


def extract_features(
    row: dict[str, Any],
    *,
    max_budget: float = 10.0,
) -> StepFeatures:
    """Turn a TraceCollector training row into a fixed-length feature vector."""
    agent = str(row.get("agent_name", ""))
    one_hot = [1 if agent == known else 0 for known in KNOWN_AGENTS]

    iteration = int(row.get("iteration", 0) or 0)
    cost_so_far = float(row.get("cost_so_far", 0.0) or 0.0)
    cost_ratio = cost_so_far / max_budget if max_budget > 0 else 0.0

    output = str(row.get("output_summary", "") or "")
    output_length = len(output)
    lowered = output.lower()
    error_keywords = 1 if any(kw in lowered for kw in ERROR_KEYWORDS) else 0

    step_index = int(row.get("step_index", 0) or 0)
    step_position = step_index / EXPECTED_TOTAL_STEPS

    return StepFeatures(
        agent_one_hot=one_hot,
        iteration=iteration,
        cost_ratio=cost_ratio,
        output_length=output_length,
        error_keywords=error_keywords,
        step_position=step_position,
    )


# ---------------------------------------------------------------------------
# ConfidenceProbe
# ---------------------------------------------------------------------------


@dataclass
class ProbeMetadata:
    n_samples: int = 0
    n_positive: int = 0
    calibrated: bool = False
    min_samples_required: int = 200
    feature_dim: int = 0


def _load_sklearn() -> Optional[dict[str, Any]]:
    """Best-effort sklearn import. Returns None if not installed."""
    try:
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
        from sklearn.metrics import (  # type: ignore[import-untyped]
            accuracy_score,
            brier_score_loss,
            confusion_matrix,
        )

        return {
            "CalibratedClassifierCV": CalibratedClassifierCV,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "accuracy_score": accuracy_score,
            "brier_score_loss": brier_score_loss,
            "confusion_matrix": confusion_matrix,
        }
    except ImportError:
        return None


class ConfidenceProbe:
    """sklearn-backed build-success probe, with file persistence."""

    def __init__(self, model_path: Path | str = DEFAULT_PROBE_PATH) -> None:
        self.model_path = Path(model_path).expanduser()
        self.model: Any = None
        self.metadata = ProbeMetadata()
        # Try to auto-load a previously-trained model. Failure is fine.
        self._try_load()

    # ---------------------------------------------------------------- load
    def _try_load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with self.model_path.open("rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload.get("model")
                meta = payload.get("metadata")
                if isinstance(meta, dict):
                    known = {f for f in ProbeMetadata.__dataclass_fields__}
                    self.metadata = ProbeMetadata(
                        **{k: v for k, v in meta.items() if k in known}
                    )
        except Exception as exc:
            logger.warning("failed to load probe from %s: %s", self.model_path, exc)
            self.model = None

    # ---------------------------------------------------------------- train
    def train(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        min_samples: int = 200,
        max_budget: float = 10.0,
    ) -> ProbeMetadata:
        """Fit a calibrated GradientBoosting probe on trace rows.

        `rows` is typically `TraceCollector.get_training_data()`. When
        fewer than `min_samples` rows are supplied we refuse to train
        (returns a metadata row flagged uncalibrated — graph routing
        falls back to the "proceed" default).
        """
        rows_list = list(rows)
        n = len(rows_list)
        if n < int(min_samples):
            logger.warning(
                "probe.train: only %d rows (< %d min); refusing to train",
                n, min_samples,
            )
            self.metadata = ProbeMetadata(
                n_samples=n, min_samples_required=int(min_samples)
            )
            return self.metadata

        sk = _load_sklearn()
        if sk is None:
            logger.warning("sklearn not installed; probe.train is a no-op")
            self.metadata = ProbeMetadata(
                n_samples=n, min_samples_required=int(min_samples)
            )
            return self.metadata

        X, y = self._build_matrix(rows_list, max_budget=max_budget)
        feature_dim = len(X[0]) if X else 0

        base = sk["GradientBoostingClassifier"](
            n_estimators=_GBDT_N_ESTIMATORS,
            max_depth=_GBDT_MAX_DEPTH,
            learning_rate=_GBDT_LEARNING_RATE,
            random_state=0,
        )
        calibrated = True
        # Calibration needs enough samples per class — fall back to the
        # raw classifier if the minority class is too thin.
        min_class = min(y.count(0), y.count(1)) if y else 0
        if min_class >= _CALIBRATION_CV:
            clf = sk["CalibratedClassifierCV"](
                base, cv=_CALIBRATION_CV, method=_CALIBRATION_METHOD
            )
        else:
            logger.info(
                "too-thin minority class (%d); skipping calibration", min_class
            )
            clf = base
            calibrated = False

        clf.fit(X, y)
        self.model = clf
        self.metadata = ProbeMetadata(
            n_samples=n,
            n_positive=sum(1 for label in y if label == 1),
            calibrated=calibrated,
            min_samples_required=int(min_samples),
            feature_dim=feature_dim,
        )
        self._save()
        return self.metadata

    # ---------------------------------------------------------------- predict
    def predict_confidence(
        self,
        features: dict[str, Any] | StepFeatures | list[float],
        *,
        max_budget: float = 10.0,
    ) -> float:
        """Return P(build succeeds). 1.0 when no model is available."""
        if self.model is None:
            return 1.0
        try:
            vec = self._coerce_features(features, max_budget=max_budget)
            proba = self.model.predict_proba([vec])
        except Exception as exc:
            logger.warning("predict_confidence failed: %s", exc)
            return 1.0
        # predict_proba returns [[p(0), p(1)]]; we want P(build_passed=1)
        row = proba[0]
        if len(row) < 2:
            return float(row[0])
        return float(row[1])

    def should_escalate(self, confidence: float) -> str:
        """Spec thresholds: >0.8 proceed / >0.4 escalate / else abort."""
        c = float(confidence)
        if c > THRESH_PROCEED:
            return "proceed"
        if c > THRESH_ESCALATE:
            return "escalate"
        return "abort"

    # ---------------------------------------------------------------- evaluate
    def evaluate(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        max_budget: float = 10.0,
    ) -> dict[str, Any]:
        """Return accuracy / Brier / confusion matrix on a held-out set.

        When sklearn is missing or the probe isn't trained, returns a
        dict with `trained=False` and no scores.
        """
        if self.model is None:
            return {"trained": False, "reason": "probe_not_trained"}
        sk = _load_sklearn()
        if sk is None:
            return {"trained": True, "reason": "sklearn_missing"}
        X, y = self._build_matrix(list(rows), max_budget=max_budget)
        if not X:
            return {"trained": True, "reason": "no_data"}
        try:
            probs = self.model.predict_proba(X)[:, 1]
            preds = [1 if p >= 0.5 else 0 for p in probs]
            acc = float(sk["accuracy_score"](y, preds))
            brier = float(sk["brier_score_loss"](y, probs))
            cm = sk["confusion_matrix"](y, preds).tolist()
        except Exception as exc:
            return {"trained": True, "error": str(exc)}
        return {
            "trained": True,
            "n": len(y),
            "accuracy": acc,
            "brier": brier,
            "confusion_matrix": cm,
        }

    # ---------------------------------------------------------------- persist
    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "metadata": {
                "n_samples": self.metadata.n_samples,
                "n_positive": self.metadata.n_positive,
                "calibrated": self.metadata.calibrated,
                "min_samples_required": self.metadata.min_samples_required,
                "feature_dim": self.metadata.feature_dim,
            },
        }
        with self.model_path.open("wb") as f:
            pickle.dump(payload, f)

    # ---------------------------------------------------------------- internals
    def _build_matrix(
        self,
        rows: list[dict[str, Any]],
        *,
        max_budget: float,
    ) -> tuple[list[list[float]], list[int]]:
        X: list[list[float]] = []
        y: list[int] = []
        for r in rows:
            feats = extract_features(r, max_budget=max_budget)
            X.append(feats.to_vector())
            # bool -> int
            outcome = r.get("build_passed")
            if outcome is None:
                continue  # defensive: skip unlabeled rows
            y.append(1 if bool(outcome) else 0)
        # Align (drop any X rows without y — extract_features is pure so
        # this can only happen if an unlabeled row slipped through).
        X = X[: len(y)]
        return X, y

    def _coerce_features(
        self,
        features: dict[str, Any] | StepFeatures | list[float],
        *,
        max_budget: float,
    ) -> list[float]:
        if isinstance(features, list):
            return [float(x) for x in features]
        if isinstance(features, StepFeatures):
            return features.to_vector()
        return extract_features(features, max_budget=max_budget).to_vector()


# ---------------------------------------------------------------------------
# Module-level helpers — used by graph.py routing
# ---------------------------------------------------------------------------


_global_probe: Optional[ConfidenceProbe] = None


def get_default_probe() -> ConfidenceProbe:
    """Return a cached ConfidenceProbe pointed at the default path."""
    global _global_probe
    if _global_probe is None:
        _global_probe = ConfidenceProbe(DEFAULT_PROBE_PATH)
    return _global_probe


def set_default_probe(probe: Optional[ConfidenceProbe]) -> None:
    """Override the module-level cache — tests only."""
    global _global_probe
    _global_probe = probe


def is_probe_routing_enabled() -> bool:
    """Honor BELIEF_ENABLE_PROBE env var AND require a trained model.

    Default OFF (no model file -> no routing change). An operator who
    has trained a probe enables it by setting BELIEF_ENABLE_PROBE=1.
    """
    if os.environ.get("BELIEF_ENABLE_PROBE", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return False
    return get_default_probe().model is not None


__all__ = [
    "ConfidenceProbe",
    "DEFAULT_PROBE_PATH",
    "EXPECTED_TOTAL_STEPS",
    "ERROR_KEYWORDS",
    "KNOWN_AGENTS",
    "ProbeMetadata",
    "StepFeatures",
    "THRESH_ESCALATE",
    "THRESH_PROCEED",
    "extract_features",
    "get_default_probe",
    "is_probe_routing_enabled",
    "set_default_probe",
]
