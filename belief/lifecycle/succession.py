"""Succession modes (mycorrhizal Stage 7, Area 8).

Mycorrhizal communities turn over as an ecosystem develops: pioneer ruderals
dominate disturbed sites (high growth, low specificity), mid-successional
generalists build biomass, late-successional specialists invest in
recalcitrant biomass that builds long-term soil capital (Martínez-García et
al. 2015; Clemmensen et al. 2015). The architectural translation: the right
policy at 10 builds is different from the right policy at 100,000.

Three modes, transitioned by *metrics* not raw time:

* ``PIONEER``  (soil < 1,000 nutrients): lenient onboarding, gentle
  sanctions, aggressive decomposition (extract everything), generous
  retention. Goal: build up biomass + diversity.
* ``MID``      (1,000–10,000, OR hub set not yet stable): full reciprocity
  ledger active, sanctions on, FSRS pruning begins, hubs come online.
* ``MATURE``   (≥10,000 AND hub set stable AND exchange-rate variance low):
  strict onboarding, firm sanctions, selective decomposition, consolidation
  pipeline active (export stable nutrients to a versioned "humus" artifact).

Other subsystems *query* the manager for mode-dependent policy at decision
points rather than hardcoding values. To keep the existing test gate green,
the onboarding gate + sanctions engine are NOT forced to consult the manager
this stage — they accept an optional manager and fall back to their own
defaults when none is supplied (consumers-deferred, like Stages 4-6).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("belief.lifecycle.succession")

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_STATE_PATH = _BELIEF_HOME / "succession_state.json"

# Mode-transition thresholds (nutrient counts).
PIONEER_CEILING = 1_000
MATURE_FLOOR = 10_000
# Exchange-rate variance below which the agent population is "converged"
# (the alpha/beta-diversity convergence finding from Martínez-García 2015).
DEFAULT_VARIANCE_CONVERGENCE = 0.05


class SuccessionMode(str, Enum):
    PIONEER = "pioneer"
    MID = "mid"
    MATURE = "mature"


@dataclass(frozen=True)
class SuccessionPolicy:
    """Mode-dependent policy knobs. Subsystems read these instead of
    hardcoding. Values shift from permissive (pioneer) to strict (mature)."""

    mode: SuccessionMode
    onboarding_min_value: float  # demo-task value bar
    sanction_strength: float  # multiplier on sanction thresholds (higher = firmer)
    fsrs_decay_multiplier: float  # 1.0 = standard; <1 slower retention
    decomposition_aggressiveness: float  # 1.0 = extract everything; <1 selective
    consolidation_active: bool  # mature-mode humus export

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "onboarding_min_value": self.onboarding_min_value,
            "sanction_strength": self.sanction_strength,
            "fsrs_decay_multiplier": self.fsrs_decay_multiplier,
            "decomposition_aggressiveness": self.decomposition_aggressiveness,
            "consolidation_active": self.consolidation_active,
        }


_POLICIES: dict[SuccessionMode, SuccessionPolicy] = {
    SuccessionMode.PIONEER: SuccessionPolicy(
        mode=SuccessionMode.PIONEER,
        onboarding_min_value=0.0,  # any validated output admits
        sanction_strength=0.5,  # gentle
        fsrs_decay_multiplier=0.5,  # slow decay → generous retention
        decomposition_aggressiveness=1.0,  # extract everything
        consolidation_active=False,
    ),
    SuccessionMode.MID: SuccessionPolicy(
        mode=SuccessionMode.MID,
        onboarding_min_value=1.0,
        sanction_strength=1.0,  # standard
        fsrs_decay_multiplier=1.0,
        decomposition_aggressiveness=0.75,
        consolidation_active=False,
    ),
    SuccessionMode.MATURE: SuccessionPolicy(
        mode=SuccessionMode.MATURE,
        onboarding_min_value=2.0,  # higher bar
        sanction_strength=1.5,  # firm
        fsrs_decay_multiplier=1.0,
        decomposition_aggressiveness=0.4,  # selective — only high-quality frags
        consolidation_active=True,
    ),
}


def policy_for(mode: SuccessionMode) -> SuccessionPolicy:
    return _POLICIES[mode]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SuccessionStatus:
    mode: SuccessionMode
    nutrient_count: int
    hub_count: int
    hub_stable: bool
    exchange_variance: Optional[float]
    computed_at: datetime


class SuccessionManager:
    """Computes the current succession mode from soil + topology metrics,
    persists transitions, and serves mode-dependent policy.

    Dependencies are injected and all optional so the manager degrades
    gracefully: missing soil → count 0 (PIONEER); missing hub registry →
    hub_stable defaults False (keeps the system out of MATURE until the
    real signal exists)."""

    def __init__(
        self,
        soil=None,
        hub_registry=None,
        reciprocity_ledger=None,
        state_path: Path = _DEFAULT_STATE_PATH,
        variance_convergence: float = DEFAULT_VARIANCE_CONVERGENCE,
    ) -> None:
        self._soil = soil
        self._hubs = hub_registry
        self._ledger = reciprocity_ledger
        self.state_path = Path(state_path).expanduser()
        self.variance_convergence = float(variance_convergence)

    # ── Metric collection ───────────────────────────────────────────────

    def _nutrient_count(self) -> int:
        if self._soil is None:
            return 0
        try:
            return int(self._soil.count())
        except Exception:  # pragma: no cover — soil optional
            return 0

    def _hub_metrics(self) -> tuple[int, bool]:
        """Return (hub_count, hub_stable). Without a hub registry, hubs are
        treated as not-yet-stable so the system can't reach MATURE on hub
        grounds alone."""
        if self._hubs is None:
            return (0, False)
        try:
            hubs = self._hubs.current_hubs()
        except Exception:  # pragma: no cover
            return (0, False)
        # "Stable" here means: at least one hub exists. A richer stability
        # signal (no churn over 30d) belongs to a later session; documented.
        return (len(hubs), len(hubs) > 0)

    def _exchange_variance(self) -> Optional[float]:
        if self._ledger is None:
            return None
        try:
            rows = self._ledger.rank_agents(window="30d")
        except Exception:  # pragma: no cover
            return None
        rates = [r.exchange_rate for r in rows]
        if len(rates) < 2:
            return None
        mean = sum(rates) / len(rates)
        return sum((x - mean) ** 2 for x in rates) / len(rates)

    # ── Mode computation ────────────────────────────────────────────────

    def compute_mode(self) -> SuccessionStatus:
        n = self._nutrient_count()
        hub_count, hub_stable = self._hub_metrics()
        variance = self._exchange_variance()

        if n < PIONEER_CEILING:
            mode = SuccessionMode.PIONEER
        elif (
            n >= MATURE_FLOOR
            and hub_stable
            and (variance is not None and variance < self.variance_convergence)
        ):
            mode = SuccessionMode.MATURE
        else:
            mode = SuccessionMode.MID

        return SuccessionStatus(
            mode=mode,
            nutrient_count=n,
            hub_count=hub_count,
            hub_stable=hub_stable,
            exchange_variance=variance,
            computed_at=_utcnow(),
        )

    def current_mode(self) -> SuccessionMode:
        return self.compute_mode().mode

    def current_policy(self) -> SuccessionPolicy:
        return policy_for(self.current_mode())

    # ── Transition logging ──────────────────────────────────────────────

    def recompute_and_log(self) -> SuccessionStatus:
        """Recompute the mode and log a transition if it changed from the
        last persisted mode. Returns the new status."""
        status = self.compute_mode()
        prev = self._load_last_mode()
        if prev != status.mode.value:
            logger.info(
                "Succession transition: %s → %s (nutrients=%d, hubs=%d, var=%s)",
                prev,
                status.mode.value,
                status.nutrient_count,
                status.hub_count,
                status.exchange_variance,
            )
            self._save_mode(status)
        return status

    def _load_last_mode(self) -> Optional[str]:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f).get("mode")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _save_mode(self, status: SuccessionStatus) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": status.mode.value,
                        "nutrient_count": status.nutrient_count,
                        "computed_at": status.computed_at.isoformat(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:  # pragma: no cover — best-effort
            logger.debug("succession state save skipped: %s", e)


# ── Mature-mode consolidation ───────────────────────────────────────────────


def consolidate_humus(
    soil,
    out_path: Path,
    min_fsrs_stability: float = 30.0,
    limit: int = 1000,
) -> dict:
    """Export stable, high-FSRS-strength nutrients into a versioned "humus"
    artifact — a self-contained JSON collection publishable independently of
    the engine. Mature-mode only (the caller checks the policy flag).

    Best-effort over the soil's tool/principle collections. Returns a
    manifest dict. Does not delete anything from soil — consolidation is
    additive, like soil organic-matter accumulation."""
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported: list[dict] = []
    try:
        # Pull active tools as the canonical "stable capital" to export.
        from belief.memory.tool_registry import ToolRegistry

        registry = ToolRegistry(soil)
        for tool in registry.get_active_tools():
            if tool.fsrs_stability >= min_fsrs_stability:
                exported.append(
                    {
                        "id": tool.id,
                        "name": tool.name,
                        "description": tool.description,
                        "fsrs_stability": tool.fsrs_stability,
                        "use_count": tool.use_count,
                    }
                )
            if len(exported) >= limit:
                break
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug("humus consolidation export skipped: %s", e)

    manifest = {
        "humus_version": _utcnow().strftime("%Y-%m-%dT%H-%M-%SZ"),
        "min_fsrs_stability": min_fsrs_stability,
        "exported_count": len(exported),
        "items": exported,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ── CLI ──────────────────────────────────────────────────────────────────


def cli_show() -> str:
    """`belief succession` — show current mode + policy. Lazily wires soil +
    hub registry + ledger from their defaults."""
    soil = None
    hubs = None
    ledger = None
    try:
        from belief.memory.soil import Soil

        soil = Soil(Path("~/.belief-engine/soil").expanduser())
    except Exception:  # pragma: no cover
        pass
    try:
        from belief.memory.reciprocity import get_default_ledger
        from belief.routing._store import RoutingStore
        from belief.routing.hubs import HubRegistry

        ledger = get_default_ledger()
        hubs = HubRegistry(RoutingStore(), ledger)
    except Exception:  # pragma: no cover
        pass

    mgr = SuccessionManager(soil=soil, hub_registry=hubs, reciprocity_ledger=ledger)
    status = mgr.compute_mode()
    policy = policy_for(status.mode)
    var_str = (
        f"{status.exchange_variance:.4f}"
        if status.exchange_variance is not None
        else "n/a (need ≥2 agents)"
    )
    return (
        f"Succession mode: {status.mode.value.upper()}\n"
        f"  nutrient count:       {status.nutrient_count}\n"
        f"  hub count / stable:   {status.hub_count} / {status.hub_stable}\n"
        f"  exchange variance:    {var_str}\n"
        f"  thresholds:           PIONEER < {PIONEER_CEILING} ≤ MID < "
        f"{MATURE_FLOOR} ≤ MATURE\n"
        f"  policy:\n"
        f"    onboarding bar:     {policy.onboarding_min_value}\n"
        f"    sanction strength:  {policy.sanction_strength}\n"
        f"    decomposition aggr: {policy.decomposition_aggressiveness}\n"
        f"    consolidation:      {policy.consolidation_active}"
    )
