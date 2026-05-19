"""Signal alphabet — the five-token vocabulary (mycorrhizal Stage 4, Area 1).

Plants under attack release methyl jasmonate (MeJA), methyl salicylate (MeSA),
ethylene, and short blends of volatile terpenes — a tiny chemical alphabet
that carries dozens of bits of meaning through *temporal sequencing* and
*ratio encoding* rather than per-event richness (Babikova et al. 2013;
Gilbert & Johnson 2017). The Belief Engine's signal vocabulary follows the
same design pattern: five token types, magnitude-scalar per emission, and
expressivity comes from sequences and joint distributions over a moving
time window rather than per-token richness.

Tokens (functionally specialized, deliberately tight):

* ``STRESS``   — agent under load / encountering difficulty
* ``DISCOVER`` — agent found a new pattern, primitive, or capability
* ``REQUEST``  — agent asking the engine for help
* ``OFFER``    — agent contributing validated output back
* ``WARN``     — agent observed a failure mode worth propagating

Composition is by token pairs / triples over a time window; the
``concentration`` and ``joint_concentration`` reads in
``belief.signal.store`` are how the protocol's expressivity surfaces.

Single-event semantics are intentionally weak — Cheong et al. 2011
(*Science* 334:354) found ~0.92 bits per NF-κB pulse in mammalian cells,
and Selimkhanov et al. 2014 demonstrated dynamics carry orders of magnitude
more. The Belief Engine's analogue: don't try to pack rich meaning into
``magnitude``; pack it into *sequences* the receiver integrates over time.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ── The closed alphabet ────────────────────────────────────────────────────

SignalToken = Literal["STRESS", "DISCOVER", "REQUEST", "OFFER", "WARN"]

#: Frozen tuple of the alphabet — useful for ``for t in SIGNAL_TOKENS`` and
#: for asserting the closed-set invariant in tests.
SIGNAL_TOKENS: tuple[SignalToken, ...] = (
    "STRESS",
    "DISCOVER",
    "REQUEST",
    "OFFER",
    "WARN",
)

# Maximum serialized size of the optional payload field. The five tokens
# already carry the protocol's intent; payload is for diagnostic context,
# not for stuffing whole prompts. Enforced on validate.
MAX_PAYLOAD_BYTES = 200


# ── The Signal model ───────────────────────────────────────────────────────


class Signal(BaseModel):
    """One emission from one agent into the soil's signal stream.

    Idempotency: emitters MUST set ``idempotency_key`` so dropped /
    replayed sends don't double-count concentration. The default
    derives a stable digest from (agent_id, token, timestamp,
    magnitude, payload) — adequate for honest senders but trivially
    collidable; production emitters should supply their own key.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    token: SignalToken
    magnitude: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: Optional[dict] = None
    idempotency_key: Optional[str] = None

    @field_validator("agent_id")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("agent_id must not be whitespace-only")
        return v

    @field_validator("payload")
    @classmethod
    def _payload_size_cap(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        # Strict serialization (no ``default=`` fallback) so the validator
        # actually enforces the documented "payload must be JSON-serializable"
        # contract — a ``default=str`` clause would silently stringify
        # arbitrary objects and the validator would lie about what it accepts.
        try:
            blob = json.dumps(v, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise ValueError(f"payload must be JSON-serializable: {e}") from e
        if len(blob.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload exceeds {MAX_PAYLOAD_BYTES}-byte cap (serialized: {len(blob)} bytes)"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def _require_tz_aware(cls, v: datetime) -> datetime:
        # Naive datetimes are ambiguous across systems; reject them
        # rather than silently assuming UTC.
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    def derived_idempotency_key(self) -> str:
        """Compute a stable digest of the signal's content.

        Used when the caller didn't supply ``idempotency_key``. SHA-256
        over a canonical serialization of (agent_id, token, magnitude,
        timestamp, payload) so two emitters that derive the same key for
        the same logical event get collapsed by the store.
        """
        canonical = json.dumps(
            {
                "a": self.agent_id,
                "t": self.token,
                "m": round(self.magnitude, 6),
                "ts": self.timestamp.isoformat(),
                "p": self.payload,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def effective_idempotency_key(self) -> str:
        return self.idempotency_key or self.derived_idempotency_key()
