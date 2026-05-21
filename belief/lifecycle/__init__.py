"""Lifecycle management (mycorrhizal Stage 7, Area 8).

Ecosystem-maturation organs — currently the succession-mode framework that
adjusts engine policy as the soil layer grows from sparse (pioneer) to dense
(mature).
"""

from belief.lifecycle.succession import (
    SuccessionManager,
    SuccessionMode,
    SuccessionPolicy,
)

__all__ = ["SuccessionManager", "SuccessionMode", "SuccessionPolicy"]
