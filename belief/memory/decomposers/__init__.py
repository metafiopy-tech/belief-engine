"""Three-tier decomposition (mycorrhizal Stage 7, Area 4).

Saprotrophic fungi break down dead biomass with enzyme suites matched to
substrate difficulty (Janusz et al. 2017): white-rot peroxidases for
recalcitrant lignin, brown-rot Fenton chemistry for cellulose, soft-rot for
the easy fraction. The architectural translation: *every* build — including
failures — should yield extractable value, with three tiers reflecting how
hard the substrate is to decompose.

* ``easy``        — the "cellulase" path. Local failures (typo, bad import,
  one broken function). AST-walk the code, harvest every node that parses
  cleanly as a reusable primitive.
* ``structural``  — the "hemicellulase" path. Integration failures (parts
  worked, composition didn't). Extract the import + call graph as candidate
  compositions with a failure annotation.
* ``recalcitrant``— the "laccase/peroxidase" path. Opaque/systemic failures
  (model loop, hallucinated API, external crash). Most is unrecoverable;
  what remains is negative evidence — a failure signature.

The dispatcher inspects a build outcome and routes to one or more tiers. A
build can be processed by several paths. This runs ALONGSIDE the existing
LLM decomposer (which keeps doing pattern/antipattern/skeleton/covenant
extraction) — it's additive, never a replacement, so existing behavior is
preserved.
"""

from belief.memory.decomposers.dispatcher import (
    DecompositionResult,
    decompose_failed_build,
)
from belief.memory.decomposers.easy import extract_clean_fragments
from belief.memory.decomposers.recalcitrant import extract_failure_signature
from belief.memory.decomposers.structural import extract_composition_edges

__all__ = [
    "DecompositionResult",
    "decompose_failed_build",
    "extract_clean_fragments",
    "extract_composition_edges",
    "extract_failure_signature",
]
