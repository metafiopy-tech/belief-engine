"""Module-level prompt constants for cross-domain synthesis (SE Session 3).

Mirrors the existing ``prompts.py`` convention: every prompt is a
module-level constant; callers format with ``.format(...)``. Auditable
and patch-friendly.

Four-pass design:

  - PASS 1 (FREEFORM): brainstorm cross-domain analogies between the
    user-submitted words. Free prose. Output is unstructured.
  - PASS 2 (PREDICATE-FORM FORCING): extract the deepest structural
    similarity as a predicate (name + arity + roles + Marr level)
    that holds for BOTH source and target with the same signature.
  - PASS 3 (ANTI-RATIONALIZATION): list at least 2 surface-level
    attributes that LOOK shared but were rejected as decorative.
  - PASS 4 (STRUCTURER): emit the full StructuralMechanism JSON.

Critic pass uses an INDEPENDENT context: it sees only the candidate
mechanism JSON, not the four-pass intermediate prose, so it can
critique without being primed by the synthesizer's chain-of-thought.
"""


# ---------------------------------------------------------------------------
# Pass 1 — free-form brainstorm
# ---------------------------------------------------------------------------

FREEFORM_PROMPT = """\
You are a cross-domain mechanism synthesizer. Two concepts are about
to be compared for structural similarity:

  source: {source}
  target: {target}

Brainstorm the relationship between them in free prose. Cover:

  - Surface attributes that LOOK similar (color, size, count, shape).
  - Deep mechanisms / processes that genuinely structurally match
    (e.g. "both pre-classify a signal at the transducer before
    downstream compute").
  - Near-misses: structurally similar systems where the analogy
    BREAKS at a specific slot.

Aim for ~200-400 words. Do not produce JSON. Do not hedge with
"perhaps" or "arguably" -- if the comparison is weak, say so plainly
and pick the strongest available structural claim.
"""


# ---------------------------------------------------------------------------
# Pass 2 — predicate-form forcing
# ---------------------------------------------------------------------------

PREDICATE_FORCING_PROMPT = """\
From your prior brainstorm, extract the SINGLE deepest structural
similarity as a typed predicate.

source: {source}
target: {target}

Prior brainstorm:
{freeform}

A predicate has:

  - name: snake_case verb-ish (e.g. ``pre_classify_signal``,
    ``broadcast_then_prune``, ``allocate_via_pheromone_decay``).
    DO NOT use attribute-style prefixes like ``has_``, ``is_``,
    ``contains_``, ``owns_``, ``lacks_`` -- those describe
    properties, not relations.
  - arity: positive integer, the number of argument slots.
  - roles: list of role names, length == arity. Each role names what
    occupies that slot in the relation (``transducer``, ``signal``,
    ``compressor``, ``coordinator``). Roles should be process-y, not
    descriptive (NOT ``red``, ``small``, ``many``).
  - marr_level: one of ``computational``, ``algorithmic``, or
    ``implementation``. The same predicate must hold at the SAME
    Marr level for source and target.

Return strict JSON:
{{"name": str,
  "arity": int,
  "roles": list[str],
  "marr_level": "computational"|"algorithmic"|"implementation",
  "rationale": str (<=80 words, why this predicate captures the
                    deepest structural match rather than a surface
                    attribute)}}
"""


# ---------------------------------------------------------------------------
# Pass 3 — anti-rationalization
# ---------------------------------------------------------------------------

ANTI_RATIONALIZATION_PROMPT = """\
You proposed the predicate ``{predicate_name}/{arity}`` for
{source} <-> {target}. Now list AT LEAST TWO surface-level
attributes you noticed but rejected as decorative.

These must be:

  - Things that LOOK shared between source and target.
  - But which fail to constitute a structural mechanism.
  - Typically descriptive properties (``has_many_X``,
    ``is_compact``, ``has_color_channels``, ``lacks_central_node``).

Return strict JSON:
{{"considered_and_rejected_attributes": list[str] (>=2 items)}}

Each entry is a short phrase, not a sentence.
"""


# ---------------------------------------------------------------------------
# Pass 4 — structurer (full StructuralMechanism JSON)
# ---------------------------------------------------------------------------

STRUCTURER_PROMPT = """\
Produce the full StructuralMechanism JSON for {source} <-> {target}
using the predicate, attributes, and reasoning developed across the
prior passes.

Predicate (forced in pass 2):
{predicate_json}

Considered-and-rejected attributes (forced in pass 3):
{rejected_attributes_json}

Brainstorm:
{freeform}

Output strict JSON with EXACTLY these top-level keys:

{{
  "mechanism_id":                      string, slug like "{source}-{target}-{{predicate_name}}",
  "source_domain":                     "{source}",
  "target_domain":                     "{target}",
  "predicate_in_source":               PredicateInstance (same as predicate above),
  "predicate_in_target":               PredicateInstance (IDENTICAL signature: same name, arity, roles, marr_level),
  "higher_order_relations":            list of objects, each with EXACTLY these two keys:
                                          "name":    str (snake_case identifier for THIS relation,
                                                     e.g. "reduces_downstream_compute" -- NOT
                                                     "relation_name", NOT "name_of_relation"),
                                          "relates": list[str] of >=2 distinct predicate names.
                                       MUST contain at least one entry whose ``relates`` includes
                                       the predicate name AND at least one OTHER distinct
                                       predicate name (e.g. "downstream_compute", "energy_cost",
                                       "signal_redundancy"),
  "near_miss":                         {{"description": str (concrete counterexample, 30-80 words),
                                          "breaks_at_argument": "predicate_in_(source|target).argument[N]"
                                                                where N is a real index in [0, arity-1]}},
  "considered_and_rejected_attributes":list[str] (>=2 entries from pass 3),
  "domain_evidence":                   list[DomainEvidence] -- list at least one
                                       {{"domain", "citation", "excerpt"}} per side. If no real
                                       evidence is at hand, use placeholder citations like
                                       "general_knowledge:cross_domain_synthesizer" so Session 4's
                                       retrieval layer can replace them. NEVER invent fake paper IDs
                                       that look real.
}}

The JSON must validate against the schema. In particular:

  - predicate_in_source and predicate_in_target MUST share name,
    arity, roles, AND marr_level.
  - higher_order_relations.relates MUST contain >=2 distinct entries.
  - near_miss.breaks_at_argument index MUST be < arity.

Return JSON only. No prose.
"""


# ---------------------------------------------------------------------------
# Critic — single-call CoVe over the candidate mechanism
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """\
You are an INDEPENDENT critic of a cross-domain structural-mechanism
claim. You see ONLY the candidate JSON below; you do not see the
synthesizer's reasoning or its earlier brainstorm. Critique it as if
the synthesizer might be wrong.

Candidate StructuralMechanism:
{mechanism_json}

Run the following 8 checks. For each, output ``passed: true`` only
if the check is clearly satisfied. ``reason`` should be at most one
sentence.

  1. Predicate is NOT attribute-style. The predicate name is verb-ish
     (``pre_classify_signal``, ``broadcast_then_prune``), not a property
     prefix (``has_``, ``is_``, ``contains_``, ``lacks_``).
  2. Roles are process-y, not descriptive. At least one role names a
     process participant (``transducer``, ``compressor``,
     ``coordinator``), not a static property (``red``, ``many``).
  3. Higher-order relation names describe processes or causation
     (``reduces_downstream_compute``, ``constrains``, ``enables``),
     not bare labels (``related_to``, ``is_a``).
  4. NearMiss plausibly fits the domains -- the counterexample
     references something coherent in source or target rather than a
     non-sequitur.
  5. NearMiss breaks_at_argument is a substantive failure point --
     the named slot is one where a real domain-system would actually
     fail, not a meaningless filler index.
  6. The considered_and_rejected_attributes are SURFACE-level
     properties (descriptive), not themselves relational mechanisms.
     If any rejected attribute reads like a real predicate, the
     synthesizer probably picked the wrong main predicate.
  7. The predicate transfers non-trivially to the target domain.
     The same name+arity+roles can be APPLIED in the target with a
     coherent meaning, not just by re-labeling.
  8. The analogy is non-trivial. It is not "both have parts" or
     "both involve information" -- it makes a specific claim about
     HOW one process is structurally like another.

Return strict JSON:
{{"verdict": "ACCEPT"|"REJECT",
  "checks": [
    {{"id": 1, "name": str, "passed": bool, "reason": str}},
    ...8 entries...
  ]}}

ACCEPT iff all 8 checks pass. Otherwise REJECT.
"""


# ---------------------------------------------------------------------------
# Few-shot examples (3 named pairs from the SE plan).
#
# These are spec-faithful exemplars used by the synthesizer at boot
# time. Each is a complete, valid StructuralMechanism. The format
# string includes them inline so the LLM sees concrete shape.
#
# Future tuning: add 5-7 more pairs and let DSPy/GEPA optimize the
# prompt weights against them. Session 5's atomization fan-out also
# benefits from a richer example bank.
# ---------------------------------------------------------------------------

FEW_SHOT_MANTIS_SHRIMP_CAMERA = """\
{{
  "mechanism_id": "mantis_shrimp-digital_camera-pre_classify_signal",
  "source_domain": "mantis_shrimp",
  "target_domain": "digital_camera",
  "predicate_in_source": {{
    "name": "pre_classify_signal",
    "arity": 2,
    "roles": ["transducer", "signal"],
    "marr_level": "algorithmic"
  }},
  "predicate_in_target": {{
    "name": "pre_classify_signal",
    "arity": 2,
    "roles": ["transducer", "signal"],
    "marr_level": "algorithmic"
  }},
  "higher_order_relations": [
    {{"name": "reduces_downstream_compute",
      "relates": ["pre_classify_signal", "downstream_compute"]}}
  ],
  "near_miss": {{
    "description": "A bee's compound eye also has many photoreceptors but signals are summed in the optic ganglia rather than pre-classified at the transducer.",
    "breaks_at_argument": "predicate_in_source.argument[0]"
  }},
  "considered_and_rejected_attributes": [
    "has_many_color_channels",
    "is_compact"
  ],
  "domain_evidence": [
    {{"domain": "biology",
      "citation": "general_knowledge:mantis_shrimp_color_vision",
      "excerpt": "16 photoreceptor classes pre-classify at the retina."}},
    {{"domain": "computing",
      "citation": "general_knowledge:image_sensor_pre_isp",
      "excerpt": "Per-pixel pre-classification reduces ISP throughput requirements."}}
  ]
}}
"""


FEW_SHOT_MYCORRHIZAL_CDN = """\
{{
  "mechanism_id": "mycorrhizal_network-cdn-allocate_via_demand_signal",
  "source_domain": "mycorrhizal_network",
  "target_domain": "content_delivery_network",
  "predicate_in_source": {{
    "name": "allocate_via_demand_signal",
    "arity": 3,
    "roles": ["allocator", "demand_signal", "resource"],
    "marr_level": "algorithmic"
  }},
  "predicate_in_target": {{
    "name": "allocate_via_demand_signal",
    "arity": 3,
    "roles": ["allocator", "demand_signal", "resource"],
    "marr_level": "algorithmic"
  }},
  "higher_order_relations": [
    {{"name": "shifts_resource_toward_high_demand",
      "relates": ["allocate_via_demand_signal", "high_demand_node"]}}
  ],
  "near_miss": {{
    "description": "A static round-robin DNS pool also distributes load across nodes, but it does so without a demand signal -- it cannot shift mass toward hot regions and fails when demand becomes skewed.",
    "breaks_at_argument": "predicate_in_target.argument[1]"
  }},
  "considered_and_rejected_attributes": [
    "has_many_nodes",
    "is_decentralized"
  ],
  "domain_evidence": [
    {{"domain": "biology",
      "citation": "general_knowledge:mycorrhizal_carbon_trade",
      "excerpt": "Fungi shift carbon allocation toward host trees signaling photosynthate demand."}},
    {{"domain": "computing",
      "citation": "general_knowledge:cdn_request_routing",
      "excerpt": "Edge caches receive content allocations weighted by request load."}}
  ]
}}
"""


FEW_SHOT_SLIME_MOLD_ROUTING = """\
{{
  "mechanism_id": "slime_mold-routing_table-prune_via_flux_decay",
  "source_domain": "slime_mold",
  "target_domain": "routing_table",
  "predicate_in_source": {{
    "name": "prune_via_flux_decay",
    "arity": 2,
    "roles": ["edge", "flux_history"],
    "marr_level": "algorithmic"
  }},
  "predicate_in_target": {{
    "name": "prune_via_flux_decay",
    "arity": 2,
    "roles": ["edge", "flux_history"],
    "marr_level": "algorithmic"
  }},
  "higher_order_relations": [
    {{"name": "concentrates_throughput_on_used_paths",
      "relates": ["prune_via_flux_decay", "throughput"]}}
  ],
  "near_miss": {{
    "description": "Random-walk routers also explore many edges but do not retain or decay flux history, so they revisit pruned dead-ends and never converge on shortest paths.",
    "breaks_at_argument": "predicate_in_source.argument[1]"
  }},
  "considered_and_rejected_attributes": [
    "has_branching_topology",
    "is_self_organizing"
  ],
  "domain_evidence": [
    {{"domain": "biology",
      "citation": "general_knowledge:tero_slime_mold_routing",
      "excerpt": "Physarum thickens tubes carrying high flux and prunes idle ones."}},
    {{"domain": "computing",
      "citation": "general_knowledge:routing_protocol_decay",
      "excerpt": "Adaptive routing protocols penalize edges with stale or zero flux."}}
  ]
}}
"""


FEW_SHOT_LIBRARY = (
    FEW_SHOT_MANTIS_SHRIMP_CAMERA
    + "\n---\n"
    + FEW_SHOT_MYCORRHIZAL_CDN
    + "\n---\n"
    + FEW_SHOT_SLIME_MOLD_ROUTING
)


__all__ = [
    "ANTI_RATIONALIZATION_PROMPT",
    "CRITIC_PROMPT",
    "FEW_SHOT_LIBRARY",
    "FEW_SHOT_MANTIS_SHRIMP_CAMERA",
    "FEW_SHOT_MYCORRHIZAL_CDN",
    "FEW_SHOT_SLIME_MOLD_ROUTING",
    "FREEFORM_PROMPT",
    "PREDICATE_FORCING_PROMPT",
    "STRUCTURER_PROMPT",
]
