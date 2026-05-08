"""AskNature biomimicry taxonomy reference (SE Session 4).

The AskNature project (Biomimicry Institute) classifies biological
strategies into a 3-level hierarchy:

  - 8 GROUPS at the top (broad functional categories)
  - 30 SUBGROUPS in the middle
  - ~160 FUNCTIONS at the leaves

The Synthesis Engine uses these tags as taxonomy slots on
``StructuralMechanism`` instances stored in the bio-primitives store.
A validator rejects unknown tags so prompts/few-shots can't drift the
vocabulary by typo.

This module ships a *representative* slice of the leaf-level functions
across all 8 groups -- enough to validate the structure and exercise
the test suite. Future tuning (Session 5+) can extend the FUNCTIONS
tuple as more cross-domain mechanisms surface real biology grounding;
the validator's allow-list grows alongside.

Source: https://asknature.org/ (functional groupings reproduced for
reference; the function list is a curation, not an exhaustive copy).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Top-level groups -- these 8 are stable across AskNature releases.
# ---------------------------------------------------------------------------

GROUPS: tuple[str, ...] = (
    "get_or_store_resources",
    "move_or_stay_put",
    "maintain_physical_integrity",
    "modify",
    "make",
    "process_information",
    "attach_or_detach",
    "adapt_or_survive",
)


# ---------------------------------------------------------------------------
# Subgroups -- 30 mid-level functional categories.
# ---------------------------------------------------------------------------

SUBGROUPS: dict[str, tuple[str, ...]] = {
    "get_or_store_resources": (
        "capture",
        "absorb",
        "store",
        "manage_disturbance",
    ),
    "move_or_stay_put": (
        "actively_move",
        "passively_move",
        "stay_put",
    ),
    "maintain_physical_integrity": (
        "manage_structural_forces",
        "manage_wear",
        "manage_temperature",
    ),
    "modify": (
        "physically_modify",
        "chemically_modify",
        "modify_state",
    ),
    "make": (
        "make_or_assemble",
        "transform_or_use_energy",
        "self_replicate",
        "self_repair",
    ),
    "process_information": (
        "sense_signals",
        "process_signals",
        "send_signals",
        "store_or_retrieve_information",
    ),
    "attach_or_detach": (
        "attach",
        "detach",
        "permit_movement",
        "prevent_movement",
    ),
    "adapt_or_survive": (
        "adapt_to_change",
        "cooperate_or_compete",
        "coordinate",
        "modify_behaviorally",
        "build_resilience",
    ),
}


# ---------------------------------------------------------------------------
# Leaf-level functions -- representative slice. Each function string is
# valid as a taxonomy tag on a StructuralMechanism. The set is the
# allow-list the validator consults.
# ---------------------------------------------------------------------------

FUNCTIONS: tuple[str, ...] = (
    # capture
    "capture_chemical_entities",
    "capture_kinetic_energy",
    "capture_radiant_energy",
    "capture_thermal_energy",
    # absorb
    "absorb_chemical_entities",
    "absorb_radiant_energy",
    "absorb_solid_objects",
    "absorb_thermal_energy",
    # store
    "store_chemical_entities",
    "store_kinetic_energy",
    "store_information",
    "store_solid_objects",
    "store_thermal_energy",
    # manage_disturbance
    "buffer_against_disturbance",
    "manage_compression",
    "manage_impact",
    # actively_move
    "fly_or_glide",
    "swim_or_move_in_water",
    "move_on_or_through_solids",
    "move_through_air",
    "walk_or_climb",
    # passively_move
    "be_carried_by_wind",
    "be_carried_by_water",
    # stay_put
    "anchor",
    "attach_temporarily",
    "resist_passive_movement",
    # manage_structural_forces
    "manage_compression",
    "manage_tension",
    "manage_shear",
    "manage_torsion",
    # manage_wear
    "resist_abrasion",
    "self_clean",
    # manage_temperature
    "regulate_temperature",
    "shed_heat",
    "retain_heat",
    "tolerate_temperature_extremes",
    # physically_modify
    "physically_break_down",
    "physically_assemble",
    "shape",
    "physically_separate",
    # chemically_modify
    "chemically_break_down",
    "chemically_assemble",
    "chemically_transform",
    # modify_state
    "change_phase",
    "change_color",
    "change_size_or_shape",
    # make_or_assemble
    "self_assemble",
    "assemble_with_external_aid",
    "fabricate_with_minimal_energy",
    # transform_or_use_energy
    "convert_chemical_to_kinetic",
    "convert_chemical_to_thermal",
    "convert_radiant_to_chemical",
    "convert_radiant_to_kinetic",
    # self_replicate
    "self_replicate",
    "reproduce_sexually",
    "reproduce_asexually",
    # self_repair
    "self_heal",
    "regenerate_tissue",
    # sense_signals
    "sense_chemical_signals",
    "sense_electrical_signals",
    "sense_kinetic_signals",
    "sense_optical_signals",
    "sense_thermal_signals",
    # process_signals
    "filter_signals",
    "amplify_signals",
    "compare_signals",
    "transduce_signals",
    "pre_classify_signals",
    "compress_signals",
    # send_signals
    "send_chemical_signals",
    "send_electrical_signals",
    "send_optical_signals",
    "send_acoustic_signals",
    # store_or_retrieve_information
    "store_information",
    "retrieve_information",
    "share_information",
    # attach
    "permanently_attach",
    "reversibly_attach",
    "self_assemble_attachment",
    # detach
    "actively_detach",
    "passively_detach",
    # permit_movement
    "permit_axial_motion",
    "permit_rotation",
    "permit_translation",
    # prevent_movement
    "lock",
    "stiffen",
    # adapt_to_change
    "adapt_phenotype",
    "adapt_behavior",
    "adapt_population",
    # cooperate_or_compete
    "cooperate_within_species",
    "cooperate_across_species",
    "compete_for_resources",
    # coordinate
    "coordinate_through_chemicals",
    "coordinate_through_signals",
    "allocate_via_demand_signal",
    # modify_behaviorally
    "navigate",
    "communicate",
    "learn",
    # build_resilience
    "diversify",
    "redundancy",
    "modularity",
    "prune_via_flux_decay",
)


_FUNCTION_SET: frozenset[str] = frozenset(FUNCTIONS)
_GROUP_SET: frozenset[str] = frozenset(GROUPS)
_SUBGROUP_SET: frozenset[str] = frozenset(sub for subs in SUBGROUPS.values() for sub in subs)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_function(tag: str) -> bool:
    """Return True iff ``tag`` is a known leaf-level function name."""
    return tag in _FUNCTION_SET


def is_valid_group(tag: str) -> bool:
    return tag in _GROUP_SET


def is_valid_subgroup(tag: str) -> bool:
    return tag in _SUBGROUP_SET


def validate_tags(tags: list[str]) -> None:
    """Raise ``ValueError`` if any tag is unknown.

    Tags may reference any level of the hierarchy (group, subgroup, or
    function). Mixed-level tagging is allowed -- for instance a
    mechanism might be tagged both ``process_information`` (group) and
    ``pre_classify_signals`` (function leaf) when the leaf is the
    primary fit but the broader group also applies.
    """
    if not isinstance(tags, list):
        raise TypeError(f"tags must be a list of strings, got {type(tags).__name__}")
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError(f"each tag must be a string, got {type(tag).__name__}")
        if tag in _FUNCTION_SET or tag in _SUBGROUP_SET or tag in _GROUP_SET:
            continue
        raise ValueError(
            f"unknown AskNature taxonomy tag: {tag!r}. Tags must be one of "
            f"the {len(_GROUP_SET)} groups, {len(_SUBGROUP_SET)} subgroups, "
            f"or {len(_FUNCTION_SET)} functions."
        )


__all__ = [
    "FUNCTIONS",
    "GROUPS",
    "SUBGROUPS",
    "is_valid_function",
    "is_valid_group",
    "is_valid_subgroup",
    "validate_tags",
]
