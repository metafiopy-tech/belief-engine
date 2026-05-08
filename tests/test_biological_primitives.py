"""Tests for the BiologicalPrimitiveStore + AskNature taxonomy (SE S4)."""

from __future__ import annotations

import json

import pytest

from belief.memory.asknature_taxonomy import (
    FUNCTIONS,
    GROUPS,
    SUBGROUPS,
    is_valid_function,
    is_valid_group,
    is_valid_subgroup,
    validate_tags,
)
from belief.memory.biological_primitives import (
    COLLECTION_NAME,
    BiologicalPrimitiveStore,
    NeighborMechanism,
)
from belief.photosynthesis.synthesis import prompts_cross_domain as prompts
from belief.photosynthesis.synthesis.structural_mechanism import StructuralMechanism


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _few_shot(name: str) -> StructuralMechanism:
    """Resolve a named few-shot template into a validated mechanism."""
    txt = getattr(prompts, name).replace("{{", "{").replace("}}", "}")
    return StructuralMechanism.model_validate(json.loads(txt))


@pytest.fixture
def fresh_store(tmp_path) -> BiologicalPrimitiveStore:
    """A truly isolated store per test.

    chromadb's EphemeralClient shares state across instances in the
    same process, so two tests that both construct
    ``BiologicalPrimitiveStore()`` would see each other's writes. We
    pin each test to its own tmp_path so the PersistentClient gives
    us real isolation.
    """
    return BiologicalPrimitiveStore(persist_dir=tmp_path / "bio_store")


@pytest.fixture
def mantis_camera() -> StructuralMechanism:
    return _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")


@pytest.fixture
def mycorrhizal_cdn() -> StructuralMechanism:
    return _few_shot("FEW_SHOT_MYCORRHIZAL_CDN")


@pytest.fixture
def slime_routing() -> StructuralMechanism:
    return _few_shot("FEW_SHOT_SLIME_MOLD_ROUTING")


# ---------------------------------------------------------------------------
# AskNature taxonomy
# ---------------------------------------------------------------------------


class TestAskNatureTaxonomy:
    def test_eight_top_groups(self) -> None:
        assert len(GROUPS) == 8

    def test_thirty_subgroups_total(self) -> None:
        flat = [s for subs in SUBGROUPS.values() for s in subs]
        assert len(flat) == 30

    def test_every_group_has_subgroups(self) -> None:
        assert set(SUBGROUPS.keys()) == set(GROUPS)

    def test_function_count_in_session4_target(self) -> None:
        # SE plan target is ~160; representative slice is currently 102.
        # Assert at least 100 so future curation can grow without
        # requiring an exact number.
        assert len(FUNCTIONS) >= 100

    def test_validate_accepts_groups_subgroups_functions_mixed(self) -> None:
        validate_tags(["process_information", "sense_signals", "pre_classify_signals"])

    def test_validate_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown AskNature taxonomy tag"):
            validate_tags(["definitely_not_a_real_tag"])

    def test_validate_rejects_partial_match(self) -> None:
        # Substring matches should NOT pass -- the tag must be exact.
        with pytest.raises(ValueError):
            validate_tags(["pre_classify"])  # function is pre_classify_signals

    def test_validate_type_error_on_non_list(self) -> None:
        with pytest.raises(TypeError):
            validate_tags("process_information")  # type: ignore[arg-type]

    def test_validate_type_error_on_non_string_tag(self) -> None:
        with pytest.raises(TypeError):
            validate_tags([42])  # type: ignore[list-item]

    def test_predicate_helpers(self) -> None:
        assert is_valid_group("process_information")
        assert is_valid_subgroup("sense_signals")
        assert is_valid_function("pre_classify_signals")
        assert not is_valid_function("nonsense")

    def test_validate_accepts_empty_list(self) -> None:
        # No tags is a valid state -- mechanisms can be untagged.
        validate_tags([])


# ---------------------------------------------------------------------------
# BiologicalPrimitiveStore -- core operations
# ---------------------------------------------------------------------------


class TestStoreCore:
    def test_initial_count_is_zero(self, fresh_store) -> None:
        assert fresh_store.count() == 0

    def test_collection_name_pinned(self) -> None:
        # Don't drift from the 5-collection naming convention.
        assert COLLECTION_NAME == "belief_biological_primitives"

    def test_add_returns_mechanism_id_when_present(self, fresh_store, mantis_camera) -> None:
        doc_id = fresh_store.add(mantis_camera)
        assert doc_id == mantis_camera.mechanism_id

    def test_add_validates_taxonomy_tags(self, fresh_store, mantis_camera) -> None:
        with pytest.raises(ValueError, match="unknown AskNature taxonomy tag"):
            fresh_store.add(mantis_camera, taxonomy_tags=["totally_made_up"])

    def test_add_with_valid_tags_succeeds(self, fresh_store, mantis_camera) -> None:
        fresh_store.add(
            mantis_camera,
            taxonomy_tags=["process_information", "pre_classify_signals"],
        )
        assert fresh_store.count() == 1

    def test_add_rejects_unknown_tags_atomically(self, fresh_store, mantis_camera) -> None:
        """Validation runs BEFORE the chromadb upsert -- bad tags must
        not leave a half-written record."""
        with pytest.raises(ValueError):
            fresh_store.add(mantis_camera, taxonomy_tags=["bad_tag"])
        assert fresh_store.count() == 0

    def test_add_two_mechanisms(self, fresh_store, mantis_camera, mycorrhizal_cdn) -> None:
        fresh_store.add(mantis_camera)
        fresh_store.add(mycorrhizal_cdn)
        assert fresh_store.count() == 2


# ---------------------------------------------------------------------------
# Novelty score
# ---------------------------------------------------------------------------


class TestNovelty:
    def test_empty_store_returns_one(self, fresh_store, mantis_camera) -> None:
        assert fresh_store.novelty_score(mantis_camera) == 1.0

    def test_exact_match_returns_zero(self, fresh_store, mantis_camera) -> None:
        fresh_store.add(mantis_camera)
        score = fresh_store.novelty_score(mantis_camera)
        # Hash-embedder may not produce perfect zero distance, but it
        # should be near zero for an identical document.
        assert score < 0.01

    def test_different_mechanism_has_nonzero_novelty(
        self, fresh_store, mantis_camera, mycorrhizal_cdn
    ) -> None:
        fresh_store.add(mantis_camera)
        score = fresh_store.novelty_score(mycorrhizal_cdn)
        assert 0.0 < score <= 1.0

    def test_novelty_in_unit_interval(
        self, fresh_store, mantis_camera, mycorrhizal_cdn, slime_routing
    ) -> None:
        fresh_store.add(mantis_camera)
        fresh_store.add(mycorrhizal_cdn)
        for m in (mantis_camera, mycorrhizal_cdn, slime_routing):
            score = fresh_store.novelty_score(m)
            assert 0.0 <= score <= 1.0

    def test_repeated_add_doesnt_change_novelty(self, fresh_store, mantis_camera) -> None:
        """Adding the same mechanism twice (upsert) doesn't change novelty."""
        fresh_store.add(mantis_camera)
        s1 = fresh_store.novelty_score(mantis_camera)
        fresh_store.add(mantis_camera)
        s2 = fresh_store.novelty_score(mantis_camera)
        assert abs(s1 - s2) < 1e-6
        assert fresh_store.count() == 1  # upsert, not double-insert


# ---------------------------------------------------------------------------
# query_nearest
# ---------------------------------------------------------------------------


class TestQueryNearest:
    def test_empty_store_returns_empty_list(self, fresh_store) -> None:
        assert fresh_store.query_nearest("anything", top_k=5) == []

    def test_zero_top_k_returns_empty(self, fresh_store, mantis_camera) -> None:
        fresh_store.add(mantis_camera)
        assert fresh_store.query_nearest("anything", top_k=0) == []

    def test_returns_neighbor_mechanism_objects(self, fresh_store, mantis_camera) -> None:
        fresh_store.add(mantis_camera, taxonomy_tags=["process_information"])
        results = fresh_store.query_nearest("mantis pre_classify", top_k=1)
        assert len(results) == 1
        n = results[0]
        assert isinstance(n, NeighborMechanism)
        assert n.mechanism.predicate_in_source.name == "pre_classify_signal"
        assert "process_information" in n.taxonomy_tags

    def test_results_ordered_by_weighted_score(
        self, fresh_store, mantis_camera, mycorrhizal_cdn, slime_routing
    ) -> None:
        for m in (mantis_camera, mycorrhizal_cdn, slime_routing):
            fresh_store.add(m)
        results = fresh_store.query_nearest("mantis_shrimp pre_classify camera", top_k=3)
        scores = [n.weighted_score for n in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_carries_fsrs_fields(self, fresh_store, mantis_camera) -> None:
        fresh_store.add(mantis_camera)
        results = fresh_store.query_nearest("mantis_shrimp", top_k=1)
        assert len(results) == 1
        # Fresh records have last_review=0 -> elapsed_days=0 ->
        # retrievability=1.0 by the FSRS power-law.
        assert results[0].fsrs_retrievability == pytest.approx(1.0, abs=0.01)

    def test_top_k_caps_results(
        self, fresh_store, mantis_camera, mycorrhizal_cdn, slime_routing
    ) -> None:
        fresh_store.add(mantis_camera)
        fresh_store.add(mycorrhizal_cdn)
        fresh_store.add(slime_routing)
        results = fresh_store.query_nearest("anything", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Round-trip integrity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_query_reconstructs_full_mechanism(self, fresh_store, mantis_camera) -> None:
        fresh_store.add(mantis_camera)
        results = fresh_store.query_nearest("mantis", top_k=1)
        rt = results[0].mechanism
        assert rt == mantis_camera

    def test_taxonomy_tags_round_trip(self, fresh_store, mantis_camera) -> None:
        tags = ["process_information", "pre_classify_signals", "sense_signals"]
        fresh_store.add(mantis_camera, taxonomy_tags=tags)
        results = fresh_store.query_nearest("mantis", top_k=1)
        assert set(results[0].taxonomy_tags) == set(tags)


# ---------------------------------------------------------------------------
# Cross-domain generator integration -- bio_store priming
# ---------------------------------------------------------------------------


class TestCrossDomainPriming:
    def test_synthesize_with_bio_store_adds_after_acceptance(
        self, fresh_store, mantis_camera
    ) -> None:
        """When synthesize_cross_domain produces an accepted mechanism
        AND a bio_store is provided, the mechanism is added to the
        store."""
        import asyncio

        from belief.photosynthesis.synthesis.cross_domain_generator import (
            synthesize_cross_domain,
        )

        # Build a four-pass response anchored to mantis_camera
        mech_json = mantis_camera.model_dump_json()
        responses = [
            "freeform brainstorm",
            json.dumps(
                {
                    "name": mantis_camera.predicate_in_source.name,
                    "arity": mantis_camera.predicate_in_source.arity,
                    "roles": list(mantis_camera.predicate_in_source.roles),
                    "marr_level": mantis_camera.predicate_in_source.marr_level,
                    "rationale": "x",
                }
            ),
            json.dumps(
                {
                    "considered_and_rejected_attributes": list(
                        mantis_camera.considered_and_rejected_attributes
                    )
                }
            ),
            mech_json,
        ]
        i = [0]

        async def gen(prompt, *, temperature, max_tokens):
            r = responses[i[0]]
            i[0] += 1
            return r

        async def critic(prompt, *, temperature, max_tokens):
            return json.dumps(
                {
                    "verdict": "ACCEPT",
                    "checks": [
                        {"id": cid, "name": f"c{cid}", "passed": True, "reason": "ok"}
                        for cid in range(1, 9)
                    ],
                }
            )

        assert fresh_store.count() == 0
        result = asyncio.run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                critic_client=critic,
                bio_store=fresh_store,
            )
        )
        assert result.spec is not None
        assert fresh_store.count() == 1

    def test_freeform_prompt_includes_primer_when_bio_store_has_neighbors(
        self, fresh_store, mantis_camera, mycorrhizal_cdn
    ) -> None:
        """The freeform pass must receive a 'PRIOR MECHANISMS' block
        when the bio_store has neighbors near the query."""
        import asyncio

        from belief.photosynthesis.synthesis.cross_domain_generator import (
            synthesize_cross_domain,
        )

        # Pre-populate the store
        fresh_store.add(mantis_camera, taxonomy_tags=["process_information"])

        captured_prompts: list[str] = []

        async def gen(prompt, *, temperature, max_tokens):
            captured_prompts.append(prompt)
            # Return a malformed pass-2 response to short-circuit early
            # (we only care about pass 1's prompt content here).
            if len(captured_prompts) == 1:
                return "freeform"
            return "not json"

        result = asyncio.run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                bio_store=fresh_store,
            )
        )
        # Generator short-circuited at pass 2 -- expected
        assert result.spec is None
        # But pass 1's prompt should have included the primer
        assert len(captured_prompts) >= 1
        assert "PRIOR MECHANISMS" in captured_prompts[0]
        assert "pre_classify_signal" in captured_prompts[0]

    def test_freeform_prompt_no_primer_when_bio_store_empty(
        self, fresh_store, mantis_camera
    ) -> None:
        """An empty bio_store contributes no primer text."""
        import asyncio

        from belief.photosynthesis.synthesis.cross_domain_generator import (
            synthesize_cross_domain,
        )

        captured_prompts: list[str] = []

        async def gen(prompt, *, temperature, max_tokens):
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                return "freeform"
            return "not json"

        asyncio.run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                bio_store=fresh_store,  # empty
            )
        )
        assert "PRIOR MECHANISMS" not in captured_prompts[0]

    def test_no_bio_store_means_no_primer(self, mantis_camera) -> None:
        """When bio_store=None, the freeform pass has no priming."""
        import asyncio

        from belief.photosynthesis.synthesis.cross_domain_generator import (
            synthesize_cross_domain,
        )

        captured_prompts: list[str] = []

        async def gen(prompt, *, temperature, max_tokens):
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                return "freeform"
            return "not json"

        asyncio.run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert "PRIOR MECHANISMS" not in captured_prompts[0]

    def test_critic_rejection_does_not_add_to_store(self, fresh_store, mantis_camera) -> None:
        """REJECTed mechanisms must NOT be added to the bio_store."""
        import asyncio

        from belief.photosynthesis.synthesis.cross_domain_generator import (
            synthesize_cross_domain,
        )

        responses = [
            "freeform",
            json.dumps(
                {
                    "name": mantis_camera.predicate_in_source.name,
                    "arity": mantis_camera.predicate_in_source.arity,
                    "roles": list(mantis_camera.predicate_in_source.roles),
                    "marr_level": mantis_camera.predicate_in_source.marr_level,
                    "rationale": "x",
                }
            ),
            json.dumps(
                {
                    "considered_and_rejected_attributes": list(
                        mantis_camera.considered_and_rejected_attributes
                    )
                }
            ),
            mantis_camera.model_dump_json(),
        ]
        i = [0]

        async def gen(prompt, *, temperature, max_tokens):
            r = responses[i[0]]
            i[0] += 1
            return r

        async def critic_rejecting(prompt, *, temperature, max_tokens):
            return json.dumps(
                {
                    "verdict": "REJECT",
                    "checks": [
                        {"id": cid, "name": f"c{cid}", "passed": cid != 7, "reason": "x"}
                        for cid in range(1, 9)
                    ],
                }
            )

        result = asyncio.run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                critic_client=critic_rejecting,
                bio_store=fresh_store,
            )
        )
        assert result.spec is None
        assert result.reason == "critic_rejected"
        assert fresh_store.count() == 0
