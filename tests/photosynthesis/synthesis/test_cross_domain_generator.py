"""Tests for the cross-domain synthesizer + critic (SE Session 3).

The four-pass synthesizer and the CoVe critic are exercised here with
canned LLM responses replayed FIFO from a tiny fake client. No real
LLM tokens are spent; the goal is to pin schema validation, the
four-pass control flow, and the critic's accept/reject behavior.
"""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable


from belief.photosynthesis.synthesis import prompts_cross_domain as prompts
from belief.photosynthesis.synthesis.cross_domain_critic import (
    critique,
)
from belief.photosynthesis.synthesis.cross_domain_generator import (
    synthesize_cross_domain,
)
from belief.photosynthesis.synthesis.structural_mechanism import (
    StructuralMechanism,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _few_shot_json(name: str) -> str:
    """Return a few-shot string with .format-style escapes resolved."""
    txt = getattr(prompts, name)
    return txt.replace("{{", "{").replace("}}", "}")


def _few_shot_mech(name: str) -> StructuralMechanism:
    return StructuralMechanism.model_validate(json.loads(_few_shot_json(name)))


def _make_fake_client(responses: list[str]) -> Callable[..., Awaitable[str]]:
    """FIFO fake LLM client. Replays canned responses in order."""
    state = {"i": 0}

    async def gen(prompt: str, *, temperature: float, max_tokens: int) -> str:
        idx = state["i"]
        if idx >= len(responses):
            raise AssertionError(
                f"fake client exhausted (asked for response {idx + 1}; "
                f"only {len(responses)} canned)"
            )
        state["i"] += 1
        return responses[idx]

    gen.calls_made = state  # type: ignore[attr-defined]
    return gen


def _accept_critic_response() -> str:
    """Canned critic response: ACCEPT verdict with all 8 checks passing.

    Note: the critic's deterministic short-circuit runs checks 1 and 2
    before the LLM is called, so the LLM only needs to return checks
    3-8 (which the merge logic accepts). For test simplicity we
    return all 8.
    """
    return json.dumps(
        {
            "verdict": "ACCEPT",
            "checks": [
                {"id": cid, "name": f"c{cid}", "passed": True, "reason": "ok"}
                for cid in range(1, 9)
            ],
        }
    )


def _reject_critic_response() -> str:
    return json.dumps(
        {
            "verdict": "REJECT",
            "checks": [
                {"id": cid, "name": f"c{cid}", "passed": cid != 7, "reason": "..."}
                for cid in range(1, 9)
            ],
        }
    )


def _four_pass_responses(few_shot_name: str) -> list[str]:
    """Build a canonical four-pass response set anchored to a few-shot."""
    mech = _few_shot_mech(few_shot_name)
    return [
        # Pass 1: free-form brainstorm (any prose)
        f"Both {mech.source_domain} and {mech.target_domain} share a "
        f"deep mechanism around {mech.predicate_in_source.name}. ...",
        # Pass 2: predicate JSON
        json.dumps(
            {
                "name": mech.predicate_in_source.name,
                "arity": mech.predicate_in_source.arity,
                "roles": list(mech.predicate_in_source.roles),
                "marr_level": mech.predicate_in_source.marr_level,
                "rationale": "deepest match",
            }
        ),
        # Pass 3: anti-rationalization
        json.dumps(
            {"considered_and_rejected_attributes": list(mech.considered_and_rejected_attributes)}
        ),
        # Pass 4: full mechanism JSON
        _few_shot_json(few_shot_name),
    ]


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class TestCriticDeterministicChecks:
    def test_attribute_style_predicate_short_circuits(self) -> None:
        """A predicate named 'has_many_sensors' must REJECT before the
        LLM is called -- check 1 catches it."""
        bad = StructuralMechanism.model_validate(
            json.loads(_few_shot_json("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        )
        # Mutate to attribute-style name in both predicates (signature
        # equality enforced by schema, so we change both).
        bad_data = json.loads(bad.model_dump_json())
        bad_data["predicate_in_source"]["name"] = "has_many_sensors"
        bad_data["predicate_in_target"]["name"] = "has_many_sensors"
        # And we have to update the higher-order relation to point at
        # the new name so schema validation still passes.
        bad_data["higher_order_relations"][0]["relates"] = [
            "has_many_sensors",
            "downstream_compute",
        ]
        bad = StructuralMechanism.model_validate(bad_data)

        # No critic_client needed -- short-circuit.
        async def never_called(prompt, *, temperature, max_tokens):
            raise AssertionError("critic_client must not be called on short-circuit")

        result = _run(critique(bad, critic_client=never_called))
        assert result.verdict == "REJECT"
        assert result.short_circuited is True
        assert result.checks[0].id == 1
        assert result.checks[0].passed is False

    def test_descriptive_only_roles_short_circuit(self) -> None:
        """A predicate whose roles are all static descriptors fails check 2."""
        bad_data = json.loads(_few_shot_json("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        bad_data["predicate_in_source"]["name"] = "has_color_count"
        bad_data["predicate_in_target"]["name"] = "has_color_count"
        bad_data["predicate_in_source"]["roles"] = ["color", "count"]
        bad_data["predicate_in_target"]["roles"] = ["color", "count"]
        bad_data["higher_order_relations"][0]["relates"] = [
            "has_color_count",
            "downstream_compute",
        ]
        bad = StructuralMechanism.model_validate(bad_data)

        async def never_called(prompt, *, temperature, max_tokens):
            raise AssertionError

        # Check 1 (attribute prefix) actually fires first -- both
        # checks reject this fixture. We pin verdict + short_circuit.
        result = _run(critique(bad, critic_client=never_called))
        assert result.verdict == "REJECT"
        assert result.short_circuited is True


class TestCriticLLMPath:
    def test_accept_when_all_checks_pass(self) -> None:
        good = _few_shot_mech("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        critic = _make_fake_client([_accept_critic_response()])
        result = _run(critique(good, critic_client=critic))
        assert result.verdict == "ACCEPT"
        assert result.short_circuited is False
        assert all(c.passed for c in result.checks)

    def test_reject_when_any_llm_check_fails(self) -> None:
        good = _few_shot_mech("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        critic = _make_fake_client([_reject_critic_response()])
        result = _run(critique(good, critic_client=critic))
        assert result.verdict == "REJECT"
        # The failing LLM check is id=7 in the response
        check_7 = next(c for c in result.checks if c.id == 7)
        assert check_7.passed is False

    def test_malformed_critic_response_rejects(self) -> None:
        good = _few_shot_mech("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        critic = _make_fake_client(["not json at all"])
        result = _run(critique(good, critic_client=critic))
        assert result.verdict == "REJECT"

    def test_critic_client_exception_rejects_with_error(self) -> None:
        good = _few_shot_mech("FEW_SHOT_MANTIS_SHRIMP_CAMERA")

        async def boom(prompt, *, temperature, max_tokens):
            raise RuntimeError("network down")

        result = _run(critique(good, critic_client=boom))
        assert result.verdict == "REJECT"
        assert result.error is not None
        assert "network down" in result.error


# ---------------------------------------------------------------------------
# Synthesizer four-pass control flow
# ---------------------------------------------------------------------------


class TestSynthesizeCrossDomain:
    def test_single_word_short_circuits(self) -> None:
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp"],
                bundle_id="b1",
                generator_client=_make_fake_client([]),
            )
        )
        assert result.spec is None
        assert result.reason == "too_few_words"

    def test_mantis_shrimp_camera_full_pipeline(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        critic = _make_fake_client([_accept_critic_response()])
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                critic_client=critic,
            )
        )
        assert result.spec is not None
        assert result.reason == "accepted"
        assert result.spec.structural_mechanism is not None
        # Loose match per acceptance criteria from the SE plan
        pred_name = result.spec.structural_mechanism.predicate_in_source.name
        assert "pre_classif" in pred_name or "compress" in pred_name

    def test_mycorrhizal_cdn_full_pipeline(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MYCORRHIZAL_CDN"))
        critic = _make_fake_client([_accept_critic_response()])
        result = _run(
            synthesize_cross_domain(
                words=["mycorrhizal_network", "content_delivery_network"],
                bundle_id="b2",
                generator_client=gen,
                critic_client=critic,
            )
        )
        assert result.spec is not None
        assert result.reason == "accepted"
        assert result.spec.structural_mechanism.predicate_in_source.name == (
            "allocate_via_demand_signal"
        )

    def test_slime_mold_routing_full_pipeline(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_SLIME_MOLD_ROUTING"))
        critic = _make_fake_client([_accept_critic_response()])
        result = _run(
            synthesize_cross_domain(
                words=["slime_mold", "routing_table"],
                bundle_id="b3",
                generator_client=gen,
                critic_client=critic,
            )
        )
        assert result.spec is not None
        assert result.spec.structural_mechanism.predicate_in_source.name == ("prune_via_flux_decay")

    def test_no_critic_runs_without_critique(self) -> None:
        """When critic_client is None, the synthesizer skips the critic
        and emits the spec with reason='accepted_no_critic'."""
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                critic_client=None,
            )
        )
        assert result.spec is not None
        assert result.reason == "accepted_no_critic"
        assert result.critic is None

    def test_critic_rejection_drops_spec(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        critic = _make_fake_client([_reject_critic_response()])
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
                critic_client=critic,
            )
        )
        assert result.spec is None
        assert result.reason == "critic_rejected"
        # Mechanism should still be populated (it's the rejected
        # candidate -- callers can inspect it).
        assert result.mechanism is not None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestSynthesizerFailureModes:
    def test_freeform_pass_error(self) -> None:
        async def boom(prompt, *, temperature, max_tokens):
            raise RuntimeError("no api key")

        result = _run(
            synthesize_cross_domain(
                words=["a", "b"],
                bundle_id="b1",
                generator_client=boom,
            )
        )
        assert result.spec is None
        assert result.reason == "freeform_pass_error"

    def test_predicate_validation_error(self) -> None:
        # Pass 2 returns a predicate whose name is not snake_case
        gen = _make_fake_client(
            [
                "freeform brainstorm",
                json.dumps(
                    {
                        "name": "InvalidName",
                        "arity": 2,
                        "roles": ["a", "b"],
                        "marr_level": "algorithmic",
                    }
                ),
            ]
        )
        result = _run(
            synthesize_cross_domain(
                words=["a", "b"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert result.spec is None
        assert result.reason == "predicate_validation_error"

    def test_anti_rationalization_too_few(self) -> None:
        gen = _make_fake_client(
            [
                "freeform",
                json.dumps(
                    {
                        "name": "ok_predicate",
                        "arity": 2,
                        "roles": ["transducer", "signal"],
                        "marr_level": "algorithmic",
                    }
                ),
                json.dumps({"considered_and_rejected_attributes": ["only_one"]}),
            ]
        )
        result = _run(
            synthesize_cross_domain(
                words=["a", "b"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert result.spec is None
        assert result.reason == "anti_rationalization_too_few"

    def test_structurer_schema_invalid(self) -> None:
        # Pass 4 returns a mechanism whose predicates don't match
        bad_mech = {
            "mechanism_id": "x",
            "source_domain": "s",
            "target_domain": "t",
            "predicate_in_source": {
                "name": "ok_predicate",
                "arity": 2,
                "roles": ["transducer", "signal"],
                "marr_level": "algorithmic",
            },
            "predicate_in_target": {
                "name": "different_predicate",  # name mismatch!
                "arity": 2,
                "roles": ["transducer", "signal"],
                "marr_level": "algorithmic",
            },
            "higher_order_relations": [{"name": "x", "relates": ["ok_predicate", "other"]}],
            "near_miss": {
                "description": "x",
                "breaks_at_argument": "predicate_in_source.argument[0]",
            },
            "considered_and_rejected_attributes": ["a", "b"],
        }
        gen = _make_fake_client(
            [
                "freeform",
                json.dumps(
                    {
                        "name": "ok_predicate",
                        "arity": 2,
                        "roles": ["transducer", "signal"],
                        "marr_level": "algorithmic",
                    }
                ),
                json.dumps({"considered_and_rejected_attributes": ["a", "b"]}),
                json.dumps(bad_mech),
            ]
        )
        result = _run(
            synthesize_cross_domain(
                words=["a", "b"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert result.spec is None
        assert result.reason == "schema_invalid"

    def test_structurer_parse_error_returns_clean_reason(self) -> None:
        gen = _make_fake_client(
            [
                "freeform",
                json.dumps(
                    {
                        "name": "ok_predicate",
                        "arity": 2,
                        "roles": ["transducer", "signal"],
                        "marr_level": "algorithmic",
                    }
                ),
                json.dumps({"considered_and_rejected_attributes": ["a", "b"]}),
                "absolutely not json",
            ]
        )
        result = _run(
            synthesize_cross_domain(
                words=["a", "b"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert result.spec is None
        assert result.reason == "structurer_parse_error"


# ---------------------------------------------------------------------------
# GoalSpec wrapper
# ---------------------------------------------------------------------------


class TestGoalSpecWrapping:
    def test_goalspec_carries_structural_mechanism(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert result.spec.structural_mechanism is not None
        # Round-trip
        rt = type(result.spec).model_validate_json(result.spec.model_dump_json())
        assert rt.structural_mechanism == result.spec.structural_mechanism

    def test_goalspec_source_citation_links_back_to_bundle(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="bundle_xyz",
                generator_client=gen,
            )
        )
        assert "bundle_xyz" in result.spec.source_citation

    def test_as_generator_result_adapts_cleanly(self) -> None:
        gen = _make_fake_client(_four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA"))
        xd_result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        gr = xd_result.as_generator_result()
        assert gr.spec is xd_result.spec
        assert gr.reason == xd_result.reason


# ---------------------------------------------------------------------------
# LLM field-drift normalization (SE Session 7.8 hotfix)
#
# Surfaced from a live `belief synth words mantis_shrimp,camera` run:
# Sonnet 4.5 systematically emits `relation_name` for
# HigherOrderRelation.name. ALL 10 bundles got rejected with the same
# Pydantic ValidationError ("Field required: higher_order_relations[N].name").
# The normalizer fixes a whitelist of known drifts before validation.
# ---------------------------------------------------------------------------


class TestLLMFieldDriftNormalization:
    def test_normalize_renames_relation_name_to_name(self) -> None:
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        data = {
            "higher_order_relations": [
                {"relation_name": "reduces_compute", "relates": ["a", "b"]},
                {"relation_name": "scans_geometry", "relates": ["a", "c"]},
            ],
        }
        out = _normalize_llm_field_drift(data)
        assert out["higher_order_relations"][0]["name"] == "reduces_compute"
        assert out["higher_order_relations"][1]["name"] == "scans_geometry"
        assert "relation_name" not in out["higher_order_relations"][0]
        assert "relation_name" not in out["higher_order_relations"][1]

    def test_normalize_leaves_correct_name_alone(self) -> None:
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        data = {
            "higher_order_relations": [
                {"name": "already_correct", "relates": ["a", "b"]},
            ],
        }
        out = _normalize_llm_field_drift(data)
        assert out["higher_order_relations"][0]["name"] == "already_correct"

    def test_normalize_prefers_name_when_both_present(self) -> None:
        """If both name AND relation_name exist, schema's name wins."""
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        data = {
            "higher_order_relations": [
                {
                    "name": "schema_name",
                    "relation_name": "drift_name",
                    "relates": ["a", "b"],
                },
            ],
        }
        out = _normalize_llm_field_drift(data)
        assert out["higher_order_relations"][0]["name"] == "schema_name"
        # The drift field is left alone -- model_validate would have ignored
        # an extra key anyway, and we don't want to silently mutate when
        # the schema field is already present.

    def test_normalize_handles_predicate_name_drift(self) -> None:
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        data = {
            "predicate_in_source": {
                "predicate_name": "foo",
                "arity": 2,
                "roles": ["a", "b"],
                "marr_level": "algorithmic",
            },
            "predicate_in_target": {
                "predicate_name": "foo",
                "arity": 2,
                "roles": ["a", "b"],
                "marr_level": "algorithmic",
            },
        }
        out = _normalize_llm_field_drift(data)
        assert out["predicate_in_source"]["name"] == "foo"
        assert out["predicate_in_target"]["name"] == "foo"
        assert "predicate_name" not in out["predicate_in_source"]
        assert "predicate_name" not in out["predicate_in_target"]

    def test_normalize_handles_near_miss_description_drift(self) -> None:
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        data = {
            "near_miss": {
                "near_miss_description": "a counterexample",
                "breaks_at_argument": "predicate_in_source.argument[0]",
            },
        }
        out = _normalize_llm_field_drift(data)
        assert out["near_miss"]["description"] == "a counterexample"

    def test_normalize_no_op_on_non_dict(self) -> None:
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        assert _normalize_llm_field_drift(None) is None
        assert _normalize_llm_field_drift("string") == "string"
        assert _normalize_llm_field_drift(42) == 42
        assert _normalize_llm_field_drift([1, 2, 3]) == [1, 2, 3]

    def test_normalize_no_op_on_missing_keys(self) -> None:
        from belief.photosynthesis.synthesis.cross_domain_generator import (
            _normalize_llm_field_drift,
        )

        # Empty / partial dicts -- normalize should not raise.
        assert _normalize_llm_field_drift({}) == {}
        assert _normalize_llm_field_drift({"unrelated": "key"}) == {"unrelated": "key"}

    def test_full_synthesizer_accepts_relation_name_drift(self) -> None:
        """End-to-end: structurer emits relation_name, generator
        normalizes, validation passes, GoalSpec gets returned."""
        # Build the canonical four-pass set, then drift the structurer's
        # higher_order_relations so the relation key is 'relation_name'.
        canonical = _four_pass_responses("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        drifted_structurer = json.loads(canonical[3])
        for rel in drifted_structurer["higher_order_relations"]:
            rel["relation_name"] = rel.pop("name")
        canonical[3] = json.dumps(drifted_structurer)

        gen = _make_fake_client(canonical)
        result = _run(
            synthesize_cross_domain(
                words=["mantis_shrimp", "digital_camera"],
                bundle_id="b1",
                generator_client=gen,
            )
        )
        assert result.spec is not None, f"expected success, got reason={result.reason}"
        assert result.mechanism is not None
        assert result.mechanism.higher_order_relations[0].name
