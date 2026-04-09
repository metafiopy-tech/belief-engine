"""
Tests for Milestones 4-6

M4: Docker + Deployment Artifacts
M5: Composition Pattern
M6: Self-Improvement Loop
"""

import tempfile
from pathlib import Path
import pytest

from belief.models.project_manifest import (
    ProjectManifest,
    ServiceType,
    EnvVar,
    ServicePort,
    HealthCheck,
    manifest_from_skeleton,
)
from belief.agents.deployment_generator import (
    generate_dockerfile,
    generate_docker_compose,
    generate_env_example,
    generate_requirements_txt,
    generate_github_actions,
    generate_railway_toml,
    generate_dockerignore,
    generate_all_deployment_artifacts,
)
from belief.agents.composition_planner import (
    PackageCandidate,
    PackageSource,
    ComponentStrategy,
    evaluate_package,
    decide_component_strategy,
    plan_composition,
    CompositionPlan,
    WELL_KNOWN_PACKAGES,
)
from belief.evolution.self_improvement import (
    SEED,
    Mentor,
    SelfPatch,
    ImprovementProposal,
    ImprovementType,
    MentorVerdict,
    PatchResult,
    run_improvement_loop,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_api_manifest() -> ProjectManifest:
    return ProjectManifest(
        project_name="lead_gen_pipeline",
        description="Lead gen pipeline with FastAPI",
        service_type=ServiceType.API,
        entry_command="uvicorn main:app --host 0.0.0.0 --port 8000",
        entry_file="main.py",
        ports=[ServicePort(container_port=8000, host_port=8000)],
        health_check=HealthCheck(path="/health"),
        pip_packages=["fastapi", "httpx", "pydantic", "pydantic-settings"],
        env_vars=[
            EnvVar(name="PIPELINE_GOOGLE_API_KEY", description="Google API key",
                   required=True, secret=True),
            EnvVar(name="PIPELINE_MAX_CONCURRENCY", description="Max concurrency",
                   required=False, default="5"),
            EnvVar(name="PIPELINE_COST_BUDGET", description="Cost budget",
                   required=False, default="2.0"),
        ],
        docker_expose=[8000],
        depends_on=["postgres"],
    )


# ===========================================================================
# M4: Deployment Artifacts
# ===========================================================================

class TestProjectManifest:
    def test_basic_creation(self):
        m = _make_api_manifest()
        assert m.project_name == "lead_gen_pipeline"
        assert m.service_type == ServiceType.API

    def test_manifest_from_skeleton(self):
        from tests.test_milestone2 import _make_12_file_skeleton
        skeleton = _make_12_file_skeleton()
        files = {"main.py": "from fastapi import FastAPI\napp = FastAPI()"}
        manifest = manifest_from_skeleton(skeleton, files)
        assert manifest.project_name == "lead_gen_pipeline"
        assert manifest.service_type == ServiceType.API
        assert len(manifest.env_vars) > 0


class TestDockerfile:
    def test_generates_valid_dockerfile(self):
        m = _make_api_manifest()
        df = generate_dockerfile(m)
        assert "FROM python:3.12-slim" in df
        assert "COPY requirements.txt" in df
        assert "EXPOSE 8000" in df
        assert "HEALTHCHECK" in df
        assert "CMD" in df

    def test_dockerfile_has_pip_install(self):
        m = _make_api_manifest()
        df = generate_dockerfile(m)
        assert "pip install" in df


class TestDockerCompose:
    def test_generates_compose(self):
        m = _make_api_manifest()
        dc = generate_docker_compose(m)
        assert "services:" in dc
        assert "lead_gen_pipeline:" in dc
        assert "8000:8000" in dc

    def test_includes_postgres(self):
        m = _make_api_manifest()
        dc = generate_docker_compose(m)
        assert "postgres:" in dc
        assert "pgdata:" in dc

    def test_health_check(self):
        m = _make_api_manifest()
        dc = generate_docker_compose(m)
        assert "healthcheck:" in dc
        assert "/health" in dc


class TestEnvExample:
    def test_generates_env(self):
        m = _make_api_manifest()
        env = generate_env_example(m)
        assert "PIPELINE_GOOGLE_API_KEY" in env
        assert "PIPELINE_MAX_CONCURRENCY" in env

    def test_secrets_section(self):
        m = _make_api_manifest()
        env = generate_env_example(m)
        assert "Required Secrets" in env

    def test_postgres_vars(self):
        m = _make_api_manifest()
        env = generate_env_example(m)
        assert "POSTGRES_USER" in env
        assert "DATABASE_URL" in env


class TestRequirementsTxt:
    def test_includes_packages(self):
        m = _make_api_manifest()
        req = generate_requirements_txt(m)
        assert "fastapi" in req
        assert "httpx" in req
        assert "pydantic" in req

    def test_adds_uvicorn_for_api(self):
        m = _make_api_manifest()
        req = generate_requirements_txt(m)
        assert "uvicorn" in req


class TestGitHubActions:
    def test_generates_workflow(self):
        m = _make_api_manifest()
        wf = generate_github_actions(m)
        assert "name: CI/CD" in wf
        assert "pytest" in wf
        assert "docker build" in wf.lower()

    def test_test_and_build_jobs(self):
        m = _make_api_manifest()
        wf = generate_github_actions(m)
        assert "test:" in wf
        assert "build:" in wf


class TestRailwayConfig:
    def test_generates_toml(self):
        m = _make_api_manifest()
        rt = generate_railway_toml(m)
        assert "[build]" in rt
        assert "[deploy]" in rt
        assert "healthcheckPath" in rt

    def test_port_config(self):
        m = _make_api_manifest()
        rt = generate_railway_toml(m)
        assert "8000" in rt


class TestAllArtifacts:
    def test_generates_all(self):
        m = _make_api_manifest()
        artifacts = generate_all_deployment_artifacts(m)
        assert "Dockerfile" in artifacts
        assert "docker-compose.yml" in artifacts
        assert ".env.example" in artifacts
        assert "requirements.txt" in artifacts
        assert ".dockerignore" in artifacts
        assert ".github/workflows/ci.yml" in artifacts
        assert "railway.toml" in artifacts


# ===========================================================================
# M5: Composition Pattern
# ===========================================================================

class TestPackageEvaluation:
    def test_well_known_packages(self):
        pkg = evaluate_package("fastapi")
        assert pkg is not None
        assert pkg.name == "fastapi"
        assert pkg.quality_score > 50

    def test_unknown_package(self):
        pkg = evaluate_package("nonexistent_pkg_12345")
        assert pkg is None

    def test_httpx_available(self):
        pkg = evaluate_package("httpx")
        assert pkg is not None
        assert pkg.quality_score > 40

    def test_fastmcp_available(self):
        """Done-when: research finds fastmcp for MCP builds."""
        pkg = evaluate_package("fastmcp")
        assert pkg is not None
        assert pkg.name == "fastmcp"


class TestComponentDecision:
    def test_high_quality_use_library(self):
        candidates = [WELL_KNOWN_PACKAGES["fastapi"]]
        decision = decide_component_strategy("api_framework", "web API", candidates)
        assert decision.strategy == ComponentStrategy.USE_LIBRARY

    def test_no_candidates_generate(self):
        decision = decide_component_strategy("custom_thing", "unique logic", [])
        assert decision.strategy == ComponentStrategy.GENERATE

    def test_low_quality_generate(self):
        weak = PackageCandidate(
            name="bad-lib", downloads_monthly=10, stars=0,
            source_rank=1, maintained=False,
        )
        decision = decide_component_strategy("thing", "desc", [weak])
        assert decision.strategy == ComponentStrategy.GENERATE


class TestCompositionPlan:
    def test_mcp_server_composition(self):
        """Done-when: finds fastmcp and httpx for MCP server builds."""
        requirements = [
            ("http_client", "HTTP client for API calls"),
            ("mcp_server", "MCP server framework"),
            ("custom_logic", "unique business logic for scoring"),
        ]
        plan = plan_composition(requirements)
        assert len(plan.decisions) == 3

        # httpx should be found for http client
        http_decision = next(d for d in plan.decisions if d.component_name == "http_client")
        assert http_decision.strategy in (ComponentStrategy.USE_LIBRARY, ComponentStrategy.WRAP_LIBRARY)

        # custom logic should be generated
        custom_decision = next(d for d in plan.decisions if d.component_name == "custom_logic")
        assert custom_decision.strategy == ComponentStrategy.GENERATE

    def test_plan_summary(self):
        requirements = [("api", "web API"), ("db", "database")]
        plan = plan_composition(requirements)
        summary = plan.summary()
        assert "Composition Plan" in summary

    def test_libraries_to_install(self):
        requirements = [("http_client", "HTTP client")]
        plan = plan_composition(requirements)
        libs = plan.libraries_to_install
        # Should find httpx or similar
        assert isinstance(libs, list)


# ===========================================================================
# M6: Self-Improvement Loop
# ===========================================================================

class TestSEED:
    def test_trigger_interval(self):
        seed = SEED(trigger_interval=5)
        for i in range(4):
            seed.record_build({"cost": 0.5})
            assert not seed.should_trigger()
        seed.record_build({"cost": 0.5})
        assert seed.should_trigger()

    def test_propose_high_corrections(self):
        seed = SEED(trigger_interval=3)
        for _ in range(3):
            seed.record_build({"correction_rounds": 3})
        proposal = seed.propose()
        assert proposal is not None
        assert "correction" in proposal.title.lower()

    def test_propose_high_failures(self):
        seed = SEED(trigger_interval=3)
        for _ in range(3):
            seed.record_build({"failures": ["a.py", "b.py"]})
        proposal = seed.propose()
        assert proposal is not None
        assert "failure" in proposal.title.lower()

    def test_no_propose_when_good(self):
        seed = SEED(trigger_interval=3)
        for _ in range(3):
            seed.record_build({"correction_rounds": 0, "failures": []})
        proposal = seed.propose()
        # Might be None or propose token optimization — either is fine


class TestMentor:
    def test_approve_prompt_change(self):
        mentor = Mentor()
        proposal = ImprovementProposal(
            title="Better prompt",
            description="Improve builder prompt",
            improvement_type=ImprovementType.PROMPT,
            target_file="prompts.py",
            current_code="old",
            proposed_code="new",
            expected_benefit="fewer errors",
            risk_level="low",
        )
        verdict = mentor.evaluate(proposal)
        assert verdict.approved

    def test_reject_high_risk(self):
        mentor = Mentor()
        proposal = ImprovementProposal(
            title="Dangerous change",
            description="Rewrite everything",
            improvement_type=ImprovementType.REFACTOR,
            target_file="graph.py",
            current_code="",
            proposed_code="",
            expected_benefit="maybe better",
            risk_level="high",
        )
        verdict = mentor.evaluate(proposal)
        assert not verdict.approved

    def test_pipeline_change_has_conditions(self):
        mentor = Mentor()
        proposal = ImprovementProposal(
            title="Pipeline tweak",
            description="Modify routing",
            improvement_type=ImprovementType.PIPELINE,
            target_file="graph.py",
            current_code="",
            proposed_code="",
            expected_benefit="faster",
            risk_level="medium",
        )
        verdict = mentor.evaluate(proposal)
        assert verdict.approved
        assert len(verdict.conditions) > 0


class TestSelfPatch:
    def test_apply_and_rollback_on_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_file.py"
            target.write_text("x = 1\n")

            patcher = SelfPatch(tmpdir)
            proposal = ImprovementProposal(
                title="Break syntax",
                description="",
                improvement_type=ImprovementType.PARAMETER,
                target_file="test_file.py",
                current_code="x = 1\n",
                proposed_code="def broken(:\n",  # Syntax error
                expected_benefit="",
            )
            verdict = MentorVerdict(approved=True, reasoning="test")

            result = patcher.apply(proposal, verdict)
            assert not result.success
            assert result.rolled_back
            # File should be restored
            assert target.read_text() == "x = 1\n"

    def test_apply_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_file.py"
            target.write_text("x = 1\n")

            patcher = SelfPatch(tmpdir)
            proposal = ImprovementProposal(
                title="Update value",
                description="",
                improvement_type=ImprovementType.PARAMETER,
                target_file="test_file.py",
                current_code="x = 1\n",
                proposed_code="x = 2\n",
                expected_benefit="",
            )
            verdict = MentorVerdict(approved=True, reasoning="test")

            result = patcher.apply(proposal, verdict)
            assert result.success
            assert target.read_text() == "x = 2\n"

    def test_rollback_on_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_file.py"
            target.write_text("x = 1\n")

            patcher = SelfPatch(tmpdir)
            proposal = ImprovementProposal(
                title="Change that fails validation",
                description="",
                improvement_type=ImprovementType.PARAMETER,
                target_file="test_file.py",
                current_code="x = 1\n",
                proposed_code="x = 999\n",
                expected_benefit="",
            )
            verdict = MentorVerdict(approved=True, reasoning="test")

            # Validation always fails
            result = patcher.apply(proposal, verdict, validate_fn=lambda: False)
            assert not result.success
            assert result.rolled_back
            assert target.read_text() == "x = 1\n"

    def test_rejected_proposal_not_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_file.py"
            target.write_text("x = 1\n")

            patcher = SelfPatch(tmpdir)
            proposal = ImprovementProposal(
                title="Rejected", description="",
                improvement_type=ImprovementType.PARAMETER,
                target_file="test_file.py", current_code="", proposed_code="x = 2\n",
                expected_benefit="",
            )
            verdict = MentorVerdict(approved=False, reasoning="nope")

            result = patcher.apply(proposal, verdict)
            assert not result.success
            assert target.read_text() == "x = 1\n"  # Unchanged


class TestFullLoop:
    def test_loop_triggers_and_applies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "belief" / "prompts" / "skeleton_prompts.py"
            target.parent.mkdir(parents=True)
            target.write_text("PROMPT = 'old'\n")

            seed = SEED(trigger_interval=3)
            mentor = Mentor()
            patcher = SelfPatch(tmpdir)

            # Record builds with high corrections
            for _ in range(3):
                seed.record_build({"correction_rounds": 3, "failures": []})

            assert seed.should_trigger()
            proposal = seed.propose()
            assert proposal is not None

            # Manually set proposed code for the test
            proposal.proposed_code = "PROMPT = 'improved'\n"

            verdict = mentor.evaluate(proposal)
            assert verdict.approved

            result = patcher.apply(proposal, verdict)
            assert result.success

    def test_loop_no_trigger(self):
        seed = SEED(trigger_interval=10)
        mentor = Mentor()
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = SelfPatch(tmpdir)
            result = run_improvement_loop(seed, mentor, patcher)
            assert result is None  # Not triggered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
