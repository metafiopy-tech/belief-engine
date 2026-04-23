"""Synthesizer Agent — polish code + generate deployment artifacts (M4).

After polishing code, generates:
- Dockerfile, docker-compose.yml (if API/server detected)
- .env.example (from config schemas or code scanning)
- GitHub Actions CI/CD workflow
- Railway deployment config

Deployment artifacts are generated deterministically from a ProjectManifest
derived from the SkeletonArtifact (if available) or code analysis.
"""

from __future__ import annotations

import ast
import logging
import re

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.state import Phase, UnifiedState
from belief.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_PROMPT

logger = logging.getLogger("belief.agents.synthesizer")


class SynthesizerAgent(BaseAgent):
    role = ModelRole.SYNTHESIZER
    name = "Synthesizer"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.SYNTHESIS
        if not state.code_files:
            state.warnings.append("Synthesizer: no code files")
            state.phase = Phase.VALIDATION
            return state

        # Session 4 (v3.2): in local mode, route the polish call through a
        # smaller/faster model (default qwen2.5-coder:1.5b).  The 1.5B is
        # ~40-60 tok/s on M2 Air vs 5-8 tok/s for 14B — a 180s polish
        # becomes ~8s.  The synthesizer's prompt is short enough that the
        # quality drop from 14B → 1.5B is (per ablation) acceptable.
        # Override via env: SYNTHESIZER_POLISH_MODEL=... (empty disables swap).
        import os as _os

        _polish_model = _os.environ.get("SYNTHESIZER_POLISH_MODEL", "qwen2.5-coder:1.5b").strip()
        _original_local_model: str | None = None
        try:
            _mode_val = getattr(self.router.mode, "value", str(self.router.mode))
        except Exception:
            _mode_val = ""
        if _polish_model and _mode_val == "local":
            _original_local_model = self.router.local_model
            self.router.local_model = _polish_model
            logger.info(
                "Synthesizer: routing polish through %s (local) instead of %s",
                _polish_model,
                _original_local_model,
            )

        llm = LLMClient(self.router)
        try:
            spec = state.requirement_spec
            goal = spec.goal if spec else state.user_goal

            # Milestone B: Skip files over 6000 chars from polishing —
            # the LLM truncates large files, replacing working code with broken code.
            # Also skip test files — they don't need polishing and waste tokens.
            #
            # Skip skeleton files too: they were generated deterministically
            # from the architect's artifact and must remain exact. An LLM
            # polish pass on a skeleton file drops exports (seen: `init_db`
            # vanished from `database.py` after synthesis, breaking every
            # downstream `from database import init_db`). The debugger's
            # additive-only guard doesn't help here because synthesis
            # runs in a different code path after debugging.
            skeleton_files: set[str] = set()
            if getattr(state, "skeleton_files", None):
                skeleton_files = set(state.skeleton_files.keys())

            polish_candidates = {
                f: c
                for f, c in sorted(state.code_files.items())
                if len(c) <= 6000
                and "/test" not in f
                and not f.startswith("test")
                and f.endswith(".py")
                and f not in skeleton_files
            }
            # Include non-.py files (config, requirements, etc.) without size limit
            for f, c in state.code_files.items():
                if not f.endswith(".py") and f not in ("README.md",):
                    polish_candidates[f] = c

            if not polish_candidates:
                logger.info("Synthesizer: no files eligible for polishing (all too large or tests)")
                state.phase = Phase.VALIDATING
                # Still generate deployment artifacts
                state.code_files = _generate_deployment_artifacts(state)
                return state

            code_str = "\n\n".join(
                f"--- {f} ---\n{c}" for f, c in sorted(polish_candidates.items())
            )

            prompt = SYNTHESIZER_PROMPT.format(
                goal=goal,
                goal_refined=spec.goal_refined if spec else goal,
                acceptance_criteria="\n".join(
                    f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)
                )
                if spec
                else "  (none)",
                credentials="\n".join(f"  - {c.name}: {c.env_var}" for c in spec.credentials)
                if spec and spec.credentials
                else "  (none)",
                code_files=code_str,
            )
            raw = await llm.generate_text(
                role=self.role,
                system=SYNTHESIZER_SYSTEM,
                prompt=prompt,
                temperature=0.3,
                complexity=state.complexity_score,
            )
            polished = _parse_files(raw)
            if polished:
                merged = dict(state.code_files)
                accepted = 0
                rejected = 0
                for fname, content in polished.items():
                    if fname.endswith(".py"):
                        if not _valid_python(content):
                            logger.warning(f"Synthesizer: rejected {fname} (syntax error)")
                            rejected += 1
                            continue

                        # Milestone B: If polished version is >20% shorter than original
                        # and original was valid Python, the polish was truncated — keep original
                        original = state.code_files.get(fname, "")
                        if original and len(content) < len(original) * 0.8:
                            if _valid_python(original):
                                logger.warning(
                                    f"Synthesizer: rejected {fname} "
                                    f"(polished version {len(content)} chars is >20% shorter "
                                    f"than original {len(original)} chars — likely truncated)"
                                )
                                rejected += 1
                                continue

                    merged[fname] = content
                    accepted += 1

                state.code_files = merged
                msg = f"Synthesizer: polished {accepted} file(s)"
                if rejected:
                    msg += f", rejected {rejected} (syntax errors)"
                logger.info(msg)

        except Exception as e:
            logger.warning(f"Synthesizer error: {e}")
            state.warnings.append(f"Synthesizer error: {e}")
        finally:
            await llm.close()
            # Session 4: restore the router's original local model so
            # downstream agents use the primary model, not the polish one.
            if _original_local_model is not None:
                self.router.local_model = _original_local_model

        # Ensure basics
        state.code_files = _ensure_basics(state.code_files, state.user_goal)

        # --- M4: Generate deployment artifacts ---
        state.code_files = _generate_deployment_artifacts(state)

        # --- run.sh: bootstrap venv + launch entry point ---
        state.code_files = _ensure_run_script(state.code_files)

        state.phase = Phase.VALIDATION
        return state


def _generate_deployment_artifacts(state: UnifiedState) -> dict[str, str]:
    """Generate Dockerfile, compose, CI/CD etc. if applicable.

    Uses ProjectManifest derived from SkeletonArtifact or code analysis.
    Only generates if the project appears to be a server/API.
    """
    files = dict(state.code_files)

    try:
        from belief.models.project_manifest import (
            manifest_from_skeleton,
            ProjectManifest,
            ServiceType,
        )
        from belief.tools.deployment_generator import generate_all_deployment_artifacts

        # Try skeleton-based manifest first
        skeleton = state.skeleton_artifact
        if skeleton:
            from belief.models.skeleton import SkeletonArtifact

            if isinstance(skeleton, dict):
                skeleton = SkeletonArtifact.model_validate(skeleton)
            manifest = manifest_from_skeleton(skeleton, files)
        else:
            # Fallback: scan code for patterns
            all_code = "\n".join(files.values())
            is_api = any(kw in all_code.lower() for kw in ["fastapi", "flask", "uvicorn", ".run("])
            if not is_api:
                return files  # Not a server project, skip deployment

            manifest = ProjectManifest(
                project_name="automation",
                service_type=ServiceType.API if is_api else ServiceType.CLI,
                pip_packages=list(
                    {
                        line.strip()
                        for line in files.get("requirements.txt", "").splitlines()
                        if line.strip() and not line.startswith("#")
                    }
                ),
            )

        # Only generate if it's a server/API
        if manifest.service_type not in (ServiceType.API, ServiceType.WORKER):
            return files

        artifacts = generate_all_deployment_artifacts(manifest)

        # Don't overwrite existing files
        for fname, content in artifacts.items():
            if fname not in files:
                files[fname] = content
                logger.info(f"Synthesizer: generated deployment artifact {fname}")

        logger.info(f"Synthesizer: added {len(artifacts)} deployment artifacts")

    except Exception as e:
        logger.debug(f"Synthesizer: deployment generation skipped: {e}")

    return files


def _valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _parse_files(raw: str) -> dict[str, str]:
    if "###FILE:" not in raw:
        return {}
    files = {}
    parts = re.split(r"###FILE:\s*", raw)
    for part in parts[1:]:
        nl = part.find("\n")
        if nl == -1:
            continue
        fname = part[:nl].strip()
        if not fname:
            continue
        content = re.sub(r"###END\s*$", "", part[nl + 1 :])
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        files[fname] = content
    return files


_ENTRY_PRIORITY = (
    "main.py",
    "app.py",
    "server.py",
    "run.py",
    "backend/main.py",
    "backend/app.py",
    "src/main.py",
    "src/app.py",
)


def _detect_entry_point(files: dict[str, str]) -> tuple[str, str] | None:
    """Pick the entry point script and return (relative_path, launch_cmd).

    ``launch_cmd`` is the shell snippet to actually run the file —
    ``uvicorn module:app --host 0.0.0.0 --port "${PORT:-8000}"`` when an
    ASGI app is detected, otherwise ``python <path>``. Returns None
    when no plausible entry point exists.
    """
    py_files = [
        f for f in files if f.endswith(".py") and not f.startswith("test") and "/test" not in f
    ]
    if not py_files:
        return None

    chosen: str | None = None
    for candidate in _ENTRY_PRIORITY:
        if candidate in files:
            chosen = candidate
            break
    if chosen is None:
        # Fall back: shortest top-level .py file path that isn't a test.
        top_level = [f for f in py_files if "/" not in f]
        if top_level:
            chosen = sorted(top_level, key=len)[0]
        else:
            chosen = sorted(py_files, key=len)[0]

    content = files.get(chosen, "")
    is_asgi = bool(re.search(r"\bFastAPI\s*\(", content) or re.search(r"\bStarlette\s*\(", content))
    has_main_block = "__main__" in content

    if is_asgi and not has_main_block:
        # Convert path to module: backend/main.py -> backend.main
        module = chosen[:-3].replace("/", ".")
        # Pick the ASGI variable name (default: app)
        m = re.search(r"^\s*(\w+)\s*=\s*FastAPI\s*\(", content, re.MULTILINE)
        var = m.group(1) if m else "app"
        launch = f'uvicorn {module}:{var} --host 0.0.0.0 --port "${{PORT:-8000}}"'
    else:
        launch = f'python "{chosen}"'

    return chosen, launch


def _ensure_run_script(files: dict[str, str]) -> dict[str, str]:
    """Write a run.sh that bootstraps a venv and launches the entry point.

    Idempotent — re-running the script reuses an existing ``.venv`` and
    skips ``pip install`` when ``requirements.txt`` is empty or absent.
    Set ``BELIEF_RUN_SKIP_INSTALL=1`` to bypass dependency installation
    on subsequent runs once the venv is warm.
    """
    if "run.sh" in files:
        return files

    entry = _detect_entry_point(files)
    if entry is None:
        return files
    entry_path, launch_cmd = entry

    has_requirements = bool(files.get("requirements.txt", "").strip())
    install_block = (
        '  if [ -z "${BELIEF_RUN_SKIP_INSTALL:-}" ] && [ -s requirements.txt ]; then\n'
        '    "$VENV/bin/pip" install --quiet --disable-pip-version-check -r requirements.txt\n'
        "  fi\n"
        if has_requirements
        else "  :\n"
    )

    script = f"""#!/usr/bin/env bash
# Auto-generated by belief-engine synthesizer.
# Bootstraps an isolated venv next to the build and launches the entry point.
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
{install_block}fi

export PYTHONPATH="${{PYTHONPATH:-}}${{PYTHONPATH:+:}}."
export PATH="$PWD/$VENV/bin:$PATH"

exec {launch_cmd} "$@"
"""

    out = dict(files)
    out["run.sh"] = script
    logger.info(f"Synthesizer: generated run.sh (entry={entry_path})")
    return out


def _ensure_basics(files: dict[str, str], goal: str) -> dict[str, str]:
    result = dict(files)
    if "README.md" not in result:
        result["README.md"] = (
            f"# Automation\n\n{goal}\n\n## Setup\n\n1. `pip install -r requirements.txt`\n2. Set env vars\n3. `python main.py`\n"
        )
    if any(f.endswith(".py") for f in result) and "requirements.txt" not in result:
        result["requirements.txt"] = "# No external dependencies\n"
    return result
