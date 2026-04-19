"""
Episode Recorder — records build traces to the belief_episodes collection.

Each build produces an episode: a structured trace of what happened,
what dependencies were used, whether tests passed, and what code
patterns were present.  The crystallizer analyzes these episodes to
discover new covenants.

Wired into the decomposer node so every build is recorded automatically.
"""

from __future__ import annotations

import ast
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger("belief.memory.episode_recorder")

# Frameworks that indicate an API project
_API_FRAMEWORKS = {"fastapi", "flask", "django", "starlette", "sanic", "falcon", "bottle"}


def record_episode(soil, build_state: dict[str, Any]) -> str:
    """Record a build trace to the belief_episodes collection.

    Extracts structured features from the build state that the
    crystallizer's invariant templates can query.

    Args:
        soil:        Soil instance with access to collections.
        build_state: The LangGraph state dict from the build pipeline.

    Returns:
        The episode ID.
    """
    code_files = build_state.get("code_files", {})
    goal = build_state.get("user_goal", "")

    # Extract dependencies from requirements.txt
    deps = _extract_dependencies(code_files.get("requirements.txt", ""))

    # Analyze code for structural features
    features = _analyze_code(code_files)

    # Build the episode record
    episode = {
        "goal": goal[:500],
        "trace_id": build_state.get("run_id", f"ep-{uuid.uuid4().hex[:12]}"),
        "passed": _is_passing(build_state),
        "score": _get_score(build_state),
        "cost_usd": _get_cost(build_state),
        "file_count": len(code_files),
        "test_count": features["test_count"],
        "dependencies": deps,
        "has_api_framework": features["has_api_framework"],
        "has_health_endpoint": features["has_health_endpoint"],
        "has_click_conftest": features["has_click_conftest"],
        "bare_except_count": features["bare_except_count"],
        "print_count": features["print_count"],
        "has_entry_point": features["has_entry_point"],
        "has_dockerfile": features["has_dockerfile"],
        "dockerfile_has_expose": features["dockerfile_has_expose"],
        "hardcoded_secret_count": features["hardcoded_secret_count"],
        "has_error_handler": features["has_error_handler"],
        "mixed_sync_async": features["mixed_sync_async"],
    }

    # Store in episodes collection
    episode_id = episode["trace_id"]
    episodes_col = soil._collections.get("belief_episodes")
    if episodes_col is None:
        logger.warning("Episode recorder: belief_episodes collection not found")
        return episode_id

    # Flatten for ChromaDB metadata (no nested lists in metadata)
    metadata = {}
    for k, v in episode.items():
        if isinstance(v, list):
            metadata[k] = ",".join(str(x) for x in v) if v else ""
        elif isinstance(v, bool):
            metadata[k] = 1 if v else 0
        elif isinstance(v, (int, float, str)):
            metadata[k] = v
        else:
            metadata[k] = str(v)

    episodes_col.upsert(
        ids=[episode_id],
        documents=[f"build episode: {goal[:200]}"],
        metadatas=[metadata],
    )

    logger.debug(f"Recorded episode {episode_id} (passed={episode['passed']})")
    return episode_id


# ── Feature extraction ──────────────────────────────────────────────────────


def _extract_dependencies(requirements_txt: str) -> list[str]:
    """Parse package names from requirements.txt."""
    if not requirements_txt.strip():
        return []

    deps = []
    for line in requirements_txt.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Extract package name before any version specifier
        pkg = re.split(r"[>=<!~\[\s]", line)[0].strip().lower()
        if pkg:
            deps.append(pkg)
    return deps


def _analyze_code(code_files: dict[str, str]) -> dict[str, Any]:
    """Analyze code files for structural features."""
    features = {
        "test_count": 0,
        "has_api_framework": False,
        "has_health_endpoint": False,
        "has_click_conftest": False,
        "bare_except_count": 0,
        "print_count": 0,
        "has_entry_point": False,
        "has_dockerfile": False,
        "dockerfile_has_expose": False,
        "hardcoded_secret_count": 0,
        "has_error_handler": False,
        "mixed_sync_async": False,
    }

    all_code = "\n".join(code_files.values())

    # API framework detection
    for framework in _API_FRAMEWORKS:
        if framework in all_code.lower():
            features["has_api_framework"] = True
            break

    # Health endpoint
    features["has_health_endpoint"] = bool(
        re.search(r'["\'/]health["\']', all_code)
        or re.search(r'@\w+\.(get|route)\s*\(\s*["\']\/health', all_code)
    )

    # Entry point
    features["has_entry_point"] = (
        "main.py" in code_files
        or "app.py" in code_files
        or '__name__ == "__main__"' in all_code
        or "__name__ == '__main__'" in all_code
    )

    # Dockerfile
    features["has_dockerfile"] = "Dockerfile" in code_files
    if features["has_dockerfile"]:
        dockerfile = code_files.get("Dockerfile", "")
        features["dockerfile_has_expose"] = "EXPOSE" in dockerfile

    # Error handler middleware
    features["has_error_handler"] = bool(
        re.search(r"exception_handler|error_handler|@app\.errorhandler", all_code)
    )

    # Hardcoded secrets (crude pattern)
    secret_patterns = [
        r'(api_key|secret_key|password|token)\s*=\s*["\'][^"\']{8,}["\']',
        r'sk-[a-zA-Z0-9]{20,}',
        r'ghp_[a-zA-Z0-9]{36}',
    ]
    for pattern in secret_patterns:
        features["hardcoded_secret_count"] += len(re.findall(pattern, all_code, re.IGNORECASE))

    # Per-file analysis
    has_async_endpoints = False
    has_sync_endpoints = False

    for fname, code in code_files.items():
        if not fname.endswith(".py"):
            continue

        # Test count
        if "test" in fname.lower():
            features["test_count"] += code.count("def test_")
            features["test_count"] += code.count("async def test_")

        # Click conftest
        if fname == "conftest.py" and "click" in code.lower():
            features["has_click_conftest"] = True

        # Print count (excluding test files)
        if "test" not in fname.lower():
            features["print_count"] += len(re.findall(r'\bprint\s*\(', code))

        # AST-based analysis
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Bare except
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    features["bare_except_count"] += 1

                # Async vs sync endpoints
                if isinstance(node, ast.AsyncFunctionDef):
                    if any(
                        isinstance(d, ast.Call) for d in node.decorator_list
                    ):
                        has_async_endpoints = True
                elif isinstance(node, ast.FunctionDef):
                    if any(
                        isinstance(d, ast.Call) for d in node.decorator_list
                    ):
                        has_sync_endpoints = True
        except SyntaxError:
            pass

    features["mixed_sync_async"] = has_async_endpoints and has_sync_endpoints

    return features


def _is_passing(state: dict) -> bool:
    """Check if the build passed."""
    validation = state.get("validation_result")
    if validation:
        v = (validation.get("verdict") if isinstance(validation, dict)
             else getattr(validation, "verdict", None))
        if v:
            verdict = v.value if hasattr(v, "value") else str(v)
            return verdict == "pass"
    return False


def _get_score(state: dict) -> float:
    """Extract the build score."""
    validation = state.get("validation_result")
    if validation:
        score = (validation.get("weighted_score", 0.0) if isinstance(validation, dict)
                 else getattr(validation, "weighted_score", 0.0))
        return float(score)
    return 0.0


def _get_cost(state: dict) -> float:
    """Extract the build cost."""
    return float(state.get("total_cost_usd", 0.0))
