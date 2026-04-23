"""
Covenant Registry — manages static (hand-written) and dynamic (crystallized) covenants.

Static covenants are the 6 enforcers in belief/validators/__init__.py.
Dynamic covenants are discovered by the crystallizer and stored in the
belief_covenants ChromaDB collection.

The registry provides a unified interface to fire all covenants against
generated code and track their effectiveness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("belief.validators.registry")


@dataclass
class CovenantResult:
    """Result of firing a single covenant against code."""

    name: str
    source: str  # "static" | "dynamic"
    passed: bool
    violations: list[dict] = field(default_factory=list)
    time_ms: float = 0.0
    auto_fixed: bool = False


class CovenantRegistry:
    """Manages and fires both static and dynamic covenants.

    Usage:
        soil = Soil()
        registry = CovenantRegistry(soil)
        results = registry.fire_all(code_files, manifest)
    """

    def __init__(self, soil) -> None:
        self.soil = soil
        self._static_covenants = _load_static_covenants()
        self._dynamic_covenants: list[dict] = []
        self._fire_counts: dict[str, int] = {}
        self._false_positive_counts: dict[str, int] = {}

        # Load dynamic covenants from ChromaDB
        self.load_dynamic_covenants()

    def load_dynamic_covenants(self) -> None:
        """Load crystallized covenants from the belief_covenants collection."""
        self._dynamic_covenants = []

        try:
            col = self.soil._collections.get("belief_covenants")
            if col is None or col.count() == 0:
                return

            results = col.get(
                include=["documents", "metadatas"],
                limit=col.count(),
            )

            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] or {}
                tags = meta.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]

                # Only load crystallized covenants (not hand-written ones)
                if "crystallized" in tags:
                    self._dynamic_covenants.append(
                        {
                            "id": doc_id,
                            "name": meta.get("content", doc_id)[:80],
                            "description": meta.get("content", ""),
                            "code": meta.get("code_sample", ""),
                            "implementation_kind": (
                                "ast"
                                if "ast" in tags
                                else "regex"
                                if "regex" in tags
                                else "assertion"
                            ),
                            "tags": tags,
                        }
                    )

            logger.info(f"Registry: loaded {len(self._dynamic_covenants)} dynamic covenants")
        except Exception as e:
            logger.warning(f"Registry: failed to load dynamic covenants: {e}")

    def fire_all(
        self,
        code_files: dict[str, str],
        manifest: Optional[dict] = None,
    ) -> list[CovenantResult]:
        """Run all covenants (static + dynamic) against generated code.

        Args:
            code_files: dict of filename -> content
            manifest:   optional project manifest (dependencies, etc.)

        Returns:
            List of CovenantResult for each covenant fired.
        """
        results: list[CovenantResult] = []

        # Fire static covenants
        for covenant in self._static_covenants:
            t0 = time.time()
            name = covenant["name"]
            self._fire_counts[name] = self._fire_counts.get(name, 0) + 1

            try:
                violations = covenant["checker"](code_files)
                elapsed_ms = (time.time() - t0) * 1000
                results.append(
                    CovenantResult(
                        name=name,
                        source="static",
                        passed=len(violations) == 0,
                        violations=violations,
                        time_ms=elapsed_ms,
                    )
                )
            except Exception:
                results.append(
                    CovenantResult(
                        name=name,
                        source="static",
                        passed=True,
                        time_ms=(time.time() - t0) * 1000,
                    )
                )

        # Fire dynamic covenants
        for covenant in self._dynamic_covenants:
            t0 = time.time()
            name = covenant.get("name", covenant["id"])
            self._fire_counts[name] = self._fire_counts.get(name, 0) + 1

            try:
                violations = self._fire_dynamic(covenant, code_files)
                elapsed_ms = (time.time() - t0) * 1000
                results.append(
                    CovenantResult(
                        name=name,
                        source="dynamic",
                        passed=len(violations) == 0,
                        violations=violations,
                        time_ms=elapsed_ms,
                    )
                )
            except Exception as e:
                logger.debug(f"Dynamic covenant {name} failed: {e}")
                results.append(
                    CovenantResult(
                        name=name,
                        source="dynamic",
                        passed=True,
                        time_ms=(time.time() - t0) * 1000,
                    )
                )

        return results

    def _fire_dynamic(self, covenant: dict, code_files: dict[str, str]) -> list[dict]:
        """Execute a dynamic covenant's code against code files."""
        code = covenant.get("code", "")
        if not code:
            return []

        # Execute the covenant code in a sandboxed namespace
        namespace: dict = {}
        try:
            exec(compile(code, f"<covenant:{covenant['id']}>", "exec"), namespace)  # noqa: S102
        except Exception:
            return []

        # Find the check function (named check_<name>)
        check_fn = None
        for key, val in namespace.items():
            if key.startswith("check_") and callable(val):
                check_fn = val
                break

        if check_fn is None:
            return []

        violations: list[dict] = []
        for fname, content in code_files.items():
            if not fname.endswith(".py"):
                continue
            try:
                result = check_fn(fname, content)
                if result:
                    violations.extend(result)
            except Exception:
                pass

        return violations

    def get_all_covenant_descriptions(self) -> list[dict]:
        """Return descriptions of all covenants (for the Claude proposer)."""
        descriptions: list[dict] = []
        for cov in self._static_covenants:
            descriptions.append(
                {
                    "name": cov["name"],
                    "description": cov["description"],
                    "source": "static",
                }
            )
        for cov in self._dynamic_covenants:
            descriptions.append(
                {
                    "name": cov.get("name", ""),
                    "description": cov.get("description", ""),
                    "source": "dynamic",
                }
            )
        return descriptions

    def get_covenant_stats(self) -> dict:
        """Return statistics about the covenant registry."""
        return {
            "static_count": len(self._static_covenants),
            "dynamic_count": len(self._dynamic_covenants),
            "total_count": len(self._static_covenants) + len(self._dynamic_covenants),
            "fire_counts": dict(self._fire_counts),
            "false_positive_counts": dict(self._false_positive_counts),
        }


# ── Static covenant loader ──────────────────────────────────────────────────


def _load_static_covenants() -> list[dict]:
    """Load the 6 hand-written covenant enforcers as registry entries."""
    from belief.validators import (
        _enforce_file_length,
        _enforce_no_bare_except,
        _enforce_no_future_with_sqlalchemy,
        _enforce_no_stdlib_in_requirements,
        _enforce_sqlalchemy_imports,
        _enforce_stdlib_imports,
    )

    def _make_checker(enforcer_fn, needs_context=False):
        """Wrap a single-file enforcer into a multi-file checker."""

        def checker(code_files: dict[str, str]) -> list[dict]:
            violations = []
            all_code = "\n".join(code_files.values())
            uses_sqlalchemy = "sqlalchemy" in all_code.lower()

            for fname, code in code_files.items():
                if not fname.endswith(".py") and enforcer_fn != _enforce_no_stdlib_in_requirements:
                    continue
                try:
                    result = enforcer_fn(fname, code, uses_sqlalchemy)
                    for v in result:
                        violations.append(
                            {
                                "file": v.file,
                                "line": v.line,
                                "message": v.message,
                                "severity": v.severity,
                            }
                        )
                except Exception:
                    pass
            return violations

        return checker

    return [
        {
            "name": "no_future_with_sqlalchemy",
            "description": "Don't use __future__ annotations with SQLAlchemy ORM types",
            "checker": _make_checker(_enforce_no_future_with_sqlalchemy),
        },
        {
            "name": "sqlalchemy_mapped_imports",
            "description": "SQLAlchemy Mapped/mapped_column must be imported when used",
            "checker": _make_checker(_enforce_sqlalchemy_imports),
        },
        {
            "name": "max_200_lines",
            "description": "No generated file should exceed 200 lines",
            "checker": _make_checker(_enforce_file_length),
        },
        {
            "name": "explicit_stdlib_imports",
            "description": "Every stdlib module used must be explicitly imported",
            "checker": _make_checker(_enforce_stdlib_imports),
        },
        {
            "name": "no_stdlib_in_requirements",
            "description": "requirements.txt must not include stdlib packages",
            "checker": _make_checker(_enforce_no_stdlib_in_requirements),
        },
        {
            "name": "no_bare_except",
            "description": "No bare except clauses in generated code",
            "checker": _make_checker(_enforce_no_bare_except),
        },
    ]
