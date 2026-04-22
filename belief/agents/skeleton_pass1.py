"""Skeleton Pass 1 Node — Milestones 1-3 integrated.

Sits between Architect and Builder in the graph:
  architect → skeleton_pass1 → builder

When a SkeletonArtifact exists:
1. Topological sort → build plan (M2: dependency DAG)
2. Generate skeleton files in DAG order (M1: deterministic)
3. Parse into SymbolRegistry
4. Build compressed, budget-aware symbol context (M3: context compression)
5. Merge skeleton files into code_files
6. Log compression metrics

Zero LLM calls — all deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("belief.agents.skeleton_pass1")


def _promote_database_files_to_skeleton(skeleton: Any) -> None:
    """Flip `skeleton=True` on any database module in the file tree.

    Mutates the SkeletonArtifact in place. Fires when the project uses
    a SQLAlchemy-family dependency and there's a file in the tree whose
    basename is `database.py` or `db.py` (package-scoped layouts like
    `blog_engine/database.py` also match). The architect's prune step
    can mark these as implementation files, which bypasses the
    deterministic skeleton generator entirely.
    """
    try:
        deps = {
            d.split("[")[0].split("=")[0].split("<")[0].split(">")[0].strip().lower()
            for d in (skeleton.external_dependencies or [])
        }
    except AttributeError:
        return
    if not (deps & {"sqlalchemy", "sqlmodel"}):
        return

    promoted = 0
    for entry in skeleton.file_tree:
        basename = entry.path.rsplit("/", 1)[-1]
        if basename in ("database.py", "db.py") and not entry.skeleton:
            entry.skeleton = True
            promoted += 1
    if promoted:
        logger.info(
            f"SkeletonPass1: promoted {promoted} database file(s) to skeleton=True"
        )


async def skeleton_pass1_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate skeleton files from SkeletonArtifact."""
    result = dict(state)

    skeleton_raw = state.get("skeleton_artifact")
    if skeleton_raw is None:
        logger.info("SkeletonPass1: no skeleton_artifact — passthrough")
        return result

    # TypeScript bypass: if the project has .ts files, skip Python skeleton
    # generation entirely. TypeScript's type system handles interfaces inline
    # — the builder generates everything including types.
    code_files = state.get("code_files", {})
    file_manifest = state.get("file_manifest")
    all_filenames = list(code_files.keys())
    if file_manifest:
        if isinstance(file_manifest, dict):
            all_filenames.extend(file_manifest.get("files", {}).keys() if isinstance(file_manifest.get("files"), dict) else [])
        elif hasattr(file_manifest, "files"):
            all_filenames.extend(f.filename if hasattr(f, "filename") else str(f) for f in (file_manifest.files or []))
    has_typescript = any(f.endswith((".ts", ".tsx", ".jsx")) for f in all_filenames)
    if has_typescript:
        logger.info("SkeletonPass1: TypeScript project detected — skipping Python skeleton generation")
        return result

    try:
        from belief.models.skeleton import SkeletonArtifact
        from belief.models.symbol_registry import SymbolRegistry
        from belief.agents.skeleton_builder import generate_all_skeletons

        # Hydrate
        if isinstance(skeleton_raw, dict):
            skeleton = SkeletonArtifact.model_validate(skeleton_raw)
        else:
            skeleton = skeleton_raw

        # Force-promote database modules to skeleton=True when the
        # project uses SQLAlchemy. The architect's prune step sometimes
        # leaves `blog_engine/database.py` (and siblings) marked as
        # implementation files, which means my deterministic generator
        # never fires on them and the LLM builder writes them instead —
        # frequently with case-mismatched `settings.database_url`
        # against a `DATABASE_URL` config field, or missing `Base`/
        # `init_db` exports. Seen on every tier-4 build before this fix.
        _promote_database_files_to_skeleton(skeleton)

        # --- M2: Topological sort for build plan ---
        build_plan_summary = ""
        try:
            from belief.models.dependency_dag import create_build_plan
            build_plan = create_build_plan(skeleton)
            build_plan_summary = (
                f"{build_plan.total_skeleton_files} skeleton + "
                f"{build_plan.total_impl_files} impl, "
                f"{len(build_plan.impl_order)} parallel levels, "
                f"max parallelism: {build_plan.topo_result.max_parallelism()}"
            )
            logger.info(f"SkeletonPass1: build plan — {build_plan_summary}")
        except Exception as e:
            logger.debug(f"SkeletonPass1: DAG sort skipped: {e}")

        # --- M1: Generate skeleton files deterministically ---
        # Cache layer: skeleton generation is deterministic given the
        # skeleton spec, so the same goal-path pair always produces the
        # same files.  We cache at the (files, registry_context) level so
        # the identical build on a rerun skips the generator entirely.
        from belief.cache.skeleton_cache import get_or_generate_skeleton

        skeleton_spec = {
            "file_tree": [
                {
                    "path": e.path,
                    "role": getattr(e, "role", None),
                    "skeleton": bool(getattr(e, "skeleton", False)),
                }
                for e in skeleton.file_tree
            ],
            "external_dependencies": sorted(
                skeleton.external_dependencies or []
            ),
            "framework": getattr(skeleton, "framework", "") or "",
            "language": getattr(skeleton, "language", "python"),
        }

        def _generate() -> dict:
            """Fresh-generate path — only runs on cache miss."""
            local_registry = SymbolRegistry()
            generated_files = generate_all_skeletons(skeleton, local_registry)
            try:
                ctx = local_registry.full_registry_context()
            except Exception:
                ctx = ""
            return {
                "files": generated_files,
                "registry_context": ctx,
                "registry_file_count": len(local_registry.all_files()),
            }

        cached_payload, cache_hit = get_or_generate_skeleton(
            skeleton_spec, _generate
        )
        skeleton_files = dict(cached_payload.get("files", {}))
        registry_context = cached_payload.get("registry_context", "") or ""
        cached_file_count = int(cached_payload.get("registry_file_count", 0))
        if cache_hit:
            logger.info(
                f"SkeletonPass1: cache HIT — {len(skeleton_files)} skeleton files "
                f"({cached_file_count} registry entries)"
            )
        else:
            logger.info(
                f"SkeletonPass1: generated {len(skeleton_files)} skeleton files"
            )

        # --- M3: Compression-summary logging ---
        # Compression itself happens inside the generator path now (the
        # registry context is stored alongside the files).  We keep the
        # estimate_tokens telemetry for observability.
        compression_summary = ""
        try:
            from belief.models.context_compression import estimate_tokens

            full_tokens = estimate_tokens(registry_context)
            compression_summary = (
                f"registry={full_tokens} tokens, "
                f"{cached_file_count} files"
            )
            logger.info(f"SkeletonPass1: context compression — {compression_summary}")
        except ImportError:
            logger.debug("SkeletonPass1: context_compression not available")

        # Merge skeleton files into code_files
        code_files = dict(state.get("code_files", {}))
        code_files.update(skeleton_files)

        # Write results to state
        result["skeleton_files"] = skeleton_files
        result["code_files"] = code_files
        result["skeleton_registry_context"] = registry_context

        # Serialize skeleton_artifact back to dict for LangGraph state
        if isinstance(skeleton, SkeletonArtifact):
            result["skeleton_artifact"] = skeleton.model_dump()

        # Store build plan metadata in warnings for visibility
        if build_plan_summary:
            result.setdefault("warnings", list(state.get("warnings", [])))
            # Don't add as warning, just log it

        logger.info(
            f"SkeletonPass1 complete: {len(skeleton_files)} skeleton files, "
            f"{cached_file_count} registry entries"
        )

    except Exception as e:
        logger.warning(f"SkeletonPass1 failed: {e} — builder will generate without skeletons")
        result.setdefault("warnings", list(state.get("warnings", [])))
        result["warnings"].append(f"SkeletonPass1 failed: {e}")

    return result
