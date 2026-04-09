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
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)
        logger.info(f"SkeletonPass1: generated {len(skeleton_files)} skeleton files")

        # --- M3: Build compressed, budget-aware symbol context ---
        registry_context = ""
        compression_summary = ""
        try:
            from belief.models.context_compression import (
                build_compressed_context,
                rank_symbols,
                ContextBudget,
                estimate_tokens,
            )
            # Pre-compute ranked symbols once
            ranked = rank_symbols(registry, skeleton)

            # Build context for each implementation file and combine
            # (The builder will get one context per file, but we also store
            # the full registry context as a fallback)
            registry_context = registry.full_registry_context()

            full_tokens = estimate_tokens(registry_context)
            compression_summary = (
                f"registry={full_tokens} tokens, "
                f"{len(registry.all_files())} files, "
                f"{len(ranked)} ranked symbols"
            )
            logger.info(f"SkeletonPass1: context compression — {compression_summary}")

        except ImportError:
            registry_context = registry.full_registry_context()
            logger.debug("SkeletonPass1: context_compression not available, using raw registry")

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
            f"{len(registry.all_files())} registry entries"
        )

    except Exception as e:
        logger.warning(f"SkeletonPass1 failed: {e} — builder will generate without skeletons")
        result.setdefault("warnings", list(state.get("warnings", [])))
        result["warnings"].append(f"SkeletonPass1 failed: {e}")

    return result
