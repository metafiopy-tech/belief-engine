"""Builder Agent — Skeleton-Aware (Milestone 1)

When a SkeletonArtifact exists:
- Skips skeleton files (already generated in Pass 1)
- Uses compressed symbol registry context instead of file summaries
- Generates implementation files against typed contracts

When no skeleton exists (fallback):
- Behaves exactly like the old builder (file summaries as context)
"""

from __future__ import annotations

import logging
import re

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.state import Phase, UnifiedState

logger = logging.getLogger("belief.agents.builder")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BUILDER_SYSTEM = """You are the Builder Agent. Write complete, working code.
Generate each file fully — no placeholders, no TODOs, no "implement this".
Every function must have a real implementation. Handle errors gracefully.
Match the language of the file being generated (Python for .py, TypeScript for .ts/.tsx).

PYTHON COVENANTS:
- When using SQLAlchemy 2.x with Mapped/mapped_column, do NOT use
  `from __future__ import annotations` — it breaks ORM type resolution.
- Always use SQLAlchemy 2.x style: `Mapped[str]`, not `str`.

TYPESCRIPT COVENANTS (violations crash the build):
- Relative imports MUST have .js extension: import { foo } from './utils.js'
- NEVER use __dirname — use import.meta.dirname
- NEVER use require() — use import
- x402: ExactEvmScheme from '@x402/evm/exact/server', NOT '@x402/evm'
- x402: HTTPFacilitatorClient from '@x402/core/server', NOT '@x402/core'
- x402: @x402/types and @x402/client DO NOT EXIST
- MCP: NEVER bare '@modelcontextprotocol/sdk' — use subpaths with .js
- MCP: zod@^3.25 is mandatory peer dep
- ethers v6: top-level imports from 'ethers', NEVER ethers.providers.* or @ethersproject/*
- ethers v6: native bigint, NOT BigNumber. parseLog() returns null — always null-check.
- Express 5: wildcard '/{*splat}' not '*', error handlers use ErrorRequestHandler type
- Vitest: import { describe, it, expect, vi } from 'vitest', vi.fn() NOT jest.fn()"""

BUILDER_PROMPT_LEGACY = """Write the code for this file:

GOAL: {goal}
FILE: {filename}
PURPOSE: {purpose}
PUBLIC INTERFACE: {public_interface}
DEPENDENCIES: {depends_on}
IS ENTRY POINT: {is_entry_point}

ARCHITECTURE NOTES:
{architecture_notes}

OTHER FILES IN THIS PROJECT:
{other_files_summary}

{gap_context}

Write complete, working Python code for {filename}.
Output ONLY the raw code — no markdown fences, no explanation.
Include all imports, all functions, all error handling.
If this is the entry point, include an if __name__ == "__main__" block."""

BUILDER_PROMPT_SKELETON = """Write the code for this file:

GOAL: {goal}
FILE: {filename}
PURPOSE: {purpose}
DEPENDENCIES: {depends_on}
IS ENTRY POINT: {is_entry_point}

## Symbol Context (importable signatures from skeleton files)
```python
{symbol_context}
```

## Skeleton Code (typed interfaces this file implements)
```python
{skeleton_code}
```

ARCHITECTURE NOTES:
{architecture_notes}

{gap_context}

CRITICAL RULES:
1. Import from the exact module paths shown in the symbol context.
2. Match ALL method signatures exactly — same parameter names, types, return types.
3. Do NOT redefine any class that already exists in the skeleton files.
4. If implementing an ABC, inherit from it and implement every abstract method.
5. Include error handling with the project's custom exceptions.

Write complete, working Python code for {filename}.
Output ONLY the raw code — no markdown fences, no explanation."""


class BuilderAgent(BaseAgent):
    role = ModelRole.BUILDER
    name = "Builder"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.BUILDING
        manifest = state.file_manifest

        if not manifest or not manifest.files:
            state.errors.append("Builder: no file manifest")
            state.phase = Phase.TESTING
            return state

        # ── Inject covenants into builder system prompt ──
        # The recomposer puts nutrient_context into state, which includes
        # covenants (immutable rules). The builder MUST see these.
        builder_system = BUILDER_SYSTEM
        nutrient_ctx = getattr(state, "nutrient_context", "") or ""
        if not nutrient_ctx and hasattr(state, "__dict__"):
            # State might be a dict-like
            nutrient_ctx = ""

        # Extract covenants from the nutrient context block
        covenant_lines = []
        if nutrient_ctx:
            for line in nutrient_ctx.split("\n"):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                line_upper = line_stripped.upper()
                # Match: "COVENANT:", "- COVENANT:", "## COVENANTS", or lines containing "IMMUTABLE"
                if (
                    "COVENANT" in line_upper
                    and (
                        line_stripped.startswith("-")
                        or line_stripped.startswith("COVENANT")
                        or ":" in line_stripped
                    )
                ) or "IMMUTABLE" in line_upper:
                    covenant_lines.append(line_stripped)

        if covenant_lines:
            covenant_block = "\n".join(covenant_lines)
            builder_system = (
                f"{BUILDER_SYSTEM}\n\n"
                f"MANDATORY RULES (covenants from institutional memory):\n"
                f"{covenant_block}\n"
                f"These rules are NON-NEGOTIABLE. Violating them causes build failures."
            )
            logger.info(f"Builder: injected {len(covenant_lines)} covenants into system prompt")

        # Store for _generate_one_file to use
        self._builder_system = builder_system

        # Detect skeleton mode
        has_skeleton = bool(state.skeleton_artifact and state.skeleton_registry_context)
        skeleton_file_set = set(state.skeleton_files.keys()) if state.skeleton_files else set()

        llm = LLMClient(self.router)
        code_files: dict[str, str] = dict(state.code_files)

        # Gap context for re-builds
        gap_context = ""
        if state.gap_report and state.gap_report.gaps:
            gap_context = "## GAPS FROM PREVIOUS BUILD (fix these)\n" + "\n".join(
                f"- [{g.severity.value}] {g.description}: {g.suggested_fix}"
                for g in state.gap_report.gaps[:5]
            )

        # ── M1: Build dependency DAG for correct ordering ──
        try:
            from belief.agents.parallel_planner import build_plan_from_manifest

            build_plan = build_plan_from_manifest(manifest)
            logger.info(
                f"Builder: DAG ordering — {len(build_plan.levels)} levels, "
                f"{build_plan.total_files} files"
            )
        except Exception as e:
            logger.debug(f"Builder: DAG ordering failed ({e}), using manifest order")
            from belief.agents.parallel_planner import build_plan_from_files

            build_plan = build_plan_from_files([f.filename for f in manifest.files])

        file_spec_map = {f.filename: f for f in manifest.files}

        # ── M2: Generate files level by level, parallel within each level ──
        # Token bucket rate limiter prevents 429 errors during parallel generation
        import asyncio

        try:
            from belief.hardening import AsyncTokenBucket

            rate_limiter = AsyncTokenBucket(rate=0.8, burst=5)  # ~48 RPM with burst of 5
        except ImportError:
            rate_limiter = None
        sem = asyncio.Semaphore(6)  # Concurrency cap as fallback

        try:
            for level in build_plan.levels:
                # Filter files at this level
                files_to_build = []
                for filename in level.files:
                    file_spec = file_spec_map.get(filename)
                    if file_spec is None:
                        continue
                    if file_spec.filename in ("README.md", "requirements.txt"):
                        continue
                    if file_spec.filename in skeleton_file_set:
                        continue
                    files_to_build.append(file_spec)

                if not files_to_build:
                    continue

                if len(files_to_build) == 1:
                    # Single file — no parallelism overhead
                    file_spec = files_to_build[0]
                    code = await self._generate_one_file(
                        llm,
                        file_spec,
                        state,
                        manifest,
                        code_files,
                        has_skeleton,
                        gap_context,
                    )
                    if code:
                        code_files[file_spec.filename] = code
                        logger.info(f"Builder: wrote {file_spec.filename} ({len(code)} chars)")
                else:
                    # Multiple files at this level — generate in parallel
                    logger.info(
                        f"Builder: level {level.level} — generating "
                        f"{len(files_to_build)} files in parallel"
                    )

                    async def _gen_with_sem(fs):
                        async with sem:
                            if rate_limiter:
                                await rate_limiter.acquire()
                            return fs.filename, await self._generate_one_file(
                                llm,
                                fs,
                                state,
                                manifest,
                                code_files,
                                has_skeleton,
                                gap_context,
                            )

                    results = await asyncio.gather(
                        *[_gen_with_sem(fs) for fs in files_to_build],
                        return_exceptions=True,
                    )

                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning(f"Builder: parallel gen failed: {result}")
                            continue
                        fname, code = result
                        if code:
                            code_files[fname] = code
                            logger.info(f"Builder: wrote {fname} ({len(code)} chars)")

        except (ConnectionError, ValueError) as e:
            logger.error(f"Builder failed: {e}")
            state.errors.append(f"Builder failed: {e}")
        finally:
            await llm.close()

        # Generate requirements.txt from actual imports
        code_files["requirements.txt"] = _extract_requirements(code_files)

        state.code_files = code_files
        state.phase = Phase.TESTING
        return state

    async def _generate_one_file(
        self,
        llm,
        file_spec,
        state,
        manifest,
        code_files,
        has_skeleton: bool,
        gap_context: str,
    ) -> str | None:
        """Generate a single file. Used by both serial and parallel paths."""
        # ── M3: Build repo map from completed files for context ──
        repo_map_context = ""
        if len(code_files) > 3:
            try:
                from belief.agents.repo_map import RepoMap

                repo_map = RepoMap.from_code_files(code_files)
                repo_map_context = repo_map.format_for_file(
                    target_file=file_spec.filename,
                    dependencies=file_spec.depends_on,
                    max_tokens=3000,
                )
            except Exception as e:
                logger.debug(f"Builder: repo map failed ({e})")

        # ── Move 5: Inject API contract so builder knows what to implement ──
        contract_context = ""
        skeleton = state.skeleton_artifact
        if skeleton:
            from belief.models.skeleton import SkeletonArtifact as SA

            if isinstance(skeleton, dict):
                try:
                    skeleton = SA.model_validate(skeleton)
                except Exception:
                    skeleton = None
            if skeleton and hasattr(skeleton, "format_contract"):
                contract = skeleton.format_contract()
                if contract:
                    contract_context = f"\n{contract}\n"
                    repo_map_context = (
                        repo_map_context + contract_context
                        if repo_map_context
                        else contract_context
                    )

        # Choose prompt based on skeleton mode
        if has_skeleton:
            prompt = self._build_skeleton_prompt(
                file_spec,
                state,
                code_files,
                gap_context,
                repo_map_context=repo_map_context,
            )
        else:
            prompt = self._build_legacy_prompt(
                file_spec,
                manifest,
                state,
                gap_context,
                repo_map_context=repo_map_context,
            )

        try:
            # Add language-specific system prompt additions
            system = getattr(self, "_builder_system", BUILDER_SYSTEM)
            try:
                from belief.languages import detect_language, get_adapter

                lang = detect_language(file_spec.filename)
                adapter = get_adapter(lang)
                lang_additions = adapter.get_system_prompt_additions()
                if lang_additions:
                    system = f"{system}\n\nLANGUAGE: {lang.value.upper()}\n{lang_additions}"
            except Exception as e:
                logger.debug(f"Language adapter failed for {file_spec.filename}: {e}")

            # Session 2 (v3.2): inject Pydantic v2 cheatsheet when this
            # file will touch pydantic/fastapi/langchain/BaseSettings.
            # Append-only to preserve the Session 1 num_keep=512 prefix
            # cache; ``_trigger_*`` checks live in prompts/cheatsheets.py.
            try:
                from belief.prompts.cheatsheets import (
                    pydantic_v2_cheatsheet_for_file_spec,
                )

                cheatsheet = pydantic_v2_cheatsheet_for_file_spec(
                    file_spec,
                    user_goal=state.user_goal if state.user_goal else None,
                )
                if cheatsheet:
                    system = f"{system}{cheatsheet}"
                    logger.debug(
                        "Builder: injected Pydantic v2 cheatsheet for %s",
                        file_spec.filename,
                    )
            except Exception as e:  # pragma: no cover
                logger.debug(
                    "Pydantic v2 cheatsheet injection failed for %s: %s",
                    file_spec.filename,
                    e,
                )

            # Inject protocol skeleton as reference for TypeScript builds
            if file_spec.filename.endswith((".ts", ".tsx")):
                try:
                    from belief.prompts.protocol_skeletons import get_skeleton

                    goal_lower = state.user_goal.lower() if state.user_goal else ""
                    protocol_map = {
                        "x402": ["x402", "payment", "micropayment", "paywall"],
                        "mcp": ["mcp", "model context protocol", "mcp server", "mcp tool"],
                        "a2a": ["a2a", "agent2agent", "agent card", "agent-to-agent"],
                        "erc8004": ["erc-8004", "erc8004", "agent identity", "agent registration"],
                    }

                    # Stage 1: Inject current API documentation (v0-style)
                    from belief.validators.typescript_fixup import get_api_docs_for_goal

                    api_docs = get_api_docs_for_goal(goal_lower)
                    if api_docs:
                        prompt = f"CURRENT API DOCUMENTATION (use these exact patterns):\n{api_docs}\n\n{prompt}"

                    for proto, keywords in protocol_map.items():
                        if any(kw in goal_lower for kw in keywords):
                            skeleton = get_skeleton(proto)
                            for skel_fname, skel_content in skeleton.items():
                                if file_spec.filename.endswith(skel_fname.split("/")[-1]):
                                    prompt = (
                                        f"REFERENCE SKELETON (use this exact pattern, adapt to the goal):\n"
                                        f"```typescript\n{skel_content[:2000]}\n```\n\n{prompt}"
                                    )
                                    break
                            break
                except Exception as e:
                    logger.debug(
                        f"Protocol skeleton injection failed for {file_spec.filename}: {e}"
                    )

            raw, truncated = await llm.generate_long_text(
                role=self.role,
                system=system,
                prompt=prompt,
                temperature=0.2,
                complexity=state.complexity_score,
            )
            if truncated:
                # The model hit the output ceiling even after continuation —
                # storing the partial would ship a file cut mid-statement
                # (the bug that produced 8/23 non-parsing files). Drop it and
                # let the missing-file path / compile gate flag it instead.
                logger.warning(
                    "Builder: %s still truncated after continuation cap — discarding partial file",
                    file_spec.filename,
                )
                return None
            return _strip_fences(raw)
        except Exception as e:
            logger.warning(f"Builder: failed to generate {file_spec.filename}: {e}")
            return None

    def _build_skeleton_prompt(
        self, file_spec, state, code_files, gap_context, repo_map_context: str = ""
    ):
        """Build prompt using compressed symbol context (M1+M3 path).

        Uses per-file budget-aware context compression when available,
        falls back to full registry context.
        """
        # Find skeleton code for this file's dependencies
        skeleton_code = "# No skeleton file for this implementation"
        if state.skeleton_files:
            for dep in file_spec.depends_on or []:
                if dep in state.skeleton_files:
                    skeleton_code = state.skeleton_files[dep]
                    break

        # Try per-file compressed context (M3)
        symbol_context = state.skeleton_registry_context  # fallback
        try:
            from belief.models.skeleton import SkeletonArtifact
            from belief.models.symbol_registry import SymbolRegistry
            from belief.models.context_compression import build_compressed_context

            skeleton = state.skeleton_artifact
            if isinstance(skeleton, dict):
                skeleton = SkeletonArtifact.model_validate(skeleton)

            if skeleton:
                # Rebuild registry from skeleton files
                registry = SymbolRegistry()
                for path, code in (state.skeleton_files or {}).items():
                    try:
                        registry.register_source(code, path)
                    except SyntaxError:
                        pass
                # Also register any already-built impl files
                for path, code in code_files.items():
                    if path not in (state.skeleton_files or {}):
                        try:
                            registry.register_source(code, path)
                        except SyntaxError:
                            pass

                compressed = build_compressed_context(
                    file_path=file_spec.filename,
                    skeleton=skeleton,
                    registry=registry,
                )
                symbol_context = compressed.context_text
                logger.debug(
                    f"Builder: compressed context for {file_spec.filename}: "
                    f"{compressed.token_count} tokens, "
                    f"{compressed.symbols_included}/{compressed.symbols_total} symbols"
                )
        except Exception as e:
            logger.debug(f"Builder: per-file compression failed ({e}), using full registry")

        return BUILDER_PROMPT_SKELETON.format(
            goal=state.requirement_spec.goal if state.requirement_spec else state.user_goal,
            filename=file_spec.filename,
            purpose=file_spec.purpose,
            depends_on=", ".join(file_spec.depends_on) if file_spec.depends_on else "none",
            is_entry_point=file_spec.is_entry_point,
            symbol_context=symbol_context,
            skeleton_code=skeleton_code,
            architecture_notes=state.file_manifest.architecture_notes
            if state.file_manifest
            else "",
            gap_context=gap_context,
        )

    def _build_legacy_prompt(
        self, file_spec, manifest, state, gap_context, repo_map_context: str = ""
    ):
        """Build prompt using old file summaries (fallback path)."""
        if repo_map_context:
            other_files = repo_map_context
        else:
            other_files = "\n".join(
                f"  - {f.filename}: {f.purpose} (exports: {f.public_interface})"
                for f in manifest.files
                if f.filename != file_spec.filename
            )
        return BUILDER_PROMPT_LEGACY.format(
            goal=state.requirement_spec.goal if state.requirement_spec else state.user_goal,
            filename=file_spec.filename,
            purpose=file_spec.purpose,
            public_interface=file_spec.public_interface,
            depends_on=", ".join(file_spec.depends_on) if file_spec.depends_on else "none",
            is_entry_point=file_spec.is_entry_point,
            architecture_notes=manifest.architecture_notes,
            other_files_summary=other_files,
            gap_context=gap_context,
        )


def _strip_fences(raw: str) -> str:
    code = raw.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code


# ---------------------------------------------------------------------------
# Requirements extraction (unchanged from original)
# ---------------------------------------------------------------------------

_STDLIB = frozenset(
    {
        "__future__",
        "abc",
        "argparse",
        "ast",
        "asyncio",
        "base64",
        "collections",
        "configparser",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "email",
        "ftplib",
        "functools",
        "getpass",
        "glob",
        "hashlib",
        "hmac",
        "html",
        "http",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "locale",
        "logging",
        "math",
        "mimetypes",
        "multiprocessing",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "pprint",
        "queue",
        "random",
        "re",
        "secrets",
        "shutil",
        "signal",
        "site",
        "socket",
        "sqlite3",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "typing",
        "typing_extensions",
        "unicodedata",
        "unittest",
        "urllib",
        "uuid",
        "venv",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
        "zlib",
        # Test and project-local modules — never real pip packages
        "pytest",
        "pytest_asyncio",
        "conftest",
        "test",
        "tests",
        "setup",
        "server",
        "app",
        "main",
        "config",
        "utils",
        "helpers",
        "models",
        "run",
        "tools",
        "client",
        "handler",
        "worker",
        "task",
        "core",
        "api",
        "pipeline",
        "schemas",
        "crud",
        "database",
        "exceptions",
        "routes",
        "seed",
        "src",
    }
)

_IMPORT_TO_PIP = {
    "fastmcp": "fastmcp",
    "httpx": "httpx",
    "requests": "requests",
    "flask": "flask",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "aiohttp": "aiohttp",
    "selenium": "selenium",
    "playwright": "playwright",
    "chromadb": "chromadb",
    "ollama": "ollama",
    "anthropic": "anthropic",
    "openai": "openai",
    "rich": "rich",
    "click": "click",
    "typer": "typer",
    "numpy": "numpy",
    "pandas": "pandas",
    "googlemaps": "googlemaps",
    "dns": "dnspython",
    "tenacity": "tenacity",
    "cachetools": "cachetools",
    "pydantic_settings": "pydantic-settings",
    "sqlalchemy": "sqlalchemy",
    "alembic": "alembic",
    "jinja2": "jinja2",
    "markdown": "markdown",
    "starlette": "starlette",
    "celery": "celery",
    "redis": "redis",
    "motor": "motor",
    "watchdog": "watchdog",
    "GitPython": "gitpython",
    "gitpython": "gitpython",
    # Common mismatches where import name ≠ pip name
    "github": "PyGithub",
    "dateutil": "python-dateutil",
    "jose": "python-jose",
    "multipart": "python-multipart",
    "jwt": "PyJWT",
    "attr": "attrs",
    "serial": "pyserial",
    "usb": "pyusb",
    "nacl": "PyNaCl",
    "Crypto": "pycryptodome",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "skimage": "scikit-image",
    "wx": "wxPython",
    "telegram": "python-telegram-bot",
    "magic": "python-magic",
    "gi": "PyGObject",
}


def _extract_requirements(code_files: dict[str, str]) -> str:
    third_party = set()
    for fname, content in code_files.items():
        if not fname.endswith(".py"):
            continue
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("import "):
                parts = line.split()
                if len(parts) >= 2:
                    mod = parts[1].split(".")[0].split(",")[0]
                    if mod not in _STDLIB and mod.isidentifier():
                        third_party.add(mod)
            elif line.startswith("from ") and " import " in line:
                mod = line.split()[1].split(".")[0]
                if mod not in _STDLIB and mod.isidentifier():
                    third_party.add(mod)

    # Filter project-local modules
    project_files = {f.replace(".py", "").replace("/", ".").split(".")[0] for f in code_files}
    third_party -= project_files

    deps = sorted({_IMPORT_TO_PIP.get(m, m) for m in third_party})
    if not deps:
        return "# No external dependencies\n"
    return "\n".join(deps) + "\n"
