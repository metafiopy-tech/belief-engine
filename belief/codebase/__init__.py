"""Codebase Ingestion — Tier 7 Brownfield Support.

Ingests an existing codebase and builds a structured representation:
  1. File tree with language detection
  2. Symbol index (classes, functions, exports per file)
  3. Dependency graph (who imports whom)
  4. Test suite discovery
  5. Compressed repo map for LLM context

This is the foundation for Agentless-style localization:
  repo → file → function → line

Usage:
    codebase = Codebase.from_directory("/path/to/repo")
    print(codebase.summary())
    relevant = codebase.localize("fix the login endpoint")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from belief.languages import (
    Language,
    detect_language,
    get_adapter,
    ExportedSymbol,
)

logger = logging.getLogger("belief.codebase")

# Files/directories to always skip
_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
    ".next",
    ".nuxt",
    "coverage",
    "htmlcov",
}

_SKIP_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

_MAX_FILE_SIZE = 100_000  # Skip files over 100KB


@dataclass
class FileInfo:
    """Metadata about a single file in the codebase."""

    path: str
    language: Language
    size: int
    line_count: int
    exports: list[ExportedSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # List of imported module names
    is_test: bool = False
    is_config: bool = False


@dataclass
class ImportEdge:
    """A dependency: source_file imports from target_file."""

    source: str
    target: str
    symbols: list[str] = field(default_factory=list)


@dataclass
class Codebase:
    """Structured representation of an existing codebase."""

    root: str
    files: dict[str, FileInfo] = field(default_factory=dict)
    edges: list[ImportEdge] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    languages: set[Language] = field(default_factory=set)

    @classmethod
    def from_directory(cls, path: str | Path) -> Codebase:
        """Ingest a codebase from a directory."""
        root = Path(path).resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        cb = cls(root=str(root))
        cb._scan_files(root)
        cb._build_dependency_graph()
        cb._discover_tests()
        logger.info(
            f"Codebase: {len(cb.files)} files, {len(cb.edges)} dependencies, "
            f"{len(cb.test_files)} tests, languages={[lang.value for lang in cb.languages]}"
        )
        return cb

    def _scan_files(self, root: Path) -> None:
        """Walk the directory tree and index all source files."""
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]

            rel_dir = os.path.relpath(dirpath, root)
            if rel_dir == ".":
                rel_dir = ""

            for fname in filenames:
                if fname in _SKIP_FILES or fname.startswith("."):
                    continue

                full_path = Path(dirpath) / fname
                rel_path = os.path.join(rel_dir, fname) if rel_dir else fname

                # Skip large files
                try:
                    size = full_path.stat().st_size
                except OSError:
                    continue
                if size > _MAX_FILE_SIZE:
                    continue

                lang = detect_language(fname)

                # Read content
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                line_count = content.count("\n") + 1

                # Parse exports via language adapter
                try:
                    adapter = get_adapter(lang)
                    exports = adapter.parse_exports(content, rel_path)
                except Exception:
                    exports = []

                # Extract imports
                imports = _extract_imports(content, lang)

                # Detect test/config files
                is_test = _is_test_file(rel_path, lang)
                is_config = _is_config_file(rel_path)

                info = FileInfo(
                    path=rel_path,
                    language=lang,
                    size=size,
                    line_count=line_count,
                    exports=exports,
                    imports=imports,
                    is_test=is_test,
                    is_config=is_config,
                )
                self.files[rel_path] = info
                self.languages.add(lang)

                if is_test:
                    self.test_files.append(rel_path)
                if is_config:
                    self.config_files.append(rel_path)

    def _build_dependency_graph(self) -> None:
        """Build import edges between files."""
        # Map module names to file paths
        module_map: dict[str, str] = {}
        for fpath in self.files:
            if fpath.endswith(".py"):
                module = fpath.replace("/", ".").replace(".py", "")
                module_map[module] = fpath
                # Also map the last component (e.g., "models" → "app/models.py")
                parts = module.split(".")
                if len(parts) > 1:
                    module_map[parts[-1]] = fpath
            elif fpath.endswith((".ts", ".tsx", ".js", ".jsx")):
                # TypeScript: map relative paths
                base = fpath.rsplit(".", 1)[0]
                module_map[base] = fpath

        for fpath, info in self.files.items():
            for imp in info.imports:
                # Resolve the import to a file
                target = module_map.get(imp)
                if not target:
                    # Try sub-module resolution
                    parts = imp.split(".")
                    for i in range(len(parts), 0, -1):
                        candidate = ".".join(parts[:i])
                        if candidate in module_map:
                            target = module_map[candidate]
                            break

                if target and target != fpath:
                    self.edges.append(
                        ImportEdge(
                            source=fpath,
                            target=target,
                            symbols=[],  # Could extract specific symbols
                        )
                    )

    def _discover_tests(self) -> None:
        """Discover test files and test runners."""
        # Already collected during _scan_files
        # Also look for CI config
        ci_patterns = [
            ".github/workflows",
            "Makefile",
            "tox.ini",
            "pytest.ini",
            "setup.cfg",
            "pyproject.toml",
        ]
        for fpath in self.files:
            for pat in ci_patterns:
                if pat in fpath:
                    if fpath not in self.config_files:
                        self.config_files.append(fpath)

    # ── Localization (Agentless-style) ──────────────────────────────────

    def localize_files(self, query: str, max_files: int = 5) -> list[str]:
        """Agentless Phase 1: Identify the most relevant files for a query.

        Uses keyword matching + dependency proximity to rank files.
        Returns the top N file paths.
        """
        scores: dict[str, float] = {}
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))

        for fpath, info in self.files.items():
            if info.is_test or info.is_config:
                continue

            score = 0.0

            # 1. Filename match
            fname_words = set(re.findall(r"\w+", fpath.lower()))
            overlap = query_words & fname_words
            score += len(overlap) * 5.0

            # 2. Export name match
            for exp in info.exports:
                exp_words = set(re.findall(r"\w+", exp.name.lower()))
                overlap = query_words & exp_words
                score += len(overlap) * 3.0

            # 3. Size penalty (prefer smaller, more focused files)
            if info.line_count > 200:
                score *= 0.8

            if score > 0:
                scores[fpath] = score

        # Sort by score descending
        ranked = sorted(scores.keys(), key=lambda f: scores[f], reverse=True)
        return ranked[:max_files]

    def localize_functions(
        self, file_path: str, query: str, max_functions: int = 3
    ) -> list[ExportedSymbol]:
        """Agentless Phase 2: Within a file, find the most relevant functions/classes."""
        info = self.files.get(file_path)
        if not info:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))

        scored = []
        for exp in info.exports:
            exp_words = set(re.findall(r"\w+", exp.name.lower()))
            sig_words = set(re.findall(r"\w+", exp.signature.lower()))
            score = len(query_words & exp_words) * 3.0 + len(query_words & sig_words) * 1.0
            if score > 0:
                scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:max_functions]]

    def get_file_content(self, file_path: str) -> str:
        """Read a file's content from disk."""
        full = Path(self.root) / file_path
        try:
            return full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def get_dependents(self, file_path: str) -> list[str]:
        """Files that import from this file (will be affected by changes)."""
        return [e.source for e in self.edges if e.target == file_path]

    def get_dependencies(self, file_path: str) -> list[str]:
        """Files this file imports from."""
        return [e.target for e in self.edges if e.source == file_path]

    def get_affected_tests(self, file_path: str) -> list[str]:
        """Test files that might be affected by changes to this file."""
        affected = set()
        # Direct test files for this module
        base = file_path.replace("/", "_").replace(".py", "")
        for tf in self.test_files:
            if base in tf or file_path.split("/")[-1].replace(".py", "") in tf:
                affected.add(tf)

        # Tests that import from this file
        for edge in self.edges:
            if (
                edge.target == file_path
                and self.files.get(edge.source, FileInfo("", Language.PYTHON, 0, 0)).is_test
            ):
                affected.add(edge.source)

        return list(affected)

    # ── Repo Map ────────────────────────────────────────────────────────

    def generate_repo_map(self, max_tokens: int = 2000) -> str:
        """Generate a compressed repo map for LLM context.

        Shows file paths + top exports, budgeted to fit within max_tokens.
        Prioritizes files with more dependencies (more important).
        """
        # Score files by importance (number of dependents)
        dep_counts: dict[str, int] = {}
        for edge in self.edges:
            dep_counts[edge.target] = dep_counts.get(edge.target, 0) + 1

        # Sort: most-depended-on files first
        ranked = sorted(
            self.files.keys(),
            key=lambda f: dep_counts.get(f, 0),
            reverse=True,
        )

        lines = []
        char_count = 0
        char_budget = max_tokens * 4  # ~4 chars per token

        for fpath in ranked:
            info = self.files[fpath]
            if info.is_test or info.is_config:
                continue

            # File header
            header = f"{fpath} ({info.line_count} lines)"

            # Top exports
            export_lines = []
            for exp in info.exports[:8]:
                export_lines.append(f"  {exp.signature}")

            block = header + "\n" + "\n".join(export_lines) if export_lines else header
            block_len = len(block)

            if char_count + block_len > char_budget:
                break

            lines.append(block)
            char_count += block_len

        return "\n\n".join(lines)

    def summary(self) -> str:
        """One-line summary of the codebase."""
        total_lines = sum(f.line_count for f in self.files.values())
        return (
            f"{len(self.files)} files, {total_lines:,} lines, "
            f"{len(self.edges)} dependencies, {len(self.test_files)} tests, "
            f"languages: {', '.join(lang.value for lang in sorted(self.languages, key=lambda x: x.value))}"
        )


# ── Import extraction ────────────────────────────────────────────────────────


def _extract_imports(code: str, language: Language) -> list[str]:
    """Extract imported module names from source code."""
    imports = []

    if language == Language.PYTHON:
        # from X import Y → X
        for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", code, re.MULTILINE):
            imports.append(m.group(1))
        # import X → X
        for m in re.finditer(r"^\s*import\s+([\w.]+)", code, re.MULTILINE):
            imports.append(m.group(1))

    elif language == Language.TYPESCRIPT:
        # import { X } from 'Y' or import X from 'Y'
        for m in re.finditer(r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""", code):
            mod = m.group(1)
            if mod.startswith("."):
                imports.append(mod.lstrip("./"))
            else:
                imports.append(mod)

    elif language == Language.GO:
        # import "package/name"
        for m in re.finditer(r'"([^"]+)"', code):
            imports.append(m.group(1))

    return imports


def _is_test_file(path: str, language: Language) -> bool:
    """Check if a file is a test file."""
    base = path.split("/")[-1]
    if language == Language.PYTHON:
        return (
            base.startswith("test_")
            or base.endswith("_test.py")
            or "/tests/" in path
            or path.startswith("tests/")
        )
    if language == Language.TYPESCRIPT:
        return ".test." in base or ".spec." in base or "/tests/" in path or "/__tests__/" in path
    if language == Language.GO:
        return base.endswith("_test.go")
    return False


def _is_config_file(path: str) -> bool:
    """Check if a file is a config/build file."""
    config_names = {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env",
        ".env.example",
        "package.json",
        "tsconfig.json",
        "go.mod",
        "go.sum",
        "requirements.txt",
        "Pipfile",
        "tox.ini",
        "pytest.ini",
        ".gitignore",
        ".dockerignore",
        "README.md",
        "LICENSE",
    }
    return path.split("/")[-1] in config_names
