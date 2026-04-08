"""Language Adapter Pattern — Tier 6 Multi-Language Support.

Every language the Belief Engine can generate implements this ABC.
The engine stays language-agnostic; adapters handle scaffolding,
verification, testing, and AST parsing per language.

Usage:
    adapter = get_adapter("python")
    result = adapter.verify_code(code, project_path)
    exports = adapter.parse_exports(code, filename)
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("belief.languages")


class Language(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"


@dataclass
class VerificationResult:
    """Result of verifying generated code."""
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


@dataclass
class ExportedSymbol:
    """A public symbol exported from a source file."""
    name: str
    kind: str  # "class", "function", "variable", "interface", "struct", "type"
    file_path: str
    line: int = 0
    signature: str = ""  # e.g., "def process(data: RawData) -> EnrichedData"


class LanguageAdapter(ABC):
    """Abstract base for language-specific adapters.

    Each adapter knows how to scaffold, verify, test, and parse
    code in its target language.
    """

    @property
    @abstractmethod
    def language(self) -> Language:
        """Which language this adapter handles."""
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions for this language (e.g., ['.py'])."""
        ...

    @property
    @abstractmethod
    def test_file_patterns(self) -> list[str]:
        """Glob patterns for test files (e.g., ['test_*.py', '*_test.py'])."""
        ...

    @abstractmethod
    def scaffold_project(self, project_name: str, dependencies: list[str]) -> dict[str, str]:
        """Generate project scaffolding files (package.json, pyproject.toml, go.mod, etc).

        Returns dict of filename → content.
        """
        ...

    @abstractmethod
    def verify_code(self, code: str, filename: str) -> VerificationResult:
        """Verify a single file's syntax/types without executing it.

        For Python: ast.parse()
        For TypeScript: tsc --noEmit (or basic parse)
        For Go: go vet / syntax check
        """
        ...

    @abstractmethod
    def parse_exports(self, code: str, filename: str) -> list[ExportedSymbol]:
        """Extract public symbols (classes, functions, types) from a file.

        Used by the tester to generate accurate import statements.
        """
        ...

    @abstractmethod
    def get_system_prompt_additions(self) -> str:
        """Language-specific additions to the builder's system prompt.

        E.g., "Use TypeScript strict mode. Prefer interfaces over types."
        """
        ...

    @abstractmethod
    def get_import_statement(self, symbol: str, from_module: str) -> str:
        """Generate an import statement for this language.

        Python: "from module import Symbol"
        TypeScript: "import { Symbol } from './module'"
        Go: "import \"package/module\""
        """
        ...

    def is_test_file(self, filename: str) -> bool:
        """Check if a filename matches test file patterns."""
        import fnmatch
        base = filename.split("/")[-1]
        return any(fnmatch.fnmatch(base, pat) for pat in self.test_file_patterns)

    def is_source_file(self, filename: str) -> bool:
        """Check if a filename is a source file for this language."""
        return any(filename.endswith(ext) for ext in self.file_extensions)


# ── Adapter Registry ─────────────────────────────────────────────────────────

_adapters: dict[Language, LanguageAdapter] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    """Register a language adapter."""
    _adapters[adapter.language] = adapter


def get_adapter(language: str | Language) -> LanguageAdapter:
    """Get the adapter for a language. Defaults to Python."""
    if isinstance(language, str):
        try:
            language = Language(language.lower())
        except ValueError:
            language = Language.PYTHON

    if language not in _adapters:
        # Lazy-load Python adapter (always available)
        from belief.languages.python_adapter import PythonAdapter
        register_adapter(PythonAdapter())

    adapter = _adapters.get(language)
    if adapter is None:
        raise ValueError(f"No adapter registered for {language}. Available: {list(_adapters.keys())}")
    return adapter


def detect_language(filename: str) -> Language:
    """Detect language from a filename."""
    ext = Path(filename).suffix.lower()
    mapping = {
        ".py": Language.PYTHON,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".js": Language.TYPESCRIPT,  # JS uses the TS adapter
        ".jsx": Language.TYPESCRIPT,
        ".go": Language.GO,
    }
    return mapping.get(ext, Language.PYTHON)


def detect_project_languages(code_files: dict[str, str]) -> set[Language]:
    """Detect all languages used in a project."""
    languages = set()
    for fname in code_files:
        lang = detect_language(fname)
        languages.add(lang)
    return languages
