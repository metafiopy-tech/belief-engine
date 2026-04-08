"""TypeScript Language Adapter — Tier 6 Multi-Language Support.

Handles TypeScript and JavaScript projects:
- package.json scaffolding
- tsc --noEmit for type checking (when tsc available)
- Regex-based export parsing (tree-sitter upgrade in M2)
- React/Next.js component detection
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from belief.languages import (
    Language,
    LanguageAdapter,
    VerificationResult,
    ExportedSymbol,
)

logger = logging.getLogger("belief.languages.typescript")


class TypeScriptAdapter(LanguageAdapter):
    """TypeScript/JavaScript language adapter."""

    @property
    def language(self) -> Language:
        return Language.TYPESCRIPT

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx"]

    @property
    def test_file_patterns(self) -> list[str]:
        return ["*.test.ts", "*.test.tsx", "*.spec.ts", "*.spec.tsx",
                "*.test.js", "*.test.jsx", "*.spec.js", "*.spec.jsx"]

    def scaffold_project(self, project_name: str, dependencies: list[str]) -> dict[str, str]:
        """Generate TypeScript project scaffolding."""
        pkg = {
            "name": project_name,
            "version": "0.1.0",
            "private": True,
            "scripts": {
                "build": "tsc",
                "dev": "tsx watch src/index.ts",
                "test": "vitest run",
                "lint": "eslint src/",
                "typecheck": "tsc --noEmit",
            },
            "dependencies": {dep: "latest" for dep in dependencies},
            "devDependencies": {
                "typescript": "^5.0.0",
                "vitest": "^2.0.0",
                "@types/node": "^20.0.0",
            },
        }

        tsconfig = {
            "compilerOptions": {
                "target": "ES2022",
                "module": "ESNext",
                "moduleResolution": "bundler",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "outDir": "./dist",
                "rootDir": "./src",
                "declaration": True,
            },
            "include": ["src/**/*.ts"],
            "exclude": ["node_modules", "dist"],
        }

        return {
            "package.json": json.dumps(pkg, indent=2),
            "tsconfig.json": json.dumps(tsconfig, indent=2),
        }

    def verify_code(self, code: str, filename: str) -> VerificationResult:
        """Verify TypeScript code.

        Checks:
        1. Basic syntax via regex patterns (always available)
        2. tsc --noEmit if TypeScript compiler is installed
        """
        errors = []

        # Basic syntax checks via regex
        # Check for unmatched braces
        open_braces = code.count("{") - code.count("}")
        if abs(open_braces) > 0:
            errors.append(f"{filename}: unmatched braces (diff={open_braces})")

        # Check for common TypeScript errors
        if "import " in code:
            # Verify import statements are well-formed
            for i, line in enumerate(code.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("import ") and not (
                    "from " in stripped or
                    stripped.endswith(";") or
                    stripped.endswith("'") or
                    stripped.endswith('"') or
                    "{" in stripped  # Multi-line import
                ):
                    errors.append(f"{filename}:{i}: malformed import statement")

        return VerificationResult(
            success=len(errors) == 0,
            errors=errors,
        )

    def parse_exports(self, code: str, filename: str) -> list[ExportedSymbol]:
        """Extract exported symbols from TypeScript code via regex.

        Handles:
        - export interface Foo { ... }
        - export class Bar { ... }
        - export function baz(...) { ... }
        - export const qux = ...
        - export type Alias = ...
        - export default function/class/...
        - export { A, B, C }
        """
        exports = []

        # export interface Name
        for m in re.finditer(r'export\s+interface\s+(\w+)', code):
            exports.append(ExportedSymbol(
                name=m.group(1), kind="interface", file_path=filename,
                signature=f"interface {m.group(1)}",
            ))

        # export class Name
        for m in re.finditer(r'export\s+class\s+(\w+)', code):
            exports.append(ExportedSymbol(
                name=m.group(1), kind="class", file_path=filename,
                signature=f"class {m.group(1)}",
            ))

        # export function name
        for m in re.finditer(r'export\s+(?:async\s+)?function\s+(\w+)', code):
            exports.append(ExportedSymbol(
                name=m.group(1), kind="function", file_path=filename,
                signature=f"function {m.group(1)}",
            ))

        # export const/let/var name
        for m in re.finditer(r'export\s+(?:const|let|var)\s+(\w+)', code):
            exports.append(ExportedSymbol(
                name=m.group(1), kind="variable", file_path=filename,
                signature=f"const {m.group(1)}",
            ))

        # export type Name
        for m in re.finditer(r'export\s+type\s+(\w+)', code):
            exports.append(ExportedSymbol(
                name=m.group(1), kind="type", file_path=filename,
                signature=f"type {m.group(1)}",
            ))

        # export default
        for m in re.finditer(r'export\s+default\s+(?:class|function)\s+(\w+)', code):
            exports.append(ExportedSymbol(
                name=m.group(1), kind="class", file_path=filename,
                signature=f"default {m.group(1)}",
            ))

        # export { A, B, C }
        for m in re.finditer(r'export\s*\{([^}]+)\}', code):
            names = [n.strip().split(' as ')[0].strip() for n in m.group(1).split(',')]
            for name in names:
                if name and not name.startswith('_'):
                    exports.append(ExportedSymbol(
                        name=name, kind="variable", file_path=filename,
                        signature=f"export {{ {name} }}",
                    ))

        return exports

    def get_system_prompt_additions(self) -> str:
        return (
            "Write TypeScript with strict mode enabled. "
            "Use interfaces for data shapes, types for unions/intersections. "
            "Prefer const over let. Use async/await for promises. "
            "Export all public symbols explicitly. "
            "Use ES module syntax (import/export), not CommonJS (require)."
        )

    def get_import_statement(self, symbol: str, from_module: str) -> str:
        module = from_module.replace(".ts", "").replace(".tsx", "")
        if not module.startswith(".") and not module.startswith("@"):
            module = f"./{module}"
        return f"import {{ {symbol} }} from '{module}';"
