"""TypeScript Language Adapter — ESM-First Protocol Generation.

Handles TypeScript and JavaScript projects with strict ESM discipline
required for x402, MCP, A2A, and ERC-8004 SDKs.

Critical ESM covenants (from research):
1. __dirname is undefined in ESM — use import.meta.dirname (Node 20.11+)
2. File extensions mandatory — import './utils.js' even when source is utils.ts
3. require() doesn't exist in ESM — use dynamic import()
4. CJS cannot require ESM, but ESM can import CJS
5. Named imports from CJS may fail — use default import + destructure
6. JSON imports require assertion — import x from './x.json' with { type: 'json' }
7. Package exports field acts as whitelist — deep imports may break
8. Dual-loading hazard — same package as CJS+ESM creates two instances
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from belief.languages import (
    Language,
    LanguageAdapter,
    VerificationResult,
    ExportedSymbol,
)

logger = logging.getLogger("belief.languages.typescript")


# ── Protocol-specific dependency sets ────────────────────────────────────────

PROTOCOL_DEPS = {
    "x402": {
        "dependencies": {
            "@x402/express": "^2.3.0",
            "@x402/evm": "^2.5.0",
            "@x402/core": "^2.3.0",  # MANDATORY peer dep — x402/express won't resolve without it
            "express": "^5.1.0",
        },
        "devDependencies": {
            "@types/express": "^5.0.0",
            "supertest": "^7.0.0",
            "@types/supertest": "^6.0.0",
        },
    },
    "mcp": {
        "dependencies": {
            "@modelcontextprotocol/sdk": "^1.29.0",
            "zod": "^3.25.0",  # MANDATORY peer dep — SDK breaks without it. NOT zod v4.
            "express": "^5.1.0",
        },
        "devDependencies": {
            "@types/express": "^5.0.0",
        },
    },
    "a2a": {
        "dependencies": {
            "@a2a-js/sdk": "latest",
            "express": "^5.1.0",
        },
        "devDependencies": {
            "@types/express": "^5.0.0",
        },
    },
    "erc8004": {
        "dependencies": {
            "agent0-sdk": "^1.7.0",
            "ethers": "^6.16.0",  # v6 — NOT v5. Top-level imports, native bigint.
        },
    },
    "bittensor": {
        # Bittensor is Python — no TypeScript deps
    },
}


class TypeScriptAdapter(LanguageAdapter):
    """TypeScript/JavaScript language adapter with ESM-first discipline."""

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

    def scaffold_project(
        self,
        project_name: str,
        dependencies: list[str],
        protocols: list[str] | None = None,
    ) -> dict[str, str]:
        """Generate TypeScript project scaffolding with ESM-first config.

        Args:
            project_name: npm package name
            dependencies: additional npm packages
            protocols: optional list of protocol presets (x402, mcp, a2a, erc8004)
        """
        # Build dependency maps
        deps = {}
        dev_deps = {
            "typescript": "^5.7.0",
            "@types/node": "^22.0.0",
            "vitest": "^3.0.0",
            "tsx": "^4.0.0",
        }

        # Add protocol-specific deps
        for proto in (protocols or []):
            proto_deps = PROTOCOL_DEPS.get(proto, {})
            deps.update(proto_deps.get("dependencies", {}))
            dev_deps.update(proto_deps.get("devDependencies", {}))

        # Add explicit dependencies (pin known packages, use ^ for unknown)
        known_versions = {
            "express": "^5.0.0",
            "fastify": "^5.0.0",
            "hono": "^4.0.0",
            "zod": "^3.25.0",
            "ethers": "^6.0.0",
            "dotenv": "^16.0.0",
            "cors": "^2.8.5",
        }
        for dep in dependencies:
            if dep not in deps:
                deps[dep] = known_versions.get(dep, "*")

        # Auto-add type declarations and test utilities for detected frameworks
        if "express" in deps:
            dev_deps["@types/express"] = "^5.0.0"
            dev_deps["supertest"] = "^7.0.0"
            dev_deps["@types/supertest"] = "^6.0.0"
        if "cors" in deps:
            dev_deps["@types/cors"] = "^2.8.0"

        pkg = {
            "name": project_name,
            "version": "0.1.0",
            "type": "module",  # CRITICAL — ESM mode
            "private": True,
            "scripts": {
                "build": "tsc",
                "start": "node dist/index.js",
                "dev": "tsx watch src/index.ts",
                "test": "vitest run",
                "typecheck": "tsc --noEmit",
            },
            "dependencies": deps,
            "devDependencies": dev_deps,
        }

        # NodeNext resolution — required for MCP SDK, x402, and most modern packages
        # Node.js 24 native TS: erasableSyntaxOnly bans enum/namespace/parameter properties
        tsconfig = {
            "compilerOptions": {
                "target": "ESNext",
                "module": "NodeNext",
                "moduleResolution": "NodeNext",
                "strict": True,
                "verbatimModuleSyntax": True,
                "erasableSyntaxOnly": True,  # Node.js 24 native TS support
                "isolatedModules": True,
                "esModuleInterop": True,
                "resolveJsonModule": True,
                "noUncheckedIndexedAccess": True,
                "outDir": "./dist",
                "rootDir": "./src",
                "sourceMap": True,
                "declaration": True,
                "skipLibCheck": True,
            },
            "include": ["src/**/*.ts"],
            "exclude": ["node_modules", "dist"],
        }

        files = {
            "package.json": json.dumps(pkg, indent=2),
            "tsconfig.json": json.dumps(tsconfig, indent=2),
            "vitest.config.ts": (
                'import { defineConfig } from "vitest/config";\n\n'
                "export default defineConfig({\n"
                "  test: {\n"
                '    include: ["src/**/*.test.ts", "src/**/*.spec.ts"],\n'
                "    globals: false,\n"
                "  },\n"
                "});\n"
            ),
        }

        # Add .env.example for protocol configs
        env_lines = []
        if protocols:
            if "x402" in protocols:
                env_lines.extend([
                    "# x402 Payment Configuration",
                    "PAY_TO_ADDRESS=0xYourWalletAddress",
                    "CDP_API_KEY_ID=",
                    "CDP_API_KEY_SECRET=",
                    "X402_NETWORK=eip155:84532  # Base Sepolia testnet",
                    "X402_FACILITATOR=https://x402.org/facilitator",
                ])
            if "erc8004" in protocols:
                env_lines.extend([
                    "# ERC-8004 Identity",
                    "PRIVATE_KEY=0x...",
                    "PINATA_JWT=",
                    "CHAIN_ID=84532  # Base Sepolia",
                ])
        if env_lines:
            files[".env.example"] = "\n".join(env_lines) + "\n"

        # Add vitest config — zero-config but explicit for clarity
        # Research: vitest has moduleResolution conflicts with NodeNext,
        # so test files use the vitest transformer (no tsc needed for tests)
        files["vitest.config.ts"] = (
            'import { defineConfig } from "vitest/config";\n\n'
            "export default defineConfig({\n"
            "  test: {\n"
            '    include: ["src/**/*.test.ts", "tests/**/*.test.ts"],\n'
            "    globals: false,  // Require explicit imports from 'vitest'\n"
            "  },\n"
            "});\n"
        )

        return files

    def verify_code(self, code: str, filename: str) -> VerificationResult:
        """Verify TypeScript code for common ESM and type errors."""
        errors = []

        # ESM covenant checks
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Covenant 1: No __dirname in ESM
            if "__dirname" in stripped and "import.meta" not in stripped:
                errors.append(f"{filename}:{i}: __dirname is undefined in ESM — use import.meta.dirname")

            # Covenant 3: No require() in ESM
            if "require(" in stripped and not stripped.startswith("//") and "createRequire" not in code:
                errors.append(f"{filename}:{i}: require() is not available in ESM — use import()")

            # Covenant 2: Missing .js extension in relative imports
            if stripped.startswith("import ") and "from '" in stripped:
                match = re.search(r"from\s+'(\.[^']+)'", stripped)
                if match:
                    path = match.group(1)
                    if not path.endswith((".js", ".json", ".mjs", ".cjs")):
                        # Relative import without extension
                        errors.append(f"{filename}:{i}: ESM requires file extensions — use '{path}.js'")

            # Check for common TypeScript anti-patterns
            if ": any" in stripped and not stripped.startswith("//"):
                pass  # Warning only, not error

        # Brace matching
        open_braces = code.count("{") - code.count("}")
        if abs(open_braces) > 1:
            errors.append(f"{filename}: unmatched braces (diff={open_braces})")

        return VerificationResult(
            success=len(errors) == 0,
            errors=errors,
        )

    def parse_exports(self, code: str, filename: str) -> list[ExportedSymbol]:
        """Extract exported symbols from TypeScript code via regex."""
        exports = []

        patterns = [
            (r'export\s+interface\s+(\w+)', "interface"),
            (r'export\s+class\s+(\w+)', "class"),
            (r'export\s+(?:async\s+)?function\s+(\w+)', "function"),
            (r'export\s+(?:const|let|var)\s+(\w+)', "variable"),
            (r'export\s+type\s+(\w+)', "type"),
            (r'export\s+enum\s+(\w+)', "enum"),
            (r'export\s+default\s+(?:class|function)\s+(\w+)', "default"),
        ]

        for pattern, kind in patterns:
            for m in re.finditer(pattern, code):
                exports.append(ExportedSymbol(
                    name=m.group(1), kind=kind, file_path=filename,
                    signature=f"{kind} {m.group(1)}",
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
        return """TYPESCRIPT ESM GENERATION RULES (MANDATORY):
1. ALL imports must use ES module syntax (import/export), NEVER require().
2. Relative imports MUST include .js extension: import { foo } from './utils.js'
   (even though the source file is .ts — TypeScript resolves .js to .ts at compile time)
3. NEVER use __dirname or __filename — use import.meta.dirname (Node 20.11+)
4. package.json MUST include "type": "module"
5. tsconfig.json MUST use "module": "NodeNext" and "moduleResolution": "NodeNext"
6. Use Zod for runtime validation at system boundaries (API inputs, env vars)
7. Prefer discriminated unions over class hierarchies
8. Use unknown instead of any for truly unknown types
9. All async functions must have explicit return types
10. Export all public symbols — no default exports for library code"""

    def get_import_statement(self, symbol: str, from_module: str) -> str:
        module = from_module.replace(".ts", ".js").replace(".tsx", ".js")
        if not module.startswith(".") and not module.startswith("@"):
            module = f"./{module}"
        return f"import {{ {symbol} }} from '{module}';"
