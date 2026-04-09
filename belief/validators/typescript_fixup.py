"""TypeScript Fixup Pipeline — Three-Stage v0-Style Error Correction.

Inspired by Vercel v0's approach (reduced error rates by double digits):
  Stage 1: API doc injection into builder prompts (pre-generation)
  Stage 2: Streaming fixup — find-and-replace patterns on raw output (during generation)
  Stage 3: Deterministic autofixers + covenant enforcement (post-generation)

The key insight from v0: "Your product's moat cannot be your system prompt."
The moat is the fixup pipeline that deterministically corrects LLM output.

Node.js 24 compatibility:
  - Native TypeScript type-stripping (no ts-node needed)
  - Requires erasableSyntaxOnly: no enum, no namespace, no parameter properties
  - Requires verbatimModuleSyntax: must use 'import type' for type-only imports

Usage:
    from belief.validators.typescript_fixup import fixup_typescript_output
    fixed_files = fixup_typescript_output(code_files, goal="x402 payment API")
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("belief.validators.ts_fixup")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: API Documentation for Prompt Injection
# ═══════════════════════════════════════════════════════════════════════════════

# Current API surfaces — injected into builder prompts when protocol detected.
# These are the MINIMUM correct patterns that prevent the most common LLM errors.

API_DOCS = {
    "x402": """## @x402/express V2 API (current)
CORRECT imports:
  import { paymentMiddleware } from "@x402/express";
  import { ExactEvmScheme } from "@x402/evm/exact/server";
  import { HTTPFacilitatorClient } from "@x402/core/server";

CORRECT paymentMiddleware signature:
  app.use(paymentMiddleware(routesConfig, resourceServer));
  // routesConfig: { "GET /path": { accepts: [{ scheme, price, network, payTo }] } }
  // resourceServer: new x402ResourceServer(facilitatorClient).register(network, scheme)

WRONG (V1 — will crash):
  paymentMiddleware(payToAddress, routes, facilitator)  // V1 arg order
  "@x402/types"  // doesn't exist
  "@x402/client"  // doesn't exist
  network: "base-sepolia"  // must be CAIP-2: "eip155:84532"
""",

    "mcp": """## @modelcontextprotocol/sdk v1.26+ API
CORRECT imports (ALL require .js extension with NodeNext):
  import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
  import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
  import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

WRONG (will crash with ERR_PACKAGE_PATH_NOT_EXPORTED):
  import { McpServer } from "@modelcontextprotocol/sdk";  // NO bare import

PEER DEPENDENCY: zod@^3.25.0 (NOT zod v4)

TOOL REGISTRATION:
  server.tool("name", "description", { param: z.string() }, async ({ param }) => ({
    content: [{ type: "text", text: result }]
  }));
""",

    "ethers": """## ethers v6.16 API (NOT v5)
ALL imports are top-level — NO namespaces:
  import { JsonRpcProvider, Wallet, Contract, parseEther, formatEther,
           ZeroAddress, getBytes, toBeHex, Interface } from "ethers";

WRONG v5 patterns (LLMs generate these constantly):
  ethers.providers.JsonRpcProvider → JsonRpcProvider
  ethers.utils.parseEther → parseEther
  ethers.constants.AddressZero → ZeroAddress
  new Web3Provider() → new BrowserProvider()
  BigNumber.from(x) → just use native bigint: 42n
  contract.address → contract.target
  provider.getSigner() → await provider.getSigner() (async in v6!)
  iface.parseLog(log) → returns null, doesn't throw — MUST null-check
  import from "@ethersproject/*" → import from "ethers"
""",

    "express5": """## Express 5.2 API
PATH MATCHING (breaking changes from v4):
  app.get("*", handler)      → app.get("/{*splat}", handler)
  "/users/:id?"              → "/users{/:id}"
  "/files/:path(.*)"         → "/files/{*path}"

ERROR HANDLING:
  - Async errors auto-caught (no express-async-errors needed)
  - Error handlers MUST use ErrorRequestHandler type:
    const errHandler: ErrorRequestHandler = (err, req, res, next) => { ... };

OTHER CHANGES:
  - req.body is undefined without body parser (was {} in v4)
  - res.redirect("back") removed — use req.get("Referrer")
  - req.query is read-only
""",
}


def get_api_docs_for_goal(goal: str) -> str:
    """Extract relevant API documentation based on goal keywords."""
    goal_lower = goal.lower()
    docs = []

    keyword_map = {
        "x402": ["x402", "payment gate", "paywall", "micropayment"],
        "mcp": ["mcp", "model context protocol", "mcp server", "mcp tool"],
        "ethers": ["ethers", "erc-8004", "erc8004", "blockchain", "smart contract", "solidity", "web3"],
        "express5": ["express", "fastify", "rest api", "http server"],
    }

    for doc_key, keywords in keyword_map.items():
        if any(kw in goal_lower for kw in keywords):
            docs.append(API_DOCS[doc_key])

    return "\n".join(docs)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Streaming Fixup — Find-and-Replace on Raw Output
# ═══════════════════════════════════════════════════════════════════════════════

# These patterns are applied line-by-line as the LLM output streams.
# Each is a (regex_pattern, replacement) tuple.
# They fix the most common LLM errors WITHOUT understanding context.

STREAMING_FIXES: list[tuple[str, str]] = [
    # ethers v5 → v6 (the #1 LLM hallucination for blockchain code)
    (r"new\s+ethers\.providers\.JsonRpcProvider\b", "new JsonRpcProvider"),
    (r"new\s+ethers\.providers\.WebSocketProvider\b", "new WebSocketProvider"),
    (r"ethers\.utils\.parseEther\b", "parseEther"),
    (r"ethers\.utils\.formatEther\b", "formatEther"),
    (r"ethers\.utils\.keccak256\b", "keccak256"),
    (r"ethers\.utils\.arrayify\b", "getBytes"),
    (r"ethers\.utils\.hexlify\b", "toBeHex"),
    (r"ethers\.constants\.AddressZero\b", "ZeroAddress"),
    (r"ethers\.constants\.MaxUint256\b", "MaxUint256"),
    (r"new\s+Web3Provider\b", "new BrowserProvider"),
    (r"BigNumber\.from\((\d+)\)", r"\1n"),  # BigNumber.from(42) → 42n

    # Express v4 → v5
    (r"""app\.(get|post|put|delete|all)\(\s*['"](\*)['"]\s*,""",
     r"""app.\1("/{*splat}","""),

    # jest → vitest (in test files only — applied globally but harmless in non-test)
    (r"\bjest\.fn\(\)", "vi.fn()"),
    (r"\bjest\.mock\(", "vi.mock("),
    (r"\bjest\.spyOn\(", "vi.spyOn("),
    (r"\bjest\.clearAllMocks\(\)", "vi.clearAllMocks()"),
]


def apply_streaming_fixes(code: str) -> str:
    """Apply streaming find-and-replace fixes to raw LLM output.

    These are deterministic regex substitutions that run in <10ms.
    They catch the patterns LLMs generate most frequently.
    """
    fixed = code
    fixes_applied = 0

    for pattern, replacement in STREAMING_FIXES:
        new_code = re.sub(pattern, replacement, fixed)
        if new_code != fixed:
            fixes_applied += 1
            fixed = new_code

    if fixes_applied:
        logger.info(f"Streaming fixup: {fixes_applied} pattern(s) corrected")

    return fixed


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Deterministic Autofixers (Post-Generation)
# ═══════════════════════════════════════════════════════════════════════════════

def fix_enum_to_const(code: str) -> str:
    """Replace TypeScript enum with 'as const' objects.

    Node.js 24 with erasableSyntaxOnly rejects enum declarations.
    This is a deterministic transformation.

    Before: enum Status { Active = "active", Inactive = "inactive" }
    After:  const Status = { Active: "active", Inactive: "inactive" } as const;
            type Status = typeof Status[keyof typeof Status];
    """
    def replace_enum(match):
        name = match.group(1)
        body = match.group(2)

        # Parse enum members
        members = []
        for member_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']+)["\']', body):
            members.append((member_match.group(1), member_match.group(2)))

        if not members:
            # Numeric enum — just use string literal union
            return match.group(0)  # Keep as-is (too complex to auto-fix)

        # Build const object
        entries = ", ".join(f'{k}: "{v}"' for k, v in members)
        return (
            f"const {name} = {{ {entries} }} as const;\n"
            f"type {name} = typeof {name}[keyof typeof {name}];"
        )

    return re.sub(
        r"(?:export\s+)?enum\s+(\w+)\s*\{([^}]+)\}",
        replace_enum,
        code,
    )


def fix_namespace_removal(code: str) -> str:
    """Remove namespace declarations (not erasable in Node.js 24).

    Converts: namespace Foo { export interface Bar { ... } }
    To:       interface FooBar { ... }  (prefixed with namespace name)
    """
    # Simple case: just strip the namespace wrapper
    def replace_namespace(match):
        # name = match.group(1)
        body = match.group(2)
        # Dedent the body
        lines = body.split("\n")
        dedented = []
        for line in lines:
            if line.startswith("  "):
                dedented.append(line[2:])
            elif line.startswith("\t"):
                dedented.append(line[1:])
            else:
                dedented.append(line)
        return "\n".join(dedented)

    return re.sub(
        r"(?:export\s+)?namespace\s+(\w+)\s*\{([\s\S]*?)\n\}",
        replace_namespace,
        code,
    )


def fix_import_type(code: str) -> str:
    """Add 'type' keyword to type-only imports for verbatimModuleSyntax.

    Detects imports that only import types (interface, type alias)
    and converts them to 'import type { ... }' form.

    This is a heuristic — it checks if the imported name starts with
    an uppercase letter and is not used as a value in the file.
    """
    lines = code.split("\n")
    new_lines = []

    for line in lines:
        # Match: import { Something } from '...'
        match = re.match(
            r"^(import\s+)\{([^}]+)\}(\s+from\s+['\"][^'\"]+['\"];?\s*)$",
            line.strip(),
        )
        if match and "import type" not in line:
            imports = [s.strip() for s in match.group(2).split(",")]
            # Check if ALL imports look like types (uppercase first letter, no value usage)
            all_types = all(
                imp.split(" as ")[0].strip()[0:1].isupper()
                for imp in imports if imp.strip()
            )
            # Don't convert if any import is clearly a value (function, class instance)
            value_indicators = ["create", "make", "get", "parse", "format", "new"]
            has_value = any(
                any(vi in imp.lower() for vi in value_indicators)
                for imp in imports
            )

            if all_types and not has_value:
                # Check if any of these names are used as values (not just types) in the code
                used_as_value = False
                rest_of_code = "\n".join(lines)
                for imp in imports:
                    name = imp.split(" as ")[-1].strip()
                    # Used as value: new Name(), Name(), Name.something
                    if re.search(rf"\bnew\s+{re.escape(name)}\b", rest_of_code):
                        used_as_value = True
                        break
                    if re.search(rf"\b{re.escape(name)}\(", rest_of_code):
                        used_as_value = True
                        break

                if not used_as_value:
                    line = line.replace("import {", "import type {", 1)

        new_lines.append(line)

    return "\n".join(new_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def fixup_typescript_output(
    code_files: dict[str, str],
    goal: str = "",
) -> dict[str, str]:
    """Apply the full three-stage fixup pipeline to generated TypeScript.

    Stage 1: API docs (injected at prompt time — not applied here)
    Stage 2: Streaming fixes (regex patterns)
    Stage 3: Deterministic autofixers (enum→const, namespace removal, import type)

    Then runs the covenant enforcer for structural validation.
    """
    fixed = {}
    total_fixes = 0

    for fname, content in code_files.items():
        if not fname.endswith((".ts", ".tsx", ".js", ".jsx")):
            fixed[fname] = content
            continue

        new_content = content

        # Stage 2: Streaming fixes
        new_content = apply_streaming_fixes(new_content)

        # Stage 3a: enum → as const (Node.js 24 compatibility)
        new_content = fix_enum_to_const(new_content)

        # Stage 3b: namespace removal (Node.js 24 compatibility)
        new_content = fix_namespace_removal(new_content)

        # Stage 3c: import type for type-only imports (verbatimModuleSyntax)
        new_content = fix_import_type(new_content)

        if new_content != content:
            total_fixes += 1

        fixed[fname] = new_content

    # Stage 3d: Run covenant enforcer
    from belief.validators.typescript_covenants import enforce_ts_covenants
    fixed, ts_result = enforce_ts_covenants(fixed, auto_fix=True)
    total_fixes += ts_result.fixes_applied

    if total_fixes:
        logger.info(f"TypeScript fixup: {total_fixes} total corrections applied")

    return fixed
