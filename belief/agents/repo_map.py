"""
Repository Map — AST-based context compression for multi-file generation.

Instead of sending full file contents as context to the builder, extract
only the public API: class names, method signatures, function signatures,
and top-level constants. A 200-line file compresses to ~10-15 lines.

For a 50-file project, this reduces cross-file context from ~40K tokens
to ~3-4K tokens — within any model's effective context window.

The repo map is ranked by dependency relevance:
  1. Direct dependencies → full signatures with docstrings
  2. One-hop dependencies → class + function names only
  3. Everything else → just file names and module docstrings

Source: TIER_4_5_SCALING_PLAN.md Milestone 3
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

logger = logging.getLogger("belief.agents.repo_map")


def extract_signatures(code: str, filename: str = "") -> str:
    """Extract public API from Python code using ast.parse().

    Returns a compact representation of the file's public interface:
    - Module docstring (first line)
    - Import statements (just the names imported)
    - Class declarations with base classes and method signatures
    - Top-level function signatures
    - Top-level constants with type hints

    Example output:
        # models.py — Data models for the pipeline
        class RawData(BaseModel):
            source: str
            payload: dict[str, Any]
            timestamp: datetime

        class EnrichedData(RawData):
            score: float
            confidence: float

    A 200-line file typically compresses to 10-20 lines.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"# {filename} — (syntax error, cannot parse)"

    lines: list[str] = []

    # Module docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        lines.append(f"# {filename} — {first_line}")
    else:
        lines.append(f"# {filename}")

    for node in ast.iter_child_nodes(tree):
        # Class definitions
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = [_name_str(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"class {node.name}{base_str}:")

            # Class docstring
            class_doc = ast.get_docstring(node)
            if class_doc:
                first_line = class_doc.strip().split("\n")[0]
                lines.append(f'    """{first_line}"""')

            # Fields (assignments with type annotations)
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                    if name.startswith("_"):
                        continue
                    type_str = _annotation_str(item.annotation)
                    if item.value is not None:
                        default = _value_str(item.value)
                        lines.append(f"    {name}: {type_str} = {default}")
                    else:
                        lines.append(f"    {name}: {type_str}")

            # Methods (just signatures)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    params = _params_str(item.args)
                    returns = _annotation_str(item.returns) if item.returns else "None"
                    lines.append(f"    {prefix} {item.name}({params}) -> {returns}")

            lines.append("")

        # Top-level functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            params = _params_str(node.args)
            returns = _annotation_str(node.returns) if node.returns else "None"
            func_doc = ast.get_docstring(node)
            if func_doc:
                first_line = func_doc.strip().split("\n")[0]
                lines.append(f"{prefix} {node.name}({params}) -> {returns}")
                lines.append(f'    """{first_line}"""')
            else:
                lines.append(f"{prefix} {node.name}({params}) -> {returns}")
            lines.append("")

        # Top-level assignments with type annotations
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if not name.startswith("_"):
                type_str = _annotation_str(node.annotation)
                lines.append(f"{name}: {type_str}")

    return "\n".join(lines)


def extract_signatures_brief(code: str, filename: str = "") -> str:
    """Extract minimal API — just class names and function names.

    Used for non-dependency files (one-hop or further).
    Even more compact than full signatures.

    Example:
        # utils.py: classes=[Validator, Formatter], functions=[validate_input, format_output]
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"# {filename} — (syntax error)"

    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            functions.append(node.name)

    parts = []
    if classes:
        parts.append(f"classes=[{', '.join(classes)}]")
    if functions:
        parts.append(f"functions=[{', '.join(functions)}]")

    return f"# {filename}: {', '.join(parts)}" if parts else f"# {filename}: (empty)"


class RepoMap:
    """Compressed representation of a multi-file project.

    Usage:
        repo_map = RepoMap.from_code_files(code_files)
        context = repo_map.format_for_file("server.py", dependencies=["models.py", "base.py"])
    """

    def __init__(self, signatures: dict[str, str], briefs: dict[str, str]) -> None:
        self._signatures = signatures  # Full signatures per file
        self._briefs = briefs  # Brief one-liners per file

    @classmethod
    def from_code_files(cls, code_files: dict[str, str]) -> RepoMap:
        """Build repo map from all generated code files."""
        signatures = {}
        briefs = {}
        for fname, code in code_files.items():
            if not fname.endswith(".py"):
                continue
            if "test" in fname.lower():
                continue  # Skip test files in repo map
            signatures[fname] = extract_signatures(code, fname)
            briefs[fname] = extract_signatures_brief(code, fname)
        return cls(signatures, briefs)

    def format_for_file(
        self,
        target_file: str,
        dependencies: Optional[list[str]] = None,
        max_tokens: int = 4000,
    ) -> str:
        """Format the repo map for generating a specific file.

        Priority:
        1. Direct dependencies → full signatures
        2. Everything else → brief one-liners

        Args:
            target_file: The file being generated (excluded from output)
            dependencies: Files this file imports from (get full signatures)
            max_tokens: Approximate token budget (~4 chars per token)
        """
        deps = set(dependencies or [])
        sections: list[str] = []
        char_budget = max_tokens * 4

        # Direct dependencies get full signatures
        if deps:
            sections.append("## Direct Dependencies (full API):")
            for dep in sorted(deps):
                if dep in self._signatures and dep != target_file:
                    sig = self._signatures[dep]
                    sections.append(sig)
                    sections.append("")

        # Everything else gets brief one-liners
        other_files = [f for f in sorted(self._briefs.keys()) if f != target_file and f not in deps]
        if other_files:
            sections.append("## Other Project Files:")
            for fname in other_files:
                sections.append(self._briefs[fname])

        result = "\n".join(sections)

        # Truncate if over budget (keep dependencies, cut others)
        if len(result) > char_budget:
            # Keep dependency section, truncate other files
            dep_section = []
            other_section = []
            in_other = False
            for line in result.split("\n"):
                if "## Other Project Files:" in line:
                    in_other = True
                if in_other:
                    other_section.append(line)
                else:
                    dep_section.append(line)

            dep_text = "\n".join(dep_section)
            remaining = char_budget - len(dep_text)
            if remaining > 100:
                other_text = "\n".join(other_section)[:remaining]
                result = dep_text + "\n" + other_text
            else:
                result = dep_text

        return result

    def format_overview(self, max_tokens: int = 2000) -> str:
        """Format a brief overview of the entire project.

        Used when no specific target file — gives the architect/planner
        a high-level view of what exists.
        """
        lines = ["## Project API Overview:"]
        for fname in sorted(self._briefs.keys()):
            lines.append(self._briefs[fname])

        result = "\n".join(lines)
        if len(result) > max_tokens * 4:
            result = result[: max_tokens * 4] + "\n# ... (truncated)"
        return result


# ── AST helpers ──────────────────────────────────────────────────────────────


def _name_str(node) -> str:
    """Convert an AST name node to string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_name_str(node.value)}[{_annotation_str(node.slice)}]"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return "..."


def _annotation_str(node) -> str:
    """Convert a type annotation AST node to string."""
    if node is None:
        return "Any"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_name_str(node.value)}[{_annotation_str(node.slice)}]"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Tuple):
        return ", ".join(_annotation_str(e) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_annotation_str(node.left)} | {_annotation_str(node.right)}"
    if isinstance(node, ast.List):
        return f"[{', '.join(_annotation_str(e) for e in node.elts)}]"
    return "..."


def _params_str(args: ast.arguments) -> str:
    """Convert function arguments to a compact parameter string."""
    parts = []
    # Regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        name = arg.arg
        if name == "self" or name == "cls":
            parts.append(name)
            continue
        type_str = _annotation_str(arg.annotation) if arg.annotation else ""
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            default = _value_str(args.defaults[default_idx])
            if type_str:
                parts.append(f"{name}: {type_str} = {default}")
            else:
                parts.append(f"{name}={default}")
        else:
            if type_str:
                parts.append(f"{name}: {type_str}")
            else:
                parts.append(name)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


def _value_str(node) -> str:
    """Convert a default value AST node to a compact string."""
    if isinstance(node, ast.Constant):
        r = repr(node.value)
        return r if len(r) < 30 else "..."
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return f"{_name_str(node.func)}(...)"
    if isinstance(node, (ast.List, ast.Tuple)):
        return "[]" if isinstance(node, ast.List) else "()"
    if isinstance(node, ast.Dict):
        return "{}"
    return "..."
