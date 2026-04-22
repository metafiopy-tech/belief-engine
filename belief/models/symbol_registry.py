"""
Symbol Registry — Milestone 1

After each file is generated, the registry parses it with `ast` and extracts
all exported symbols (classes, functions, constants) with their signatures.

When generating file N, the Builder receives the compressed symbol registry
from files 1..N-1 instead of full file contents (~1K tokens vs ~30K).

This is the foundation for Milestone 3's full context compression.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Symbol types
# ---------------------------------------------------------------------------

@dataclass
class FunctionSymbol:
    """An exported function or method."""
    name: str
    params: str           # Full parameter string with annotations
    return_type: str       # Return annotation as string, or "None"
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None

    def signature_line(self) -> str:
        prefix = "async def" if self.is_async else "def"
        return f"{prefix} {self.name}({self.params}) -> {self.return_type}: ..."


@dataclass
class ClassSymbol:
    """An exported class with its methods and attributes."""
    name: str
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionSymbol] = field(default_factory=list)
    class_attributes: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None

    def signature_block(self) -> str:
        lines = []
        base_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines.append(f"class {self.name}{base_str}:")
        if self.docstring:
            lines.append(f'    """{self.docstring}"""')
        for attr_name, attr_type in self.class_attributes:
            lines.append(f"    {attr_name}: {attr_type}")
        for method in self.methods:
            lines.append(f"    {method.signature_line()}")
        if not self.class_attributes and not self.methods:
            lines.append("    ...")
        return "\n".join(lines)


@dataclass
class ConstantSymbol:
    """A module-level constant or type alias."""
    name: str
    type_annotation: Optional[str] = None
    value_repr: Optional[str] = None  # Short repr of the value if simple

    def signature_line(self) -> str:
        if self.type_annotation:
            return f"{self.name}: {self.type_annotation}"
        if self.value_repr:
            return f"{self.name} = {self.value_repr}"
        return self.name


# ---------------------------------------------------------------------------
# File entry in the registry
# ---------------------------------------------------------------------------

@dataclass
class FileSymbols:
    """All exported symbols from a single file."""
    file_path: str
    module_path: str          # Python import path, e.g. "models.lead"
    classes: list[ClassSymbol] = field(default_factory=list)
    functions: list[FunctionSymbol] = field(default_factory=list)
    constants: list[ConstantSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # Raw import lines

    def all_symbol_names(self) -> list[str]:
        names = [c.name for c in self.classes]
        names += [f.name for f in self.functions]
        names += [c.name for c in self.constants]
        return names

    def as_compressed_context(self) -> str:
        """
        Render this file's symbols as a compact importable reference.
        This is what gets injected into the Builder's prompt.
        """
        lines = [f"# --- {self.module_path} ---"]
        for const in self.constants:
            lines.append(const.signature_line())
        for func in self.functions:
            lines.append(func.signature_line())
        for cls in self.classes:
            lines.append(cls.signature_block())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AST Extraction
# ---------------------------------------------------------------------------

def _annotation_to_str(node: Optional[ast.expr]) -> str:
    """Convert an AST annotation node to a string representation."""
    if node is None:
        return "None"
    return ast.unparse(node)


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names."""
    decorators = []
    for dec in node.decorator_list:
        decorators.append(ast.unparse(dec))
    return decorators


def _get_docstring(node: ast.AST) -> Optional[str]:
    """Extract first-line docstring from a class or function."""
    ds = ast.get_docstring(node)
    if ds:
        # Just the first line for compression
        first_line = ds.strip().split("\n")[0]
        return first_line
    return None


def _extract_params(args: ast.arguments) -> str:
    """Convert function arguments to a parameter string."""
    parts = []

    # Regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        param = arg.arg
        if arg.annotation:
            param += f": {ast.unparse(arg.annotation)}"
        default_idx = i - defaults_offset
        if default_idx >= 0:
            param += f" = {ast.unparse(args.defaults[default_idx])}"
        parts.append(param)

    # *args
    if args.vararg:
        va = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            va += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(va)
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        param = arg.arg
        if arg.annotation:
            param += f": {ast.unparse(arg.annotation)}"
        if args.kw_defaults[i] is not None:
            param += f" = {ast.unparse(args.kw_defaults[i])}"
        parts.append(param)

    # **kwargs
    if args.kwarg:
        kw = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            kw += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(kw)

    return ", ".join(parts)


def _extract_class(node: ast.ClassDef) -> ClassSymbol:
    """Extract a ClassSymbol from an AST ClassDef node."""
    bases = [ast.unparse(b) for b in node.bases]
    methods = []
    class_attrs = []

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private/dunder methods except __init__
            if item.name.startswith("_") and item.name != "__init__":
                continue
            methods.append(FunctionSymbol(
                name=item.name,
                params=_extract_params(item.args),
                return_type=_annotation_to_str(item.returns),
                is_async=isinstance(item, ast.AsyncFunctionDef),
                decorators=_get_decorators(item),
                docstring=_get_docstring(item),
            ))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            class_attrs.append((
                item.target.id,
                ast.unparse(item.annotation)
            ))

    return ClassSymbol(
        name=node.name,
        bases=bases,
        methods=methods,
        class_attributes=class_attrs,
        decorators=_get_decorators(node),
        docstring=_get_docstring(node),
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSymbol:
    """Extract a FunctionSymbol from an AST node."""
    return FunctionSymbol(
        name=node.name,
        params=_extract_params(node.args),
        return_type=_annotation_to_str(node.returns),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=_get_decorators(node),
        docstring=_get_docstring(node),
    )


def _extract_constant(node: ast.AnnAssign | ast.Assign) -> Optional[ConstantSymbol]:
    """Extract a module-level constant or type alias."""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        value_repr = ast.unparse(node.value) if node.value else None
        # Truncate long values
        if value_repr and len(value_repr) > 80:
            value_repr = value_repr[:77] + "..."
        return ConstantSymbol(
            name=node.target.id,
            type_annotation=ast.unparse(node.annotation),
            value_repr=value_repr,
        )
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.isupper():
            value_repr = ast.unparse(node.value) if node.value else None
            if value_repr and len(value_repr) > 80:
                value_repr = value_repr[:77] + "..."
            return ConstantSymbol(
                name=target.id,
                value_repr=value_repr,
            )
    return None


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract import lines for reference."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            imports.append(f"from {module} import {names}")
    return imports


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(source_code: str, file_path: str, project_root: str = "") -> FileSymbols:
    """
    Parse a Python source file and extract all exported symbols.

    Args:
        source_code: The Python source code to parse.
        file_path: Relative file path (e.g. "models/lead.py").
        project_root: Optional project root for module path computation.

    Returns:
        FileSymbols with all extracted symbols.

    Raises:
        SyntaxError: If the source code cannot be parsed.
    """
    tree = ast.parse(source_code)

    # Compute module path from file path
    module_path = file_path.replace("/", ".").replace("\\", ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    if module_path.endswith(".__init__"):
        module_path = module_path[:-9]

    classes = []
    functions = []
    constants = []

    for node in tree.body:
        # Skip private names (single underscore prefix)
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                classes.append(_extract_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(_extract_function(node))
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            const = _extract_constant(node)
            if const:
                constants.append(const)

    return FileSymbols(
        file_path=file_path,
        module_path=module_path,
        classes=classes,
        functions=functions,
        constants=constants,
        imports=_extract_imports(tree),
    )


def parse_file_from_path(path: Path, project_root: Path | None = None) -> FileSymbols:
    """Convenience: parse a file from disk."""
    source = path.read_text()
    rel_path = str(path.relative_to(project_root)) if project_root else str(path)
    return parse_file(source, rel_path, str(project_root or ""))


# ---------------------------------------------------------------------------
# Symbol Registry — accumulates symbols across files
# ---------------------------------------------------------------------------

class SymbolRegistry:
    """
    Accumulates parsed symbols from all generated files.

    Used by the Builder to get compressed context when generating file N:
    instead of including full source of files 1..N-1, it includes only
    their symbol signatures (~1K tokens vs ~30K).
    """

    def __init__(self):
        self._files: dict[str, FileSymbols] = {}  # keyed by file_path

    def register(self, file_symbols: FileSymbols) -> None:
        """Register symbols from a generated file."""
        self._files[file_symbols.file_path] = file_symbols

    def register_source(self, source_code: str, file_path: str) -> FileSymbols:
        """Parse and register a source file in one step."""
        symbols = parse_file(source_code, file_path)
        self.register(symbols)
        return symbols

    def get(self, file_path: str) -> Optional[FileSymbols]:
        """Get symbols for a specific file."""
        return self._files.get(file_path)

    def all_files(self) -> list[FileSymbols]:
        """All registered files in insertion order."""
        return list(self._files.values())

    def resolve_symbol(self, symbol_name: str) -> Optional[tuple[str, str]]:
        """
        Find which module exports a given symbol name.
        Returns (module_path, symbol_name) or None.
        """
        for fs in self._files.values():
            if symbol_name in fs.all_symbol_names():
                return (fs.module_path, symbol_name)
        return None

    def context_for_file(self, file_path: str, dependency_paths: list[str]) -> str:
        """
        Build compressed context for generating a specific file.

        Only includes symbols from the file's direct dependencies,
        not the entire registry.

        Args:
            file_path: The file about to be generated.
            dependency_paths: File paths this file depends on.

        Returns:
            Compressed context string with symbol signatures.
        """
        sections = []
        for dep_path in dependency_paths:
            dep_symbols = self._files.get(dep_path)
            if dep_symbols:
                sections.append(dep_symbols.as_compressed_context())

        if not sections:
            return "# No dependencies registered yet."

        header = f"# Symbol context for {file_path}\n# Dependencies: {', '.join(dependency_paths)}\n"
        return header + "\n\n".join(sections)

    def full_registry_context(self) -> str:
        """Full compressed context from all registered files."""
        if not self._files:
            return "# No files registered yet."
        sections = [fs.as_compressed_context() for fs in self._files.values()]
        return "\n\n".join(sections)
