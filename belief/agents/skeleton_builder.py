"""
SkeletonBuilder Agent — Milestone 1, Pass 1

Takes a SkeletonArtifact from the Architect and generates skeleton files:
- Pydantic models (from model_chains)
- Abstract base classes (from abc_definitions)
- Protocol definitions (from protocol_definitions)
- Config schemas (from config_schemas)
- Exception hierarchies (from exception_specs)

All files have typed signatures and `pass` bodies — no implementation logic.
Runs on Haiku for cost efficiency.

After each file is generated, it's parsed by the SymbolRegistry so that
downstream files (and Pass 2 implementation files) have access to the
compressed symbol context.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belief.models.skeleton import ModelSpec, SkeletonArtifact
    from belief.models.symbol_registry import SymbolRegistry

logger = logging.getLogger(__name__)


# Canonical package names we treat as SQLAlchemy-family.
# Extend here rather than adding new matching logic in each generator.
_SQLA_PACKAGES = {"sqlalchemy", "sqlmodel"}
# Symbols that identify a file as the authoritative DB module, regardless
# of filename or external_dependencies wording.
_DB_EXPORT_SYMBOLS = {"Base", "engine", "SessionLocal", "get_db", "init_db"}


def _normalized_deps(skeleton: SkeletonArtifact) -> set[str]:
    """Lowercased external deps, stripped of version specifiers and extras.

    `"SQLAlchemy>=2.0"` → `"sqlalchemy"`; `"sqlalchemy[asyncio]==2.0.25"` → `"sqlalchemy"`.
    """
    out: set[str] = set()
    for dep in skeleton.external_dependencies:
        base = re.split(r"[<>=!~\[;\s]", dep.strip(), maxsplit=1)[0].lower()
        if base:
            out.add(base)
    return out


def _uses_sqlalchemy(skeleton: SkeletonArtifact) -> bool:
    """True if the project declares any SQLAlchemy-family dependency."""
    return bool(_normalized_deps(skeleton) & _SQLA_PACKAGES)


def _module_path(file_path: str) -> str:
    """Convert `pkg/models.py` → `pkg.models` for use in import statements."""
    return file_path.replace("/", ".").removesuffix(".py")


def _resolve_external_bases(
    skeleton: SkeletonArtifact,
    source_file: str,
    needed: set[str],
) -> tuple[dict[str, set[str]], set[str]]:
    """Map each needed base class to the module that exports it.

    Walks `skeleton.dependency_edges` looking for edges where
    `source == source_file` and the edge's `symbols` contain names we need.
    Returns `({module_path: {symbol, ...}}, unresolved_symbols)`.
    """
    resolved: dict[str, set[str]] = {}
    remaining = set(needed)

    for edge in skeleton.dependency_edges:
        if edge.source != source_file:
            continue
        hits = remaining & set(edge.symbols or ())
        if not hits:
            continue
        module_path = _module_path(edge.target)
        resolved.setdefault(module_path, set()).update(hits)
        remaining -= hits
        if not remaining:
            break

    return resolved, remaining


# Crude type-annotation → SQLAlchemy Column type mapping. Covers the
# common cases seen in tier 3 schemas; anything unrecognized falls back
# to `String`, which pydantic-happy JSON string fields round-trip fine.
_SQLA_COLUMN_TYPES = {
    "int": "Integer",
    "str": "String",
    "bytes": "String",
    "bool": "Integer",
    "float": "String",
    "datetime": "DateTime",
    "date": "DateTime",
}


def _literal_default(default: str | None) -> str | None:
    """Return a syntactically valid Python literal for a field default.

    The architect stores defaults as raw strings, so `"sqlite:///./x.db"`
    arrives as the bare characters `sqlite:///./x.db`. Emitting that into
    `Field(default=...)` produces invalid syntax. This helper uses
    `ast.literal_eval` as the authoritative test: if the raw default
    parses as a Python literal (None/True/False, a number, a quoted
    string, a list/dict/tuple literal), pass it through; otherwise
    treat it as a plain string and `repr()` it so embedded quotes and
    backslashes are escaped safely.
    """
    if default is None:
        return None
    if not isinstance(default, str):
        return repr(default)
    try:
        ast.literal_eval(default)
        return default
    except (ValueError, SyntaxError):
        return repr(default)


def _derive_tablename(class_name: str) -> str:
    """Convert a camelCase class name to a snake_case plural table name."""
    out: list[str] = []
    for i, ch in enumerate(class_name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    name = "".join(out)
    return name if name.endswith("s") else name + "s"


def _orm_column_type(type_annotation: str) -> str:
    """Map a Python type annotation string to a SQLAlchemy Column type.

    Strips `Optional[...]`, `list[...]`, and other generic wrappers to get
    at the inner type. Unknown types fall back to `String`.
    """
    annot = type_annotation.strip()
    # Peel a single level of generic wrapping, e.g. `Optional[int]` → `int`.
    for prefix in ("Optional[", "list[", "List[", "Mapped["):
        if annot.startswith(prefix) and annot.endswith("]"):
            annot = annot[len(prefix):-1].strip()
            break
    annot_lower = annot.lower()
    # Long-text heuristic: any "text" or "content" style ends up as Text.
    if annot_lower in _SQLA_COLUMN_TYPES:
        return _SQLA_COLUMN_TYPES[annot_lower]
    return "String"


def _emit_orm_body(lines: list[str], model: ModelSpec) -> None:
    """Emit a valid SQLAlchemy ORM class body from a ModelSpec.

    - Pulls `__tablename__` from a dunder field if present, otherwise
      derives it from the class name.
    - Emits every non-dunder field as `Column(...)`.
    - Guarantees a primary-key `id` column — SQLAlchemy rejects mapped
      classes without one, and the architect often omits it.
    """
    tablename = _derive_tablename(model.name)
    column_fields: list = []
    has_id = False

    for f in model.fields:
        if f.name == "__tablename__":
            if f.default:
                raw = f.default.strip().strip('"').strip("'")
                if raw:
                    tablename = raw
            continue
        if f.name.startswith("__") and f.name.endswith("__"):
            continue  # other dunder metadata — ignore, builder can add later
        column_fields.append(f)
        if f.name == "id":
            has_id = True

    lines.append(f'    __tablename__ = "{tablename}"')
    lines.append("")
    if not has_id:
        lines.append("    id = Column(Integer, primary_key=True, index=True)")

    for f in column_fields:
        col_type = _orm_column_type(f.type_annotation)
        if f.name == "id":
            lines.append(f"    id = Column({col_type}, primary_key=True, index=True)")
            continue
        nullable = f.default is not None or "Optional" in f.type_annotation
        nullable_str = "True" if nullable else "False"
        lines.append(f"    {f.name} = Column({col_type}, nullable={nullable_str})")


def _file_is_db_module(skeleton: SkeletonArtifact, file_path: str) -> bool:
    """True if `file_path` is (or should be) the authoritative DB module.

    Three independent signals, any one sufficient:
      1. Basename is `database.py` or `db.py`.
      2. Any dependency edge targets this file requesting DB exports
         (`Base`, `engine`, `SessionLocal`, `get_db`, `init_db`).
      3. Project uses SQLAlchemy and the file basename contains "db".
    """
    base = file_path.rsplit("/", 1)[-1]
    if base in ("database.py", "db.py"):
        return True
    for edge in skeleton.dependency_edges:
        if edge.target == file_path and set(edge.symbols or ()) & _DB_EXPORT_SYMBOLS:
            return True
    return False


# ---------------------------------------------------------------------------
# Skeleton Code Generators (deterministic — no LLM needed for most skeletons)
# ---------------------------------------------------------------------------

def _generate_model_chain_code(skeleton: SkeletonArtifact, file_path: str) -> str | None:
    """
    Generate Pydantic model code for all models in chains that belong to this file.
    This is deterministic — model chains are fully specified in the SkeletonArtifact.
    """
    models_in_file = []
    for chain in skeleton.model_chains:
        for model in chain.models:
            if model.file_path == file_path:
                models_in_file.append(model)

    if not models_in_file:
        return None

    # Detect if project uses SQLAlchemy (avoid __future__ annotations)
    uses_sqlalchemy = _uses_sqlalchemy(skeleton)

    lines = [
        '"""Auto-generated Pydantic models — skeleton (Pass 1)."""',
        "",
    ]

    # Only add __future__ if NOT using SQLAlchemy (it breaks Mapped type resolution)
    if not uses_sqlalchemy:
        lines.append("from __future__ import annotations")
        lines.append("")

    lines.extend([
        "from typing import Optional",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
    ])

    # Collect base classes that are in the same file
    local_names = {m.name for m in models_in_file}

    # Collect imports for base classes from other files.
    # Emit REAL imports (not TODO comments) so the generated class body
    # parses and executes. For each external base, look up the dependency
    # edge that imports it and resolve the target module path.
    external_bases = set()
    for model in models_in_file:
        if model.base_class != "BaseModel" and model.base_class not in local_names:
            external_bases.add(model.base_class)

    if external_bases:
        resolved, unresolved = _resolve_external_bases(
            skeleton, file_path, external_bases
        )
        for module_path, names in sorted(resolved.items()):
            names_str = ", ".join(sorted(names))
            lines.append(f"from {module_path} import {names_str}")
        if unresolved:
            # Still log so we can see what the architect didn't wire up,
            # but don't leave unresolved names — the class body below
            # would reference undefined symbols.
            logger.warning(
                f"skeleton_builder: {file_path} references unresolved base(s) "
                f"{sorted(unresolved)} — no dependency edge found"
            )
            # Heuristic fallback: if the project uses SQLAlchemy and a
            # `Base` is unresolved, import it from the conventional
            # database module. This matches the database-generator output.
            if uses_sqlalchemy and "Base" in unresolved:
                lines.append("from database import Base")
                unresolved.discard("Base")
            for name in sorted(unresolved):
                lines.append(f"# TODO: Import external base: {name}")
        lines.append("")

    # Track whether any ORM (non-pydantic) class appears in this file so
    # we can decide whether to emit SQLAlchemy Column imports at the top.
    any_orm_class = any(
        m.base_class != "BaseModel"
        and m.base_class not in local_names
        for m in models_in_file
    )
    if any_orm_class:
        lines.append("from sqlalchemy import Column, DateTime, Integer, String, Text")
        lines.append("")

    for model in models_in_file:
        # Pydantic Field(...) syntax only applies when the model actually
        # inherits from `BaseModel` (or chains from another BaseModel).
        # A class inheriting from SQLAlchemy's `Base` is an ORM class and
        # rejects pydantic `Field(...)`. If the architect put `__tablename__`
        # in `model.fields`, naively emitting it as a pydantic field yields
        # `__tablename__: str = Field(default='contacts', ...)`, which
        # DeclarativeBase reads as a FieldInfo for the table name and
        # blows up with `ArgumentError: could not assemble any primary
        # key columns`.
        #
        # For ORM classes we emit real Column(...) bodies here. Emitting
        # `pass` and deferring to the builder doesn't work — the builder's
        # contract is "don't touch skeleton files", so an empty ORM class
        # ships to pytest and hits `does not have __tablename__`.
        base_is_pydantic = (
            model.base_class == "BaseModel"
            or model.base_class in local_names
        )

        # Class definition
        lines.append("")
        if model.docstring:
            lines.append(f'class {model.name}({model.base_class}):')
            lines.append(f'    """{model.docstring}"""')
        else:
            lines.append(f"class {model.name}({model.base_class}):")

        if not model.fields and not model.validators:
            if not base_is_pydantic:
                # Empty ORM class still needs a __tablename__ or
                # SQLAlchemy will reject it.
                tablename = _derive_tablename(model.name)
                lines.append(f'    __tablename__ = "{tablename}"')
                lines.append("    id = Column(Integer, primary_key=True, index=True)")
            else:
                lines.append("    pass")
            lines.append("")
            continue

        if not base_is_pydantic:
            _emit_orm_body(lines, model)
            lines.append("")
            continue

        # Fields (pydantic path only). Defensive filter: never emit dunder
        # attributes like `__tablename__` as pydantic fields even if the
        # architect put them in `model.fields` — they're class-level
        # metadata, not data fields.
        for f in model.fields:
            if f.name.startswith("__") and f.name.endswith("__"):
                continue
            # model_config is class-level Pydantic v2 config, not a Field
            if f.name == "model_config":
                lines.append("    model_config = ConfigDict(from_attributes=True)")
                continue
            desc = f.description.replace('"', "'") if f.description else ""
            default = _literal_default(f.default)
            if desc:
                if default is not None:
                    lines.append(
                        f'    {f.name}: {f.type_annotation} = Field(default={default}, description="{desc}")'
                    )
                else:
                    lines.append(
                        f'    {f.name}: {f.type_annotation} = Field(description="{desc}")'
                    )
            else:
                if default is not None:
                    lines.append(f"    {f.name}: {f.type_annotation} = {default}")
                else:
                    lines.append(f"    {f.name}: {f.type_annotation}")

        # Validator stubs
        for v in model.validators:
            lines.append("")
            lines.append(f"    def {v}(self):")
            lines.append("        pass  # TODO: implement validator")

        lines.append("")

    return "\n".join(lines)


def _generate_abc_code(skeleton: SkeletonArtifact, file_path: str) -> str | None:
    """Generate ABC code for all ABCs that belong to this file."""
    abcs_in_file = [a for a in skeleton.abc_definitions if a.file_path == file_path]
    if not abcs_in_file:
        return None

    lines = [
        '"""Auto-generated abstract base classes — skeleton (Pass 1)."""',
        "",
        "from __future__ import annotations",
        "",
        "from abc import ABC, abstractmethod",
        "from typing import Any, Optional",
        "",
    ]

    for abc_def in abcs_in_file:
        base_str = ", ".join(abc_def.base_classes)
        if abc_def.docstring:
            lines.append(f"class {abc_def.name}({base_str}):")
            lines.append(f'    """{abc_def.docstring}"""')
        else:
            lines.append(f"class {abc_def.name}({base_str}):")

        if not abc_def.methods and not abc_def.class_attributes:
            lines.append("    pass")
            lines.append("")
            continue

        # Class attributes
        for attr in abc_def.class_attributes:
            if attr.default is not None:
                lines.append(f"    {attr.name}: {attr.type_annotation} = {attr.default}")
            else:
                lines.append(f"    {attr.name}: {attr.type_annotation}")

        if abc_def.class_attributes:
            lines.append("")

        # Methods
        for method in abc_def.methods:
            decorators = []
            if method.is_abstract:
                decorators.append("@abstractmethod")
            # MethodSignature has no decorators field — additional
            # decorators (e.g. @property) would need to be added to the model
            for dec in decorators:
                lines.append(f"    {dec}")

            prefix = "async def" if method.is_async else "def"
            lines.append(f"    {prefix} {method.name}({method.params}) -> {method.return_type}:")
            if method.docstring:
                lines.append(f'        """{method.docstring}"""')
            lines.append(f"        ...  # TODO: implement {method.name}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


def _generate_protocol_code(skeleton: SkeletonArtifact, file_path: str) -> str | None:
    """Generate Protocol code for all protocols that belong to this file."""
    protos_in_file = [p for p in skeleton.protocol_definitions if p.file_path == file_path]
    if not protos_in_file:
        return None

    lines = [
        '"""Auto-generated Protocol definitions — skeleton (Pass 1)."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Protocol, runtime_checkable",
        "",
    ]

    for proto in protos_in_file:
        lines.append("@runtime_checkable")
        if proto.docstring:
            lines.append(f"class {proto.name}(Protocol):")
            lines.append(f'    """{proto.docstring}"""')
        else:
            lines.append(f"class {proto.name}(Protocol):")

        if not proto.methods and not proto.attributes:
            lines.append("    ...")
            lines.append("")
            continue

        for attr in proto.attributes:
            lines.append(f"    {attr.name}: {attr.type_annotation}")

        for method in proto.methods:
            prefix = "async def" if method.is_async else "def"
            lines.append(f"    {prefix} {method.name}({method.params}) -> {method.return_type}: ...")

        lines.append("")

    return "\n".join(lines)


def _generate_database_code(skeleton: SkeletonArtifact, file_path: str) -> str | None:
    """Generate SQLAlchemy database setup code for database.py files.

    This is the AUTHORITATIVE database setup. It exports:
    - engine: SQLAlchemy engine
    - SessionLocal: session factory
    - Base: DeclarativeBase for models
    - get_db: FastAPI dependency
    - init_db: create all tables

    Fires when EITHER:
      * the file is a DB module by name or by dep-edge evidence, AND
      * the project declares a SQLAlchemy-family dependency OR a
        dependency edge into this file names a DB export symbol.

    Previously this returned None when `external_dependencies` contained
    `"sqlalchemy>=2.0"` instead of the bare `"sqlalchemy"` — which made
    the generator fall through to `_generate_config_code` and emit a
    bare `DatabaseConfig` class with no engine/Base/get_db. See
    belief/agents/skeleton_builder.py root-cause analysis.

    The builder must NOT overwrite this file.
    The debugger must NOT replace exports — only add to them.
    """
    if not _file_is_db_module(skeleton, file_path):
        return None

    # Must also have some SQLAlchemy evidence — either a dep or a dep-edge
    # into this file requesting DB exports. Without that we'd clobber a
    # file that happens to be named `db.py` in a non-SQL project.
    has_dep_edge_signal = any(
        edge.target == file_path and set(edge.symbols or ()) & _DB_EXPORT_SYMBOLS
        for edge in skeleton.dependency_edges
    )
    if not (_uses_sqlalchemy(skeleton) or has_dep_edge_signal):
        return None

    db_core = '''"""Database setup — SQLAlchemy 2.x engine, session, and base."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """Dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
'''

    # If the architect also attached a ConfigSchema to this file
    # (e.g. a DatabaseConfig with SQLALCHEMY_DATABASE_URL), we used to
    # lose the engine/Base because _generate_config_code fired instead.
    # Now we merge: append the config class body below the engine block.
    config_body = _generate_config_code(skeleton, file_path)
    if config_body is None:
        return db_core

    # Strip the config body's duplicate module docstring + __future__ line,
    # keep the imports and class definitions.
    config_tail = _strip_module_header(config_body)
    return db_core + "\n" + config_tail


def _strip_module_header(source: str) -> str:
    """Drop leading module docstring and `from __future__ import annotations`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines()
    drop_until = 0
    for node in tree.body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        )
        if is_docstring or is_future:
            drop_until = max(drop_until, node.end_lineno or 0)
        else:
            break
    return "\n".join(lines[drop_until:]).lstrip("\n")


def _generate_config_code(skeleton: SkeletonArtifact, file_path: str) -> str | None:
    """Generate config/settings code."""
    configs_in_file = [c for c in skeleton.config_schemas if c.file_path == file_path]
    if not configs_in_file:
        return None

    lines = [
        '"""Auto-generated configuration schemas — skeleton (Pass 1)."""',
        "",
        "from __future__ import annotations",
        "",
        "from pydantic import ConfigDict, Field",
        "from pydantic_settings import BaseSettings",
        "",
    ]

    for config in configs_in_file:
        if config.docstring:
            lines.append(f"class {config.name}(BaseSettings):")
            lines.append(f'    """{config.docstring}"""')
        else:
            lines.append(f"class {config.name}(BaseSettings):")

        if config.env_prefix:
            lines.append(f"    model_config = ConfigDict(env_prefix='{config.env_prefix}')")
            lines.append("")

        if not config.fields:
            lines.append("    pass")
        else:
            for f in config.fields:
                default_literal = _literal_default(f.default)
                desc = f.description.replace('"', "'") if f.description else ""
                if desc:
                    if default_literal is not None:
                        lines.append(
                            f'    {f.name}: {f.type_annotation} = Field(default={default_literal}, description="{desc}")'
                        )
                    else:
                        lines.append(
                            f'    {f.name}: {f.type_annotation} = Field(description="{desc}")'
                        )
                else:
                    if default_literal is not None:
                        lines.append(f"    {f.name}: {f.type_annotation} = {default_literal}")
                    else:
                        lines.append(f"    {f.name}: {f.type_annotation}")

        lines.append("")

    return "\n".join(lines)


def _generate_exception_code(skeleton: SkeletonArtifact, file_path: str) -> str | None:
    """Generate exception hierarchy code."""
    exceptions_in_file = [e for e in skeleton.exception_specs if e.file_path == file_path]
    if not exceptions_in_file:
        return None

    lines = [
        '"""Auto-generated exception hierarchy — skeleton (Pass 1)."""',
        "",
    ]

    for exc in exceptions_in_file:
        if exc.docstring:
            lines.append(f"class {exc.name}({exc.base_class}):")
            lines.append(f'    """{exc.docstring}"""')
        else:
            lines.append(f"class {exc.name}({exc.base_class}):")

        if exc.message_template:
            lines.append("    def __init__(self, **kwargs):")
            lines.append(f'        super().__init__(f"{exc.message_template}")')
        else:
            lines.append("    pass")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generation orchestrator
# ---------------------------------------------------------------------------

def generate_skeleton_file(
    skeleton: SkeletonArtifact,
    file_path: str,
    registry: SymbolRegistry,
) -> str | None:
    """
    Generate a single skeleton file from the SkeletonArtifact spec.

    Tries each generator in order. Most skeleton files are deterministic —
    the SkeletonArtifact has enough information to generate them without
    an LLM call.

    After successful generation, the file is parsed and registered in the
    SymbolRegistry for downstream context.

    Args:
        skeleton: The full SkeletonArtifact from the Architect.
        file_path: The specific file to generate.
        registry: The SymbolRegistry to update after generation.

    Returns:
        Generated source code, or None if no generators matched.
    """
    # Short-circuit on non-Python files. Every generator here emits
    # Python source and the downstream `ast.parse` check will fail on
    # anything else (seen: `requirements.txt` landing here and producing
    # spurious `invalid syntax` errors in the logs).
    if not file_path.endswith(".py"):
        return None

    generators = [
        _generate_model_chain_code,
        _generate_abc_code,
        _generate_protocol_code,
        _generate_database_code,   # Must come before config (database.py is often tagged CONFIG)
        _generate_config_code,
        _generate_exception_code,
    ]

    for gen in generators:
        result = gen(skeleton, file_path)
        if result is not None:
            # Validate syntax
            try:
                ast.parse(result)
            except SyntaxError as e:
                logger.error(f"Skeleton generation produced invalid syntax for {file_path}: {e}")
                return None

            # Register symbols
            try:
                registry.register_source(result, file_path)
            except SyntaxError:
                pass  # Already logged above if it fails

            logger.info(f"Generated skeleton: {file_path} ({len(result.splitlines())} lines)")
            return result

    return None


def generate_all_skeletons(
    skeleton: SkeletonArtifact,
    registry: SymbolRegistry,
) -> dict[str, str]:
    """
    Generate all skeleton files (Pass 1) from a SkeletonArtifact.

    Returns a dict of {file_path: source_code} for all successfully
    generated skeleton files.
    """
    results = {}
    skeleton_files = skeleton.skeleton_files()

    logger.info(f"Pass 1: Generating {len(skeleton_files)} skeleton files")

    for entry in skeleton_files:
        code = generate_skeleton_file(skeleton, entry.path, registry)
        if code is not None:
            results[entry.path] = code
        else:
            logger.warning(
                f"No generator matched for skeleton file: {entry.path} (role={entry.role}). "
                f"This file will need LLM generation in Pass 2."
            )

    logger.info(f"Pass 1 complete: {len(results)}/{len(skeleton_files)} files generated deterministically")
    return results
