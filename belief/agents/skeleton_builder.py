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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belief.models.skeleton import SkeletonArtifact
    from belief.models.symbol_registry import SymbolRegistry

logger = logging.getLogger(__name__)


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
    deps_lower = {d.lower() for d in skeleton.external_dependencies}
    uses_sqlalchemy = any(d in deps_lower for d in ("sqlalchemy", "sqlmodel"))

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
        "from pydantic import BaseModel, Field",
        "",
    ])

    # Collect base classes that are in the same file
    local_names = {m.name for m in models_in_file}

    # Collect imports for base classes from other files
    external_bases = set()
    for model in models_in_file:
        if model.base_class != "BaseModel" and model.base_class not in local_names:
            external_bases.add(model.base_class)

    # We'll leave external imports as comments for now —
    # the Builder (Pass 2) or manual wiring will resolve them
    if external_bases:
        lines.append(f"# TODO: Import external bases: {', '.join(sorted(external_bases))}")
        lines.append("")

    for model in models_in_file:
        # Class definition
        lines.append("")
        if model.docstring:
            lines.append(f'class {model.name}({model.base_class}):')
            lines.append(f'    """{model.docstring}"""')
        else:
            lines.append(f"class {model.name}({model.base_class}):")

        if not model.fields and not model.validators:
            lines.append("    pass")
            lines.append("")
            continue

        # Fields
        for f in model.fields:
            if f.description:
                if f.default is not None:
                    lines.append(
                        f'    {f.name}: {f.type_annotation} = Field(default={f.default}, description="{f.description}")'
                    )
                else:
                    lines.append(
                        f'    {f.name}: {f.type_annotation} = Field(description="{f.description}")'
                    )
            else:
                if f.default is not None:
                    lines.append(f"    {f.name}: {f.type_annotation} = {f.default}")
                else:
                    lines.append(f"    {f.name}: {f.type_annotation}")

        # Validator stubs
        for v in model.validators:
            lines.append("")
            lines.append(f"    def {v}(self):")
            lines.append(f"        pass  # TODO: implement validator")

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

    Handles the common pattern: engine + session + Base for SQLAlchemy 2.x.
    Prevents the recurring __future__ + SQLAlchemy conflict.
    """
    base = file_path.split("/")[-1] if "/" in file_path else file_path
    if base not in ("database.py", "db.py"):
        return None

    # Check if project uses SQLAlchemy
    deps_lower = {d.lower() for d in skeleton.external_dependencies}
    if not any(d in deps_lower for d in ("sqlalchemy", "sqlmodel")):
        return None

    # Generate SQLAlchemy 2.x database setup
    # NOTE: Do NOT use `from __future__ import annotations` — it breaks
    # SQLAlchemy's Mapped type resolution at class definition time.
    return '''"""Database setup — SQLAlchemy 2.x engine, session, and base."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

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
'''


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
        "from pydantic import Field",
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
            lines.append(f"    model_config = dict(env_prefix='{config.env_prefix}')")
            lines.append("")

        if not config.fields:
            lines.append("    pass")
        else:
            for f in config.fields:
                if f.description:
                    if f.default is not None:
                        lines.append(
                            f'    {f.name}: {f.type_annotation} = Field(default={f.default}, description="{f.description}")'
                        )
                    else:
                        lines.append(
                            f'    {f.name}: {f.type_annotation} = Field(description="{f.description}")'
                        )
                else:
                    if f.default is not None:
                        lines.append(f"    {f.name}: {f.type_annotation} = {f.default}")
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
            lines.append(f"    def __init__(self, **kwargs):")
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
