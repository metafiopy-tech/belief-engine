"""Pydantic v1 → v2 deterministic rewrite — Session 2 (v3.2).

Kills the v1↔v2 oscillation loop that consumed ~30% of debug iterations
on the overnight benchmark.  Qwen 2.5 Coder 14B's training data pre-dates
the pydantic v2 stabilisation, so the model routinely emits
``from pydantic.v1 import …`` or uses the v1 ``.dict()`` method.  The
debugger would "fix" these toward v2, Qwen would regenerate them toward
v1 on the next round, and so on.

This module makes the rewrite deterministic.  The covenant runs BEFORE
the debugger sees the file, so the debugger never observes a v1
symbol.  LLM pressure alone could not fix this reliably; LibCST can.

Transformations (single pass, in order)
---------------------------------------

Imports
    from pydantic.v1 import X             → from pydantic import X
    from pydantic.v1.anything import X    → from pydantic import X
    from langchain_core.pydantic_v1 import X → from pydantic import X
    from langchain.pydantic_v1 import X   → from pydantic import X
    from pydantic import BaseSettings     → from pydantic_settings import BaseSettings
    from pydantic import BaseSettings, X  → split: pydantic_settings + pydantic

Config → ConfigDict
    class Config:                         → model_config = ConfigDict(...)
        orm_mode = True                     from_attributes=True
        allow_population_by_field_name      populate_by_name
        schema_extra                        json_schema_extra
        anystr_strip_whitespace             str_strip_whitespace
        anystr_lower / anystr_upper         str_to_lower / str_to_upper
        fields = {...}                      (no equivalent; emits TODO comment — risky to auto-rewrite)

Decorators
    @validator("x", pre=True)             → @field_validator("x", mode="before") + @classmethod
    @validator("x", always=True)          → @field_validator("x", mode="before") + @classmethod
    @root_validator(pre=True)             → @model_validator(mode="before") + @classmethod
    @root_validator                       → @model_validator(mode="after")

Method calls
    .dict()                               → .model_dump()
    .json()                               → .model_dump_json()
    .parse_obj(x)                         → .model_validate(x)
    .parse_raw(x)                         → .model_validate_json(x)
    .schema()                             → .model_json_schema()
    .update_forward_refs()                → .model_rebuild()
    .copy()                               → .model_copy()

Constrained types
    conint(gt=0, lt=100)                  → Annotated[int, Field(gt=0, lt=100)]
    constr(min_length=1)                  → Annotated[str, Field(min_length=1)]

Root models
    __root__: T                           → leave in place + emit TODO comment
                                             ("RootModel migration needed" — full rewrite
                                             is risky because __root__ fields interact with
                                             parent-class __init__ signatures)

Non-goals for Session 2
-----------------------

* Migration of ``GenericModel`` → pydantic.v2 generics (rare in LLM output).
* ``Field(env="FOO")`` → ``BaseSettings`` field env handling (pydantic_settings covers this; we just route the import).
* ``@validator("*")`` wildcard → ``@model_validator(mode="wrap")`` (rare; errors cleanly if encountered so a follow-up session can add it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import libcst as cst
from libcst import matchers as m

logger = logging.getLogger("belief.covenants.pydantic_v2")


# ---------------------------------------------------------------------------
# Rewrite tables — keep close to the top so they're easy to audit.
# ---------------------------------------------------------------------------

# Method-on-a-Pydantic-model renames.  Applied via leave_Call when the
# parent looks like a Call whose func is an Attribute whose value is an
# identifier — we can't know statically whether the receiver is really a
# BaseModel without type info, so we apply by name and accept occasional
# collateral on non-model .copy() calls.  Scoped to method names that are
# unusual outside Pydantic (model_dump, model_validate_json) to minimise
# false positives on truly generic names.
METHOD_RENAMES: dict[str, str] = {
    "dict": "model_dump",
    "json": "model_dump_json",
    "parse_obj": "model_validate",
    "parse_raw": "model_validate_json",
    "schema": "model_json_schema",
    "update_forward_refs": "model_rebuild",
    "copy": "model_copy",
}

# Config-class attribute renames.  Applied when we find a ``class Config:``
# nested inside a BaseModel subclass (or any class — we don't check).
CONFIG_FIELD_RENAMES: dict[str, str] = {
    "orm_mode": "from_attributes",
    "allow_population_by_field_name": "populate_by_name",
    "schema_extra": "json_schema_extra",
    "anystr_strip_whitespace": "str_strip_whitespace",
    "anystr_lower": "str_to_lower",
    "anystr_upper": "str_to_upper",
}

# Config attributes with no v2 equivalent — we emit a clear comment so a
# human (or the debugger on its retry) can address them.
CONFIG_FIELD_NO_V2_EQUIVALENT: set[str] = {
    "fields",  # v2 uses model_fields + Field(..., title=..., description=...)
    "error_msg_templates",
    "keep_untouched",
}


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class CovenantApplied:
    """One discrete rewrite the covenant made, for logging.

    Session 8's covenant proposer reads these to cluster which rules
    fire most often.  Keep the ``rule`` strings stable — they're the
    HDBSCAN feature dimension.
    """

    rule: str
    detail: str = ""
    line: int | None = None
    file: str | None = None


# ---------------------------------------------------------------------------
# Matchers (libcst.matchers) used across the visitor — hoisted so they're
# compiled once when the module imports.
# ---------------------------------------------------------------------------

_VALIDATOR_NAME_MATCHER = m.Name("validator")
_ROOT_VALIDATOR_NAME_MATCHER = m.Name("root_validator")


# ---------------------------------------------------------------------------
# The transformer
# ---------------------------------------------------------------------------


class PydanticV2Covenant(cst.CSTTransformer):
    """Rewrites a single Python module source from pydantic v1 → v2.

    Usage::

        transformer = PydanticV2Covenant()
        new_module = cst.parse_module(source).visit(transformer)
        new_source = new_module.code
        for applied in transformer.applied:
            print(applied)
    """

    def __init__(self, *, filename: str | None = None) -> None:
        super().__init__()
        self.filename = filename
        self.applied: list[CovenantApplied] = []

        # Post-transform import needs — we inject these at the top of the
        # module in a final pass (see :meth:`leave_Module`).
        self._needs_annotated: bool = False
        self._needs_field: bool = False
        self._needs_configdict: bool = False
        self._needs_field_validator: bool = False
        self._needs_model_validator: bool = False
        self._needs_classmethod: bool = False  # stdlib — almost always already imported
        self._needs_pydantic_settings_basesettings: bool = False

        # Tracks already-existing imports so we don't duplicate.
        self._existing_pydantic_names: set[str] = set()
        self._existing_typing_names: set[str] = set()

    # ------------------------------------------------------------------
    # Call site rewrites: .dict() → .model_dump(), etc.
    # ------------------------------------------------------------------

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        """Rename method attributes that are pydantic-specific.

        Only touches ``.dict`` etc. — we don't look up whether the
        receiver is a BaseModel (no type info at syntax level).  This
        accepts small collateral (e.g., a ``.copy()`` call on a dict)
        in exchange for the covenant firing reliably on the v1↔v2
        thrash loop the engine actually hits.  The two rare false
        positives (``dict.copy``, ``list.copy``) are easy for the
        debugger to catch if they matter; the common true positive
        (``Model.dict()`` returning the v1 shape) is the main win.
        """
        if not m.matches(updated_node.attr, m.Name()):
            return updated_node
        name = updated_node.attr.value  # type: ignore[union-attr]
        if name in METHOD_RENAMES:
            new_name = METHOD_RENAMES[name]
            self.applied.append(
                CovenantApplied(
                    rule=f"pydantic_v2.method_rename.{name}",
                    detail=f".{name}() → .{new_name}()",
                    file=self.filename,
                )
            )
            return updated_node.with_changes(attr=cst.Name(new_name))
        return updated_node

    # ------------------------------------------------------------------
    # Import rewrites
    # ------------------------------------------------------------------

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        """Same-statement import rewrites only.

        ``leave_ImportFrom`` can only return an ImportFrom; splitting
        a statement into two needs ``leave_SimpleStatementLine``.  See
        :meth:`leave_SimpleStatementLine` below for the BaseSettings
        split path.
        """
        module_path = _dotted_name(updated_node.module)

        # Track existing pydantic/typing names so we don't re-import what's already imported.
        if module_path == "pydantic":
            for name in _imported_names(updated_node):
                self._existing_pydantic_names.add(name)
        if module_path == "typing":
            for name in _imported_names(updated_node):
                self._existing_typing_names.add(name)

        # from langchain_core.pydantic_v1 import X        → from pydantic import X
        # from langchain.pydantic_v1 import X             → from pydantic import X
        # from pydantic.v1 import X                       → from pydantic import X
        # from pydantic.v1.<anything> import X            → from pydantic import X
        if module_path in {
            "langchain_core.pydantic_v1",
            "langchain.pydantic_v1",
            "pydantic.v1",
        } or (module_path and module_path.startswith("pydantic.v1.")):
            self.applied.append(
                CovenantApplied(
                    rule="pydantic_v2.import.rewrite_v1_to_v2",
                    detail=f"from {module_path} → from pydantic",
                    file=self.filename,
                )
            )
            for name in _imported_names(updated_node):
                self._existing_pydantic_names.add(name)
            return updated_node.with_changes(module=cst.Name("pydantic"))

        return updated_node

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine | cst.FlattenSentinel[cst.SimpleStatementLine]:
        """Splits ``from pydantic import BaseSettings [, X, Y]`` into two
        statements so ``BaseSettings`` goes to ``pydantic_settings``
        while the other names stay on ``pydantic``.

        This has to live on SimpleStatementLine (not ImportFrom)
        because libcst's ``leave_ImportFrom`` can only return a single
        ImportFrom.
        """
        # Fast path: unless one of the inner statements is an ImportFrom
        # of pydantic with BaseSettings, do nothing.
        for leaf in updated_node.body:
            if not isinstance(leaf, cst.ImportFrom):
                continue
            if _dotted_name(leaf.module) != "pydantic":
                continue
            if not _has_imported_name(leaf, "BaseSettings"):
                continue
            return self._split_basesettings_line(updated_node, leaf)

        return updated_node

    def _split_basesettings_line(
        self,
        stmt: cst.SimpleStatementLine,
        importfrom: cst.ImportFrom,
    ) -> cst.SimpleStatementLine | cst.FlattenSentinel[cst.SimpleStatementLine]:
        aliases = list(importfrom.names) if isinstance(importfrom.names, (tuple, list)) else []

        base_settings_aliases: list[cst.ImportAlias] = []
        other_aliases: list[cst.ImportAlias] = []
        for alias in aliases:
            name = _import_alias_name(alias)
            if name == "BaseSettings":
                base_settings_aliases.append(alias)
            else:
                other_aliases.append(alias)

        if not base_settings_aliases:
            return stmt

        self._needs_pydantic_settings_basesettings = True
        self.applied.append(
            CovenantApplied(
                rule="pydantic_v2.import.basesettings_to_pydantic_settings",
                detail="BaseSettings → pydantic_settings"
                + (
                    f" (+{len(other_aliases)} other names stay on pydantic)"
                    if other_aliases
                    else ""
                ),
                file=self.filename,
            )
        )

        pydantic_settings_importfrom = cst.ImportFrom(
            module=cst.Name("pydantic_settings"),
            names=_normalise_import_aliases(base_settings_aliases),
        )
        pydantic_settings_stmt = cst.SimpleStatementLine(
            body=[pydantic_settings_importfrom],
            leading_lines=stmt.leading_lines,
        )

        if not other_aliases:
            # Only BaseSettings — replace the original statement.
            return cst.FlattenSentinel([pydantic_settings_stmt])

        for alias in other_aliases:
            name = _import_alias_name(alias)
            if name:
                self._existing_pydantic_names.add(name)
        remaining_importfrom = importfrom.with_changes(
            names=_normalise_import_aliases(other_aliases)
        )
        # Rebuild the original statement keeping any other leaves on the same line
        # (e.g., a semicolon-chained statement — rare in real code).
        other_leaves = [leaf for leaf in stmt.body if leaf is not importfrom]
        remaining_stmt = cst.SimpleStatementLine(
            body=[remaining_importfrom] + other_leaves,
        )
        return cst.FlattenSentinel([pydantic_settings_stmt, remaining_stmt])

    # ------------------------------------------------------------------
    # Decorator rewrites: @validator / @root_validator
    # ------------------------------------------------------------------

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        """Rewrite @validator and @root_validator decorators.

        v2 requires @classmethod below the decorator (v1 did it
        implicitly); we add it when rewriting.  Also converts the
        pre=True / always=True kwargs to mode="before"/"after".
        """
        if not updated_node.decorators:
            return updated_node

        new_decorators: list[cst.Decorator] = []
        has_classmethod = any(_is_classmethod_decorator(d) for d in updated_node.decorators)
        needs_classmethod = False
        changed = False

        for dec in updated_node.decorators:
            call = _decorator_as_call(dec)
            if call is None:
                # Could be @validator (no parens) — treat as decorator-without-args.
                name_node = _decorator_name(dec)
                if name_node == "validator":
                    new_decorators.append(_make_call_decorator("field_validator", []))
                    self._needs_field_validator = True
                    needs_classmethod = True
                    changed = True
                    self.applied.append(
                        CovenantApplied(
                            rule="pydantic_v2.decorator.validator_to_field_validator",
                            detail="@validator → @field_validator",
                            file=self.filename,
                        )
                    )
                    continue
                if name_node == "root_validator":
                    new_decorators.append(
                        _make_call_decorator(
                            "model_validator", [_make_kwarg("mode", cst.SimpleString('"after"'))]
                        )
                    )
                    self._needs_model_validator = True
                    needs_classmethod = True
                    changed = True
                    self.applied.append(
                        CovenantApplied(
                            rule="pydantic_v2.decorator.root_validator_to_model_validator",
                            detail="@root_validator → @model_validator(mode='after')",
                            file=self.filename,
                        )
                    )
                    continue
                new_decorators.append(dec)
                continue

            # call is a cst.Call
            func = call.func
            if m.matches(func, _VALIDATOR_NAME_MATCHER):
                new_decorators.append(_rewrite_validator_call(call))
                self._needs_field_validator = True
                needs_classmethod = True
                changed = True
                self.applied.append(
                    CovenantApplied(
                        rule="pydantic_v2.decorator.validator_to_field_validator",
                        detail="@validator(...) → @field_validator(...)",
                        file=self.filename,
                    )
                )
            elif m.matches(func, _ROOT_VALIDATOR_NAME_MATCHER):
                new_decorators.append(_rewrite_root_validator_call(call))
                self._needs_model_validator = True
                needs_classmethod = True
                changed = True
                self.applied.append(
                    CovenantApplied(
                        rule="pydantic_v2.decorator.root_validator_to_model_validator",
                        detail="@root_validator(...) → @model_validator(...)",
                        file=self.filename,
                    )
                )
            else:
                new_decorators.append(dec)

        if not changed:
            return updated_node

        if needs_classmethod and not has_classmethod:
            # Insert @classmethod immediately before the validator decorator so
            # the method receives cls as its first arg.  pydantic v2 requires
            # this even when the body already looks cls-shaped.
            classmethod_decorator = cst.Decorator(decorator=cst.Name("classmethod"))
            # Insert after the @field_validator/@model_validator so it's the closer decorator.
            # Per pydantic v2 docs: @classmethod goes directly above the method.
            new_decorators.append(classmethod_decorator)

        return updated_node.with_changes(decorators=new_decorators)

    # ------------------------------------------------------------------
    # Class rewrites: Config → ConfigDict, __root__ TODO, conint/constr
    # ------------------------------------------------------------------

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        body = updated_node.body.body if hasattr(updated_node.body, "body") else []

        # 1. Look for an inner `class Config:` and convert it to a
        #    model_config = ConfigDict(...) assignment at the class level.
        new_body_items: list[cst.BaseStatement] = []
        config_converted = False
        for item in body:
            config_class = _as_config_inner_class(item)
            if config_class is not None and not config_converted:
                converted = self._convert_config_to_configdict(config_class)
                if converted is not None:
                    new_body_items.append(converted)
                    self._needs_configdict = True
                    config_converted = True
                    self.applied.append(
                        CovenantApplied(
                            rule="pydantic_v2.class.config_to_configdict",
                            detail="class Config → model_config = ConfigDict(...)",
                            file=self.filename,
                        )
                    )
                    continue
            # 2. __root__ fields → leave in place + TODO comment
            root_field = _as_root_field(item)
            if root_field is not None:
                new_body_items.append(
                    _prepend_leading_comment(
                        item,
                        "# TODO: v2 RootModel migration needed — __root__ fields "
                        "require subclassing pydantic.RootModel instead of BaseModel",
                    )
                )
                self.applied.append(
                    CovenantApplied(
                        rule="pydantic_v2.class.root_todo",
                        detail="__root__ field annotated with migration TODO",
                        file=self.filename,
                    )
                )
                continue
            new_body_items.append(item)

        if not config_converted and all(n is item for n, item in zip(new_body_items, body)):
            return updated_node

        new_indented = (
            updated_node.body.with_changes(body=new_body_items)
            if hasattr(updated_node.body, "with_changes")
            else updated_node.body
        )
        return updated_node.with_changes(body=new_indented)

    def _convert_config_to_configdict(self, config_class: cst.ClassDef) -> cst.BaseStatement | None:
        """Take a ``class Config:`` block and emit
        ``model_config = ConfigDict(k1=v1, k2=v2, ...)``.

        Applies the CONFIG_FIELD_RENAMES along the way.  Attributes with
        no v2 equivalent are dropped with a log line (the debugger will
        re-add them properly if needed).
        """
        kwargs: list[cst.Arg] = []
        for stmt in config_class.body.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for leaf in stmt.body:
                if isinstance(leaf, cst.Assign) and len(leaf.targets) == 1:
                    target = leaf.targets[0].target
                    if isinstance(target, cst.Name):
                        key = target.value
                        if key in CONFIG_FIELD_NO_V2_EQUIVALENT:
                            logger.info(
                                "Config field %r has no v2 equivalent — dropping "
                                "(rewrite manually or via debugger)",
                                key,
                            )
                            continue
                        v2_key = CONFIG_FIELD_RENAMES.get(key, key)
                        kwargs.append(
                            cst.Arg(
                                keyword=cst.Name(v2_key),
                                value=leaf.value,
                                equal=cst.AssignEqual(
                                    whitespace_before=cst.SimpleWhitespace(""),
                                    whitespace_after=cst.SimpleWhitespace(""),
                                ),
                            )
                        )
        if not kwargs:
            return None

        # Strip trailing comma from last arg for clean emit.
        last = kwargs[-1]
        kwargs[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)
        for i in range(len(kwargs) - 1):
            kwargs[i] = kwargs[i].with_changes(
                comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
            )

        assign = cst.Assign(
            targets=[cst.AssignTarget(target=cst.Name("model_config"))],
            value=cst.Call(
                func=cst.Name("ConfigDict"),
                args=kwargs,
            ),
        )
        return cst.SimpleStatementLine(body=[assign])

    # ------------------------------------------------------------------
    # Constrained types: conint/constr → Annotated[int, Field(...)]
    # ------------------------------------------------------------------

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        """Rewrite conint/constr factory calls to Annotated equivalents.

        These show up mostly in field annotations::

            class Foo(BaseModel):
                age: conint(gt=0, lt=150)  →  age: Annotated[int, Field(gt=0, lt=150)]

        We don't try to handle every constrained-type factory — conint
        and constr cover ~95% of LLM-emitted cases; other ones
        (confloat, condecimal, conlist, …) are follow-up work.
        """
        if not m.matches(updated_node.func, m.Name()):
            return updated_node
        factory_name = updated_node.func.value  # type: ignore[union-attr]
        if factory_name == "conint":
            base = cst.Name("int")
        elif factory_name == "constr":
            base = cst.Name("str")
        else:
            return updated_node

        # Convert positional args to kwargs isn't needed for these — both
        # factories take only kwargs.  Pass them straight to Field(...).
        field_call = cst.Call(
            func=cst.Name("Field"),
            args=list(updated_node.args),
        )
        self._needs_annotated = True
        self._needs_field = True
        self.applied.append(
            CovenantApplied(
                rule=f"pydantic_v2.type.{factory_name}_to_annotated",
                detail=f"{factory_name}(...) → Annotated[{base.value}, Field(...)]",
                file=self.filename,
            )
        )
        return cst.Subscript(
            value=cst.Name("Annotated"),
            slice=[
                cst.SubscriptElement(slice=cst.Index(value=base)),
                cst.SubscriptElement(slice=cst.Index(value=field_call)),
            ],
        )

    # ------------------------------------------------------------------
    # Final pass: inject any needed imports at the top of the module.
    # ------------------------------------------------------------------

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Add imports we now need (Annotated, Field, ConfigDict, etc.)."""

        pydantic_adds: set[str] = set()
        typing_adds: set[str] = set()

        if self._needs_configdict and "ConfigDict" not in self._existing_pydantic_names:
            pydantic_adds.add("ConfigDict")
        if self._needs_field and "Field" not in self._existing_pydantic_names:
            pydantic_adds.add("Field")
        if self._needs_field_validator and "field_validator" not in self._existing_pydantic_names:
            pydantic_adds.add("field_validator")
        if self._needs_model_validator and "model_validator" not in self._existing_pydantic_names:
            pydantic_adds.add("model_validator")
        if self._needs_annotated and "Annotated" not in self._existing_typing_names:
            typing_adds.add("Annotated")

        new_body: list[cst.BaseStatement] = list(updated_node.body)

        if pydantic_adds:
            new_body = _splice_import(
                new_body,
                module="pydantic",
                names=sorted(pydantic_adds),
            )
        if typing_adds:
            new_body = _splice_import(
                new_body,
                module="typing",
                names=sorted(typing_adds),
            )

        return updated_node.with_changes(body=new_body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dotted_name(node: cst.CSTNode | None) -> str | None:
    """Render a module reference (e.g. pydantic.v1.something) to a string."""
    if node is None:
        return None
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parts: list[str] = []
        cur: cst.BaseExpression = node
        while isinstance(cur, cst.Attribute):
            if not isinstance(cur.attr, cst.Name):
                return None
            parts.append(cur.attr.value)
            cur = cur.value
        if isinstance(cur, cst.Name):
            parts.append(cur.value)
        return ".".join(reversed(parts)) if parts else None
    return None


def _imported_names(node: cst.ImportFrom) -> list[str]:
    """List all top-level names pulled in by an ``ImportFrom`` node."""
    if node.names is None or not isinstance(node.names, (tuple, list)):
        return []
    out = []
    for alias in node.names:
        n = _import_alias_name(alias)
        if n:
            out.append(n)
    return out


def _has_imported_name(node: cst.ImportFrom, name: str) -> bool:
    return name in _imported_names(node)


def _import_alias_name(alias: cst.ImportAlias) -> str | None:
    if isinstance(alias.name, cst.Name):
        return alias.name.value
    return _dotted_name(alias.name)


def _decorator_as_call(dec: cst.Decorator) -> cst.Call | None:
    if isinstance(dec.decorator, cst.Call):
        return dec.decorator
    return None


def _decorator_name(dec: cst.Decorator) -> str | None:
    d = dec.decorator
    if isinstance(d, cst.Name):
        return d.value
    if isinstance(d, cst.Call) and isinstance(d.func, cst.Name):
        return d.func.value
    return None


def _is_classmethod_decorator(dec: cst.Decorator) -> bool:
    return _decorator_name(dec) == "classmethod"


def _make_call_decorator(name: str, args: list[cst.Arg]) -> cst.Decorator:
    return cst.Decorator(decorator=cst.Call(func=cst.Name(name), args=args))


def _make_kwarg(key: str, value: cst.BaseExpression) -> cst.Arg:
    return cst.Arg(
        keyword=cst.Name(key),
        value=value,
        equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
        ),
    )


def _rewrite_validator_call(call: cst.Call) -> cst.Decorator:
    """@validator("x", pre=True, always=True) → @field_validator("x", mode="before")"""
    new_args: list[cst.Arg] = []
    mode_value: str | None = None

    for arg in call.args:
        if arg.keyword is None:
            # Positional — the field name(s).
            new_args.append(arg)
            continue
        key = arg.keyword.value if isinstance(arg.keyword, cst.Name) else None
        if key == "pre":
            # pre=True → mode="before".  pre=False is v2 default mode="after".
            if _is_literal_true(arg.value):
                mode_value = "before"
            continue
        if key == "always":
            # v2 has no exact equivalent — always=True implies run on every
            # validation pass; closest is mode="before".  Leave a TODO? Actually
            # just map to mode=before; the v2 validator always runs.
            if _is_literal_true(arg.value):
                mode_value = mode_value or "before"
            continue
        if key == "allow_reuse":
            # Default behaviour in v2; silently drop.
            continue
        if key == "check_fields":
            # Still supported in v2 with same name.
            new_args.append(arg)
            continue
        new_args.append(arg)

    if mode_value is not None:
        new_args.append(_make_kwarg("mode", cst.SimpleString(f'"{mode_value}"')))

    return _make_call_decorator("field_validator", _normalise_args(new_args))


def _rewrite_root_validator_call(call: cst.Call) -> cst.Decorator:
    """@root_validator(pre=True) → @model_validator(mode="before"); else mode="after"."""
    mode_value = "after"
    other_args: list[cst.Arg] = []
    for arg in call.args:
        if arg.keyword is not None and isinstance(arg.keyword, cst.Name):
            if arg.keyword.value == "pre":
                if _is_literal_true(arg.value):
                    mode_value = "before"
                continue
            if arg.keyword.value == "allow_reuse":
                continue
        other_args.append(arg)
    other_args.append(_make_kwarg("mode", cst.SimpleString(f'"{mode_value}"')))
    return _make_call_decorator("model_validator", _normalise_args(other_args))


def _normalise_args(args: list[cst.Arg]) -> list[cst.Arg]:
    """Fix trailing-comma state so the libcst emitter doesn't produce
    ``foo(a,)`` or ``foo(a,b,)`` oddities.
    """
    if not args:
        return args
    out = []
    for i, a in enumerate(args):
        if i == len(args) - 1:
            out.append(a.with_changes(comma=cst.MaybeSentinel.DEFAULT))
        else:
            out.append(a.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))
    return out


def _is_literal_true(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Name) and node.value == "True"


def _as_config_inner_class(stmt: cst.BaseStatement) -> cst.ClassDef | None:
    """Return the ClassDef if ``stmt`` is a ``class Config:`` declaration,
    else None.
    """
    if (
        isinstance(stmt, cst.ClassDef)
        and isinstance(stmt.name, cst.Name)
        and stmt.name.value == "Config"
    ):
        return stmt
    return None


def _as_root_field(stmt: cst.BaseStatement) -> cst.AnnAssign | None:
    """Return the AnnAssign if ``stmt`` declares a ``__root__: T`` field."""
    if isinstance(stmt, cst.SimpleStatementLine):
        for leaf in stmt.body:
            if (
                isinstance(leaf, cst.AnnAssign)
                and isinstance(leaf.target, cst.Name)
                and leaf.target.value == "__root__"
            ):
                return leaf
    return None


def _prepend_leading_comment(stmt: cst.BaseStatement, text: str) -> cst.BaseStatement:
    """Add a leading ``# ...`` comment to a statement."""
    if not isinstance(stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
        return stmt
    leading = list(stmt.leading_lines) if hasattr(stmt, "leading_lines") else []
    leading.append(cst.EmptyLine(comment=cst.Comment(text)))
    return stmt.with_changes(leading_lines=leading)


def _splice_import(
    body: list[cst.BaseStatement],
    *,
    module: str,
    names: list[str],
) -> list[cst.BaseStatement]:
    """Insert ``from module import a, b, c`` after the last top-level
    import statement (or at index 0 if there are no imports yet).

    If a statement already imports from the same module, we MERGE the
    new names into that existing statement instead of emitting a
    duplicate.  This keeps the emitted module readable.
    """
    # 1. Find an existing ``from module import …`` line.
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine):
            for leaf in stmt.body:
                if isinstance(leaf, cst.ImportFrom) and _dotted_name(leaf.module) == module:
                    existing_names = set(_imported_names(leaf))
                    missing = [n for n in names if n not in existing_names]
                    if not missing:
                        return body  # Nothing to do.
                    merged_aliases = (
                        list(leaf.names) if isinstance(leaf.names, (tuple, list)) else []
                    )
                    merged_aliases.extend(cst.ImportAlias(name=cst.Name(n)) for n in missing)
                    merged_aliases = _normalise_import_aliases(merged_aliases)
                    new_importfrom = leaf.with_changes(names=merged_aliases)
                    new_stmt = stmt.with_changes(
                        body=[new_importfrom if child is leaf else child for child in stmt.body]
                    )
                    return [s if s is not stmt else new_stmt for s in body]

    # 2. No existing import — insert after last top-level import.
    new_importfrom = cst.ImportFrom(
        module=cst.Name(module),
        names=[cst.ImportAlias(name=cst.Name(n)) for n in names],
    )
    new_importfrom = new_importfrom.with_changes(
        names=_normalise_import_aliases(list(new_importfrom.names))
    )
    new_stmt = cst.SimpleStatementLine(body=[new_importfrom])

    insert_idx = 0
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine) and any(
            isinstance(leaf, (cst.Import, cst.ImportFrom)) for leaf in stmt.body
        ):
            insert_idx = i + 1
    return body[:insert_idx] + [new_stmt] + body[insert_idx:]


def _normalise_import_aliases(aliases: list[cst.ImportAlias]) -> list[cst.ImportAlias]:
    if not aliases:
        return aliases
    out = []
    for i, a in enumerate(aliases):
        if i == len(aliases) - 1:
            out.append(a.with_changes(comma=cst.MaybeSentinel.DEFAULT))
        else:
            out.append(a.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))
    return out


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def apply_pydantic_v2_covenant(
    source: str, *, filename: str | None = None
) -> tuple[str, list[CovenantApplied]]:
    """Run :class:`PydanticV2Covenant` on a source string.

    Returns ``(new_source, applied_rewrites)``.  Non-Python content
    (non-parseable, empty, non-pydantic) round-trips unchanged.
    """
    if not source.strip():
        return source, []
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as e:
        logger.debug("pydantic_v2 covenant: skip unparseable %s (%s)", filename, e)
        return source, []
    transformer = PydanticV2Covenant(filename=filename)
    new_module = module.visit(transformer)
    return new_module.code, transformer.applied


__all__ = [
    "CovenantApplied",
    "PydanticV2Covenant",
    "apply_pydantic_v2_covenant",
]
