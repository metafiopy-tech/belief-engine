"""Hermetic tests for the Session 2 (v3.2) covenant pipeline.

No LLM calls, no network, no Ollama.  Every test is a
before → after source-string comparison run through the full
:func:`belief.covenants.enforce_python_covenants` pipeline.

The tests are organised by rewrite category to mirror the session
doc's acceptance criteria.  A handful of extra "guardrail" tests
verify idempotence and no-ops on unrelated code.

Run with::

    python3 -m pytest tests/test_pydantic_covenant.py -v
"""

from __future__ import annotations

import textwrap


from belief.covenants import (
    enforce_python_covenants,
    enforce_python_covenants_on_files,
)
from belief.covenants.forbidden_imports import (
    is_stdlib_name,
)
from belief.covenants.pydantic_v2 import apply_pydantic_v2_covenant


def _run(src: str, filename: str = "my_models.py") -> tuple[str, list]:
    return enforce_python_covenants(textwrap.dedent(src), filename=filename)


# ---------------------------------------------------------------------------
# 1. Import rewrites
# ---------------------------------------------------------------------------


class TestImportRewrites:
    def test_pydantic_v1_to_v2_basic(self) -> None:
        src = "from pydantic.v1 import BaseModel"
        out, applied = _run(src)
        assert "pydantic.v1" not in out
        assert "from pydantic import BaseModel" in out
        assert any(a.rule.startswith("pydantic_v2.import") for a in applied)

    def test_pydantic_v1_submodule_flattened(self) -> None:
        src = "from pydantic.v1.fields import Field"
        out, _ = _run(src)
        assert "pydantic.v1" not in out
        assert "from pydantic import Field" in out

    def test_langchain_core_pydantic_v1_rewrite(self) -> None:
        src = "from langchain_core.pydantic_v1 import Field"
        out, applied = _run(src)
        assert "langchain_core.pydantic_v1" not in out
        assert "from pydantic import Field" in out
        assert applied  # something was applied

    def test_langchain_pydantic_v1_rewrite(self) -> None:
        src = "from langchain.pydantic_v1 import BaseModel"
        out, _ = _run(src)
        assert "langchain.pydantic_v1" not in out
        assert "from pydantic import BaseModel" in out

    def test_basesettings_moves_to_pydantic_settings(self) -> None:
        src = "from pydantic import BaseSettings"
        out, _ = _run(src)
        assert "from pydantic_settings import BaseSettings" in out
        # ...and the old pydantic-import line for BaseSettings must be gone.
        assert "from pydantic import BaseSettings" not in out

    def test_basesettings_split_preserves_other_pydantic_names(self) -> None:
        src = "from pydantic import BaseSettings, Field, BaseModel"
        out, _ = _run(src)
        assert "from pydantic_settings import BaseSettings" in out
        # Both Field and BaseModel stay on pydantic.
        assert "from pydantic import" in out
        assert "Field" in out
        assert "BaseModel" in out


# ---------------------------------------------------------------------------
# 2. Config class → ConfigDict
# ---------------------------------------------------------------------------


class TestConfigToConfigDict:
    def test_orm_mode_becomes_from_attributes(self) -> None:
        src = """
            from pydantic import BaseModel

            class User(BaseModel):
                name: str

                class Config:
                    orm_mode = True
        """
        out, _ = _run(src)
        assert "model_config = ConfigDict(" in out
        assert "from_attributes=True" in out
        assert "orm_mode" not in out
        # Inner class Config should be gone.
        assert "class Config" not in out

    def test_all_field_renames(self) -> None:
        src = """
            from pydantic import BaseModel

            class X(BaseModel):
                name: str
                class Config:
                    orm_mode = True
                    allow_population_by_field_name = True
                    anystr_strip_whitespace = True
                    schema_extra = {"example": 1}
        """
        out, _ = _run(src)
        # All the v1 names must disappear as *isolated identifiers*.
        # We check for the `name=` form in the ConfigDict call because
        # some v1 names are substrings of their v2 replacements
        # (e.g., `schema_extra` is a substring of `json_schema_extra`).
        import re

        for v1 in (
            "orm_mode",
            "allow_population_by_field_name",
            "anystr_strip_whitespace",
            "schema_extra",
        ):
            pattern = rf"(?<![_A-Za-z0-9]){v1}\s*="
            assert not re.search(pattern, out), (
                f"{v1} should have been renamed but appears as kwarg in: {out}"
            )
        for v2 in (
            "from_attributes",
            "populate_by_name",
            "str_strip_whitespace",
            "json_schema_extra",
        ):
            assert v2 in out, f"{v2} should be present"


# ---------------------------------------------------------------------------
# 3. Validator decorators
# ---------------------------------------------------------------------------


class TestValidatorRewrites:
    def test_validator_pre_true_becomes_field_validator_mode_before(self) -> None:
        src = """
            from pydantic import BaseModel, validator

            class Foo(BaseModel):
                x: int

                @validator("x", pre=True)
                def check(cls, v):
                    return v
        """
        out, _ = _run(src)
        assert "@field_validator" in out
        assert 'mode="before"' in out
        assert "@classmethod" in out
        assert "@validator" not in out  # old decorator gone

    def test_validator_without_parens(self) -> None:
        # Bare @validator (no call) — rare but we still handle it.
        src = """
            from pydantic import BaseModel, validator

            class Foo(BaseModel):
                x: int

                @validator("x")
                def check(cls, v):
                    return v
        """
        out, _ = _run(src)
        assert "@field_validator" in out
        assert "@classmethod" in out
        assert "@validator" not in out

    def test_root_validator_pre_becomes_model_validator_before(self) -> None:
        src = """
            from pydantic import BaseModel, root_validator

            class Foo(BaseModel):
                x: int

                @root_validator(pre=True)
                def check(cls, values):
                    return values
        """
        out, _ = _run(src)
        assert "@model_validator" in out
        assert 'mode="before"' in out
        assert "@classmethod" in out

    def test_root_validator_default_becomes_model_validator_after(self) -> None:
        src = """
            from pydantic import BaseModel, root_validator

            class Foo(BaseModel):
                x: int

                @root_validator
                def check(cls, values):
                    return values
        """
        out, _ = _run(src)
        assert "@model_validator" in out
        assert 'mode="after"' in out


# ---------------------------------------------------------------------------
# 4. Method rewrites
# ---------------------------------------------------------------------------


class TestMethodRewrites:
    def test_dict_and_json_rewritten(self) -> None:
        # Force the prepass to fire by including a pydantic import — the
        # prepass is conservative and won't touch non-pydantic code on
        # purpose (idempotence; see TestGuardrails).
        src = """
            from pydantic import BaseModel

            m = BaseModel()
            a = m.dict()
            b = m.json()
        """
        out, _ = _run(src)
        assert ".model_dump()" in out
        assert ".model_dump_json()" in out

    def test_parse_obj_and_parse_raw(self) -> None:
        src = """
            from pydantic import BaseModel

            class M(BaseModel):
                x: int

            M.parse_obj({"x": 1})
            M.parse_raw('{"x": 1}')
        """
        out, _ = _run(src)
        assert ".model_validate(" in out
        assert ".model_validate_json(" in out
        assert ".parse_obj(" not in out
        assert ".parse_raw(" not in out


# ---------------------------------------------------------------------------
# 5. __root__ — TODO comment, NOT rewrite
# ---------------------------------------------------------------------------


class TestRootField:
    def test_root_field_gets_todo_not_rewrite(self) -> None:
        src = """
            from pydantic import BaseModel

            class Wrapper(BaseModel):
                __root__: list[int]
        """
        out, applied = _run(src)
        # The __root__ field is intentionally left in place — the
        # TODO comment is the signal, not a structural rewrite.
        assert "__root__" in out
        assert "TODO" in out and "RootModel migration" in out
        assert any(a.rule == "pydantic_v2.class.root_todo" for a in applied)


# ---------------------------------------------------------------------------
# 6. Constrained types
# ---------------------------------------------------------------------------


class TestConstrainedTypes:
    def test_conint_becomes_annotated(self) -> None:
        src = """
            from pydantic import BaseModel, conint

            class Age(BaseModel):
                years: conint(gt=0, lt=150)
        """
        out, _ = _run(src)
        assert "Annotated[int, Field(gt=0, lt=150)]" in out
        assert "from typing import" in out and "Annotated" in out

    def test_constr_becomes_annotated(self) -> None:
        src = """
            from pydantic import BaseModel, constr

            class Name(BaseModel):
                value: constr(min_length=1, max_length=50)
        """
        out, _ = _run(src)
        assert "Annotated[str, Field(" in out
        assert "min_length=1" in out


# ---------------------------------------------------------------------------
# 7. Stdlib stripped from requirements.txt
# ---------------------------------------------------------------------------


class TestForbiddenImports:
    def test_timeit_stripped(self) -> None:
        src = "fastapi==0.100.0\ntimeit\npydantic>=2\n"
        out, applied = enforce_python_covenants(src, filename="requirements.txt")
        assert "timeit" not in out
        assert "fastapi" in out
        assert "pydantic" in out
        assert any("stdlib" in a.rule for a in applied)

    def test_uuid_stripped(self) -> None:
        src = "uuid\nrequests==2.28\n"
        out, _ = enforce_python_covenants(src, filename="requirements.txt")
        assert out.splitlines()[0].startswith("requests")
        assert "uuid" not in out

    def test_non_requirements_file_passthrough(self) -> None:
        # A .py file named 'requirements' should NOT have stdlib refs
        # stripped — those would be legitimate imports.
        src = "import timeit\ntimeit.default_timer()"
        out, _ = enforce_python_covenants(src, filename="my_benchmark.py")
        assert "import timeit" in out

    def test_is_stdlib_name_known_cases(self) -> None:
        assert is_stdlib_name("timeit")
        assert is_stdlib_name("uuid")
        assert is_stdlib_name("json")
        assert not is_stdlib_name("requests")
        assert not is_stdlib_name("pydantic")
        assert not is_stdlib_name("fastapi")


# ---------------------------------------------------------------------------
# 8. Idempotence + non-pydantic passthrough (guardrails)
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_non_pydantic_code_unchanged(self) -> None:
        src = textwrap.dedent("""
            def fibonacci(n: int) -> int:
                if n < 2:
                    return n
                return fibonacci(n - 1) + fibonacci(n - 2)

            print(fibonacci(10))
        """).lstrip()
        out, applied = enforce_python_covenants(src, filename="fib.py")
        # The prepass should short-circuit — no rewrites, identical output.
        assert out == src
        assert applied == []

    def test_idempotent_on_already_v2_code(self) -> None:
        src = textwrap.dedent("""
            from pydantic import BaseModel, ConfigDict, field_validator

            class User(BaseModel):
                model_config = ConfigDict(from_attributes=True)
                name: str

                @field_validator("name")
                @classmethod
                def check(cls, v):
                    return v
        """).lstrip()
        out1, _ = enforce_python_covenants(src, filename="u.py")
        out2, _ = enforce_python_covenants(out1, filename="u.py")
        # Running twice gives the same result — the covenant doesn't
        # drift on already-v2 code.
        assert out1 == out2

    def test_unparseable_source_round_trips_unchanged(self) -> None:
        src = "this is not valid python ::: ¯\\_(ツ)_/¯"
        out, applied = apply_pydantic_v2_covenant(src)
        assert out == src
        assert applied == []


# ---------------------------------------------------------------------------
# 9. Bulk entry point (what belief.graph calls)
# ---------------------------------------------------------------------------


class TestBulkEntryPoint:
    def test_mixed_files_handled_per_type(self) -> None:
        files = {
            "models.py": "from pydantic.v1 import BaseModel\nclass F(BaseModel):\n    x: int\n",
            "requirements.txt": "fastapi\ntimeit\npydantic>=2\nuuid\n",
            "fib.py": "def fib(n):\n    return 1 if n<2 else fib(n-1)+fib(n-2)\n",
        }
        fixed, applied = enforce_python_covenants_on_files(files)
        # Python file: v1 rewritten
        assert "pydantic.v1" not in fixed["models.py"]
        assert "from pydantic import BaseModel" in fixed["models.py"]
        # requirements.txt: stdlib stripped
        assert "timeit" not in fixed["requirements.txt"]
        assert "uuid" not in fixed["requirements.txt"]
        assert "fastapi" in fixed["requirements.txt"]
        # Non-pydantic file: unchanged
        assert fixed["fib.py"] == files["fib.py"]
        # At least three rules fired (v1-rewrite, stdlib x2).
        assert len(applied) >= 3


# ---------------------------------------------------------------------------
# 10. Cheatsheet trigger — prompts/cheatsheets.py helper
# ---------------------------------------------------------------------------


class TestCheatsheetTrigger:
    def test_triggers_on_models_filename(self) -> None:
        from belief.prompts.cheatsheets import should_inject_pydantic_v2_cheatsheet

        assert should_inject_pydantic_v2_cheatsheet(filename="src/models.py")
        assert should_inject_pydantic_v2_cheatsheet(filename="app/schemas.py")
        assert should_inject_pydantic_v2_cheatsheet(filename="settings.py")

    def test_does_not_trigger_on_plain_filename(self) -> None:
        from belief.prompts.cheatsheets import should_inject_pydantic_v2_cheatsheet

        assert not should_inject_pydantic_v2_cheatsheet(filename="fib.py")
        assert not should_inject_pydantic_v2_cheatsheet(filename="main.py")

    def test_triggers_on_pydantic_import(self) -> None:
        from belief.prompts.cheatsheets import should_inject_pydantic_v2_cheatsheet

        assert should_inject_pydantic_v2_cheatsheet(planned_imports=["pydantic", "os"])
        assert should_inject_pydantic_v2_cheatsheet(planned_imports=["langchain_core.tools"])

    def test_triggers_on_fastapi_goal(self) -> None:
        from belief.prompts.cheatsheets import should_inject_pydantic_v2_cheatsheet

        assert should_inject_pydantic_v2_cheatsheet(
            user_goal="Build a FastAPI service that stores users",
        )

    def test_cheatsheet_text_has_v1_v2_contrast(self) -> None:
        from belief.prompts.cheatsheets import load_pydantic_v2_cheatsheet

        text = load_pydantic_v2_cheatsheet()
        assert text, "cheatsheet file must ship with the package"
        # Basic sanity — must contain at least one Wrong/Right pair.
        assert "orm_mode" in text
        assert "from_attributes" in text
        # Must document the @classmethod requirement.
        assert "@classmethod" in text
