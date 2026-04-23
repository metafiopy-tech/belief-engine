# Pydantic v2 quick reference (injected by Session 2 covenant)

You are generating code for Python 3.14 with **Pydantic v2**. The v1
API is frozen and partially broken on 3.14 (langchain_core's own v1
shim emits a UserWarning on every import). Use the v2 API.

Contrast these patterns. Left column is wrong — will be rewritten by
the covenant enforcer and you will see the edit. Right column is the
target.

## Imports

    ❌ from pydantic.v1 import BaseModel
    ✅ from pydantic import BaseModel

    ❌ from langchain_core.pydantic_v1 import Field
    ✅ from pydantic import Field

    ❌ from pydantic import BaseSettings
    ✅ from pydantic_settings import BaseSettings

## Model config

    ❌ class User(BaseModel):
           name: str
           class Config:
               orm_mode = True
               allow_population_by_field_name = True

    ✅ from pydantic import BaseModel, ConfigDict
       class User(BaseModel):
           model_config = ConfigDict(from_attributes=True, populate_by_name=True)
           name: str

    Config field renames you'll see:
        orm_mode                       → from_attributes
        allow_population_by_field_name → populate_by_name
        schema_extra                   → json_schema_extra
        anystr_strip_whitespace        → str_strip_whitespace

## Validators

    ❌ from pydantic import BaseModel, validator
       class Foo(BaseModel):
           x: int
           @validator("x", pre=True)
           def check(cls, v):
               return v

    ✅ from pydantic import BaseModel, field_validator
       class Foo(BaseModel):
           x: int
           @field_validator("x", mode="before")
           @classmethod                           # REQUIRED in v2
           def check(cls, v):
               return v

    @root_validator → @model_validator(mode="after") or (mode="before")

## Method names on model instances

    ❌ m.dict()            ✅ m.model_dump()
    ❌ m.json()            ✅ m.model_dump_json()
    ❌ M.parse_obj(x)      ✅ M.model_validate(x)
    ❌ M.parse_raw(s)      ✅ M.model_validate_json(s)
    ❌ m.schema()          ✅ m.model_json_schema()
    ❌ m.copy()            ✅ m.model_copy()
    ❌ m.update_forward_refs() ✅ m.model_rebuild()

## Constrained types

    ❌ years: conint(gt=0, lt=150)
    ✅ years: Annotated[int, Field(gt=0, lt=150)]     # from typing import Annotated

    ❌ name: constr(min_length=1)
    ✅ name: Annotated[str, Field(min_length=1)]

## Root models (handle manually — covenant leaves a TODO)

    ❌ class ListModel(BaseModel):
           __root__: list[int]

    ✅ from pydantic import RootModel
       class ListModel(RootModel[list[int]]):
           pass

## What NOT to do

- Do not import from `pydantic.v1`. The covenant enforcer will rewrite
  it and the debugger will see your rewrite only, not the original.
- Do not pin `pydantic<2`. This project requires v2+.
- Do not add stdlib names like `uuid`, `timeit`, `typing`, `json`,
  `datetime` to `requirements.txt`. They are stdlib. The covenant
  enforcer strips them silently.
