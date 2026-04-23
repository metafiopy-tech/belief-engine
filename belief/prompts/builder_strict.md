# Builder — strict quality prompt (Session 4, v3.2)

Appended to the builder's system prompt when the synthesizer router is
enabled.  The intent: reduce the need for a polish pass by demanding
final-quality code upfront.

If the router still fires the synthesizer (tests fail, ruff > 3, CC > 12,
lines > 150), that's a signal that this block wasn't forceful enough —
iterate on the wording, not on adding more polish infrastructure.

---

## MANDATORY QUALITY CONSTRAINTS

Your output is the FINAL deliverable. There is no polish pass. Treat
every line like it's about to be reviewed by a senior engineer.

### Ruff-clean under E, F, B, UP, I

- **E** — syntax / PEP 8 errors (E501 line length is the one exception:
  lines up to 100 chars are acceptable).
- **F** — undefined names, unused imports. Every import must be used.
- **B** — bugbear warnings (mutable default args, except without
  re-raise, etc.).
- **UP** — pyupgrade. Use modern syntax: ``dict`` / ``list`` / ``X | None``
  not ``Dict`` / ``List`` / ``Optional[X]``. Use ``match`` statements
  when they fit. Use f-strings, not ``.format()``.
- **I** — imports sorted. stdlib, third-party, local — one blank line
  between each block. Alphabetical within a block.

### Type hints on every public surface

- Every public function and method has a complete type annotation.
- Use ``from __future__ import annotations`` unless the file imports
  SQLAlchemy ORM types (``Mapped``, ``mapped_column``) — see the
  ``no_future_with_sqlalchemy`` covenant.
- Pydantic models: ``field: type`` is enough; don't re-annotate in
  ``Field(...)`` unless you need a constraint.

### Complexity ≤ 10 per function

- If a function is branching enough to hit cyclomatic complexity 10,
  extract a helper with a clearly-named purpose.
- Refuse to write nested ``if`` / ``for`` / ``try`` pyramids. Flatten
  via early return or lifted helpers.

### No debug residue

- Zero ``TODO`` / ``FIXME`` / ``XXX`` comments.
- Zero commented-out code.
- Zero ``print`` statements for debugging.  Use ``logging`` if a
  runtime signal is genuinely needed.
- Zero ``breakpoint()`` / ``pdb`` calls.

### No preamble, no postscript

- Output ONLY the final code.
- NO "# Here is the implementation:" lead-ins.
- NO "I hope this helps!" trailing prose.
- If the file needs a module docstring, write one — but keep it
  focused on what the module DOES, not what it used to do or how
  you were asked to write it.

---

## Contrastive examples

### ❌ Wrong — long function, unused import, emoji comment, v1 style

```python
from typing import Optional, List
import os
import json

def process_items(items: List[dict], flag: bool = False) -> Optional[dict]:
    # 🚀 Process all the items!
    result = {}
    for item in items:
        if item.get("type") == "a":
            if item.get("value") is not None:
                if item["value"] > 0:
                    if flag:
                        result[item["key"]] = item["value"] * 2
                    else:
                        result[item["key"]] = item["value"]
                else:
                    if flag:
                        result[item["key"]] = 0
                    else:
                        result[item["key"]] = None
        else:
            pass  # TODO: handle non-a types
    if not result:
        return None
    return result
```

### ✅ Right — clear structure, modern syntax, no dead code

```python
from __future__ import annotations


def process_items(items: list[dict], flag: bool = False) -> dict | None:
    """Return a dict of processed items, or None if nothing processed."""
    result = {k: v for item in items for k, v in _process_one(item, flag).items() if v is not None}
    return result or None


def _process_one(item: dict, flag: bool) -> dict:
    if item.get("type") != "a":
        return {}
    value = item.get("value")
    if value is None:
        return {}
    if value > 0:
        return {item["key"]: value * 2 if flag else value}
    return {item["key"]: 0 if flag else None}
```

### ❌ Wrong — pydantic v1, leading prose, debug print

```python
# Here's my implementation of the User model:

from pydantic.v1 import BaseModel, validator

class User(BaseModel):
    name: str
    email: str

    class Config:
        orm_mode = True

    @validator("email")
    def check_email(cls, v):
        print(f"checking {v}")  # debug
        assert "@" in v, "not a valid email"
        return v

# Let me know if you need anything else!
```

### ✅ Right — pydantic v2, no prose, no debug

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    email: str

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("email must contain @")
        return v
```

---

When you finish a file, re-read what you wrote against this list. If
anything would trip ruff or radon, fix it BEFORE emitting. The polish
pass is going to be gated on these exact signals — if they fire, you've
made more work for the pipeline, not less.
