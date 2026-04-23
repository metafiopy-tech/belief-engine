"""Session 2 (v3.2) — deterministic covenant pipeline.

Runs upstream of the debugger so Qwen's v1↔v2 oscillation loop can't
survive.  The 4-stage pipeline is applied to every generated Python
file AND to requirements.txt before those files are seen by the
executor, tester, or debugger.

Stage 1 — regex prepass (fast-path)
    Cheap substring / regex checks to decide whether the expensive
    Stages 2-4 need to run at all.  If a file doesn't mention
    ``pydantic`` / ``langchain`` / ``BaseSettings`` anywhere, we skip
    LibCST parsing + ruff + bump-pydantic entirely.

Stage 2 — LibCST transformers
    :class:`~belief.covenants.pydantic_v2.PydanticV2Covenant` rewrites
    imports, Config → ConfigDict, validators, method calls, constrained
    types.  :func:`~belief.covenants.forbidden_imports.apply_forbidden_imports_covenant`
    strips stdlib names from requirements.txt.  This stage is the
    largest behavioural change; Stages 3-4 just clean up what it
    leaves behind.

Stage 3 — ruff --fix
    ``ruff check --select UP,F401,F811,I --fix --stdin-filename <name>``
    removes unused imports (UP / F401), unpins legacy syntax, sorts
    imports (I).  Subprocess — tolerates ruff being missing by
    skipping the stage with a debug log.

Stage 4 — bump-pydantic CLI
    ``bump-pydantic --no-input <path>`` — only invoked when Stage 1
    saw a pydantic import AND bump-pydantic is on $PATH.  It catches
    patterns the LibCST transformer is conservative about (notably
    some GenericModel shapes).  Subprocess — tolerates missing CLI.

Public API
----------

    enforce_python_covenants(source: str, filename: str = "") →
        tuple[str, list[CovenantApplied]]

    enforce_python_covenants_on_files(code_files: dict[str, str]) →
        tuple[dict[str, str], list[CovenantApplied]]
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from belief.covenants.forbidden_imports import (
    apply_forbidden_imports_covenant,
)
from belief.covenants.pydantic_v2 import (
    CovenantApplied as _PydanticCovenantApplied,
    apply_pydantic_v2_covenant,
)

# Both covenants define a CovenantApplied dataclass with the same
# schema; expose the pydantic_v2 one as the public type so callers
# don't have to reason about which module it came from.
CovenantApplied = _PydanticCovenantApplied

logger = logging.getLogger("belief.covenants")


# ---------------------------------------------------------------------------
# Stage 1 — regex prepass
# ---------------------------------------------------------------------------

# If any of these patterns match, the file is a candidate for the full
# pipeline.  Otherwise we skip LibCST entirely (keeps covenant cost
# negligible on the many non-pydantic files in a typical build).
_PREPASS_TRIGGERS = re.compile(
    r"""(?x)
    \b(
        pydantic(\.v1|_settings)?        # pydantic / pydantic.v1 / pydantic_settings
        | langchain(_core)?\.pydantic_v1 # langchain_core.pydantic_v1 etc.
        | @validator\b
        | @root_validator\b
        | \.dict\(\)                     # .dict() method call
        | \.parse_obj\(
        | \.parse_raw\(
        | BaseSettings\b
        | conint\(
        | constr\(
        | __root__
        | class\s+Config\s*:             # inner Config class
    )
    """
)


def _prepass_should_run_pydantic(source: str) -> bool:
    """Cheap substring test: does this source look like it might need
    the pydantic_v2 transformer?

    A false negative here means a v1 pattern slips through to the
    debugger.  A false positive only costs extra LibCST parse time.
    So we bias toward running — the patterns above are broad on purpose.
    """
    return bool(_PREPASS_TRIGGERS.search(source))


def _source_imports_pydantic(source: str) -> bool:
    """Stricter test for Stage 4: only invoke bump-pydantic if the file
    actually imports pydantic in any form.  Avoids subprocess cost on
    files that happened to mention ``BaseSettings`` in a comment.
    """
    if "import pydantic" in source:
        return True
    if re.search(r"from\s+pydantic(\.|\s+import)", source):
        return True
    if re.search(r"from\s+pydantic_settings\s+import", source):
        return True
    return False


# ---------------------------------------------------------------------------
# Stage 3 — ruff --fix
# ---------------------------------------------------------------------------


def _run_ruff_fix(source: str, filename: str) -> tuple[str, bool]:
    """Pipe ``source`` through ``ruff check --select UP,F401,F811,I --fix --stdin-filename …``.

    Returns ``(new_source, changed)``.  If ruff isn't on PATH or the
    subprocess errors, returns ``(source, False)`` without raising —
    covenants are upstream of the debugger, not upstream of ``pip``, so
    skipping a stage never blocks a build.
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        logger.debug("ruff not on PATH; skipping Stage 3")
        return source, False
    try:
        proc = subprocess.run(
            [
                ruff,
                "check",
                "--select",
                "UP,F401,F811,I",
                "--fix",
                "--fix-only",
                "--stdin-filename",
                filename,
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ruff check timed out on %s; skipping Stage 3", filename)
        return source, False
    except OSError as e:  # pragma: no cover
        logger.debug("ruff invocation failed on %s: %s", filename, e)
        return source, False

    # ruff --fix writes the transformed code to stdout; non-zero exit
    # codes can still mean "fixed some, left some unfixable" — don't
    # treat that as a hard failure.  If stdout is empty we fall back.
    if not proc.stdout:
        return source, False
    new_source = proc.stdout
    return new_source, new_source != source


# ---------------------------------------------------------------------------
# Stage 4 — bump-pydantic
# ---------------------------------------------------------------------------


def _run_bump_pydantic(source: str, filename: str) -> tuple[str, bool]:
    """Run bump-pydantic on a single file via a tempdir.

    bump-pydantic operates on paths, not stdin, so we drop the source
    into a temp file, run the CLI, read back the result.  Subprocess
    timeout is 20s — this is a slow tool and we don't want to let it
    hold up a build.
    """
    bump = shutil.which("bump-pydantic")
    if bump is None:
        logger.debug("bump-pydantic not on PATH; skipping Stage 4")
        return source, False
    with tempfile.TemporaryDirectory(prefix="belief_covenant_") as td:
        tmp_path = Path(td) / (Path(filename).name or "file.py")
        tmp_path.write_text(source)
        try:
            proc = subprocess.run(
                [bump, "--no-input", str(tmp_path.parent)],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            logger.warning("bump-pydantic timed out on %s; skipping Stage 4", filename)
            return source, False
        except OSError as e:  # pragma: no cover
            logger.debug("bump-pydantic invocation failed: %s", e)
            return source, False
        if proc.returncode not in (0, 1):
            # 0 = no changes, 1 = changes applied, other codes = errors.
            logger.debug(
                "bump-pydantic exit=%d on %s; ignoring output",
                proc.returncode,
                filename,
            )
            return source, False
        new_source = tmp_path.read_text()
        return new_source, new_source != source


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def enforce_python_covenants(
    source: str,
    filename: str = "",
) -> tuple[str, list[CovenantApplied]]:
    """Run the 4-stage covenant pipeline on a single file.

    See module docstring for stage details.  Non-Python / non-requirements
    files round-trip unchanged with an empty applied list.
    """
    applied: list[CovenantApplied] = []

    # Route requirements.txt-style files to the forbidden_imports path.
    if filename and (
        filename.lower().endswith(".txt")
        and ("requirement" in filename.lower() or "constraint" in filename.lower())
    ):
        new_source, reqs_applied = apply_forbidden_imports_covenant(source, filename=filename)
        applied.extend(reqs_applied)
        return new_source, applied

    # Python source path.  Stage 1 — prepass:
    if not filename.endswith(".py") and filename:
        return source, []
    if not _prepass_should_run_pydantic(source):
        return source, []

    # Stage 2 — LibCST PydanticV2Covenant
    new_source = source
    stage2_source, stage2_applied = apply_pydantic_v2_covenant(
        new_source, filename=filename or None
    )
    applied.extend(stage2_applied)
    new_source = stage2_source

    # Stage 3 — ruff --fix
    if new_source != source:
        ruff_target_filename = filename or "file.py"
        stage3_source, stage3_changed = _run_ruff_fix(new_source, ruff_target_filename)
        if stage3_changed:
            applied.append(
                CovenantApplied(
                    rule="ruff_fix.post_libcst_cleanup",
                    detail="ruff --fix removed unused imports and sorted blocks",
                    file=filename or None,
                )
            )
            new_source = stage3_source

    # Stage 4 — bump-pydantic (only when source genuinely imports pydantic)
    if _source_imports_pydantic(new_source):
        stage4_source, stage4_changed = _run_bump_pydantic(new_source, filename or "file.py")
        if stage4_changed:
            applied.append(
                CovenantApplied(
                    rule="bump_pydantic.residual_rewrites",
                    detail="bump-pydantic CLI caught patterns LibCST left behind",
                    file=filename or None,
                )
            )
            new_source = stage4_source

    return new_source, applied


def enforce_python_covenants_on_files(
    code_files: dict[str, str],
) -> tuple[dict[str, str], list[CovenantApplied]]:
    """Bulk form — called by :mod:`belief.graph` covenant_enforce node.

    Iterates :func:`enforce_python_covenants` across every file in
    ``code_files`` (including requirements.txt).  Returns a new dict
    plus the accumulated CovenantApplied list across all files.
    """
    fixed = dict(code_files)
    all_applied: list[CovenantApplied] = []
    for fname, src in list(fixed.items()):
        new_src, applied = enforce_python_covenants(src, filename=fname)
        if applied:
            all_applied.extend(applied)
        if new_src != src:
            fixed[fname] = new_src
    return fixed, all_applied


__all__ = [
    "CovenantApplied",
    "enforce_python_covenants",
    "enforce_python_covenants_on_files",
]
