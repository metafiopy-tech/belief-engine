"""Self-improvement pipeline — SEED, Mentor, and SelfPatch.

SEED: Every N builds, review accumulated remainders and propose one improvement.
Mentor: Evaluate proposals for safety. Approve or reject.
SelfPatch: Apply approved patches, snapshot before changes, rollback on failure.

Source: seed.py, mentor.py, self_patch.py, pipeline.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("belief.evolution")


# ── Proposal Model ────────────────────────────────────────────────────────────

class ImprovementProposal(BaseModel):
    """A structured self-improvement proposal from SEED."""
    title: str
    what: str  # What the change does
    why: str   # Why it's needed (linked to remainders)
    target_file: str  # Which file to modify
    code: str = ""    # The actual code change
    confidence: str = "MEDIUM"  # HIGH / MEDIUM / LOW
    proposed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending / approved / rejected / applied / rolled_back
    remainder_sources: list[str] = Field(default_factory=list)


# ── SEED — Self-Evolving Enhancement Driver ───────────────────────────────────

class SEED:
    """Reviews accumulated remainders and proposes targeted improvements.

    Source: seed.py

    Constraints:
    - ONE proposal per cycle
    - Must contain actual working Python code
    - Must target specific file and function
    - Prefer reliability fixes over new features
    """

    TRIGGER_EVERY = 10  # builds between proposals

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._build_count = 0
        self._proposals_file = project_root / ".belief-engine" / "proposals.json"
        self._proposals_file.parent.mkdir(parents=True, exist_ok=True)

    def tick(self) -> bool:
        """Called after every build. Returns True if it's time to propose."""
        self._build_count += 1
        return self._build_count % self.TRIGGER_EVERY == 0

    async def propose(self, remainders: list[str], llm=None) -> Optional[ImprovementProposal]:
        """Generate one improvement proposal from accumulated remainders + soil nutrients.

        Phase 3: SEED now reads from the ChromaDB soil to inform its proposals.
        Covenants become mandatory constraints. Antipatterns become focus areas.
        """
        if not llm:
            return None

        # ── Phase 3: Read from soil ──
        soil_context = ""
        try:
            from belief.memory.soil import Soil
            from belief.memory.nutrients import NutrientType
            soil = Soil(Path("~/.belief-engine/soil").expanduser())

            # Get all covenants — these are immutable rules to enforce
            covenants = soil.retrieve("", nutrient_type=NutrientType.COVENANT, min_retrievability=0.0)
            if covenants:
                covenant_text = "\n".join(f"  COVENANT: {c.content}" for c in covenants)
                soil_context += f"\nACTIVE COVENANTS (must be enforced):\n{covenant_text}\n"

            # Get recent antipatterns — these are recurring failures to fix
            antipatterns = soil.retrieve("build failure error", nutrient_type=NutrientType.ANTIPATTERN, n=5)
            if antipatterns:
                anti_text = "\n".join(f"  ANTIPATTERN: {a.content}" for a in antipatterns)
                soil_context += f"\nRECURRING FAILURES (prioritize fixing these):\n{anti_text}\n"

            # Soil stats for context
            soil_context += f"\nSoil stats: {soil.count()} nutrients, {soil.count_by_type()}\n"

        except Exception as e:
            logger.debug(f"SEED: soil read failed ({e}), using remainders only")

        if not remainders and not soil_context:
            return None

        remainder_text = "\n".join(f"- {r}" for r in remainders[-20:]) if remainders else "(none)"

        try:
            raw = await llm.generate_text(
                role="latios",
                system=(
                    "You are SEED, the self-improvement engine. "
                    "Analyze accumulated gaps AND soil nutrients from past builds "
                    "and propose ONE targeted fix. "
                    "COVENANTS are immutable rules — your fix must enforce or strengthen them. "
                    "ANTIPATTERNS are recurring failures — prioritize fixes that prevent them. "
                    "The fix must be a specific code change to a specific file. "
                    "Prefer reliability fixes over new features. "
                    "Respond ONLY with valid JSON."
                ),
                prompt=(
                    f"Accumulated gaps from recent builds:\n{remainder_text}\n\n"
                    f"{soil_context}\n"
                    f"Project files: {[f.name for f in self.project_root.glob('belief/**/*.py')]}\n\n"
                    '{"title": "...", "what": "...", "why": "...", '
                    '"target_file": "belief/agents/xxx.py", "confidence": "HIGH|MEDIUM|LOW", '
                    '"code": "the actual python code to add or replace"}'
                ),
                temperature=0.3,
                max_tokens=2000,
            )
            # Parse JSON
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                proposal = ImprovementProposal(
                    **data,
                    remainder_sources=remainders[-5:] if remainders else [],
                )
                self._save_proposal(proposal)
                return proposal
        except Exception as e:
            logger.warning(f"SEED proposal generation failed: {e}")

        return None

    def _save_proposal(self, proposal: ImprovementProposal) -> None:
        proposals = self._load_proposals()
        proposals.append(proposal.model_dump())
        self._proposals_file.write_text(json.dumps(proposals[-50:], indent=2))

    def _load_proposals(self) -> list:
        if self._proposals_file.exists():
            try:
                return json.loads(self._proposals_file.read_text())
            except Exception:
                pass
        return []

    def get_pending(self) -> list[ImprovementProposal]:
        proposals = self._load_proposals()
        return [ImprovementProposal(**p) for p in proposals if p.get("status") == "pending"]


# ── Mentor — The Immune System ────────────────────────────────────────────────

class Mentor:
    """Evaluates proposals for safety before application.

    Source: mentor.py

    Checks:
    - Is this actually a code change (not emotional/confused output)?
    - Does it have valid Python syntax?
    - What's the blast radius?
    - Has something similar been tried and failed?
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._history_file = project_root / ".belief-engine" / "patch_history.json"

    async def evaluate(self, proposal: ImprovementProposal,
                       llm=None) -> tuple[bool, str]:
        """Evaluate a proposal. Returns (approved: bool, reason: str)."""

        # Check 1: Does it have code?
        if not proposal.code or len(proposal.code.strip()) < 10:
            return False, "Proposal has no substantive code."

        # Check 2: Valid Python syntax?
        if proposal.target_file.endswith(".py"):
            try:
                ast.parse(proposal.code)
            except SyntaxError as e:
                return False, f"Syntax error in proposed code: {e}"

        # Check 3: Target file exists?
        target = self.project_root / proposal.target_file
        if not target.exists():
            return False, f"Target file does not exist: {proposal.target_file}"

        # Check 4: Has this been tried before and failed?
        history = self._load_history()
        for entry in history:
            if entry.get("status") == "rolled_back" and entry.get("title") == proposal.title:
                return False, f"Similar patch was previously rolled back: {entry.get('reason', 'unknown')}"

        # Check 5: LLM safety review (if available)
        if llm:
            try:
                verdict = await llm.generate_text(
                    role="latios",
                    system=(
                        "You are Mentor, the immune system. Evaluate this self-modification proposal. "
                        "Is it safe? Is it useful? Will it break anything? "
                        "Respond with APPROVE or REJECT followed by one sentence."
                    ),
                    prompt=(
                        f"Title: {proposal.title}\n"
                        f"What: {proposal.what}\n"
                        f"Target: {proposal.target_file}\n"
                        f"Code:\n{proposal.code[:1000]}\n"
                    ),
                    temperature=0.1,
                    max_tokens=100,
                )
                if "REJECT" in verdict.upper():
                    return False, verdict
            except Exception:
                pass  # If LLM fails, continue with deterministic checks only

        # Check 6: Confidence threshold
        if proposal.confidence == "LOW":
            return False, "LOW confidence proposals require manual approval."

        return True, "All checks passed."

    def _load_history(self) -> list:
        if self._history_file.exists():
            try:
                return json.loads(self._history_file.read_text())
            except Exception:
                pass
        return []

    def _save_history(self, entry: dict) -> None:
        history = self._load_history()
        history.append(entry)
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history_file.write_text(json.dumps(history[-50:], indent=2))


# ── SelfPatch — Apply Changes ────────────────────────────────────────────────

class SelfPatch:
    """Apply approved patches with snapshot and rollback.

    Source: self_patch.py

    Every patch:
    1. Snapshot current file
    2. Apply the change
    3. Syntax-check the result
    4. If broken → rollback → notify
    5. If good → log → notify
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._snapshots_dir = project_root / ".belief-engine" / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, filepath: Path) -> Path:
        """Snapshot a file before patching."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = self._snapshots_dir / f"{filepath.name}.{ts}.bak"
        shutil.copy2(filepath, snap_path)
        logger.info(f"Snapshot: {filepath.name} → {snap_path.name}")
        return snap_path

    def apply(self, proposal: ImprovementProposal) -> tuple[bool, str]:
        """Apply a patch. Returns (success, message)."""
        target = self.project_root / proposal.target_file
        if not target.exists():
            return False, f"Target not found: {proposal.target_file}"

        # Snapshot
        snap = self.snapshot(target)

        try:
            # Determine if this is a full file replacement or an append
            code = proposal.code.strip()
            is_full_file = (
                code.startswith('"""') or code.startswith("import ")
                or code.startswith("from ") or code.startswith("#!")
            ) and len(code) > 500

            if is_full_file:
                target.write_text(code)
                action = "replaced"
            else:
                with open(target, "a") as f:
                    f.write(f"\n\n# ── Self-patch {datetime.now().strftime('%Y-%m-%d %H:%M')} ──\n")
                    f.write(f"# {proposal.title}\n")
                    f.write(code)
                action = "appended"

            # Verify syntax
            if target.suffix == ".py":
                try:
                    ast.parse(target.read_text())
                except SyntaxError as e:
                    # Rollback
                    shutil.copy2(snap, target)
                    return False, f"Patch broke syntax — rolled back. Error: {e}"

            logger.info(f"Patch applied: {proposal.title} ({action} {target.name})")
            return True, f"Patch applied ({action}): {proposal.title}"

        except Exception as e:
            # Rollback on any error
            shutil.copy2(snap, target)
            return False, f"Patch failed — rolled back. Error: {e}"

    def rollback(self, filepath: Path) -> tuple[bool, str]:
        """Roll back to the most recent snapshot."""
        snaps = sorted(self._snapshots_dir.glob(f"{filepath.name}.*"), reverse=True)
        if not snaps:
            return False, "No snapshots found."
        shutil.copy2(snaps[0], filepath)
        return True, f"Rolled back to {snaps[0].name}"

    def list_snapshots(self) -> list[str]:
        return sorted(s.name for s in self._snapshots_dir.iterdir())
