"""
Prompt Store — manages optimized prompts across versions.

Stores optimized prompt instructions extracted by DSPy compilation
so they can be reloaded without re-running optimization.  Each save
is tagged with a version ID (from the evolutionary archive) and a
timestamp.

Storage: JSON files in ~/.belief-engine/prompts/
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("belief.optimization.prompt_store")


class PromptStore:
    """Manages optimized prompts across versions.

    Usage:
        store = PromptStore()
        store.save({"planner.plan": "Optimized instructions..."}, "v-abc123")
        prompts = store.load_latest()
    """

    def __init__(self, store_dir: str = "~/.belief-engine/prompts") -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, prompts: dict[str, str], version_id: str) -> Path:
        """Save optimized prompts for a version.

        Creates a JSON file named {version_id}.json with the prompts
        and metadata.

        Returns the path to the saved file.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        data = {
            "version_id": version_id,
            "timestamp": timestamp,
            "prompts": prompts,
        }

        filename = f"{version_id}.json"
        path = self.store_dir / filename
        path.write_text(json.dumps(data, indent=2))

        logger.info(
            f"Saved {len(prompts)} optimized prompts for version {version_id}"
        )
        return path

    def load_latest(self) -> Optional[dict[str, str]]:
        """Load the most recent optimized prompts.

        Returns None if no saved prompts exist.
        """
        files = sorted(self.store_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if not files:
            return None

        latest = files[-1]
        try:
            data = json.loads(latest.read_text())
            return data.get("prompts", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load latest prompts: {e}")
            return None

    def load_for_version(self, version_id: str) -> Optional[dict[str, str]]:
        """Load prompts for a specific version.

        Returns None if no prompts exist for this version.
        """
        path = self.store_dir / f"{version_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return data.get("prompts", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load prompts for {version_id}: {e}")
            return None

    def list_versions(self) -> list[dict]:
        """List all saved prompt versions with metadata.

        Returns list of dicts with keys: version_id, timestamp, prompt_count.
        """
        versions: list[dict] = []
        for path in sorted(self.store_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                versions.append({
                    "version_id": data.get("version_id", path.stem),
                    "timestamp": data.get("timestamp", ""),
                    "prompt_count": len(data.get("prompts", {})),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return versions

    def delete_version(self, version_id: str) -> bool:
        """Delete prompts for a specific version.

        Returns True if deleted, False if not found.
        """
        path = self.store_dir / f"{version_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
