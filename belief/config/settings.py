"""Central configuration — loaded once at startup from .env file.

Source: forge/config.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


class Settings:
    """All configuration in one place. No scattered os.getenv() calls."""

    # API keys
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    brave_api_key: str = os.environ.get("BRAVE_API_KEY", "")
    github_token: str = os.environ.get("GITHUB_TOKEN", "")

    # Budget controls
    max_cost_per_build: float = float(os.environ.get("BELIEF_MAX_COST_PER_BUILD", "10.0"))
    max_iterations: int = int(os.environ.get("BELIEF_MAX_ITERATIONS", "3"))
    max_steps_per_task: int = int(os.environ.get("BELIEF_MAX_STEPS_PER_TASK", "25"))

    # Local models
    ollama_enabled: bool = os.environ.get("OLLAMA_ENABLED", "false").lower() in ("true", "1", "yes")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # Telegram
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Storage
    db_path: str = os.environ.get("BELIEF_DB_PATH", "~/.belief-engine/builds.db")
    chroma_path: str = os.environ.get("BELIEF_CHROMA_PATH", "~/.belief-engine/chroma")
    output_dir: str = os.environ.get("BELIEF_OUTPUT_DIR", "./output")
    cache_dir: str = os.environ.get("BELIEF_CACHE_DIR", "./cache")

    # Logging
    log_level: str = os.environ.get("BELIEF_LOG_LEVEL", "INFO")
    log_json: bool = os.environ.get("BELIEF_LOG_JSON", "false").lower() in ("true", "1", "yes")

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_full_path(self) -> Path:
        p = Path(self.db_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# Singleton
settings = Settings()
