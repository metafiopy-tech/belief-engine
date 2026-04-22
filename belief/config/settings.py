"""Central configuration — loaded once at startup from .env file.

Source: forge/config.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


@dataclass
class Settings:
    """All configuration in one place. Instantiated from environment."""

    # API keys
    anthropic_api_key: str = ""
    brave_api_key: str = ""
    github_token: str = ""

    # Budget controls
    max_cost_per_build: float = 10.0
    max_iterations: int = 3
    max_steps_per_task: int = 25

    # Local models
    ollama_enabled: bool = False
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Storage
    db_path: str = "~/.belief-engine/builds.db"
    chroma_path: str = "~/.belief-engine/chroma"
    output_dir: str = "./output"
    cache_dir: str = "./cache"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    def __post_init__(self) -> None:
        """Expand user paths consistently."""
        self.db_path = str(Path(self.db_path).expanduser())
        self.chroma_path = str(Path(self.chroma_path).expanduser())

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            brave_api_key=os.environ.get("BRAVE_API_KEY", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            max_cost_per_build=float(os.environ.get("BELIEF_MAX_COST_PER_BUILD", "10.0")),
            max_iterations=int(os.environ.get("BELIEF_MAX_ITERATIONS", "3")),
            max_steps_per_task=int(os.environ.get("BELIEF_MAX_STEPS_PER_TASK", "25")),
            ollama_enabled=os.environ.get("OLLAMA_ENABLED", "false").lower() in ("true", "1", "yes"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            db_path=os.environ.get("BELIEF_DB_PATH", "~/.belief-engine/builds.db"),
            chroma_path=os.environ.get("BELIEF_CHROMA_PATH", "~/.belief-engine/chroma"),
            output_dir=os.environ.get("BELIEF_OUTPUT_DIR", "./output"),
            cache_dir=os.environ.get("BELIEF_CACHE_DIR", "./cache"),
            log_level=os.environ.get("BELIEF_LOG_LEVEL", "INFO"),
            log_json=os.environ.get("BELIEF_LOG_JSON", "false").lower() in ("true", "1", "yes"),
        )

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
settings = Settings.from_env()
