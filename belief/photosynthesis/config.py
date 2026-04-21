"""Runtime configuration for the Photosynthesis daemon.

Everything in this module reads from environment variables with
safe defaults, so the same code can run locally (developer laptop)
or on a VPS under systemd without code changes. Paths default to
/var/lib and /var/log when running as the `photo` service user; the
tests override them to temp dirs.

Secrets (ANTHROPIC_API_KEY, GH_TOKEN, STACKEX_KEY, REDDIT_*) live in
/etc/photosynthesis/env with chmod 600 on production. Locally, a
dotenv file or the shell environment is fine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STATE_DIR = Path(os.environ.get("PHOTO_STATE_DIR", "/var/lib/photosynthesis"))
DEFAULT_LOG_DIR = Path(os.environ.get("PHOTO_LOG_DIR", "/var/log/photosynthesis"))
DEFAULT_CONFIG_DIR = Path(os.environ.get("PHOTO_CONFIG_DIR", "/etc/photosynthesis"))


@dataclass(frozen=True)
class Cadences:
    """Source-polling intervals in seconds.

    Pulled verbatim from the design doc's rate-limit table. Never poll
    a source faster than it wants to be polled; rate limit bans cascade
    into long-term filter gaps.
    """

    github_search_s: int = 6 * 3600        # 6h
    github_releases_s: int = 15 * 60       # 15 min
    pypi_s: int = 10 * 60                  # 10 min
    stackoverflow_s: int = 30 * 60         # 30 min
    hackernews_s: int = 15 * 60            # 15 min
    arxiv_s: int = 6 * 3600                # 6h
    filter_pass_s: int = 30 * 60           # 30 min — cascade-run interval
    synthesis_cycle_s: int = 2 * 3600      # 2h — Session 4 goal synthesis


@dataclass(frozen=True)
class FilterThresholds:
    """Cosine / score thresholds for the cascade.

    stage2_coarse is the "keep through" gate; signals below it fail the
    filter outright. stage2_high marks signals high-enough to bypass the
    more expensive stage-3 embedding step (they're obvious keeps).
    stage3_threshold is the MiniLM cosine cutoff. All tunable — start
    with the spec defaults and let real-world data adjust them.
    """

    stage2_coarse: float = 0.12
    stage2_high: float = 0.30
    stage3_threshold: float = 0.35


@dataclass(frozen=True)
class FilterBudget:
    """How many survivors to hand to the next pipeline stage."""

    top_k_for_llm: int = 20


@dataclass(frozen=True)
class PhotoConfig:
    """Everything the daemon needs to boot. Immutable by design."""

    # Filesystem
    state_dir: Path = field(default_factory=lambda: DEFAULT_STATE_DIR)
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    config_dir: Path = field(default_factory=lambda: DEFAULT_CONFIG_DIR)

    # Scheduler
    cadences: Cadences = field(default_factory=Cadences)
    job_misfire_grace_s: int = 300
    scheduler_max_workers: int = 4

    # Filter
    thresholds: FilterThresholds = field(default_factory=FilterThresholds)
    budget: FilterBudget = field(default_factory=FilterBudget)

    # Tracked deps — used by github_releases and pypi targeted fetch
    tracked_deps: tuple[str, ...] = (
        "langgraph",
        "anthropic",
        "pydantic",
        "fastapi",
        "click",
        "chromadb",
        "httpx",
        "typer",
        "openai",
        "langchain",
    )

    # Stack Overflow tags to follow
    stackoverflow_tags: tuple[str, ...] = (
        "python",
        "fastapi",
        "click",
        "mcp",
        "anthropic-claude",
        "langchain",
    )

    # HackerNews: only Show HN
    hn_tags: str = "show_hn"

    # arXiv categories
    arxiv_categories: tuple[str, ...] = ("cs.AI", "cs.CL", "cs.SE")

    # Secret/env lookups
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    github_token_env: str = "GH_TOKEN"
    stackex_key_env: str = "STACKEX_KEY"

    # Derived paths
    @property
    def signals_db(self) -> Path:
        return self.state_dir / "signals.sqlite"

    @property
    def jobs_db(self) -> Path:
        return self.state_dir / "jobs.sqlite"

    @property
    def keywords_file(self) -> Path:
        return self.config_dir / "domain_keywords.yaml"

    @property
    def pending_sessions_dir(self) -> Path:
        return self.state_dir / "pending_sessions"

    @property
    def archive_chroma_dir(self) -> Path:
        return self.state_dir / "archive_chroma"


def load_config() -> PhotoConfig:
    """Return a PhotoConfig from environment, with directory creation.

    Lightweight — safe to call multiple times. Anything that needs a
    full DB connection belongs in state.py, not here.
    """
    cfg = PhotoConfig()
    return cfg


__all__ = [
    "Cadences",
    "FilterBudget",
    "FilterThresholds",
    "PhotoConfig",
    "load_config",
]
