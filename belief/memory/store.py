"""Build memory — remembers what was built before.

Session state: flat JSON for current run.
Build store: SQLite with TF-IDF similarity search across past builds.

Source: memory.py, store.py, indexer.py
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("belief.memory")


# ── Session State ─────────────────────────────────────────────────────────────

class SessionState:
    """Flat JSON session state for the current run.

    Source: memory.py
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._file = session_dir / "session.json"
        self._state: dict = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except Exception:
                pass
        return {"created": datetime.now().isoformat(), "facts": {}, "remainders": []}

    def save(self) -> None:
        self._file.write_text(json.dumps(self._state, indent=2, default=str))

    def set_fact(self, key: str, value) -> None:
        self._state["facts"][key] = value
        self.save()

    def get_fact(self, key: str, default=None):
        return self._state.get("facts", {}).get(key, default)

    def add_remainder(self, remainder: str) -> None:
        self._state.setdefault("remainders", []).append({
            "t": datetime.now().isoformat(),
            "r": remainder,
        })
        # Keep last 100
        if len(self._state["remainders"]) > 100:
            self._state["remainders"] = self._state["remainders"][-100:]
        self.save()

    def get_remainders(self, n: int = 10) -> list[str]:
        return [r["r"] for r in self._state.get("remainders", [])[-n:]]


# ── Build Record ──────────────────────────────────────────────────────────────

class BuildRecord(BaseModel):
    """A completed build stored in the build memory."""
    run_id: str
    goal: str
    goal_refined: str = ""
    file_summaries: dict[str, str] = Field(default_factory=dict)
    quality_scores: dict[str, float] = Field(default_factory=dict)
    output_path: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    verdict: str = ""
    tags: list[str] = Field(default_factory=list)
    completed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Build Store with Similarity Search ────────────────────────────────────────

class BuildStore:
    """SQLite-backed build history with TF-IDF similarity search.

    Source: store.py + indexer.py

    When you ask "build me an MCP server", this searches past builds
    and finds the closest prior build to use as a reference.
    No embeddings needed — pure TF-IDF cosine similarity over goals.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._idf_cache: dict[str, float] = {}

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS builds (
                run_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                goal_refined TEXT DEFAULT '',
                file_summaries TEXT DEFAULT '{}',
                quality_scores TEXT DEFAULT '{}',
                output_path TEXT DEFAULT '',
                cost_usd REAL DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                verdict TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                completed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_builds_goal ON builds(goal);
        """)
        self._conn.commit()

    def store(self, record: BuildRecord) -> None:
        """Store a completed build."""
        self._conn.execute(
            """INSERT OR REPLACE INTO builds
               (run_id, goal, goal_refined, file_summaries, quality_scores,
                output_path, cost_usd, duration_seconds, verdict, tags, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.run_id, record.goal, record.goal_refined,
             json.dumps(record.file_summaries), json.dumps(record.quality_scores),
             record.output_path, record.cost_usd, record.duration_seconds,
             record.verdict, json.dumps(record.tags), record.completed_at),
        )
        self._conn.commit()
        self._idf_cache.clear()  # Invalidate cache
        logger.info(f"Build stored: {record.run_id} ({record.goal[:50]})")

    def find_similar(self, goal: str, n: int = 3) -> list[BuildRecord]:
        """Find the N most similar past builds using TF-IDF cosine similarity."""
        rows = self._conn.execute("SELECT * FROM builds ORDER BY completed_at DESC").fetchall()
        if not rows:
            return []

        # Tokenize the query
        query_tokens = _tokenize(goal)
        if not query_tokens:
            return []

        # Build IDF from all stored goals
        all_docs = [_tokenize(row["goal"]) for row in rows]
        idf = _compute_idf(all_docs)

        # TF-IDF vector for query
        query_vec = _tfidf_vector(query_tokens, idf)

        # Score each stored build
        scored = []
        for i, row in enumerate(rows):
            doc_vec = _tfidf_vector(all_docs[i], idf)
            sim = _cosine_similarity(query_vec, doc_vec)
            if sim > 0.1:  # Minimum relevance threshold
                record = BuildRecord(
                    run_id=row["run_id"], goal=row["goal"],
                    goal_refined=row["goal_refined"],
                    file_summaries=json.loads(row["file_summaries"]),
                    quality_scores=json.loads(row["quality_scores"]),
                    output_path=row["output_path"],
                    cost_usd=row["cost_usd"],
                    duration_seconds=row["duration_seconds"],
                    verdict=row["verdict"],
                    tags=json.loads(row["tags"]),
                    completed_at=row["completed_at"],
                )
                scored.append((sim, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:n]]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM builds").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()


# ── TF-IDF Helpers (stdlib only, no sklearn needed) ───────────────────────────

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "in", "it", "to", "of", "and", "or", "for",
    "with", "on", "at", "by", "this", "that", "from", "was", "be", "are",
    "as", "has", "have", "been", "its", "but", "not", "we", "our", "my",
    "build", "create", "make", "write", "python", "script", "using",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize and clean text for TF-IDF."""
    return [
        w for w in text.lower().split()
        if len(w) > 2 and w not in _STOP_WORDS and w.isalpha()
    ]


def _compute_idf(docs: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency."""
    n = len(docs)
    if n == 0:
        return {}
    df: Counter = Counter()
    for doc in docs:
        for word in set(doc):
            df[word] += 1
    return {word: math.log(n / count) for word, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a document."""
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: (count / total) * idf.get(word, 0) for word, count in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
