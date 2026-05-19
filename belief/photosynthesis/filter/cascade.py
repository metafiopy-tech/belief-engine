"""Four-stage cascading relevance filter.

Given a list of raw signal texts, drop the irrelevant ones as cheaply
as possible. Stages get progressively more expensive; only the top ~20
survivors reach LLM scoring in the next pipeline step.

    stage 0 : bloom blocklist  (~15 KB RAM, O(1) per item)
    stage 1 : compiled keyword regex (~200 terms)
    stage 2 : TF-IDF cosine vs past-goal corpus
    stage 3 : MiniLM sentence embedding vs ChromaDB domain centroids

Heavy deps (pybloom-live, scikit-learn, sentence-transformers) are
lazy-loaded on first use. A caller that only wants the cheap stages
(e.g. unit tests) can instantiate the filter without them.

Design invariants:
- Stage ordering is fixed. No stage is allowed to *promote* a signal
  that failed an earlier stage. Stages are gates, not scorers.
- Each signal's `stage_reached` records the last stage it survived —
  0 means "blocked at stage 0", 3 means "passed stage 3".
- `filter_score` is the score assigned by whichever stage made the
  final keep/drop decision (higher = more relevant).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


logger = logging.getLogger("belief.photosynthesis.filter")


# Session 0 (v3.2): explicit offline gate. When BELIEF_OFFLINE=1 is set,
# the stage-3 embedding path raises a *clear* RuntimeError instead of
# silently falling through to Hugging Face's hub, which was the original
# failure mode in the non-hermetic tests (unreachable HF during CI runs
# produced opaque HTTPError chains). Stages 0–2 remain fully usable.
def _offline_mode() -> bool:
    """True when BELIEF_OFFLINE is set to a truthy value."""
    v = os.environ.get("BELIEF_OFFLINE", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


class OfflineModeError(RuntimeError):
    """Raised when stage 3 is invoked under BELIEF_OFFLINE=1.

    Stages 0–2 (blocklist, keyword, TF-IDF) are all-local and do not
    raise.  Stage 3 requires a SentenceTransformer checkpoint which
    would otherwise be fetched from Hugging Face — that is exactly
    what BELIEF_OFFLINE exists to forbid.
    """


class Stage(IntEnum):
    """Cascade stage labels (matches stage_reached in the DB)."""

    BLOCKED = 0
    KEYWORD = 1
    TFIDF = 2
    EMBED = 3


@dataclass
class FilterResult:
    """Per-signal filter outcome."""

    signal_id: Optional[int] = None
    text: str = ""
    stage_reached: int = 0
    filter_score: float = 0.0
    kept: bool = False
    reason: str = ""

    # Optional diagnostics (populated by higher stages)
    keyword_hits: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------


def _load_bloom() -> Any:
    try:
        from pybloom_live import BloomFilter  # type: ignore[import-untyped]

        return BloomFilter
    except ImportError:
        return None


def _load_sklearn() -> Any:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

        return TfidfVectorizer, cosine_similarity
    except ImportError:
        return None


def _load_sentence_transformers() -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        return SentenceTransformer
    except ImportError:
        return None


def _load_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-untyped]

        return np
    except ImportError:
        return None


def _load_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# CascadingRelevanceFilter
# ---------------------------------------------------------------------------


class CascadingRelevanceFilter:
    """Four-stage filter matching the Photosynthesis design doc."""

    def __init__(
        self,
        keywords_path: Optional[Path] = None,
        *,
        blocklist: Optional[Iterable[str]] = None,
        corpus: Optional[Sequence[str]] = None,
        centroids: Optional[Any] = None,
        stage2_coarse: float = 0.12,
        stage2_high: float = 0.30,
        stage3_threshold: float = 0.35,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        self._keywords_path = keywords_path
        self._blocklist_items = list(blocklist or [])
        self._corpus = list(corpus or [])
        self._centroids = centroids

        self.stage2_coarse = stage2_coarse
        self.stage2_high = stage2_high
        self.stage3_threshold = stage3_threshold
        self._embed_model_name = embed_model_name
        self._batch_size = batch_size

        # Lazy state
        self._keyword_regex: Optional[re.Pattern[str]] = None
        self._keyword_terms: list[str] = []
        self._bloom: Any = None
        self._tfidf: Any = None
        self._tfidf_matrix: Any = None
        self._cosine: Any = None
        self._embed_model: Any = None
        self._np: Any = None

        self._lock = threading.Lock()

    # ---------------------------- stage 0: bloom blocklist
    def _ensure_bloom(self) -> None:
        if self._bloom is not None:
            return
        BloomFilter = _load_bloom()
        if BloomFilter is None:
            # Without pybloom-live we fall back to an ordinary set —
            # fine at 10k entries; the spec size of 15KB is a nice-to-have.
            self._bloom = set(self._blocklist_items)
            return
        # capacity 10k at 0.1% fpr ≈ 15 KB
        self._bloom = BloomFilter(capacity=10_000, error_rate=0.001)
        for item in self._blocklist_items:
            self._bloom.add(item)

    def _blocked(self, text: str) -> bool:
        """Stage 0: cheap identity blocklist (domains, authors)."""
        if not self._blocklist_items:
            return False
        self._ensure_bloom()
        lowered = text.lower()
        for item in self._blocklist_items:
            if item and item.lower() in lowered:
                return True
        return False

    # ---------------------------- stage 1: keyword regex
    def _ensure_keyword_regex(self) -> None:
        if self._keyword_regex is not None:
            return
        terms = list(self._keyword_terms)
        if not terms and self._keywords_path:
            terms = self._load_keyword_file(self._keywords_path)
        self._keyword_terms = terms
        if not terms:
            self._keyword_regex = re.compile(r"(?!x)x")  # never matches
            return
        # Word-bounded, case-insensitive, OR of escaped terms.
        pattern = r"\b(?:" + "|".join(re.escape(t) for t in sorted(set(terms))) + r")\b"
        self._keyword_regex = re.compile(pattern, re.IGNORECASE)

    def _load_keyword_file(self, path: Path) -> list[str]:
        yaml = _load_yaml()
        if yaml is None:
            logger.warning("pyyaml not installed; stage-1 keyword filter is empty.")
            return []
        try:
            data = yaml.safe_load(Path(path).read_text())
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Could not read keyword file %s: %s", path, exc)
            return []
        return [str(k) for k in (data or {}).get("keywords", [])]

    def _keyword_hits(self, text: str) -> list[str]:
        self._ensure_keyword_regex()
        if self._keyword_regex is None:
            return []
        return self._keyword_regex.findall(text or "")

    # ---------------------------- stage 2: TF-IDF cosine
    def _ensure_tfidf(self) -> None:
        if self._tfidf is not None:
            return
        sklearn = _load_sklearn()
        if sklearn is None or not self._corpus:
            return
        TfidfVectorizer, cosine_similarity = sklearn
        self._cosine = cosine_similarity
        self._tfidf = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        self._tfidf_matrix = self._tfidf.fit_transform(self._corpus)

    def _tfidf_score(self, text: str) -> float:
        """Returns the max cosine to any corpus item, or -1.0 if TF-IDF is unavailable."""
        self._ensure_tfidf()
        if self._tfidf is None or self._tfidf_matrix is None:
            return -1.0
        try:
            vec = self._tfidf.transform([text or ""])
            sims = self._cosine(vec, self._tfidf_matrix)
            return float(sims.max()) if sims.size else 0.0
        except Exception:
            return 0.0

    # ---------------------------- stage 3: MiniLM embedding
    def _ensure_embed_model(self) -> None:
        if self._embed_model is not None:
            return
        # Session 0: explicit offline gate — refuse before we try to
        # construct SentenceTransformer, which would otherwise attempt
        # to pull the model from Hugging Face on first call.
        if _offline_mode():
            raise OfflineModeError(
                f"embedding model {self._embed_model_name!r} requested in "
                "offline mode (BELIEF_OFFLINE=1); stage 3 is disabled. "
                "Unset BELIEF_OFFLINE or avoid stage-3 invocation."
            )
        SentenceTransformer = _load_sentence_transformers()
        if SentenceTransformer is None:
            return
        # Phase 1 (2026-05-18): fail-closed when the model isn't
        # already cached locally. Tests in tests/photosynthesis/
        # test_filter.py exercise the no-corpus path with
        # sentence-transformers installed but the model not
        # pre-cached; they expect the cascade to skip stage 3
        # cleanly rather than attempt a Hugging Face download.
        # Scoping HF_HUB_OFFLINE to this one constructor call makes
        # a missing cache raise a catchable OSError instead of
        # either silently downloading or hanging on a network
        # timeout. Set BELIEF_EMBED_ALLOW_DOWNLOAD=1 to opt back in
        # to in-band downloads (useful when populating a fresh
        # machine's cache).
        allow_download = os.environ.get("BELIEF_EMBED_ALLOW_DOWNLOAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._embed_model = SentenceTransformer(self._embed_model_name, device="cpu")
        except (OSError, ValueError) as exc:
            logger.warning(
                "stage-3 embedding model %r not available locally (%s); "
                "cascade will skip stage 3. Set "
                "BELIEF_EMBED_ALLOW_DOWNLOAD=1 to permit Hugging Face "
                "downloads, or pre-cache the model.",
                self._embed_model_name,
                exc,
            )
            return
        finally:
            if not allow_download:
                if prev_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_offline
        try:
            self._embed_model.max_seq_length = 256
        except AttributeError:
            pass
        self._np = _load_numpy()

    def _embed_score(self, text: str) -> float:
        """Cosine vs loaded centroids; -1.0 if unavailable."""
        self._ensure_embed_model()
        if self._embed_model is None or self._centroids is None or self._np is None:
            return -1.0
        vec = self._embed_model.encode([text or ""], batch_size=self._batch_size)
        v = vec[0]
        # Normalize
        norm = float(self._np.linalg.norm(v)) or 1.0
        v = v / norm
        sims = self._centroids @ v  # centroids expected pre-normalized
        return float(self._np.max(sims))

    # ---------------------------- main entry point
    def score(
        self,
        signals: Sequence[dict | str],
    ) -> list[FilterResult]:
        """Score a batch of signals.

        Each signal may be either a plain string (used as the scoring
        text) or a dict with keys 'text' and optionally 'signal_id'.
        """
        results: list[FilterResult] = []

        with self._lock:
            self._ensure_keyword_regex()

            # Batch prep for stage 3 (embed only survivors to save cost)
            stage3_candidates: list[int] = []

            for idx, raw in enumerate(signals):
                text, sid = _coerce_signal(raw)
                r = FilterResult(signal_id=sid, text=text)

                # Stage 0
                if self._blocked(text):
                    r.reason = "blocklist"
                    r.stage_reached = Stage.BLOCKED
                    results.append(r)
                    continue

                # Stage 1
                hits = self._keyword_hits(text)
                r.keyword_hits = hits
                if not hits:
                    r.reason = "no keyword"
                    r.stage_reached = Stage.BLOCKED
                    results.append(r)
                    continue
                r.stage_reached = Stage.KEYWORD
                r.filter_score = min(1.0, len(hits) / 5.0)

                # Stage 2
                t2 = self._tfidf_score(text)
                if t2 >= 0:
                    if t2 < self.stage2_coarse:
                        r.reason = "low tfidf"
                        results.append(r)
                        continue
                    r.stage_reached = Stage.TFIDF
                    r.filter_score = t2
                    # High-confidence bypass of stage 3
                    if t2 >= self.stage2_high:
                        r.kept = True
                        results.append(r)
                        continue
                stage3_candidates.append(len(results))
                results.append(r)

            # Stage 3 (batch)
            if stage3_candidates:
                self._ensure_embed_model()
                for i in stage3_candidates:
                    r = results[i]
                    t3 = self._embed_score(r.text)
                    if t3 < 0:
                        # Embed model unavailable; keep what stage 2 decided.
                        continue
                    r.stage_reached = Stage.EMBED
                    r.filter_score = t3
                    r.kept = t3 >= self.stage3_threshold
                    if not r.kept:
                        r.reason = "low embed"

        return results

    def top_k_for_llm(
        self,
        scored: Sequence[FilterResult],
        k: int = 20,
    ) -> list[FilterResult]:
        """Return the top-k kept results by filter_score, descending."""
        kept = [r for r in scored if r.kept]
        kept.sort(key=lambda r: r.filter_score, reverse=True)
        return kept[:k]


def _coerce_signal(raw: dict | str) -> tuple[str, Optional[int]]:
    if isinstance(raw, str):
        return raw, None
    text = str(raw.get("text") or raw.get("title") or "")
    sid = raw.get("signal_id")
    try:
        sid_int = int(sid) if sid is not None else None
    except (TypeError, ValueError):
        sid_int = None
    return text, sid_int


# ---------------------------------------------------------------------------
# Throughput diagnostic (used by tests)
# ---------------------------------------------------------------------------


def throughput(filter_: CascadingRelevanceFilter, signals: Sequence[str]) -> float:
    """Returns signals-per-second when running the full pipeline."""
    start = time.monotonic()
    filter_.score(signals)
    elapsed = max(1e-9, time.monotonic() - start)
    return len(signals) / elapsed


__all__ = [
    "CascadingRelevanceFilter",
    "FilterResult",
    "Stage",
    "throughput",
]
