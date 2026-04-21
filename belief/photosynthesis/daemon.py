"""Photosynthesis daemon — APScheduler process entrypoint.

`python -m belief.photosynthesis.daemon` boots the scheduler, registers
six harvest jobs plus one filter pass, and blocks until SIGTERM or
SIGINT. Under systemd the service unit sets Type=simple and
Restart=on-failure; the scheduler's graceful shutdown reaches in-flight
jobs via `sched.shutdown(wait=True)`.

Heavy dependencies (apscheduler, httpx) are imported lazily so the rest
of `belief.photosynthesis` is still importable when the `[photosynthesis]`
extra isn't installed.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any, Awaitable, Callable

from belief.photosynthesis.config import PhotoConfig, load_config
from belief.photosynthesis.sources import (
    arxiv,
    github_releases,
    github_search,
    hackernews,
    pypi,
    stackoverflow,
)
from belief.photosynthesis.state import PhotosynthesisState


logger = logging.getLogger("belief.photosynthesis.daemon")


# Mapping of source module to cadence attribute on Cadences
HARVESTERS: list[tuple[str, Any, str]] = [
    ("github_search", github_search, "github_search_s"),
    ("github_releases", github_releases, "github_releases_s"),
    ("pypi", pypi, "pypi_s"),
    ("stackoverflow", stackoverflow, "stackoverflow_s"),
    ("hackernews", hackernews, "hackernews_s"),
    ("arxiv", arxiv, "arxiv_s"),
]


def _load_scheduler() -> Any:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
        from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED  # type: ignore[import-untyped]
        from apscheduler.executors.pool import ThreadPoolExecutor  # type: ignore[import-untyped]
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore  # type: ignore[import-untyped]

        return {
            "BackgroundScheduler": BackgroundScheduler,
            "EVENT_JOB_ERROR": EVENT_JOB_ERROR,
            "EVENT_JOB_MISSED": EVENT_JOB_MISSED,
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "SQLAlchemyJobStore": SQLAlchemyJobStore,
        }
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "apscheduler is required to run the Photosynthesis daemon. "
            "Install the [photosynthesis] extra: pip install -e '.[photosynthesis]'"
        ) from exc


class PhotosynthesisDaemon:
    """Orchestrates scheduling + dispatch for the six source harvesters."""

    def __init__(self, config: PhotoConfig | None = None) -> None:
        self.config = config or load_config()
        self.state = PhotosynthesisState(str(self.config.signals_db))
        self.scheduler: Any = None
        self._shutting_down = False
        self._loop = asyncio.new_event_loop()

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Build the scheduler and start it. Returns immediately."""
        apsc = _load_scheduler()

        # Lazy: httpx client is inside each harvest dispatch
        from zoneinfo import ZoneInfo

        jobstores = {
            "default": apsc["SQLAlchemyJobStore"](url=f"sqlite:///{self.config.jobs_db}")
        }
        executors = {
            "default": apsc["ThreadPoolExecutor"](self.config.scheduler_max_workers),
        }
        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": self.config.job_misfire_grace_s,
        }

        self.scheduler = apsc["BackgroundScheduler"](
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=ZoneInfo("UTC"),
        )

        # Register harvest jobs
        for name, module, cadence_attr in HARVESTERS:
            seconds = getattr(self.config.cadences, cadence_attr)
            self.scheduler.add_job(
                self._make_harvest_callback(name, module),
                trigger="interval",
                seconds=seconds,
                id=f"harvest:{name}",
                name=f"harvest:{name}",
                replace_existing=True,
            )

        # Register the filter pass
        self.scheduler.add_job(
            self._filter_pass,
            trigger="interval",
            seconds=self.config.cadences.filter_pass_s,
            id="filter_pass",
            name="filter_pass",
            replace_existing=True,
        )

        # Register the synthesis cycle (Session 4)
        self.scheduler.add_job(
            self._synthesis_cycle_entry,
            trigger="interval",
            seconds=self.config.cadences.synthesis_cycle_s,
            id="synthesis_cycle",
            name="synthesis_cycle",
            replace_existing=True,
        )

        # Error listeners
        self.scheduler.add_listener(
            self._on_job_error,
            apsc["EVENT_JOB_ERROR"] | apsc["EVENT_JOB_MISSED"],
        )

        self.scheduler.start()
        logger.info("Photosynthesis started")

    def shutdown(self) -> None:
        """Gracefully stop the scheduler; idempotent."""
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=True)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
        logger.info("Photosynthesis stopped")

    # ------------------------------------------------------------ harvest glue
    def _make_harvest_callback(
        self,
        name: str,
        module: Any,
    ) -> Callable[[], None]:
        """Build a sync-callable that runs the async harvester to completion."""

        def run() -> None:
            try:
                asyncio.run(self._run_harvest(name, module))
            except Exception:
                logger.exception("harvest %s failed", name)

        return run

    async def _run_harvest(self, name: str, module: Any) -> None:
        from belief.core.http import get_async_client

        async with get_async_client(timeout=30.0) as client:
            try:
                new_seeds = await module.harvest(client, self.state, self.config)
            except Exception:
                logger.exception("harvest %s raised", name)
                return
        logger.info("harvest %s: %d new signals", name, len(new_seeds))

    # -------------------------------------------------------- filter pass glue
    def _filter_pass(self) -> None:
        """Run pending signals through the cascading filter."""
        try:
            from belief.photosynthesis.filter.cascade import (
                CascadingRelevanceFilter,
            )
        except Exception:
            logger.exception("cascade filter unavailable")
            return

        rows = self.state.pending_signals(limit=1000)
        if not rows:
            return

        filt = CascadingRelevanceFilter(
            keywords_path=self.config.keywords_file,
        )

        texts = [
            {"signal_id": row["id"], "text": f"{row['title']} {row['summary']}"}
            for row in rows
        ]

        results = filt.score(texts)
        for result in results:
            if result.signal_id is None:
                continue
            status = "kept" if result.kept else "filtered"
            self.state.update_filter_result(
                result.signal_id,
                stage_reached=int(result.stage_reached),
                filter_score=float(result.filter_score),
                status=status,
            )
        logger.info(
            "filter_pass: %d scored, %d kept",
            len(results),
            sum(1 for r in results if r.kept),
        )

    # -------------------------------------------------------- synthesis glue
    def _synthesis_cycle_entry(self) -> None:
        """Sync APScheduler entry point for the async synthesis cycle.

        Guarded by the Session-5 kill switch; for Session 4 the stub
        allows every call through. See
        belief.photosynthesis.synthesis.cycle.run_synthesis_cycle for
        the actual pipeline.
        """
        try:
            from belief.photosynthesis.safety import kill_switch
            from belief.photosynthesis.synthesis.cycle import (
                run_synthesis_cycle_sync,
            )
        except Exception:
            logger.exception("synthesis cycle unavailable")
            return

        gated = kill_switch(tag="synthesis")(run_synthesis_cycle_sync)
        try:
            result = gated(self.state, self.config)
            logger.info("synthesis_cycle: %s", result)
        except Exception:
            logger.exception("synthesis cycle failed")

    # --------------------------------------------------------- event listeners
    def _on_job_error(self, event: Any) -> None:
        logger.error(
            "job %s error: %s",
            getattr(event, "job_id", "?"),
            getattr(event, "exception", None),
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _install_signal_handlers(daemon: PhotosynthesisDaemon) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        logger.info("signal %d received; shutting down", signum)
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    """Start the daemon and block until a signal arrives."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    daemon = PhotosynthesisDaemon()
    daemon.start()
    _install_signal_handlers(daemon)
    # Block forever; APScheduler's BackgroundScheduler runs in a thread.
    try:
        signal.pause()
    except AttributeError:  # pragma: no cover - Windows
        import time

        while not daemon._shutting_down:
            time.sleep(60)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["PhotosynthesisDaemon", "main"]
