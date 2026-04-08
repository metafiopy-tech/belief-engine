"""Telegram notification support for Belief Engine.

Sends notifications for:
- Build completion (pass/fail, cost, time)
- SEED proposals
- Health issues

Source: spawn_gate.py _tg_send pattern
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from belief.config.settings import settings

logger = logging.getLogger("belief.tools.notify")


def send_telegram(text: str, parse_mode: str = "Markdown") -> bool:
    """Send a message to Telegram. Returns True on success."""
    token = settings.telegram_token
    chat_id = settings.telegram_chat_id

    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping notification")
        return False

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": text[:4000],  # Telegram limit
            "parse_mode": parse_mode,
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        logger.debug(f"Telegram send failed: {e}")
        return False


def notify_build_complete(
    run_id: str, goal: str, verdict: str, cost: float,
    elapsed: float, file_count: int,
) -> None:
    """Notify about a completed build."""
    emoji = "✅" if verdict == "pass" else "⚠️"
    msg = (
        f"{emoji} *Build Complete*\n\n"
        f"Goal: {goal[:100]}\n"
        f"Verdict: `{verdict}`\n"
        f"Cost: ${cost:.4f}\n"
        f"Time: {elapsed:.0f}s\n"
        f"Files: {file_count}\n"
        f"Run: `{run_id}`"
    )
    send_telegram(msg)


def notify_seed_proposal(title: str, target: str, confidence: str) -> None:
    """Notify about a SEED improvement proposal."""
    msg = (
        f"🌱 *SEED Proposal*\n\n"
        f"Title: {title}\n"
        f"Target: `{target}`\n"
        f"Confidence: {confidence}\n\n"
        f"Run `belief-approve` to apply."
    )
    send_telegram(msg)


def notify_health_issue(issues: list[str]) -> None:
    """Notify about health issues."""
    msg = (
        f"⚠️ *Health Issues*\n\n" +
        "\n".join(f"• {i}" for i in issues)
    )
    send_telegram(msg)
