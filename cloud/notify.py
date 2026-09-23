"""Alert relay: posts digests to the Cloudflare worker, which pushes via ntfy.

The worker owns delivery (ntfy push now; it keeps retry/backoff for ntfy's rate
limits). This module only has to hand it a message, so the scanner never depends
on a carrier or an SMTP provider being reachable from the runner.
"""
from __future__ import annotations

import logging
import time

import requests

from cloud.config import (ALERT_CHUNK_CHARS, ALERT_CHUNK_DELAY, ALERT_TOKEN,
                          ALERT_URL, HTTP_TIMEOUT)

logger = logging.getLogger(__name__)


def configured() -> bool:
    return bool(ALERT_URL and ALERT_TOKEN)


def chunk_message(message: str, limit: int = 0) -> list[str]:
    """Split a digest on line boundaries so no push exceeds the worker's cap."""
    cap = limit or ALERT_CHUNK_CHARS
    if len(message) <= cap:
        return [message]
    parts: list[str] = []
    current = ""
    for line in message.splitlines(keepends=True):
        if len(current) + len(line) > cap and current:
            parts.append(current.rstrip("\n"))
            current = ""
        while len(line) > cap:
            parts.append(line[:cap])
            line = line[cap:]
        current += line
    if current.strip():
        parts.append(current.rstrip("\n"))
    return parts


def send(message: str, url: str = "", token: str = "", dry_run: bool = False) -> dict:
    """Send one digest, splitting it when needed. Returns a delivery summary."""
    target = (url or ALERT_URL).rstrip("/")
    auth = token or ALERT_TOKEN
    parts = chunk_message(message)
    if dry_run:
        return {"sent": False, "dry_run": True, "parts": len(parts), "message": message}
    if not target or not auth:
        return {"sent": False, "error": "ALERT_URL / ALERT_TOKEN not configured",
                "parts": len(parts), "message": message}

    delivered = []
    for index, part in enumerate(parts):
        headline = part if len(parts) == 1 else f"[{index + 1}/{len(parts)}]\n{part}"
        try:
            response = requests.post(f"{target}/send", json={"message": headline},
                                     headers={"x-alert-token": auth,
                                              "content-type": "application/json"},
                                     timeout=HTTP_TIMEOUT)
            body = response.text[:300]
            delivered.append({"part": index + 1, "status": response.status_code, "body": body})
            if response.status_code >= 400:
                logger.warning("Alert part %d rejected: HTTP %d %s",
                               index + 1, response.status_code, body)
        except requests.RequestException as error:
            delivered.append({"part": index + 1, "status": 0, "body": str(error)})
            logger.warning("Alert part %d failed: %s", index + 1, error)
        if index < len(parts) - 1:
            time.sleep(ALERT_CHUNK_DELAY)

    ok = all(item["status"] == 200 for item in delivered)
    return {"sent": ok, "parts": len(parts), "delivered": delivered}
