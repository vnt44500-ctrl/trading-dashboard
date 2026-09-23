"""Bulk market-data provider built on Alpaca's free Market Data API.

Why this exists next to ``data/market_data.py``: Yahoo's chart API is one
symbol per HTTP request, so covering the whole US market (~13,000 tradable
symbols) would take 13,000 requests. Alpaca's free "Basic" plan exposes *bulk*
endpoints - one request returns many symbols - which is what makes a
whole-market scan fit inside a free compute and rate-limit budget.

Verified against the live API (Sept 2026):

* ``GET /v2/assets``               full US equity universe in one request
* ``GET /v2/stocks/bars``          paginated bulk OHLCV, any timeframe
* ``GET /v2/stocks/snapshots``     latest trade + daily bar, ~2,000 symbols
* ``GET /v1beta3/crypto/us/bars``  crypto OHLCV (trades around the clock)

Free-plan constraints this module is designed around:

* **200 requests/minute, account-wide.** Every parallel shard shares the same
  key, so the pacer here is per-process and the caller is expected to divide
  the allowance by the number of shards (``run_cloud_scan.py`` does).
* **Recent SIP data is not permitted** on the free plan (HTTP 403 for
  snapshot requests), so snapshots must use ``feed=iex``. IEX prints are real
  exchange trades (a few percent of consolidated volume), not the full tape.
* Historical daily/weekly bars *are* available on the SIP feed.

Nothing in the Streamlit app imports this module; the dashboard keeps using
``data/market_data.py``. It exists for the headless cloud scan only.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_BASE = "https://data.alpaca.markets"
TRADE_BASE = "https://paper-api.alpaca.markets"

# Alpaca caps a single bars response at 10,000 bars. Chunking by symbol keeps
# each request comfortably inside that, and page tokens handle the rest.
MAX_BARS_PER_RESPONSE = 10000
DEFAULT_RPM = 200

_CREDENTIAL_ALIASES = (
    ("ALPACA_API_KEY", "APCA_API_KEY_ID"),
    ("ALPACA_API_SECRET", "APCA_API_SECRET_KEY"),
)


def credentials_available() -> bool:
    """True when an Alpaca key/secret pair is present in the environment."""
    return all(any(os.getenv(name) for name in aliases) for aliases in _CREDENTIAL_ALIASES)


def _credential(primary: str, alias: str) -> str:
    return os.getenv(primary) or os.getenv(alias) or ""


class BulkMarketData:
    """Bulk quotes and OHLCV for whole-market scans.

    Every public method returns plain dicts/DataFrames so callers never need to
    know about pagination, feeds, or rate limiting.
    """

    def __init__(self, key: str | None = None, secret: str | None = None,
                 requests_per_minute: int = DEFAULT_RPM):
        self.key = key if key is not None else _credential("ALPACA_API_KEY", "APCA_API_KEY_ID")
        self.secret = secret if secret is not None else _credential("ALPACA_API_SECRET", "APCA_API_SECRET_KEY")
        self.rpm = max(1, int(requests_per_minute))
        self._recent: deque[float] = deque()
        self.request_count = 0

    # -- transport ---------------------------------------------------------
    @property
    def headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    def configured(self) -> bool:
        return bool(self.key and self.secret)

    def _pace(self) -> None:
        """Block until this process is inside its share of the rate limit.

        A sliding one-minute window is used rather than a fixed sleep so short
        bursts stay fast while the rolling average is still respected.
        """
        while True:
            now = time.monotonic()
            while self._recent and now - self._recent[0] >= 60.0:
                self._recent.popleft()
            if len(self._recent) < self.rpm:
                self._recent.append(now)
                return
            time.sleep(max(0.05, 60.0 - (now - self._recent[0]) + 0.01))

    def _get(self, base: str, path: str, params: dict | None = None,
             attempts: int = 4) -> dict:
        """GET with rate-limit pacing and backoff on 429/5xx."""
        if not self.configured():
            raise RuntimeError("Alpaca credentials missing (ALPACA_API_KEY / ALPACA_API_SECRET)")
        last_error: Exception | None = None
        for attempt in range(attempts):
            self._pace()
            try:
                response = requests.get(f"{base}{path}", params=params or {},
                                        headers=self.headers, timeout=60)
                self.request_count += 1
                if response.status_code == 200:
                    return response.json()
                # 429 / 5xx are transient; 403 on snapshots means the free plan
                # cannot read recent SIP data, which the caller fixes with
                # feed=iex. Retrying that is pointless.
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = float(response.headers.get("Retry-After") or 0) or (2 ** attempt)
                    logger.warning("Alpaca %s returned %s; retrying in %.1fs",
                                   path, response.status_code, wait)
                    time.sleep(wait)
                    last_error = RuntimeError(f"HTTP {response.status_code} for {path}")
                    continue
                raise RuntimeError(f"HTTP {response.status_code} for {path}: {response.text[:200]}")
            except requests.RequestException as error:
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Alpaca request failed after {attempts} attempts: {last_error}")
