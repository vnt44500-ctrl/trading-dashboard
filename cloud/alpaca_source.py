"""Bulk market-data access through Alpaca's free market-data API.

Why this exists
---------------
``data/market_data.py`` (the dashboard's own provider) asks Yahoo Finance for
one symbol per HTTP request. That caps any single run at a few hundred symbols
and makes a market-wide sweep impossible. Alpaca's free "Basic" plan instead
exposes *bulk* endpoints - one request returns many symbols - so the identical
signal pipeline can cover every US equity in about a hundred calls.

Capacities measured live against the free plan (September 2026)
---------------------------------------------------------------
``GET /v2/assets``                 14,356 US equities in one response
``GET /v2/stocks/snapshots``       ~2,000 symbols per request
``GET /v2/stocks/bars``            a page holds roughly 10,000 bars, so the
                                   symbol count per page is derived from the
                                   requested history length
rate limit                         200 requests / minute

Free-plan quirks this module absorbs
------------------------------------
* Recent SIP data is not entitled (``subscription does not permit querying
  recent SIP data``), so snapshots must pass ``feed=iex``.
* Historical (non-recent) SIP bars *are* entitled, so daily history uses
  ``feed=sip`` for full-market coverage while intraday uses ``feed=iex``.
* Timeframes need very different page sizes, so the chunk size is computed
  from the expected bar count instead of being hard-coded.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import requests

from cloud.config import (ALPACA_BAR_LIMIT, ALPACA_DATA_URL, ALPACA_KEY,
                          ALPACA_RATE_PER_MINUTE, ALPACA_SECRET,
                          ALPACA_TRADE_URL, HTTP_TIMEOUT)

logger = logging.getLogger(__name__)

STOCK_BARS_PATH = "/v2/stocks/bars"
CRYPTO_BARS_PATH = "/v1beta3/crypto/us/bars"

# Trading sessions per calendar day, used to size a page before fetching it.
_BARS_PER_DAY = {"1Day": 5.0 / 7.0, "1Hour": 7.0, "1Week": 1.0 / 7.0}


class AlpacaError(RuntimeError):
    """Raised when Alpaca returns a response the scan cannot use."""


@dataclass
class BarMatrix:
    """A ``(symbols x timestamps)`` view of close and volume history.

    Rows that never traded on a given timestamp keep ``NaN`` closes, which the
    cross-detection helpers treat as "no signal" rather than a false one.
    """

    symbols: list[str]
    timestamps: list[int]
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:


class BulkSource:
    """Rate-limited, paginated client for Alpaca's bulk endpoints."""

    def __init__(self, key: str = "", secret: str = "", timeout: int = 0):
        self.key = key or ALPACA_KEY
        self.secret = secret or ALPACA_SECRET
        self.timeout = timeout or HTTP_TIMEOUT
        self._calls: deque[float] = deque()
        self.calls_made = 0
        if not (self.key and self.secret):
            raise AlpacaError("ALPACA_API_KEY / ALPACA_API_SECRET are not set")

    # --- transport ----------------------------------------------------------
    @property
    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    def _throttle(self) -> None:
        """Keep the trailing 60-second call count under the plan's limit."""
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60.0:
            self._calls.popleft()
        if len(self._calls) >= ALPACA_RATE_PER_MINUTE:
            sleep_for = 60.0 - (now - self._calls[0]) + 0.05
            logger.info("Rate budget reached; pausing %.1fs", sleep_for)
            time.sleep(max(sleep_for, 0.05))
            now = time.monotonic()
            while self._calls and now - self._calls[0] > 60.0:
                self._calls.popleft()
        self._calls.append(now)

    def _get(self, url: str, params: dict, attempts: int = 4) -> dict:
        last_error = ""
        for attempt in range(1, attempts + 1):
            self._throttle()
            try:
                response = requests.get(url, params=params, headers=self._headers,
                                        timeout=self.timeout)
            except requests.RequestException as error:
                last_error = str(error)
            else:
                self.calls_made += 1
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code in (400, 401, 403):
                    # A permission or payload problem will not fix itself.
                    raise AlpacaError(f"{url} -> {last_error}")
            if attempt < attempts:
                time.sleep(attempt * 1.5)
        raise AlpacaError(f"{url} failed after {attempts} attempts: {last_error}")

        return len(self.symbols)
