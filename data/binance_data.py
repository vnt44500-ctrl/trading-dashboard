"""Free public Binance spot intraday history provider."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"


def _binance_symbol(symbol: str) -> str:
    return symbol.upper().replace("-USD", "USDT").replace("-USDT", "USDT")


def get_quote(symbol: str) -> dict:
    """Return a public Binance 24-hour quote for a USD crypto symbol."""
    response = requests.get(BINANCE_TICKER_URL, params={"symbol": _binance_symbol(symbol)}, timeout=15)
    response.raise_for_status()
    payload = response.json()
    return {
        "price": float(payload["lastPrice"]),
        "prev_close": float(payload["prevClosePrice"]),
        "source": "Binance public API",
        "volume": float(payload["volume"]),
    }


def get_history(symbol: str = "BTCUSDT", interval: str = "1h", years: int = 5) -> pd.DataFrame:
    """Download genuine public Binance candles with chronological pagination."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows = []
    while start_ms < end_ms:
        response = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": _binance_symbol(symbol), "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_start = int(batch[-1][0]) + 1
        if next_start <= start_ms:
            break
        start_ms = next_start
        if len(batch) < 1000:
            break
        time.sleep(0.08)

    if not rows:
        return pd.DataFrame()
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    data = pd.DataFrame(rows, columns=columns)
    data.index = pd.to_datetime(data.pop("open_time"), unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[["open", "high", "low", "close", "volume"]].sort_index().loc[~data.index.duplicated()]