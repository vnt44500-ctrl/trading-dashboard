"""Environment-driven configuration for the cloud market scanner.

Every knob is read from the environment so the identical code runs locally
(for testing) and on a scheduled CI runner. Secrets — the Alpaca key pair and
the alert token — are injected by CI secret storage and never committed.
"""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# --- Alpaca market data ------------------------------------------------------
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_TRADE_URL = os.getenv("ALPACA_TRADE_URL", "https://paper-api.alpaca.markets")

# The free "Basic" market-data plan allows 200 requests/minute. Stay just under
# it so a long paginated sweep never trips the limiter mid-run.
ALPACA_RATE_PER_MINUTE = _int("ALPACA_RATE_PER_MINUTE", 180)
ALPACA_SYMBOL_CHUNK = _int("ALPACA_SYMBOL_CHUNK", 250)
ALPACA_BAR_LIMIT = _int("ALPACA_BAR_LIMIT", 10000)

# --- Alert relay (Cloudflare worker -> ntfy push) ----------------------------
ALERT_URL = os.getenv("ALERT_URL", "")
ALERT_TOKEN = os.getenv("ALERT_TOKEN", "")
# The worker truncates a single message at 1600 characters, so longer digests
# are posted as several messages.
ALERT_CHUNK_CHARS = _int("ALERT_CHUNK_CHARS", 1450)
ALERT_CHUNK_DELAY = _float("ALERT_CHUNK_DELAY", 1.5)

# --- Scan scope --------------------------------------------------------------
# Liquidity gates. A whole-market sweep without them reports hundreds of
# penny-stock crossovers, which buries the signals that matter.
MIN_PRICE = _float("SCAN_MIN_PRICE", 1.0)
MIN_DOLLAR_VOLUME = _float("SCAN_MIN_DOLLAR_VOLUME", 1_000_000.0)
# How many of the most liquid names get the hourly (short-horizon) sweep.
SHORT_UNIVERSE_SIZE = _int("SCAN_SHORT_UNIVERSE", 600)
MAX_SIGNALS_PER_ALERT = _int("SCAN_MAX_SIGNALS", 30)
# A signal re-firing inside this window is treated as the same event.
DEDUP_HOURS = _int("SCAN_DEDUP_HOURS", 12)
STATE_PATH = os.getenv("SCAN_STATE_PATH", "cloud/state/signal_state.json")
HTTP_TIMEOUT = _int("SCAN_HTTP_TIMEOUT", 45)
