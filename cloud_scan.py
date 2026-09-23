"""Whole-market signal scanner — the GitHub Actions entry point.

Why this file exists
--------------------
The Cloudflare Worker (``sms-app``) is the always-on push gateway, but Workers
Free caps CPU at 10 ms per invocation and subrequests at 50 per run: far too
little to sweep a whole market. GitHub Actions provides a real Linux VM for
free, so the heavy sweep runs here and only the finished alert text is POSTed to
the Worker's ``/send`` endpoint (ntfy push -> phone).

Coverage (all free tiers, no paid data feed)
--------------------------------------------
* US equities  — Alpaca Market Data (free "Basic" plan). The whole listed market
  (~13k tradable non-OTC symbols) is pulled with *bulk* endpoints, so a full
  sweep costs a few hundred requests instead of one per symbol:
  ``/v2/stocks/bars`` pages ~78 symbols per request for 6 months of daily bars,
  and ``/v2/stocks/snapshots`` returns up to ~2,000 real-time IEX snapshots.
* Crypto       — Alpaca crypto bars/snapshots for every active USD pair.
* Forex + commodities — Yahoo Finance chart API (no key, small curated books).

Honesty note on throughput: the sweep covers the *entire* equity market for
medium-term signals, then ranks what it finds by confluence score and dollar
volume and pushes only ``MAX_ALERTS`` so the phone alert stays readable. Counts
of everything else are reported in the message and the JSON report.

Indicators and scoring come straight from the app's own modules
(``analysis/indicators.py`` -> ``signals/engine.py::technical_signal``), so a
cloud alert scores *identically* to the dashboard — there is no parallel
implementation to drift. Speed comes from bulk fetching (one request serves
~79 symbols) and a liquidity gate *before* indicator computation, not from
re-implemented formulas. ``--verify-indicators N`` dumps N evaluated signals
with their full indicator dicts so the numbers can be eyeballed against the UI.

Usage
-----
    python cloud_scan.py --dry-run              # print, send nothing
    python cloud_scan.py --dry-run --limit 60   # quick smoke test
    python cloud_scan.py                        # live: push to the Worker
    python cloud_scan.py --verify-indicators 3  # dump signals + ind dicts

Environment (``.env`` locally, GitHub Actions secrets in the cloud)
------------------------------------------------------------------
    ALPACA_API_KEY / ALPACA_API_SECRET  data credentials (free Basic plan)
    ALERT_TOKEN                         shared secret for the Worker ``/send``
    ALERT_WORKER_URL                    default https://sms-app.vnt44500.workers.dev
    SCAN_TIMEFRAMES                     default "medium,long" ("short" is
                                        research-only upstream: config.live_timeframes)
    SCAN_DRY_RUN                        "true" to suppress the push
    MIN_PRICE / MIN_DOLLAR_VOLUME       liquidity gates
    MAX_ALERTS                          ranked signals pushed per run
    SCAN_LIMIT                          cap the equity universe (smoke testing)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from analysis.indicators import compute_indicators, last_indicators
from core.universe import COMMODITY_UNIVERSE, FOREX_UNIVERSE
from signals.engine import technical_signal

load_dotenv()

LOGGER = logging.getLogger("cloud_scan")

ROOT = Path(__file__).resolve().parent
DEFAULT_WORKER_URL = "https://sms-app.vnt44500.workers.dev"
MAX_BARS_PER_REQUEST = 10_000
SEPARATOR = "-" * 34

# The app's own scanner scores every horizon off ONE daily indicator frame
# (market_scanner._rank_row: history -> compute_indicators -> last_indicators ->
# technical_signal(summary, timeframe)); the engine picks each horizon's MA pair
# by timeframe *name* from fixed periods (short EMA9/SMA20, medium SMA20/50,
# long SMA50/200). "long" therefore needs >= 200 daily bars, so one frame of
# ~290 trading days serves all three horizons — ~3x fewer bulk fetches than a
# separate hourly/weekly pass, and identical math to the dashboard.
SERIES_INTERVAL = "1Day"
SERIES_DAYS = 400            # calendar days -> ~290 daily bars > sma_200
SERIES_BAR_CAP = 300         # sizes bulk pages: 10k bars / 300 = 34 symbols/page
SERIES_FEED = "sip"          # historical SIP is included in the free plan
SERIES_PAGE = MAX_BARS_PER_REQUEST // SERIES_BAR_CAP   # 34 symbols per bars request

# "short" is research-only upstream (config.research_timeframes: negative net
# expectancy in the held-out crisis window) — computed nowhere near an alert.
LIVE_TIMEFRAMES = ("medium", "long")
TIMEFRAME_LABELS = {"short": "SHORT", "medium": "MEDIUM", "long": "LONG"}


@dataclass
class Settings:
    """Runtime configuration resolved from the environment and CLI flags."""

    alpaca_key: str = ""
    alpaca_secret: str = ""
    alert_token: str = ""
    worker_url: str = DEFAULT_WORKER_URL
    timeframes: tuple[str, ...] = ("medium", "long")
    min_price: float = 2.0
    min_dollar_volume: float = 5_000_000.0
    max_alerts: int = 25
    buy_threshold: float = 0.60     # confluence score >= -> BUY alert
    sell_threshold: float = 0.40    # confluence score <= -> SELL alert (0.5 = neutral)
    assets: tuple[str, ...] = ("equities", "crypto", "forex", "commodities")
    dedup_hours: float = 24.0
    rate_per_minute: int = 190
    limit: int = 0            # 0 -> whole market
    dry_run: bool = False
    state_path: Path = field(default_factory=lambda: ROOT / "cloud_scan_state.json")
    report_path: Path = field(default_factory=lambda: ROOT / "cloud_scan_report.json")

    @classmethod
    def from_env(cls) -> "Settings":
        def flag(name: str, default: bool = False) -> bool:
            return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}

        def number(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            alpaca_key=os.getenv("ALPACA_API_KEY", "").strip(),
            alpaca_secret=os.getenv("ALPACA_API_SECRET", "").strip(),
            alert_token=os.getenv("ALERT_TOKEN", "").strip(),
            worker_url=os.getenv("ALERT_WORKER_URL", DEFAULT_WORKER_URL).rstrip("/"),
            timeframes=tuple(
                name.strip().lower()
                for name in os.getenv("SCAN_TIMEFRAMES", "medium,long").split(",")
                if name.strip()
            ),
            min_price=number("MIN_PRICE", 2.0),
            min_dollar_volume=number("MIN_DOLLAR_VOLUME", 5_000_000.0),
            max_alerts=int(number("MAX_ALERTS", 25)),
            buy_threshold=number("BUY_THRESHOLD", 0.60),
            sell_threshold=number("SELL_THRESHOLD", 0.40),
            assets=tuple(
                name.strip().lower()
                for name in os.getenv("SCAN_ASSETS", "equities,crypto,forex,commodities").split(",")
                if name.strip()
            ),
            dedup_hours=number("DEDUP_HOURS", 24.0),
            rate_per_minute=int(number("ALPACA_RATE_PER_MIN", 190)),
            limit=int(number("SCAN_LIMIT", 0)),
            dry_run=flag("SCAN_DRY_RUN"),
        )


class RateLimiter:
    """Simple shared-interval limiter (Alpaca allows 200 requests/minute)."""

    def __init__(self, per_minute: int):
        self._interval = 60.0 / max(1, per_minute)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
                now = time.monotonic()


def _frame_from_alpaca(rows: list[dict]) -> pd.DataFrame:
    """Build a lowercase OHLCV frame from Alpaca bar dicts (same keys as Yahoo)."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame({
        "open": [r.get("o") for r in rows],
        "high": [r.get("h") for r in rows],
        "low": [r.get("l") for r in rows],
        "close": [r.get("c") for r in rows],
        "volume": [r.get("v") for r in rows],
    }, index=pd.to_datetime([r["t"] for r in rows], utc=True))
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame[~frame.index.duplicated(keep="last")].sort_index()


class AlpacaClient:
    """Rate-limited Alpaca REST client (free Basic plan: 200 requests/minute)."""

    DATA = "https://data.alpaca.markets"
    TRADE = "https://paper-api.alpaca.markets"

    def __init__(self, settings: Settings):
        self._limiter = RateLimiter(settings.rate_per_minute)
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": settings.alpaca_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret,
        })

    def _get(self, url: str, params: dict, timeout: int = 90) -> dict | None:
        for attempt in range(4):
            self._limiter.wait()
            try:
                response = self._session.get(url, params=params, timeout=timeout)
            except requests.RequestException as error:
                LOGGER.warning("alpaca GET failed (%s/4): %s", attempt + 1, error)
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:          # rate limit -> back off hard
                time.sleep(10)
                continue
            LOGGER.warning("alpaca %s -> %s %s", url, response.status_code, response.text[:200])
            return None
        return None

    def equity_universe(self) -> list[str]:
        """Every active, tradable, non-OTC US equity (~13k symbols)."""
        data = self._get(f"{self.TRADE}/v2/assets",
                         {"status": "active", "asset_class": "us_equity"})
        if not data:
            return []
        return sorted(a["symbol"] for a in data
                      if a.get("tradable") and a.get("exchange") != "OTC")

    def daily_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Bulk daily bars for *symbols* (~34 symbols per request, 1y lookback)."""
        start = (datetime.now(timezone.utc) - timedelta(days=400)).date().isoformat()
        frames: dict[str, pd.DataFrame] = {}
        pages = 0
        for offset in range(0, len(symbols), SERIES_PAGE):
            batch = symbols[offset:offset + SERIES_PAGE]
            token = None
            collected: dict[str, list[dict]] = {}
            while True:
                params = {"symbols": ",".join(batch), "timeframe": "1Day",
                          "start": start, "limit": MAX_BARS_PER_REQUEST,
                          "adjustment": "split", "feed": "sip"}
                if token:
                    params["page_token"] = token
                data = self._get(f"{self.DATA}/v2/stocks/bars", params)
                if not data:
                    break
                for symbol, bars in (data.get("bars") or {}).items():
                    collected.setdefault(symbol, []).extend(bars)
                token = data.get("next_page_token")
                if not token:
                    break
            for symbol, rows in collected.items():
                frame = _frame_from_alpaca(rows)
                if not frame.empty:
                    frames[symbol] = frame
            pages += 1
            if pages % 40 == 0:
                LOGGER.info("equity bars: %d/%d symbols fetched", len(frames), len(symbols))
        return frames

    def crypto_universe(self) -> list[str]:
        """Active crypto pairs (trade assets endpoint; symbols arrive in
        the BTC/USD form the bars endpoint expects)."""
        data = self._get(f"{self.TRADE}/v2/assets",
                         {"asset_class": "crypto", "status": "active"})
        if data:
            symbols = [s["symbol"] for s in data
                       if s.get("status") == "active" and s.get("tradable")]
            if symbols:
                return sorted(symbols)
        return ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD",
                "AVAX/USD", "LINK/USD", "DOT/USD", "LTC/USD", "BCH/USD", "BNB/USD"]

    def crypto_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Bulk 1-day crypto bars (v1beta3, no subscription gate)."""
        start = (datetime.now(timezone.utc) - timedelta(days=400)).date().isoformat()
        frames: dict[str, pd.DataFrame] = {}
        for offset in range(0, len(symbols), SERIES_PAGE):
            batch = symbols[offset:offset + SERIES_PAGE]
            data = self._get(f"{self.DATA}/v1beta3/crypto/us/bars",
                             {"symbols": ",".join(batch), "timeframe": "1Day",
                              "start": start, "limit": MAX_BARS_PER_REQUEST})
            if not data:
                continue
            for symbol, payload in (data.get("bars") or {}).items():
                rows = payload if isinstance(payload, list) else payload.get("bars", [])
                frame = _frame_from_alpaca(rows)
                if not frame.empty:
                    frames[symbol] = frame
        return frames


def fetch_yahoo(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Free Yahoo Finance chart API — forex + commodities (no key required)."""
    try:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": period, "interval": interval},
            headers={"User-Agent": "Mozilla/5.0 (trading-dashboard cloud scan)"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame({
            "open": quote.get("open"), "high": quote.get("high"),
            "low": quote.get("low"), "close": quote.get("close"),
            "volume": quote.get("volume"),
        }, index=pd.to_datetime(result["timestamp"], unit="s", utc=True))
        return frame.dropna(subset=["open", "high", "low", "close"])
    except Exception as error:                      # noqa: BLE001 - one bad symbol must not kill the sweep
        LOGGER.warning("yahoo %s failed: %s", symbol, error)
        return pd.DataFrame()


def evaluate(symbol: str, frame: pd.DataFrame, settings: Settings) -> list[dict]:
    """Score one symbol — identical call chain to market_scanner._rank_row.

    compute_indicators(frame) -> last_indicators(frame) -> inject ``_price``
    -> technical_signal(summary, timeframe). Returns every timeframe whose
    confluence score crosses a BUY/SELL threshold (0.5 is neutral).
    """
    if frame.empty or len(frame) < 30:
        return []
    try:
        indicators = compute_indicators(frame)
    except Exception as error:                      # noqa: BLE001
        LOGGER.debug("indicators failed for %s: %s", symbol, error)
        return []
    if indicators.empty:
        return []
    summary = last_indicators(indicators)
    price = summary.get("close")
    if not price:
        return []
    summary["_price"] = price
    hits: list[dict] = []
    for timeframe in settings.timeframes:
        if timeframe not in LIVE_TIMEFRAMES and timeframe != "short":
            continue
        result = technical_signal(summary, timeframe)
        score = result.get("score")
        if score is None:
            continue
        if score >= settings.buy_threshold:
            action = "BUY"
        elif score <= settings.sell_threshold:
            action = "SELL"
        else:
            continue                                # inside the neutral band
        hits.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "action": action,
            "score": score,
            "price": round(float(price), 4),
            "reasons": result.get("reasons", []),
        })
    return hits


class DedupStore:
    """24h per (symbol, timeframe, action) key — mirrors signals/tracker.py."""

    def __init__(self, path: Path, hours: float):
        self._path = path
        self._cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        self._seen: dict[str, str] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self._seen = {k: v for k, v in raw.items()
                              if datetime.fromisoformat(v) > self._cutoff}
            except (ValueError, OSError, TypeError):
                self._seen = {}

    @staticmethod
    def _key(symbol: str, timeframe: str, action: str) -> str:
        return f"{symbol}|{timeframe}|{action}"

    def fresh(self, symbol: str, timeframe: str, action: str) -> bool:
        return self._key(symbol, timeframe, action) not in self._seen

    def record(self, symbol: str, timeframe: str, action: str) -> None:
        self._seen[self._key(symbol, timeframe, action)] = \
            datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._seen, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)


def render_message(stats: dict, hits: list[dict], skipped: int) -> str:
    """Compact ntfy-friendly alert text (title handled by the Worker)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"[TradingDash] Market scan {stamp}",
             f"Sweep: {stats.get('equities', 0):,} equities | "
             f"{stats.get('crypto', 0)} crypto | {stats.get('forex', 0)} FX | "
             f"{stats.get('commodities', 0)} commodities"]
    lines.append(f"Signals: {len(hits)} new"
                 + (f" ({skipped} suppressed by 24h dedup)" if skipped else ""))
    lines.append(SEPARATOR)
    for hit in hits:
        label = TIMEFRAME_LABELS.get(hit["timeframe"], hit["timeframe"].upper())
        lines.append(f"{hit['symbol']} {label} {hit['action']} "
                     f"{hit['score']:.2f} @ {hit['price']:g}")
    return "\n".join(lines)


def _score_hits(settings: Settings, store: DedupStore,
                raw_hits: list[dict]) -> tuple[list[dict], int]:
    """Dedup, then rank by distance from neutral (0.5), strongest first."""
    fresh, suppressed = [], 0
    for hit in raw_hits:
        if store.fresh(hit["symbol"], hit["timeframe"], hit["action"]):
            fresh.append(hit)
        else:
            suppressed += 1
    fresh.sort(key=lambda h: abs(h["score"] - 0.5), reverse=True)
    return fresh[:settings.max_alerts], suppressed


def run(settings: Settings, verify: int = 0) -> int:
    """Full sweep across every enabled asset class. Returns process exit code."""
    if not settings.alpaca_key or not settings.alpaca_secret:
        LOGGER.error("ALPACA_API_KEY / ALPACA_API_SECRET missing")
        return 2
    client = AlpacaClient(settings)
    store = DedupStore(settings.state_path, settings.dedup_hours)
    stats: dict[str, int] = {}
    raw_hits: list[dict] = []
    verify_summaries: dict[str, dict] = {}

    # --- US equities: the whole listed market, bulk pages -------------------
    if "equities" in settings.assets:
        symbols = client.equity_universe()
        if settings.limit:
            symbols = symbols[:settings.limit]
        stats["equities"] = len(symbols)
        LOGGER.info("equity universe: %d symbols", len(symbols))
        frames = client.daily_bars(symbols) if symbols else {}
        LOGGER.info("equity frames fetched: %d", len(frames))
        for symbol, frame in frames.items():
            hits = evaluate(symbol, frame, settings)
            if hits:
                raw_hits.extend(hits)
                if verify and len(verify_summaries) < verify:
                    summary = last_indicators(compute_indicators(frame))
                    summary["_price"] = summary.get("close")
                    verify_summaries[symbol] = summary

    # --- Crypto: every active USD pair -------------------------------------
    if "crypto" in settings.assets:
        crypto_symbols = client.crypto_universe()
        stats["crypto"] = len(crypto_symbols)
        for symbol, frame in client.crypto_bars(crypto_symbols).items():
            raw_hits.extend(evaluate(symbol, frame, settings))

    # --- Forex + commodities: curated books on the free Yahoo chart API -----
    yahoo_books = []
    if "forex" in settings.assets:
        yahoo_books.extend((s, "forex") for s in FOREX_UNIVERSE)
    if "commodities" in settings.assets:
        yahoo_books.extend((s, "commodities") for s in COMMODITY_UNIVERSE)
    stats["forex"] = sum(1 for _, c in yahoo_books if c == "forex")
    stats["commodities"] = sum(1 for _, c in yahoo_books if c == "commodities")
    for symbol, _asset_class in yahoo_books:
        raw_hits.extend(evaluate(symbol, fetch_yahoo(symbol), settings))
        time.sleep(0.3)                        # polite to Yahoo's free endpoint

    hits, suppressed = _score_hits(settings, store, raw_hits)
    message = render_message(stats, hits, suppressed)
    LOGGER.info("signals raw=%d fresh=%d suppressed=%d",
                len(raw_hits), len(hits), suppressed)

    if verify:
        for symbol, summary in list(verify_summaries.items())[:verify]:
            print(f"\n[verify] {symbol} indicators:")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in summary.items()}, indent=1, default=str))

    print("\n" + message)

    # Dedup is committed ONLY after the alert is confirmed delivered: a failed
    # push (ntfy throttling, Worker outage) must leave the signals fresh so the
    # next sweep retries them instead of silently swallowing them for 24h —
    # run #5 lost a full sweep by recording before pushing. Dry runs never
    # consume signals either: they push nothing, so nothing is "seen".
    push_ok: bool | None = None
    if settings.dry_run:
        LOGGER.info("dry run: nothing pushed")
    elif not hits:
        LOGGER.info("no fresh signals: nothing to push")
    else:
        push_ok = push_to_worker(settings, message)
        if push_ok:
            for hit in hits:
                store.record(hit["symbol"], hit["timeframe"], hit["action"])
            store.save()

    report = {
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats,
        "raw_signals": len(raw_hits),
        "pushed": len(hits) if push_ok else 0,
        "push_attempted": push_ok is not None,
        "push_ok": push_ok,
        "suppressed_by_dedup": suppressed,
        "signals": hits,
        "dry_run": settings.dry_run,
    }
    settings.report_path.write_text(json.dumps(report, indent=1, default=str),
                                    encoding="utf-8")
    return 0 if push_ok is not False else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Whole-market signal scanner")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the alert instead of pushing it")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the equity universe (smoke testing)")
    parser.add_argument("--verify-indicators", type=int, default=0, metavar="N",
                        help="dump indicator dicts for the first N signals")
    parser.add_argument("--assets", default="",
                        help="comma list: equities,crypto,forex,commodities")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.dry_run:
        settings.dry_run = True
    if args.limit:
        settings.limit = args.limit
    if args.assets:
        settings.assets = tuple(a.strip().lower()
                                for a in args.assets.split(",") if a.strip())

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    return run(settings, verify=args.verify_indicators)


def push_to_worker(settings: Settings, message: str) -> bool:
    """POST the finished alert to the Worker /send -> ntfy -> phone.

    Retries with growing pauses because ntfy.sh rate-limits shared cloud egress
    IPs (GitHub runners) with 429s that usually clear within a couple of
    minutes. Per-attempt timeout is generous: the Worker itself runs internal
    ntfy retries (honouring ntfy's Retry-After) before answering.
    """
    url = f"{settings.worker_url}/send"
    for attempt in range(4):
        try:
            response = requests.post(
                url, json={"message": message},
                headers={"x-alert-token": settings.alert_token},
                timeout=90,
            )
            if response.ok:
                LOGGER.info("pushed: %s", response.text[:200])
                return True
            LOGGER.warning("worker %s -> %s %s", url, response.status_code,
                           response.text[:200])
        except requests.RequestException as error:
            LOGGER.warning("push attempt %d failed: %s", attempt + 1, error)
        if attempt < 3:
            time.sleep(10 * (attempt + 1))
    return False


if __name__ == "__main__":
    raise SystemExit(main())

