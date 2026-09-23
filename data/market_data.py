"""Market data provider using yfinance (free) for quotes and history.

Handles retries, rate-limit friendliness, and caching.
"""
import logging
import time
from datetime import datetime
import re

import pandas as pd
import yfinance as yf
import requests

from config import config

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Fetch real-time quotes and historical OHLCV data."""

    def __init__(self):
        self._ticker_cache = {}

    def _ticker(self, symbol: str):
        key = symbol.upper()
        if key not in self._ticker_cache:
            self._ticker_cache[key] = yf.Ticker(key)
        return self._ticker_cache[key]

    def _normalize_option_symbol(self, symbol: str):
        cleaned = symbol.strip().upper()
        match = re.match(r"^([A-Z0-9.\-]+)\s+(\d{4}-\d{2}-\d{2})\s+(CALL|PUT)\s+([0-9.]+)$", cleaned)
        if match:
            return match.group(1), {
                "label": cleaned,
                "expiry": match.group(2),
                "option_type": match.group(3).lower(),
                "strike": float(match.group(4)),
            }
        if " CALL" in cleaned or " PUT" in cleaned or " OPTION" in cleaned:
            base = re.split(r"\s+(?:CALL|PUT|OPTION)\b", cleaned, maxsplit=1)[0].strip()
            return base, {"label": cleaned}
        return cleaned, None

    def get_quote(self, symbol: str) -> dict:
        """Return the latest quote for a symbol with retries."""
        base_symbol, option_label = self._normalize_option_symbol(symbol)
        for attempt in range(config.max_retries):
            try:
                t = self._ticker(base_symbol)
                info = t.info or {}
                fast = t.fast_info
                price = None

                if fast and getattr(fast, "last_price", None):
                    price = fast.last_price
                elif info.get("regularMarketPrice"):
                    price = info["regularMarketPrice"]
                elif info.get("currentPrice"):
                    price = info["currentPrice"]

                if price is None and not option_label:
                    raise ValueError(f"No Yahoo price returned for {base_symbol}")

                if option_label:
                    option_price = price
                    if option_label.get("expiry"):
                        chain = t.option_chain(option_label["expiry"])
                        options = getattr(chain, option_label["option_type"] + "s")
                        match = options[options["strike"] == option_label["strike"]]
                        if not match.empty:
                            contract = match.iloc[0]
                            option_price = contract.get("lastPrice") or contract.get("ask") or contract.get("bid")
                    return {
                        "symbol": base_symbol,
                        "price": option_price,
                        "prev_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Yahoo Finance Options",
                        "option_label": option_label["label"],
                        "name": info.get("shortName") or info.get("longName") or base_symbol,
                    }

                return {
                    "symbol": base_symbol,
                    "price": price,
                    "prev_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                    "open": info.get("open") or info.get("regularMarketOpen"),
                    "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                    "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
                    "volume": info.get("volume") or info.get("regularMarketVolume"),
                    "market_cap": info.get("marketCap"),
                    "pe": info.get("trailingPE"),
                    "eps": info.get("trailingEps"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "currency": info.get("currency"),
                    "name": info.get("shortName") or info.get("longName") or base_symbol,
                    "currentRatio": info.get("currentRatio"),
                    "debtToEquity": info.get("debtToEquity"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Yahoo Finance",
                }
            except Exception as error:
                logger.warning("Quote fetch failed for %s (attempt %d): %s", base_symbol, attempt + 1, error)
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_backoff * (attempt + 1))
                else:
                    return self._crypto_fallback_quote(base_symbol)
        return self._crypto_fallback_quote(base_symbol)

    def _crypto_fallback_quote(self, symbol: str) -> dict:
        """Try public crypto feeds, then give up with an explicit diagnostic.

        Binance is attempted before CoinGecko because it returns the same
        OHLC-style payload the rest of the app expects. The CoinGecko lookup
        previously relied on a module-level dict that was only populated when
        the user happened to search for the coin in the current session, so it
        returned ``price: None`` for most symbols.
        """
        if "-USD" in symbol.upper() or "-USDT" in symbol.upper():
            try:
                from data.binance_data import get_quote as get_binance_quote
                quote = get_binance_quote(symbol)
                if quote.get("price"):
                    quote["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    quote.setdefault("name", symbol)
                    return quote
            except Exception as error:
                logger.warning("Binance quote fallback failed for %s: %s", symbol, error)
            return self._get_coingecko_quote(symbol)
        return {"symbol": symbol, "price": None, "name": symbol,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    def _get_coingecko_quote(self, symbol: str) -> dict:
        try:
            from core.assets import COINGECKO_SYMBOL_IDS
            coin_id = COINGECKO_SYMBOL_IDS.get(symbol.upper())
            if not coin_id:
                return {"symbol": symbol, "price": None, "name": symbol, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=5,
            )
            response.raise_for_status()
            values = response.json().get(coin_id, {})
            return {
                "symbol": symbol,
                "price": values.get("usd"),
                "prev_close": None,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CoinGecko",
                "name": symbol,
            }
        except (requests.RequestException, ValueError, TypeError):
            return {"symbol": symbol, "price": None, "name": symbol, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV history as a DataFrame with normalized columns.

        The current (in-progress) session bar is removed. Yahoo Finance returns
        that bar with ``open``/``high``/``low``/``close`` set to NaN and only a
        partial ``volume`` value, which previously poisoned every indicator
        computed from the final row.
        """
        for attempt in range(config.max_retries):
            try:
                df = self._ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    cleaned = self._drop_incomplete_bars(df)
                    if not cleaned.empty:
                        return cleaned
                    logger.warning("All bars for %s were incomplete; returning empty frame.", symbol)
                    return pd.DataFrame()
            except Exception as error:
                logger.warning("History fetch failed for %s (attempt %d): %s", symbol, attempt + 1, error)
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_backoff * (attempt + 1))
        return pd.DataFrame()

    @staticmethod
    def _drop_incomplete_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows without a usable OHLC set, then de-duplicate the index."""
        price_columns = [column for column in ("open", "high", "low", "close") if column in df.columns]
        if not price_columns:
            return df
        cleaned = df.dropna(subset=price_columns)
        cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")]
        return cleaned.sort_index()

    def get_intraday(self, symbol: str, interval: str = "1m", period: str = "1d") -> pd.DataFrame:
        """Return recent intraday data for live ticks."""
        return self.get_history(symbol, period=period, interval=interval)

    def get_option_expiries(self, symbol: str):
        try:
            return self._ticker(symbol).options
        except Exception:
            return []

    def get_option_chain(self, symbol: str, expiry: str = None):
        try:
            ticker = self._ticker(symbol)
            expiries = ticker.options
            if expiry is None and expiries:
                expiry = expiries[0]
            if not expiry:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
            chain = ticker.option_chain(expiry)
            return {"calls": chain.calls, "puts": chain.puts, "expiry": expiry}
        except Exception:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": expiry}


provider = MarketDataProvider()