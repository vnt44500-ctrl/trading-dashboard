"""Market-wide ranking for liquid instruments available through Yahoo Finance."""
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Callable, Dict, Iterable, List
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.fundamentals import analyze_fundamentals

logger = logging.getLogger(__name__)

STOCK_CANDIDATES = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "LLY",
    "JPM", "V", "UNH", "XOM", "MA", "COST", "WMT", "HD", "NFLX", "AMD", "ORCL",
    "CRM", "BAC", "KO", "PEP", "ADBE", "QCOM", "INTC", "CSCO", "MCD", "IBM",
]
CRYPTO_CANDIDATES = [
    "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "XRP-USD", "SOL-USD", "USDC-USD", "DOGE-USD",
    "ADA-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "TON-USD", "SHIB-USD", "DOT-USD", "BCH-USD",
    "LTC-USD", "NEAR-USD", "UNI-USD", "ATOM-USD", "XLM-USD", "ETC-USD", "FIL-USD", "HBAR-USD",
]
CURRENCY_CANDIDATES = [
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X", "CNY=X",
    "HKD=X", "SGD=X", "INR=X", "SEK=X", "NOK=X", "MXN=X", "BRL=X", "ZAR=X", "TRY=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "USDMXN=X", "USDTRY=X",
]
COMMODITY_CANDIDATES = [
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "PA=F", "ZC=F", "ZS=F",
    "ZW=F", "KE=F", "CT=F", "CC=F", "KC=F", "SB=F", "OJ=F", "LE=F", "HE=F", "LBS=F",
]


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _call_with_timeout(function, timeout_seconds: int):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(function)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        # The worker thread cannot be killed, but cancelling keeps the pool
        # from queueing more work; the leaked socket closes when the HTTP
        # request's own timeout fires. Log it so throttled symbols are visible
        # instead of silently disappearing from the scan.
        future.cancel()
        logger.warning("Scan call timed out after %ss and was abandoned.", timeout_seconds)
        return None
    except Exception as error:
        logger.warning("Scan call failed: %s", error)
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _class_label(asset_class: str) -> str:
    return {"stock": "Stocks", "crypto": "Cryptos", "currency": "Currencies", "option": "Options", "commodity": "Commodities"}[asset_class]


def _historical_signal_stats(indicator_df: pd.DataFrame, technical_signal) -> dict:
    """Score each completed bar and evaluate only the next five bars."""
    outcomes = {"BUY": [], "SELL": []}
    timeframes = ("short", "medium", "long")
    for index in range(max(50, len(indicator_df) - 260), len(indicator_df) - 5):
        row = indicator_df.iloc[index].to_dict()
        row["_price"] = float(indicator_df["close"].iloc[index])
        prices = indicator_df["close"].iloc[index + 1:index + 6]
        for timeframe in timeframes:
            score = technical_signal(row, timeframe)["score"]
            action = "BUY" if score >= 0.60 else "SELL" if score <= 0.40 else "HOLD"
            if action != "HOLD":
                outcomes[action].append(bool(prices.iloc[-1] > row["_price"] if action == "BUY" else prices.iloc[-1] < row["_price"]))
    completed = outcomes["BUY"] + outcomes["SELL"]
    return {
        "Historical signals": len(completed),
        "Historical success %": round(sum(completed) / len(completed) * 100, 1) if completed else 0.0,
    }


def _rank_row(symbol: str, asset_class: str, quote: dict, history: pd.DataFrame, indicators, technical_signal) -> dict | None:
    if history.empty:
        return None
    # A NaN quote passes `not quote.get("price")`, so it used to reach
    # `round(price, 6)` and raise TypeError, which the caller swallowed and the
    # symbol silently disappeared from the scan.
    price = _safe_float(quote.get("price"))
    if price is None or not np.isfinite(price) or price <= 0:
        logger.warning("Skipping %s: no usable price (raw=%r).", symbol, quote.get("price"))
        return None
    indicator_df = indicators.compute_indicators(history)
    summary = indicators.last_indicators(indicator_df)
    if not summary:
        logger.warning("Skipping %s: no indicators were produced.", symbol)
        return None
    summary["_price"] = price
    technical = technical_signal(summary, "medium")
    fundamental = {"score": 0.5, "data_quality": 0.0}
    if asset_class == "stock":
        fundamental = analyze_fundamentals(SimpleNamespace(asset_class="stock"), quote)
    timeframe_signals = {
        timeframe: technical_signal({**summary, "_price": price}, timeframe)["score"]
        for timeframe in ("short", "medium", "long")
    }
    stats = _historical_signal_stats(indicator_df, technical_signal)
    atr_value = _safe_float(summary.get("atr"))
    volatility = _safe_float(summary.get("volatility"))
    var_value = _safe_float(summary.get("var"))
    risk_parts = [abs(value) for value in (volatility, var_value) if value is not None]
    if atr_value is not None and price:
        risk_parts.append(abs(atr_value / price))
    risk = min(100.0, max(0.0, (sum(risk_parts) / len(risk_parts) * 100) if risk_parts else 50.0))
    confidence = abs(technical["score"] - 0.5) * 2
    opportunity = max(0.0, min(100.0, confidence * 100))
    rank_score = opportunity * 0.7 + (100 - risk) * 0.3
    action = "BUY" if technical["score"] >= 0.60 else "SELL" if technical["score"] <= 0.40 else "HOLD"
    reason_text = "; ".join(technical["reasons"][:3]) if technical["reasons"] else "Mixed technical confluence"
    return {
        "Symbol": symbol,
        "Asset": _class_label(asset_class),
        "Action": action,
        "Action basis": "Technical-only scan",
        "Technical score": round(technical["score"] * 100, 1),
        "Short signal": "BUY" if timeframe_signals["short"] >= 0.60 else "SELL" if timeframe_signals["short"] <= 0.40 else "HOLD",
        "Medium signal": "BUY" if timeframe_signals["medium"] >= 0.60 else "SELL" if timeframe_signals["medium"] <= 0.40 else "HOLD",
        "Long signal": "BUY" if timeframe_signals["long"] >= 0.60 else "SELL" if timeframe_signals["long"] <= 0.40 else "HOLD",
        "Price": round(price, 6),
        "Signal confidence": round(opportunity, 1),
        "Risk": round(risk, 1),
        "ATR %": round((abs(atr_value / price) * 100) if atr_value and price else 0, 2),
        "Volatility %": round((abs(volatility) * 100) if volatility is not None else 0, 2),
        "VaR %": round((abs(var_value) * 100) if var_value is not None else 0, 2),
        "Rank score": round(rank_score, 1),
        "RSI": round(_safe_float(summary.get("rsi")) or 0, 1),
        "Smart VWAP": round(_safe_float(summary.get("smart_vwap")) or 0, 6),
        "Pineify Bias": round(_safe_float(summary.get("pineify_bias")) or 0, 6),
        **stats,
        "Data source": quote.get("source", "Yahoo Finance"),
        "Fundamental score": round(float(fundamental.get("score", 0.5)) * 100, 1),
        "Fundamental data quality": round(float(fundamental.get("data_quality", 0.0)) * 100, 1),
        "Reason": f"Score {technical['score']:.2f}: {reason_text}",
    }


def _scan_symbol(symbol: str, asset_class: str, provider, indicators, technical_signal) -> dict | None:
    try:
        quote = {}
        if asset_class == "crypto":
            from data.binance_data import get_history as get_binance_history, get_quote as get_binance_quote
            try:
                quote = get_binance_quote(symbol)
                history = get_binance_history(symbol, interval="1h", years=1)
            except Exception:
                history = provider.get_history(symbol, period="6mo", interval="1d")
                latest_price = _safe_float(history["close"].iloc[-1]) if not history.empty else None
                quote = {"price": latest_price, "prev_close": _safe_float(history["close"].iloc[-2]) if len(history) > 1 else None, "source": "Yahoo Finance fallback"}
        else:
            history = provider.get_history(symbol, period="6mo", interval="1d")
            quote = provider.get_quote(symbol)
            if not quote.get("price") and not history.empty:
                quote["price"] = _safe_float(history["close"].iloc[-1])
            quote["source"] = quote.get("source", "Yahoo Finance")
        if history.empty:
            logger.warning("Skipping %s: provider returned no history.", symbol)
            return None
        return _rank_row(symbol, asset_class, quote, history, indicators, technical_signal)
    except Exception:
        logger.exception("Scan failed for %s (%s); symbol excluded from results.", symbol, asset_class)
        return None


def _scan_symbols(symbols: Iterable[str], asset_class: str, provider, indicators, technical_signal, progress: Callable | None = None) -> List[dict]:
    """Scan symbols serially.

    yfinance shares a session internally and returns empty frames when several
    history requests run concurrently, so this stays single-threaded; the
    previous `ThreadPoolExecutor(max_workers=1)` wrapper only added overhead.
    """
    rows = []
    symbols = list(symbols)
    for index, symbol in enumerate(symbols, start=1):
        row = _scan_symbol(symbol, asset_class, provider, indicators, technical_signal)
        if row:
            rows.append(row)
        if progress:
            progress(index / len(symbols))
    logger.info("Scanned %d %s symbols, %d usable.", len(symbols), asset_class, len(rows))
    return rows


def _scan_option_underlying(symbol: str, provider, indicators, technical_signal) -> List[dict]:
    rows = []
    try:
        history = provider.get_history(symbol, period="6mo", interval="1d")
        if history.empty:
            return rows
        quote = {"price": _safe_float(history["close"].iloc[-1])}
        base = _rank_row(symbol, "stock", quote, history, indicators, technical_signal)
        if not base:
            return rows
        expiries = _call_with_timeout(lambda: provider.get_option_expiries(symbol), 8) or []
        if not expiries:
            return rows
        chain = _call_with_timeout(lambda: provider.get_option_chain(symbol, expiries[0]), 12)
        if not chain:
            return rows
        underlying_price = float(quote["price"])
        for option_type, option_df in (("Call", chain.get("calls")), ("Put", chain.get("puts"))):
            if option_df is None or option_df.empty or "strike" not in option_df:
                continue
            option_df = option_df.copy()
            option_df["distance"] = (option_df["strike"] - underlying_price).abs()
            option_df = option_df.sort_values("distance").head(2)
            for _, contract in option_df.iterrows():
                last_price = _safe_float(contract.get("lastPrice")) or _safe_float(contract.get("ask")) or _safe_float(contract.get("bid"))
                if last_price is None:
                    continue
                open_interest = _safe_float(contract.get("openInterest")) or 0
                volume = _safe_float(contract.get("volume")) or 0
                bid = _safe_float(contract.get("bid")) or 0
                ask = _safe_float(contract.get("ask")) or 0
                spread_pct = ((ask - bid) / last_price * 100) if ask > bid and last_price else 0.0
                implied_volatility = _safe_float(contract.get("impliedVolatility")) or 0
                expiry_date = pd.to_datetime(chain.get("expiry", expiries[0]), errors="coerce")
                days_to_expiry = max(0, int((expiry_date - pd.Timestamp.now()).days)) if not pd.isna(expiry_date) else 0
                liquidity_penalty = min(35.0, max(0.0, 20.0 - min(20.0, (open_interest + volume) / 100)))
                liquidity_penalty += min(25.0, spread_pct)
                direction = base["Signal confidence"] if base["Action"] == "BUY" else 100 - base["Signal confidence"]
                risk = min(100.0, base["Risk"] + liquidity_penalty + 10)
                rows.append({
                    "Symbol": f"{symbol} {chain.get('expiry', expiries[0])} {option_type} {contract['strike']}",
                    "Asset": "Options",
                    "Action": base["Action"] if option_type == "Call" else ("SELL" if base["Action"] == "BUY" else "BUY" if base["Action"] == "SELL" else "HOLD"),
                    "Price": round(last_price, 4), "Signal confidence": round(direction, 1),
                    "Risk": round(risk, 1), "Rank score": round(direction * 0.65 + (100 - risk) * 0.35, 1),
                    "ATR %": base["ATR %"], "Volatility %": base["Volatility %"], "VaR %": base["VaR %"],
                    "Spread %": round(spread_pct, 2), "Implied Volatility %": round(implied_volatility * 100, 2),
                    "Open Interest": int(open_interest), "Volume": int(volume), "Days to Expiry": days_to_expiry,
                    "RSI": base["RSI"], "Smart VWAP": base["Smart VWAP"], "Pineify Bias": base["Pineify Bias"],
                    "Reason": f"Underlying {symbol}: {base['Reason']}; open interest {int(open_interest):,}.",
                })
    except Exception:
        logger.exception("Option scan failed for underlying %s.", symbol)
        return []
    return rows


def _scan_options(provider, indicators, technical_signal, underlyings: Iterable[str], progress: Callable | None = None) -> List[dict]:
    rows = []
    underlyings = list(underlyings)
    completed = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(_scan_option_underlying, symbol, provider, indicators, technical_signal) for symbol in underlyings]
        for future in as_completed(futures):
            completed += 1
            rows.extend(future.result())
            if progress:
                progress(completed / len(underlyings))
    return rows


def scan_market(provider, indicators, technical_signal, progress: Callable | None = None, group_callback: Callable | None = None) -> Dict[str, pd.DataFrame]:
    """Scan available candidates and return five ranked top-20 tables."""
    groups = [
        ("Stocks", lambda: _scan_symbols(STOCK_CANDIDATES[:20], "stock", provider, indicators, technical_signal, progress)),
        ("Cryptos", lambda: _scan_symbols(CRYPTO_CANDIDATES[:20], "crypto", provider, indicators, technical_signal, progress)),
        ("Currencies", lambda: _scan_symbols(CURRENCY_CANDIDATES[:20], "currency", provider, indicators, technical_signal, progress)),
        ("Commodities", lambda: _scan_symbols(COMMODITY_CANDIDATES[:20], "commodity", provider, indicators, technical_signal, progress)),
        ("Options", lambda: _scan_options(provider, indicators, technical_signal, STOCK_CANDIDATES[:5], progress)),
    ]
    results = {}
    for name, scan_group in groups:
        try:
            rows = scan_group()
        except Exception:
            rows = []
        table = pd.DataFrame(rows).sort_values("Rank score", ascending=False).head(20) if rows else pd.DataFrame()
        results[name] = table
        if group_callback:
            group_callback(name, table)
    return results


def scanned_at() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
