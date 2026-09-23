"""Asset universe: symbols, names, classes for searchable instruments."""
import re

import requests
import yfinance as yf

COINGECKO_SYMBOL_IDS = {}

ASSET_UNIVERSE = {
    "AAPL": {"name": "Apple Inc.", "class": "stock", "exchange": "NASDAQ"},
    "MSFT": {"name": "Microsoft Corp.", "class": "stock", "exchange": "NASDAQ"},
    "GOOGL": {"name": "Alphabet Inc.", "class": "stock", "exchange": "NASDAQ"},
    "AMZN": {"name": "Amazon.com Inc.", "class": "stock", "exchange": "NASDAQ"},
    "META": {"name": "Meta Platforms", "class": "stock", "exchange": "NASDAQ"},
    "NVDA": {"name": "NVIDIA Corp.", "class": "stock", "exchange": "NASDAQ"},
    "TSLA": {"name": "Tesla Inc.", "class": "stock", "exchange": "NASDAQ"},
    "JPM": {"name": "JPMorgan Chase", "class": "stock", "exchange": "NYSE"},
    "V": {"name": "Visa Inc.", "class": "stock", "exchange": "NYSE"},
    "JNJ": {"name": "Johnson & Johnson", "class": "stock", "exchange": "NYSE"},
    "XOM": {"name": "Exxon Mobil", "class": "stock", "exchange": "NYSE"},
    "SPY": {"name": "SPDR S&P 500 ETF", "class": "stock", "exchange": "NYSE"},
    "QQQ": {"name": "Invesco QQQ ETF", "class": "stock", "exchange": "NASDAQ"},
    "EURUSD=X": {"name": "EUR/USD", "class": "forex", "exchange": "FX"},
    "GBPUSD=X": {"name": "GBP/USD", "class": "forex", "exchange": "FX"},
    "USDJPY=X": {"name": "USD/JPY", "class": "forex", "exchange": "FX"},
    "USDINR=X": {"name": "USD/INR", "class": "forex", "exchange": "FX"},
    "AUDUSD=X": {"name": "AUD/USD", "class": "forex", "exchange": "FX"},
    "USDCAD=X": {"name": "USD/CAD", "class": "forex", "exchange": "FX"},
    "GC=F": {"name": "Gold Futures", "class": "commodity", "exchange": "COMEX"},
    "SI=F": {"name": "Silver Futures", "class": "commodity", "exchange": "COMEX"},
    "CL=F": {"name": "Crude Oil WTI", "class": "commodity", "exchange": "NYMEX"},
    "BZ=F": {"name": "Brent Crude", "class": "commodity", "exchange": "ICE"},
    "NG=F": {"name": "Natural Gas", "class": "commodity", "exchange": "NYMEX"},
    "HG=F": {"name": "Copper Futures", "class": "commodity", "exchange": "COMEX"},
    "ZC=F": {"name": "Corn Futures", "class": "commodity", "exchange": "CBOT"},
    "BTC-USD": {"name": "Bitcoin", "class": "crypto", "exchange": "Crypto"},
    "ETH-USD": {"name": "Ethereum", "class": "crypto", "exchange": "Crypto"},
    "SOL-USD": {"name": "Solana", "class": "crypto", "exchange": "Crypto"},
    "BNB-USD": {"name": "BNB", "class": "crypto", "exchange": "Crypto"},
    "XRP-USD": {"name": "XRP", "class": "crypto", "exchange": "Crypto"},
    "ADA-USD": {"name": "Cardano", "class": "crypto", "exchange": "Crypto"},
    "DOGE-USD": {"name": "Dogecoin", "class": "crypto", "exchange": "Crypto"},
}


def _asset_class_from_symbol(symbol: str) -> str:
    s = symbol.upper()
    if re.search(r"\b(CALL|PUT|OPTION)\b", s):
        return "option"
    if s.endswith("=F") or s.endswith("F"):
        return "commodity"
    if "-USD" in s or "-USDT" in s or ("USD" in s and "=" not in s and "-" in s):
        return "crypto"
    if "=" in s:
        return "forex"
    return "stock"


def _search_yahoo(query: str, limit: int = 8):
    q = query.strip()
    if not q:
        return []

    try:
        results = yf.search(q, max_results=limit)
        matches = []
        for quote in results.get("quotes", []):
            symbol = quote.get("symbol")
            if not symbol:
                continue
            name = quote.get("shortname") or quote.get("longname") or symbol
            kind = (quote.get("quoteType") or "EQUITY").upper()
            if kind == "EQUITY":
                asset_class = "stock"
            elif kind == "CRYPTOCURRENCY":
                asset_class = "crypto"
            elif kind in {"CURRENCY", "FOREX"}:
                asset_class = "forex"
            elif kind == "FUTURE":
                asset_class = "commodity"
            else:
                asset_class = "stock"
            matches.append((symbol, {"name": name, "class": asset_class, "exchange": quote.get("exchange") or "Yahoo"}))
        return matches
    except Exception:
        return []


def _search_coingecko(query: str, limit: int = 8):
    """Use CoinGecko's public search endpoint as a crypto-only fallback."""
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": query.strip()},
            timeout=5,
        )
        response.raise_for_status()
        matches = []
        for coin in response.json().get("coins", [])[:limit]:
            symbol = coin.get("symbol", "").upper()
            if not symbol:
                continue
            ticker = f"{symbol}-USD"
            COINGECKO_SYMBOL_IDS[ticker] = coin.get("id")
            matches.append((
                ticker,
                {"name": coin.get("name") or symbol, "class": "crypto", "exchange": "CoinGecko", "coingecko_id": coin.get("id")},
            ))
        return matches
    except (requests.RequestException, ValueError, TypeError):
        return []
def _search_option_chain(query: str, limit: int = 4):
    q = query.strip().upper()
    option_match = re.search(r"([A-Z0-9\-\.]+)\s*(CALL|PUT|OPTION)", q)
    if not option_match:
        return []
    base_symbol = option_match.group(1)
    option_type = option_match.group(2)
    try:
        ticker = yf.Ticker(base_symbol)
        expiries = ticker.options or []
        if not expiries:
            return []
        results = []
        for expiry in expiries[: min(len(expiries), 3)]:
            chain = ticker.option_chain(expiry)
            option_df = chain.calls if option_type.upper() == "CALL" else chain.puts
            if option_df.empty:
                continue
            best = option_df.iloc[0]
            display = f"{base_symbol} {expiry} {option_type.upper()} {best.get('strike', 'ATM')}"
            results.append((display, {"name": f"{base_symbol} {option_type.upper()} {expiry}", "class": "option", "exchange": "Options"}))
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def search_assets(query: str, limit: int = 25):
    """Return local and remote suggestions once at least three characters exist."""
    q = query.strip().upper()
    if not q:
        return list(ASSET_UNIVERSE.items())[:limit]

    results = []
    for sym, meta in ASSET_UNIVERSE.items():
        if q in sym or q in meta["name"].upper():
            results.append((sym, meta))

    # Keep an exact symbol at the top so reruns cannot hide the user's query.
    results.sort(key=lambda item: (item[0].upper() != q, not item[0].upper().startswith(q), item[0]))

    # Avoid remote requests for one- or two-character input. This prevents
    # noisy suggestions and rate limits while the user is still typing.
    if len(q) < 3:
        return results[:limit]

    if len(results) < limit:
        yahoo_matches = _search_yahoo(q, limit=limit - len(results))
        for item in yahoo_matches:
            symbol, meta = item
            if symbol.upper() not in {s.upper() for s, _ in results}:
                results.append((symbol, meta))

    if len(results) < limit:
        crypto_matches = _search_coingecko(q, limit=limit - len(results))
        for item in crypto_matches:
            symbol, meta = item
            if symbol.upper() not in {s.upper() for s, _ in results}:
                results.append(item)

    if len(results) < limit:
        option_matches = _search_option_chain(q, limit=max(1, limit - len(results)))
        for item in option_matches:
            symbol, meta = item
            if symbol.upper() not in {s.upper() for s, _ in results}:
                results.append((symbol, meta))

    if q and not any(sym.upper() == q for sym, _ in results):
        cls = _asset_class_from_symbol(q)
        results.append((q, {"name": f"{q} (Custom)", "class": cls, "exchange": "Unknown"}))

    return results[:limit]


def get_asset(symbol: str):
    """Return an Asset model for a symbol if it exists, else construct a generic one."""
    from core.models import Asset

    meta = ASSET_UNIVERSE.get(symbol.upper())
    if meta:
        return Asset(
            symbol=symbol.upper(),
            name=meta["name"],
            asset_class=meta["class"],
            exchange=meta.get("exchange"),
        )
    return Asset(symbol=symbol.upper(), name=symbol.upper(), asset_class=_asset_class_from_symbol(symbol))