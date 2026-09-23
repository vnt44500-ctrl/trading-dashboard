"""Full-market symbol books for the multi-asset alert scanner.

Organised by asset class. The stock book is the live S&P 500 constituent list
(fetched at runtime from a maintained dataset, with a static liquid-name
fallback for offline runs); the other classes are curated liquid instruments
that all trade on Yahoo Finance with daily OHLCV, so the whole pipeline
(indicators -> ML evidence -> operating threshold -> regime gate -> alert)
runs unchanged across classes.

Options are intentionally not a separate universe: the validated signal fires
on the UNDERLYING (stock/ETF), and the alert points at the matching option
strategy on that underlying. Options themselves have no long daily price
history to validate a signal model on - pretending otherwise would break the
framework's honesty standard.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")

# Static fallback (offline / dataset unavailable): ~110 most liquid S&P names.
STOCKS_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "LLY",
    "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "ABBV", "MRK",
    "PEP", "KO", "CVX", "ADBE", "WMT", "BAC", "CRM", "MCD", "CSCO", "TMO",
    "ACN", "ABT", "LIN", "NFLX", "AMD", "DIS", "QCOM", "TXN", "AMGN", "CAT",
    "VZ", "IBM", "NEE", "PM", "UNP", "ORCL", "HON", "RTX", "LOW", "SPGI",
    "INTU", "AMAT", "GS", "BA", "DE", "ISRG", "SCHW", "PLD", "BKNG", "MMM",
    "T", "COP", "SLB", "F", "GM", "BLK", "MS", "GE", "LMT", "SYK",
    "ADP", "GILD", "PGR", "MDLZ", "REGN", "ADI", "LRCX", "PANW", "SNPS", "CDNS",
    "MU", "MRVL", "KLAC", "ABNB", "SBUX", "NKE", "TJX", "CMCSA", "TFC", "USB",
    "PNC", "COF", "AXP", "MET", "AIG", "TRV", "ALL", "EMR", "GD", "NOC",
    "HES", "PSX", "KMI", "OKE", "WMB", "DUK", "SO", "AEP", "EXC", "XEL",
]

ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EFA", "EEM", "VWO", "TLT",
    "IEF", "SHY", "LQD", "HYG", "GLD", "SLV", "USO", "UNG", "XLE", "XLF",
    "XLI", "XLV", "XLK", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH",
    "VNQ",
]

# Futures (direct exposure) for the commodity class.
COMMODITY_UNIVERSE = [
    "GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "CL=F", "BZ=F", "NG=F",
    "RB=F", "HO=F", "ZW=F", "ZC=F", "ZS=F", "KC=F", "SB=F", "CC=F", "CT=F",
]

FOREX_UNIVERSE = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "DX-Y.NYB",
]

CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD",
    "ATOM-USD", "ETC-USD", "XLM-USD",
]

_CLASS_ALIASES = {"stock": "stock", "stocks": "stock", "equity": "stock",
                  "etf": "etf", "commodity": "commodity", "commodities": "commodity",
                  "forex": "forex", "currency": "forex", "currencies": "forex",
                  "crypto": "crypto"}

_STATIC_UNIVERSES = {"etf": ETF_UNIVERSE, "commodity": COMMODITY_UNIVERSE,
                     "forex": FOREX_UNIVERSE, "crypto": CRYPTO_UNIVERSE}
