# Trading Signals Dashboard

A professional real-time market analysis app built with **Streamlit** that fetches
live data for **stocks, forex, commodities, and crypto**, performs **fundamental +
technical + news-sentiment** analysis, and generates actionable **BUY / SELL / HOLD**
signals — with backtesting and a pluggable broker adapter for future automated trading.

## Features

- 📈 **Real-time prices & candlestick charts** with multi-resolution history.
- 🔍 **Instrument search** across stocks, FX pairs, commodities, and crypto.
- 📐 **Comprehensive technical indicators**: SMA, EMA, VWAP, RSI, MACD,
  Bollinger Bands, Stochastic, OBV, ATR, ADX, Williams %R, support/resistance,
  Value-at-Risk (VaR), volatility.
- 🧠 **Fundamental analysis** (P/E, EPS, dividend yield, beta, market cap).
- 📰 **News sentiment** via free sources (yfinance news + optional NewsAPI).
- 🎯 **Composite signal engine** with confidence scoring and full rationale.
- 🛡️ **Risk-aware signals** with ATR stops, account-risk position sizing, model versions,
  regime context, and duplicate-signal protection.
- 🔄 **Realistic backtesting** with next-bar fills, commission, slippage, ATR exits,
  profit factor, expectancy, and walk-forward validation.
- 📊 **Persistent market scans** with risk breakdowns and option liquidity fields.
- ⏰ **Independent scheduler** for scans without an open browser session.
- 🤖 **Pluggable broker adapter** (paper trading now; IBKR/Alpaca/Binance ready).

## Project Structure

```
trading-dashboard/
├── app.py                 # Streamlit entry point
├── config.py              # Central configuration
├── requirements.txt
├── core/                  # Models + asset universe
├── data/                  # Market & news data providers
├── analysis/              # Indicators, fundamentals, sentiment
├── run_market_scheduler.py # Browser-independent scan scheduler
├── signals/               # Signal generation engine
├── backtest/              # Backtesting engine
├── broker/                # Broker adapter interface
└── ui/                    # Streamlit dashboard & charts
```

## Install & Run

```powershell
cd trading-dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser (usually http://localhost:8501).

## Optional API Keys (`.env`)

Create a `.env` file for extra features (not required for core functionality):

```
NEWS_API_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ACTIVE_BROKER=paper
```

## Scheduled scanning

Run one scan from PowerShell:

```powershell
.\.venv\Scripts\python.exe run_market_scheduler.py --once
```

Run daily at a local time while the scheduler process is open:

```powershell
.\.venv\Scripts\python.exe run_market_scheduler.py --time 16:00
```

For unattended operation, register the `--once` command with Windows Task Scheduler.
The dashboard reads the latest saved result from `market_scan_history.json`.

## Data coverage and swaps

Yahoo Finance is suitable for delayed research data but does not provide reliable
institutional swaps, complete global listings, or guaranteed real-time quotes. The
`data/provider_interfaces.py` contract is the integration boundary for a broker or
institutional source such as IBKR, Alpaca, Binance, Polygon, Refinitiv, or Bloomberg.
Swap scanning requires a provider with those instruments and credentials.

## Broker automation

To automate trading later, extend `BrokerAdapter` in `broker/adapter.py` for your
broker (e.g. Interactive Brokers, Alpaca, Binance) and set `ACTIVE_BROKER` accordingly.

> **Disclaimer**: This tool is for educational/analysis purposes only and is not
> financial advice. Always validate signals and use appropriate risk management.