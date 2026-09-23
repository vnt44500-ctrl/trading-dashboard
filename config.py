"""Central configuration for the trading dashboard."""
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Data source toggles
    use_news: bool = True
    news_cache_minutes: int = 15

    # Rate limiting / retry
    max_retries: int = 3
    retry_backoff: float = 1.0

    # Indicator defaults
    short_ma: int = 20
    long_ma: int = 50
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    var_confidence: float = 0.95

    # Portfolio and execution assumptions used by signals/backtests
    account_equity: float = 10000.0
    risk_per_trade: float = 0.005
    max_position_fraction: float = 0.25
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005
    # Estimated cost of one round trip (commissions + slippage), as a percent
    # of trade value. Used by the evidence engine's operating-point selection so
    # a threshold whose edge cannot clear its own costs is never selected.
    signal_round_trip_cost_pct: float = 0.2
    atr_stop_multiple: float = 2.0
    reward_risk_multiple: float = 2.0
    signal_model_version: str = "confluence-v2"
    signal_dedup_hours: int = 24

    # Live-timeframe policy, set by the multi-window validation (Sept 2026):
    # medium is the lead (its edge replicated in three independent held-out
    # windows), long is secondary (positive in every window, largest samples),
    # short stays research-only (negative net expectancy in the crisis window).
    # Research-only timeframes are still computed and displayed, but their
    # live actions are forced to HOLD with an explanatory reason.
    lead_timeframe: str = "medium"
    live_timeframes: tuple = ("medium", "long")
    research_timeframes: tuple = ("short",)

    # API keys (optional, loaded from .env)
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", ""))
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))

    # Broker adapter (pluggable)
    active_broker: str = os.getenv("ACTIVE_BROKER", "paper")


config = Config()