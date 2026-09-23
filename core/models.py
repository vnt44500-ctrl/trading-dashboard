"""Model classes shared across the application."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Asset:
    """A tradeable asset."""
    symbol: str
    name: str
    asset_class: str  # stock | forex | commodity | crypto | option
    exchange: Optional[str] = None


@dataclass
class Signal:
    """A generated buy/sell/hold recommendation."""
    asset: Asset
    action: str  # BUY | SELL | HOLD
    confidence: float  # 0..1
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    timeframe: str = "medium"             # short | medium | long
    expected_duration: str = ""           # human readable duration
    reasons: List[str] = field(default_factory=list)
    strategy_name: str = ""
    strategy_summary: str = ""
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    # Internal fields for self-learning
    id: str = field(default_factory=lambda: "")
    resolved: bool = False
    success: bool = False
    realized_return: float = 0.0
    model_version: str = "confluence-v2"
    risk_score: float = 0.0
    position_size: float = 0.0
    raw_action: str = "HOLD"
    live_eligible: bool = False