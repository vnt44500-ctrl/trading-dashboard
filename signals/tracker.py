"""Self-learning tracker module.

Records signals, evaluates them against new prices, and dynamically adjusts
the weights (technical vs fundamental vs sentiment) to maximize profitability.
"""
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

from core.models import Signal
from config import config

TRACKER_FILE = "signal_tracker.json"

# Base default weights customized per timeframe
DEFAULT_WEIGHTS = {
    "short": {"technical": 0.70, "fundamental": 0.05, "sentiment": 0.25},
    "medium": {"technical": 0.50, "fundamental": 0.30, "sentiment": 0.20},
    "long": {"technical": 0.20, "fundamental": 0.70, "sentiment": 0.10},
}


class SignalTracker:
    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(TRACKER_FILE):
            try:
                with open(TRACKER_FILE, "r") as f:
                    data = json.load(f)
                    if "weights" not in data:
                        data["weights"] = DEFAULT_WEIGHTS.copy()
                    return data
            except Exception:
                pass
        return {"signals": [], "weights": DEFAULT_WEIGHTS.copy(), "performance": {}}
    def _save(self):
        """Atomically persist tracker state.

        A plain truncating write can interleave between concurrent sessions and
        leave a corrupt JSON file, which then silently falls back to defaults.
        """
        temporary = f"{TRACKER_FILE}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
        os.replace(temporary, TRACKER_FILE)

    def record_signal(self, signal: Signal):
        """Save a generated signal so the AI can evaluate it later."""
        # Only record if we actually took a directional stance
        if signal.action == "HOLD":
            return

        cutoff = datetime.now() - timedelta(hours=config.signal_dedup_hours)
        for existing in self.data["signals"]:
            if (
                existing.get("symbol") == signal.asset.symbol
                and existing.get("timeframe") == signal.timeframe
                and existing.get("action") == signal.action
                and not existing.get("resolved")
                and datetime.fromisoformat(existing["timestamp"]) >= cutoff
            ):
                signal.id = existing["id"]
                return
            
        sig_id = str(uuid.uuid4())
        signal.id = sig_id
        
        self.data["signals"].append({
            "id": sig_id,
            "symbol": signal.asset.symbol,
            "timeframe": signal.timeframe,
            "action": signal.action,
            "entry_price": signal.price,
            "confidence": signal.confidence,
            "tech_score": signal.technical_score,
            "fund_score": signal.fundamental_score,
            "sent_score": signal.sentiment_score,
            "timestamp": signal.timestamp.isoformat(),
            "resolved": False,
            "model_version": signal.model_version,
            "risk_score": signal.risk_score,
            "position_size": signal.position_size,
        })
        self._save()

    def get_weights(self, timeframe: str) -> Dict[str, float]:
        """Return the running optimized weights for a given timeframe."""
        return self.data["weights"].get(timeframe, DEFAULT_WEIGHTS[timeframe]).copy()

    def evaluate_and_learn(self, symbol: str, current_price: float):
        """Check past unresolved signals for this symbol and adjust weights if resolved."""
        now = datetime.now()
        updated = False
        
        # Time horizons required to evaluate a signal
        horizons = {
            "short": 3,     # evaluate after 3 days
            "medium": 21,   # evaluate after ~3 weeks
            "long": 90      # evaluate after ~3 months
        }

        for sig in self.data["signals"]:
            if sig["symbol"] == symbol and not sig.get("resolved"):
                sig_date = datetime.fromisoformat(sig["timestamp"])
                days_since = (now - sig_date).days
                horizon = horizons.get(sig["timeframe"], 21)
                
                # If enough time has passed to judge the signal
                if days_since >= horizon:
                    entry = sig["entry_price"]
                    action = sig["action"]
                    pct_change = (current_price - entry) / entry
                    
                    if action == "BUY":
                        success = pct_change > 0.01  # >1% profit
                        realized = pct_change
                    else: # SELL
                        success = pct_change < -0.01 # >1% drop
                        realized = -pct_change
                        
                    sig["resolved"] = True
                    sig["success"] = success
                    sig["realized_return"] = realized
                    sig["market_direction"] = 1 if pct_change > 0 else -1

                    self._update_weights(sig["timeframe"], sig, market_direction=sig["market_direction"])
                    updated = True

        if updated:
            self._save()

    WEIGHT_FLOOR = 0.05
    WEIGHT_CEILING = 0.80
    LEARNING_RATE = 0.02

    def _update_weights(self, timeframe: str, sig: dict, market_direction: int):
        """Move each component's weight toward whichever side the market took.

        The previous implementation compared the component's bullishness with the
        *action* and then increased the weight in both the "component agreed and
        the trade won" and the "component disagreed and the trade lost" cases.
        Because sentiment is almost always neutral (0.5, therefore "not bullish")
        and most trades lost, that inverted branch increased sentiment's weight
        on nearly every loss until it held 94% of the short-horizon weight.
        """
        weights = self.data["weights"].setdefault(timeframe, dict(DEFAULT_WEIGHTS[timeframe]))

        stances = {
            "technical": self._stance(sig.get("tech_score")),
            "fundamental": self._stance(sig.get("fund_score")),
            "sentiment": self._stance(sig.get("sent_score")),
        }

        for component, stance in stances.items():
            if stance == 0:
                continue  # the component abstained; it has no claim to reward or blame
            aligned = stance == market_direction
            weights[component] = float(
                min(self.WEIGHT_CEILING, max(self.WEIGHT_FLOOR, weights[component] + (self.LEARNING_RATE if aligned else -self.LEARNING_RATE)))
            )

        total = sum(weights.values())
        if total > 0:
            for component in weights:
                weights[component] = round(weights[component] / total, 6)
        self.data["weights"][timeframe] = weights

    @staticmethod
    def _stance(score) -> int:
        """Classify a component score as bullish (+1), bearish (-1) or abstaining (0)."""
        if score is None:
            return 0
        if score > 0.55:
            return 1
        if score < 0.45:
            return -1
        return 0

    def reset_weights(self):
        """Restore the documented default weights for every timeframe."""
        self.data["weights"] = {timeframe: dict(values) for timeframe, values in DEFAULT_WEIGHTS.items()}
        self._save()

    def prune(self, keep_resolved: int = 2000):
        """Cap the on-disk signal log so the state file cannot grow without bound."""
        signals = self.data.get("signals", [])
        if len(signals) <= keep_resolved:
            return
        unresolved = [s for s in signals if not s.get("resolved")]
        resolved = [s for s in signals if s.get("resolved")]
        self.data["signals"] = unresolved + resolved[-keep_resolved:]
        self._save()

    def get_stats(self) -> dict:
        """Return analytics on AI performance."""
        signals = self.data.get("signals", [])
        resolved = [s for s in signals if s.get("resolved")]
        if not resolved:
            return {"tot": len(signals), "res": 0, "win_rate": 0.0, "avg_return": 0.0}
            
        wins = sum(1 for s in resolved if s.get("success"))
        returns = [s.get("realized_return", 0) for s in resolved]
        avg_ret = sum(returns) / len(returns)
        
        return {
            "tot": len(signals),
            "res": len(resolved),
            "win_rate": wins / len(resolved),
            "avg_return": avg_ret
        }

tracker = SignalTracker()