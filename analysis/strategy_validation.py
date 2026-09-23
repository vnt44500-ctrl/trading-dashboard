"""Historical validation and research metadata for dashboard strategies."""
from __future__ import annotations

import pandas as pd

from backtest.engine import backtest
from signals.engine import technical_signal


VALIDATION_THRESHOLD = 0.80
MINIMUM_TRADES = 10
TRAIN_FRACTION = 0.70


def _candidate_signal(timeframe: str, buy_threshold: float, sell_threshold: float):
    def signal(row):
        return technical_signal(row, timeframe)

    return signal


def optimize_strategy(df: pd.DataFrame, allow_short: bool = True) -> dict:
    """Search transparent rule variants, then score the winner on a holdout.

    Parameters are selected only on the first 70% of history. The final 30% is
    never used for selection and is the only result used for live eligibility.
    """
    if df.empty or len(df) < 160:
        return {"eligible": False, "error": "Need at least 160 bars to optimize and validate a strategy."}

    split = int(len(df) * TRAIN_FRACTION)
    train = df.iloc[:split]
    holdout = df.iloc[split:]
    candidates = [
        {"timeframe": timeframe, "buy_threshold": buy, "sell_threshold": sell}
        for timeframe in ("short", "medium", "long")
        for buy, sell in ((0.60, 0.40), (0.62, 0.38), (0.65, 0.35), (0.68, 0.32))
    ]
    rows = []
    for candidate in candidates:
        signal_fn = _candidate_signal(**candidate)
        result = backtest(
            train,
            allow_short=allow_short,
            signal_fn=signal_fn,
            buy_threshold=candidate["buy_threshold"],
            sell_threshold=candidate["sell_threshold"],
        )
        if "error" not in result and result["num_trades"] >= MINIMUM_TRADES:
            rows.append({**candidate, "train_win_rate": result["win_rate"], "train_return": result["total_return"], "train_trades": result["num_trades"]})
    if not rows:
        return {"eligible": False, "error": "No candidate produced enough training trades."}

    rows.sort(key=lambda row: (row["train_win_rate"], row["train_return"]), reverse=True)
    winner = rows[0]
    holdout_result = backtest(
        holdout,
        allow_short=allow_short,
        signal_fn=_candidate_signal(winner["timeframe"], winner["buy_threshold"], winner["sell_threshold"]),
        buy_threshold=winner["buy_threshold"],
        sell_threshold=winner["sell_threshold"],
    )
    if "error" in holdout_result:
        return {"eligible": False, "error": holdout_result["error"], "candidates": pd.DataFrame(rows)}
    trades = int(holdout_result["num_trades"])
    success_ratio = float(holdout_result["win_rate"])
    return {
        "eligible": success_ratio >= VALIDATION_THRESHOLD and trades >= MINIMUM_TRADES,
        "success_ratio": success_ratio,
        "threshold": VALIDATION_THRESHOLD,
        "minimum_trades": MINIMUM_TRADES,
        "trades": trades,
        "total_return": holdout_result["total_return"],
        "max_drawdown": holdout_result["max_drawdown"],
        "sharpe": holdout_result["sharpe"],
        "equity_curve": holdout_result["equity_curve"],
        "candidate": winner,
        "candidates": pd.DataFrame(rows),
        "train_bars": len(train),
        "holdout_bars": len(holdout),
    }


def validate_strategy(df: pd.DataFrame, allow_short: bool = True) -> dict:
    """Evaluate the default technical strategy without changing its rules."""
    result = backtest(df, allow_short=allow_short)
    if "error" in result:
        return {
            "eligible": False,
            "success_ratio": 0.0,
            "minimum_trades": MINIMUM_TRADES,
            "threshold": VALIDATION_THRESHOLD,
            "error": result["error"],
        }
    success_ratio = float(result["win_rate"])
    trades = int(result["num_trades"])
    return {
        "eligible": success_ratio >= VALIDATION_THRESHOLD and trades >= MINIMUM_TRADES,
        "success_ratio": success_ratio,
        "minimum_trades": MINIMUM_TRADES,
        "threshold": VALIDATION_THRESHOLD,
        "trades": trades,
        "total_return": result["total_return"],
        "max_drawdown": result["max_drawdown"],
        "sharpe": result["sharpe"],
        "equity_curve": result["equity_curve"],
    }


def strategy_catalog() -> list[dict]:
    return [
        {
            "name": "ML ensemble gate",
            "type": "Classification + regression",
            "method": "Combines logistic classification for direction, linear regression for expected return, and random-forest regression for nonlinear return patterns.",
            "steps": "1. Build lagged technical features. 2. Train only on the first 70% of five-year data. 3. Blend model scores into BUY, SELL, or HOLD. 4. Evaluate direction accuracy and profitable trades on the untouched final 30%.",
        },
        {
            "name": "AI-optimized composite",
            "type": "Strategy",
            "method": "Combines technical score, fundamental score, and news sentiment using timeframe-specific weights.",
            "steps": "1. Calculate indicators. 2. Score trend, momentum, volume, VWAP, cloud, and regime. 3. Blend fundamental and sentiment scores. 4. BUY >= 0.60, SELL <= 0.40, otherwise HOLD.",
        },
        {
            "name": "ATR risk sizing",
            "type": "Risk method",
            "method": "Sizes a position from account risk divided by ATR-based stop distance, capped by maximum portfolio fraction.",
            "steps": "1. Set stop at two ATRs or 0.5% minimum. 2. Risk 0.5% of account equity. 3. Cap notional exposure at 25%. 4. Apply commission and slippage.",
        },
        {
            "name": "Walk-forward validation",
            "type": "Validation method",
            "method": "Tests sequential out-of-sample windows after a training history, reducing look-ahead bias.",
            "steps": "1. Reserve the training window. 2. Test the next fixed window. 3. Move forward and repeat. 4. Review profitable-fold rate, return, drawdown, and trade count.",
        },
    ]


def indicator_catalog() -> list[dict]:
    return [
        {"name": "SMA / EMA", "method": "Moving averages estimate trend direction and crossover momentum.", "steps": "Compare fast and slow averages; positive spread supports BUY, negative spread supports SELL."},
        {"name": "RSI", "method": "Relative Strength Index measures recent momentum on a 0-100 scale.", "steps": "Below 30 contributes bullish reversal evidence; above 70 contributes bearish pullback evidence."},
        {"name": "MACD", "method": "Difference between fast and slow exponential averages with a signal line.", "steps": "MACD above its signal contributes bullish momentum; below contributes bearish momentum."},
        {"name": "Smart VWAP", "method": "Blends volume-weighted average price with the short EMA as a price anchor.", "steps": "Price above the anchor supports bullish confluence; below supports bearish confluence."},
        {"name": "Ichimoku cloud", "method": "Uses conversion, base, and shifted cloud levels to describe market structure.", "steps": "Price above the cloud supports bullish structure; below the cloud supports bearish structure."},
        {"name": "Supertrend", "method": "ATR-derived trailing bands classify the current direction without using future bars.", "steps": "Maintain causal upper and lower bands; BUY-side confluence requires bullish direction, while SELL-side confluence requires bearish direction."},
        {"name": "ATR / ADX / CMF", "method": "Measures volatility, trend strength, and buying or selling pressure.", "steps": "ATR sets stops, ADX contextualizes trend strength, and CMF confirms pressure when extreme."},
    ]