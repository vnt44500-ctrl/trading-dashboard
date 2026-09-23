"""Backtesting module for the composite signal strategy.

Simulates a long-only strategy (with optional short for SELL signals)
on historical OHLCV data using the technical signal logic.
"""
import numpy as np
import pandas as pd

from analysis import indicators as ind
from signals.engine import technical_signal
from config import config


def backtest(
    df: pd.DataFrame,
    allow_short: bool = False,
    commission_rate: float | None = None,
    slippage_rate: float | None = None,
    risk_per_trade: float | None = None,
    signal_fn=None,
    buy_threshold: float = 0.62,
    sell_threshold: float = 0.38,
    signal_scores=None,
) -> dict:
    """Run a next-bar, risk-sized backtest with costs and ATR exits."""
    if df.empty or len(df) < 60:
        return {"error": "Insufficient data for backtest."}

    data = ind.compute_indicators(df)
    commission_rate = config.commission_rate if commission_rate is None else commission_rate
    slippage_rate = config.slippage_rate if slippage_rate is None else slippage_rate
    risk_per_trade = config.risk_per_trade if risk_per_trade is None else risk_per_trade
    equity = []
    cash = config.account_equity
    position = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    trades = []
    pending_action = None
    initial = cash

    def close_position(fill_price: float):
        nonlocal cash, position, entry_price, stop_price, target_price
        if position == 0:
            return
        notional = abs(position * fill_price)
        exit_fill = fill_price * (1 - slippage_rate if position > 0 else 1 + slippage_rate)
        cash += position * exit_fill
        cash -= notional * commission_rate
        pnl = (exit_fill - entry_price) * position
        trades.append(pnl)
        position = 0.0
        entry_price = stop_price = target_price = 0.0

    def open_position(action: str, fill_price: float, stop_distance: float):
        nonlocal cash, position, entry_price, stop_price, target_price
        equity_now = cash + position * fill_price
        quantity = min(
            equity_now * risk_per_trade / stop_distance,
            equity_now * config.max_position_fraction / max(fill_price, 1e-9),
        )
        if quantity <= 0:
            return
        if action == "BUY":
            position = quantity
            entry_price = fill_price * (1 + slippage_rate)
            stop_price = entry_price - stop_distance
            target_price = entry_price + stop_distance * config.reward_risk_multiple
            cash -= quantity * entry_price + abs(quantity * entry_price) * commission_rate
        elif action == "SELL" and allow_short:
            position = -quantity
            entry_price = fill_price * (1 - slippage_rate)
            stop_price = entry_price + stop_distance
            target_price = entry_price - stop_distance * config.reward_risk_multiple
            cash += quantity * entry_price - abs(quantity * entry_price) * commission_rate

    for i in range(len(data)):
        row = data.iloc[i].to_dict()
        row = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        close = float(data["close"].iloc[i])
        open_price = float(data["open"].iloc[i])
        high = float(data["high"].iloc[i])
        low = float(data["low"].iloc[i])

        if position > 0 and (low <= stop_price or high >= target_price):
            close_position(stop_price if low <= stop_price else target_price)
        elif position < 0 and (high >= stop_price or low <= target_price):
            close_position(stop_price if high >= stop_price else target_price)

        if pending_action and i > 0:
            atr_value = float(row.get("atr") or close * 0.01)
            stop_distance = max(atr_value * config.atr_stop_multiple, open_price * 0.005)
            if (pending_action == "BUY" and position < 0) or (pending_action == "SELL" and position > 0):
                close_position(open_price)
            if position == 0:
                open_position(pending_action, open_price, stop_distance)
        pending_action = None

        if i >= 2:
            row["_price"] = close
            if signal_scores is not None:
                sig = {"score": float(signal_scores[i])}
            else:
                sig = signal_fn(row) if signal_fn else technical_signal(row, "short")
            pending_action = "BUY" if sig["score"] >= buy_threshold else ("SELL" if sig["score"] <= sell_threshold else None)

        equity.append(cash + position * close)

    if position != 0:
        close_position(float(data["close"].iloc[-1]))
        equity[-1] = cash

    data["equity"] = equity
    data["returns"] = data["equity"].pct_change().fillna(0)

    total_return = (equity[-1] - initial) / initial
    n = len(equity)
    annual_return = (1 + total_return) ** (252 / max(n, 1)) - 1
    daily_std = float(np.std(data["returns"]))
    sharpe = (data["returns"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    max_dd = _max_drawdown(equity)
    win_rate = (sum(1 for t in trades if t > 0) / max(len(trades), 1)) if trades else 0.0
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    expectancy = sum(trades) / len(trades) if trades else 0.0

    return {
        "equity_curve": data[["close", "equity"]],
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "num_trades": len(trades),
        "profit_factor": round(float(profit_factor), 3) if np.isfinite(profit_factor) else None,
        "expectancy": round(float(expectancy), 2),
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "risk_per_trade": risk_per_trade,
        "final_equity": round(equity[-1], 2),
    }


def _max_drawdown(equity) -> float:
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def walk_forward_backtest(
    df: pd.DataFrame,
    train_bars: int = 120,
    test_bars: int = 60,
    allow_short: bool = False,
    signal_fn=None,
) -> dict:
    """Evaluate fixed rules on sequential out-of-sample windows.

    The training window prevents very short histories from being treated as
    valid tests; strategy parameters are intentionally not fitted on the test
    window, which keeps the result honest and reproducible.
    """
    if df.empty or len(df) < train_bars + test_bars:
        return {"error": f"Need at least {train_bars + test_bars} bars for walk-forward validation."}
    folds = []
    start = train_bars
    while start + test_bars <= len(df):
        test = df.iloc[start:start + test_bars]
        result = backtest(test, allow_short=allow_short, signal_fn=signal_fn)
        if "error" not in result:
            folds.append({
                "fold": len(folds) + 1,
                "start": str(test.index[0]),
                "end": str(test.index[-1]),
                "return": result["total_return"],
                "sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
                "win_rate": result["win_rate"],
                "trades": result["num_trades"],
                "expectancy": result["expectancy"],
            })
        start += test_bars
    if not folds:
        return {"error": "No valid out-of-sample folds were produced."}
    fold_df = pd.DataFrame(folds)
    return {
        "folds": fold_df,
        "fold_count": len(folds),
        "profitable_fold_rate": float((fold_df["return"] > 0).mean()),
        "average_return": float(fold_df["return"].mean()),
        "average_sharpe": float(fold_df["sharpe"].mean()),
        "worst_drawdown": float(fold_df["max_drawdown"].max()),
    }