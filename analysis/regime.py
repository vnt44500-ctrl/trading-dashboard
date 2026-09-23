"""Market-regime detection used as a signal context filter."""
import numpy as np
import pandas as pd


def detect_regime(df: pd.DataFrame) -> dict:
    """Classify trend and volatility from historical OHLCV data."""
    if df.empty or len(df) < 30:
        return {"name": "unknown", "confidence": 0.0, "reason": "Insufficient history for regime detection."}
    close = df["close"].astype(float)
    returns = close.pct_change().dropna()
    short = close.rolling(20).mean().iloc[-1]
    long = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
    trend_gap = (short - long) / max(abs(long), 1e-9)
    volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    rolling_mean = returns.rolling(20).mean().abs().iloc[-1]
    rolling_std = returns.rolling(20).std().replace(0, np.nan).iloc[-1]
    directional_strength = rolling_mean / rolling_std if pd.notna(rolling_std) else 0.0
    directional_strength = float(directional_strength) if pd.notna(directional_strength) else 0.0
    volatility = float(volatility) if pd.notna(volatility) else 0.0
    if volatility >= 0.45:
        volatility_name = "high-volatility"
    elif volatility <= 0.15:
        volatility_name = "low-volatility"
    else:
        volatility_name = "normal-volatility"
    if trend_gap >= 0.02 and directional_strength >= 0.15:
        trend_name = "bull"
    elif trend_gap <= -0.02 and directional_strength >= 0.15:
        trend_name = "bear"
    else:
        trend_name = "sideways"
    name = f"{trend_name}/{volatility_name}"
    confidence = min(1.0, abs(trend_gap) * 8 + min(directional_strength, 1.0) * 0.35)
    return {
        "name": name,
        "trend": trend_name,
        "volatility_state": volatility_name,
        "confidence": round(float(confidence), 3),
        "trend_gap": round(float(trend_gap), 5),
        "annualized_volatility": round(volatility, 4),
        "directional_strength": round(directional_strength, 4),
        "reason": f"20/50 moving-average gap {trend_gap:.2%}; directional strength {directional_strength:.2f}; annualized volatility {volatility:.2%}.",
    }
