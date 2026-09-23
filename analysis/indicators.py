"""Technical indicators engine.

Computes a comprehensive set of indicators:
- Moving averages (SMA, EMA, VWAP)
- RSI, MACD, Bollinger Bands, Stochastic
- OBV, ATR, ADX, Ichimoku (partial), Williams %R
- Support/Resistance levels
- Value-at-Risk (VaR) and volatility
- Open Interest (when available via info)
"""
import numpy as np
import pandas as pd
from ta.trend import PSARIndicator
from ta.volatility import KeltnerChannel

from config import config


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = sma(series, period)
    sd = series.rolling(window=period).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return lower, mid, upper


def stochastic(high, low, close, k_period: int = 14, d_period: int = 3):
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high, low, close, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_val = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di


def williams_r(high, low, close, period: int = 14) -> pd.Series:
    high_max = high.rolling(window=period).max()
    low_min = low.rolling(window=period).min()
    return -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)


def vwap(high, low, close, volume) -> pd.Series:
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def chaikin_money_flow(high, low, close, volume, period: int = 20) -> pd.Series:
    """CMF: Measures buying/selling pressure via money flow multiplier."""
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_vol = mf_mult * volume
    return mf_vol.rolling(window=period).sum() / volume.rolling(window=period).sum().replace(0, np.nan)

def supertrend_direction(high, low, close, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Calculate causal Supertrend direction: 1 bullish, -1 bearish."""
    atr_value = atr(high, low, close, period)
    midpoint = (high + low) / 2
    upper = midpoint + multiplier * atr_value
    lower = midpoint - multiplier * atr_value
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(0, index=close.index, dtype=int)
    for index in range(1, len(close)):
        previous = index - 1
        if close.iloc[previous] <= final_upper.iloc[previous]:
            final_upper.iloc[index] = min(upper.iloc[index], final_upper.iloc[previous])
        if close.iloc[previous] >= final_lower.iloc[previous]:
            final_lower.iloc[index] = max(lower.iloc[index], final_lower.iloc[previous])
        if direction.iloc[previous] >= 0:
            direction.iloc[index] = 1 if close.iloc[index] >= final_lower.iloc[index] else -1
        else:
            direction.iloc[index] = -1 if close.iloc[index] <= final_upper.iloc[index] else 1
    return direction

def ichimoku_cloud(high, low):
    """Ichimoku Kinko Hyo components."""
    tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2 
    kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    return tenkan, kijun, senkou_a, senkou_b

def volume_profile_poc(close: pd.Series, volume: pd.Series, window: int = 60, bins: int = 15) -> pd.Series:
    """Rolling Point of Control (Volume Profile).

    Returns a *series* rather than a single scalar. The previous version
    returned one value computed over the entire frame and broadcast it to every
    bar, so historical bars implicitly "knew" the current volume profile and the
    backtest was optimistically biased. Each value here uses only the trailing
    ``window`` bars.
    """
    prices = np.asarray(close, dtype=float)
    volumes = np.asarray(volume, dtype=float)
    output = np.full(len(prices), np.nan)
    for index in range(len(prices)):
        start = max(0, index - window + 1)
        window_prices = prices[start:index + 1]
        window_volumes = volumes[start:index + 1]
        usable = np.isfinite(window_prices) & np.isfinite(window_volumes)
        window_prices, window_volumes = window_prices[usable], window_volumes[usable]
        if window_prices.size == 0 or window_volumes.sum() <= 0:
            output[index] = np.nan
            continue
        histogram, edges = np.histogram(window_prices, bins=bins, weights=window_volumes)
        if histogram.sum() <= 0:
            output[index] = window_prices[-1]
            continue
        best = int(histogram.argmax())
        output[index] = (edges[best] + edges[best + 1]) / 2.0
    return pd.Series(output, index=close.index)

def ttm_squeeze(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20) -> pd.Series:
    """Squeeze Momentum (TTM Squeeze). True if BB is inside Keltner Channels (Volatility crunch)."""
    # Bollinger
    std = df['close'].rolling(bb_period).std()
    sma20 = df['close'].rolling(bb_period).mean()
    bb_upper = sma20 + 2.0 * std
    bb_lower = sma20 - 2.0 * std

    # Keltner
    atr_val = atr(df['high'], df['low'], df['close'], kc_period)
    kc_upper = sma20 + 1.5 * atr_val
    kc_lower = sma20 - 1.5 * atr_val

    # Squeeze is on when BB is inside Keltner
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    return squeeze_on

def _pivot_flags(high: np.ndarray, low: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized pivot detection using a centred rolling window."""
    size = len(high)
    pivot_high = np.zeros(size, dtype=bool)
    pivot_low = np.zeros(size, dtype=bool)
    if size < 2 * window + 1:
        return pivot_high, pivot_low
    from numpy.lib.stride_tricks import sliding_window_view

    high_windows = sliding_window_view(high, 2 * window + 1)
    low_windows = sliding_window_view(low, 2 * window + 1)
    inner = slice(window, size - window)
    pivot_high[inner] = high[inner] >= high_windows.max(axis=1)
    pivot_low[inner] = low[inner] <= low_windows.min(axis=1)
    return pivot_high, pivot_low


def causal_support_resistance(df: pd.DataFrame, window: int = 10, memory: int = 40) -> pd.DataFrame:
    """Nearest support and resistance *known at each bar*.

    A pivot is only confirmed ``window`` bars after it forms, so a historical
    bar is never allowed to see a pivot that had not been confirmed yet. The
    previous implementation selected the three lowest pivot highs over the
    entire history, which for a rising market always produced resistance levels
    far below the current price and made the resistance rule dead code.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    size = len(df)
    pivot_high, pivot_low = _pivot_flags(high, low, window)

    high_indices = np.flatnonzero(pivot_high)
    low_indices = np.flatnonzero(pivot_low)
    high_confirmed = high_indices + window
    low_confirmed = low_indices + window

    support = np.full(size, np.nan)
    resistance = np.full(size, np.nan)
    for index in range(size):
        price = close[index]
        if not np.isfinite(price):
            continue
        known_highs = high_indices[:np.searchsorted(high_confirmed, index, side="right")][-memory:]
        known_lows = low_indices[:np.searchsorted(low_confirmed, index, side="right")][-memory:]
        if known_highs.size:
            above = high[known_highs][high[known_highs] > price]
            if above.size:
                resistance[index] = float(above.min())
        if known_lows.size:
            below = low[known_lows][low[known_lows] < price]
            if below.size:
                support[index] = float(below.max())
    return pd.DataFrame({"nearest_support": support, "nearest_resistance": resistance}, index=df.index)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: worst expected loss at given confidence."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(np.percentile(clean, (1 - confidence) * 100))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the DataFrame (in place copy)."""
    if df.empty:
        return df
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out.get("volume", pd.Series(0, index=out.index))

    out["sma_short"] = sma(close, 20)
    out["sma_long"] = sma(close, 50)
    out["sma_200"] = sma(close, 200)
    out["ema_short"] = ema(close, 9)
    out["ema_long"] = ema(close, 21)
    out["vwap"] = vwap(high, low, close, volume)
    out["smart_vwap"] = (out["vwap"] + out["ema_short"]) / 2
    out["pineify_bias"] = out["ema_short"] - out["sma_long"]
    out["pineify_trend"] = np.where(out["pineify_bias"] >= 0, 1, 0)

    out["rsi"] = rsi(close, config.rsi_period)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = bollinger_bands(close, config.bb_period, config.bb_std)
    out["stoch_k"], out["stoch_d"] = stochastic(high, low, close)
    out["obv"] = obv(close, volume)
    out["atr"] = atr(high, low, close)
    out["adx"], out["plus_di"], out["minus_di"] = adx(high, low, close)
    out["williams_r"] = williams_r(high, low, close)
    out["cmf"] = chaikin_money_flow(high, low, close, volume)
    out["supertrend_direction"] = supertrend_direction(high, low, close)
    out["tenkan"], out["kijun"], out["senkou_a"], out["senkou_b"] = ichimoku_cloud(high, low)

    # Premium TradingView Indicators
    psar = PSARIndicator(high, low, close)
    out["psar"] = psar.psar()
    out["vp_poc"] = volume_profile_poc(close, volume)
    out["ttm_squeeze"] = ttm_squeeze(out)

    kc = KeltnerChannel(high, low, close)
    out["kc_lower"] = kc.keltner_channel_lband()
    out["kc_upper"] = kc.keltner_channel_hband()

    out["returns"] = close.pct_change()
    out["volume_sma"] = volume.rolling(window=20).mean()
    out["volatility"] = out["returns"].rolling(window=config.short_ma).std() * np.sqrt(252)

    # Causal structure levels. These replace the previous whole-frame
    # computation so that a historical bar cannot see future pivots.
    levels = causal_support_resistance(out)
    out["nearest_support"] = levels["nearest_support"]
    out["nearest_resistance"] = levels["nearest_resistance"]

    # Per-bar regime so backtests and live signals see identical context. The
    # previous code injected a single whole-period regime reading into live
    # signals only, which made backtest and live behaviour inconsistent.
    out["regime_bias"] = np.where(
        out["sma_short"] > out["sma_long"] * 1.02, 1.0,
        np.where(out["sma_short"] < out["sma_long"] * 0.98, -1.0, 0.0),
    )

    return out


def compute_var(returns: pd.Series) -> float:
    return value_at_risk(returns, config.var_confidence)


def last_indicators(df: pd.DataFrame) -> dict:
    """Return the latest values of key indicators as a flat dict."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    keys = [
        "sma_short", "sma_long", "sma_200", "ema_short", "ema_long", "vwap",
        "smart_vwap", "pineify_bias", "pineify_trend",
        "open", "close", "volume", "volume_sma", "returns",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_lower", "bb_mid", "bb_upper",
        "stoch_k", "stoch_d", "obv", "atr", "adx", "plus_di", "minus_di",
        "williams_r", "volatility", "cmf", "tenkan", "kijun", "senkou_a", "senkou_b",
        "psar", "supertrend_direction", "vp_poc", "ttm_squeeze", "kc_lower", "kc_upper",
        "nearest_support", "nearest_resistance", "regime_bias",
    ]
    out = {k: (round(float(last[k]), 4) if pd.notna(last.get(k)) else None) for k in keys}

    # Flat lists are kept for display compatibility; they are derived from the
    # causal per-bar levels rather than from a whole-frame pivot scan.
    out["support"] = [out["nearest_support"]] if out.get("nearest_support") is not None else []
    out["resistance"] = [out["nearest_resistance"]] if out.get("nearest_resistance") is not None else []
    out["var"] = round(compute_var(df["returns"]), 4)
    return out