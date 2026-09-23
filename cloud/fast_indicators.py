"""Vectorised indicators for whole-market scans.

The dashboard's ``analysis.indicators`` module operates one pandas Series at a
time. That is the right shape for a single-symbol analysis, but a whole-market
run evaluates thousands of symbols, so this module computes the same formulas
across a 2-D ``(symbols x bars)`` matrix in one pass.

Formulas intentionally mirror ``analysis/indicators.py`` (Wilder RSI via
``ewm(alpha=1/period, adjust=False)`` and simple rolling means) so a cloud
signal and a dashboard signal agree on the same data.
"""
from __future__ import annotations

import numpy as np


def rolling_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    """Row-wise simple moving average over the trailing ``window`` columns."""
    rows, cols = matrix.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if cols < window:
        return out
    cumulative = np.cumsum(matrix, axis=1)
    head = np.concatenate([np.zeros((rows, 1)), cumulative[:, : cols - window]], axis=1)
    out[:, window - 1:] = (cumulative[:, window - 1:] - head) / window
    return out


def ewm(matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise exponential smoothing, equivalent to pandas ``ewm(adjust=False)``."""
    rows, cols = matrix.shape
    out = np.empty((rows, cols), dtype=float)
    if cols == 0:
        return out
    out[:, 0] = matrix[:, 0]
    for column in range(1, cols):
        out[:, column] = alpha * matrix[:, column] + (1.0 - alpha) * out[:, column - 1]
    return out


def wilder_rsi(matrix: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI across every row, pushed through ``fillna(50)`` like the app."""
    if matrix.shape[1] < 2:
        return np.full(matrix.shape, 50.0)
    delta = np.diff(matrix, axis=1)
    gain = np.clip(delta, 0.0, None)
    loss = -np.clip(delta, None, 0.0)
    alpha = 1.0 / period
    average_gain = ewm(gain, alpha)
    average_loss = ewm(loss, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(average_loss > 0, average_gain / average_loss, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + ratio))
    rsi = np.where((average_loss <= 0) & (average_gain > 0), 100.0,
                   np.where((average_loss <= 0) & (average_gain <= 0), 50.0, rsi))
    # The RSI series is one column shorter than prices; pad the front with 50 so
    # every row keeps the same time axis as the closing-price matrix.
    padded = np.concatenate([np.full((matrix.shape[0], 1), 50.0), rsi], axis=1)
    return np.clip(padded, 0.0, 100.0)


def resample_last(matrix: np.ndarray, buckets: list[int]) -> np.ndarray:
    """Collapse columns into groups, keeping the last value of each group.

    ``buckets`` maps each column index to a group id (e.g. an ISO week number).
    Used to derive weekly bars from daily bars without a second API fetch.
    """
    if matrix.shape[1] == 0:
        return matrix
    group_ids = np.asarray(buckets)
    outputs = []
    for group in np.unique(group_ids):
        columns = np.flatnonzero(group_ids == group)
        outputs.append(matrix[:, columns[-1]])
    return np.column_stack(outputs)


def last_value(matrix: np.ndarray) -> np.ndarray:
    """Return only the most recent column, or NaN where a row is empty."""
    if matrix.shape[1] == 0:
        return np.full(matrix.shape[0], np.nan)
    return matrix[:, -1]


def previous_value(matrix: np.ndarray) -> np.ndarray:
    """Return the second-to-last column, or NaN where the row is too short."""
    if matrix.shape[1] < 2:
        return np.full(matrix.shape[0], np.nan)
    return matrix[:, -2]


def crossed_above(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Boolean column of fresh bullish crosses (previous bar was not above)."""
    now = last_value(fast) > last_value(slow)
    before = previous_value(fast) > previous_value(slow)
    valid = np.isfinite(last_value(fast)) & np.isfinite(last_value(slow)) & \
        np.isfinite(previous_value(fast)) & np.isfinite(previous_value(slow))
    return now & ~before & valid


def crossed_below(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Boolean column of fresh bearish crosses (previous bar was not below)."""
    now = last_value(fast) < last_value(slow)
    before = previous_value(fast) < previous_value(slow)
    valid = np.isfinite(last_value(fast)) & np.isfinite(last_value(slow)) & \
        np.isfinite(previous_value(fast)) & np.isfinite(previous_value(slow))
    return now & ~before & valid


def crossed_up_through(values: np.ndarray, level: float) -> np.ndarray:
    """Boolean column for a recovery back above ``level`` (e.g. RSI leaving 30)."""
    now = last_value(values) >= level
    before = previous_value(values) < level
    return now & before & np.isfinite(last_value(values)) & np.isfinite(previous_value(values))


def crossed_down_through(values: np.ndarray, level: float) -> np.ndarray:
    """Boolean column for a drop back below ``level`` (e.g. RSI leaving 70)."""
    now = last_value(values) <= level
    before = previous_value(values) > level
    return now & before & np.isfinite(last_value(values)) & np.isfinite(previous_value(values))
