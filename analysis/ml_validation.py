"""Back-compatibility adapter for the leakage-safe evidence engine.

The dashboard and signal engine used to import ``validate_ml_ensemble`` from
this module, whose old implementation trained on the first 70% and tested on
the final 30% with a single split, then demanded an unreachable 80/80 gate.

The real implementation now lives in ``analysis.signal_models``: causal
features, ATR barrier labels, a gradient-boosting / ridge / trend-forecaster
ensemble, purged + embargoed walk-forward folds, isotonic calibration, and an
operating-point sweep that reports the honest precision/coverage trade-off.

This module keeps the historical entry points and maps them through.
"""

from __future__ import annotations

import pandas as pd

from analysis import signal_models as sm


def validate_ml_ensemble(df: pd.DataFrame, timeframe: str = "medium",
                         allow_short: bool = True) -> dict:
    """Validate the signal ensemble for one timeframe on the given history.

    ``allow_short`` is accepted for call-site compatibility; the evidence
    engine measures long-side precision (the emitted BUY rule) and mirrors it
    for SELL decisions via P(down) = 1 - P(up).
    """
    return sm.analyze_timeframe(df, timeframe, sm.DEFAULT_TARGET_PRECISION)


def regime_gate_for(df: pd.DataFrame) -> dict:
    """Causal regime-gate state for the most recent bar, no model needed.

    The gate depends only on trailing market state (benchmark trend, volatility
    percentiles), so it can be attached to pooled/panel evidence that has no
    per-asset fitted bundle.
    """
    gate: dict = {}
    try:
        features = sm.build_features(df)
    except Exception:
        return gate
    if features.empty:
        return gate
    row = features.iloc[-1]
    for column in ("long_size_mult", "short_size_mult", "mkt_trend", "vol_regime_pctile"):
        try:
            value = row[column]
            gate[column] = None if pd.isna(value) else round(float(value), 4)
        except Exception:
            gate[column] = None
    return gate


def score_latest_bar(df: pd.DataFrame, bundle) -> float | None:
    """Calibrated P(up) for the most recent bar using a fitted bundle."""
    probability, _ = score_latest_with_gate(df, bundle)
    return probability


def score_latest_with_gate(df: pd.DataFrame, bundle) -> tuple:
    """Calibrated P(up) *and* the causal regime gate for the most recent bar.

    The gate columns (long/short sizing multipliers) are computed from trailing
    benchmark trend and volatility percentiles in ``signal_models`` — the same
    state the engine needs to size or block the entry it is about to emit.
    """
    gate: dict = {}
    if bundle is None:
        return None, gate
    try:
        features = sm.build_features(df)
    except Exception:
        return None, gate
    if features.empty:
        return None, gate
    latest = features.iloc[[-1]]
    probability = None
    try:
        probability = float(bundle.predict_proba(latest)[0])
    except Exception:
        probability = None
    row = latest.iloc[0]
    for column in ("long_size_mult", "short_size_mult", "mkt_trend", "vol_regime_pctile"):
        try:
            value = row[column]
            gate[column] = None if pd.isna(value) else round(float(value), 4)
        except Exception:
            gate[column] = None
    return probability, gate