"""Pooled multi-asset panel model with cross-sectional features.

Per-symbol ensembles see only one history, so they cannot learn where an asset
stands *relative* to its peers or to the overall market. This module pools
several liquid symbols into one training frame and adds cross-sectional
features that only exist in a panel:

* ``xs_mom_rank``     - percentile rank of the 20-bar momentum among peers on the same date
* ``xs_rel_strength`` - 20-bar momentum minus the peer average on the same date
* ``xs_vol_rank``     - percentile rank of realised volatility among peers

Every cross-sectional feature uses only information available on its own date,
so the panel stays causal. Training and evaluation reuse the leakage-safe
machinery from ``analysis.signal_models`` (ATR barrier labels, purged and
embargoed walk-forward folds, isotonic calibration, operating-point sweep),
with the purge gap scaled by the number of symbols per date so that no
training row can see a test row's label window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analysis import signal_models as sm

logger = logging.getLogger(__name__)

PANEL_XS_FEATURES = ["xs_mom_rank", "xs_rel_strength", "xs_vol_rank"]
PANEL_FEATURES = sm.FEATURES + PANEL_XS_FEATURES
MIN_PANEL_SYMBOLS = 3
MIN_PANEL_DATES = 250


def prepare_panel_assets(frames, benchmark=None):
    """Build causal features once per symbol, shared across timeframes.

    Returns ``{symbol: {"features": DataFrame, "enriched": DataFrame}}``.
    Symbols that fail (empty history, unusable indicators) are dropped with a
    logged warning instead of aborting the whole panel.
    """
    if benchmark is None:
        benchmark = sm._benchmark_frame()
    prepared = {}
    for symbol, history in (frames or {}).items():
        try:
            if history is None or history.empty:
                raise ValueError("empty history")
            enriched = sm._enrich(history)
            features = sm.build_features(enriched, benchmark=benchmark)
            if features.empty:
                raise ValueError("no features could be built")
            prepared[symbol] = {"features": features, "enriched": enriched}
        except Exception as error:
            logger.warning("Panel preparation skipped %s: %s", symbol, error)
    return prepared


def build_panel(prepared, horizon):
    """Stack per-symbol feature/label frames into one pooled training frame.

    The result is sorted by date with ``date``/``symbol`` columns and the three
    cross-sectional features computed per date, then rows without a usable
    label are dropped (features are forward-filled per symbol, never across
    symbols, and labels are never filled).
    """
    parts = []
    for symbol, bundle in prepared.items():
        try:
            labels = sm.build_labels(bundle["enriched"], horizon)
            if labels.direction.dropna().empty:
                continue
            frame = bundle["features"].copy()
            frame["date"] = frame.index
            frame["symbol"] = symbol
            frame["y_direction"] = labels.direction
            frame["y_relative"] = labels.relative_direction
            frame["y_return"] = labels.forward_return
            frame["y_long"] = labels.long_return
            frame["y_short"] = labels.short_return
            parts.append(frame)
        except Exception as error:
            logger.warning("Panel labels skipped %s: %s", symbol, error)
    if len(parts) < MIN_PANEL_SYMBOLS:
        return None

    panel = pd.concat(parts).sort_values("date", kind="stable").reset_index(drop=True)
    grouped_by_date = panel.groupby("date", sort=False)
    panel["xs_mom_rank"] = grouped_by_date["mom_20"].rank(pct=True)
    panel["xs_rel_strength"] = panel["mom_20"] - grouped_by_date["mom_20"].transform("mean")
    panel["xs_vol_rank"] = grouped_by_date["volatility"].rank(pct=True)

    # Forward-fill features within each symbol only; a symbol's gap must never
    # be filled with another symbol's values.
    panel[PANEL_FEATURES] = panel.groupby("symbol", sort=False)[PANEL_FEATURES].ffill()
    panel = panel.dropna(subset=["y_relative", "y_direction", "y_return"])
    if panel.empty or panel["date"].nunique() < MIN_PANEL_DATES:
        return None
    return panel.reset_index(drop=True)


def _panel_fold_bounds(size: int, n_folds: int, horizon: int, embargo: int,
                       symbols_per_date: int) -> list:
    """Expanding-window folds with a purge gap measured in *dates*.

    The single-symbol gap of ``horizon + embargo`` rows becomes
    ``(horizon + embargo) * symbols_per_date`` rows, because that many rows
    share one date in the stacked frame.
    """
    gap = (horizon + embargo) * max(1, symbols_per_date)
    minimum = (n_folds + 1) * 40 * max(1, symbols_per_date)
    if size < minimum:
        return []
    fold_size = size // (n_folds + 1)
    bounds = []
    for fold in range(1, n_folds + 1):
        train_end = fold_size * fold
        test_start = train_end + gap
        test_end = min(size, test_start + fold_size)
        if test_start >= size or test_end - test_start < 20 * max(1, symbols_per_date):
            continue
        bounds.append((train_end, test_start, test_end))
    return bounds


def panel_cross_val(panel: pd.DataFrame, horizon: int, n_folds: int = sm.N_FOLDS) -> pd.DataFrame:
    """Out-of-fold raw panel predictions, mirroring ``signal_models.cross_val_scores``."""
    symbols_per_date = max(1, int(panel.groupby("date").size().median()))
    bounds = _panel_fold_bounds(len(panel), n_folds, horizon, sm.EMBARGO_BARS, symbols_per_date)
    collected = []
    for fold_number, (train_end, test_start, test_end) in enumerate(bounds, start=1):
        train_frame = panel.iloc[:train_end]
        test_frame = panel.iloc[test_start:test_end]
        try:
            bundle = sm.fit_bundle(train_frame, feature_columns=PANEL_FEATURES)
        except ValueError:
            continue
        collected.append(pd.DataFrame({
            "score": bundle.raw_scores(test_frame[PANEL_FEATURES]),
            "y": test_frame["y_relative"].to_numpy(),
            "absolute_up": test_frame["y_direction"].to_numpy(),
            "forward_return": test_frame["y_return"].to_numpy(),
            "long_return": test_frame["y_long"].to_numpy(),
            "short_return": test_frame["y_short"].to_numpy(),
            "forecast": bundle.forecast(test_frame[PANEL_FEATURES]),
            "fold": fold_number,
        }, index=test_frame.index))
    if not collected:
        return pd.DataFrame(columns=["score", "y", "absolute_up", "forward_return",
                                     "long_return", "short_return", "forecast", "fold"])
    return pd.concat(collected).sort_index()


def _live_probabilities(bundle, panel: pd.DataFrame, prepared: dict) -> dict:
    """Calibrated P(up) for every panel symbol at its most recent bar.

    The snapshot for the cross-sectional ranks is each symbol's own latest row,
    so a symbol that stopped updating earlier is ranked against the others'
    latest state rather than silently dropped.
    """
    snapshot = panel.groupby("symbol", sort=False).tail(1).set_index("symbol")
    probabilities = {}
    for symbol in prepared:
        if symbol not in snapshot.index:
            continue
        try:
            row = snapshot.loc[[symbol], PANEL_FEATURES]
            probabilities[symbol] = round(float(bundle.predict_proba(row)[0]), 4)
        except Exception as error:
            logger.warning("Panel live score failed for %s: %s", symbol, error)
    return probabilities


def rank_diagnostics(panel: pd.DataFrame, out_of_fold: pd.DataFrame) -> dict:
    """Cross-sectional skill of the pooled model: daily rank IC and tercile spread.

    A pooled model's natural job is ranking peers against each other, not
    calling absolute direction on its own. These diagnostics measure exactly
    that, out-of-sample: the Spearman correlation between the model's score and
    realised forward returns within each date (the information coefficient),
    and the return spread between the top and bottom score terciles — the edge
    a long-short portfolio would capture.
    """
    if out_of_fold.empty:
        return {}
    scored = pd.DataFrame({
        "date": panel.loc[out_of_fold.index, "date"].to_numpy(),
        "score": out_of_fold["score"].to_numpy(),
        "forward_return": out_of_fold["forward_return"].to_numpy(),
    }).dropna()
    if scored.empty:
        return {}

    def _daily_ic(group: pd.DataFrame) -> float:
        if len(group) < 4 or group["score"].nunique() < 2 or group["forward_return"].nunique() < 2:
            return np.nan
        return float(group["score"].corr(group["forward_return"], method="spearman"))

    def _tercile_spread(group: pd.DataFrame) -> float:
        if len(group) < 6:
            return np.nan
        rank_pct = group["score"].rank(pct=True)
        top = group.loc[rank_pct >= 2.0 / 3.0, "forward_return"].mean()
        bottom = group.loc[rank_pct <= 1.0 / 3.0, "forward_return"].mean()
        if not np.isfinite(top) or not np.isfinite(bottom):
            return np.nan
        return float(top - bottom)

    daily_ic = scored.groupby("date", sort=True).apply(_daily_ic).dropna()
    spreads = scored.groupby("date", sort=True).apply(_tercile_spread).dropna()
    result = {"dates_evaluated": int(len(daily_ic))}
    if len(daily_ic):
        result.update({
            "ic_mean": round(float(daily_ic.mean()), 4),
            "ic_ir": round(float(daily_ic.mean() / daily_ic.std()), 3) if daily_ic.std() > 0 else None,
            "ic_positive_share": round(float((daily_ic > 0).mean()), 3),
        })
    if len(spreads):
        result.update({
            "spread_mean": round(float(spreads.mean()), 6),
            "spread_positive_share": round(float((spreads > 0).mean()), 3),
            "spread_dates": int(len(spreads)),
        })
    return result


def analyze_panel(prepared: dict, timeframe: str = "medium",
                  target_precision: float = sm.DEFAULT_TARGET_PRECISION,
                  n_folds: int = sm.N_FOLDS) -> dict:
    """Full leakage-safe evaluation of the pooled panel ensemble for one timeframe.

    Returns the same result shape as ``signal_models.analyze_timeframe`` plus
    ``scope='panel'``, the participating ``symbols``, and per-symbol
    ``live_probabilities`` for the most recent bar.
    """
    horizon = sm.TIMEFRAME_HORIZONS.get(timeframe, 15)
    symbols = sorted(prepared)
    result = {"available": False, "timeframe": timeframe, "horizon": horizon,
              "target_precision": target_precision, "models": None, "error": None,
              "scope": "panel", "symbols": symbols}

    panel = build_panel(prepared, horizon)
    if panel is None:
        result["error"] = (f"Panel needs at least {MIN_PANEL_SYMBOLS} symbols with "
                           f"{MIN_PANEL_DATES}+ usable dates of history; got {len(symbols)}.")
        return result

    out_of_fold = panel_cross_val(panel, horizon, n_folds)
    if out_of_fold.empty:
        result["error"] = "Panel walk-forward validation produced no folds (history too short)."
        return result

    probabilities = sm.cross_validated_probabilities(out_of_fold)
    valid = probabilities.notna()
    scores = probabilities[valid].to_numpy()
    labels = out_of_fold.loc[valid, "y"].to_numpy()
    forward = out_of_fold.loc[valid, "forward_return"].to_numpy()
    long_r = out_of_fold.loc[valid, "long_return"].to_numpy()
    short_r = out_of_fold.loc[valid, "short_return"].to_numpy()
    absolute = out_of_fold.loc[valid, "absolute_up"].to_numpy()

    diagnostics = sm.classifier_diagnostics(scores, labels)
    sweep = sm.sweep_operating_points(scores, labels, forward, long_r, short_r, absolute)
    operating = sm.select_operating_point(sweep, target_precision)
    reliability = sm.reliability_table(scores, labels)
    rank_objective = rank_diagnostics(panel, out_of_fold)

    try:
        bundle = sm.fit_bundle(panel, feature_columns=PANEL_FEATURES)
    except ValueError as error:
        result["error"] = str(error)
        return result
    bundle.calibrator = sm.fit_calibrator(out_of_fold["score"].to_numpy(), out_of_fold["y"].to_numpy())

    base_rate = float(labels.mean()) if labels.size else None
    result.update({
        "available": True,
        "models": bundle,
        "panel_rows": int(len(panel)),
        "panel_dates": int(panel["date"].nunique()),
        "folds": int(out_of_fold["fold"].nunique()),
        "validated_bars": int(labels.size),
        "base_rate": round(base_rate, 4) if base_rate is not None else None,
        "diagnostics": diagnostics,
        "sweep": sweep,
        "operating": operating,
        "reliability": reliability,
        "rank_objective": rank_objective,
        "live_probabilities": _live_probabilities(bundle, panel, prepared),
        # Backwards-compatible keys used by the dashboard panel and engine.
        "success_ratio": operating.get("success"),
        "trade_win_rate": operating.get("success"),
        "model_hit_rate": diagnostics.get("roc_auc"),
        "threshold": target_precision,
        "trades": operating.get("signals") or 0,
        "coverage": operating.get("coverage"),
        "eligible": bool(operating.get("achieved")),
        "reason": operating.get("reason"),
    })
    return result
