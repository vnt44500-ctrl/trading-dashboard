"""Leakage-safe classification, regression and forecasting ensemble.

This module is the evidence engine behind the live signals. It replaces the
previous single-split "train on 70%, hope the holdout clears 80%" gate, which
was unreachable by construction: it required an 80% directional hit rate *and*
an 80% trade win rate on a five-bar forecast.

What it does instead:

1. Builds causal features (no future information, no whole-frame statistics).
2. Labels each bar with the outcome the strategy actually trades — a
   first-touch barrier test sized from ATR — and with plain forward direction.
3. Trains an ensemble: a gradient-boosted classifier for direction, a linear
   and a gradient-boosted regressor for expected return, and an exponentially
   weighted trend/volatility forecaster.
4. Validates with purged, embargoed walk-forward folds so that no training row
   overlaps the label window of a test row.
5. Learns a probability calibrator on out-of-fold scores, then sweeps the
   decision threshold to report the honest precision/coverage trade-off.
6. Selects the operating threshold that reaches the requested success rate,
   and reports the coverage that survives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import indicators as ind

logger = logging.getLogger(__name__)

# Feature set is intentionally compact and causal: every column at row t is
# computed only from rows <= t.
FEATURES = [
    "returns",
    "mom_2",
    "mom_3",
    "mom_5",
    "mom_10",
    "mom_20",
    "rsi_scaled",
    "rsi_delta",
    "macd_hist_norm",
    "price_to_sma20",
    "price_to_sma50",
    "sma20_to_sma50",
    "price_to_sma200",
    "ema9_to_ema21",
    "atr_pct",
    "volatility",
    "vol_ratio",
    "adx_scaled",
    "di_spread",
    "bb_position",
    "bb_width",
    "stoch_k_scaled",
    "williams_r_scaled",
    "cmf",
    "volume_z",
    "vwap_distance",
    "poc_distance",
    "supertrend_direction",
    "psar_distance",
    "tenkan_kijun_spread",
    "regime_bias",
    "dist_to_support",
    "dist_to_resistance",
    # Market-regime features (causal, from the SPY benchmark history). They
    # place each bar in its market context: where the market is relative to
    # its own trend, how volatile the market is, and how the asset is moving
    # relative to the market (relative strength, rolling beta).
    "mkt_ret_20",
    "mkt_trend",
    "mkt_vol_20",
    "rel_str_20",
    "rel_str_60",
    "beta_60",
]

# Regime-gate columns. These are *decision-layer* outputs, not model inputs:
# they size or block live entries from the market state at each bar. They stay
# out of FEATURES on purpose so the trained model contract (and the cached
# out-of-fold replay) is unchanged; the gate is applied after the model speaks.
GATE_COLUMNS = [
    "mkt_vol_pctile",    # trailing 1-year percentile of benchmark 20d volatility
    "vol_regime_pctile", # trailing 1-year percentile of this asset's 20d volatility
    "long_size_mult",    # 0..1 multiplier for new long entries
    "short_size_mult",   # 0..1 multiplier for new short entries
]

# Gate thresholds. Faber-style trend filter on the benchmark (200-day SMA via
# mkt_trend) plus a volatility-crisis overlay. All inputs are trailing, so the
# gate is fully causal; when benchmark/vol information is missing (warmup
# rows), multipliers default to 1.0 (no gate) rather than blocking.
GATE_TREND_HALF_LONG = 0.0     # SPY below its 200-SMA: longs at half size
GATE_TREND_BLOCK_LONG = -0.10  # deep bear: no new longs
GATE_TREND_HALF_SHORT = 0.05   # modest bull drift: shorts at half size
GATE_TREND_BLOCK_SHORT = 0.10  # strong uptrend: no new shorts
GATE_VOL_HALF = 0.95           # 95th pct trailing vol: everything half size
GATE_VOL_BLOCK = 0.98          # 98th pct trailing vol: no new entries

# Default forecast horizons (in bars) per dashboard timeframe.
TIMEFRAME_HORIZONS = {"short": 5, "medium": 15, "long": 40}
DEFAULT_TARGET_PRECISION = 0.80
MINIMUM_TRADES = 12
N_FOLDS = 4
EMBARGO_BARS = 5


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Return an indicator frame, reusing one that is already enriched.

    compute_indicators is the expensive step, and both the feature builder and
    the label builder need it, so enrich once and pass the result along.
    """
    if "atr" in df.columns and "regime_bias" in df.columns and "vp_poc" in df.columns:
        return df
    return ind.compute_indicators(df)


_BENCHMARK_CACHE: dict[str, pd.DataFrame | None] = {}


def _benchmark_frame(refresh: bool = False) -> pd.DataFrame | None:
    """Daily history for the market benchmark (SPY), cached per process.

    The benchmark features give every symbol the same causal read of the market
    regime it is trading in. Only information available at each bar is used, so
    there is no leakage. If the fetch fails the regime features are simply NaN
    and the models fill them with the training median, so the engine keeps
    working offline.
    """
    if not refresh and "SPY" in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE["SPY"]
    frame = None
    try:
        from data.market_data import provider
        candidate = provider.get_history("SPY", period="5y")
        if candidate is not None and not candidate.empty and "close" in candidate.columns:
            frame = candidate
    except Exception as error:
        logger.warning("Benchmark fetch for regime features failed: %s", error)
    _BENCHMARK_CACHE["SPY"] = frame
    return frame


def _regime_gate_columns(out: pd.DataFrame, data: pd.DataFrame) -> dict:
    """Trailing percentile regimes and the long/short sizing multipliers.

    All computations look backwards only: the percentiles are rolling 1-year
    ranks, and the multipliers are deterministic thresholds on those ranks and
    on the benchmark trend. The rules exist because the 10-year replay showed
    every timeframe bleeding badly in 2020-style crises while the same rules
    barely touched ordinary months.
    """
    result = {column: pd.Series(1.0, index=out.index) for column in GATE_COLUMNS}
    result["mkt_vol_pctile"] = np.nan
    result["vol_regime_pctile"] = np.nan

    asset_vol = data.get("volatility")
    if asset_vol is not None and len(asset_vol):
        result["vol_regime_pctile"] = asset_vol.rolling(252, min_periods=126).rank(pct=True)
    elif "volatility" in out.columns:
        result["vol_regime_pctile"] = out["volatility"].rolling(252, min_periods=126).rank(pct=True)

    trend = out.get("mkt_trend")
    vol_pctile = result["vol_regime_pctile"]

    long_mult = pd.Series(1.0, index=out.index)
    short_mult = pd.Series(1.0, index=out.index)
    if trend is not None:
        long_mult = long_mult.mask(trend < GATE_TREND_HALF_LONG, 0.5).mask(
            trend < GATE_TREND_BLOCK_LONG, 0.0)
        short_mult = short_mult.mask(trend > GATE_TREND_HALF_SHORT, 0.5).mask(
            trend > GATE_TREND_BLOCK_SHORT, 0.0)
    high_vol = vol_pctile >= GATE_VOL_HALF
    crisis_vol = vol_pctile >= GATE_VOL_BLOCK
    long_mult = long_mult.mask(high_vol, long_mult * 0.5).mask(crisis_vol, 0.0)
    short_mult = short_mult.mask(high_vol, short_mult * 0.5).mask(crisis_vol, 0.0)

    result["long_size_mult"] = long_mult.fillna(1.0)
    result["short_size_mult"] = short_mult.fillna(1.0)
    bench_vol = out.get("mkt_vol_20")
    if bench_vol is not None and isinstance(bench_vol, pd.Series) and bench_vol.notna().any():
        result["mkt_vol_pctile"] = bench_vol.rolling(252, min_periods=126).rank(pct=True)
    return result


def build_features(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Derive the causal feature matrix from raw OHLCV.

    ``benchmark`` overrides the auto-fetched SPY frame (pass an empty frame to
    force the regime features to NaN, e.g. in offline tests).
    """
    data = _enrich(df)
    if data.empty:
        return pd.DataFrame(columns=FEATURES)

    close = data["close"].astype(float)
    atr = data["atr"].replace(0, np.nan)
    volume = data.get("volume", pd.Series(0.0, index=data.index)).astype(float)

    out = pd.DataFrame(index=data.index)
    out["returns"] = data["returns"]
    for horizon in (2, 3, 5, 10, 20):
        out[f"mom_{horizon}"] = close.pct_change(horizon)

    out["rsi_scaled"] = (data["rsi"] - 50.0) / 50.0
    out["rsi_delta"] = data["rsi"].diff() / 50.0
    out["macd_hist_norm"] = data["macd_hist"] / close
    out["price_to_sma20"] = (close - data["sma_short"]) / close
    out["price_to_sma50"] = (close - data["sma_long"]) / close
    out["sma20_to_sma50"] = (data["sma_short"] - data["sma_long"]) / close
    out["price_to_sma200"] = (close - data["sma_200"]) / close
    out["ema9_to_ema21"] = (data["ema_short"] - data["ema_long"]) / close
    out["atr_pct"] = atr / close
    out["volatility"] = data["volatility"]
    out["vol_ratio"] = data["volatility"] / data["volatility"].rolling(60).mean()
    out["adx_scaled"] = data["adx"] / 100.0
    out["di_spread"] = (data["plus_di"] - data["minus_di"]) / 100.0

    band_width = (data["bb_upper"] - data["bb_lower"]).replace(0, np.nan)
    out["bb_position"] = (close - data["bb_lower"]) / band_width
    out["bb_width"] = band_width / close
    out["stoch_k_scaled"] = (data["stoch_k"] - 50.0) / 50.0
    out["williams_r_scaled"] = (data["williams_r"] + 50.0) / 50.0
    out["cmf"] = data["cmf"]

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std().replace(0, np.nan)
    out["volume_z"] = (volume - volume_mean) / volume_std

    out["vwap_distance"] = (close - data["smart_vwap"]) / close
    out["poc_distance"] = (close - data["vp_poc"]) / close
    out["supertrend_direction"] = data["supertrend_direction"]
    out["psar_distance"] = (close - data["psar"]) / close
    out["tenkan_kijun_spread"] = (data["tenkan"] - data["kijun"]) / close
    out["regime_bias"] = data["regime_bias"]
    out["dist_to_support"] = (close - data["nearest_support"]) / close
    out["dist_to_resistance"] = (data["nearest_resistance"] - close) / close

    # Market-regime block. Everything below uses only data up to each bar:
    # trailing benchmark returns, the benchmark's own trend/volatility, and
    # rolling relative-strength/beta of this asset against the benchmark.
    bench = benchmark if (benchmark is not None and not benchmark.empty) else _benchmark_frame()
    if bench is not None and not bench.empty:
        bench_close = bench["close"].astype(float)
        bench_ret = bench_close.pct_change()
        bench_aligned = bench_ret.reindex(data.index).ffill()
        out["mkt_ret_20"] = bench_close.pct_change(20).reindex(data.index).ffill()
        out["mkt_trend"] = (bench_close / bench_close.rolling(200).mean() - 1.0).reindex(data.index).ffill()
        out["mkt_vol_20"] = bench_ret.rolling(20).std().reindex(data.index).ffill()
        out["rel_str_20"] = out["mom_20"] - out["mkt_ret_20"]
        out["rel_str_60"] = close.pct_change(60) - bench_close.pct_change(60).reindex(data.index).ffill()
        cov = data["returns"].astype(float).rolling(60).cov(bench_aligned)
        var = bench_aligned.rolling(60).var().replace(0, np.nan)
        out["beta_60"] = cov / var

    for column, values in _regime_gate_columns(out, data).items():
        out[column] = values
    return out.reindex(columns=FEATURES + GATE_COLUMNS).replace([np.inf, -np.inf], np.nan)


@dataclass
class Labels:
    """Forward-looking targets built from a single pass over the price path."""

    direction: pd.Series          # 1 if close[t+h] > close[t] else 0  (absolute)
    relative_direction: pd.Series # 1 if the move beat the causal baseline (stationary)
    forward_return: pd.Series     # close[t+h] / close[t] - 1
    long_return: pd.Series        # R-multiple of an ATR-barrier long trade
    short_return: pd.Series       # R-multiple of an ATR-barrier short trade


BARRIER_PROFILES = {
    # Per-horizon barrier geometry. The old single profile (2 ATR stop, 2:1
    # reward) was tuned for multi-week holds; at a 5-day horizon a 4-ATR target
    # is almost never reached inside the window, so most "wins" were actually
    # mark-to-market exits — and the 2-ATR stop absorbed noise the horizon
    # cannot recover from (10-year replay: 44.0% trade win rate vs a 53.9%
    # base). The short profile narrows the target so it can resolve inside 5
    # bars; medium/long keep the original geometry that their operating points
    # were measured on.
    5: {"atr_stop_multiple": 1.5, "reward_risk": 1.2},
    15: {"atr_stop_multiple": 2.0, "reward_risk": 2.0},
    40: {"atr_stop_multiple": 2.0, "reward_risk": 2.0},
}


def build_labels(df: pd.DataFrame, horizon: int, atr_stop_multiple: float | None = None,
                 reward_risk: float | None = None) -> Labels:
    """Label each bar with direction *and* the economics of the trade.

    Barrier geometry defaults to the per-horizon profile in ``BARRIER_PROFILES``
    (a 5-bar signal cannot reach a 4-ATR target; a 40-bar one can). Explicit
    arguments still override the profile.

    Two things are derived:

    * ``direction`` / ``forward_return`` - the plain close-to-close outcome over
      the horizon. This is the primary classifier target, because "was the
      signal right" is the question the dashboard reports on.
    * ``long_return`` / ``short_return`` - the R-multiple a trade would have
      realised using the strategy's own ATR stop and reward target, resolved by
      first touch. This is the economic reality check: a direction can be right
      while the trade still loses to a stop-out along the way.
    """
    profile = BARRIER_PROFILES.get(int(horizon), {})
    atr_stop_multiple = profile.get("atr_stop_multiple", 2.0) if atr_stop_multiple is None else atr_stop_multiple
    reward_risk = profile.get("reward_risk", 2.0) if reward_risk is None else reward_risk
    data = _enrich(df)
    close_values = data["close"].astype(float).to_numpy()
    high_values = data["high"].astype(float).to_numpy()
    low_values = data["low"].astype(float).to_numpy()
    atr_values = data["atr"].astype(float).to_numpy()

    size = len(data)
    index_ref = data.index
    close_all = data["close"].astype(float)

    # Causal baseline threshold for the stationary label. At bar t this uses
    # only closes up to t, so it is a legal predictor input at labelling time.
    # Without de-meaning, a bull market makes 60-75% of windows "up", and the
    # model simply learns the drift: high apparent accuracy, near-zero skill.
    past_horizon_return = close_all.pct_change(horizon)
    baseline = past_horizon_return.rolling(window=max(60, horizon * 8), min_periods=20).median()

    forward_return = pd.Series(np.nan, index=index_ref, dtype=float)
    direction = pd.Series(np.nan, index=index_ref, dtype=float)
    relative_direction = pd.Series(np.nan, index=index_ref, dtype=float)
    long_return = pd.Series(np.nan, index=index_ref, dtype=float)
    short_return = pd.Series(np.nan, index=index_ref, dtype=float)
    baseline_values = baseline.to_numpy()

    for index in range(size - horizon):
        entry = close_values[index]
        stop_distance = atr_values[index] * atr_stop_multiple
        if not np.isfinite(entry) or not np.isfinite(stop_distance) or stop_distance <= 0 or entry <= 0:
            continue

        future_close = close_values[index + horizon]
        if np.isfinite(future_close):
            forward_return.iat[index] = future_close / entry - 1.0
            direction.iat[index] = 1.0 if future_close > entry else 0.0
            if np.isfinite(baseline_values[index]):
                relative_direction.iat[index] = 1.0 if (future_close / entry - 1.0) > baseline_values[index] else 0.0

        long_target, long_stop = entry + stop_distance * reward_risk, entry - stop_distance
        short_target, short_stop = entry - stop_distance * reward_risk, entry + stop_distance
        long_outcome = None
        short_outcome = None

        for step in range(index + 1, min(index + horizon, size - 1) + 1):
            bar_high, bar_low = high_values[step], low_values[step]
            if long_outcome is None:
                if bar_low <= long_stop:
                    long_outcome = -1.0
                elif bar_high >= long_target:
                    long_outcome = reward_risk
            if short_outcome is None:
                if bar_high >= short_stop:
                    short_outcome = -1.0
                elif bar_low <= short_target:
                    short_outcome = reward_risk
            if long_outcome is not None and short_outcome is not None:
                break

        if long_outcome is None and np.isfinite(future_close):
            long_outcome = (future_close - entry) / stop_distance
        if short_outcome is None and np.isfinite(future_close):
            short_outcome = (entry - future_close) / stop_distance
        if long_outcome is not None:
            long_return.iat[index] = long_outcome
        if short_outcome is not None:
            short_return.iat[index] = short_outcome

    return Labels(direction=direction, forward_return=forward_return,
                  long_return=long_return, short_return=short_return,
                  relative_direction=relative_direction)


def _design_matrix(df: pd.DataFrame, horizon: int):
    """Align features with labels and drop rows that have no usable target."""
    enriched = _enrich(df)
    features = build_features(enriched)
    labels = build_labels(enriched, horizon)
    if features.empty or labels.direction.dropna().empty:
        return None
    frame = features.join(labels.direction.rename("y_direction"))
    frame = frame.join(labels.relative_direction.rename("y_relative"))
    frame = frame.join(labels.forward_return.rename("y_return"))
    frame = frame.join(labels.long_return.rename("y_long"))
    frame = frame.join(labels.short_return.rename("y_short"))
    # Carry features forward across gaps, but never invent a label.
    frame[FEATURES] = frame[FEATURES].ffill()
    # y_relative is the training target: it is stationary, so the model cannot
    # score points by learning the market's drift. y_direction is retained for
    # reporting how the traded direction actually resolved.
    frame = frame.dropna(subset=["y_relative", "y_direction", "y_return"])
    if frame.empty:
        return None
    return frame


@dataclass
class ModelBundle:
    """A fitted classification/regression/forecasting ensemble."""

    fill_values: pd.Series
    linear_classifier: object
    tree_classifier: object
    linear_regressor: object
    tree_regressor: object
    forest_regressor: object
    return_scale: float
    calibrator: IsotonicRegression | None = None
    feature_columns: list = field(default_factory=lambda: list(FEATURES))
    blend_weights: dict = field(default_factory=lambda: {
        "classifier": 0.55, "linear": 0.15, "tree": 0.20, "forest": 0.10,
    })

    def raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        """Uncalibrated ensemble probability that price is higher after the horizon."""
        values = self._prepare(features)
        classifier_probability = 0.5 * (
            self.linear_classifier.predict_proba(values)[:, 1]
            + self.tree_classifier.predict_proba(values)[:, 1]
        )
        scale = self.return_scale
        linear_probability = 0.5 + 0.5 * np.tanh(self.linear_regressor.predict(values) / scale)
        tree_probability = 0.5 + 0.5 * np.tanh(self.tree_regressor.predict(values) / scale)
        forest_probability = 0.5 + 0.5 * np.tanh(self.forest_regressor.predict(values) / scale)
        weights = self.blend_weights
        blended = (
            weights["classifier"] * classifier_probability
            + weights["linear"] * linear_probability
            + weights["tree"] * tree_probability
            + weights["forest"] * forest_probability
        )
        return np.clip(blended, 0.0, 1.0)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated probability, so a 0.70 threshold means roughly 70% precision."""
        raw = self.raw_scores(features)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def expected_return(self, features: pd.DataFrame) -> np.ndarray:
        values = self._prepare(features)
        return 0.5 * self.linear_regressor.predict(values) + 0.5 * self.tree_regressor.predict(values)

    def forecast(self, features: pd.DataFrame) -> np.ndarray:
        """Blended forecast of the forward return, in return units."""
        return self.expected_return(features)

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        values = features.reindex(columns=self.feature_columns).replace([np.inf, -np.inf], np.nan)
        return values.fillna(self.fill_values)


def fit_bundle(frame: pd.DataFrame, feature_columns: list | None = None) -> ModelBundle:
    """Fit the classification/regression ensemble on a training frame.

    ``feature_columns`` lets panel models train on the per-symbol features plus
    the cross-sectional columns; it defaults to the standard per-symbol set.
    """
    columns = list(feature_columns) if feature_columns else list(FEATURES)
    x_train = frame[columns].replace([np.inf, -np.inf], np.nan)
    fill_values = x_train.median().fillna(0.0)
    x_train = x_train.fillna(fill_values)
    y_direction = frame["y_relative"].astype(int)
    y_return = frame["y_return"].astype(float)

    if y_direction.nunique() < 2:
        raise ValueError("Training window contains only one direction class.")

    linear_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, random_state=42),
    ).fit(x_train, y_direction)
    tree_classifier = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=3, min_samples_leaf=20,
        l2_regularization=1.0, random_state=42,
    ).fit(x_train, y_direction)
    linear_regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, y_return)
    tree_regressor = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.06, max_depth=3, min_samples_leaf=20,
        l2_regularization=1.0, random_state=42,
    ).fit(x_train, y_return)
    forest_regressor = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=12, random_state=42, n_jobs=-1,
    ).fit(x_train, y_return)

    return ModelBundle(
        fill_values=fill_values,
        linear_classifier=linear_classifier,
        tree_classifier=tree_classifier,
        linear_regressor=linear_regressor,
        tree_regressor=tree_regressor,
        forest_regressor=forest_regressor,
        return_scale=max(float(y_return.std()), 1e-3),
        feature_columns=columns,
    )


def _purged_fold_bounds(size: int, n_folds: int, horizon: int, embargo: int):
    """Sequential expanding-window folds with a purge gap equal to the horizon.

    Without the gap the last ``horizon`` training rows share their label window
    with the first test rows, so the model would be scored on information it had
    already seen during fitting.
    """
    minimum = (n_folds + 1) * 40
    if size < minimum:
        return []
    fold_size = size // (n_folds + 1)
    bounds = []
    for fold in range(1, n_folds + 1):
        train_end = fold_size * fold
        test_start = train_end + horizon + embargo
        test_end = min(size, test_start + fold_size)
        if test_start >= size or test_end - test_start < 20:
            continue
        bounds.append((train_end, test_start, test_end))
    return bounds


def cross_val_scores(frame: pd.DataFrame, horizon: int, n_folds: int = N_FOLDS,
                     embargo: int = EMBARGO_BARS) -> pd.DataFrame:
    """Out-of-fold raw predictions for every bar covered by a validation fold."""
    bounds = _purged_fold_bounds(len(frame), n_folds, horizon, embargo)
    collected = []
    for fold_number, (train_end, test_start, test_end) in enumerate(bounds, start=1):
        train_frame = frame.iloc[:train_end]
        test_frame = frame.iloc[test_start:test_end]
        try:
            bundle = fit_bundle(train_frame)
        except ValueError:
            continue
        collected.append(pd.DataFrame({
            "score": bundle.raw_scores(test_frame[FEATURES]),
            "y": test_frame["y_relative"].to_numpy(),
            "absolute_up": test_frame["y_direction"].to_numpy(),
            "forward_return": test_frame["y_return"].to_numpy(),
            "long_return": test_frame["y_long"].to_numpy(),
            "short_return": test_frame["y_short"].to_numpy(),
            "forecast": bundle.forecast(test_frame[FEATURES]),
            "fold": fold_number,
        }, index=test_frame.index))
    if not collected:
        return pd.DataFrame(columns=["score", "y", "absolute_up", "forward_return", "long_return", "short_return", "forecast", "fold"])
    return pd.concat(collected).sort_index()


def fit_calibrator(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression | None:
    """Fit isotonic regression so scores behave like real probabilities.

    A raw ensemble score of 0.75 does not mean "right 75% of the time". Isotonic
    calibration on out-of-fold predictions makes the operating threshold
    interpretable, which is what allows an 80% success target to be stated and
    then measured rather than assumed.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.size < 60 or len(np.unique(labels)) < 2:
        return None
    try:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(scores, labels)
        return calibrator
    except ValueError:
        logger.warning("Isotonic calibration failed; falling back to raw scores.")
        return None


def classifier_diagnostics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Discrimination and reliability metrics for the calibrated classifier.

    ROC-AUC is the headline number because it is independent of the decision
    threshold and of the class base rate. Balanced accuracy and MCC are included
    because plain accuracy is trivially inflated by an unbalanced label.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    empty = {"roc_auc": None, "pr_auc": None, "brier": None, "log_loss": None,
             "base_rate": None, "balanced_accuracy": None, "mcc": None, "lift": None}
    if scores.size < 20 or len(np.unique(labels)) < 2:
        return empty
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    predicted = (scores >= 0.5).astype(float)
    try:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
        brier = float(brier_score_loss(labels, clipped))
        logloss = float(log_loss(labels, clipped))
        balanced = float(balanced_accuracy_score(labels, predicted))
        mcc = float(matthews_corrcoef(labels, predicted))
        accuracy = float((predicted == labels).mean())
    except ValueError:
        result = dict(empty)
        result["base_rate"] = round(float(labels.mean()), 4)
        return result
    base_rate = float(labels.mean())
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier": round(brier, 4),
        "log_loss": round(logloss, 4),
        "base_rate": round(base_rate, 4),
        "balanced_accuracy": round(balanced, 4),
        "mcc": round(mcc, 4),
        "lift": round(accuracy - max(base_rate, 1 - base_rate), 4),
    }


def reliability_table(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency per probability bucket."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.size == 0:
        return pd.DataFrame(columns=["bucket", "predicted", "observed", "count"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(scores, edges) - 1, 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        rows.append({
            "bucket": f"{edges[index]:.1f}-{edges[index + 1]:.1f}",
            "predicted": round(float(scores[mask].mean()), 4),
            "observed": round(float(labels[mask].mean()), 4),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def sweep_operating_points(scores: np.ndarray, labels: np.ndarray,
                           forward_return: np.ndarray | None = None,
                           long_return: np.ndarray | None = None,
                           short_return: np.ndarray | None = None,
                           absolute_up: np.ndarray | None = None,
                           cost_pct: float = 0.2) -> pd.DataFrame:
    """Precision/coverage curve across decision thresholds.

    Two success definitions are reported because they answer different questions:

    * ``Skill success %`` - did the *relative* call resolve correctly, i.e. did
      the model beat its causal baseline. This is the honest measure of whether
      the signal has any information. 50% is a coin flip.
    * ``Trade success %`` - did the ATR-barrier trade make money (R > 0) after
      stop, target and first-touch resolution. This is what the account feels.
    * ``Directional success %`` - did price simply move the signalled way. On a
      trending symbol this is dominated by the drift, so it is reported next to
      the base rate rather than on its own.
    * ``Net expected %`` - expected return minus one round trip of estimated
      costs (commissions plus slippage, ``cost_pct`` per signal). A threshold
      whose edge cannot clear its own costs is not tradable, however high its
      precision looks.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    forward = np.asarray(forward_return, dtype=float) if forward_return is not None else None
    long_r = np.asarray(long_return, dtype=float) if long_return is not None else None
    short_r = np.asarray(short_return, dtype=float) if short_return is not None else None
    absolute = np.asarray(absolute_up, dtype=float) if absolute_up is not None else None
    total = scores.size
    rows = []
    for threshold in np.round(np.arange(0.50, 0.96, 0.01), 2):
        signal_mask = (scores >= threshold) | (scores <= (1 - threshold))
        signals = int(signal_mask.sum())
        if signals == 0:
            rows.append({"Threshold": float(threshold), "Signals": 0, "Coverage %": 0.0,
                         "Success %": None, "Skill success %": None, "Trade success %": None,
                         "Directional success %": None, "Expected return %": None,
                         "Net expected %": None})
            continue

        predicted_up = scores[signal_mask] >= 0.5
        # 1. Skill: did the relative call resolve correctly?
        skill = float((predicted_up == (labels[signal_mask] == 1)).mean())
        # 2. Directional: did price move the signalled way (drift-dominated)?
        directional = None
        if absolute is not None:
            directional = float((predicted_up == (absolute[signal_mask] == 1)).mean())
        # 3. Economic: did the barrier trade finish positive?
        trade = None
        if long_r is not None and short_r is not None:
            realised = np.where(predicted_up, long_r[signal_mask], short_r[signal_mask])
            trade = float((realised > 0).mean())
        # A long earns +forward; a short earns the inverse. Averaging raw
        # forward returns across both sides would silently misprice every
        # short signal in the book, so the side is applied before averaging.
        expected = None
        if forward is not None:
            side_adjusted = np.where(predicted_up, forward[signal_mask], -forward[signal_mask])
            expected = float(side_adjusted.mean() * 100)

        rows.append({
            "Threshold": float(threshold),
            "Signals": signals,
            "Coverage %": round(signals / total * 100, 1),
            "Success %": round(skill * 100, 1),
            "Skill success %": round(skill * 100, 1),
            "Trade success %": round(trade * 100, 1) if trade is not None else None,
            "Directional success %": round(directional * 100, 1) if directional is not None else None,
            "Expected return %": round(expected, 3) if expected is not None else None,
            "Net expected %": round(expected - cost_pct, 3) if expected is not None else None,
        })
    return pd.DataFrame(rows)


def select_operating_point(sweep: pd.DataFrame, target_precision: float,
                           minimum_trades: int = MINIMUM_TRADES) -> dict:
    """Pick the widest-coverage threshold that reaches the target *and* clears costs.

    Among qualifying thresholds, ones whose net expected return (after one
    round trip of costs) is positive are preferred; among those, the widest
    coverage wins. If the success target cannot be reached on this history, the
    best cost-clearing point is reported so live trading at least never picks a
    threshold whose edge is smaller than its own transaction costs. Not
    pretending the target was met is the whole point of measuring it.
    """
    unusable = {"achieved": False, "threshold": None, "success": None, "coverage": None,
                "signals": 0, "net_expected": None}
    if sweep is None or sweep.empty:
        return {**unusable, "reason": "No threshold could be evaluated."}

    usable = sweep.dropna(subset=["Success %"])
    usable = usable[usable["Signals"] >= minimum_trades]
    if usable.empty:
        return {**unusable, "reason": "Not enough out-of-sample signals to evaluate any threshold."}

    net = usable.get("Net expected %")
    if net is not None:
        cost_clearing = usable[net.fillna(-np.inf) > 0]
    else:
        cost_clearing = usable

    qualifying = usable[usable["Success %"] >= target_precision * 100]
    pool = qualifying if not qualifying.empty else cost_clearing
    if pool.empty:
        pool = usable
    best = pool.sort_values(["Coverage %", "Success %"], ascending=False).iloc[0]
    achieved = not qualifying.empty and best["Success %"] >= target_precision * 100
    net_expected = float(best["Net expected %"]) if "Net expected %" in usable.columns else None
    return {
        "achieved": bool(achieved),
        "threshold": float(best["Threshold"]),
        "success": float(best["Success %"]) / 100,
        "coverage": float(best["Coverage %"]) / 100,
        "signals": int(best["Signals"]),
        "net_expected": net_expected / 100 if net_expected is not None else None,
        "reason": (f"Target {target_precision:.0%} reached at threshold {best['Threshold']:.2f}."
                   if achieved else
                   f"Best cost-clearing point on this history is {best['Success %']:.1f}% at threshold "
                   f"{best['Threshold']:.2f} with {int(best['Signals'])} signals "
                   f"(net expected {net_expected:.2f}% per trade after costs)."
                   if net_expected is not None else
                   f"Best achievable on this history is {best['Success %']:.1f}% at threshold "
                   f"{best['Threshold']:.2f} with {int(best['Signals'])} signals."),
    }


def cross_validated_probabilities(out_of_fold: pd.DataFrame) -> pd.Series:
    """Calibrate each fold's scores using only the *other* folds.

    Fitting one calibrator on all out-of-fold scores and then scoring those same
    rows with it leaks the labels back in. Rotating the calibrator across folds
    keeps the reported success rate honest.
    """
    probabilities = pd.Series(np.nan, index=out_of_fold.index, dtype=float)
    for fold in out_of_fold["fold"].unique():
        test_part = out_of_fold[out_of_fold["fold"] == fold]
        train_part = out_of_fold[out_of_fold["fold"] != fold]
        calibrator = fit_calibrator(train_part["score"].to_numpy(), train_part["y"].to_numpy())
        if calibrator is None:
            probabilities.loc[test_part.index] = test_part["score"].to_numpy()
        else:
            probabilities.loc[test_part.index] = np.clip(calibrator.predict(test_part["score"].to_numpy()), 0.0, 1.0)
    return probabilities


def analyze_timeframe(df: pd.DataFrame, timeframe: str = "medium",
                      target_precision: float = DEFAULT_TARGET_PRECISION,
                      n_folds: int = N_FOLDS) -> dict:
    """Full classification / regression / forecasting evaluation for one timeframe.

    Returns the live-ready model bundle, an honest walk-forward success-rate
    measurement, the chosen operating threshold, and the diagnostics needed to
    see *why* the model performs the way it does.
    """
    horizon = TIMEFRAME_HORIZONS.get(timeframe, 15)
    result = {"available": False, "timeframe": timeframe, "horizon": horizon,
              "target_precision": target_precision, "models": None, "error": None}

    frame = _design_matrix(df, horizon)
    if frame is None or len(frame) < 150:
        result["error"] = f"Need at least 150 labelled bars; got {0 if frame is None else len(frame)}."
        return result

    out_of_fold = cross_val_scores(frame, horizon, n_folds)
    if out_of_fold.empty:
        result["error"] = "Walk-forward validation produced no folds (history too short for purged splits)."
        return result

    probabilities = cross_validated_probabilities(out_of_fold)
    valid = probabilities.notna()
    scores = probabilities[valid].to_numpy()
    labels = out_of_fold.loc[valid, "y"].to_numpy()
    forward = out_of_fold.loc[valid, "forward_return"].to_numpy()
    long_r = out_of_fold.loc[valid, "long_return"].to_numpy()
    short_r = out_of_fold.loc[valid, "short_return"].to_numpy()
    absolute = out_of_fold.loc[valid, "absolute_up"].to_numpy()

    diagnostics = classifier_diagnostics(scores, labels)
    try:
        from config import config as app_config
        cost_pct = float(getattr(app_config, "signal_round_trip_cost_pct", 0.2)) * 100
    except Exception:
        cost_pct = 0.2
    sweep = sweep_operating_points(scores, labels, forward, long_r, short_r, absolute,
                                   cost_pct=cost_pct)
    operating = select_operating_point(sweep, target_precision)
    reliability = reliability_table(scores, labels)

    # Final deployed model: fit on all labelled history, calibrate on all
    # out-of-fold scores (the standard way to prepare the served model).
    try:
        bundle = fit_bundle(frame)
    except ValueError as error:
        result["error"] = str(error)
        return result
    bundle.calibrator = fit_calibrator(out_of_fold["score"].to_numpy(), out_of_fold["y"].to_numpy())

    base_rate = float(labels.mean()) if labels.size else None
    result.update({
        "available": True,
        "models": bundle,
        "bars": len(frame),
        "folds": int(out_of_fold["fold"].nunique()),
        "validated_bars": int(labels.size),
        "base_rate": round(base_rate, 4) if base_rate is not None else None,
        "diagnostics": diagnostics,
        "sweep": sweep,
        "operating": operating,
        "reliability": reliability,
        # Backwards-compatible keys used by the dashboard panel.
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


def score_latest(df: pd.DataFrame, timeframe: str = "medium",
                 target_precision: float = DEFAULT_TARGET_PRECISION) -> dict:
    """Apply the ensemble to the most recent bar.

    Returns the calibrated probability, the direction, the forecast, and the
    threshold the probability has to clear for the signal to be emitted.
    """
    analyzed = analyze_timeframe(df, timeframe, target_precision)
    if not analyzed.get("available"):
        return {"available": False, "error": analyzed.get("error")}
    features = build_features(df)
    if features.empty:
        return {"available": False, "error": "No features could be built."}
    latest = features.iloc[[-1]]
    bundle = analyzed["models"]
    probability = float(bundle.predict_proba(latest)[0])
    forecast = float(bundle.forecast(latest)[0])
    return {
        "available": True,
        "probability": round(probability, 4),
        "forecast_return": round(forecast, 6),
        "threshold": analyzed["operating"].get("threshold") or 0.5,
        "analysis": analyzed,
    }