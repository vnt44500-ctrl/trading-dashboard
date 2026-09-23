"""Long-history walk-forward replay of the app's own signal engine.

Runs the app's committed per-symbol engine (``analysis.signal_models``) across
the top-50 liquid US stocks on ~10 years of daily data. Each timeframe is
judged at its own holding horizon (short=5, medium=15, long=40 bars) using the
engine's purged + embargoed expanding-window folds. Nothing here re-implements
features, labels, folds, calibration or the operating-point rule — the harness
calls the engine's own functions, so the numbers describe the app exactly as it
runs today.

What is reported per timeframe (pooled across all evaluated symbols):

* ROC-AUC of the out-of-fold calibrated probabilities (does the model rank
  outcomes better than chance anywhere in 10 years of data).
* Success rate of *fired signals* at each threshold against the fair
  horizon-matched base rates: the skill base (label base rate), the drift base
  (P(price moves up over the horizon)) and the barrier-trade base (how often
  the same ATR-barrier trade wins with no signal at all).
* Expected return per signal before and after round-trip costs.
* Coverage — how often the app would actually speak.
* The pooled operating point: does the app's 80% target get reached
  out-of-sample across 50 symbols and 10 years?
* Year-by-year stability of the pooled result (a breakdown of the same run,
  not a separate windowed test).
* Runtime health: seconds per symbol, skipped symbols, fold counts.

Usage::

    python -m analysis.walk_forward --symbols 6          # smoke test
    python -m analysis.walk_forward                       # full 50-symbol run
    python -m analysis.walk_forward --fresh               # ignore row cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running both as a module (python -m analysis.walk_forward) and as a
# plain script (python analysis\walk_forward.py) by putting the project root
# on sys.path before the app imports below.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis import signal_models as sm  # noqa: E402
from config import config  # noqa: E402

logger = logging.getLogger(__name__)

# Top-50 liquid US large caps (S&P 100 core by typical dollar volume).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "AVGO", "PG", "MA", "HD", "MRK", "PEP",
    "KO", "COST", "ABBV", "WMT", "MCD", "CSCO", "CRM", "BAC", "ADBE", "PFE",
    "CVX", "TMO", "ACN", "ABT", "NFLX", "DIS", "LIN", "AMD", "INTC", "WFC",
    "QCOM", "TXN", "DHR", "NKE", "CMCSA", "ORCL", "CAT", "VZ", "NEE", "AMGN",
]
TIMEFRAMES = ("short", "medium", "long")
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "data" / "cache" / "walk_forward"
ROWS_CACHE = CACHE_DIR / "oof_rows.pkl"
REPORT_PATH = _ROOT / "analysis" / "walk_forward_report.json"
PROGRESS_PATH = _ROOT / "analysis" / "walk_forward_progress.log"


def _log(message: str) -> None:
    """Progress line to stdout and a log file so long runs can be watched."""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _clean(value):
    """Make numpy/pandas values JSON-safe."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _with_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """pandas 3.0: read_csv(parse_dates=True) can leave a string index; coerce it.

    The engine tolerates a string index, but year grouping, log formatting and
    the year-by-year breakdown all need real timestamps.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    return frame


def fetch_history(symbol: str, period: str = "10y") -> pd.DataFrame | None:
    """10y daily history via the app's provider, cached on disk for resumability."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace(".", "_").replace("-", "_")
    cache_file = CACHE_DIR / f"{safe}_{period}.csv"
    if cache_file.exists():
        try:
            frame = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not frame.empty:
                return _with_datetime_index(frame)
        except Exception:
            pass
    from data.market_data import provider
    for attempt in range(2):
        try:
            frame = provider.get_history(symbol, period=period)
            if frame is not None and not frame.empty and "close" in frame.columns:
                frame.to_csv(cache_file)
                return _with_datetime_index(frame)
        except Exception as error:
            logger.warning("History fetch %s attempt %d failed: %s", symbol, attempt + 1, error)
            time.sleep(3 * (attempt + 1))
    _log(f"FETCH-FAIL {symbol}: no usable {period} history; symbol skipped")
    return None


def seed_benchmark(period: str = "10y") -> None:
    """Give the engine's regime features the full test-span SPY context.

    The engine's benchmark cache normally holds ~5y of SPY; seeding it with the
    replay span keeps ``mkt_*``/``rel_str_*``/``beta_60`` defined across the
    whole 10 years. The features stay causal (trailing windows only).
    """
    spy = fetch_history("SPY", period=period)
    if spy is not None:
        sm._BENCHMARK_CACHE["SPY"] = spy
        _log(f"benchmark SPY seeded: {len(spy)} bars, {spy.index[0]:%Y-%m} to {spy.index[-1]:%Y-%m}")
    else:
        _log("benchmark SPY unavailable; regime features will be median-filled by the engine")


def prepare_symbol(frame: pd.DataFrame):
    """Enrichment + causal features once per symbol, shared by all timeframes."""
    try:
        if frame is None or frame.empty or len(frame) < 400:
            raise ValueError("history too short")
        enriched = sm._enrich(frame)
        features = sm.build_features(enriched)
        if features.empty:
            raise ValueError("no features could be built")
        return enriched, features
    except Exception as error:
        _log(f"PREP-FAIL: {error}")
        return None


def evaluate_symbol(symbol: str, prepared, timeframes) -> list:
    """Engine-native out-of-fold rows for each timeframe of one symbol."""
    enriched, features = prepared
    entries = []
    for timeframe in timeframes:
        horizon = sm.TIMEFRAME_HORIZONS[timeframe]
        try:
            labels = sm.build_labels(enriched, horizon)
            if labels.direction.dropna().empty:
                continue
            frame = features.copy()
            frame["y_relative"] = labels.relative_direction
            frame["y_direction"] = labels.direction
            frame["y_return"] = labels.forward_return
            frame["y_long"] = labels.long_return
            frame["y_short"] = labels.short_return
            # Engine parity with _design_matrix: carry features forward across
            # gaps but never invent a label, then drop rows without a usable
            # target so fit_bundle never sees a NaN label (which made every
            # fold raise and be silently swallowed, producing zero rows).
            frame[sm.FEATURES] = frame[sm.FEATURES].ffill()
            frame = frame.dropna(subset=["y_relative", "y_direction", "y_return"])
            if frame.empty:
                continue
            oof = sm.cross_val_scores(frame, horizon)
        except Exception as error:
            _log(f"TF-FAIL {symbol}/{timeframe}: {error}")
            continue
        if oof.empty:
            continue
        oof = oof.dropna(subset=["y"])
        if oof.empty:
            continue
        entries.append({"timeframe": timeframe, "horizon": horizon, "oof": oof,
                        "folds": int(oof["fold"].nunique())})
    return entries


def run_replay(symbols: list, period: str = "10y", timeframes=TIMEFRAMES) -> pd.DataFrame:
    """Fetch + replay every symbol; returns one master out-of-fold table."""
    seed_benchmark(period)
    started = time.time()
    pieces, used, skipped = [], [], []
    for number, symbol in enumerate(symbols, start=1):
        t0 = time.time()
        history = fetch_history(symbol, period=period)
        prepared = prepare_symbol(history)
        if prepared is None:
            skipped.append(symbol)
            continue
        entries = evaluate_symbol(symbol, prepared, timeframes)
        if not entries:
            skipped.append(symbol)
            continue
        for entry in entries:
            oof = entry["oof"]
            pieces.append(oof.assign(
                symbol=symbol, timeframe=entry["timeframe"], horizon=entry["horizon"],
                date=oof.index, year=oof.index.year))
        used.append(symbol)
        _log(f"({number}/{len(symbols)}) {symbol}: {len(entries)} timeframes, "
             f"{time.time() - t0:.1f}s")
    if not pieces:
        return pd.DataFrame()
    # Stack to a UNIQUE row index. The same calendar dates repeat across
    # symbols, and a duplicated datetime index makes every ``.loc`` row
    # selection in the engine expand rows and misalign (pandas 3.0
    # list-indexer error). The panel model does the same reset in
    # ``build_panel``. Dates ride along in the ``date``/``year`` columns.
    master = pd.concat(pieces).sort_values("date", kind="stable").reset_index(drop=True)
    master.attrs = {"used": used, "skipped": skipped, "seconds": round(time.time() - started, 1)}
    return master


def year_breakdown(valid_group: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The same run split by calendar year — does one lucky stretch carry the result?

    ``valid_group`` is the timeframe's rows with calibrated probabilities
    attached (column ``prob``). Signal firing uses the pooled operating
    threshold, exactly as the live app would have spoken at that bar.
    """
    fired = (valid_group["prob"] >= threshold) | (valid_group["prob"] <= 1 - threshold)
    up = valid_group["prob"] >= 0.5
    realised = pd.Series(
        np.where(up, valid_group["long_return"], valid_group["short_return"]),
        index=valid_group.index,
    )
    rows = []
    for year, part in valid_group.groupby("year"):
        mask = fired.loc[part.index]
        row = {"Year": int(year), "OOF bars": int(len(part)), "Signals": int(mask.sum()),
               "Drift base %": round(float(part["absolute_up"].mean()) * 100, 1)}
        if mask.any():
            sel = part.index[mask]
            skill = float((up.loc[sel] == (part.loc[sel, "y"] == 1)).mean()) * 100
            row["Skill success %"] = round(skill, 1)
            row["Trade success %"] = round(float((realised.loc[sel] > 0).mean()) * 100, 1)
            row["Skill edge pp"] = round(skill - float(part["y"].mean()) * 100, 1)
        else:
            row["Skill success %"] = row["Trade success %"] = row["Skill edge pp"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_timeframe(group: pd.DataFrame,
                       target_precision: float = sm.DEFAULT_TARGET_PRECISION) -> dict:
    """Pooled engine-native evaluation of one timeframe across all symbols.

    Mirrors ``signal_models.analyze_timeframe`` on the pooled out-of-fold rows:
    rotating-fold calibration, diagnostics, the full threshold sweep, the
    operating-point rule — then the fair no-signal bases on the *same* rows and
    a year-by-year split of the pooled result.
    """
    probs = sm.cross_validated_probabilities(group)
    valid = probs.notna()
    scores = probs[valid].to_numpy()
    labels = group.loc[valid, "y"].to_numpy()
    absolute = group.loc[valid, "absolute_up"].to_numpy()
    forward = group.loc[valid, "forward_return"].to_numpy()
    long_r = group.loc[valid, "long_return"].to_numpy()
    short_r = group.loc[valid, "short_return"].to_numpy()

    diagnostics = sm.classifier_diagnostics(scores, labels)
    sweep = sm.sweep_operating_points(scores, labels, forward, long_r, short_r, absolute)
    operating = sm.select_operating_point(sweep, target_precision)

    # Fair "no signal" bases, pooled over exactly the rows the signals were judged on.
    skill_base = float(labels.mean())        # always guess "up vs its own baseline"
    drift_base = float(absolute.mean())      # always guess "price goes up"
    trade_base = float((long_r > 0).mean())  # always-long ATR-barrier trade

    threshold = float(operating.get("threshold") or 0.60)
    fired = (scores >= threshold) | (scores <= 1 - threshold)
    up = scores >= 0.5
    realised = np.where(up, long_r, short_r)
    round_trip_cost = 0.001  # 10 bps round trip, liquid large caps

    result = {
        "rows": int(labels.size),
        "folds": int(group["fold"].nunique()),
        "symbols": int(group["symbol"].nunique()),
        "roc_auc": diagnostics.get("roc_auc"),
        "skill_base": round(skill_base, 4),
        "drift_base": round(drift_base, 4),
        "trade_base": round(trade_base, 4),
        "operating": operating,
        "threshold": threshold,
    }
    if fired.any():
        skill_success = float((up[fired] == (labels[fired] == 1)).mean())
        trade_success = float((realised[fired] > 0).mean())
        result.update({
            "signals": int(fired.sum()),
            "coverage": round(float(fired.mean()), 4),
            "skill_success": round(skill_success, 4),
            "trade_success": round(trade_success, 4),
            "directional_success": round(float((up[fired] == (absolute[fired] == 1)).mean()), 4),
            "expected_return_pct": round(float(forward[fired].mean() * 100), 3),
            "net_expected_return_pct": round(float((forward[fired].mean() - round_trip_cost) * 100), 3),
            "avg_trade_r": round(float(realised[fired].mean()), 3),
            "skill_edge": round(skill_success - skill_base, 4),
            "trade_edge": round(trade_success - trade_base, 4),
        })
    else:
        result.update({"signals": 0, "coverage": 0.0, "skill_success": None,
                       "trade_success": None, "directional_success": None,
                       "expected_return_pct": None, "net_expected_return_pct": None,
                       "avg_trade_r": None, "skill_edge": None, "trade_edge": None})
    result["sweep"] = sweep
    result["years"] = year_breakdown(group.loc[valid].assign(prob=probs[valid]), threshold)
    return result


def build_report(master: pd.DataFrame, target_precision: float = sm.DEFAULT_TARGET_PRECISION,
                 timeframes=TIMEFRAMES, period: str = "10y") -> dict:
    """Evaluate every timeframe on the pooled out-of-fold rows and assemble the report."""
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "analysis.signal_models (the app's committed engine, unchanged)",
        "period": period,
        "target_precision": target_precision,
        "symbols_used": list(master.attrs.get("used", [])),
        "symbols_skipped": list(master.attrs.get("skipped", [])),
        "replay_seconds": master.attrs.get("seconds"),
        "oof_rows": int(len(master)),
        "timeframes": {},
    }
    for timeframe in timeframes:
        group = master[master["timeframe"] == timeframe]
        if group.empty:
            report["timeframes"][timeframe] = {"error": "no out-of-fold rows for this timeframe"}
            continue
        try:
            result = evaluate_timeframe(group, target_precision)
        except Exception as error:
            report["timeframes"][timeframe] = {"error": f"evaluation failed: {error}"}
            continue
        sweep = result.pop("sweep", None)
        years = result.pop("years", None)
        headline = []
        if sweep is not None and not sweep.empty:
            picks = sweep[sweep["Threshold"].isin(THRESHOLDS) & sweep["Success %"].notna()]
            headline = picks.to_dict("records")
        result["horizon"] = int(group["horizon"].iloc[0])
        result["headline_thresholds"] = headline
        result["years"] = years.to_dict("records") if years is not None else []
        report["timeframes"][timeframe] = result

    verdicts = {}
    for timeframe, res in report["timeframes"].items():
        if res.get("error"):
            verdicts[timeframe] = res["error"]
            continue
        if not res.get("signals"):
            verdicts[timeframe] = "No signals fired at the operating threshold."
            continue
        verdicts[timeframe] = (
            f"{res['signals']} signals ({res['coverage'] * 100:.1f}% of bars): "
            f"skill {res['skill_success'] * 100:.1f}% vs {res['skill_base'] * 100:.1f}% base "
            f"({res['skill_edge'] * 100:+.1f} pp); trade {res['trade_success'] * 100:.1f}% vs "
            f"{res['trade_base'] * 100:.1f}% base ({res['trade_edge'] * 100:+.1f} pp); "
            f"net expected {res['net_expected_return_pct']:+.2f}% per trade after costs."
        )
    report["verdicts"] = verdicts
    return report


def print_summary(report: dict) -> None:
    """Plain-language readout of the report, one block per timeframe."""
    skipped = ", ".join(report.get("symbols_skipped", []))
    _log("=" * 74)
    _log("WALK-FORWARD REPLAY RESULT — the app's own engine, out-of-sample folds only")
    _log(f"  history: {report['period']} daily | symbols evaluated: {len(report['symbols_used'])}"
         + (f" | skipped: {skipped}" if skipped else ""))
    _log(f"  out-of-fold bars: {report['oof_rows']:,} | replay took {report['replay_seconds']}s")
    for timeframe, res in report["timeframes"].items():
        if res.get("error"):
            _log(f"[{timeframe}] {res['error']}")
            continue
        roc = res.get("roc_auc")
        _log("-" * 74)
        _log(f"[{timeframe} @ {res['horizon']}-bar hold]  ROC-AUC: "
             f"{'n/a' if roc is None else f'{roc:.3f}'}  (0.5 = no ranking skill)")
        op = res.get("operating") or {}
        _log(f"  operating point: {op.get('reason')}")
        if res.get("signals"):
            _log(f"  fired signals : {res['signals']:,} ({res['coverage'] * 100:.1f}% coverage)")
            _log(f"  skill success : {res['skill_success'] * 100:.1f}%  vs base "
                 f"{res['skill_base'] * 100:.1f}%  -> edge {res['skill_edge'] * 100:+.1f} pp")
            _log(f"  trade success : {res['trade_success'] * 100:.1f}%  vs base "
                 f"{res['trade_base'] * 100:.1f}%  -> edge {res['trade_edge'] * 100:+.1f} pp")
            _log(f"  expected move : {res['expected_return_pct']:+.2f}% per trade, "
                 f"{res['net_expected_return_pct']:+.2f}% after ~10 bps round trip")
            years = res.get("years") or []
            edges = [(row["Year"], row["Skill edge pp"]) for row in years
                     if isinstance(row.get("Skill edge pp"), (int, float))
                     and not np.isnan(row["Skill edge pp"])]
            if edges:
                best = max(edges, key=lambda item: item[1])
                worst = min(edges, key=lambda item: item[1])
                _log(f"  year spread   : best {best[0]} ({best[1]:+.1f} pp), "
                     f"worst {worst[0]} ({worst[1]:+.1f} pp), {len(edges)} years with signals")
        else:
            _log("  no signals fired at the operating threshold on this pooled history.")
    _log("=" * 74)
    for timeframe, verdict in report.get("verdicts", {}).items():
        _log(f"VERDICT {timeframe}: {verdict}")
    _log("=" * 74)


def save_report(report: dict, path: Path = REPORT_PATH) -> Path:
    """Persist the JSON report (numpy-safe) for the dashboard and future reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    return path


def _meta_path() -> Path:
    return ROWS_CACHE.with_suffix(".meta.json")


def _save_rows(master: pd.DataFrame) -> None:
    """Cache the expensive out-of-fold replay so re-runs skip straight to evaluation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    master.to_pickle(ROWS_CACHE)
    meta = {
        "used": list(master.attrs.get("used", [])),
        "skipped": list(master.attrs.get("skipped", [])),
        "seconds": master.attrs.get("seconds"),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_rows() -> pd.DataFrame | None:
    if not ROWS_CACHE.exists():
        return None
    try:
        master = pd.read_pickle(ROWS_CACHE)
        needed = {"symbol", "timeframe", "horizon", "fold", "y", "absolute_up",
                  "forward_return", "long_return", "short_return", "score", "forecast"}
        if master.empty or not needed.issubset(master.columns):
            return None
        if _meta_path().exists():
            meta = json.loads(_meta_path().read_text(encoding="utf-8"))
            master.attrs = {"used": meta.get("used", []),
                            "skipped": meta.get("skipped", []),
                            "seconds": meta.get("seconds")}
        return master
    except Exception as error:
        _log(f"row cache unreadable ({error}); replaying from scratch")
        return None


def main() -> None:
    """Resumable CLI: replay (cached), evaluate every timeframe, print + save the report."""
    parser = argparse.ArgumentParser(description="Long-history walk-forward replay of the app's signal engine")
    parser.add_argument("--symbols", type=int, default=0,
                        help="limit to the first N universe symbols (smoke test)")
    parser.add_argument("--period", default="10y", help="history span to replay (default 10y)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the cached out-of-fold rows and replay everything")
    parser.add_argument("--target", type=float, default=sm.DEFAULT_TARGET_PRECISION,
                        help="operating-point precision target (default from the app config)")
    args = parser.parse_args()

    symbols = UNIVERSE[: args.symbols] if args.symbols else list(UNIVERSE)
    full_run = not args.symbols  # only the full universe shares the resumable row cache
    _log(f"walk-forward replay: {len(symbols)} symbols, {args.period} daily, "
         f"timeframes {', '.join(TIMEFRAMES)}")

    master = _load_rows() if (full_run and not args.fresh) else None
    if master is not None:
        saved = "?"
        if _meta_path().exists():
            try:
                saved = json.loads(_meta_path().read_text(encoding="utf-8")).get("saved", "?")
            except Exception:
                pass
        _log(f"using cached out-of-fold rows: {len(master):,} rows across "
             f"{master['symbol'].nunique()} symbols (saved {saved}); pass --fresh to replay")
    else:
        master = run_replay(symbols, period=args.period)
        if master.empty:
            _log("REPLAY FAILED: no out-of-fold rows were produced; nothing to evaluate.")
            return
        if full_run:
            _save_rows(master)
            _log(f"replay cached: {len(master):,} rows, {master['symbol'].nunique()} symbols, "
                 f"{master.attrs.get('seconds')}s")
        else:
            _log(f"smoke replay done: {len(master):,} rows (not cached; cache is full-run only)")

    report = build_report(master, target_precision=args.target, period=args.period)
    print_summary(report)
    path = save_report(report)
    _log(f"report saved: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    main()
