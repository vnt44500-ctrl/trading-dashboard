"""Main dashboard render function for the Streamlit app."""
import threading
from datetime import date, datetime

import pandas as pd
import streamlit as st

from data.market_data import provider
from data.news_data import news_provider
from analysis import indicators as ind
from analysis.fundamentals import analyze_fundamentals
from analysis.sentiment import analyze_news_sentiment
from signals.engine import generate_signals, technical_signal
from backtest.engine import backtest, walk_forward_backtest
from broker.adapter import get_broker
from ui.charts import price_chart, equity_chart, gauge_chart
from core.assets import search_assets, get_asset
from signals.tracker import tracker
from analysis.market_scanner import scan_market, scanned_at
from analysis.regime import detect_regime
from analysis.ml_validation import (regime_gate_for, score_latest_bar,
                                    score_latest_with_gate, validate_ml_ensemble)
from analysis import panel_models
from analysis.strategy_validation import indicator_catalog, strategy_catalog
from data.scan_store import load_scan, save_scan
from notify import notify_signals_async


@st.cache_data(ttl=3600, show_spinner=False)
def _validate_cached(history, timeframe: str, allow_short: bool):
    """One leakage-safe walk-forward validation per timeframe, cached 1h."""
    result = validate_ml_ensemble(history, timeframe=timeframe, allow_short=allow_short)
    if result.get("available") and result.get("models") is not None:
        probability, gate = score_latest_with_gate(history, result["models"])
        result["live_probability"] = probability
        result["gate"] = gate
    return result


PANEL_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "XOM", "UNH", "SPY"]


@st.cache_resource
def _panel_cache():
    """Process-wide pooled-model state shared by every session.

    ``streamlit.cache_resource`` keeps exactly one instance per server process,
    so the panel histories are fetched and the model trained once and reused
    across reruns, sessions, and timeframes. All heavy work happens on a
    background thread; the UI thread only reads the state dict.
    """
    return {"fetching": False, "ready": False, "error": None,
            "symbols": [], "prepared": None, "results": {}}


def _fetch_panel_frames(symbols, period: str = "5y") -> dict:
    frames = {}
    for symbol in symbols:
        try:
            frame = provider.get_history(symbol, period=period)
        except Exception:
            frame = pd.DataFrame()
        if frame is not None and len(frame) >= 400:
            frames[symbol] = frame
    return frames


def _panel_ensure() -> dict:
    """Kick off the pooled-panel build once, then report current status."""
    state = _panel_cache()
    if not state.get("fetching") and not state.get("ready") and state.get("error") is None:

        def _worker():
            try:
                frames = _fetch_panel_frames(PANEL_UNIVERSE)
                if len(frames) < panel_models.MIN_PANEL_SYMBOLS:
                    state["error"] = "Not enough symbols returned usable history for the panel model."
                    return
                state["symbols"] = sorted(frames)
                prepared = panel_models.prepare_panel_assets(frames)
                state["prepared"] = prepared
                state["ready"] = True
                for tf in ("short", "medium", "long"):
                    try:
                        state["results"][tf] = panel_models.analyze_panel(prepared, timeframe=tf)
                    except Exception as error:
                        state["results"][tf] = {"available": False, "error": str(error)}
            except Exception as error:
                state["error"] = str(error)
            finally:
                state["fetching"] = False

        state["fetching"] = True
        threading.Thread(target=_worker, daemon=True).start()
    return state


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_quote(symbol: str):
    return provider.get_quote(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_history(symbol: str, period: str):
    return provider.get_history(symbol, period=period)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_news(symbol: str):
    return news_provider.get_news(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def _search_assets_cached(query: str):
    return search_assets(query)


def _get_indicator_reasoning(ind_summary: dict):
    indicator_reasons = []
    if ind_summary.get("rsi") is not None:
        rsi = ind_summary["rsi"]
        if rsi >= 70:
            indicator_reasons.append({"name": "RSI", "signal": "Overbought", "strategy": "Wait for momentum exhaustion or bearish rejection before entering short or tightening stops.", "value": rsi})
        elif rsi <= 30:
            indicator_reasons.append({"name": "RSI", "signal": "Oversold", "strategy": "Look for reversal confirmation and bullish continuation near support.", "value": rsi})
        else:
            indicator_reasons.append({"name": "RSI", "signal": "Balanced momentum", "strategy": "Trend strength is neutral; wait for confluence from price, VWAP, and volume.", "value": rsi})

    if ind_summary.get("macd") is not None and ind_summary.get("macd_signal") is not None:
        diff = ind_summary["macd"] - ind_summary["macd_signal"]
        if diff > 0:
            indicator_reasons.append({"name": "MACD", "signal": "Bullish crossover", "strategy": "Momentum is constructive; favor continuation setups while price holds above short-term support.", "value": diff})
        else:
            indicator_reasons.append({"name": "MACD", "signal": "Bearish crossover", "strategy": "Momentum favors downside; reduce risk and wait for a defensive structure.", "value": diff})

    if ind_summary.get("smart_vwap") is not None:
        current = ind_summary.get("smart_vwap")
        indicator_reasons.append({"name": "Smart VWAP", "signal": "Volume-weighted trend anchor", "strategy": "Use the Smart VWAP as a dynamic benchmark; price above it supports bullish continuation, below it supports defensive positioning.", "value": current})

    if ind_summary.get("pineify_bias") is not None:
        bias = ind_summary["pineify_bias"]
        indicator_reasons.append({"name": "Pineify Bias", "signal": "Bullish" if bias >= 0 else "Bearish", "strategy": "Trend direction is aligned with the EMA/SMA gap; use this as a filter for confirming long or short entries.", "value": bias})

    return indicator_reasons


def _render_signal_expander(signal):
    st.subheader(f"{signal.timeframe.title()} signal")
    st.markdown(f"**Action:** {signal.action}  |  **Confidence:** {signal.confidence * 100:.1f}%")
    st.caption(f"Model: {signal.model_version} | Estimated risk: {signal.risk_score:.1f}/100 | Risk-based size: {signal.position_size:.4f} units")
    st.write(signal.strategy_summary)
    st.caption(signal.strategy_name)
    with st.expander("Detailed reasoning"):
        for reason in signal.reasons:
            st.markdown(f"- {reason}")


def _render_market_scan(results: dict, scan_time: str):
    st.header("Market opportunity scanner")
    st.caption("Scanner actions are technical-only research signals. The instrument dashboard adds fundamentals, news, ML validation, and live eligibility, so its action may differ. Crypto rows use Binance public data when available; other instruments use Yahoo Finance.")
    st.caption(f"Last scan: {scan_time}")
    for number, name in enumerate(["Stocks", "Cryptos", "Currencies", "Options", "Commodities"], start=1):
        table = results.get(name)
        st.subheader(f"{number}. {name}")
        if table is None or table.empty:
            st.info(f"No {name.lower()} data was available in this scan.")
        else:
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Signal confidence": st.column_config.ProgressColumn("Signal confidence", min_value=0, max_value=100, format="%.1f"),
                    "Risk": st.column_config.ProgressColumn("Estimated risk", min_value=0, max_value=100, format="%.1f"),
                    "Rank score": st.column_config.NumberColumn("Rank score", format="%.1f"),
                    "Technical score": st.column_config.NumberColumn("Technical score", format="%.1f"),
                    "Price": st.column_config.NumberColumn("Price", format="%.6f"),
                    "ATR %": st.column_config.NumberColumn("ATR %", format="%.2f"),
                    "Volatility %": st.column_config.NumberColumn("Volatility %", format="%.2f"),
                    "VaR %": st.column_config.NumberColumn("VaR %", format="%.2f"),
                    "Spread %": st.column_config.NumberColumn("Spread %", format="%.2f"),
                    "Implied Volatility %": st.column_config.NumberColumn("Implied volatility %", format="%.2f"),
                    "Historical success %": st.column_config.NumberColumn("Historical success %", format="%.1f"),
                    "Historical signals": st.column_config.NumberColumn("Historical signals", format="%d"),
                    "Fundamental score": st.column_config.NumberColumn("Fundamental score", format="%.1f"),
                    "Fundamental data quality": st.column_config.NumberColumn("Fundamental data quality", format="%.1f"),
                },
            )


def _run_market_scan():
    st.session_state.market_scan_results = {}
    st.session_state.market_scan_status = "running"
    st.session_state.market_scan_error = None
    progress = st.progress(0, text="Scanning market candidates...")
    try:
        partial_results = {}

        def save_group(name, table):
            partial_results[name] = table
            st.session_state.market_scan_results = dict(partial_results)
            st.session_state.market_scan_time = scanned_at()

        results = scan_market(provider, ind, technical_signal, progress.progress, save_group)
        st.session_state.market_scan_results = results
        st.session_state.market_scan_time = scanned_at()
        save_scan(results)
        st.session_state.market_scan_status = "complete"
    except Exception as error:
        st.session_state.market_scan_status = "failed"
        st.session_state.market_scan_error = str(error)
    finally:
        st.session_state.market_scan_pause_refresh = True
        progress.empty()


def render_dashboard():
    st.title("📈 Trading Signals Dashboard")
    st.caption("Fundamental + Technical + News sentiment → actionable BUY / SELL / HOLD signals.")
    if "market_scan_results" not in st.session_state:
        stored_scan = load_scan()
        if stored_scan:
            st.session_state.market_scan_results, st.session_state.market_scan_time = stored_scan
    scan_now = False
    scheduled_scan = False
    with st.sidebar:
        st.title("📊 Trading Signals")
        st.markdown("Real-time analysis across stocks, forex, commodities & crypto.")

        asset_class = st.selectbox(
            "Asset class",
            ["All", "Stock", "Forex", "Commodity", "Crypto", "Option"],
        )

        query = st.text_input(
            "Search symbol or name",
            placeholder="Type at least 3 characters: AAPL, BTC, EUR, GOLD",
            key="asset_query",
        )
        if st.button(
            "Refresh market data",
            icon=":material/refresh:",
            width="stretch",
            help="Clear cached quotes, history, news, and evidence models, then refetch.",
        ):
            st.cache_data.clear()
            st.rerun()
        if query and len(query.strip()) < 3:
            st.caption("Type at least 3 characters to load market suggestions.")
        if query:
            results = _search_assets_cached(query)
        else:
            results = _search_assets_cached("")

        if asset_class != "All":
            results = [(s, m) for s, m in results if m["class"].lower() == asset_class.lower()]

        symbol = None
        if results:
            labels = [f"{s} — {m['name']}" for s, m in results]
            selection = st.selectbox(
                "Suggestions",
                labels,
                index=0,
                key="asset_suggestion",
                help="Search by symbol or name. Exact and local matches appear first.",
            )
            if selection:
                symbol = selection.split(" — ")[0]
        elif query.strip():
            symbol = query.strip().upper()
            st.caption(f"Using typed symbol: {symbol}")

        st.divider()
        st.markdown("**Analysis settings**")
        period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
        allow_short = st.checkbox(
            "Allow short positions (SELL opens short)",
            value=True,
            help="When enabled, a SELL opens or reverses into a short position. When disabled, SELL can only close a long position.",
        )

        st.divider()
        st.markdown("**Whole-market scanner**")
        scan_now = st.button("Run market scan", type="primary", width="stretch")
        schedule_enabled = st.checkbox("Run scan at a specific time", value=False)
        scheduled_time = st.time_input("Scheduled scan time", value=datetime.now().time(), disabled=not schedule_enabled)
        if schedule_enabled and scheduled_time is not None:
            schedule_key = f"{st.date_input('Scheduled date', value=date.today())} {scheduled_time}"
            scheduled_scan = str(datetime.now())[:16] >= schedule_key[:16] and st.session_state.get("last_schedule_key") != schedule_key
            if scheduled_scan:
                st.session_state.last_schedule_key = schedule_key

    if scan_now or scheduled_scan:
        st.session_state.market_scan_requested = True
        st.session_state.market_scan_pause_refresh = True

    if st.session_state.pop("market_scan_requested", False):
        with st.spinner("Scanning stocks, cryptos, currencies, options, and commodities..."):
            _run_market_scan()

    main_scan_now = st.button("Run top-20 market scan", type="primary", width="stretch")
    if main_scan_now:
        st.session_state.market_scan_requested = True
        st.session_state.market_scan_pause_refresh = True
        st.rerun()

    if "market_scan_results" in st.session_state:
        _render_market_scan(st.session_state.market_scan_results, st.session_state.get("market_scan_time", "unknown"))
        if st.session_state.get("market_scan_status") == "running":
            st.info("The scan is still running. Completed categories are shown above; options may be unavailable from the current data provider.")
        st.divider()
    elif st.session_state.get("market_scan_status") == "failed":
        with st.container(border=True):
            st.subheader("Market opportunity scanner")
            st.error(f"The market scan could not complete: {st.session_state.get('market_scan_error', 'unknown error')}")
            st.info("Try again after checking the network connection or Yahoo Finance availability.")
    else:
        with st.container(border=True):
            st.subheader("Market opportunity scanner")
            st.write("Run the scanner to rank the top 20 available stocks, cryptos, currencies, options, and commodities by composite opportunity and estimated risk.")
            st.info("No market scan has been run yet. Use the button above or the scanner controls in the sidebar.")

    if not symbol:
        st.info("👈 Use the sidebar to search and select an instrument to begin.")
        return

    asset = get_asset(symbol)

    with st.spinner(f"Fetching data for {asset.symbol}..."):
        quote = _fetch_quote(asset.symbol)
        hist = _fetch_history(asset.symbol, period)
        validation_hist = hist if period == "5y" else _fetch_history(asset.symbol, "5y")
        news = _fetch_news(asset.symbol)

    if not quote.get("price"):
        st.error(f"Could not fetch data for {asset.symbol}. It may be invalid or temporarily unavailable.")
        return

    ind_df = ind.compute_indicators(hist)
    ind_summary = ind.last_indicators(ind_df)
    regime = detect_regime(hist)
    ind_summary["regime"] = regime["name"]
    ind_summary["regime_reason"] = regime["reason"]
    fundamental = analyze_fundamentals(asset, quote)
    sentiment = analyze_news_sentiment(news)

    panel_state = _panel_ensure()
    validations = {
        tf: _validate_cached(validation_hist, tf, allow_short)
        for tf in ("short", "medium", "long")
    }
    evidence_compare = {}
    scope_pref = st.session_state.get("evidence_scope", "Auto (best measured)")
    if panel_state.get("ready"):
        for tf in ("short", "medium", "long"):
            panel_result = panel_state["results"].get(tf)
            if not isinstance(panel_result, dict) or not panel_result.get("available"):
                continue  # panel still training this timeframe; keep single-asset evidence
            panel_result = dict(panel_result)
            panel_result["live_probability"] = (panel_result.get("live_probabilities") or {}).get(asset.symbol)
            # Panel evidence has no per-asset fitted bundle, but the regime
            # gate depends only on trailing market state — attach it so the
            # engine's sizing/blocking rule applies to panel-chosen evidence too.
            if "gate" not in panel_result:
                panel_result["gate"] = regime_gate_for(validation_hist)
            single = validations.get(tf) or {}
            single_success = (single.get("operating") or {}).get("success") or 0.0
            panel_success = (panel_result.get("operating") or {}).get("success") or 0.0
            if scope_pref.startswith("Panel"):
                chosen = "panel"
            elif scope_pref == "Single asset":
                chosen = "single"
            else:
                # Auto: whichever evidence measured better out-of-sample wins.
                prefer_panel = (
                    (panel_result.get("eligible") and not single.get("eligible"))
                    or (panel_result.get("eligible") == single.get("eligible")
                        and panel_success > single_success)
                )
                chosen = "panel" if prefer_panel else "single"
            evidence_compare[tf] = {
                "single_success": single_success,
                "panel_success": panel_success,
                "single_eligible": bool(single.get("eligible")),
                "panel_eligible": bool(panel_result.get("eligible")),
                "chosen": chosen,
                "panel_auc": panel_result.get("diagnostics", {}).get("roc_auc"),
                "panel_symbols": len(panel_result.get("symbols") or []),
            }
            if chosen == "panel":
                validations[tf] = panel_result
            else:
                # Skill-weighted hybrid: blend the pooled model's live
                # probability into the single-asset evidence with weight
                # proportional to its demonstrated out-of-sample edge
                # (w = 0 until the panel AUC beats 0.5, capped at 0.5).
                panel_auc = panel_result.get("diagnostics", {}).get("roc_auc")
                weight = 0.0 if panel_auc is None else max(0.0, min(0.5, (panel_auc - 0.5) * 2.0))
                evidence_compare[tf]["blend_weight"] = round(weight, 3)
                single_prob = single.get("live_probability")
                panel_prob = panel_result.get("live_probability")
                if weight > 0 and single_prob is not None and panel_prob is not None:
                    blended = (1.0 - weight) * single_prob + weight * panel_prob
                    validations[tf]["live_probability"] = blended
                    evidence_compare[tf]["single_probability"] = single_prob
                    evidence_compare[tf]["panel_probability"] = panel_prob
                    evidence_compare[tf]["blended_probability"] = blended
    signals_dict = generate_signals(asset, quote, ind_summary, fundamental, sentiment, validation=validations)
    # Side-effect-free SMS fan-out: runs on a daemon thread, no-ops entirely
    # unless the operator opted in via NOTIFY_SMS_* env vars. It never changes
    # any signal, score, or action — it only reads the finished Signal objects.
    try:
        notify_signals_async(signals_dict)
    except Exception:
        pass  # alerting must never disturb the render path
    sig_short = signals_dict["short"]
    sig_med = signals_dict["medium"]
    sig_long = signals_dict["long"]

    col1, col2, col3 = st.columns([2, 1, 1])
    price = quote.get("price")
    prev_close = quote.get("prev_close")
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    with col1:
        st.markdown(f"## {asset.name} ({asset.symbol})")
        st.markdown(f"**{asset.asset_class.title()}** · {asset.exchange or ''}")
        st.caption(f"Last updated: {quote.get('last_updated', 'N/A')} (source: {quote.get('source', 'Yahoo Finance')})")
        delta = f"{change_pct:+.2f}%" if change_pct else ""
        st.metric("Last Price", f"{price:,.4f}" if price else "N/A", delta)

    with col2:
        st.metric("Var (daily)", f"{ind_summary.get('var', 0) * 100:.2f}%")
        st.metric("RSI", f"{ind_summary.get('rsi', 0):.1f}")

    with col3:
        st.metric("Smart VWAP", f"{ind_summary.get('smart_vwap', 0):,.4f}")
        st.metric("Pineify Bias", f"{ind_summary.get('pineify_bias', 0):,.4f}")
        st.caption(f"Regime: {regime['name']} ({regime['confidence'] * 100:.0f}% context confidence)")

    st.divider()

    st.markdown("### 🎯 Multi-Timeframe Signals")
    sc1, sc2, sc3 = st.columns(3)

    def render_sig_card(s, col):
        with col:
            color = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#ffb300"}[s.action]
            st.markdown(f"#### {s.timeframe.title()} Term")
            st.caption(f"⏱ {s.expected_duration}")
            display_action = s.action if s.live_eligible or s.raw_action == "HOLD" else f"{s.action} (research {s.raw_action})"
            st.markdown(
                f"<div style='background:{color};padding:12px;border-radius:8px;text-align:center;"
                f"font-size:20px;font-weight:bold;color:white;margin-bottom:10px;'>{display_action} "
                f"({s.confidence*100:.0f}% Conf)</div>",
                unsafe_allow_html=True,
            )
            with st.expander("AI Strategy Logic"):
                st.write(s.strategy_summary)
                st.caption(s.strategy_name)
                for reason in s.reasons:
                    st.markdown(f"- {reason}")

    render_sig_card(sig_short, sc1)
    render_sig_card(sig_med, sc2)
    render_sig_card(sig_long, sc3)

    st.divider()
    st.plotly_chart(price_chart(ind_df, asset.symbol, ind_summary), width="stretch")

    tabs = st.tabs(["📌 Reasons", "📐 Indicators", "📰 News", "🔄 Backtest", "📚 Research", "🤖 Automation", "🧠 AI Tracker"])

    with tabs[0]:
        _render_signal_expander(sig_short)
        st.divider()
        _render_signal_expander(sig_med)
        st.divider()
        _render_signal_expander(sig_long)

    with tabs[1]:
        st.subheader("Pineify Signals & Overlays")
        reason_rows = _get_indicator_reasoning(ind_summary)
        if not reason_rows:
            st.info("Indicator analysis is not available for this symbol yet.")
        else:
            for row in reason_rows:
                st.markdown(f"**{row['name']}**: {row['signal']} — {row['strategy']}")
                st.caption(f"value: {row['value']}")

        st.divider()
        st.subheader("Smart VWAP / Signal Breakdown")
        if ind_summary:
            for key in ["rsi", "macd", "macd_signal", "smart_vwap", "vwap", "pineify_bias", "sma_short", "sma_long"]:
                if key in ind_summary:
                    st.metric(key, f"{ind_summary[key]:,.4f}" if isinstance(ind_summary[key], float) else ind_summary[key])

    with tabs[2]:
        st.subheader("Recent News & Sentiment")
        for item in sentiment.get("items", []):
            s = item.get("sentiment", 0)
            emoji = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "⚪")
            st.markdown(f"{emoji} **{item.get('title')}** — *{item.get('source')}*")
            if item.get("link"):
                st.markdown(f"[Read more]({item.get('link')})")
            st.caption(item.get("published", ""))

    with tabs[3]:
        st.subheader("Backtest")
        st.markdown("#### Evidence engine")
        st.caption(
            "Purged, embargoed walk-forward validation with calibrated probabilities. "
            "The operating point is the lowest P(up) threshold whose out-of-sample precision reaches the 80% target."
        )
        scope = st.radio("Model scope", ["Auto (best measured)", "Panel (12-asset pooled)", "Single asset"],
                         horizontal=True, key="evidence_scope")
        panel_available = any((v or {}).get("scope") == "panel" for v in validations.values())
        if scope.startswith("Panel") and not panel_available:
            state = _panel_cache()
            if state.get("fetching") or (state.get("ready") and not state.get("error")):
                st.info("Building the pooled panel model in the background (fetching 12 histories, then training short/medium/long walk-forward folds). Once ready it appears here automatically; single-asset evidence is shown meanwhile.")
            elif state.get("error"):
                st.warning(f"Panel model unavailable: {state['error']}")
            else:
                st.info("The panel model has not produced evidence for this timeframe yet; showing the single-asset evidence instead.")
        elif scope == "Single asset":
            st.caption("Showing the per-symbol ensemble for the selected asset only.")
        evidence_tf = st.selectbox("Evidence timeframe", ["short", "medium", "long"], index=1, key="evidence_tf")
        evidence = validations.get(evidence_tf, {})
        compare = evidence_compare.get(evidence_tf)
        if compare:
            winner = "pooled panel" if compare["chosen"] == "panel" else "single-asset"
            caption_text = (
                f"Auto-selection compares measured out-of-sample precision: "
                f"single-asset {compare['single_success'] * 100:.1f}% vs pooled panel {compare['panel_success'] * 100:.1f}% "
                f"→ {winner} evidence in use for {evidence_tf}."
            )
            panel_auc = compare.get("panel_auc")
            if panel_auc is not None:
                caption_text += f" Panel ROC-AUC {panel_auc:.3f}"
                weight = compare.get("blend_weight", 0.0)
                caption_text += (
                    f", blended into the live probability with skill weight {weight:.2f}."
                    if weight > 0
                    else " (no demonstrated skill edge; the live probability stays single-asset)."
                )
            st.caption(caption_text)
        if evidence.get("scope") == "panel":
            st.caption(f"Pooled across {len(evidence.get('symbols', []))} symbols · "
                       f"{evidence.get('panel_rows', 0):,} training rows · "
                       f"{evidence.get('panel_dates', 0)} dates · label: drift-relative direction")
            rank_obj = evidence.get("rank_objective") or {}
            if rank_obj.get("ic_mean") is not None:
                st.caption(
                    f"Cross-sectional rank IC {rank_obj['ic_mean']:.3f} "
                    f"(IC-IR {(rank_obj.get('ic_ir') or 0):.2f}, positive on "
                    f"{(rank_obj.get('ic_positive_share') or 0) * 100:.0f}% of "
                    f"{rank_obj.get('dates_evaluated', 0)} dates) · "
                    f"top-minus-bottom tercile spread {(rank_obj.get('spread_mean') or 0) * 100:.2f}%/period "
                    f"(positive on {(rank_obj.get('spread_positive_share') or 0) * 100:.0f}% of dates)"
                )
            live_probs = evidence.get("live_probabilities") or {}
            if live_probs:
                with st.expander("Panel live P(up) by symbol (latest bar)"):
                    st.dataframe(
                        pd.DataFrame([{"Symbol": sym, "P(up)": prob}
                                      for sym, prob in sorted(live_probs.items(), key=lambda kv: -kv[1])]),
                        hide_index=True,
                        width="stretch",
                    )
        operating = evidence.get("operating") or {}
        if evidence.get("error"):
            st.info(evidence["error"])
        elif evidence.get("available"):
            diagnostics = evidence.get("diagnostics") or {}
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("ROC-AUC (model skill)", f"{(diagnostics.get('roc_auc') or 0.5):.3f}")
            e2.metric("Up-move base rate", f"{(evidence.get('base_rate') or 0) * 100:.1f}%")
            e3.metric("Operating precision", f"{(operating.get('success') or 0) * 100:.1f}%")
            e4.metric("Signal coverage", f"{(operating.get('coverage') or 0) * 100:.1f}%")
            if evidence.get("eligible"):
                st.success(f"{operating.get('reason')} Live {evidence_tf.title()} signals fire only when the calibrated probability clears {operating.get('threshold'):.2f}.")
            else:
                st.warning(f"{operating.get('reason')} Live {evidence_tf.title()} signals stay gated to HOLD until the model reaches the 80% precision target out-of-sample.")
            live_probability = evidence.get("live_probability")
            if live_probability is not None:
                st.caption(
                    f"Latest calibrated P(up) for {asset.symbol}: {live_probability * 100:.1f}% "
                    f"(operating threshold {operating.get('threshold') or 0.5:.2f})."
                )
            gate_info = evidence.get("gate") or {}
            if gate_info:
                trend_value = gate_info.get("mkt_trend")
                if trend_value is None:
                    st.caption("Regime gate: benchmark data unavailable; no gating applied.")
                else:
                    st.caption(
                        f"Regime gate — new longs ×{gate_info.get('long_size_mult')}, "
                        f"new shorts ×{gate_info.get('short_size_mult')} · "
                        f"SPY vs its 200-SMA {trend_value:+.1%} · "
                        f"asset vol percentile {gate_info.get('vol_regime_pctile')}"
                    )
            sweep = evidence.get("sweep")
            if sweep is not None and len(sweep):
                with st.expander("Precision / coverage sweep (out-of-sample)"):
                    st.dataframe(sweep, hide_index=True, width="stretch")
            reliability = evidence.get("reliability")
            if reliability is not None and len(reliability):
                with st.expander("Calibration reliability (predicted vs observed)"):
                    st.dataframe(reliability, hide_index=True, width="stretch")

        st.divider()
        st.markdown("#### Pooled panel evidence")
        st.caption(
            f"A single cross-sectional model pooled over {len(panel_state.get('symbols') or [])} liquid symbols, with "
            "same-date relative-strength and volatility-rank features. In Auto mode it backs an "
            "instrument only when it measures better out-of-sample than the per-symbol ensemble."
        )
        panel_evidence = (panel_state.get("results") or {}).get(evidence_tf)
        if not isinstance(panel_evidence, dict):
            panel_evidence = {}
        if panel_evidence.get("available"):
            panel_operating = panel_evidence.get("operating") or {}
            panel_diagnostics = panel_evidence.get("diagnostics") or {}
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Panel ROC-AUC", f"{(panel_diagnostics.get('roc_auc') or 0.5):.3f}")
            p2.metric("Panel precision", f"{(panel_operating.get('success') or 0) * 100:.1f}%")
            p3.metric("Panel coverage", f"{(panel_operating.get('coverage') or 0) * 100:.1f}%")
            p4.metric("Panel signals", f"{panel_operating.get('signals') or 0}")
            st.caption(
                f"Pooled over {len(panel_evidence.get('symbols') or [])} symbols "
                f"({panel_evidence.get('panel_rows')} rows, {panel_evidence.get('panel_dates')} dates, "
                f"{panel_evidence.get('folds')} purged walk-forward folds). {panel_evidence.get('reason') or ''}"
            )
            rank_obj = panel_evidence.get("rank_objective") or {}
            if rank_obj.get("ic_mean") is not None:
                st.caption(
                    f"Ranking skill: daily rank IC {rank_obj['ic_mean']:.3f} "
                    f"(IC-IR {(rank_obj.get('ic_ir') or 0):.2f}) · long-short tercile spread "
                    f"{(rank_obj.get('spread_mean') or 0) * 100:.2f}%/period, positive on "
                    f"{(rank_obj.get('spread_positive_share') or 0) * 100:.0f}% of dates"
                )
            if panel_evidence.get("eligible"):
                st.success("Panel operating point meets the 80% precision target; panel-governed timeframes may emit signals when a symbol's calibrated probability clears the threshold.")
            else:
                st.warning("Panel operating point does not meet the 80% precision target; panel-governed timeframes stay HOLD.")
            panel_probabilities = panel_evidence.get("live_probabilities") or {}
            if panel_probabilities:
                panel_threshold = panel_operating.get("threshold") or 0.5
                probability_rows = [
                    {
                        "Symbol": symbol,
                        "P(up)": probability,
                        "Clears threshold": "yes" if probability >= panel_threshold else "no",
                    }
                    for symbol, probability in sorted(panel_probabilities.items(), key=lambda item: item[1], reverse=True)
                ]
                st.dataframe(pd.DataFrame(probability_rows), hide_index=True, width="stretch")
        else:
            st.info(panel_evidence.get("error") or "Panel evidence is not available yet.")

        manual_scenario = st.selectbox(
            "Manual scenario",
            ["Default technical rules", "BUY-only scenario", "SELL-only scenario"],
            help="Run the selected rule over the loaded history to inspect a particular directional scenario.",
        )
        manual_signal_fn = None
        if manual_scenario == "BUY-only scenario":
            manual_signal_fn = lambda row: {"score": 1.0}
        elif manual_scenario == "SELL-only scenario":
            manual_signal_fn = lambda row: {"score": 0.0}
        if st.button("Run Backtest", key="run_bt"):
            with st.spinner("Running backtest..."):
                result = backtest(ind_df, allow_short=allow_short, signal_fn=manual_signal_fn)
            if "error" in result:
                st.warning(result["error"])
            else:
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Return", f"{result['total_return']*100:.2f}%")
                m2.metric("Annual Return", f"{result['annual_return']*100:.2f}%")
                m3.metric("Sharpe", result["sharpe"])
                m4.metric("Max Drawdown", f"{result['max_drawdown']*100:.2f}%")
                m5.metric("Win Rate", f"{result['win_rate']*100:.1f}%")
                m6.metric("Trades", result["num_trades"])
                b1, b2 = st.columns(2)
                b1.metric("Profit Factor", result.get("profit_factor", "N/A"))
                b2.metric("Expectancy / Trade", f"${result.get('expectancy', 0):.2f}")
                st.plotly_chart(equity_chart(result["equity_curve"]), width="stretch")
        else:
            st.write("Click 'Run Backtest' to evaluate the strategy on historical data.")

        if st.button("Run walk-forward validation", key="run_wf"):
            with st.spinner("Evaluating sequential out-of-sample windows..."):
                walk_forward = walk_forward_backtest(ind_df, allow_short=allow_short)
            if "error" in walk_forward:
                st.warning(walk_forward["error"])
            else:
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Validation folds", walk_forward["fold_count"])
                w2.metric("Profitable folds", f"{walk_forward['profitable_fold_rate'] * 100:.1f}%")
                w3.metric("Average return", f"{walk_forward['average_return'] * 100:.2f}%")
                w4.metric("Worst drawdown", f"{walk_forward['worst_drawdown'] * 100:.2f}%")
                st.dataframe(walk_forward["folds"], hide_index=True, width="stretch")

    with tabs[4]:
        st.subheader("Strategy and indicator research")
        st.caption("Review the exact methods used by the app before testing a scenario.")
        research_type = st.segmented_control("Research catalog", ["Strategies", "Indicators"], default="Strategies")
        catalog = strategy_catalog() if research_type == "Strategies" else indicator_catalog()
        for item in catalog:
            with st.container(border=True):
                st.markdown(f"**{item['name']}** · {item['type']}" if item.get("type") else f"**{item['name']}**")
                st.write(item["method"])
                st.caption(item["steps"])

    with tabs[5]:
        st.subheader("Automation (Broker Integration)")
        st.markdown("A pluggable broker adapter is built in. Currently running in **paper trading** mode.")
        if "paper_broker" not in st.session_state:
            st.session_state.paper_broker = get_broker()
        broker = st.session_state.paper_broker
        if st.button("Connect Paper Broker"):
            ok = broker.connect()
            st.success("Connected to Paper Broker." if ok else "Connection failed.")

        st.markdown("**Place a paper order**")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            order_side = st.selectbox("Side", ["buy", "sell"])
        with bc2:
            order_qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        with bc3:
            st.write("")
            if st.button("Place Order"):
                order = broker.place_order(asset.symbol, order_side, order_qty)
                st.json(order)

        st.markdown("**Current paper positions**")
        st.write(broker.get_positions())

        st.info("To go live later: implement an adapter for your broker (IBKR / Alpaca / Binance) by extending `BrokerAdapter` in `broker/adapter.py`, then set `ACTIVE_BROKER` in your `.env`.")

    with tabs[6]:
        st.subheader("🧠 Self-Learning Tracker")
        st.markdown("The AI tracks its past signals, checks if they were profitable, and slightly adjusts its strategies/weights to maximize future win probability.")
        stats = tracker.get_stats()
        asc1, asc2, asc3, asc4 = st.columns(4)
        asc1.metric("Total Signals Logged", stats["tot"])
        asc2.metric("Signals Resolved (Expired)", stats["res"])
        asc3.metric("AI Win Rate", f"{stats['win_rate']*100:.1f}%")
        asc4.metric("Average Return per Trade", f"{stats['avg_return']*100:.2f}%")

        st.markdown("**Current Optimized Strategy Weights:**")
        w_df = []
        for tf, w in tracker.data["weights"].items():
            w_df.append({"Timeframe": tf.title(), "Technical": f"{w['technical']*100:.1f}%", "Fundamental": f"{w['fundamental']*100:.1f}%", "Sentiment": f"{w['sentiment']*100:.1f}%"})
        st.table(w_df)