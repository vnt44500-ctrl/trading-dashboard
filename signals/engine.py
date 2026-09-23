"""Signal generation engine.

Combines technical, fundamental, and news-sentiment analysis into a
consolidated BUY / SELL / HOLD signal with confidence and reasons.
"""

import math

from core.models import Signal
from signals.tracker import tracker
from config import config

# Relative contribution of each technical component. Only the ratios matter:
# the evidence sum is normalised by this capacity before being squashed into a
# probability-like score, which is what prevents the previous `max(0, min(1))`
# saturation (three separate timeframes all reporting exactly 1.000).
COMPONENT_WEIGHTS = {
    "trend": 0.12,
    "rsi_extreme": 0.15,
    "rsi_momentum": 0.05,
    "macd": 0.08,
    "price_action": 0.06,
    "volume": 0.04,
    "vwap": 0.06,
    "poc": 0.05,
    "cloud": 0.15,
    "tenkan": 0.05,
    "squeeze": 0.10,
    "psar": 0.06,
    "supertrend": 0.08,
    "cmf": 0.08,
    "structure": 0.05,
    "regime": 0.04,
}
EVIDENCE_CAPACITY = sum(COMPONENT_WEIGHTS.values())
SCORE_SCALE = 0.8


def squash(evidence: float) -> float:
    """Map raw evidence in [-capacity, +capacity] to (0, 1) without saturation."""
    normalized = max(-1.0, min(1.0, evidence / EVIDENCE_CAPACITY))
    return round(0.5 + 0.5 * math.tanh(SCORE_SCALE * normalized), 4)


def _risk_metrics(ind: dict, price: float | None) -> dict:
    if not price or price <= 0:
        return {"risk_score": 100.0, "stop_distance": 0.0, "position_size": 0.0}
    atr_value = ind.get("atr") or 0.0
    atr_pct = abs(atr_value / price) if atr_value else 0.0
    volatility = abs(ind.get("volatility") or 0.0)
    var_value = abs(ind.get("var") or 0.0)
    components = [min(volatility, 1.0), min(var_value * 2, 1.0), min(atr_pct * 3, 1.0)]
    risk_score = round(min(100.0, max(0.0, sum(components) / len(components) * 100)), 1)
    stop_distance = max(float(atr_value) * config.atr_stop_multiple, price * 0.005)
    risk_budget = config.account_equity * config.risk_per_trade
    raw_size = risk_budget / stop_distance if stop_distance else 0.0
    max_size = config.account_equity * config.max_position_fraction / price
    return {"risk_score": risk_score, "stop_distance": stop_distance, "position_size": min(raw_size, max_size)}

def technical_signal(ind: dict, timeframe: str) -> dict:
    """Score technical indicators based on timeframe context."""
    reasons = []
    evidence = 0.0
    price = ind.get("_price")

    def contrib(component: str, direction: int, msg: str):
        nonlocal evidence
        evidence += COMPONENT_WEIGHTS[component] * direction
        reasons.append(msg)

    # Dynamic MA selection based on timeframe. `sma_200` must be tested with
    # `or` rather than a `.get` default, because last_indicators always creates
    # the key (with a None value) so a default never applies.
    if timeframe == "short":
        fast, slow = ind.get("ema_short"), ind.get("sma_short")
    elif timeframe == "long":
        fast, slow = ind.get("sma_long"), ind.get("sma_200") or ind.get("sma_long")
    else:  # medium
        fast, slow = ind.get("sma_short"), ind.get("sma_long")
    rsi = ind.get("rsi")

    if fast is not None and slow is not None:
        if fast > slow:
            contrib("trend", 1, f"Trend: Fast MA > Slow MA ({timeframe} trend up).")
        else:
            contrib("trend", -1, f"Trend: Fast MA < Slow MA ({timeframe} trend down).")

    # RSI logic changes slightly by timeframe
    if rsi is not None:
        if rsi < 30:
            contrib("rsi_extreme", 1, f"RSI deeply oversold ({rsi:.1f}) — potential bounce.")
        elif rsi > 70:
            contrib("rsi_extreme", -1, f"RSI heavily overbought ({rsi:.1f}) — pullback expected.")
        elif timeframe == "short" and rsi > 50:
            contrib("rsi_momentum", 1, "RSI > 50 (short-term momentum).")
        elif timeframe == "short" and rsi < 50:
            contrib("rsi_momentum", -1, "RSI < 50 (short-term weakness).")

    # MACD
    macd = ind.get("macd")
    macd_signal = ind.get("macd_signal")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            contrib("macd", 1, "MACD Bullish Crossover.")
        else:
            contrib("macd", -1, "MACD Bearish Crossover.")

    # Price action and volume provide confirmation for indicator alignment.
    open_price = ind.get("open")
    close_price = ind.get("close", price)
    if open_price is not None and close_price is not None:
        if close_price > open_price:
            contrib("price_action", 1, "Price action closed bullish on the latest bar.")
        elif close_price < open_price:
            contrib("price_action", -1, "Price action closed bearish on the latest bar.")

    volume = ind.get("volume")
    volume_sma = ind.get("volume_sma")
    if volume is not None and volume_sma and volume_sma > 0:
        if volume > volume_sma and ind.get("returns") is not None:
            direction = 1 if ind["returns"] > 0 else -1
            label = "buying" if direction > 0 else "selling"
            contrib("volume", direction, f"Volume confirms {label} pressure ({volume / volume_sma:.1f}x 20-bar average).")
        else:
            reasons.append("Volume is below its 20-bar average, so the move has weaker confirmation.")

    # Smart VWAP and Pineify Bias are confirmation filters, not independent signals.
    smart_vwap = ind.get("smart_vwap")
    if price is not None and smart_vwap is not None:
        if price > smart_vwap:
            contrib("vwap", 1, f"Price is above Smart VWAP ({smart_vwap:.2f}), supporting bullish confluence.")
        elif price < smart_vwap:
            contrib("vwap", -1, f"Price is below Smart VWAP ({smart_vwap:.2f}), supporting bearish confluence.")

    pineify_bias = ind.get("pineify_bias")
    if pineify_bias is not None and price is not None:
        if pineify_bias > 0:
            contrib("vwap", 1, f"Pineify Bias is positive ({pineify_bias:.2f}), confirming short-term trend strength.")
        elif pineify_bias < 0:
            contrib("vwap", -1, f"Pineify Bias is negative ({pineify_bias:.2f}), confirming short-term trend weakness.")

    # Support/resistance turns the indicator alignment into a location-aware
    # signal. These come from the causal per-bar levels, so a backtest sees the
    # same numbers a live signal would have seen.
    support = ind.get("nearest_support")
    resistance = ind.get("nearest_resistance")
    if price is not None and price > 0:
        if support is not None and 0 < (price - support) / price < 0.03:
            contrib("structure", 1, f"Price is near support at {support:.2f}, improving bullish risk/reward.")
        elif resistance is not None and 0 < (resistance - price) / price < 0.03:
            contrib("structure", -1, f"Price is near resistance at {resistance:.2f}, limiting bullish follow-through.")

    # Ichimoku Cloud (Premium structural)
    tenkan = ind.get("tenkan")
    kijun = ind.get("kijun")
    senkou_a = ind.get("senkou_a")
    senkou_b = ind.get("senkou_b")
    if None not in (tenkan, kijun, senkou_a, senkou_b, price):
        cloud_top = max(senkou_a, senkou_b)
        cloud_bot = min(senkou_a, senkou_b)
        if price > cloud_top:
            contrib("cloud", 1, "Price above Ichimoku Cloud (Strong Bullish context).")
        elif price < cloud_bot:
            contrib("cloud", -1, "Price below Ichimoku Cloud (Strong Bearish context).")

        if timeframe in ("short", "medium") and tenkan > kijun:
            contrib("tenkan", 1, "Tenkan > Kijun (Ichimoku Crossover).")
            
    # Premium: Volume Profile Point of Control (POC)
    vp_poc = ind.get("vp_poc")
    if vp_poc is not None and price is not None:
        if price > vp_poc:
            contrib("poc", 1, f"Trading above Volume Profile POC ({vp_poc:.2f}) — accumulation support.")
        elif price < vp_poc:
            contrib("poc", -1, f"Trading below Volume Profile POC ({vp_poc:.2f}) — heavy volume resistance.")

    # Premium: TTM Squeeze
    squeeze_on = ind.get("ttm_squeeze")
    macd_hist = ind.get("macd_hist", 0)
    if squeeze_on:
        if macd_hist > 0:
            contrib("squeeze", 1, "TTM Squeeze ON with positive momentum (Impending Bullish Breakout).")
        else:
            contrib("squeeze", -1, "TTM Squeeze ON with negative momentum (Impending Bearish Breakout).")

    # Premium: Parabolic SAR (PSAR) Trend stop
    psar = ind.get("psar")
    if psar is not None and price is not None:
        if price > psar:
            contrib("psar", 1, "PSAR indicates ongoing bullish trend.")
        else:
            contrib("psar", -1, "PSAR indicates ongoing bearish trend.")

    supertrend_bias = ind.get("supertrend_direction")
    if supertrend_bias == 1:
        contrib("supertrend", 1, "Supertrend confirms bullish direction.")
    elif supertrend_bias == -1:
        contrib("supertrend", -1, "Supertrend confirms bearish direction.")

    # Chaikin Money Flow
    cmf = ind.get("cmf")
    if cmf is not None:
        if cmf > 0.15:
            contrib("cmf", 1, f"Strong buying pressure (CMF {cmf:.2f}).")
        elif cmf < -0.15:
            contrib("cmf", -1, f"Strong selling pressure (CMF {cmf:.2f}).")

    # Per-bar regime bias. Using the numeric column rather than a whole-period
    # string keeps historical scoring identical to live scoring.
    regime_bias = ind.get("regime_bias")
    if regime_bias:
        if regime_bias > 0:
            contrib("regime", 1, "Regime bias is bullish; trend setups receive confirmation.")
        else:
            contrib("regime", -1, "Regime bias is bearish; bullish setups receive a defensive penalty.")
    else:
        reasons.append("Regime bias is neutral; directional signals require stronger confluence.")

    return {"score": squash(evidence), "evidence": round(evidence, 4), "capacity": EVIDENCE_CAPACITY, "reasons": reasons}

def _evidence_for(validation, timeframe: str) -> dict:
    """Extract the evidence-engine result for one timeframe.

    Accepts the per-timeframe mapping produced by the dashboard
    (``{tf: analyze_timeframe result}``). A legacy single-result dict
    (containing an ``operating`` key) is treated as medium-term evidence;
    anything else means no evidence is available.
    """
    if not isinstance(validation, dict) or not validation:
        return {}
    candidate = validation.get(timeframe)
    if isinstance(candidate, dict):
        return candidate
    if "operating" in validation and timeframe == "medium":
        return validation
    return {}


def generate_signals(asset, quote: dict, ind: dict, fundamental: dict, sentiment: dict, validation: dict | None = None) -> dict:
    """Produce Short, Medium, and Long term signals dynamically."""
    
    ind = dict(ind)
    ind["_price"] = quote.get("price")
    
    # Let tracker evaluate past predictions on this asset right now so it learns
    tracker.evaluate_and_learn(asset.symbol, quote.get("price"))

    results = {}
    timeframes = {
        "short": "1-5 Days (Momentum / Swing)",
        "medium": "2-6 Weeks (Position / Trend)",
        "long": "3-12 Months (Value / Macro)"
    }
    
    # Generate 3 distinct signals
    for tf, duration in timeframes.items():
        if tf == config.lead_timeframe:
            duration = f"{duration} — LEAD"
        tech = technical_signal(ind, tf)
        fund_score = fundamental.get("score", 0.5)
        sent_score = sentiment.get("score", 0.5)

        # Get AI optimized weights dynamically for this timeframe
        weights = tracker.get_weights(tf)
        
        combined = (
            tech["score"] * weights["technical"]
            + fund_score * weights["fundamental"]
            + sent_score * weights["sentiment"]
        )

        evidence = _evidence_for(validation, tf)
        operating = evidence.get("operating") or {}
        threshold = float(operating.get("threshold") or 0.5)
        prob = evidence.get("live_probability")
        if prob is not None:
            # Blend the calibrated model probability into the composite score.
            combined = (combined + prob) / 2

        if combined >= 0.60:
            raw_action = "BUY"
        elif combined <= 0.40:
            raw_action = "SELL"
        else:
            raw_action = "HOLD"

        live_eligible = evidence.get("eligible", False) if evidence else validation is None
        research_only = tf in getattr(config, "research_timeframes", ())
        if research_only:
            # Policy: this timeframe's edge did not survive validation (negative
            # net expectancy in the crisis window), so it never fires live.
            live_eligible = False
        action = raw_action if live_eligible else "HOLD"

        # Selective emission: even when the 80% evidence gate passes, the live
        # action only fires when the calibrated probability clears the
        # operating threshold — the exact rule whose precision was measured
        # out-of-sample on purged walk-forward folds.
        gate_reason = None
        if action == "BUY" and (prob is None or prob < threshold):
            action = "HOLD"
            gate_reason = ("No calibrated probability is available for this bar; no live entry."
                           if prob is None else
                           f"Calibrated P(up) {prob:.2f} is below the evidence threshold "
                           f"{threshold:.2f}; no live entry.")
        elif action == "SELL" and (prob is None or prob > 1.0 - threshold):
            action = "HOLD"
            gate_reason = ("No calibrated probability is available for this bar; no live entry."
                           if prob is None else
                           f"Calibrated P(down) {1.0 - prob:.2f} is below the evidence threshold "
                           f"{threshold:.2f}; no live entry.")

        # Regime gate: a causal market-state multiplier (benchmark 200-SMA trend
        # and trailing volatility percentile) sizes or blocks new entries. The
        # 10-year replay showed 2020-style crises were the worst stretch for
        # every timeframe; the gate scales exposure down there instead of
        # changing the signal itself.
        gate_info = (evidence.get("gate") or {}) if evidence else {}
        size_mult = 1.0
        gate_note = None
        if action == "BUY" and gate_info.get("long_size_mult") is not None:
            size_mult = float(gate_info["long_size_mult"])
        elif action == "SELL" and gate_info.get("short_size_mult") is not None:
            size_mult = float(gate_info["short_size_mult"])
        if action in ("BUY", "SELL") and size_mult <= 0.0:
            action = "HOLD"
            gate_reason = ("Regime gate blocks new entries on this bar: benchmark trend is "
                           "against the trade or the market is in a volatility crisis.")
        elif action in ("BUY", "SELL") and size_mult < 1.0:
            gate_note = (f"Regime gate: reduced to {size_mult * 100:.0f}% size "
                         "(adverse benchmark trend or elevated volatility regime).")

        confidence = abs(combined - 0.5) * 2
        risk = _risk_metrics(ind, quote.get("price"))
        confidence *= max(0.25, 1.0 - risk["risk_score"] / 150)

        reasons = tech["reasons"]
        if research_only:
            reasons.append(
                f"{tf.title()} timeframe is research-only: its live edge did not survive "
                f"validation, so the {raw_action} suggestion is displayed but never traded."
            )
        if weights["fundamental"] >= 0.2:
            reasons += fundamental.get("reasons", [])
        if weights["sentiment"] >= 0.2:
            reasons += sentiment.get("reasons", [])
        if gate_note:
            reasons.append(gate_note)

        strat = (f"AI-Optimized Composite ({weights['technical']*100:.0f}% Tech, "
                 f"{weights['fundamental']*100:.0f}% Fund, {weights['sentiment']*100:.0f}% Sent)")

        summary = (f"Research suggestion: {raw_action}; live status: {action} for {duration}. Combined score: {combined:.2f}; "
               f"estimated risk: {risk['risk_score']:.1f}/100; "
               f"risk-based position size: {risk['position_size']:.2f} units.")
        if evidence:
            success_pct = (operating.get("success") or 0.0) * 100
            if evidence.get("eligible"):
                summary += (f" Evidence: {success_pct:.1f}% measured precision on purged walk-forward signals at "
                            f"P(up) >= {threshold:.2f} ({operating.get('signals', 0)} signals, "
                            f"{(operating.get('coverage') or 0.0) * 100:.1f}% of bars).")
            else:
                summary += (f" Evidence gate not met: best achievable precision on this history is {success_pct:.1f}%, "
                            "so the live action stays HOLD.")
                reasons = list(reasons) + [
                    "The evidence engine has not demonstrated the 80% out-of-sample precision target for this timeframe; this signal is shown as HOLD."
                ]
        if gate_reason:
            reasons.append(gate_reason)
        
        sig = Signal(
            asset=asset,
            action=action,
            confidence=round(confidence, 3),
            price=quote.get("price"),
            timeframe=tf,
            expected_duration=duration,
            reasons=list(set(reasons)),
            strategy_name=strat,
            strategy_summary=summary,
            technical_score=round(tech["score"], 3),
            fundamental_score=round(fund_score, 3),
            sentiment_score=round(sent_score, 3),
            indicators=ind,
            model_version=config.signal_model_version,
            risk_score=risk["risk_score"],
            position_size=round(risk["position_size"] * size_mult, 6),
            raw_action=raw_action,
            live_eligible=live_eligible,
        )
        
        # Track for future evaluation (self-learning)
        tracker.record_signal(sig)

        results[tf] = sig

    return results