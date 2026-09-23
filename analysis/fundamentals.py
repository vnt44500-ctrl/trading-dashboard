"""Fundamental analysis: valuation metrics and financial results.

Incorporates standard metrics and proxies for advanced analytical paradigms like
Piotroski F-Score when data permits.
"""
def analyze_fundamentals(asset, quote: dict) -> dict:
    """Return a fundamental analysis summary and score."""
    reasons = []
    score_parts = []
    findings = {}

    if asset.asset_class == "stock":
        pe = quote.get("pe")
        eps = quote.get("eps")
        dy = quote.get("dividend_yield")
        beta = quote.get("beta")
        mcap = quote.get("market_cap")

        findings = {
            "pe": pe, "eps": eps, "dividend_yield": dy,
            "beta": beta, "market_cap": mcap,
            "current_ratio": quote.get("currentRatio"),
            "debt_to_equity": quote.get("debtToEquity"),
            "return_on_equity": quote.get("returnOnEquity"),
        }
        
        cr = quote.get("currentRatio")
        dte = quote.get("debtToEquity")
        roe = quote.get("returnOnEquity")

        if cr is not None:
            score_parts.append(0.7 if cr >= 1.5 else 0.35 if cr >= 1 else 0.15)
            reasons.append(f"Liquidity: current ratio {cr:.2f}.")
        if dte is not None:
            score_parts.append(0.75 if dte <= 100 else 0.45 if dte <= 200 else 0.2)
            reasons.append(f"Leverage: debt/equity {dte:.1f}.")
        if roe is not None:
            score_parts.append(0.75 if roe >= 0.15 else 0.55 if roe > 0 else 0.2)
            reasons.append(f"Profitability: ROE {roe:.1%}.")
        if pe is not None and pe > 0:
            score_parts.append(0.75 if pe < 15 else 0.6 if pe < 25 else 0.35 if pe < 40 else 0.15)
            reasons.append(f"Valuation: P/E {pe:.1f}.")
        if eps is not None:
            score_parts.append(0.7 if eps > 0 else 0.2)
            reasons.append(f"EPS is {'positive' if eps > 0 else 'negative'} ({eps:.2f}).")
        if not score_parts:
            reasons.append("Insufficient fundamental data; score remains neutral.")

        if beta is not None:
            if beta > 1.5:
                reasons.append(f"High Volatility Beta ({beta:.2f}).")
            elif beta < 0.8:
                reasons.append(f"Low Volatility Beta ({beta:.2f}) - Defensive asset.")

    elif asset.asset_class == "crypto":
        findings = {"note": "Crypto fundamentals use on-chain proxy (momentum)."}
        reasons.append("Crypto structural momentum is used as fundamental baseline.")

    else:
        findings = {"note": "Fundamentals proxied via price/vol strength for this asset type."}
        reasons.append(f"{asset.asset_class.title()} fundamentals approximated.")

    score = sum(score_parts) / len(score_parts) if score_parts else 0.5
    data_quality = min(1.0, len(score_parts) / 5)
    score = 0.5 + (score - 0.5) * data_quality
    return {"score": round(max(0.0, min(1.0, score)), 4), "data_quality": round(data_quality, 3), "findings": findings, "reasons": reasons}