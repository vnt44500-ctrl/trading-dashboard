"""News sentiment analysis using VADER NLP pipeline.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is heavily used
in quantitative finance to parse financial headlines, negations, and intensities.
"""
from datetime import datetime, timezone
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize NLP pipeline
vader = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> float:
    """Return a score in [-1, 1] using VADER NLP."""
    if not text:
        return 0.0
    
    # Financial domain adjustments could be appended to VADER's lexicon here:
    custom_finance = {
        "beat": 2.0, "miss": -2.0, "upgrade": 2.0, "downgrade": -2.0,
        "soar": 3.0, "plunge": -3.0, "record": 1.5, "lawsuit": -2.5,
        "bullish": 2.5, "bearish": -2.5, "fraud": -3.5, "bankruptcy": -4.0,
        "dividend": 1.5, "revenue": 1.0, "growth": 1.5, "loss": -1.5,
    }
    vader.lexicon.update(custom_finance)
    
    # Generate compound polarity score
    scores = vader.polarity_scores(text)
    return scores['compound']

def analyze_news_sentiment(news: List[dict]) -> dict:
    """Aggregate sentiment over a list of news items using NLP."""
    if not news:
        return {"score": 0.5, "reasons": ["No recent news available."], "items": []}

    scored = []
    weighted_sentiment = 0.0
    total_weight = 0.0
    
    for item in news:
        text = f"{item.get('title', '')}. {item.get('summary', '')}".strip()
        s = sentiment_score(text)
        age_days = 0.0
        published = item.get("published")
        if published:
            try:
                parsed = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400)
            except (TypeError, ValueError):
                age_days = 0.0
        weight = 1 / (1 + age_days / 3)
        weighted_sentiment += s * weight
        total_weight += weight
        scored.append({**item, "sentiment": round(s, 3)})

    avg = weighted_sentiment / total_weight if total_weight else 0.0
    
    # Normalizing NLP score [-1,1] directly into [0,1] confidence probability
    normalized = (avg + 1) / 2

    reasons = []
    if avg > 0.25:
        reasons.append(f"Strongly bullish news sentiment (VADER: {avg:+.2f}).")
    elif avg > 0.05:
        reasons.append(f"Mildly positive news sentiment (VADER: {avg:+.2f}).")
    elif avg < -0.25:
        reasons.append(f"Strongly bearish news sentiment (VADER: {avg:+.2f}).")
    elif avg < -0.05:
        reasons.append(f"Mildly negative news sentiment (VADER: {avg:+.2f}).")
    else:
        reasons.append(f"News sentiment is neutral/mixed (VADER: {avg:+.2f}).")

    positive_count = sum(1 for s in scored if s["sentiment"] > 0.05)
    negative_count = sum(1 for s in scored if s["sentiment"] < -0.05)
    reasons.append(f"NLP detected {positive_count} bullish vs {negative_count} bearish contextual headlines.")

    return {"score": normalized, "data_quality": min(1.0, len(scored) / 5), "reasons": reasons, "items": scored}