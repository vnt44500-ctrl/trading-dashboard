"""News data provider using free/public sources.

Uses yfinance news (free) as primary source. Optionally supports NewsAPI
if a key is configured. Also supports a simple RSS fallback via requests.
"""
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests
import yfinance as yf

from config import config


def _to_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return None


class NewsProvider:
    """Aggregate news from free sources."""

    def get_news(self, symbol: str, limit: int = 15) -> List[dict]:
        news = []

        # Primary: yfinance news (free, no key required)
        try:
            ticker = yf.Ticker(symbol)
            items = ticker.news or []
            for item in items[:limit]:
                published = _to_dt(item.get("providerPublishTime"))
                news.append({
                    "title": item.get("title", ""),
                    "source": item.get("publisher", "yfinance"),
                    "link": item.get("link", ""),
                    "published": published.isoformat() if published else "",
                    "summary": "",
                })
        except Exception:
            pass

        # Secondary: NewsAPI if key present
        if config.news_api_key and len(news) < limit:
            news.extend(self._newsapi(symbol, limit - len(news)))

        return news[:limit]

    def _newsapi(self, symbol: str, limit: int) -> List[dict]:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol,
                "apiKey": config.news_api_key,
                "pageSize": limit,
                "sortBy": "publishedAt",
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            out = []
            for a in data.get("articles", []):
                out.append({
                    "title": a.get("title", ""),
                    "source": (a.get("source") or {}).get("name", "NewsAPI"),
                    "link": a.get("url", ""),
                    "published": a.get("publishedAt", ""),
                    "summary": a.get("description", ""),
                })
            return out
        except Exception:
            return []


news_provider = NewsProvider()