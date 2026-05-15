import os
import requests
from datetime import datetime, timedelta


def fetch_news(ticker: str, api_key: str, from_date: str, to_date: str) -> list:
    """
    Fetch company news from Finnhub for a given ticker.

    Args:
        ticker: Stock symbol, e.g. "AAPL"
        api_key: Finnhub API key
        from_date: Start date as "YYYY-MM-DD"
        to_date: End date as "YYYY-MM-DD"

    Returns:
        List of dicts with keys: id, datetime, headline, summary, source, url
    """
    base_url = os.environ["FINNHUB_BASE_URL"]
    url = f"{base_url}/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    items = resp.json()

    results = []
    seen_ids = set()
    for item in items:
        news_id = item.get("id")
        if news_id in seen_ids:
            continue
        seen_ids.add(news_id)
        results.append({
            "id": news_id,
            "datetime": datetime.utcfromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M:%S"),
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        })
    return results


def date_range(days_back: int):
    """Return (from_date, to_date) strings for the last N days."""
    today = datetime.utcnow().date()
    from_date = today - timedelta(days=days_back)
    return str(from_date), str(today)
