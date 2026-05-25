import csv
import os
from collections import defaultdict
from datetime import date, datetime


def write_csv(records: list, output_dir: str) -> str:
    """
    Write sentiment records to a timestamped CSV file.

    Returns:
        Path to the written file.
    """
    now = datetime.utcnow()
    date_folder = os.path.join(output_dir, now.strftime("%Y-%m-%d"))
    os.makedirs(date_folder, exist_ok=True)
    path = os.path.join(date_folder, f"sentiment_{now.strftime('%H%M%S')}.csv")

    fieldnames = ["run_ts", "ticker", "news_id", "news_date", "source", "headline", "sentiment", "confidence_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return path


def open_incremental(output_dir: str):
    """
    Open a timestamped CSV for row-by-row writing.
    Returns (path, DictWriter, file_handle). Caller must close the file handle.
    """
    now = datetime.utcnow()
    date_folder = os.path.join(output_dir, now.strftime("%Y-%m-%d"))
    os.makedirs(date_folder, exist_ok=True)
    path = os.path.join(date_folder, f"sentiment_{now.strftime('%H%M%S')}.csv")
    fieldnames = ["run_ts", "ticker", "news_id", "news_date", "source",
                  "headline", "sentiment", "confidence_score"]
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    fh.flush()
    return path, w, fh


def _article_to_raw_score(sentiment: str, confidence: float) -> float:
    """Convert a per-article sentiment label + confidence to a positive-class probability."""
    s = sentiment.lower()
    if s == "positive":
        return confidence
    if s == "negative":
        return 1.0 - confidence
    return 0.5  # neutral


def push_articles_to_db(records: list, endpoint: str, api_key: str | None = None):
    """
    Push article-level sentiment records for a single ticker to the server.
    Posts to <endpoint>/articles. Safe to call per-stock as articles finish.
    Server should upsert on (news_id, ticker).
    """
    import requests

    key = api_key or os.environ.get("SENTIMENT_API_KEY", "")
    if not key:
        raise ValueError("SENTIMENT_API_KEY is not set — cannot push to sentiment service.")

    articles_url = endpoint.rstrip("/") + "/articles"
    payload = [
        {
            "run_ts": r["run_ts"],
            "ticker": r["ticker"],
            "news_id": r["news_id"],
            "news_date": r["news_date"],
            "source": r["source"],
            "headline": r["headline"],
            "sentiment": r["sentiment"],
            "confidence_score": float(r["confidence_score"]) if r.get("confidence_score") else None,
        }
        for r in records
    ]
    resp = requests.post(
        articles_url,
        json=payload,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    inserted = result.get("inserted", "?")
    skipped = result.get("skipped", 0)
    print(f"  → Pushed {len(payload)} article(s) [{inserted} inserted, {skipped} skipped]")
    return result


def push_to_db(records: list, endpoint: str, api_key: str | None = None):
    """
    Aggregate article-level records by ticker and push daily sentiment scores
    to the sentiment service.  Re-runs are safe — the endpoint upserts on (date, symbol).
    """
    import requests

    key = api_key or os.environ.get("SENTIMENT_API_KEY", "")
    if not key:
        raise ValueError("SENTIMENT_API_KEY is not set — cannot push to sentiment service.")

    run_date = str(date.today())

    # Aggregate raw_score per ticker
    scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        raw = _article_to_raw_score(r.get("sentiment", "neutral"), float(r.get("confidence_score") or 0.5))
        scores[r["ticker"]].append(raw)

    payload = []
    for symbol, raw_list in scores.items():
        raw_score = sum(raw_list) / len(raw_list)
        sentiment_score = (raw_score - 0.5) * 6  # placeholder z-score: maps [0,1] → [-3,+3]
        payload.append({
            "date": run_date,
            "symbol": symbol,
            "sentiment_score": round(sentiment_score, 6),
            "raw_score": round(raw_score, 6),
            "source": "finbert",
        })

    resp = requests.post(
        endpoint,
        json=payload,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"Sentiment service: {result}  ({len(payload)} symbol(s) pushed)")
    return result
