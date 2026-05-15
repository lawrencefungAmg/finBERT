#!/usr/bin/env python3
"""
One-command sentiment analysis pipeline.

Usage:
    python run_sentiment.py                        # uses stocks.yaml defaults
    python run_sentiment.py --days 30              # override lookback window
    python run_sentiment.py --tickers AAPL MSFT    # override stock universe
    python run_sentiment.py --config my_stocks.yaml
    python run_sentiment.py --push-db http://myserver/api/sentiment
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

load_dotenv()

from pipeline import news as news_mod
from pipeline import sentiment as sentiment_mod
from pipeline import writer as writer_mod


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_stock_list(cfg: dict, tickers_override: list) -> list:
    if tickers_override:
        return [{"ticker": t, "name": t} for t in tickers_override]
    return cfg.get("stocks", [])


def print_summary(records: list, stocks: list):
    from collections import defaultdict, Counter
    by_ticker = defaultdict(list)
    for r in records:
        by_ticker[r["ticker"]].append(r["sentiment"])

    print("\n" + "=" * 55)
    print(f"{'Ticker':<8} {'Articles':>8}  {'Positive':>9} {'Negative':>9} {'Neutral':>8}")
    print("-" * 55)
    for stock in stocks:
        t = stock["ticker"]
        sentiments = by_ticker.get(t, [])
        if not sentiments:
            print(f"{t:<8} {'0':>8}  {'—':>9} {'—':>9} {'—':>8}")
            continue
        c = Counter(sentiments)
        total = len(sentiments)
        print(
            f"{t:<8} {total:>8}  "
            f"{c.get('Positive', 0):>9} "
            f"{c.get('Negative', 0):>9} "
            f"{c.get('Neutral', 0):>8}"
        )
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Run financial sentiment analysis pipeline")
    parser.add_argument("--config", default="stocks.yaml", help="Path to stocks YAML config")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER", help="Override stock universe")
    parser.add_argument("--days", type=int, help="Days of news to look back (overrides config)")
    parser.add_argument("--push-db", metavar="ENDPOINT", help="Push results to this DB endpoint URL")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config(args.config)
    settings = cfg.get("settings", {})

    stocks = build_stock_list(cfg, args.tickers)
    if not stocks:
        print("ERROR: No stocks defined. Add tickers to stocks.yaml or pass --tickers.", file=sys.stderr)
        sys.exit(1)

    days_back = args.days or settings.get("days_lookback", 7)
    lm_url = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
    lm_model = os.environ.get("LM_STUDIO_MODEL", "qwen")
    output_dir = os.environ.get("OUTPUT_DIR", "output")

    finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not finnhub_api_key:
        print("ERROR: FINNHUB_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("FINNHUB_BASE_URL"):
        print("ERROR: FINNHUB_BASE_URL is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    # Build clients
    lm_client = sentiment_mod.build_client(lm_url)
    from_date, to_date = news_mod.date_range(days_back)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Pipeline starting — {len(stocks)} stock(s), {days_back} day(s) of news ({from_date} → {to_date})")
    print(f"LM Studio: {lm_url}  model hint: {lm_model}\n")

    csv_path, csv_writer, csv_file = writer_mod.open_incremental(output_dir)
    print(f"Output: {csv_path}  (written incrementally — safe to Ctrl+C)\n")

    records = []
    interrupted = False
    try:
        for stock in stocks:
            ticker = stock["ticker"]
            print(f"[{ticker}] Fetching news ...", end=" ", flush=True)

            try:
                articles = news_mod.fetch_news(ticker, finnhub_api_key, from_date, to_date)
            except Exception as e:
                print(f"FAILED ({e})")
                continue

            total_articles = len(articles)
            print(f"{total_articles} article(s) found")

            for i, article in enumerate(articles, 1):
                text = f"{article['headline']}. {article['summary']}".strip(". ")
                if not text:
                    continue

                try:
                    result = sentiment_mod.analyze(text, lm_client, lm_model)
                except Exception as e:
                    print(f"  [{ticker}] Sentiment error on article {article['id']}: {e}")
                    continue

                record = {
                    "run_ts": run_ts,
                    "ticker": ticker,
                    "news_id": article["id"],
                    "news_date": article["datetime"],
                    "source": article["source"],
                    "headline": article["headline"],
                    "sentiment": result.get("sentiment", ""),
                    "confidence_score": result.get("confidence_score", ""),
                }
                records.append(record)
                csv_writer.writerow(record)
                csv_file.flush()
                print(f"  [{i}/{total_articles}] {result['sentiment']:8s} ({result['confidence_score']:.2f})  {article['headline'][:70]}")

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\nInterrupted — {len(records)} record(s) saved so far.")
    finally:
        csv_file.close()

    if not records:
        print("\nNo records produced — nothing to save.")
        sys.exit(0)

    status = "Partial results" if interrupted else "Done"
    print(f"\n{status}: {len(records)} record(s) → {csv_path}")

    # Optional DB push (--push-db flag takes precedence over DB_ENDPOINT in .env)
    db_endpoint = args.push_db or os.environ.get("DB_ENDPOINT", "")
    if db_endpoint and not interrupted:
        writer_mod.push_to_db(records, db_endpoint)

    print_summary(records, stocks)


if __name__ == "__main__":
    main()
