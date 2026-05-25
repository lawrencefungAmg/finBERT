#!/usr/bin/env python3
"""
One-command sentiment analysis pipeline.

Usage (from project root):
    python scripts/run_sentiment.py                        # all tickers from market-data-service (1-day lookback)
    python scripts/run_sentiment.py --days 7               # override lookback window
    python scripts/run_sentiment.py --tickers AAPL MSFT    # override stock universe (skip API call)
    python scripts/run_sentiment.py --push-db http://myserver/api/sentiment
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from pipeline import market_data as market_data_mod
from pipeline import news as news_mod
from pipeline import sentiment as sentiment_mod
from pipeline import writer as writer_mod


def print_summary(records: list, stocks: list):
    from collections import Counter, defaultdict

    by_ticker = defaultdict(list)
    for r in records:
        by_ticker[r["ticker"]].append(r["sentiment"])

    print("\n" + "=" * 55)
    print(f"{'Ticker':<12} {'Articles':>8}  {'Positive':>9} {'Negative':>9} {'Neutral':>8}")
    print("-" * 55)
    for stock in stocks:
        t = stock["ticker"]
        sentiments = by_ticker.get(t, [])
        if not sentiments:
            print(f"{t:<12} {'0':>8}  {'—':>9} {'—':>9} {'—':>8}")
            continue
        c = Counter(sentiments)
        total = len(sentiments)
        print(
            f"{t:<12} {total:>8}  "
            f"{c.get('Positive', 0):>9} "
            f"{c.get('Negative', 0):>9} "
            f"{c.get('Neutral', 0):>8}"
        )
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Run financial sentiment analysis pipeline")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Override stock universe (skips market-data-service call)")
    parser.add_argument("--days", type=int, help="Days of news to look back (default: DAYS_LOOKBACK env or 1)")
    parser.add_argument("--push-db", metavar="ENDPOINT", help="Push results to this DB endpoint URL")
    args = parser.parse_args()

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

    # Resolve stock list
    if args.tickers:
        stocks = [{"ticker": t, "name": t} for t in args.tickers]
        print(f"Stock universe: {len(stocks)} ticker(s) from --tickers override")
    else:
        market_data_url = os.environ.get("MARKET_DATA_URL", "")
        if not market_data_url:
            print("ERROR: MARKET_DATA_URL is not set. Add it to your .env file.", file=sys.stderr)
            sys.exit(1)
        try:
            stocks = market_data_mod.fetch_tickers(market_data_url)
        except Exception as e:
            print(f"ERROR: Could not fetch tickers from {market_data_url}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Stock universe: {len(stocks)} ticker(s) from {market_data_url}")

    if not stocks:
        print("ERROR: No stocks to process.", file=sys.stderr)
        sys.exit(1)

    days_back = args.days or int(os.environ.get("DAYS_LOOKBACK", "1"))
    from_date, to_date = news_mod.date_range(days_back)

    lm_client = sentiment_mod.build_client(lm_url)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    csv_path, csv_writer, csv_file = writer_mod.open_incremental(output_dir)
    print(f"Lookback: {days_back} day(s)  ({from_date} → {to_date})")
    print(f"LM Studio: {lm_url}  model hint: {lm_model}")
    print(f"Output: {csv_path}  (written incrementally — safe to Ctrl+C)\n")

    db_endpoint = args.push_db or os.environ.get("SENTIMENT_API_URL", "")

    records = []
    interrupted = False
    try:
        for stock in stocks:
            ticker = stock["ticker"]
            print(f"[{ticker}] Fetching news ...", end=" ", flush=True)

            # TODO: add a news source that supports HK stocks (Finnhub only covers US equities)
            if ticker.endswith(".HK"):
                print("SKIPPED (HK stock — Finnhub not supported)")
                continue

            try:
                articles = news_mod.fetch_news(ticker, finnhub_api_key, from_date, to_date)
            except Exception as e:
                print(f"FAILED ({e})")
                continue

            total_articles = len(articles)
            print(f"{total_articles} article(s) found")

            stock_records = []
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
                stock_records.append(record)
                records.append(record)
                csv_writer.writerow(record)
                csv_file.flush()
                print(f"  [{i}/{total_articles}] {result['sentiment']:8s} ({result['confidence_score']:.2f})  {article['headline'][:70]}")

            if db_endpoint and stock_records:
                try:
                    writer_mod.push_articles_to_db(stock_records, db_endpoint)
                except Exception as e:
                    print(f"  [{ticker}] Article push failed: {e}")

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

    print_summary(records, stocks)


if __name__ == "__main__":
    main()
