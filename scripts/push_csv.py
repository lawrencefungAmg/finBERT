#!/usr/bin/env python3
"""
Replay a saved sentiment CSV to POST /sentiment/articles.

Usage:
    python scripts/push_csv.py output/2026-05-17/sentiment_113947.csv
    python scripts/push_csv.py output/2026-05-17/*.csv          # multiple files
    python scripts/push_csv.py output/2026-05-17/sentiment_113947.csv --batch-size 200
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from pipeline import writer as writer_mod


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Coerce types to match the API schema
    for r in rows:
        r["news_id"] = int(r["news_id"])
        r["confidence_score"] = float(r["confidence_score"]) if r.get("confidence_score") else None
    return rows


def push_in_batches(records: list[dict], endpoint: str, api_key: str, batch_size: int) -> tuple[int, int]:
    total_inserted = total_skipped = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        result = writer_mod.push_articles_to_db(chunk, endpoint, api_key)
        total_inserted += result.get("inserted", 0)
        total_skipped += result.get("skipped", 0)
    return total_inserted, total_skipped


def main():
    parser = argparse.ArgumentParser(description="Replay saved sentiment CSV(s) to the articles endpoint")
    parser.add_argument("files", nargs="+", metavar="CSV", help="One or more CSV files to push")
    parser.add_argument("--batch-size", type=int, default=500, metavar="N",
                        help="Rows per POST request (default: 500)")
    parser.add_argument("--endpoint", metavar="URL",
                        help="Override SENTIMENT_API_URL env var")
    args = parser.parse_args()

    endpoint = args.endpoint or os.environ.get("SENTIMENT_API_URL", "")
    if not endpoint:
        print("ERROR: SENTIMENT_API_URL is not set — add it to .env or pass --endpoint", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("SENTIMENT_API_KEY", "")
    if not api_key:
        print("ERROR: SENTIMENT_API_KEY is not set — add it to .env", file=sys.stderr)
        sys.exit(1)

    grand_inserted = grand_skipped = 0

    for file_arg in args.files:
        path = Path(file_arg)
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue

        records = read_csv(path)
        if not records:
            print(f"SKIP: {path} is empty")
            continue

        print(f"{path.name}  ({len(records)} rows) ...", end=" ", flush=True)
        try:
            inserted, skipped = push_in_batches(records, endpoint, api_key, args.batch_size)
            print(f"inserted={inserted}  skipped={skipped}")
            grand_inserted += inserted
            grand_skipped += skipped
        except Exception as e:
            print(f"FAILED — {e}", file=sys.stderr)

    if len(args.files) > 1:
        print(f"\nTotal: inserted={grand_inserted}  skipped={grand_skipped}")


if __name__ == "__main__":
    main()
