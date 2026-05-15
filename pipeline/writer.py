import csv
import os
from datetime import datetime


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


def push_to_db(records: list, endpoint: str):
    """Placeholder for time-series DB push. Wire up when the endpoint is ready."""
    raise NotImplementedError(
        "DB push not yet configured. Call run_sentiment.py without --push-db for now."
    )
