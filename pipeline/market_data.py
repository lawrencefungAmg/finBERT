import requests


def fetch_tickers(base_url: str) -> list[dict]:
    """Fetch the firm's research-scope tickers from the market-data-service.

    Returns a list of {ticker, name} dicts compatible with the pipeline's
    existing stock record format.
    """
    r = requests.get(f"{base_url.rstrip('/')}/api/v1/tickers", timeout=10)
    r.raise_for_status()
    return [
        {"ticker": t["symbol"], "name": t.get("name", t["symbol"])}
        for t in r.json()
    ]
