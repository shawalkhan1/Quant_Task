"""Container smoke test: Polymarket reachability + live loader check."""

import os
import sys

# Allow direct execution via: python docker/smoke_test.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from src.data.live_market_loader import load_live_polymarket_data


def main() -> int:
    gamma = requests.get("https://gamma-api.polymarket.com/markets", params={"limit": 3}, timeout=20)
    print("gamma_status", gamma.status_code)
    if gamma.status_code != 200:
        return 1

    payload = gamma.json()
    print("gamma_items", len(payload) if isinstance(payload, list) else -1)
    if not isinstance(payload, list) or len(payload) == 0:
        return 1

    price_data, market_data, features = load_live_polymarket_data(
        days_back=14,
        min_volume=100.0,
        max_markets=3,
        use_cache=False,
    )

    print("price_rows", len(price_data))
    print("market_rows", len(market_data))
    print("feature_rows", len(features))
    print("unique_markets", int(market_data["market_id"].nunique()) if len(market_data) else 0)

    if len(market_data) == 0 or len(features) == 0:
        return 1

    required = {
        "market_id",
        "market_price_yes",
        "market_price_no",
        "fair_price",
        "resolution",
        "time_to_expiry_min",
        "minutes_elapsed",
    }
    missing = sorted(required - set(market_data.columns))
    print("missing_columns", missing)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
