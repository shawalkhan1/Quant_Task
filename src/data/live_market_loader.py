"""Utilities to load live Polymarket market data for app workflows."""

from typing import Tuple

import pandas as pd

from src.data.features import FeatureEngine
from src.data.polymarket_fetcher import PolymarketFetcher


def build_price_proxy_from_markets(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build OHLCV-like price data from market probability observations.

    The feature engine expects OHLCV columns. We use the aggregated YES
    probability as a proxy close price and derive narrow high/low bands.
    """
    if market_data.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    grouped_yes = market_data.groupby(market_data.index)["market_price_yes"].mean().sort_index()
    grouped_vol = market_data.groupby(market_data.index)["volume_usd"].sum().sort_index().fillna(1.0)

    price_data = pd.DataFrame(
        {
            "open": grouped_yes,
            "high": grouped_yes * 1.001,
            "low": grouped_yes * 0.999,
            "close": grouped_yes,
            "volume": grouped_vol,
        }
    )
    price_data.index.name = "timestamp"
    return price_data


def load_live_polymarket_data(
    days_back: int,
    min_volume: float,
    max_markets: int,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch live Polymarket data and return (price_data, market_data, features)."""
    fetcher = PolymarketFetcher()
    market_data = fetcher.fetch_dataset_for_backtest(
        days_back=days_back,
        min_volume=min_volume,
        max_markets=max_markets,
        use_cache=use_cache,
    )

    if market_data.empty:
        raise RuntimeError(
            "No live Polymarket market data returned by API. "
            "Check internet connectivity or lower min volume / increase lookback."
        )

    market_data = market_data.sort_index()
    price_data = build_price_proxy_from_markets(market_data)

    engine = FeatureEngine()
    features = engine.compute_all_features(price_data)
    features = engine.add_market_features(features, market_data)

    return price_data, market_data, features
