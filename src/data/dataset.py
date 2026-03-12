"""
Time-indexed dataset management.

Provides a clean API for accessing data while preventing look-ahead bias.
Supports proper train/test splitting by time.
"""

import logging
from datetime import datetime
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """
    Manages time-indexed datasets for backtesting and model training.

    KEY BIAS PREVENTION MEASURES:
    1. Data is always accessed in chronological order
    2. Train/test split is ALWAYS by time (never random)
    3. The access API prevents peeking into future data
    4. Feature computation uses only backward-looking windows
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        market_data: pd.DataFrame,
        features_data: pd.DataFrame,
    ):
        # Validate all data is time-indexed
        for name, df in [("price", price_data), ("market", market_data), ("features", features_data)]:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{name} data must have DatetimeIndex")

        self.price_data = price_data.sort_index()
        self.market_data = market_data.sort_index()
        self.features_data = features_data.sort_index()

        self._validate_alignment()
        logger.info(
            f"Dataset initialized: {len(self.price_data)} price bars, "
            f"{len(self.market_data)} market observations"
        )

    def _validate_alignment(self):
        """Check for timestamp alignment issues."""
        if len(self.market_data) > 0 and len(self.price_data) > 0:
            market_start = self.market_data.index.min()
            market_end = self.market_data.index.max()
            price_start = self.price_data.index.min()
            price_end = self.price_data.index.max()

            if market_start < price_start or market_end > price_end:
                logger.warning(
                    "Market data range extends beyond price data range. "
                    "This may cause issues."
                )

    def time_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.0,
    ) -> dict:
        """
        Split data by time into train/val/test sets.
        NEVER uses random splitting to prevent data leakage.

        Returns dict with keys: 'train', 'val', 'test'
        Each value is a dict with 'price', 'market', 'features' DataFrames.
        """
        n = len(self.price_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = self.price_data.index[:train_end]
        val_idx = self.price_data.index[train_end:val_end] if val_ratio > 0 else pd.DatetimeIndex([])
        test_idx = self.price_data.index[val_end if val_ratio > 0 else train_end:]

        splits = {}
        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            if len(idx) == 0:
                splits[name] = None
                continue
            start, end = idx[0], idx[-1]
            splits[name] = {
                "price": self.price_data.loc[start:end],
                "market": self.market_data.loc[start:end] if len(self.market_data) > 0 else pd.DataFrame(),
                "features": self.features_data.loc[start:end] if len(self.features_data) > 0 else pd.DataFrame(),
            }

        train_end_ts = train_idx[-1] if len(train_idx) > 0 else "N/A"
        test_start_ts = test_idx[0] if len(test_idx) > 0 else "N/A"
        logger.info(f"Time split - Train ends: {train_end_ts}, Test starts: {test_start_ts}")
        return splits

    def walk_forward_splits(
        self,
        train_days: int = 7,
        test_days: int = 2,
        step_days: int = 1,
    ) -> List[dict]:
        """
        Generate walk-forward optimization splits.

        Yields multiple (train, test) pairs where each test period
        follows the corresponding train period chronologically.
        """
        from datetime import timedelta

        splits = []
        total_minutes = (self.price_data.index[-1] - self.price_data.index[0]).total_seconds() / 60
        start = self.price_data.index[0]

        while True:
            train_end = start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            if test_end > self.price_data.index[-1]:
                break

            train_data = {
                "price": self.price_data.loc[start:train_end],
                "market": self.market_data.loc[start:train_end],
                "features": self.features_data.loc[start:train_end],
            }
            test_data = {
                "price": self.price_data.loc[train_end:test_end],
                "market": self.market_data.loc[train_end:test_end],
                "features": self.features_data.loc[train_end:test_end],
            }

            splits.append({"train": train_data, "test": test_data})
            start += timedelta(days=step_days)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def get_data_at(self, timestamp: pd.Timestamp) -> dict:
        """
        Get all data up to and including the given timestamp.
        This is the safe way to access data during backtesting.
        """
        return {
            "price": self.price_data.loc[:timestamp],
            "market": self.market_data.loc[:timestamp],
            "features": self.features_data.loc[:timestamp],
        }

    def get_market_ids(self) -> List[str]:
        """Return unique market IDs."""
        if "market_id" in self.market_data.columns:
            return self.market_data["market_id"].unique().tolist()
        return []

    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.price_data.index[0], self.price_data.index[-1]

    @property
    def num_bars(self) -> int:
        return len(self.price_data)

    @property
    def num_markets(self) -> int:
        if "market_id" in self.market_data.columns:
            return self.market_data["market_id"].nunique()
        return 0
