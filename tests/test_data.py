"""Tests for the data layer — fetcher, market simulator, features, dataset."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.fetcher import DataFetcher
from src.data.market_simulator import PredictionMarketSimulator
from src.data.features import FeatureEngine
from src.data.dataset import TimeSeriesDataset


# --------------------------------------------------------------------------- #
#  DataFetcher tests
# --------------------------------------------------------------------------- #
class TestDataFetcher:
    """Test the DataFetcher synthetic data generation."""

    def test_generate_synthetic_btc(self):
        fetcher = DataFetcher()
        df = fetcher._generate_synthetic_data("BTC/USDT", "1m", days=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])
        assert df["high"].ge(df["low"]).all()
        assert df["high"].ge(df["open"]).all()
        assert df["high"].ge(df["close"]).all()
        assert df["low"].le(df["open"]).all()
        assert df["low"].le(df["close"]).all()

    def test_synthetic_length(self):
        fetcher = DataFetcher()
        df = fetcher._generate_synthetic_data("ETH/USDT", "1m", days=1)
        expected = 24 * 60  # one bar per minute
        assert abs(len(df) - expected) <= 1

    def test_synthetic_prices_positive(self):
        fetcher = DataFetcher()
        df = fetcher._generate_synthetic_data("BTC/USDT", "1m", days=1)
        assert (df["close"] > 0).all()
        assert (df["volume"] > 0).all()


# --------------------------------------------------------------------------- #
#  PredictionMarketSimulator tests
# --------------------------------------------------------------------------- #
class TestMarketSimulator:
    """Test prediction market generation."""

    @pytest.fixture
    def price_data(self):
        fetcher = DataFetcher()
        return fetcher._generate_synthetic_data("BTC/USDT", "1m", days=2)

    def test_generate_markets(self, price_data):
        sim = PredictionMarketSimulator()
        markets = sim.generate_markets(price_data)
        assert isinstance(markets, pd.DataFrame)
        assert len(markets) > 0

    def test_market_columns(self, price_data):
        sim = PredictionMarketSimulator()
        markets = sim.generate_markets(price_data)
        required = [
            "market_id", "fair_price", "market_price_yes",
            "market_price_no", "resolution",
        ]
        for col in required:
            assert col in markets.columns, f"Missing column: {col}"

    def test_market_prices_bounded(self, price_data):
        sim = PredictionMarketSimulator()
        markets = sim.generate_markets(price_data)
        assert markets["market_price_yes"].between(0, 1).all()
        assert markets["market_price_no"].between(0, 1).all()

    def test_resolution_binary(self, price_data):
        sim = PredictionMarketSimulator()
        markets = sim.generate_markets(price_data)
        assert markets["resolution"].isin([0.0, 1.0]).all()


# --------------------------------------------------------------------------- #
#  FeatureEngine tests
# --------------------------------------------------------------------------- #
class TestFeatureEngine:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 500
        close = 65000 + np.cumsum(np.random.randn(n) * 50)
        idx = pd.date_range("2025-01-01", periods=n, freq="min")
        return pd.DataFrame({
            "open": close + np.random.randn(n),
            "high": close + abs(np.random.randn(n) * 10),
            "low": close - abs(np.random.randn(n) * 10),
            "close": close,
            "volume": np.random.uniform(10, 100, n),
        }, index=idx)

    def test_compute_all_features(self, sample_df):
        engine = FeatureEngine()
        result = engine.compute_all_features(sample_df)
        assert len(result.columns) > len(sample_df.columns)

    def test_no_look_ahead(self, sample_df):
        """Features should not use future data — adding rows at the end
        should NOT change earlier feature values."""
        engine = FeatureEngine()
        result1 = engine.compute_all_features(sample_df.copy())

        # Extend with 10 more rows
        extra = sample_df.iloc[-10:].copy()
        extra.index = extra.index + pd.Timedelta(minutes=len(sample_df))
        extended = pd.concat([sample_df, extra])
        engine2 = FeatureEngine()
        result2 = engine2.compute_all_features(extended)

        # Values for original rows should be identical
        common_idx = result1.dropna().index[:100]
        for col in result1.columns:
            if col in result2.columns:
                diff = (result1.loc[common_idx, col] - result2.loc[common_idx, col]).abs()
                assert diff.max() < 1e-10, f"Look-ahead detected in {col}"

    def test_features_numeric(self, sample_df):
        engine = FeatureEngine()
        result = engine.compute_all_features(sample_df)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"


# --------------------------------------------------------------------------- #
#  TimeSeriesDataset tests
# --------------------------------------------------------------------------- #
class TestTimeSeriesDataset:
    """Test dataset splitting logic."""

    @pytest.fixture
    def dataset(self):
        n = 1000
        idx = pd.date_range("2025-01-01", periods=n, freq="min")
        price_data = pd.DataFrame({
            "open": np.random.randn(n) + 100,
            "high": np.random.randn(n) + 101,
            "low": np.random.randn(n) + 99,
            "close": np.random.randn(n) + 100,
            "volume": np.random.uniform(10, 100, n),
        }, index=idx)
        market_data = pd.DataFrame({
            "market_id": [f"m{i}" for i in range(n)],
            "market_price_yes": np.random.uniform(0.3, 0.7, n),
            "market_price_no": np.random.uniform(0.3, 0.7, n),
            "fair_price": np.random.uniform(0.4, 0.6, n),
            "resolution": np.random.choice([0, 1], n),
        }, index=idx)
        features_data = pd.DataFrame({
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n),
        }, index=idx)
        return TimeSeriesDataset(price_data, market_data, features_data)

    def test_time_split(self, dataset):
        result = dataset.time_split(train_ratio=0.7)
        assert "train" in result
        assert "test" in result

    def test_num_bars(self, dataset):
        assert dataset.num_bars == 1000

    def test_date_range(self, dataset):
        start, end = dataset.date_range
        assert start < end

    def test_get_market_ids(self, dataset):
        ids = dataset.get_market_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0
