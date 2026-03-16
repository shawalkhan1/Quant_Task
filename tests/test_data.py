"""Tests for the data layer — live loader, features, and dataset utilities."""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.data.live_market_loader as live_market_loader
from src.data.features import FeatureEngine
from src.data.dataset import TimeSeriesDataset


# --------------------------------------------------------------------------- #
#  Live loader contract tests
# --------------------------------------------------------------------------- #
class TestLiveLoader:
    """Test live Polymarket loader contracts without real network dependency."""

    @pytest.fixture
    def sample_market_data(self):
        idx = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
        return pd.DataFrame(
            {
                "market_id": ["m1"] * 15 + ["m2"] * 15,
                "market_price_yes": np.linspace(0.4, 0.7, 30),
                "market_price_no": np.linspace(0.6, 0.3, 30),
                "fair_price": np.linspace(0.4, 0.7, 30),
                "resolution": [1] * 15 + [0] * 15,
                "time_to_expiry_min": list(range(15, 0, -1)) + list(range(15, 0, -1)),
                "minutes_elapsed": list(range(15)) + list(range(15)),
                "volume_usd": np.random.uniform(100, 2000, 30),
                "liquidity_usd": np.random.uniform(100, 2000, 30),
                "symbol": ["Polymarket"] * 30,
            },
            index=idx,
        )

    def test_build_price_proxy(self, sample_market_data):
        price_data = live_market_loader.build_price_proxy_from_markets(sample_market_data)
        assert isinstance(price_data, pd.DataFrame)
        assert len(price_data) > 0
        assert all(c in price_data.columns for c in ["open", "high", "low", "close", "volume"])
        assert price_data["high"].ge(price_data["low"]).all()

    def test_load_live_polymarket_data_contract(self, monkeypatch, sample_market_data):
        class _DummyFetcher:
            def fetch_dataset_for_backtest(self, **kwargs):
                return sample_market_data

        monkeypatch.setattr(live_market_loader, "PolymarketFetcher", _DummyFetcher)
        price_data, market_data, features = live_market_loader.load_live_polymarket_data(
            days_back=7,
            min_volume=100.0,
            max_markets=5,
            use_cache=False,
        )

        assert len(price_data) > 0
        assert len(market_data) > 0
        assert len(features) > 0
        assert "market_price_yes" in market_data.columns
        assert "resolution" in market_data.columns

    def test_load_live_polymarket_data_empty_raises(self, monkeypatch):
        class _DummyFetcher:
            def fetch_dataset_for_backtest(self, **kwargs):
                return pd.DataFrame()

        monkeypatch.setattr(live_market_loader, "PolymarketFetcher", _DummyFetcher)
        with pytest.raises(RuntimeError):
            live_market_loader.load_live_polymarket_data(
                days_back=7,
                min_volume=100.0,
                max_markets=5,
                use_cache=False,
            )


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
