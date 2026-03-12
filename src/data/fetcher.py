"""
Cryptocurrency price data fetcher using ccxt (Binance).
Fetches OHLCV data at 1-minute granularity and caches locally.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_ccxt():
    """Try to import ccxt; return None if unavailable."""
    try:
        import ccxt
        return ccxt
    except ImportError:
        return None


class DataFetcher:
    """
    Fetches and caches cryptocurrency OHLCV data from Binance via ccxt.
    Falls back to synthetic data generation if ccxt is unavailable.
    """

    def __init__(self, exchange_id: str = "binance", data_dir: str = None):
        from config.settings import RAW_DATA_DIR
        self.data_dir = data_dir or RAW_DATA_DIR
        self.exchange_id = exchange_id
        os.makedirs(self.data_dir, exist_ok=True)

        ccxt = _try_import_ccxt()
        if ccxt is not None:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self.exchange = exchange_class({"enableRateLimit": True})
                logger.info(f"Initialized {exchange_id} exchange via ccxt")
            except Exception as e:
                logger.warning(f"Could not init {exchange_id}: {e}. Using synthetic data.")
                self.exchange = None
        else:
            logger.warning("ccxt not installed. Will generate synthetic data.")
            self.exchange = None

    def _cache_path(self, symbol: str, timeframe: str, days: int) -> str:
        safe_symbol = symbol.replace("/", "_")
        return os.path.join(self.data_dir, f"{safe_symbol}_{timeframe}_{days}d.csv")

    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        days: int = 30,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Uses cache if available, falls back to synthetic data
        if the exchange is unavailable.
        """
        cache_path = self._cache_path(symbol, timeframe, days)

        # Check cache
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
            return df

        # Try exchange
        if self.exchange is not None:
            try:
                df = self._fetch_from_exchange(symbol, timeframe, days)
                df.to_csv(cache_path)
                logger.info(f"Fetched {len(df)} bars from {self.exchange_id}, cached to {cache_path}")
                return df
            except Exception as e:
                logger.warning(f"Exchange fetch failed: {e}. Generating synthetic data.")

        # Fallback: synthetic
        df = self._generate_synthetic_data(symbol, timeframe, days)
        df.to_csv(cache_path)
        logger.info(f"Generated {len(df)} synthetic bars, cached to {cache_path}")
        return df

    def _fetch_from_exchange(
        self, symbol: str, timeframe: str, days: int
    ) -> pd.DataFrame:
        """Fetch OHLCV from exchange using ccxt with pagination."""
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        limit = 1000
        all_candles = []

        while True:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            time.sleep(self.exchange.rateLimit / 1000)

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        return df

    def _generate_synthetic_data(
        self, symbol: str, timeframe: str, days: int
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data using geometric Brownian motion.
        This produces data with realistic statistical properties for backtesting.
        """
        np.random.seed(42 if "BTC" in symbol else 123)

        # Parameters based on typical crypto
        if "BTC" in symbol:
            initial_price = 65000.0
            annual_vol = 0.60
            avg_volume = 150.0
        elif "ETH" in symbol:
            initial_price = 3500.0
            annual_vol = 0.70
            avg_volume = 2000.0
        else:
            initial_price = 100.0
            annual_vol = 0.50
            avg_volume = 1000.0

        minutes_per_bar = 1
        total_bars = days * 24 * 60 // minutes_per_bar
        dt = minutes_per_bar / (365.25 * 24 * 60)
        drift = 0.0  # neutral drift

        # GBM for close prices
        returns = np.random.normal(drift * dt, annual_vol * np.sqrt(dt), total_bars)
        # Add some autocorrelation and mean reversion
        for i in range(1, len(returns)):
            returns[i] += -0.05 * returns[i - 1]  # slight mean reversion

        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV from close prices
        opens = np.roll(prices, 1)
        opens[0] = initial_price

        # High/Low with realistic wicks
        wick_up = np.abs(np.random.normal(0, annual_vol * np.sqrt(dt) * initial_price * 0.3, total_bars))
        wick_down = np.abs(np.random.normal(0, annual_vol * np.sqrt(dt) * initial_price * 0.3, total_bars))
        highs = np.maximum(opens, prices) + wick_up
        lows = np.minimum(opens, prices) - wick_down

        # Volume with some clustering
        log_vol = np.random.normal(np.log(avg_volume), 0.5, total_bars)
        for i in range(1, len(log_vol)):
            log_vol[i] = 0.7 * log_vol[i - 1] + 0.3 * log_vol[i]  # vol clustering
        volumes = np.exp(log_vol)

        # Timestamps
        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start_time = end_time - timedelta(minutes=total_bars)
        timestamps = pd.date_range(start=start_time, periods=total_bars, freq="1min", tz="UTC")

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=timestamps,
        )
        df.index.name = "timestamp"
        return df

    def fetch_multiple(
        self, symbols: List[str], timeframe: str = "1m", days: int = 30
    ) -> dict:
        """Fetch data for multiple symbols."""
        data = {}
        for sym in symbols:
            data[sym] = self.fetch_ohlcv(sym, timeframe, days)
        return data
