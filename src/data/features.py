"""
Feature Engineering Pipeline.

Generates technical, momentum, volatility, volume, and market-specific
features for prediction market strategies.

BIAS PREVENTION: All features use only data available at or before
the current timestamp — no look-ahead.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Computes features from OHLCV data for use in prediction market models.

    All features are strictly backward-looking to prevent look-ahead bias.
    Features are shifted by 1 bar where needed to ensure the signal
    at time t uses only data available at time t.
    """

    def __init__(self):
        self.feature_names: List[str] = []

    def compute_all_features(
        self,
        price_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.

        Parameters:
            price_data: DataFrame with open, high, low, close, volume
            market_data: Optional market data with market_price_yes, strike, etc.

        Returns:
            DataFrame with all computed features (NaN rows at the beginning are expected)
        """
        df = price_data.copy()

        # --- Price-based features ---
        df["return_1m"] = df["close"].pct_change(1)
        df["return_5m"] = df["close"].pct_change(5)
        df["return_15m"] = df["close"].pct_change(15)
        df["return_30m"] = df["close"].pct_change(30)
        df["return_60m"] = df["close"].pct_change(60)
        df["log_return_1m"] = np.log(df["close"] / df["close"].shift(1))

        # --- Momentum features ---
        df["rsi_14"] = self._rsi(df["close"], 14)
        df["rsi_7"] = self._rsi(df["close"], 7)
        macd, macd_signal, macd_hist = self._macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_histogram"] = macd_hist
        df["roc_5"] = self._rate_of_change(df["close"], 5)
        df["roc_15"] = self._rate_of_change(df["close"], 15)

        # Williams %R
        df["williams_r_14"] = self._williams_r(df["high"], df["low"], df["close"], 14)

        # --- Volatility features ---
        df["volatility_5m"] = df["log_return_1m"].rolling(5).std() * np.sqrt(525600)
        df["volatility_15m"] = df["log_return_1m"].rolling(15).std() * np.sqrt(525600)
        df["volatility_60m"] = df["log_return_1m"].rolling(60).std() * np.sqrt(525600)
        df["parkinson_vol_15"] = self._parkinson_volatility(df["high"], df["low"], 15)
        df["atr_14"] = self._atr(df["high"], df["low"], df["close"], 14)

        # Volatility ratio (short/long)
        df["vol_ratio"] = df["volatility_5m"] / (df["volatility_60m"] + 1e-10)

        # --- Volume features ---
        df["volume_sma_15"] = df["volume"].rolling(15).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_15"] + 1e-10)
        df["volume_momentum"] = df["volume"].pct_change(5)

        # VWAP
        df["vwap_15"] = (
            (df["close"] * df["volume"]).rolling(15).sum()
            / (df["volume"].rolling(15).sum() + 1e-10)
        )
        df["price_vs_vwap"] = (df["close"] - df["vwap_15"]) / (df["vwap_15"] + 1e-10)

        # --- Price position features ---
        df["price_vs_sma_15"] = (df["close"] - df["close"].rolling(15).mean()) / (
            df["close"].rolling(15).mean() + 1e-10
        )
        df["price_vs_sma_60"] = (df["close"] - df["close"].rolling(60).mean()) / (
            df["close"].rolling(60).mean() + 1e-10
        )

        # Bollinger Band position
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_position"] = (df["close"] - sma_20) / (2 * std_20 + 1e-10)

        # --- Time features (cyclical encoding) ---
        if hasattr(df.index, "hour"):
            hour = df.index.hour
            minute = df.index.minute
            day_of_week = df.index.dayofweek
        else:
            hour = pd.Series(0, index=df.index)
            minute = pd.Series(0, index=df.index)
            day_of_week = pd.Series(0, index=df.index)

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * minute / 60)
        df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        # --- Higher-order features ---
        df["return_acceleration"] = df["return_1m"] - df["return_1m"].shift(1)
        df["vol_of_vol"] = df["volatility_5m"].rolling(15).std()

        # Store feature names (exclude OHLCV columns)
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        self.feature_names = [c for c in df.columns if c not in ohlcv_cols]

        logger.info(f"Generated {len(self.feature_names)} features")
        return df

    def add_market_features(
        self,
        features_df: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add prediction-market-specific features.

        Parameters:
            features_df: DataFrame with price features
            market_data: DataFrame with market_price_yes, strike, etc.

        Returns:
            Enhanced DataFrame with market features
        """
        # Market data may have duplicate timestamps (multiple markets per bar).
        # We merge on index to replicate the feature rows for each market row.
        df = market_data.copy()

        # Join price-based features onto market data by timestamp
        feat_cols = [c for c in features_df.columns if c not in df.columns]
        if feat_cols:
            feat_sub = features_df[feat_cols]
            # Use a left join on the market index to expand features to every market row
            df = df.join(feat_sub, how="left")

        if "strike" in df.columns and "close_price" in df.columns:
            df["distance_to_strike"] = (
                (df["close_price"] - df["strike"])
                / df["strike"]
            )
            df["abs_distance_to_strike"] = df["distance_to_strike"].abs()

        if "time_to_expiry_min" in df.columns:
            df["time_to_expiry"] = df["time_to_expiry_min"] / 15.0  # normalize
            df["time_to_expiry_sq"] = df["time_to_expiry"] ** 2

        if "market_price_yes" in df.columns:
            df["market_implied_prob"] = df["market_price_yes"]
            df["market_price_momentum"] = df["market_price_yes"].diff(1)

        if "implied_vol" in df.columns:
            df["market_implied_vol"] = df["implied_vol"]

        # Update feature names
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        self.feature_names = [c for c in df.columns if c not in ohlcv_cols]

        return df

    # ──────────────────────────────────────────
    # Technical indicator helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _rate_of_change(series: pd.Series, period: int) -> pd.Series:
        return series.pct_change(period)

    @staticmethod
    def _williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return -100 * (hh - close) / (hh - ll + 1e-10)

    @staticmethod
    def _parkinson_volatility(
        high: pd.Series, low: pd.Series, window: int = 15
    ) -> pd.Series:
        log_hl = np.log(high / (low + 1e-10))
        return np.sqrt(
            (1 / (4 * np.log(2))) * (log_hl**2).rolling(window).mean()
        ) * np.sqrt(525600)

    @staticmethod
    def _atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def get_feature_names(self, include_market: bool = True) -> List[str]:
        """Return the list of computed feature names."""
        return self.feature_names.copy()
