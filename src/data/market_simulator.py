"""
Prediction Market Simulator.

Generates realistic 15-minute binary prediction markets from crypto price data.
Uses Black-Scholes digital option pricing with added noise to simulate
market inefficiencies.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class PredictionMarketSimulator:
    """
    Simulates prediction markets of the form:
      "Will BTC be above price X in 15 minutes?"

    Market prices are generated using digital option pricing (Black-Scholes)
    with Gaussian noise to simulate market inefficiencies.
    """

    def __init__(
        self,
        market_duration_minutes: int = 15,
        noise_std: float = 0.05,
        fee_rate: float = 0.02,
        risk_free_rate: float = 0.05,
        min_price: float = 0.01,
        max_price: float = 0.99,
        seed: int = 42,
    ):
        self.market_duration = market_duration_minutes
        self.noise_std = noise_std
        self.fee_rate = fee_rate
        self.risk_free_rate = risk_free_rate
        self.min_price = min_price
        self.max_price = max_price
        self.rng = np.random.RandomState(seed)

    def _estimate_volatility(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """
        Estimate rolling annualized volatility from 1-minute returns.
        Uses Parkinson volatility estimator if OHLC is available.
        """
        log_returns = np.log(prices / prices.shift(1))
        # Annualize: sqrt(525600) for minutes in a year
        vol = log_returns.rolling(window=window, min_periods=10).std() * np.sqrt(525600)
        vol = vol.bfill().fillna(0.6)  # Default ~60% annual vol
        return vol

    def _digital_option_price(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: float = 0.05,
    ) -> np.ndarray:
        """
        Black-Scholes digital (binary) call option price.

        C_digital = e^{-rT} * N(d2)

        where d2 = [ln(S/K) + (r - sigma^2/2)*T] / (sigma * sqrt(T))

        Parameters:
            S: Current price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Annualized volatility
            r: Risk-free rate
        Returns:
            Fair probability estimate
        """
        T = np.maximum(T, 1e-10)  # prevent division by zero
        sigma = np.maximum(sigma, 0.01)

        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        price = np.exp(-r * T) * norm.cdf(d2)
        return price

    def generate_markets(
        self,
        price_data: pd.DataFrame,
        strike_offset: float = 0.0,
        symbol: str = "BTC/USDT",
    ) -> pd.DataFrame:
        """
        Generate prediction markets from price data.

        For every `market_duration` minutes, creates a market:
          - Strike = price at market open + offset
          - Resolution = 1 if price at close > strike, else 0
          - Market prices at each minute = digital option price + noise

        Parameters:
            price_data: DataFrame with 'close' column and datetime index
            strike_offset: Fraction to offset strike (0 = at-the-money)
            symbol: Market identifier

        Returns:
            DataFrame with columns:
                market_id, timestamp, close_price, strike, time_to_expiry,
                fair_price, market_price_yes, market_price_no, resolution,
                implied_vol, minutes_elapsed
        """
        close_prices = price_data["close"].values
        timestamps = price_data.index
        n = len(close_prices)
        duration = self.market_duration

        # Estimate rolling volatility
        vol_series = self._estimate_volatility(price_data["close"]).values

        records = []
        market_count = 0

        for market_start in range(0, n - duration, duration):
            market_end = market_start + duration
            if market_end >= n:
                break

            # Strike price = price at market open (with optional offset)
            strike = close_prices[market_start] * (1.0 + strike_offset)

            # Resolution: did price end above strike?
            resolution = 1 if close_prices[market_end] > strike else 0

            market_id = f"{symbol.replace('/', '')}_{timestamps[market_start].strftime('%Y%m%d_%H%M')}"
            market_count += 1

            # Generate market prices at each minute within this market
            for t in range(duration + 1):
                idx = market_start + t
                if idx >= n:
                    break

                S = close_prices[idx]
                K = strike
                remaining_minutes = duration - t
                T_years = remaining_minutes / 525600.0  # Convert minutes to years

                sigma = vol_series[idx]

                # Fair price from Black-Scholes digital option
                if remaining_minutes <= 0:
                    fair_price = float(resolution)
                else:
                    fair_price = self._digital_option_price(
                        np.array([S]), np.array([K]),
                        np.array([T_years]), np.array([sigma]),
                        self.risk_free_rate,
                    )[0]

                # Add noise to simulate market inefficiency
                noise = self.rng.normal(0, self.noise_std)
                # Noise decreases as we approach expiry (markets get more efficient)
                noise_decay = remaining_minutes / duration
                market_price_yes = np.clip(
                    fair_price + noise * noise_decay,
                    self.min_price,
                    self.max_price,
                )
                market_price_no = np.clip(
                    1.0 - market_price_yes + self.rng.normal(0, 0.01),
                    self.min_price,
                    self.max_price,
                )

                records.append({
                    "market_id": market_id,
                    "timestamp": timestamps[idx],
                    "close_price": S,
                    "strike": K,
                    "time_to_expiry_min": remaining_minutes,
                    "implied_vol": sigma,
                    "fair_price": fair_price,
                    "market_price_yes": market_price_yes,
                    "market_price_no": market_price_no,
                    "resolution": resolution,
                    "minutes_elapsed": t,
                    "symbol": symbol,
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)

        logger.info(
            f"Generated {market_count} prediction markets with "
            f"{len(df)} price observations for {symbol}"
        )
        return df

    def generate_multi_strike_markets(
        self,
        price_data: pd.DataFrame,
        offsets: list = None,
        symbol: str = "BTC/USDT",
    ) -> pd.DataFrame:
        """Generate markets at multiple strike offsets for richer data."""
        if offsets is None:
            offsets = [-0.002, -0.001, 0.0, 0.001, 0.002]

        all_markets = []
        for offset in offsets:
            markets = self.generate_markets(price_data, strike_offset=offset, symbol=symbol)
            if len(markets) > 0:
                all_markets.append(markets)

        if all_markets:
            return pd.concat(all_markets).sort_index()
        return pd.DataFrame()
