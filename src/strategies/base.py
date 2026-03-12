"""
Abstract base class for all trading strategies.

Defines the interface that all strategies must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Signal:
    """Trading signal from a strategy."""
    action: str         # "trade" or "hold"
    direction: str      # "YES" or "NO"
    size: float         # Dollar amount
    predicted_prob: float
    confidence: float
    edge: float
    reason: str = ""


class BaseStrategy(ABC):
    """
    Abstract base class for prediction market trading strategies.

    All strategies must implement:
        - generate_signal(): Produce a trading signal given current data
        - name: Strategy name
    """

    def __init__(self, name: str, risk_config=None):
        self.name = name
        if risk_config is None:
            from config.settings import DEFAULT_RISK_CONFIG
            risk_config = DEFAULT_RISK_CONFIG
        self.risk_config = risk_config
        self._trade_count = 0

    @abstractmethod
    def generate_signal(
        self,
        market_row: pd.Series,
        features: Optional[pd.Series],
        timestamp: datetime,
        portfolio_value: float,
        open_positions: Dict[str, list],
    ) -> Optional[dict]:
        """
        Generate a trading signal for the given market and data.

        Parameters:
            market_row: Current market data (price, strike, time to expiry, etc.)
            features: Current feature values
            timestamp: Current time
            portfolio_value: Current portfolio value
            open_positions: Dict of open positions by market_id

        Returns:
            Signal dictionary with keys: action, direction, size, predicted_prob,
            confidence, edge. Returns None for no action.
        """
        pass

    def compute_position_size(
        self,
        edge: float,
        predicted_prob: float,
        portfolio_value: float,
        market_price: float,
    ) -> float:
        """
        Compute position size using fractional Kelly criterion.

        f* = kelly_fraction * (p*b - q) / b

        where:
            p = predicted probability of winning
            b = odds (payout ratio)
            q = 1 - p
            kelly_fraction = 0.25 (quarter Kelly for safety)
        """
        if edge <= 0 or predicted_prob <= 0 or predicted_prob >= 1:
            return 0.0

        # Odds: if we buy YES at market_price, payout is 1.0
        # So b = (1 - market_price) / market_price
        if market_price <= 0 or market_price >= 1:
            return 0.0

        b = (1.0 - market_price) / market_price
        q = 1.0 - predicted_prob

        kelly_f = (predicted_prob * b - q) / b
        kelly_f = max(kelly_f, 0.0)

        # Apply fraction
        fraction = self.risk_config.kelly_fraction
        position_pct = kelly_f * fraction

        # Cap at max position size
        position_pct = min(position_pct, self.risk_config.max_position_pct)

        size = portfolio_value * position_pct
        return max(size, 0.0)

    def _already_in_market(
        self, market_id: str, open_positions: Dict[str, list]
    ) -> bool:
        """Check if already holding a position in this market."""
        return market_id in open_positions and len(open_positions[market_id]) > 0

    def reset(self):
        """Reset strategy state."""
        self._trade_count = 0
