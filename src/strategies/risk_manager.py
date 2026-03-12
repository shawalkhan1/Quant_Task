"""
Risk Management Module.

Provides risk management utilities for all strategies, including
position sizing, exposure limits, drawdown control, and circuit breakers.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits for a strategy."""
    max_position_pct: float = 0.05
    max_total_exposure_pct: float = 0.30
    max_drawdown_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    max_trades_per_hour: int = 20
    kelly_fraction: float = 0.25
    min_edge_to_trade: float = 0.03


class RiskManager:
    """
    Centralized risk management for the trading platform.

    Responsibilities:
    1. Pre-trade validation
    2. Position sizing
    3. Exposure monitoring
    4. Drawdown circuit breaker
    5. Rate limiting
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._hourly_trade_count = 0
        self._current_hour = None
        self._peak_equity = 0.0
        self._daily_pnl = 0.0
        self._circuit_breaker_active = False

    def pre_trade_check(
        self,
        proposed_size: float,
        proposed_cost: float,
        portfolio_value: float,
        current_exposure: float,
        current_cash: float,
        edge: float,
        timestamp=None,
    ) -> Dict[str, any]:
        """
        Comprehensive pre-trade risk check.

        Returns:
            dict with 'approved' (bool) and 'reason' (str)
        """
        if self._circuit_breaker_active:
            return {"approved": False, "reason": "Circuit breaker active"}

        # Minimum edge check
        if abs(edge) < self.limits.min_edge_to_trade:
            return {
                "approved": False,
                "reason": f"Edge {edge:.4f} below minimum {self.limits.min_edge_to_trade}",
            }

        # Position size check
        if proposed_cost > portfolio_value * self.limits.max_position_pct:
            return {
                "approved": False,
                "reason": f"Position size {proposed_cost:.2f} exceeds limit "
                          f"({portfolio_value * self.limits.max_position_pct:.2f})",
            }

        # Exposure check
        new_exposure = current_exposure + proposed_cost
        if new_exposure > portfolio_value * self.limits.max_total_exposure_pct:
            return {
                "approved": False,
                "reason": f"Total exposure {new_exposure:.2f} would exceed limit",
            }

        # Cash check
        if proposed_cost > current_cash:
            return {"approved": False, "reason": "Insufficient cash"}

        # Rate limit
        if timestamp is not None:
            hour = timestamp.hour if hasattr(timestamp, "hour") else 0
            if hour != self._current_hour:
                self._hourly_trade_count = 0
                self._current_hour = hour
            if self._hourly_trade_count >= self.limits.max_trades_per_hour:
                return {"approved": False, "reason": "Hourly trade limit reached"}

        return {"approved": True, "reason": "OK"}

    def update_after_trade(self):
        """Update state after a trade."""
        self._hourly_trade_count += 1

    def check_drawdown(self, current_equity: float, initial_capital: float):
        """Check and update drawdown status."""
        self._peak_equity = max(self._peak_equity, current_equity)
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.limits.max_drawdown_pct:
                self._circuit_breaker_active = True
                logger.warning(
                    f"CIRCUIT BREAKER: Drawdown {drawdown:.2%} exceeds "
                    f"limit {self.limits.max_drawdown_pct:.2%}"
                )

    def compute_kelly_size(
        self,
        predicted_prob: float,
        market_price: float,
        portfolio_value: float,
        direction: str = "YES",
    ) -> float:
        """
        Compute position size using fractional Kelly criterion.

        For YES position at price p_market:
            b = (1 - p_market) / p_market  (odds)
            f* = (predicted_prob * b - (1-predicted_prob)) / b

        Applies kelly_fraction to reduce volatility.
        """
        if direction == "YES":
            price = market_price
            win_prob = predicted_prob
        else:
            price = 1.0 - market_price
            win_prob = 1.0 - predicted_prob

        if price <= 0 or price >= 1:
            return 0.0

        b = (1.0 - price) / price
        q = 1.0 - win_prob

        kelly_f = (win_prob * b - q) / b
        kelly_f = max(kelly_f, 0.0)

        # Apply fraction and cap
        position_pct = min(
            kelly_f * self.limits.kelly_fraction,
            self.limits.max_position_pct,
        )

        return portfolio_value * position_pct

    def reset(self):
        """Reset risk manager state."""
        self._hourly_trade_count = 0
        self._current_hour = None
        self._peak_equity = 0.0
        self._daily_pnl = 0.0
        self._circuit_breaker_active = False
