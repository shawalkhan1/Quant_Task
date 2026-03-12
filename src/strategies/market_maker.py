"""
Market Maker Strategy.

Provides liquidity to prediction markets by quoting bid/ask spreads
around an estimated fair value, with inventory control.

Mathematical Formulation:
    - Fair value  p_fair  estimated from features or Black-Scholes
    - Bid = p_fair - spread/2 + inventory_adjustment
    - Ask = p_fair + spread/2 + inventory_adjustment
    - Spread = base_spread + k * volatility
    - Inventory adjustment = -γ * net_inventory
    - Trade when market price crosses our quotes
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)


class MarketMakerStrategy(BaseStrategy):
    """
    Market Maker strategy for prediction markets.

    Provides liquidity by quoting bid/ask prices around estimated fair value.
    Uses inventory control to manage directional risk.
    """

    def __init__(
        self,
        base_spread: float = 0.03,
        vol_sensitivity: float = 0.5,
        inventory_aversion: float = 0.1,
        max_inventory: int = 10,
        risk_config=None,
    ):
        super().__init__(name="MarketMaker", risk_config=risk_config)
        self.base_spread = base_spread
        self.vol_sensitivity = vol_sensitivity
        self.inventory_aversion = inventory_aversion
        self.max_inventory = max_inventory

        # State
        self.net_inventory = 0  # Positive = long YES, negative = long NO
        self.risk_mgr = RiskManager(RiskLimits(
            max_position_pct=0.03,
            max_total_exposure_pct=0.25,
            min_edge_to_trade=0.01,
            kelly_fraction=0.15,
        ))

    def generate_signal(
        self,
        market_row: pd.Series,
        features: Optional[pd.Series],
        timestamp: datetime,
        portfolio_value: float,
        open_positions: Dict[str, list],
    ) -> Optional[dict]:
        """
        Generate market-maker trading signal.

        Logic:
        1. Estimate fair value from various inputs
        2. Compute dynamic spread based on volatility
        3. Compute bid/ask with inventory skew
        4. If market price is below our bid (cheap YES) -> buy YES
        5. If market price is above our ask (expensive YES) -> buy NO
        """
        market_id = market_row.get("market_id", "")
        time_to_expiry = market_row.get("time_to_expiry_min", 0)
        market_price_yes = market_row.get("market_price_yes", 0.5)
        market_price_no = market_row.get("market_price_no", 0.5)
        fair_price = market_row.get("fair_price", 0.5)

        # Don't trade too close to expiry or too early
        if time_to_expiry <= 2 or time_to_expiry > 14:
            return None

        # Don't exceed inventory limits
        if abs(self.net_inventory) >= self.max_inventory:
            return None

        # Skip if already in this market
        if self._already_in_market(market_id, open_positions):
            return None

        # Estimate volatility from features
        vol = 0.6  # default
        if features is not None and hasattr(features, "get"):
            vol = features.get("volatility_15m", 0.6)
            if pd.isna(vol) or vol <= 0:
                vol = 0.6

        # Dynamic spread
        spread = self.base_spread + self.vol_sensitivity * (vol / 100.0)
        spread = max(spread, 0.02)  # minimum spread

        # Inventory skew
        inv_adj = -self.inventory_aversion * self.net_inventory * 0.01

        # Bid/Ask
        bid = fair_price - spread / 2 + inv_adj
        ask = fair_price + spread / 2 + inv_adj

        # Trading logic
        direction = None
        edge = 0.0

        if market_price_yes < bid:
            # Market YES is cheap -> buy YES
            direction = "YES"
            edge = bid - market_price_yes
            predicted_prob = fair_price
        elif market_price_yes > ask:
            # Market YES is expensive -> buy NO (sell YES)
            direction = "NO"
            edge = market_price_yes - ask
            predicted_prob = 1.0 - fair_price
        else:
            return None

        # Position sizing
        size = self.risk_mgr.compute_kelly_size(
            predicted_prob=predicted_prob if direction == "YES" else (1 - predicted_prob),
            market_price=market_price_yes if direction == "YES" else market_price_no,
            portfolio_value=portfolio_value,
            direction=direction,
        )

        if size < 10.0:  # Minimum trade size
            return None

        if direction == "YES":
            self.net_inventory += 1
        else:
            self.net_inventory -= 1

        self._trade_count += 1
        return {
            "action": "trade",
            "direction": direction,
            "size": size,
            "predicted_prob": predicted_prob,
            "confidence": min(edge / spread, 1.0),
            "edge": edge,
            "reason": f"MM: price={market_price_yes:.4f}, bid={bid:.4f}, ask={ask:.4f}",
        }

    def reset(self):
        super().reset()
        self.net_inventory = 0
        self.risk_mgr.reset()
