"""
Arbitrage Strategy.

Detects pricing inconsistencies in prediction markets and trades to
capture risk-free or low-risk profits.

Mathematical Formulation:
    Type 1 — YES/NO Imbalance:
        If p_yes + p_no < 1 - fees: buy both sides for guaranteed profit
        If p_yes + p_no > 1 + fees: prices are inflated (avoid or short)

    Type 2 — Cross-Market Mispricing:
        If same underlying event is priced differently across strikes,
        detect and exploit the divergence.

    Type 3 — Fair Value Deviation:
        When market price significantly deviates from BS fair value,
        trade toward fair value.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)


class ArbitrageStrategy(BaseStrategy):
    """
    Arbitrage strategy for prediction markets.

    Exploits three types of inefficiencies:
    1. YES/NO price imbalance (p_yes + p_no ≠ 1)
    2. Cross-market arbitrage (same underlying, different strikes mispriced)
    3. Fair value deviation (market price vs model price)
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.02,
        fair_value_threshold: float = 0.06,
        cross_market_threshold: float = 0.05,
        fee_buffer: float = 0.01,
        risk_config=None,
    ):
        super().__init__(name="Arbitrage", risk_config=risk_config)
        self.imbalance_threshold = imbalance_threshold
        self.fair_value_threshold = fair_value_threshold
        self.cross_market_threshold = cross_market_threshold
        self.fee_buffer = fee_buffer

        # Cache of recent market prices for cross-market comparison
        self._market_price_cache: Dict[str, Dict] = {}

        self.risk_mgr = RiskManager(RiskLimits(
            max_position_pct=0.04,
            max_total_exposure_pct=0.25,
            min_edge_to_trade=0.02,
            kelly_fraction=0.30,  # Higher fraction for arb (lower risk)
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
        Generate arbitrage trading signal.

        Checks for three types of mispricings in priority order.
        """
        market_id = market_row.get("market_id", "")
        time_to_expiry = market_row.get("time_to_expiry_min", 0)
        market_price_yes = market_row.get("market_price_yes", 0.5)
        market_price_no = market_row.get("market_price_no", 0.5)
        fair_price = market_row.get("fair_price", 0.5)

        # Timing constraints
        if time_to_expiry <= 1 or time_to_expiry > 14:
            return None

        # Skip if already positioned
        if self._already_in_market(market_id, open_positions):
            return None

        # Update cross-market cache for this timestamp
        self._update_market_cache(market_row, timestamp)

        # ── Type 1: YES/NO Imbalance ──
        signal = self._check_imbalance(market_price_yes, market_price_no, market_id, portfolio_value)
        if signal is not None:
            return signal

        # ── Type 2: Cross-Market Arbitrage ──
        signal = self._check_cross_market(
            market_row, market_id, timestamp, portfolio_value, open_positions
        )
        if signal is not None:
            return signal

        # ── Type 3: Fair Value Deviation ──
        signal = self._check_fair_value_deviation(
            market_price_yes, market_price_no, fair_price, market_id, portfolio_value
        )
        if signal is not None:
            return signal

        return None

    def _update_market_cache(self, market_row: pd.Series, timestamp: datetime):
        """
        Cache market prices for cross-market comparison.

        Stores the latest price observation for each market so that
        different strikes on the same underlying can be compared.
        """
        market_id = market_row.get("market_id", "")
        symbol = market_row.get("symbol", "")
        strike = market_row.get("strike", 0)
        close_price = market_row.get("close_price", 0)
        time_to_expiry = market_row.get("time_to_expiry_min", 0)

        self._market_price_cache[market_id] = {
            "symbol": symbol,
            "strike": strike,
            "close_price": close_price,
            "market_price_yes": market_row.get("market_price_yes", 0.5),
            "fair_price": market_row.get("fair_price", 0.5),
            "time_to_expiry_min": time_to_expiry,
            "timestamp": timestamp,
        }

    def _check_cross_market(
        self,
        market_row: pd.Series,
        market_id: str,
        timestamp: datetime,
        portfolio_value: float,
        open_positions: Dict[str, list],
    ) -> Optional[dict]:
        """
        Cross-market arbitrage: detect mispricing across different strikes.

        If two markets share the same underlying and similar expiry, but their
        implied probabilities are inconsistent (e.g., a lower strike has a
        LOWER yes-probability than a higher strike), that's an arbitrage.

        For ordered strikes K1 < K2:
            P(price > K1) >= P(price > K2) must hold (monotonicity)
            If violated: buy cheap YES on lower strike, buy cheap NO on higher strike.
        """
        current = self._market_price_cache.get(market_id)
        if current is None:
            return None

        current_symbol = current["symbol"]
        current_strike = current["strike"]
        current_price_yes = current["market_price_yes"]
        current_fair = current["fair_price"]
        current_ttx = current["time_to_expiry_min"]

        for other_id, other in self._market_price_cache.items():
            if other_id == market_id:
                continue
            # Same underlying, similar time-to-expiry
            if other["symbol"] != current_symbol:
                continue
            if abs(other["time_to_expiry_min"] - current_ttx) > 2:
                continue

            other_strike = other["strike"]
            other_price_yes = other["market_price_yes"]

            # Check monotonicity: lower strike should have higher P(YES)
            if current_strike < other_strike:
                low_strike_yes = current_price_yes
                high_strike_yes = other_price_yes
                low_market_id = market_id
            elif current_strike > other_strike:
                low_strike_yes = other_price_yes
                high_strike_yes = current_price_yes
                low_market_id = other_id
            else:
                continue  # Same strike

            # Violation: lower strike has lower YES price
            violation = high_strike_yes - low_strike_yes
            if violation > self.cross_market_threshold:
                # The lower strike's YES is too cheap -> buy YES on it
                if low_market_id == market_id:
                    direction = "YES"
                    predicted_prob = current_fair
                    market_price = current_price_yes
                else:
                    # The higher strike's NO is too cheap (buy NO on current)
                    direction = "NO"
                    predicted_prob = 1.0 - current_fair
                    market_price = market_row.get("market_price_no", 0.5)

                edge = violation / 2.0  # Conservative edge estimate

                size = self.risk_mgr.compute_kelly_size(
                    predicted_prob=predicted_prob,
                    market_price=market_price,
                    portfolio_value=portfolio_value,
                    direction=direction,
                )
                if size < 10.0:
                    continue

                self._trade_count += 1
                return {
                    "action": "trade",
                    "direction": direction,
                    "size": size,
                    "predicted_prob": predicted_prob,
                    "confidence": min(violation / 0.15, 1.0),
                    "edge": edge,
                    "reason": f"ARB CrossMkt: {market_id} vs {other_id}, violation={violation:.4f}",
                }

        return None

    def _check_imbalance(
        self,
        p_yes: float,
        p_no: float,
        market_id: str,
        portfolio_value: float,
    ) -> Optional[dict]:
        """
        Check for YES/NO price imbalance.

        If p_yes + p_no < 1 - threshold - fees:
            The market is underpriced -> buy the cheaper side
        """
        total = p_yes + p_no
        threshold = self.imbalance_threshold + self.fee_buffer

        if total < (1.0 - threshold):
            # Both sides are cheap -> buy the side with more room
            edge = (1.0 - total) / 2
            if p_yes <= p_no:
                direction = "YES"
                predicted_prob = 1.0 - p_no  # Implied from NO price
                market_price = p_yes
            else:
                direction = "NO"
                predicted_prob = 1.0 - p_yes
                market_price = p_no

            size = self.risk_mgr.compute_kelly_size(
                predicted_prob=predicted_prob,
                market_price=market_price,
                portfolio_value=portfolio_value,
                direction=direction,
            )
            if size < 10.0:
                return None

            self._trade_count += 1
            return {
                "action": "trade",
                "direction": direction,
                "size": size,
                "predicted_prob": predicted_prob,
                "confidence": min(edge / 0.10, 1.0),
                "edge": edge,
                "reason": f"ARB Imbalance: p_yes+p_no={total:.4f} < {1-threshold:.4f}",
            }

        return None

    def _check_fair_value_deviation(
        self,
        p_yes: float,
        p_no: float,
        fair_price: float,
        market_id: str,
        portfolio_value: float,
    ) -> Optional[dict]:
        """
        Check for deviation between market price and BS fair value.

        Trade toward fair value when deviation exceeds threshold.
        """
        yes_deviation = fair_price - p_yes  # positive = YES is cheap
        no_deviation = (1.0 - fair_price) - p_no  # positive = NO is cheap

        direction = None
        edge = 0.0

        if yes_deviation > self.fair_value_threshold:
            # YES is underpriced relative to fair value
            direction = "YES"
            edge = yes_deviation
            predicted_prob = fair_price
            market_price = p_yes
        elif no_deviation > self.fair_value_threshold:
            # NO is underpriced
            direction = "NO"
            edge = no_deviation
            predicted_prob = 1.0 - fair_price
            market_price = p_no
        else:
            return None

        size = self.risk_mgr.compute_kelly_size(
            predicted_prob=predicted_prob,
            market_price=market_price,
            portfolio_value=portfolio_value,
            direction=direction,
        )
        if size < 10.0:
            return None

        self._trade_count += 1
        return {
            "action": "trade",
            "direction": direction,
            "size": size,
            "predicted_prob": predicted_prob,
            "confidence": min(edge / 0.15, 1.0),
            "edge": edge,
            "reason": f"ARB FairVal: deviation={edge:.4f}, fair={fair_price:.4f}",
        }

    def reset(self):
        super().reset()
        self.risk_mgr.reset()
        self._market_price_cache = {}
