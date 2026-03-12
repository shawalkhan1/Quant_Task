"""
Position tracking and settlement for prediction markets.

Handles open positions, settlement at market expiry, and P&L calculation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position in a prediction market."""
    market_id: str
    symbol: str
    direction: str           # "YES" or "NO"
    entry_price: float       # Price paid per contract
    size: float             # Number of contracts (dollar exposure)
    entry_time: datetime
    strategy_name: str
    predicted_prob: float
    edge: float
    confidence: float

    @property
    def cost(self) -> float:
        """Total cost of the position."""
        return self.size * self.entry_price

    @property
    def max_profit(self) -> float:
        """Maximum possible profit."""
        if self.direction == "YES":
            return self.size * (1.0 - self.entry_price)
        else:
            return self.size * self.entry_price

    @property
    def max_loss(self) -> float:
        """Maximum possible loss."""
        if self.direction == "YES":
            return self.size * self.entry_price
        else:
            return self.size * (1.0 - self.entry_price)


class PositionManager:
    """
    Manages open and closed positions for prediction market trading.

    Key responsibilities:
    - Track open positions by market
    - Settle positions when markets resolve
    - Calculate realized P&L
    - Enforce exposure limits
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions: Dict[str, List[Position]] = {}  # market_id -> [positions]
        self.closed_positions: List[dict] = []
        self.equity_history: List[dict] = []

    @property
    def total_exposure(self) -> float:
        """Total dollar exposure across all open positions."""
        total = 0.0
        for positions in self.open_positions.values():
            for pos in positions:
                total += pos.cost
        return total

    @property
    def equity(self) -> float:
        """Current equity = cash + position value (at cost)."""
        return self.cash + self.total_exposure

    @property
    def num_open_positions(self) -> int:
        return sum(len(ps) for ps in self.open_positions.values())

    def can_open_position(
        self,
        cost: float,
        max_position_pct: float = 0.05,
        max_total_exposure_pct: float = 0.30,
    ) -> bool:
        """
        Pre-trade risk check.
        """
        # Check position size limit
        if cost > self.equity * max_position_pct:
            return False
        # Check total exposure limit
        if (self.total_exposure + cost) > self.equity * max_total_exposure_pct:
            return False
        # Check sufficient cash
        if cost > self.cash:
            return False
        return True

    def open_position(self, position: Position, fees: float = 0.0) -> bool:
        """
        Open a new position.

        Returns True if successful, False if insufficient resources.
        """
        cost = position.cost + fees
        if cost > self.cash:
            logger.warning(
                f"Insufficient cash ({self.cash:.2f}) for position "
                f"cost ({cost:.2f}) in {position.market_id}"
            )
            return False

        self.cash -= cost

        if position.market_id not in self.open_positions:
            self.open_positions[position.market_id] = []
        self.open_positions[position.market_id].append(position)

        logger.debug(
            f"Opened {position.direction} position in {position.market_id}: "
            f"size={position.size:.2f}, price={position.entry_price:.4f}"
        )
        return True

    def settle_market(
        self,
        market_id: str,
        resolution: int,
        settlement_time: datetime,
    ) -> List[dict]:
        """
        Settle all positions in a resolved market.

        Parameters:
            market_id: The market being settled
            resolution: 1 (YES) or 0 (NO)
            settlement_time: When the market resolved

        Returns:
            List of settlement records
        """
        if market_id not in self.open_positions:
            return []

        settlements = []
        positions = self.open_positions.pop(market_id)

        for pos in positions:
            # Calculate P&L
            if pos.direction == "YES":
                pnl = pos.size * (resolution - pos.entry_price)
            else:  # NO
                pnl = pos.size * ((1 - resolution) - (1 - pos.entry_price))
                # Equivalently: pos.size * (pos.entry_price - resolution)

            # Return principal + profit (or loss)
            self.cash += pos.cost + pnl

            record = {
                "market_id": market_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "size": pos.size,
                "entry_time": pos.entry_time,
                "settlement_time": settlement_time,
                "resolution": resolution,
                "pnl": pnl,
                "strategy_name": pos.strategy_name,
                "predicted_prob": pos.predicted_prob,
                "edge": pos.edge,
            }
            self.closed_positions.append(record)
            settlements.append(record)

            logger.debug(
                f"Settled {pos.direction} in {market_id}: "
                f"resolution={resolution}, pnl={pnl:.2f}"
            )

        return settlements

    def record_equity(self, timestamp: datetime):
        """Record equity at a point in time for equity curve."""
        self.equity_history.append({
            "timestamp": timestamp,
            "equity": self.equity,
            "cash": self.cash,
            "exposure": self.total_exposure,
            "num_positions": self.num_open_positions,
        })

    def get_equity_curve(self) -> 'pd.DataFrame':
        """Return equity curve as DataFrame."""
        import pandas as pd
        if not self.equity_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_drawdown_series(self) -> 'pd.Series':
        """Calculate drawdown series from equity curve."""
        import pandas as pd
        eq = self.get_equity_curve()
        if len(eq) == 0:
            return pd.Series()
        peak = eq["equity"].cummax()
        drawdown = (eq["equity"] - peak) / peak
        return drawdown

    def reset(self):
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.open_positions = {}
        self.closed_positions = []
        self.equity_history = []
