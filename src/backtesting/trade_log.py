"""
Trade logging system.

Records every trade with full detail for transparency and analysis.
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Complete record of a single trade in a prediction market."""
    timestamp: datetime
    market_id: str
    symbol: str
    predicted_probability: float
    market_price: float
    direction: str              # "YES" or "NO"
    position_size: float        # Dollar amount
    entry_price: float          # Price paid
    exit_price: float           # Settlement price (0 or 1)
    realized_outcome: int       # 0 or 1
    pnl: float                  # Gross P&L
    fees: float                 # Transaction fees
    net_pnl: float              # Net P&L after fees
    strategy_name: str
    confidence: float           # Model confidence  
    edge: float                 # Predicted edge over market price
    capital_at_entry: float     # Portfolio value at time of trade
    settlement_time: Optional[datetime] = None


class TradeLogger:
    """Maintains a complete trade log for backtesting and forward testing."""

    def __init__(self):
        self.trades: List[TradeRecord] = []

    def log_trade(self, trade: TradeRecord):
        """Record a trade."""
        self.trades.append(trade)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade log to a pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = [asdict(t) for t in self.trades]
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def save(self, filepath: str):
        """Save trade log to CSV."""
        df = self.to_dataframe()
        if len(df) > 0:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} trades to {filepath}")

    def summary(self) -> dict:
        """Quick summary of the trade log."""
        if not self.trades:
            return {"total_trades": 0}
        df = self.to_dataframe()
        return {
            "total_trades": len(df),
            "winning_trades": int((df["net_pnl"] > 0).sum()),
            "losing_trades": int((df["net_pnl"] < 0).sum()),
            "total_pnl": float(df["net_pnl"].sum()),
            "avg_pnl": float(df["net_pnl"].mean()),
            "best_trade": float(df["net_pnl"].max()),
            "worst_trade": float(df["net_pnl"].min()),
            "avg_edge": float(df["edge"].mean()),
        }

    def clear(self):
        """Clear all trades."""
        self.trades = []

    def __len__(self):
        return len(self.trades)
