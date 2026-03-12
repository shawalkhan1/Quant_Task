"""
Paper Trading Engine.

Simulates real-time execution of strategies by replaying
historical data as if it were arriving in real-time.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.backtesting.position import PositionManager
from src.backtesting.trade_log import TradeRecord, TradeLogger
from src.backtesting.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading engine that simulates real-time strategy execution.

    Uses historical data but processes it in a strictly forward manner,
    simulating how the strategy would perform in live conditions.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost_pct: float = 0.01,
        slippage_pct: float = 0.005,
    ):
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct,
            slippage_pct=slippage_pct,
        )
        self.is_running = False
        self.current_step = 0
        self.total_steps = 0

    def run_paper_trade(
        self,
        strategy,
        market_data: pd.DataFrame,
        features_data: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> dict:
        """
        Run paper trading simulation.

        This is essentially a forward-only backtest on unseen data,
        simulating what would happen if the strategy were deployed live.
        """
        logger.info(f"Starting paper trading for {strategy.name}")
        self.is_running = True
        self.total_steps = len(market_data)

        results = self.engine.run(strategy, market_data, features_data, price_data)
        results["mode"] = "paper_trade"

        self.is_running = False
        logger.info(f"Paper trading complete: {results['metrics'].get('total_trades', 0)} trades")
        return results

    def get_status(self) -> dict:
        """Get current paper trading status."""
        return {
            "is_running": self.is_running,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "equity": self.engine.position_manager.equity,
            "open_positions": self.engine.position_manager.num_open_positions,
            "total_trades": len(self.engine.trade_logger),
        }
