"""
Core Backtesting Engine.

Event-driven backtesting engine for prediction market strategies.
Processes data bar-by-bar to prevent look-ahead bias.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from src.backtesting.position import Position, PositionManager
from src.backtesting.trade_log import TradeRecord, TradeLogger
from src.backtesting.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine for prediction market trading strategies.

    Architecture:
        1. Iterates through market data bar-by-bar (chronological)
        2. At each bar, feeds data to strategy for signal generation
        3. Executes trades based on signals with transaction costs
        4. Settles markets at expiry
        5. Tracks all positions, P&L, and metrics

    BIAS PREVENTION:
        - Strategy only sees data up to current bar
        - No future data is accessible during signal generation
        - Markets settle only when expiry is reached
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost_pct: float = 0.01,
        slippage_pct: float = 0.005,
        max_position_pct: float = 0.05,
        max_total_exposure_pct: float = 0.30,
        record_equity_interval: int = 15,  # Record equity every N bars
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.record_equity_interval = record_equity_interval

        # Core components
        self.position_manager = PositionManager(initial_capital)
        self.trade_logger = TradeLogger()
        self.metrics_calc = MetricsCalculator()

        # State tracking
        self._market_expiry: Dict[str, datetime] = {}
        self._market_resolution: Dict[str, int] = {}
        self._bar_count = 0

    def run(
        self,
        strategy,
        market_data: pd.DataFrame,
        features_data: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> dict:
        """
        Run a full backtest.

        Parameters:
            strategy: Strategy object with generate_signal() method
            market_data: Prediction market data (time-indexed)
            features_data: Feature data (time-indexed)
            price_data: OHLCV price data (time-indexed)

        Returns:
            Dictionary with backtest results (metrics, equity curve, trade log)
        """
        self._reset()
        logger.info(f"Starting backtest for strategy: {strategy.name}")
        logger.info(f"Data range: {market_data.index[0]} to {market_data.index[-1]}")

        # Pre-process: build market expiry and resolution lookup
        self._build_market_lookup(market_data)

        # Get unique timestamps in sorted order
        timestamps = market_data.index.unique().sort_values()

        for i, ts in enumerate(timestamps):
            self._bar_count += 1

            # 1. Get current market data at this timestamp
            current_markets = market_data.loc[[ts]] if ts in market_data.index else pd.DataFrame()

            # 2. Get features at this timestamp
            current_features = None
            if ts in features_data.index:
                feat_rows = features_data.loc[[ts]]
                current_features = feat_rows.iloc[0] if len(feat_rows) > 0 else None

            # 3. Check for market settlements
            self._settle_expired_markets(ts)

            # 4. Generate signals from strategy
            if len(current_markets) > 0:
                for _, market_row in current_markets.iterrows():
                    # Only trade if market hasn't expired yet
                    if market_row.get("time_to_expiry_min", 0) <= 1:
                        continue

                    signal = strategy.generate_signal(
                        market_row=market_row,
                        features=current_features,
                        timestamp=ts,
                        portfolio_value=self.position_manager.equity,
                        open_positions=self.position_manager.open_positions,
                    )

                    if signal is not None and signal.get("action") == "trade":
                        self._execute_trade(signal, market_row, ts, strategy.name)

            # 5. Record equity periodically
            if self._bar_count % self.record_equity_interval == 0:
                self.position_manager.record_equity(ts)

            # 6. Check drawdown circuit breaker
            if self._check_circuit_breaker():
                logger.warning("Drawdown circuit breaker triggered. Stopping.")
                break

        # Final equity record
        if len(timestamps) > 0:
            self.position_manager.record_equity(timestamps[-1])

        # Settle any remaining open positions at last known resolution
        self._settle_all_remaining(timestamps[-1] if len(timestamps) > 0 else datetime.now())

        # Compute metrics
        equity_curve = self.position_manager.get_equity_curve()
        trades_df = self.trade_logger.to_dataframe()
        metrics = self.metrics_calc.compute_all(
            equity_curve, trades_df, self.initial_capital
        )

        logger.info(f"Backtest complete: {metrics.get('total_trades', 0)} trades")
        logger.info(MetricsCalculator.format_metrics(metrics))

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": trades_df,
            "drawdown": self.position_manager.get_drawdown_series(),
            "strategy_name": strategy.name,
        }

    def _execute_trade(
        self,
        signal: dict,
        market_row: pd.Series,
        timestamp: datetime,
        strategy_name: str,
    ):
        """Execute a trade based on a strategy signal."""
        direction = signal["direction"]  # "YES" or "NO"
        size = signal.get("size", 100.0)  # Dollar amount
        predicted_prob = signal.get("predicted_prob", 0.5)
        confidence = signal.get("confidence", 0.0)
        edge = signal.get("edge", 0.0)

        # Determine entry price with slippage
        if direction == "YES":
            base_price = market_row["market_price_yes"]
            entry_price = min(base_price * (1 + self.slippage_pct), 0.99)
        else:
            base_price = market_row["market_price_no"]
            entry_price = min(base_price * (1 + self.slippage_pct), 0.99)

        # Transaction fees
        fees = size * self.transaction_cost_pct

        # Risk check
        cost = size * entry_price
        if not self.position_manager.can_open_position(
            cost + fees, self.max_position_pct, self.max_total_exposure_pct
        ):
            return

        # Create and open position
        market_id = market_row.get("market_id", "unknown")
        symbol = market_row.get("symbol", "unknown")

        position = Position(
            market_id=market_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            entry_time=timestamp,
            strategy_name=strategy_name,
            predicted_prob=predicted_prob,
            edge=edge,
            confidence=confidence,
        )

        success = self.position_manager.open_position(position, fees)
        if not success:
            return

        # Log the trade entry (P&L will be updated at settlement)
        trade = TradeRecord(
            timestamp=timestamp,
            market_id=market_id,
            symbol=symbol,
            predicted_probability=predicted_prob,
            market_price=base_price,
            direction=direction,
            position_size=size,
            entry_price=entry_price,
            exit_price=0.0,  # Updated at settlement
            realized_outcome=-1,  # Updated at settlement
            pnl=0.0,
            fees=fees,
            net_pnl=-fees,  # Updated at settlement
            strategy_name=strategy_name,
            confidence=confidence,
            edge=edge,
            capital_at_entry=self.position_manager.equity,
        )
        self.trade_logger.log_trade(trade)

    def _build_market_lookup(self, market_data: pd.DataFrame):
        """Build lookup tables for market expiry and resolution."""
        if "market_id" not in market_data.columns:
            return

        for market_id in market_data["market_id"].unique():
            mkt = market_data[market_data["market_id"] == market_id]
            # Expiry is when time_to_expiry_min == 0
            expire_rows = mkt[mkt["time_to_expiry_min"] == 0]
            if len(expire_rows) > 0:
                self._market_expiry[market_id] = expire_rows.index[-1]
                self._market_resolution[market_id] = int(expire_rows.iloc[-1]["resolution"])
            else:
                # Use last row as approximate expiry
                self._market_expiry[market_id] = mkt.index[-1]
                self._market_resolution[market_id] = int(mkt.iloc[-1]["resolution"])

    def _settle_expired_markets(self, current_time: datetime):
        """Settle any markets that have expired."""
        expired = []
        for market_id, expiry_time in self._market_expiry.items():
            if current_time >= expiry_time and market_id in self.position_manager.open_positions:
                resolution = self._market_resolution.get(market_id, 0)
                settlements = self.position_manager.settle_market(
                    market_id, resolution, current_time
                )
                # Update trade log with settlement info
                self._update_trade_settlements(settlements)
                expired.append(market_id)

    def _settle_all_remaining(self, final_time: datetime):
        """Force-settle all remaining open positions."""
        market_ids = list(self.position_manager.open_positions.keys())
        for market_id in market_ids:
            resolution = self._market_resolution.get(market_id, 0)
            settlements = self.position_manager.settle_market(
                market_id, resolution, final_time
            )
            self._update_trade_settlements(settlements)

    def _update_trade_settlements(self, settlements: List[dict]):
        """Update trade log entries with settlement information."""
        for settlement in settlements:
            # Find matching trade in log and update
            for trade in reversed(self.trade_logger.trades):
                if (trade.market_id == settlement["market_id"]
                    and trade.direction == settlement["direction"]
                    and trade.realized_outcome == -1):  # Not yet settled

                    trade.realized_outcome = settlement["resolution"]
                    trade.pnl = settlement["pnl"]
                    exit_fees = trade.position_size * self.transaction_cost_pct
                    trade.net_pnl = settlement["pnl"] - trade.fees - exit_fees
                    trade.exit_price = float(settlement["resolution"])
                    trade.settlement_time = settlement["settlement_time"]
                    break

    def _check_circuit_breaker(self) -> bool:
        """Check if drawdown exceeds circuit breaker threshold."""
        from config.settings import DEFAULT_RISK_CONFIG
        max_dd = DEFAULT_RISK_CONFIG.max_drawdown_pct

        current_equity = self.position_manager.equity
        if current_equity < self.initial_capital * (1 - max_dd):
            return True
        return False

    def _reset(self):
        """Reset engine for a new backtest."""
        self.position_manager.reset()
        self.trade_logger.clear()
        self._market_expiry = {}
        self._market_resolution = {}
        self._bar_count = 0
