"""Tests for the backtesting engine — trade log, position manager, metrics."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtesting.trade_log import TradeRecord, TradeLogger
from src.backtesting.position import Position, PositionManager
from src.backtesting.metrics import MetricsCalculator


# --------------------------------------------------------------------------- #
#  TradeLogger tests
# --------------------------------------------------------------------------- #
class TestTradeLogger:
    def _make_record(self, **overrides):
        defaults = dict(
            timestamp=pd.Timestamp("2025-01-01 00:00"),
            market_id="m1",
            symbol="BTC/USDT",
            predicted_probability=0.65,
            market_price=0.55,
            direction="YES",
            position_size=500,
            entry_price=0.55,
            exit_price=1.0,
            realized_outcome=1,
            pnl=225.0,
            fees=5.0,
            net_pnl=220.0,
            strategy_name="test",
            confidence=0.65,
            edge=0.10,
            capital_at_entry=10000,
            settlement_time=pd.Timestamp("2025-01-01 00:15"),
        )
        defaults.update(overrides)
        return TradeRecord(**defaults)

    def test_log_trade(self):
        logger = TradeLogger()
        record = self._make_record()
        logger.log_trade(record)
        assert len(logger) == 1

    def test_to_dataframe(self):
        logger = TradeLogger()
        logger.log_trade(self._make_record())
        df = logger.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_summary(self):
        logger = TradeLogger()
        logger.log_trade(self._make_record(net_pnl=100))
        logger.log_trade(self._make_record(net_pnl=-50, market_id="m2"))
        s = logger.summary()
        assert isinstance(s, dict)

    def test_clear(self):
        logger = TradeLogger()
        logger.log_trade(self._make_record())
        logger.clear()
        assert len(logger) == 0


# --------------------------------------------------------------------------- #
#  PositionManager tests
# --------------------------------------------------------------------------- #
class TestPositionManager:
    def _make_position(self, market_id="m1", direction="YES", entry_price=0.50, size=500):
        return Position(
            market_id=market_id,
            symbol="BTC/USDT",
            direction=direction,
            entry_price=entry_price,
            size=size,
            entry_time=pd.Timestamp("2025-01-01"),
            strategy_name="test",
            predicted_prob=0.6,
            edge=0.1,
            confidence=0.6,
        )

    def test_open_position(self):
        pm = PositionManager(initial_capital=10000)
        pos = self._make_position(size=500)
        pm.open_position(pos)
        assert pm.num_open_positions >= 1
        assert pm.cash < 10000

    def test_settle_market_yes_win(self):
        pm = PositionManager(initial_capital=10000)
        pos = self._make_position(direction="YES", entry_price=0.50, size=500)
        pm.open_position(pos)
        pm.settle_market(
            market_id="m1",
            resolution=1,
            settlement_time=pd.Timestamp("2025-01-01 00:15"),
        )
        assert pm.num_open_positions == 0
        # YES bet won → should be profitable
        assert pm.cash > 10000 - 500  # at least got investment back

    def test_settle_market_yes_loss(self):
        pm = PositionManager(initial_capital=10000)
        pos = self._make_position(direction="YES", entry_price=0.50, size=500)
        pm.open_position(pos)
        pm.settle_market(
            market_id="m1",
            resolution=0,
            settlement_time=pd.Timestamp("2025-01-01 00:15"),
        )
        assert pm.num_open_positions == 0
        # YES bet lost → cash should be reduced
        assert pm.cash < 10000

    def test_can_open_position_within_limits(self):
        pm = PositionManager(initial_capital=10000)
        assert pm.can_open_position(500, max_position_pct=0.05, max_total_exposure_pct=0.30)

    def test_can_open_position_exceeds_cash(self):
        pm = PositionManager(initial_capital=10000)
        assert not pm.can_open_position(11000, max_position_pct=1.0, max_total_exposure_pct=1.0)

    def test_equity_property(self):
        pm = PositionManager(initial_capital=10000)
        assert pm.equity == 10000

    def test_reset(self):
        pm = PositionManager(initial_capital=10000)
        pos = self._make_position()
        pm.open_position(pos)
        pm.reset()
        assert pm.cash == 10000
        assert pm.num_open_positions == 0


# --------------------------------------------------------------------------- #
#  MetricsCalculator tests
# --------------------------------------------------------------------------- #
class TestMetricsCalculator:
    def test_compute_all_with_trades(self):
        n = 20
        equity_curve = pd.DataFrame({
            "equity": np.cumsum(np.random.randn(n) * 50) + 10000,
        }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))

        trades_data = []
        np.random.seed(42)
        for i in range(n):
            pnl = np.random.choice([-50, -30, 20, 40, 60])
            trades_data.append({
                "net_pnl": pnl,
                "pnl": pnl + 1,
                "fees": 1,
                "direction": "YES",
                "entry_price": 0.5,
                "exit_price": 0.6 if pnl > 0 else 0.4,
                "position_size": 100,
                "market_id": f"m{i}",
            })
        trades_df = pd.DataFrame(trades_data)

        metrics = MetricsCalculator.compute_all(equity_curve, trades_df, initial_capital=10000)
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate" in metrics
        assert isinstance(metrics["total_trades"], (int, np.integer))

    def test_compute_all_no_trades(self):
        equity_curve = pd.DataFrame({"equity": [10000]})
        trades_df = pd.DataFrame()
        metrics = MetricsCalculator.compute_all(equity_curve, trades_df, initial_capital=10000)
        assert metrics["total_trades"] == 0

    def test_format_metrics(self):
        metrics = {"total_return_pct": 5.0, "sharpe_ratio": 1.5, "total_trades": 10}
        text = MetricsCalculator.format_metrics(metrics)
        assert isinstance(text, str)
        assert len(text) > 0
