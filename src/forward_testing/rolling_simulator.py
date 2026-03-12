"""
Rolling Out-of-Sample Simulator (Walk-Forward Analysis).

Implements walk-forward optimization: trains on N days, tests on M days,
then rolls forward. This prevents overfitting and provides realistic
out-of-sample performance estimates.
"""

import logging
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class RollingSimulator:
    """
    Walk-Forward Out-of-Sample Testing Engine.

    Methodology:
    1. Divide data into overlapping windows
    2. For each window: train on first portion, test on second
    3. Aggregate results across all test periods
    4. Compare in-sample vs out-of-sample performance

    This is the gold standard for strategy evaluation because:
    - No look-ahead bias
    - Tests adaptation to changing markets
    - Reveals overfitting if in-sample >> out-of-sample
    """

    def __init__(
        self,
        train_days: int = 7,
        test_days: int = 2,
        step_days: int = 1,
        initial_capital: float = 10000.0,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.initial_capital = initial_capital

    def run(
        self,
        strategy_class,
        strategy_kwargs: dict,
        market_data: pd.DataFrame,
        features_data: pd.DataFrame,
        price_data: pd.DataFrame,
        train_callback=None,
    ) -> dict:
        """
        Run walk-forward analysis.

        Parameters:
            strategy_class: Strategy class to instantiate
            strategy_kwargs: Kwargs for strategy constructor
            market_data: Full prediction market data
            features_data: Full feature data
            price_data: Full OHLCV data
            train_callback: Optional callback(strategy, train_market, train_features)
                            for retraining models on each fold

        Returns:
            Dictionary with aggregated results, per-fold results, and comparison
        """
        logger.info(
            f"Starting walk-forward analysis: "
            f"train={self.train_days}d, test={self.test_days}d, step={self.step_days}d"
        )

        # Generate time windows
        windows = self._generate_windows(price_data.index)
        logger.info(f"Generated {len(windows)} walk-forward windows")

        fold_results = []
        in_sample_metrics_list = []
        out_sample_metrics_list = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Fold {i+1}/{len(windows)}: train [{train_start} -> {train_end}], test [{test_start} -> {test_end}]")

            # Split data for this fold
            train_market = market_data.loc[train_start:train_end]
            train_features = features_data.loc[train_start:train_end]
            train_price = price_data.loc[train_start:train_end]

            test_market = market_data.loc[test_start:test_end]
            test_features = features_data.loc[test_start:test_end]
            test_price = price_data.loc[test_start:test_end]

            if len(train_market) < 100 or len(test_market) < 50:
                logger.warning(f"Fold {i+1}: insufficient data, skipping")
                continue

            # Create fresh strategy for this fold
            strategy = strategy_class(**strategy_kwargs)

            # Train if callback provided (for predictive strategy)
            if train_callback is not None:
                train_callback(strategy, train_market, train_features)

            # ── In-sample backtest ──
            in_engine = BacktestEngine(initial_capital=self.initial_capital)
            in_results = in_engine.run(strategy, train_market, train_features, train_price)
            in_sample_metrics_list.append(in_results["metrics"])

            # Reset strategy state for out-of-sample
            strategy.reset()
            if train_callback is not None:
                train_callback(strategy, train_market, train_features)

            # ── Out-of-sample backtest ──
            out_engine = BacktestEngine(initial_capital=self.initial_capital)
            out_results = out_engine.run(strategy, test_market, test_features, test_price)
            out_sample_metrics_list.append(out_results["metrics"])

            fold_results.append({
                "fold": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "in_sample": in_results["metrics"],
                "out_sample": out_results["metrics"],
                "out_equity_curve": out_results["equity_curve"],
                "out_trades": out_results["trades"],
            })

        if not fold_results:
            logger.warning("No valid walk-forward folds completed")
            return {"status": "no_valid_folds"}

        # Aggregate results
        aggregated = self._aggregate_results(fold_results, in_sample_metrics_list, out_sample_metrics_list)

        logger.info(
            f"Walk-forward complete: {len(fold_results)} folds, "
            f"Avg OOS return: {aggregated['avg_oos_return']:.4f}"
        )

        return {
            "status": "complete",
            "num_folds": len(fold_results),
            "fold_results": fold_results,
            "aggregated": aggregated,
            "in_sample_metrics": in_sample_metrics_list,
            "out_sample_metrics": out_sample_metrics_list,
        }

    def _generate_windows(self, index: pd.DatetimeIndex) -> List[tuple]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        windows = []
        start = index[0]
        end = index[-1]

        current = start
        while True:
            train_end = current + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > end:
                break

            windows.append((current, train_end, test_start, test_end))
            current += timedelta(days=self.step_days)

        return windows

    def _aggregate_results(
        self,
        fold_results: List[dict],
        in_sample_list: List[dict],
        out_sample_list: List[dict],
    ) -> dict:
        """Aggregate metrics across all folds."""

        def _safe_mean(metrics_list, key):
            vals = [m.get(key, 0) for m in metrics_list if isinstance(m.get(key), (int, float))]
            return np.mean(vals) if vals else 0.0

        def _safe_std(metrics_list, key):
            vals = [m.get(key, 0) for m in metrics_list if isinstance(m.get(key), (int, float))]
            return np.std(vals) if len(vals) > 1 else 0.0

        return {
            # In-sample averages
            "avg_is_return": _safe_mean(in_sample_list, "total_return"),
            "avg_is_sharpe": _safe_mean(in_sample_list, "sharpe_ratio"),
            "avg_is_win_rate": _safe_mean(in_sample_list, "win_rate"),
            "avg_is_trades": _safe_mean(in_sample_list, "total_trades"),

            # Out-of-sample averages
            "avg_oos_return": _safe_mean(out_sample_list, "total_return"),
            "avg_oos_sharpe": _safe_mean(out_sample_list, "sharpe_ratio"),
            "avg_oos_win_rate": _safe_mean(out_sample_list, "win_rate"),
            "avg_oos_max_dd": _safe_mean(out_sample_list, "max_drawdown"),
            "avg_oos_trades": _safe_mean(out_sample_list, "total_trades"),
            "avg_oos_profit_factor": _safe_mean(out_sample_list, "profit_factor"),

            # Stability metrics
            "oos_return_std": _safe_std(out_sample_list, "total_return"),
            "oos_sharpe_std": _safe_std(out_sample_list, "sharpe_ratio"),

            # Degradation (in-sample vs out-of-sample)
            "return_degradation": (
                _safe_mean(in_sample_list, "total_return") - _safe_mean(out_sample_list, "total_return")
            ),
            "sharpe_degradation": (
                _safe_mean(in_sample_list, "sharpe_ratio") - _safe_mean(out_sample_list, "sharpe_ratio")
            ),
        }
