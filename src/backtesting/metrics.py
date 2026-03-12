"""
Performance Metrics Calculator.

Computes comprehensive performance metrics for backtesting results.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates trading performance metrics from equity curves and trade logs.
    """

    @staticmethod
    def compute_all(
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 525600,  # minutes in a year
    ) -> Dict[str, float]:
        """
        Compute all performance metrics.

        Parameters:
            equity_curve: DataFrame with 'equity' column (time-indexed)
            trades_df: DataFrame of trade records
            initial_capital: Starting capital
            risk_free_rate: For Sharpe ratio
            periods_per_year: For annualization

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # ── Return Metrics ──
        if len(equity_curve) > 0:
            final_equity = equity_curve["equity"].iloc[-1]
            metrics["initial_capital"] = initial_capital
            metrics["final_equity"] = final_equity
            metrics["total_return"] = (final_equity - initial_capital) / initial_capital
            metrics["total_return_pct"] = metrics["total_return"] * 100

            # Period returns
            returns = equity_curve["equity"].pct_change().dropna()
            if len(returns) > 1:
                # Sharpe Ratio (annualized)
                excess_returns = returns - risk_free_rate / periods_per_year
                if returns.std() > 0:
                    metrics["sharpe_ratio"] = (
                        excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
                    )
                else:
                    metrics["sharpe_ratio"] = 0.0

                # Sortino Ratio
                downside_excess = excess_returns[excess_returns < 0]
                if len(downside_excess) > 0 and downside_excess.std() > 0:
                    metrics["sortino_ratio"] = (
                        excess_returns.mean() / downside_excess.std() * np.sqrt(periods_per_year)
                    )
                else:
                    metrics["sortino_ratio"] = 0.0

                # Volatility (annualized)
                metrics["volatility"] = returns.std() * np.sqrt(periods_per_year)
            else:
                metrics["sharpe_ratio"] = 0.0
                metrics["sortino_ratio"] = 0.0
                metrics["volatility"] = 0.0

            # ── Drawdown Metrics ──
            peak = equity_curve["equity"].cummax()
            drawdown = (equity_curve["equity"] - peak) / peak
            metrics["max_drawdown"] = abs(float(drawdown.min()))
            metrics["max_drawdown_pct"] = metrics["max_drawdown"] * 100

            # Calmar Ratio
            if metrics["max_drawdown"] > 1e-6:
                # Estimate annualized return from total return and time span
                total_minutes = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 60
                if total_minutes > 0:
                    annualized_return = (
                        (1 + metrics["total_return"]) ** (periods_per_year / total_minutes) - 1
                    )
                    metrics["calmar_ratio"] = min(
                        annualized_return / metrics["max_drawdown"], 1000.0
                    )
                else:
                    metrics["calmar_ratio"] = 0.0
            else:
                metrics["calmar_ratio"] = min(metrics["total_return"] * 100, 1000.0) if metrics["total_return"] > 0 else 0.0

            # Recovery time
            in_drawdown = drawdown < 0
            if in_drawdown.any():
                drawdown_periods = in_drawdown.astype(int)
                # Find longest consecutive drawdown
                groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
                longest = drawdown_periods.groupby(groups).sum().max()
                metrics["max_drawdown_duration_bars"] = int(longest)
            else:
                metrics["max_drawdown_duration_bars"] = 0
        else:
            metrics.update({
                "initial_capital": initial_capital,
                "final_equity": initial_capital,
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown_duration_bars": 0,
            })

        # ── Trade Metrics ──
        if len(trades_df) > 0:
            metrics["total_trades"] = len(trades_df)
            pnl_col = "net_pnl" if "net_pnl" in trades_df.columns else "pnl"

            winners = trades_df[trades_df[pnl_col] > 0]
            losers = trades_df[trades_df[pnl_col] < 0]

            metrics["winning_trades"] = len(winners)
            metrics["losing_trades"] = len(losers)
            metrics["win_rate"] = len(winners) / len(trades_df) if len(trades_df) > 0 else 0.0
            metrics["win_rate_pct"] = metrics["win_rate"] * 100

            gross_profit = winners[pnl_col].sum() if len(winners) > 0 else 0.0
            gross_loss = abs(losers[pnl_col].sum()) if len(losers) > 0 else 0.0
            metrics["gross_profit"] = float(gross_profit)
            metrics["gross_loss"] = float(gross_loss)
            metrics["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float('inf')
            )

            metrics["avg_trade_pnl"] = float(trades_df[pnl_col].mean())
            metrics["avg_winner"] = float(winners[pnl_col].mean()) if len(winners) > 0 else 0.0
            metrics["avg_loser"] = float(losers[pnl_col].mean()) if len(losers) > 0 else 0.0
            metrics["best_trade"] = float(trades_df[pnl_col].max())
            metrics["worst_trade"] = float(trades_df[pnl_col].min())
            metrics["total_fees"] = float(trades_df["fees"].sum()) if "fees" in trades_df.columns else 0.0

            # Expectancy
            metrics["expectancy"] = (
                metrics["win_rate"] * metrics["avg_winner"]
                + (1 - metrics["win_rate"]) * metrics["avg_loser"]
            )

            # Avg edge
            if "edge" in trades_df.columns:
                metrics["avg_edge"] = float(trades_df["edge"].mean())
        else:
            metrics.update({
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "win_rate_pct": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "total_fees": 0.0,
                "expectancy": 0.0,
            })

        return metrics

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        """Pretty-print metrics as a formatted string."""
        lines = [
            "═" * 50,
            "  PERFORMANCE METRICS",
            "═" * 50,
            f"  Initial Capital:     ${metrics.get('initial_capital', 0):,.2f}",
            f"  Final Equity:        ${metrics.get('final_equity', 0):,.2f}",
            f"  Total Return:        {metrics.get('total_return_pct', 0):.2f}%",
            f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}",
            f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}",
            f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%",
            f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}",
            "─" * 50,
            f"  Total Trades:        {metrics.get('total_trades', 0)}",
            f"  Win Rate:            {metrics.get('win_rate_pct', 0):.1f}%",
            f"  Profit Factor:       {metrics.get('profit_factor', 0):.3f}",
            f"  Avg Trade P&L:       ${metrics.get('avg_trade_pnl', 0):.2f}",
            f"  Best Trade:          ${metrics.get('best_trade', 0):.2f}",
            f"  Worst Trade:         ${metrics.get('worst_trade', 0):.2f}",
            f"  Total Fees:          ${metrics.get('total_fees', 0):.2f}",
            f"  Expectancy:          ${metrics.get('expectancy', 0):.2f}",
            "═" * 50,
        ]
        return "\n".join(lines)
