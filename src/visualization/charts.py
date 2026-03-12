"""
Reusable Chart Components.

Provides publication-quality Plotly charts for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict

COLORS = {
    "profit": "#00d4aa",
    "loss": "#ff6b6b",
    "neutral": "#808080",
    "primary": "#00b4d8",
    "secondary": "#e77f67",
    "accent": "#ffd166",
    "bg": "#0e1117",
    "grid": "#1e2130",
    "text": "#fafafa",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)


def equity_curve_chart(
    equity_df: pd.DataFrame,
    title: str = "Equity Curve",
    initial_capital: float = 10000.0,
    height: int = 450,
) -> go.Figure:
    """Create an equity curve chart with drawdown overlay."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=[title, "Drawdown"],
    )

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 180, 216, 0.1)",
        ),
        row=1, col=1,
    )

    # Initial capital reference
    fig.add_hline(
        y=initial_capital, line_dash="dash",
        line_color=COLORS["neutral"], row=1, col=1,
        annotation_text=f"Initial: ${initial_capital:,.0f}",
    )

    # Drawdown
    peak = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - peak) / peak * 100

    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=drawdown,
            mode="lines",
            name="Drawdown %",
            line=dict(color=COLORS["loss"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.2)",
        ),
        row=2, col=1,
    )

    fig.update_layout(**LAYOUT_DEFAULTS, height=height, showlegend=True)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def trade_scatter_chart(
    trades_df: pd.DataFrame,
    title: str = "Trade P&L Distribution",
    height: int = 400,
) -> go.Figure:
    """Scatter plot of trades colored by P&L."""
    if len(trades_df) == 0:
        return go.Figure().update_layout(**LAYOUT_DEFAULTS, title=title)

    colors = [COLORS["profit"] if pnl > 0 else COLORS["loss"] for pnl in trades_df["net_pnl"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trades_df["timestamp"] if "timestamp" in trades_df.columns else trades_df.index,
            y=trades_df["net_pnl"],
            mode="markers",
            marker=dict(color=colors, size=6, opacity=0.7),
            text=[
                f"Edge: {e:.4f}<br>Dir: {d}<br>P&L: ${p:.2f}"
                for e, d, p in zip(trades_df["edge"], trades_df["direction"], trades_df["net_pnl"])
            ],
            hoverinfo="text",
            name="Trades",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(**LAYOUT_DEFAULTS, height=height, title=title)
    fig.update_yaxes(title_text="Net P&L ($)")
    return fig


def metrics_table_figure(metrics: dict, height: int = 400) -> go.Figure:
    """Create a formatted metrics table."""
    keys = [
        "total_return_pct", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
        "calmar_ratio", "total_trades", "win_rate_pct", "profit_factor",
        "avg_trade_pnl", "best_trade", "worst_trade", "expectancy", "total_fees",
    ]
    labels = [
        "Total Return %", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown %",
        "Calmar Ratio", "Total Trades", "Win Rate %", "Profit Factor",
        "Avg Trade P&L", "Best Trade", "Worst Trade", "Expectancy", "Total Fees",
    ]
    values = []
    for k in keys:
        v = metrics.get(k, 0)
        if isinstance(v, float):
            values.append(f"{v:.3f}" if abs(v) < 100 else f"{v:,.2f}")
        else:
            values.append(str(v))

    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="#1e2130",
                font=dict(color="white", size=13),
                align="left",
            ),
            cells=dict(
                values=[labels, values],
                fill_color=[["#0e1117"] * len(labels), ["#151820"] * len(values)],
                font=dict(color="white", size=12),
                align="left",
                height=28,
            ),
        )
    ])
    fig.update_layout(**LAYOUT_DEFAULTS, height=height, title="Performance Metrics")
    return fig


def candlestick_chart(
    price_data: pd.DataFrame,
    title: str = "Price Chart",
    height: int = 450,
    overlay_series: Optional[Dict[str, pd.Series]] = None,
) -> go.Figure:
    """Create a candlestick chart with optional overlays."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data["open"],
            high=price_data["high"],
            low=price_data["low"],
            close=price_data["close"],
            increasing_line_color=COLORS["profit"],
            decreasing_line_color=COLORS["loss"],
            name="Price",
        ),
        row=1, col=1,
    )

    if overlay_series:
        for name, series in overlay_series.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series.values,
                    mode="lines", name=name,
                    line=dict(width=1.5),
                ),
                row=1, col=1,
            )

    # Volume
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data["volume"],
            name="Volume",
            marker_color=COLORS["primary"],
            opacity=0.4,
        ),
        row=2, col=1,
    )

    fig.update_layout(**LAYOUT_DEFAULTS, height=height, title=title, xaxis_rangeslider_visible=False)
    return fig


def probability_comparison_chart(
    timestamps,
    predicted_probs: np.ndarray,
    market_prices: np.ndarray,
    outcomes: Optional[np.ndarray] = None,
    title: str = "Predicted vs Market Probability",
    height: int = 400,
) -> go.Figure:
    """Compare predicted probabilities with market prices."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps, y=predicted_probs,
            mode="lines", name="Model Prediction",
            line=dict(color=COLORS["primary"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=market_prices,
            mode="lines", name="Market Price",
            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
        )
    )

    if outcomes is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=outcomes,
                mode="markers", name="Outcome",
                marker=dict(color=COLORS["accent"], size=4, symbol="diamond"),
            )
        )

    fig.update_layout(**LAYOUT_DEFAULTS, height=height, title=title)
    fig.update_yaxes(title_text="Probability", range=[-0.05, 1.05])
    return fig


def calibration_chart(
    reliability_df: pd.DataFrame,
    title: str = "Probability Calibration (Reliability Diagram)",
    height: int = 400,
) -> go.Figure:
    """Create a reliability diagram for probability calibration."""
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Perfect Calibration",
            line=dict(color=COLORS["neutral"], width=1, dash="dash"),
        )
    )

    valid = reliability_df.dropna()
    if len(valid) > 0:
        fig.add_trace(
            go.Scatter(
                x=valid["mean_predicted"],
                y=valid["fraction_positive"],
                mode="lines+markers",
                name="Model Calibration",
                line=dict(color=COLORS["primary"], width=2),
                marker=dict(size=8),
            )
        )

        # Bar chart for counts
        fig.add_trace(
            go.Bar(
                x=valid["bin_center"],
                y=valid["count"] / valid["count"].sum(),
                name="Sample Distribution",
                marker_color=COLORS["primary"],
                opacity=0.2,
                yaxis="y2",
            )
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=height,
        title=title,
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        yaxis2=dict(
            title="Distribution", overlaying="y", side="right",
            showgrid=False, range=[0, 1],
        ),
    )
    return fig


def strategy_comparison_chart(
    results_dict: Dict[str, dict],
    title: str = "Strategy Comparison — Equity Curves",
    height: int = 450,
) -> go.Figure:
    """Overlay equity curves from multiple strategies."""
    fig = go.Figure()
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#9b59b6", "#1abc9c"]

    for i, (name, result) in enumerate(results_dict.items()):
        ec = result.get("equity_curve")
        if ec is not None and len(ec) > 0:
            color = colors_list[i % len(colors_list)]
            fig.add_trace(
                go.Scatter(
                    x=ec.index, y=ec["equity"],
                    mode="lines", name=name,
                    line=dict(color=color, width=2),
                )
            )

    fig.update_layout(**LAYOUT_DEFAULTS, height=height, title=title)
    fig.update_yaxes(title_text="Equity ($)")
    return fig


def walk_forward_chart(
    fold_results: List[dict],
    title: str = "Walk-Forward Out-of-Sample Performance",
    height: int = 400,
) -> go.Figure:
    """Visualize walk-forward analysis results per fold."""
    fig = go.Figure()

    folds = list(range(1, len(fold_results) + 1))
    is_returns = [r["in_sample"].get("total_return_pct", 0) for r in fold_results]
    oos_returns = [r["out_sample"].get("total_return_pct", 0) for r in fold_results]

    fig.add_trace(
        go.Bar(x=folds, y=is_returns, name="In-Sample Return %",
               marker_color=COLORS["primary"], opacity=0.7)
    )
    fig.add_trace(
        go.Bar(x=folds, y=oos_returns, name="Out-of-Sample Return %",
               marker_color=COLORS["secondary"], opacity=0.7)
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=height, title=title,
        barmode="group",
        xaxis_title="Fold",
        yaxis_title="Return %",
    )
    return fig
