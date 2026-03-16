"""
Strategy Comparison Page — Compare multiple strategies side-by-side.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Strategy Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Strategy Comparison")
st.markdown("Run and compare all strategies on the same live Polymarket dataset.")
st.markdown("---")

# ── Controls ──
data_days = st.sidebar.slider("Lookback Days", 3, 60, 30, key="cmp_days")
min_volume = st.sidebar.number_input("Min Market Volume (USD)", 100.0, 1000000.0, 5000.0, 100.0, key="cmp_min_vol")
max_markets = st.sidebar.slider("Max Markets", 10, 200, 80, key="cmp_max_mkts")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 100000, 10000, 1000, key="cmp_cap")


@st.cache_data(ttl=3600)
def load_data(days, min_volume, max_markets):
    from src.data.live_market_loader import load_live_polymarket_data

    return load_live_polymarket_data(
        days_back=days,
        min_volume=min_volume,
        max_markets=max_markets,
        use_cache=True,
    )


def run_all_strategies(price_data, market_data, features, initial_capital):
    from src.backtesting.engine import BacktestEngine
    from src.strategies.market_maker import MarketMakerStrategy
    from src.strategies.arbitrage import ArbitrageStrategy
    from src.strategies.predictive import PredictiveStrategy

    # Use last 30% as test
    n = len(market_data)
    split_idx = int(n * 0.7)
    train_market = market_data.iloc[:split_idx]
    test_market = market_data.iloc[split_idx:]
    train_features = features.loc[:train_market.index[-1]]
    test_features = features.loc[test_market.index[0]:]

    results = {}

    # 1. Market Maker
    mm = MarketMakerStrategy()
    engine_mm = BacktestEngine(initial_capital=initial_capital)
    results["Market Maker"] = engine_mm.run(mm, test_market, test_features, price_data)

    # 2. Arbitrage
    arb = ArbitrageStrategy()
    engine_arb = BacktestEngine(initial_capital=initial_capital)
    results["Arbitrage"] = engine_arb.run(arb, test_market, test_features, price_data)

    # 3. Predictive
    pred = PredictiveStrategy(min_edge=0.05, model_type="ensemble")
    pred.train(train_features, train_market)
    engine_pred = BacktestEngine(initial_capital=initial_capital)
    results["Predictive"] = engine_pred.run(pred, test_market, test_features, price_data)

    return results


if st.button("🏁 Run All Strategies", type="primary"):
    with st.spinner("Running all three strategies..."):
        price_data, market_data, features = load_data(data_days, min_volume, max_markets)
        results = run_all_strategies(price_data, market_data, features, initial_capital)
        st.session_state["comparison_results"] = results
    st.success("All strategies complete!")

if "comparison_results" in st.session_state:
    results = st.session_state["comparison_results"]

    # ── Equity Curves Overlay ──
    st.markdown("### Equity Curves")
    from src.visualization.charts import strategy_comparison_chart
    fig = strategy_comparison_chart(results)
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics Comparison Table ──
    st.markdown("### Performance Metrics Comparison")
    comparison_data = []
    for name, result in results.items():
        m = result["metrics"]
        comparison_data.append({
            "Strategy": name,
            "Total Return %": f"{m.get('total_return_pct', 0):.2f}",
            "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.3f}",
            "Sortino Ratio": f"{m.get('sortino_ratio', 0):.3f}",
            "Max Drawdown %": f"{m.get('max_drawdown_pct', 0):.2f}",
            "Win Rate %": f"{m.get('win_rate_pct', 0):.1f}",
            "Profit Factor": f"{m.get('profit_factor', 0):.3f}",
            "Total Trades": m.get("total_trades", 0),
            "Avg Trade P&L": f"${m.get('avg_trade_pnl', 0):.2f}",
            "Expectancy": f"${m.get('expectancy', 0):.2f}",
            "Calmar Ratio": f"{m.get('calmar_ratio', 0):.3f}",
        })

    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    # ── Bar Charts ──
    st.markdown("### Visual Comparisons")
    import plotly.graph_objects as go

    col1, col2 = st.columns(2)
    with col1:
        strategies = [d["Strategy"] for d in comparison_data]
        returns = [float(d["Total Return %"]) for d in comparison_data]
        colors = ["#00d4aa" if r >= 0 else "#ff6b6b" for r in returns]
        fig = go.Figure([go.Bar(x=strategies, y=returns, marker_color=colors)])
        fig.update_layout(
            title="Total Return %", template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sharpes = [float(d["Sharpe Ratio"]) for d in comparison_data]
        colors = ["#00b4d8" if s >= 0 else "#ff6b6b" for s in sharpes]
        fig = go.Figure([go.Bar(x=strategies, y=sharpes, marker_color=colors)])
        fig.update_layout(
            title="Sharpe Ratio", template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        win_rates = [float(d["Win Rate %"]) for d in comparison_data]
        fig = go.Figure([go.Bar(x=strategies, y=win_rates, marker_color="#ffd166")])
        fig.update_layout(
            title="Win Rate %", template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        trades = [d["Total Trades"] for d in comparison_data]
        fig = go.Figure([go.Bar(x=strategies, y=trades, marker_color="#e77f67")])
        fig.update_layout(
            title="Total Trades", template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Individual Trade Logs ──
    st.markdown("### Trade Logs")
    for name, result in results.items():
        with st.expander(f"📋 {name} Trade Log"):
            trades_df = result.get("trades")
            if trades_df is not None and len(trades_df) > 0:
                st.dataframe(
                    trades_df[[
                        "timestamp", "market_id", "direction",
                        "predicted_probability", "market_price",
                        "net_pnl", "edge"
                    ]].round(4),
                    use_container_width=True,
                    height=300,
                )
            else:
                st.info(f"No trades for {name}")
else:
    st.info("Click **Run All Strategies** to compare performance.")
