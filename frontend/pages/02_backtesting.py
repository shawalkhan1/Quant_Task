"""
Backtesting Page — Run backtests with configurable parameters.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Backtesting", page_icon="📈", layout="wide")
st.title("📈 Backtesting Engine")
st.markdown("Run strategy backtests on historical prediction market data.")
st.markdown("---")

# ── Sidebar Controls ──
st.sidebar.header("Backtest Configuration")

strategy_name = st.sidebar.selectbox(
    "Strategy", ["Predictive (ML Ensemble)", "Market Maker", "Arbitrage"]
)
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 100000, 10000, 1000)
transaction_cost = st.sidebar.slider("Transaction Cost %", 0.0, 5.0, 1.0, 0.1) / 100
slippage = st.sidebar.slider("Slippage %", 0.0, 2.0, 0.5, 0.1) / 100
train_ratio = st.sidebar.slider("Train / Test Split", 0.5, 0.9, 0.7, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Strategy Parameters")

if strategy_name == "Market Maker":
    base_spread = st.sidebar.slider("Base Spread", 0.01, 0.10, 0.03, 0.01)
    vol_sensitivity = st.sidebar.slider("Volatility Sensitivity", 0.1, 2.0, 0.5, 0.1)
    inventory_aversion = st.sidebar.slider("Inventory Aversion", 0.01, 0.5, 0.1, 0.01)
elif strategy_name == "Arbitrage":
    imbalance_threshold = st.sidebar.slider("Imbalance Threshold", 0.01, 0.10, 0.02, 0.01)
    fair_value_threshold = st.sidebar.slider("Fair Value Threshold", 0.02, 0.15, 0.06, 0.01)
else:  # Predictive
    min_edge = st.sidebar.slider("Min Edge to Trade", 0.02, 0.15, 0.05, 0.01)
    ensemble_alpha = st.sidebar.slider("Ensemble α (LR weight)", 0.0, 1.0, 0.5, 0.1)
    model_type = st.sidebar.selectbox("Model Type", ["ensemble", "logistic", "gbt"])

# Data loading
st.sidebar.markdown("---")
symbol = st.sidebar.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"], key="bt_symbol")
data_days = st.sidebar.slider("Data Days", 5, 30, 15, key="bt_days")


@st.cache_data(ttl=3600)
def load_data_for_backtest(symbol, days):
    from src.data.fetcher import DataFetcher
    from src.data.market_simulator import PredictionMarketSimulator
    from src.data.features import FeatureEngine

    fetcher = DataFetcher()
    price_data = fetcher.fetch_ohlcv(symbol=symbol, days=days)

    simulator = PredictionMarketSimulator(noise_std=0.05)
    market_data = simulator.generate_markets(price_data, symbol=symbol)

    engine = FeatureEngine()
    features = engine.compute_all_features(price_data)

    return price_data, market_data, features


def run_backtest(strategy_name, price_data, market_data, features, train_ratio):
    from src.data.features import FeatureEngine
    from src.backtesting.engine import BacktestEngine

    # Time-based split
    n = len(market_data)
    split_idx = int(n * train_ratio)
    train_market = market_data.iloc[:split_idx]
    test_market = market_data.iloc[split_idx:]
    test_features = features.loc[test_market.index[0]:test_market.index[-1]] if len(test_market) > 0 else features
    train_features = features.loc[train_market.index[0]:train_market.index[-1]] if len(train_market) > 0 else features

    # Create strategy
    if strategy_name == "Market Maker":
        from src.strategies.market_maker import MarketMakerStrategy
        strategy = MarketMakerStrategy(
            base_spread=base_spread,
            vol_sensitivity=vol_sensitivity,
            inventory_aversion=inventory_aversion,
        )
        test_engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage,
        )
        results = test_engine.run(strategy, test_market, test_features, price_data)

    elif strategy_name == "Arbitrage":
        from src.strategies.arbitrage import ArbitrageStrategy
        strategy = ArbitrageStrategy(
            imbalance_threshold=imbalance_threshold,
            fair_value_threshold=fair_value_threshold,
        )
        test_engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage,
        )
        results = test_engine.run(strategy, test_market, test_features, price_data)

    else:  # Predictive
        from src.strategies.predictive import PredictiveStrategy
        strategy = PredictiveStrategy(
            min_edge=min_edge,
            ensemble_alpha=ensemble_alpha,
            model_type=model_type,
        )
        # Train on training data
        fe = FeatureEngine()
        train_feat_full = fe.compute_all_features(price_data.loc[:train_market.index[-1]])
        train_feat_with_market = fe.add_market_features(train_feat_full, train_market)
        strategy.train(train_feat_with_market, train_market)

        # Test on out-of-sample
        test_feat_full = fe.compute_all_features(price_data)
        test_feat_with_market = fe.add_market_features(test_feat_full, test_market)
        test_engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage,
        )
        results = test_engine.run(strategy, test_market, test_feat_with_market, price_data)
        results["feature_importance"] = strategy.get_feature_importance()

    results["train_size"] = len(train_market)
    results["test_size"] = len(test_market)
    return results


# ── Main Content ──
if st.button("🚀 Run Backtest", type="primary"):
    with st.spinner("Loading data and running backtest..."):
        price_data, market_data, features = load_data_for_backtest(symbol, data_days)
        results = run_backtest(strategy_name, price_data, market_data, features, train_ratio)
        st.session_state["backtest_results"] = results
        st.session_state["backtest_strategy"] = strategy_name

if "backtest_results" in st.session_state:
    results = st.session_state["backtest_results"]
    metrics = results["metrics"]

    st.markdown(f"### Results: {st.session_state.get('backtest_strategy', 'Unknown')}")

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta_color = "normal" if metrics.get("total_return", 0) >= 0 else "inverse"
        st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%",
                   delta=f"${metrics.get('final_equity', 0) - metrics.get('initial_capital', 0):,.2f}")
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
    with col3:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
    with col5:
        st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.3f}")
    with col2:
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
    with col3:
        st.metric("Expectancy", f"${metrics.get('expectancy', 0):.2f}")
    with col4:
        st.metric("Total Fees", f"${metrics.get('total_fees', 0):.2f}")

    st.markdown("---")

    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Trade Analysis", "Metrics Table", "Trade Log"])

    with tab1:
        from src.visualization.charts import equity_curve_chart
        ec = results.get("equity_curve")
        if ec is not None and len(ec) > 0:
            fig = equity_curve_chart(ec, initial_capital=initial_capital)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No equity curve data available.")

    with tab2:
        from src.visualization.charts import trade_scatter_chart
        trades_df = results.get("trades")
        if trades_df is not None and len(trades_df) > 0:
            fig = trade_scatter_chart(trades_df)
            st.plotly_chart(fig, use_container_width=True)

            # Direction breakdown
            import plotly.express as px
            dir_counts = trades_df["direction"].value_counts()
            fig2 = px.pie(values=dir_counts.values, names=dir_counts.index,
                          title="Trade Direction Distribution", template="plotly_dark")
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No trades executed.")

    with tab3:
        from src.visualization.charts import metrics_table_figure
        fig = metrics_table_figure(metrics)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        trades_df = results.get("trades")
        if trades_df is not None and len(trades_df) > 0:
            st.dataframe(
                trades_df[[
                    "timestamp", "market_id", "direction", "predicted_probability",
                    "market_price", "entry_price", "realized_outcome", "net_pnl",
                    "edge", "confidence", "strategy_name"
                ]].round(4),
                use_container_width=True,
                height=500,
            )
        else:
            st.info("No trades to display.")

    # Feature importance (for Predictive strategy)
    if "feature_importance" in results and results["feature_importance"] is not None:
        st.markdown("### Feature Importance (GBT)")
        fi = results["feature_importance"].head(15)
        import plotly.express as px
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
                     title="Top 15 Features", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to begin.")
