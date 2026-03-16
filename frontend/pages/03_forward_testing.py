"""
Forward Testing Page — Walk-forward analysis and paper trading.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Forward Testing", page_icon="⏩", layout="wide")
st.title("⏩ Forward Testing")
st.markdown("Walk-forward analysis and paper trading simulation.")
st.markdown("---")

# ── Controls ──
st.sidebar.header("Walk-Forward Config")
train_days = st.sidebar.slider("Train Period (days)", 3, 14, 7)
test_days = st.sidebar.slider("Test Period (days)", 1, 5, 2)
step_days = st.sidebar.slider("Step Size (days)", 1, 3, 1)
data_days = st.sidebar.slider("Lookback Days", 10, 90, 30, key="ft_days")
min_volume = st.sidebar.number_input("Min Market Volume (USD)", 100.0, 1000000.0, 5000.0, 100.0, key="ft_min_vol")
max_markets = st.sidebar.slider("Max Markets", 10, 200, 100, key="ft_max_mkts")

strategy_choice = st.sidebar.selectbox(
    "Strategy", ["Predictive (ML Ensemble)", "Market Maker", "Arbitrage"], key="ft_strat"
)


@st.cache_data(ttl=3600)
def load_data(days, min_volume, max_markets):
    from src.data.live_market_loader import load_live_polymarket_data

    return load_live_polymarket_data(
        days_back=days,
        min_volume=min_volume,
        max_markets=max_markets,
        use_cache=True,
    )


def get_strategy_class_and_kwargs(name):
    if name == "Market Maker":
        from src.strategies.market_maker import MarketMakerStrategy
        return MarketMakerStrategy, {"base_spread": 0.03}
    elif name == "Arbitrage":
        from src.strategies.arbitrage import ArbitrageStrategy
        return ArbitrageStrategy, {"imbalance_threshold": 0.02}
    else:
        from src.strategies.predictive import PredictiveStrategy
        return PredictiveStrategy, {"min_edge": 0.05, "model_type": "ensemble"}


def train_callback_factory(strategy_name):
    if strategy_name == "Predictive (ML Ensemble)":
        def callback(strategy, train_market, train_features):
            strategy.train(train_features, train_market)
        return callback
    return None


tab1, tab2 = st.tabs(["Walk-Forward Analysis", "Paper Trading"])

with tab1:
    st.markdown("### Walk-Forward Out-of-Sample Testing")
    st.markdown("""
    Walk-forward analysis trains on historical data and tests on subsequent
    unseen data, rolling forward through time. This prevents overfitting and
    gives a realistic estimate of strategy performance.
    """)

    if st.button("🔄 Run Walk-Forward Analysis", type="primary"):
        with st.spinner("Running walk-forward analysis (this may take a minute)..."):
            price_data, market_data, features = load_data(data_days, min_volume, max_markets)

            from src.forward_testing.rolling_simulator import RollingSimulator
            simulator = RollingSimulator(
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
            )

            strategy_class, strategy_kwargs = get_strategy_class_and_kwargs(strategy_choice)
            train_cb = train_callback_factory(strategy_choice)

            results = simulator.run(
                strategy_class=strategy_class,
                strategy_kwargs=strategy_kwargs,
                market_data=market_data,
                features_data=features,
                price_data=price_data,
                train_callback=train_cb,
            )
            st.session_state["wf_results"] = results

    if "wf_results" in st.session_state:
        results = st.session_state["wf_results"]

        if results.get("status") == "complete":
            agg = results["aggregated"]

            st.markdown("### Aggregated Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg OOS Return", f"{agg['avg_oos_return']*100:.2f}%")
            with col2:
                st.metric("Avg OOS Sharpe", f"{agg['avg_oos_sharpe']:.3f}")
            with col3:
                st.metric("Avg OOS Win Rate", f"{agg['avg_oos_win_rate']*100:.1f}%")
            with col4:
                st.metric("Num Folds", results["num_folds"])

            # Degradation analysis
            st.markdown("### In-Sample vs Out-of-Sample Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg IS Return", f"{agg['avg_is_return']*100:.2f}%")
                st.metric("Return Degradation", f"{agg['return_degradation']*100:.2f}%")
            with col2:
                st.metric("Avg IS Sharpe", f"{agg['avg_is_sharpe']:.3f}")
                st.metric("Sharpe Degradation", f"{agg['sharpe_degradation']:.3f}")

            # Walk-forward chart
            from src.visualization.charts import walk_forward_chart
            if results.get("fold_results"):
                fig = walk_forward_chart(results["fold_results"])
                st.plotly_chart(fig, use_container_width=True)

            # Per-fold details
            st.markdown("### Per-Fold Results")
            fold_data = []
            for fr in results.get("fold_results", []):
                fold_data.append({
                    "Fold": fr["fold"],
                    "Train": f"{fr['train_start'].strftime('%m/%d')} → {fr['train_end'].strftime('%m/%d')}",
                    "Test": f"{fr['test_start'].strftime('%m/%d')} → {fr['test_end'].strftime('%m/%d')}",
                    "IS Return %": f"{fr['in_sample'].get('total_return_pct', 0):.2f}",
                    "OOS Return %": f"{fr['out_sample'].get('total_return_pct', 0):.2f}",
                    "OOS Sharpe": f"{fr['out_sample'].get('sharpe_ratio', 0):.3f}",
                    "OOS Trades": fr['out_sample'].get('total_trades', 0),
                    "OOS Win Rate": f"{fr['out_sample'].get('win_rate_pct', 0):.1f}%",
                })
            st.dataframe(pd.DataFrame(fold_data), use_container_width=True)
        else:
            st.warning("Walk-forward analysis did not complete. Try adjusting parameters or data range.")

with tab2:
    st.markdown("### Paper Trading Simulation")
    st.markdown("""
    Paper trading simulates real-time execution on the most recent data segment
    (last 20% of the dataset), using a model trained on earlier data.
    """)

    if st.button("▶️ Start Paper Trading", type="primary"):
        with st.spinner("Running paper trading simulation..."):
            price_data, market_data, features = load_data(data_days, min_volume, max_markets)

            # Use last 20% as paper trading period
            split_idx = int(len(market_data) * 0.8)
            train_market = market_data.iloc[:split_idx]
            test_market = market_data.iloc[split_idx:]
            train_features = features.loc[:train_market.index[-1]]
            test_features = features.loc[test_market.index[0]:]

            strategy_class, strategy_kwargs = get_strategy_class_and_kwargs(strategy_choice)
            strategy = strategy_class(**strategy_kwargs)

            train_cb = train_callback_factory(strategy_choice)
            if train_cb:
                train_cb(strategy, train_market, train_features)

            from src.forward_testing.paper_trader import PaperTrader
            paper_trader = PaperTrader(initial_capital=10000)
            results = paper_trader.run_paper_trade(
                strategy, test_market, test_features, price_data
            )
            st.session_state["paper_results"] = results

    if "paper_results" in st.session_state:
        results = st.session_state["paper_results"]
        metrics = results["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Return", f"{metrics.get('total_return_pct', 0):.2f}%")
        with col2:
            st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")
        with col3:
            st.metric("Trades", metrics.get("total_trades", 0))
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")

        from src.visualization.charts import equity_curve_chart
        ec = results.get("equity_curve")
        if ec is not None and len(ec) > 0:
            fig = equity_curve_chart(ec, title="Paper Trading Equity Curve")
            st.plotly_chart(fig, use_container_width=True)
