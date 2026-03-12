"""
Data Explorer Page — View price data, prediction markets, and features.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Data Explorer")
st.markdown("Explore cryptocurrency price data, simulated prediction markets, and engineered features.")
st.markdown("---")

# ── Controls ──
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"], index=0)
with col2:
    days = st.slider("Days of Data", 5, 30, 15)
with col3:
    noise_std = st.slider("Market Noise σ", 0.01, 0.15, 0.05, 0.01)


@st.cache_data(ttl=3600)
def load_data(symbol, days, noise_std):
    from src.data.fetcher import DataFetcher
    from src.data.market_simulator import PredictionMarketSimulator
    from src.data.features import FeatureEngine

    fetcher = DataFetcher()
    price_data = fetcher.fetch_ohlcv(symbol=symbol, days=days)

    simulator = PredictionMarketSimulator(noise_std=noise_std)
    market_data = simulator.generate_markets(price_data, symbol=symbol)

    engine = FeatureEngine()
    features = engine.compute_all_features(price_data)

    return price_data, market_data, features


if st.button("📥 Load Data", type="primary"):
    with st.spinner("Fetching and processing data..."):
        price_data, market_data, features = load_data(symbol, days, noise_std)
        st.session_state["price_data"] = price_data
        st.session_state["market_data"] = market_data
        st.session_state["features"] = features
        st.session_state["symbol"] = symbol
    st.success(f"Loaded {len(price_data)} price bars, {len(market_data)} market observations")

if "price_data" in st.session_state:
    price_data = st.session_state["price_data"]
    market_data = st.session_state["market_data"]
    features = st.session_state["features"]

    # ── Price Chart ──
    st.markdown("### Price Data")
    from src.visualization.charts import candlestick_chart
    # Downsample for display
    display_price = price_data.iloc[::15]  # Every 15 min
    fig = candlestick_chart(display_price, title=f"{symbol} Price (15-min)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Data Stats ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bars", f"{len(price_data):,}")
    with col2:
        st.metric("Date Range", f"{price_data.index[0].strftime('%Y-%m-%d')} → {price_data.index[-1].strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Avg Price", f"${price_data['close'].mean():,.2f}")
    with col4:
        st.metric("Volatility (ann.)", f"{price_data['close'].pct_change().std() * np.sqrt(525600):.1%}")

    # ── Market Data ──
    st.markdown("### Prediction Market Data")
    if len(market_data) > 0:
        num_markets = market_data["market_id"].nunique()
        resolution_rate = market_data.groupby("market_id")["resolution"].first().mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Markets", num_markets)
        with col2:
            st.metric("YES Resolution Rate", f"{resolution_rate:.1%}")
        with col3:
            st.metric("Market Duration", "15 minutes")

        # Market price distribution
        from src.visualization.charts import probability_comparison_chart
        sample_markets = market_data["market_id"].unique()[:5]
        for mid in sample_markets:
            mkt = market_data[market_data["market_id"] == mid]
            fig = probability_comparison_chart(
                mkt.index, mkt["fair_price"].values, mkt["market_price_yes"].values,
                title=f"Market: {mid}"
            )
            st.plotly_chart(fig, use_container_width=True)
            break  # Show just one

        st.dataframe(market_data.head(50), use_container_width=True)

    # ── Features ──
    st.markdown("### Engineered Features")
    st.dataframe(features.describe().T.round(4), use_container_width=True)

    # Feature distribution
    feature_col = st.selectbox(
        "Select feature to visualize",
        [c for c in features.columns if c not in ["open", "high", "low", "close", "volume"]],
    )
    if feature_col:
        import plotly.express as px
        fig = px.histogram(
            features[feature_col].dropna(), nbins=50,
            title=f"Distribution: {feature_col}",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click **Load Data** to begin exploring.")
