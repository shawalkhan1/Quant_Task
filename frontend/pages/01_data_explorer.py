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
st.markdown("Explore live Polymarket markets, price proxy series, and engineered features.")
st.markdown("---")

# ── Controls ──
col1, col2, col3 = st.columns(3)
with col1:
    days = st.slider("Lookback Days", 3, 60, 15)
with col2:
    min_volume = st.number_input("Min Market Volume (USD)", 100.0, 1000000.0, 5000.0, 100.0)
with col3:
    max_markets = st.slider("Max Markets", 10, 200, 80)


@st.cache_data(ttl=3600)
def load_data(days, min_volume, max_markets):
    from src.data.live_market_loader import load_live_polymarket_data

    return load_live_polymarket_data(
        days_back=days,
        min_volume=min_volume,
        max_markets=max_markets,
        use_cache=True,
    )


if st.button("📥 Load Data", type="primary"):
    with st.spinner("Fetching and processing data..."):
        price_data, market_data, features = load_data(days, min_volume, max_markets)
        st.session_state["price_data"] = price_data
        st.session_state["market_data"] = market_data
        st.session_state["features"] = features
    st.success(
        f"Loaded {len(price_data)} price proxy bars, {len(market_data)} live market observations"
    )

if "price_data" in st.session_state:
    price_data = st.session_state["price_data"]
    market_data = st.session_state["market_data"]
    features = st.session_state["features"]

    # ── Price Chart ──
    st.markdown("### Price Data")
    from src.visualization.charts import candlestick_chart
    # Downsample for display
    display_price = price_data.iloc[::15]  # Every 15 min
    fig = candlestick_chart(display_price, title="Polymarket YES-Price Proxy (15-min)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Data Stats ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bars", f"{len(price_data):,}")
    with col2:
        st.metric("Date Range", f"{price_data.index[0].strftime('%Y-%m-%d')} → {price_data.index[-1].strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Avg YES Probability", f"{price_data['close'].mean():.3f}")
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
            duration_min = market_data.groupby("market_id")["total_duration_min"].first().median()
            st.metric("Median Duration", f"{duration_min:.1f} min")

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
