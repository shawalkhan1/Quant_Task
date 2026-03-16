"""
Prediction Market Trading Platform — Streamlit Dashboard

Main application entry point.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Prediction Market Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric { background: #151820; padding: 15px; border-radius: 10px; border: 1px solid #1e2130; }
    .stMetric label { color: #808080 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #00b4d8 !important; }
    div[data-testid="stSidebarContent"] { background: #0a0d14; }
    h1 { color: #00b4d8 !important; }
    h2, h3 { color: #e0e0e0 !important; }
    .stTabs [data-baseweb="tab"] { color: #808080; }
    .stTabs [aria-selected="true"] { color: #00b4d8 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 PM Trading Platform")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Navigation**

Use the pages in the sidebar to:
- 🔍 Explore data
- 📈 Run backtests
- ⏩ Forward test strategies
- ⚖️ Compare strategies
- 📐 Research analysis
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #808080; font-size: 0.8em;'>
Prediction Market Trading System<br/>
Quantitative Research Platform<br/>
v1.0.0
</div>
""", unsafe_allow_html=True)

# Main page
st.title("Prediction Market Trading Platform")
st.markdown("### Quantitative Research & Trading System for 15-Minute Crypto Markets")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Market Type", "15-min Binary")
with col2:
    st.metric("Assets", "BTC / ETH")
with col3:
    st.metric("Strategies", "3 Active")
with col4:
    st.metric("Engine", "Event-Driven")

st.markdown("---")

st.markdown("""
## Platform Overview

This platform provides a complete research and trading infrastructure for
short-horizon crypto prediction markets.

### Key Features

| Component | Description |
|-----------|-------------|
| **Data Layer** | Fetches live Polymarket market metadata and token price histories via public APIs |
| **Backtesting Engine** | Event-driven, bar-by-bar simulation with position tracking, settlement, and transaction costs |
| **Strategy Framework** | Market Maker, Arbitrage, and Predictive (ML ensemble) strategies |
| **Forward Testing** | Walk-forward optimization with rolling out-of-sample evaluation |
| **Risk Management** | Kelly criterion sizing, exposure limits, drawdown circuit breakers |
| **Visualization** | Interactive Plotly charts, equity curves, calibration diagrams |

### How to Use

Navigate using the sidebar pages:

1. **Data Explorer** — View price data, prediction markets, and features
2. **Backtesting** — Run backtests with configurable parameters
3. **Forward Testing** — Walk-forward analysis and paper trading
4. **Strategy Comparison** — Compare multiple strategies side-by-side
5. **Research Analysis** — Probability calibration, feature importance, regime analysis
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #606060; font-size: 0.9em;'>
Built as a quantitative research platform for prediction market strategy evaluation.
</div>
""", unsafe_allow_html=True)
