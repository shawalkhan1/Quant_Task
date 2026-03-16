"""
Research Analysis Page — Calibration, feature importance, regime analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Research Analysis", page_icon="📐", layout="wide")
st.title("📐 Research Analysis")
st.markdown("Probability calibration, feature importance, and regime analysis.")
st.markdown("---")

data_days = st.sidebar.slider("Lookback Days", 3, 60, 30, key="res_days")
min_volume = st.sidebar.number_input("Min Market Volume (USD)", 100.0, 1000000.0, 5000.0, 100.0, key="res_min_vol")
max_markets = st.sidebar.slider("Max Markets", 10, 200, 80, key="res_max_mkts")


@st.cache_data(ttl=3600)
def load_and_train(days, min_volume, max_markets):
    from src.data.live_market_loader import load_live_polymarket_data
    from src.strategies.predictive import PredictiveStrategy

    price_data, market_data, features = load_live_polymarket_data(
        days_back=days,
        min_volume=min_volume,
        max_markets=max_markets,
        use_cache=True,
    )

    # Train/Test split
    n = len(market_data)
    split_idx = int(n * 0.7)
    train_market = market_data.iloc[:split_idx]
    test_market = market_data.iloc[split_idx:]
    train_features = features.loc[:train_market.index[-1]]
    test_features = features.loc[test_market.index[0]:]

    # Train predictive model
    strategy = PredictiveStrategy(min_edge=0.05, model_type="ensemble")
    train_metrics = strategy.train(train_features, train_market)

    # Generate predictions on test data
    predictions = []
    actuals = []
    market_prices_list = []
    timestamps = []

    test_signal_data = test_market[
        (test_market["minutes_elapsed"] >= 3) & (test_market["minutes_elapsed"] <= 5)
    ]

    for ts, row in test_signal_data.iterrows():
        if ts in test_features.index:
            feat = test_features.loc[ts]
            if isinstance(feat, pd.DataFrame):
                feat = feat.iloc[0]
            pred = strategy.predict_probability(feat)
            predictions.append(pred)
            actuals.append(int(row["resolution"]))
            market_prices_list.append(row["market_price_yes"])
            timestamps.append(ts)

    result = {
        "predictions": np.array(predictions) if predictions else np.array([]),
        "actuals": np.array(actuals) if actuals else np.array([]),
        "market_prices": np.array(market_prices_list) if market_prices_list else np.array([]),
        "timestamps": timestamps,
        "train_metrics": train_metrics,
        "feature_importance": strategy.get_feature_importance(),
        "price_data": price_data,
        "market_data": market_data,
        "features": features,
    }
    return result


if st.button("🔬 Run Analysis", type="primary"):
    with st.spinner("Training models and analyzing..."):
        result = load_and_train(data_days, min_volume, max_markets)
        st.session_state["research_data"] = result

if "research_data" in st.session_state:
    data = st.session_state["research_data"]
    predictions = data["predictions"]
    actuals = data["actuals"]
    market_prices = data["market_prices"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Probability Calibration", "Feature Analysis",
        "Regime Analysis", "Model Comparison"
    ])

    with tab1:
        st.markdown("### Probability Calibration Analysis")
        if len(predictions) > 20:
            from src.models.calibration import CalibrationAnalyzer

            # Brier Score
            brier = CalibrationAnalyzer.brier_score(actuals, predictions)
            market_brier = CalibrationAnalyzer.brier_score(actuals, market_prices)
            ece = CalibrationAnalyzer.expected_calibration_error(actuals, predictions)
            decomp = CalibrationAnalyzer.brier_decomposition(actuals, predictions)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Brier Score", f"{brier:.4f}")
            with col2:
                st.metric("Market Brier Score", f"{market_brier:.4f}")
            with col3:
                st.metric("ECE", f"{ece:.4f}")
            with col4:
                st.metric("Skill Score", f"{decomp['skill_score']:.4f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reliability", f"{decomp['reliability']:.4f}")
            with col2:
                st.metric("Resolution", f"{decomp['resolution']:.4f}")
            with col3:
                st.metric("Uncertainty", f"{decomp['uncertainty']:.4f}")

            # Reliability Diagram
            rel_df = CalibrationAnalyzer.reliability_diagram(actuals, predictions, n_bins=10)
            from src.visualization.charts import calibration_chart
            fig = calibration_chart(rel_df)
            st.plotly_chart(fig, use_container_width=True)

            # Prediction distribution
            import plotly.express as px
            fig = px.histogram(
                x=predictions, nbins=30,
                title="Prediction Distribution",
                labels={"x": "Predicted Probability"},
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

            # Prediction vs Market
            if len(data["timestamps"]) > 0:
                from src.visualization.charts import probability_comparison_chart
                fig = probability_comparison_chart(
                    data["timestamps"], predictions, market_prices, actuals,
                    title="Model Predictions vs Market Prices"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient prediction data for calibration analysis.")

    with tab2:
        st.markdown("### Feature Importance Analysis")
        fi = data.get("feature_importance")
        if fi is not None and len(fi) > 0:
            import plotly.express as px
            top_fi = fi.head(20)
            fig = px.bar(
                top_fi, x="importance", y="feature", orientation="h",
                title="Top 20 Feature Importances (GBT)",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                yaxis=dict(autorange="reversed"),
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(fi.round(4), use_container_width=True)
        else:
            st.info("No feature importance data available.")

        # Training metrics
        tm = data.get("train_metrics", {})
        if tm:
            st.markdown("### Training Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", tm.get("n_samples", 0))
            with col2:
                st.metric("LR Brier", f"{tm.get('lr_brier', 0):.4f}")
            with col3:
                st.metric("GBT Brier", f"{tm.get('gbt_brier', 0):.4f}")

    with tab3:
        st.markdown("### Regime Analysis")
        st.markdown("""
        Analyze strategy behavior under different market conditions.
        """)

        features = data["features"]
        market_data = data["market_data"]

        if "volatility_15m" in features.columns and len(market_data) > 0:
            # Classify market regimes
            vol = features["volatility_15m"].dropna()
            vol_median = vol.median()

            high_vol_mask = vol > vol_median * 1.5
            low_vol_mask = vol < vol_median * 0.5

            st.markdown(f"**Median 15m Volatility:** {vol_median:.4f}")
            st.markdown(f"**High-Vol Bars:** {high_vol_mask.sum():,} | **Low-Vol Bars:** {low_vol_mask.sum():,}")

            # Volatility distribution
            import plotly.express as px
            fig = px.histogram(
                vol, nbins=50, title="Volatility Distribution (15-minute)",
                template="plotly_dark",
            )
            fig.add_vline(x=vol_median, line_dash="dash", line_color="red",
                          annotation_text=f"Median: {vol_median:.4f}")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

            # Resolution rates by regime
            if "resolution" in market_data.columns and "implied_vol" in market_data.columns:
                market_data_copy = market_data.copy()
                vol_med_mkt = market_data_copy["implied_vol"].median()
                market_data_copy["regime"] = np.where(
                    market_data_copy["implied_vol"] > vol_med_mkt * 1.3, "High Vol",
                    np.where(market_data_copy["implied_vol"] < vol_med_mkt * 0.7, "Low Vol", "Normal")
                )
                regime_stats = market_data_copy.groupby(["regime", "market_id"]).first().groupby("regime").agg(
                    count=("resolution", "count"),
                    yes_rate=("resolution", "mean"),
                ).round(3)
                st.dataframe(regime_stats, use_container_width=True)

    with tab4:
        st.markdown("### Alternative Model Comparison")
        st.markdown("""
        We evaluated three modeling approaches:

        | Model | Type | Purpose |
        |-------|------|---------|
        | **Logistic Regression** | Parametric | Baseline, interpretable |
        | **Gradient Boosted Trees** | Non-parametric | Captures non-linear patterns |
        | **Bayesian (Beta-Binomial)** | Bayesian | Regime-aware updating |

        **Final choice: Ensemble (LR + GBT)** — combines interpretability with predictive power.

        The Bayesian model was rejected because it adapts too slowly to rapid regime changes
        in 15-minute windows. An LSTM model was also considered but rejected due to
        overfitting on limited data and poor calibration.
        """)

        tm = data.get("train_metrics", {})
        if tm and tm.get("status") == "trained":
            comparison_df = pd.DataFrame([
                {"Model": "Logistic Regression", "Brier Score": tm.get("lr_brier", 0),
                 "Accuracy": tm.get("lr_accuracy", 0), "Type": "Parametric"},
                {"Model": "Gradient Boosted Trees", "Brier Score": tm.get("gbt_brier", 0),
                 "Accuracy": tm.get("gbt_accuracy", 0), "Type": "Non-parametric"},
                {"Model": "Ensemble (LR+GBT)", "Brier Score": tm.get("ensemble_brier", 0),
                 "Accuracy": tm.get("ensemble_accuracy", 0), "Type": "Ensemble"},
            ])
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            import plotly.express as px
            fig = px.bar(
                comparison_df, x="Model", y="Brier Score",
                color="Model", title="Brier Score by Model (Lower is Better)",
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click **Run Analysis** to begin research analysis.")
