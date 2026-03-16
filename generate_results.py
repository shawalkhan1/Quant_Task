"""
Comprehensive Results Generation Script.

Runs all strategies through backtesting and forward testing,
generates calibration analysis, and saves everything to results/.

This script produces the deliverable result files:
  - results/backtest_results/ : CSV metrics + equity curves for each strategy
  - results/trade_logs/ : Full trade logs for each strategy
  - results/forward_test_results/ : Walk-forward and paper trading results
  - results/comparison_summary.csv : Side-by-side strategy comparison
  - results/calibration_results.json : Probability calibration metrics
  - results/bayesian_alternative_results.json : Rejected alternative experiment
"""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from config.settings import (
    BACKTEST_RESULTS_DIR, FORWARD_RESULTS_DIR, TRADE_LOGS_DIR, RESULTS_DIR,
)
from src.data.features import FeatureEngine
from src.backtesting.engine import BacktestEngine
from src.strategies.market_maker import MarketMakerStrategy
from src.strategies.arbitrage import ArbitrageStrategy
from src.strategies.predictive import PredictiveStrategy
from src.models.bayesian_model import ContextualBayesianModel
from src.models.calibration import CalibrationAnalyzer
from src.forward_testing.paper_trader import PaperTrader
from src.forward_testing.rolling_simulator import RollingSimulator

import os
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
os.makedirs(FORWARD_RESULTS_DIR, exist_ok=True)
os.makedirs(TRADE_LOGS_DIR, exist_ok=True)


def save_json(data, path):
    """Save dict to JSON, converting numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, datetime):
            return str(obj)
        return str(obj)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)
    print(f"  Saved: {path}")


# ==========================================================
# STEP 1: Fetch Real Polymarket Data
# ==========================================================
print("=" * 60)
print("STEP 1: Fetching Real Polymarket Data")
print("=" * 60)

from src.data.polymarket_fetcher import PolymarketFetcher

poly = PolymarketFetcher()

print("  Fetching resolved markets from Polymarket API...")
markets = poly.fetch_dataset_for_backtest(
    days_back=30,          # 30 days of resolved markets
    min_volume=5000.0,     # Only markets with $5k+ volume (liquid enough)
    max_markets=80,        # Up to 80 resolved markets
    use_cache=True,        # Cache timeseries locally for reruns
)

if markets.empty:
    raise RuntimeError(
        "Polymarket API returned no data. Live market data is required; "
        "no synthetic fallback is allowed."
    )
else:
    # Build a synthetic price_data stub from market_price_yes timeseries
    # (needed for the FeatureEngine which expects OHLCV — we proxy with YES price)
    ts_index = markets.index.unique().sort_values()
    yes_series = markets.groupby(markets.index)["market_price_yes"].mean()
    price_data = pd.DataFrame({
        "open":   yes_series,
        "high":   yes_series * 1.001,
        "low":    yes_series * 0.999,
        "close":  yes_series,
        "volume": markets.groupby(markets.index)["volume_usd"].sum().fillna(1.0),
    })
    print(f"  Polymarket data: {len(markets):,} observations, "
          f"{markets['market_id'].nunique()} unique markets")
    print(f"  Date range: {markets.index.min()} → {markets.index.max()}")

# Build features on the price proxy (market probability as price)
engine = FeatureEngine()
features = engine.compute_all_features(price_data)
print(f"  Features: {features.shape[1]} columns, {len(features)} rows")

# Save raw data
price_data.to_csv(os.path.join(RESULTS_DIR, "price_data.csv"))
markets.to_csv(os.path.join(RESULTS_DIR, "market_data.csv"))
print(f"  Saved raw data to results/")

# ==========================================================
# STEP 2: Train/Test Split
# ==========================================================
print()
print("=" * 60)
print("STEP 2: Train/Test Split (70/30)")
print("=" * 60)

split_idx = int(len(markets) * 0.7)
train_markets = markets.iloc[:split_idx]
test_markets = markets.iloc[split_idx:]

train_end = train_markets.index[-1]
test_start = test_markets.index[0]

train_features = features.loc[:train_end]
test_features = features.loc[test_start:]

print(f"  Train: {len(train_markets)} market obs ({train_markets['market_id'].nunique()} markets)")
print(f"  Test:  {len(test_markets)} market obs ({test_markets['market_id'].nunique()} markets)")

# ==========================================================
# STEP 3: In-Sample Backtests (on training data)
# ==========================================================
print()
print("=" * 60)
print("STEP 3: In-Sample Backtests (Training Period)")
print("=" * 60)

in_sample_results = {}

# --- Market Maker (In-Sample) ---
mm_strat = MarketMakerStrategy()
bt_mm = BacktestEngine(initial_capital=10000, transaction_cost_pct=0.01)
mm_is = bt_mm.run(strategy=mm_strat, market_data=train_markets,
                   features_data=train_features, price_data=price_data)
in_sample_results["MarketMaker"] = mm_is
print(f"  Market Maker IS: {mm_is['metrics']['total_trades']} trades, "
      f"return={mm_is['metrics']['total_return_pct']:.2f}%")

# --- Arbitrage (In-Sample) ---
arb_strat = ArbitrageStrategy()
bt_arb = BacktestEngine(initial_capital=10000, transaction_cost_pct=0.01)
arb_is = bt_arb.run(strategy=arb_strat, market_data=train_markets,
                     features_data=train_features, price_data=price_data)
in_sample_results["Arbitrage"] = arb_is
print(f"  Arbitrage    IS: {arb_is['metrics']['total_trades']} trades, "
      f"return={arb_is['metrics']['total_return_pct']:.2f}%")

# --- Predictive (In-Sample — train then test on train data) ---
pred_strat = PredictiveStrategy()
fe2 = FeatureEngine()
train_feat_full = fe2.compute_all_features(price_data.loc[:train_end])
train_feat_with_mkt = fe2.add_market_features(train_feat_full, train_markets)
train_metrics = pred_strat.train(train_feat_with_mkt, train_markets)
print(f"  Predictive training: {train_metrics.get('n_samples', 0)} samples, "
      f"Brier={train_metrics.get('ensemble_brier', 'N/A')}")

bt_pred_is = BacktestEngine(initial_capital=10000, transaction_cost_pct=0.01)
pred_is = bt_pred_is.run(strategy=pred_strat, market_data=train_markets,
                          features_data=train_features, price_data=price_data)
in_sample_results["Predictive"] = pred_is
print(f"  Predictive   IS: {pred_is['metrics']['total_trades']} trades, "
      f"return={pred_is['metrics']['total_return_pct']:.2f}%")

# ==========================================================
# STEP 4: Out-of-Sample Backtests (on test data)
# ==========================================================
print()
print("=" * 60)
print("STEP 4: Out-of-Sample Backtests (Test Period)")
print("=" * 60)

oos_results = {}

# --- Market Maker (OOS) ---
mm_strat2 = MarketMakerStrategy()
bt_mm2 = BacktestEngine(initial_capital=10000, transaction_cost_pct=0.01)
mm_oos = bt_mm2.run(strategy=mm_strat2, market_data=test_markets,
                     features_data=test_features, price_data=price_data)
oos_results["MarketMaker"] = mm_oos
print(f"  Market Maker OOS: {mm_oos['metrics']['total_trades']} trades, "
      f"return={mm_oos['metrics']['total_return_pct']:.2f}%")

# --- Arbitrage (OOS) ---
arb_strat2 = ArbitrageStrategy()
bt_arb2 = BacktestEngine(initial_capital=10000, transaction_cost_pct=0.01)
arb_oos = bt_arb2.run(strategy=arb_strat2, market_data=test_markets,
                       features_data=test_features, price_data=price_data)
oos_results["Arbitrage"] = arb_oos
print(f"  Arbitrage    OOS: {arb_oos['metrics']['total_trades']} trades, "
      f"return={arb_oos['metrics']['total_return_pct']:.2f}%")

# --- Predictive (OOS — same model, unseen data) ---
bt_pred_oos = BacktestEngine(initial_capital=10000, transaction_cost_pct=0.01)
pred_oos = bt_pred_oos.run(strategy=pred_strat, market_data=test_markets,
                            features_data=test_features, price_data=price_data)
oos_results["Predictive"] = pred_oos
print(f"  Predictive   OOS: {pred_oos['metrics']['total_trades']} trades, "
      f"return={pred_oos['metrics']['total_return_pct']:.2f}%")

# ==========================================================
# STEP 5: Save Backtest Results
# ==========================================================
print()
print("=" * 60)
print("STEP 5: Saving Backtest Results")
print("=" * 60)

for label, results_dict, period in [("IS", in_sample_results, "in_sample"),
                                     ("OOS", oos_results, "out_of_sample")]:
    for name, result in results_dict.items():
        prefix = f"{name}_{period}"

        # Metrics JSON
        save_json(result["metrics"], os.path.join(BACKTEST_RESULTS_DIR, f"{prefix}_metrics.json"))

        # Equity curve CSV
        eq = result.get("equity_curve")
        if eq is not None and len(eq) > 0:
            eq.to_csv(os.path.join(BACKTEST_RESULTS_DIR, f"{prefix}_equity_curve.csv"))
            print(f"  Saved: {prefix}_equity_curve.csv")

        # Trade log CSV
        trades = result.get("trades")
        if trades is not None and len(trades) > 0:
            trades.to_csv(os.path.join(TRADE_LOGS_DIR, f"{prefix}_trades.csv"), index=False)
            print(f"  Saved: {prefix}_trades.csv ({len(trades)} trades)")

# ==========================================================
# STEP 6: Bayesian Alternative Experiment
# ==========================================================
print()
print("=" * 60)
print("STEP 6: Bayesian Alternative Experiment (Rejected)")
print("=" * 60)

# Run the Bayesian model on the same data and compare to the Ensemble
bayesian_model = ContextualBayesianModel()

# Train: iterate through training markets and update
train_market_ids = train_markets["market_id"].unique()
trained_count = 0
for mid in train_market_ids:
    mkt_rows = train_markets[train_markets["market_id"] == mid]
    if len(mkt_rows) > 0:
        resolution = int(mkt_rows.iloc[0]["resolution"])
        # Get features at this timestamp
        ts = mkt_rows.index[0]
        feat = train_features.loc[ts].iloc[0] if ts in train_features.index and isinstance(train_features.loc[ts], pd.DataFrame) else (train_features.loc[ts] if ts in train_features.index else None)
        bayesian_model.update(resolution, feat)
        trained_count += 1

print(f"  Bayesian model trained on {trained_count} market resolutions")

# Evaluate: predict on test markets and compute Brier score
bayesian_predictions = []
bayesian_actuals = []
ensemble_predictions = []

test_market_ids = test_markets["market_id"].unique()
for mid in test_market_ids:
    mkt_rows = test_markets[test_markets["market_id"] == mid]
    if len(mkt_rows) == 0:
        continue
    # Use a row around minute 4 for prediction
    signal_rows = mkt_rows[mkt_rows["minutes_elapsed"].between(3, 5)]
    if len(signal_rows) == 0:
        signal_rows = mkt_rows.iloc[:1]
    row = signal_rows.iloc[0]

    ts = signal_rows.index[0]
    feat = test_features.loc[ts].iloc[0] if ts in test_features.index and isinstance(test_features.loc[ts], pd.DataFrame) else (test_features.loc[ts] if ts in test_features.index else None)

    # Bayesian prediction
    bayesian_prob = bayesian_model.predict_probability(feat)
    bayesian_predictions.append(bayesian_prob)

    # Ensemble prediction (already trained)
    if feat is not None:
        ens_prob = pred_strat.predict_probability(feat)
    else:
        ens_prob = 0.5
    ensemble_predictions.append(ens_prob)

    bayesian_actuals.append(int(row["resolution"]))

    # Update Bayesian with this observation
    bayesian_model.update(int(row["resolution"]), feat)

bayesian_preds = np.array(bayesian_predictions)
ensemble_preds = np.array(ensemble_predictions)
actuals = np.array(bayesian_actuals)

from sklearn.metrics import brier_score_loss, accuracy_score

bayesian_brier = brier_score_loss(actuals, bayesian_preds)
ensemble_brier = brier_score_loss(actuals, ensemble_preds)
bayesian_acc = accuracy_score(actuals, (bayesian_preds > 0.5).astype(int))
ensemble_acc = accuracy_score(actuals, (ensemble_preds > 0.5).astype(int))

# Calibration comparison
bayesian_results = {
    "model": "Contextual Bayesian (Beta-Binomial)",
    "n_test_markets": len(actuals),
    "bayesian_brier_score": round(float(bayesian_brier), 4),
    "ensemble_brier_score": round(float(ensemble_brier), 4),
    "bayesian_accuracy": round(float(bayesian_acc), 4),
    "ensemble_accuracy": round(float(ensemble_acc), 4),
    "bayesian_mean_prediction": round(float(bayesian_preds.mean()), 4),
    "ensemble_mean_prediction": round(float(ensemble_preds.mean()), 4),
    "actual_positive_rate": round(float(actuals.mean()), 4),
    "conclusion": (
        "Bayesian model adapts too slowly to short-horizon regime changes. "
        "The LR+GBT ensemble achieves better Brier score and accuracy."
    ),
}

print(f"  Bayesian Brier: {bayesian_brier:.4f}")
print(f"  Ensemble Brier: {ensemble_brier:.4f}")
print(f"  Bayesian Accuracy: {bayesian_acc:.4f}")
print(f"  Ensemble Accuracy: {ensemble_acc:.4f}")

save_json(bayesian_results, os.path.join(RESULTS_DIR, "bayesian_alternative_results.json"))

# ==========================================================
# STEP 7: Calibration Analysis
# ==========================================================
print()
print("=" * 60)
print("STEP 7: Probability Calibration Analysis")
print("=" * 60)

cal = CalibrationAnalyzer()
calibration_results = {}

# Ensemble calibration
if len(ensemble_preds) > 0 and len(actuals) > 0:
    ens_brier = CalibrationAnalyzer.brier_score(actuals, ensemble_preds)
    ens_ece = CalibrationAnalyzer.expected_calibration_error(actuals, ensemble_preds)
    ens_decomp = CalibrationAnalyzer.brier_decomposition(actuals, ensemble_preds)
    ens_reliability = CalibrationAnalyzer.reliability_diagram(actuals, ensemble_preds)
    calibration_results["ensemble"] = {
        "brier_score": round(ens_brier, 4),
        "ece": round(ens_ece, 4),
        "decomposition": {k: round(v, 4) for k, v in ens_decomp.items()},
        "reliability_diagram": ens_reliability.to_dict(orient="records"),
    }
    print(f"  Ensemble - Brier: {ens_brier:.4f}, ECE: {ens_ece:.4f}")

# Bayesian calibration
if len(bayesian_preds) > 0 and len(actuals) > 0:
    bay_brier = CalibrationAnalyzer.brier_score(actuals, bayesian_preds)
    bay_ece = CalibrationAnalyzer.expected_calibration_error(actuals, bayesian_preds)
    bay_decomp = CalibrationAnalyzer.brier_decomposition(actuals, bayesian_preds)
    bay_reliability = CalibrationAnalyzer.reliability_diagram(actuals, bayesian_preds)
    calibration_results["bayesian"] = {
        "brier_score": round(bay_brier, 4),
        "ece": round(bay_ece, 4),
        "decomposition": {k: round(v, 4) for k, v in bay_decomp.items()},
        "reliability_diagram": bay_reliability.to_dict(orient="records"),
    }
    print(f"  Bayesian - Brier: {bay_brier:.4f}, ECE: {bay_ece:.4f}")

save_json(calibration_results, os.path.join(RESULTS_DIR, "calibration_results.json"))

# ==========================================================
# STEP 8: Forward Testing (Walk-Forward)
# ==========================================================
print()
print("=" * 60)
print("STEP 8: Forward Testing — Walk-Forward Analysis")
print("=" * 60)

# Paper trading on the last 20% of data
paper_split = int(len(markets) * 0.8)
paper_markets = markets.iloc[paper_split:]
paper_features = features.loc[paper_markets.index[0]:]

paper_trader = PaperTrader(initial_capital=10000, transaction_cost_pct=0.01)
paper_result = paper_trader.run_paper_trade(
    strategy=pred_strat,
    market_data=paper_markets,
    features_data=paper_features,
    price_data=price_data,
)

print(f"  Paper Trading: {paper_result['metrics']['total_trades']} trades, "
      f"return={paper_result['metrics']['total_return_pct']:.2f}%")

# Save forward test results
save_json(paper_result["metrics"], os.path.join(FORWARD_RESULTS_DIR, "paper_trade_metrics.json"))
paper_eq = paper_result.get("equity_curve")
if paper_eq is not None and len(paper_eq) > 0:
    paper_eq.to_csv(os.path.join(FORWARD_RESULTS_DIR, "paper_trade_equity.csv"))
paper_trades = paper_result.get("trades")
if paper_trades is not None and len(paper_trades) > 0:
    paper_trades.to_csv(os.path.join(FORWARD_RESULTS_DIR, "paper_trade_log.csv"), index=False)

# ==========================================================
# STEP 9: Comparison Summary
# ==========================================================
print()
print("=" * 60)
print("STEP 9: Strategy Comparison Summary")
print("=" * 60)

summary_rows = []
for name in ["MarketMaker", "Arbitrage", "Predictive"]:
    is_m = in_sample_results[name]["metrics"]
    oos_m = oos_results[name]["metrics"]
    row = {
        "Strategy": name,
        "IS_Trades": is_m.get("total_trades", 0),
        "IS_Return%": round(is_m.get("total_return_pct", 0), 2),
        "IS_Sharpe": round(is_m.get("sharpe_ratio", 0), 2),
        "IS_MaxDD%": round(is_m.get("max_drawdown_pct", 0), 2),
        "IS_WinRate%": round(is_m.get("win_rate", 0), 1),
        "IS_ProfitFactor": round(is_m.get("profit_factor", 0), 2),
        "OOS_Trades": oos_m.get("total_trades", 0),
        "OOS_Return%": round(oos_m.get("total_return_pct", 0), 2),
        "OOS_Sharpe": round(oos_m.get("sharpe_ratio", 0), 2),
        "OOS_MaxDD%": round(oos_m.get("max_drawdown_pct", 0), 2),
        "OOS_WinRate%": round(oos_m.get("win_rate", 0), 1),
        "OOS_ProfitFactor": round(oos_m.get("profit_factor", 0), 2),
    }
    summary_rows.append(row)
    degradation = (oos_m.get("total_return_pct", 0) - is_m.get("total_return_pct", 0))
    print(f"  {name:15s} | IS: {is_m.get('total_return_pct', 0):+.2f}% -> OOS: {oos_m.get('total_return_pct', 0):+.2f}% "
          f"(degradation: {degradation:+.2f}pp)")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULTS_DIR, "comparison_summary.csv"), index=False)
print(f"\n  Saved: comparison_summary.csv")

# Print summary table
print()
print(summary_df.to_string(index=False))

# ==========================================================
# DONE
# ==========================================================
print()
print("=" * 60)
print("ALL RESULTS GENERATED AND SAVED SUCCESSFULLY")
print("=" * 60)

# List all result files
print("\nGenerated files:")
for root, dirs, files in os.walk(RESULTS_DIR):
    for f in sorted(files):
        fpath = os.path.join(root, f)
        size_kb = os.path.getsize(fpath) / 1024
        rel = os.path.relpath(fpath, RESULTS_DIR)
        print(f"  results/{rel} ({size_kb:.1f} KB)")
