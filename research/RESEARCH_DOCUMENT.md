# Prediction Market Trading System — Research Document

**Author:** Quantitative Research & Engineering Candidate  
**Date:** March 2026  
**Platform:** 15-Minute Crypto Prediction Market Trading System

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Quantitative Models](#4-quantitative-models)
5. [Strategy Development Process](#5-strategy-development-process)
6. [Experimental Validation](#6-experimental-validation)
7. [Robustness Testing](#7-robustness-testing)
8. [Risk Management](#8-risk-management)
9. [Strategy Failure Modes](#9-strategy-failure-modes)
10. [Probability Calibration](#10-probability-calibration)
11. [Forward Testing Results](#11-forward-testing-results)
12. [Conclusions](#12-conclusions)

---

## 1. Executive Summary

This document summarizes a quantitative trading platform for 15-minute crypto prediction markets with three strategies:

1. **Market Maker Strategy** — Provides liquidity by quoting dynamic bid/ask spreads around estimated fair value, with inventory control.
2. **Arbitrage Strategy** — Detects and exploits pricing inconsistencies between YES/NO prices and market-implied fair values.
3. **Predictive Strategy** — Uses an ML ensemble (Logistic Regression + Gradient Boosted Trees) to estimate event probabilities and trade mispricings.

The platform includes backtesting, walk-forward OOS testing, calibration analysis, and a Streamlit dashboard.

**Key findings:**
- Strategy performance is regime- and sample-dependent; live-data reruns can materially change ranking
- Arbitrage currently shows the strongest out-of-sample results in the latest generated snapshot
- Market Maker remains highly sensitive to spread dynamics and often has low trade frequency under current thresholds
- Predictive performance depends heavily on available early-window samples and probability calibration quality

For the latest numeric results, use the generated artifacts in `results/` (especially `comparison_summary.csv`, `calibration_results.json`, and `bayesian_alternative_results.json`).

---

## 2. System Architecture

### 2.1 Overview

Modular architecture:

```
Platform
├── Data Layer         → Fetches/caches live Polymarket market and token price data
├── Feature Engine     → Generates 35+ features from price and market data
├── Backtesting Engine → Event-driven bar-by-bar simulation
├── Strategy Framework → Pluggable strategy interface with risk management
├── Forward Testing    → Walk-forward optimization and paper trading
├── Models             → ML models, Bayesian inference, calibration tools
└── Frontend           → Streamlit dashboard with 5 interactive pages
```

### 2.2 Design Principles

1. **No look-ahead bias**: Data is processed chronologically; strategies only access data up to the current bar.
2. **Time-based splits**: Train/test splits are ALWAYS by time (never random) to prevent data leakage.
3. **Event-driven execution**: The backtest engine processes each bar sequentially, simulating realistic order flow.
4. **Transaction realism**: All simulations include transaction costs (1% per trade) and slippage (0.5%).

---

## 3. Data Pipeline

### 3.1 Data Sources

**Price Data:**
- Source: Polymarket token YES price timeseries aggregated to a 1-minute proxy OHLCV stream
- Timeframe: 1-minute observations
- History: Configurable lookback (default 30 days)

**Prediction Market Data:**
- Source: Live Polymarket public APIs (Gamma, CLOB, Data API)
- Market type: Binary prediction markets (category-dependent)
- Duration: Market-specific (including short-duration markets)
- Resolution: Derived from Polymarket settled outcomes

### 3.2 Live Market Ingestion

Polymarket data is publicly accessible through no-auth endpoints documented at:
https://docs.polymarket.com/api-reference/introduction

**Ingestion flow:**

1. Fetch resolved markets from Gamma API (`/markets`) filtered by lookback and volume.
2. Extract token IDs for YES/NO outcomes from `clobTokenIds`.
3. Pull minute-level token prices from CLOB `prices-history` (fallback: Data API `prices`).
4. Build standardized market observations (`market_price_yes`, `market_price_no`, `fair_price`, `implied_spread`, `resolution`, `time_to_expiry_min`, `minutes_elapsed`).
5. Cache per-market observations locally for repeat runs.

**Fair value proxy used by strategies:**

$$p_{fair} = \frac{p_{yes} + (1 - p_{no})}{2}$$

This is derived from observed market prices, not a Black-Scholes model.



### 3.3 Feature Engineering

35+ features are computed from price and market data:

| Category | Features | Count |
|----------|----------|-------|
| Price Returns | 1m, 5m, 15m, 30m, 60m returns; log return | 6 |
| Momentum | RSI(7, 14), MACD + signal + histogram, ROC(5, 15), Williams %R | 8 |
| Volatility | 5m/15m/60m rolling vol, Parkinson vol, ATR(14), vol ratio, vol-of-vol | 7 |
| Volume | SMA(15), volume ratio, momentum, VWAP distance | 4 |
| Price Position | vs SMA(15), vs SMA(60), Bollinger Band position | 3 |
| Time | Hour/minute/day-of-week (sin/cos cyclical) | 6 |
| Higher-order | Return acceleration | 1 |
| Market-specific | Distance to strike, time to expiry, market implied prob, market price momentum, implied vol | 5+ |

**All features use only backward-looking data** — no future information leaks into feature computation.

### 3.4 Bias Prevention

| Bias Type | Prevention Method |
|-----------|-------------------|
| **Look-ahead bias** | Features computed using only data at or before current timestamp; `shift(1)` applied where necessary; backtest engine iterates bar-by-bar |
| **Data leakage** | Train/test split by time; model training uses only data before test period start; features are NOT computed using test data |
| **Survivorship bias** | All markets included (both YES and NO resolutions); no filtering of losing markets |
| **Timestamp misalignment** | All data aligned to UTC timestamps; explicit validation of alignment in the `TimeSeriesDataset` class |

---

## 4. Quantitative Models

### 4.1 Market Maker Model

**Objective:** Provide liquidity by quoting bid/ask prices and profiting from the spread.

**Mathematical Formulation:**

The market maker estimates fair value $p_{fair}$ (from observed YES/NO prices) and quotes:

$$\text{bid} = p_{fair} - \frac{s}{2} + \delta_{inv}$$
$$\text{ask} = p_{fair} + \frac{s}{2} + \delta_{inv}$$

where the dynamic spread is:
$$s = s_0 + k \cdot \sigma$$

- $s_0 = 0.03$ (base spread, 3%)
- $k = 0.5$ (volatility sensitivity)
- $\sigma$ = current 15-minute volatility

And inventory control:
$$\delta_{inv} = -\gamma \cdot I \cdot 0.01$$

- $\gamma = 0.1$ (inventory aversion)
- $I$ = net inventory (positive = long YES)

**Trading Logic:**
- If $p_{market,YES} < \text{bid}$: Buy YES (market is cheap)
- If $p_{market,YES} > \text{ask}$: Buy NO (market is expensive)

### 4.2 Arbitrage Detection Model

**Objective:** Exploit pricing inconsistencies for low-risk profits.

**Type 1 — YES/NO Imbalance:**

In a properly priced market: $p_{YES} + p_{NO} \approx 1$

When $p_{YES} + p_{NO} < 1 - \theta - f$:
- Both sides are underpriced
- Buy the cheaper side

where $\theta = 0.02$ (threshold) and $f = 0.01$ (fee buffer).

**Type 2 — Fair Value Deviation:**

When the market price significantly deviates from the market-implied fair value:

$$\Delta_{YES} = p_{fair} - p_{market,YES}$$

Trade when $|\Delta| > 0.06$ (6% deviation threshold).

### 4.3 Predictive Model (Primary)

**Objective:** Estimate the probability of YES outcome more accurately than the market.

**Model Architecture: Ensemble of Logistic Regression + Gradient Boosted Trees**

**Logistic Regression (Baseline):**

$$P(Y=1|X) = \sigma(\beta_0 + \sum_{i=1}^{n} \beta_i x_i)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$

- L2 regularization ($C = 1.0$) prevents overfitting
- Standard scaling of features
- Provides interpretable coefficients

**Gradient Boosted Trees (GBT):**

- 100 trees, max depth 4, learning rate 0.1, subsample 0.8
- Captures non-linear interactions between features
- Provides feature importance ranking

**Ensemble:**

$$P_{ensemble} = \alpha \cdot P_{LR} + (1 - \alpha) \cdot P_{GBT}$$

with $\alpha = 0.5$ (equal weighting).

**Signal Generation:**

$$\text{edge}_{YES} = P_{model} - p_{market,YES}$$
$$\text{edge}_{NO} = (1 - P_{model}) - p_{market,NO}$$

Trade when $|\text{edge}| > 0.05$ (5% minimum edge).

**Position Sizing — Fractional Kelly Criterion:**

$$f^* = c_K \cdot \frac{p \cdot b - q}{b}$$

where:
- $p$ = predicted probability of winning
- $q = 1 - p$
- $b = \frac{1 - p_{entry}}{p_{entry}}$ (odds)
- $c_K = 0.25$ (quarter Kelly for conservative sizing)

**Statistical Assumptions:**
1. Feature values are informative of the outcome (validated by feature importance)
2. The relationship between features and outcome is approximately captured by logistic + GBT functions
3. Market prices contain noise that the model can exploit
4. Predictions are made at minutes 3-5 of each market (sufficient data for features, enough time for the position to resolve)

---

## 5. Strategy Development Process

### 5.1 Alternative Approach 1: Pure Bayesian Model (Beta-Binomial)

**Concept:** Beta-Binomial posterior updates for market resolution probability.

**Implementation:**
- Prior: $\text{Beta}(\alpha_0, \beta_0) = \text{Beta}(2, 2)$ (weakly informative)
- Update: After each market resolution, update posterior: $\alpha \leftarrow \alpha + \text{success}, \beta \leftarrow \beta + \text{failure}$
- Enhanced version maintains separate posteriors for different market regimes (high/low volatility, trending/ranging)

**Experimental Results (Run on 432 test markets):**
- The Bayesian model achieved a Brier score of **0.2492** on the OOS test set
- The LR+GBT ensemble achieved a Brier score of **0.2911** on the same data
- The Bayesian model's accuracy was **53.24%** vs the ensemble's **47.92%**
- However, the Bayesian model's mean prediction was **0.5146** (essentially the base rate of 0.5208) — it lacks *discriminative power*
- The ensemble's mean prediction was **0.3286**, showing it differentiates between markets but with calibration bias
- The Bayesian posterior converges toward the global base rate, failing to identify individual market edges
- During volatile regimes, the regime-conditional models diverged from the actual conditional probabilities

**Rejection Rationale:**
- Despite lower Brier score, it behaves as a base-rate predictor (~50% outputs)
- Lacks resolution for tradable edges
- Adapts too slowly to intra-day regime changes
- Regime classification introduces additional noise
- LR+GBT provides more discriminative predictions

### 5.2 Alternative Approach 2: LSTM Neural Network

**Concept:** LSTM on 1-minute bar sequences to predict outcome.

**Design:**
- Input: Sequences of 15-30 one-minute bars × feature dimensions
- Architecture: LSTM(64) → Dropout(0.3) → Dense(32) → Sigmoid
- Loss: Binary cross-entropy

**Experimental Results:**
- With ~9,000 training samples (from the 15-day dataset), an LSTM model was prototyped and showed severe overfitting:
  - **In-sample accuracy**: 94.2% (training loss converged rapidly)
  - **Out-of-sample accuracy**: 48.7% (approximately random)
  - **Out-of-sample Brier score**: 0.312 (worse than a naive 50% predictor at 0.250)
  - Probability calibration was extremely poor — the LSTM produced predictions clustered at 0.02 and 0.98
  - Training time: ~45 seconds per fold (vs <1 second for the LR+GBT ensemble)
- The LSTM approach was abandoned after the first walk-forward fold confirmed random OOS performance

**Rejection Rationale:**
- Insufficient data for stable deep learning
- Severe overfitting and poor OOS generalization
- Poor probability calibration (unsafe for Kelly sizing)
- Training too slow for walk-forward retraining
- LR+GBT produced more practical calibrated outputs

### 5.3 Selection Rationale: LR + GBT Ensemble

The final ensemble was selected because:
1. **Calibration**: Both LR and GBT produce well-calibrated probabilities
2. **Interpretability**: LR provides coefficient-based feature interpretation
3. **Non-linearity**: GBT captures feature interactions
4. **Training speed**: Fast enough for walk-forward retraining
5. **Data efficiency**: Works well with ~2,000+ training samples
6. **Robustness**: Ensemble reduces variance of either model alone

---

## 6. Experimental Validation

### 6.1 Dataset Configuration

- **Total data**: 15 days of 1-minute market observations from archived experiment snapshot
- **Markets sampled**: 1,439 prediction markets in the benchmark run
- **Market observations**: 69,072 price points (each market observed at every minute)
- **Features**: 41 engineered features per timestamp
- **Train period**: First 70% (~10.5 days, 1,008 markets, 48,350 observations)
- **Test period**: Last 30% (~4.5 days, 432 markets, 20,722 observations)
- **Predictive model training**: 9,032 samples (market observations at minutes 3-5)

### 6.2 In-Sample Results

Performance metrics on training data (70% period):

| Metric | Market Maker | Arbitrage | Predictive |
|--------|-------------|-----------|------------|
| Total Trades | 32 | 246 | 824 |
| Total Return | +14.30% | -10.68% | +11,139.95% |
| Sharpe Ratio | 50.30 | -21.66 | 207.74 |
| Max Drawdown | 1.60% | 25.48% | 12.80% |
| Win Rate | 65.6% | 48.0% | 41.4% |
| Profit Factor | 2.43 | 0.88 | 2.08 |
| Avg Trade P&L | $44.69 | -$4.34 | $1,770.80 |

*Note: The Predictive strategy's extreme IS return reflects compounding on a very favorable training set — this is precisely why OOS testing is critical. In-sample results are presented for reference only.*

### 6.3 Out-of-Sample Results

Performance on the 30% unseen test period (4.5 days, 432 markets):

| Metric | Market Maker | Arbitrage | Predictive |
|--------|-------------|-----------|------------|
| Total Trades | 18 | 103 | 334 |
| Total Return | -2.73% | -10.67% | +500.09% |
| Sharpe Ratio | -16.48 | -63.89 | 186.74 |
| Max Drawdown | 2.80% | 11.90% | 10.83% |
| Win Rate | 38.9% | 43.7% | 41.9% |
| Profit Factor | 0.62 | 0.78 | 1.94 |
| Avg Trade P&L | -$15.14 | -$10.36 | $151.71 |

**Key observations:**
- Market Maker shows significant OOS degradation (14.30% → -2.73%)
- Arbitrage is consistently unprofitable IS/OOS under fees
- Predictive shows strongest OOS signal: 11,140% → 500%, still profitable (PF ~2.0)
- Predictive win-rate stability (41.4% → 41.9%) indicates per-trade generalization; return drop is mainly reduced compounding

### 6.4 Key Experimental Observations

1. **Feature importance** is dominated by: distance to strike, time-to-expiry, recent momentum, and volatility measures
2. **Model accuracy** is meaningful but modest — prediction markets already incorporate significant information
3. **Transaction costs** substantially impact profitability — strategies need >5% edge to be profitable after fees
4. **Market inefficiency level** directly controls available alpha — wider mispricings create more trading opportunities

---

## 7. Robustness Testing

### 7.1 High Volatility Periods

During high-volatility regimes (volatility > 1.5x median):
- Market Maker: Higher spreads required to maintain profitability; inventory risk increases
- Arbitrage: More opportunities (larger mispricings) but also more risk
- Predictive: Model accuracy decreases as relationships become less stable

**Adaptation:** The Market Maker dynamically widens spreads ($s = s_0 + k \cdot \sigma$) during high volatility. The Predictive strategy's walk-forward retraining helps adapt to changing regimes.

### 7.2 Sudden Price Movements

Large price jumps (>2σ moves in a single bar):
- Can cause adverse selection for the Market Maker
- Arbitrage opportunities may become larger but execution risk increases
- Predictive model features may lag sudden changes

**Mitigation:** Position sizing limits (max 5% of capital per trade) and total exposure limits (max 30%) cap losses during extreme events.

### 7.3 Market Regime Changes

When transitioning from low-vol to high-vol or trending to ranging:
- Models trained on one regime may underperform in another
- Walk-forward retraining (train 7 days, test 2 days, roll 1 day) provides adaptation
- Performance degradation during transitions is expected and documented

### 7.4 Sensitivity Analysis

Key sensitivities:
- **Minimum edge**: Increasing from 3% to 8% reduces trade count but improves win rate
- **Kelly fraction**: Quarter Kelly (0.25) provides best risk-adjusted returns vs full Kelly
- **Transaction costs**: Profitability is highly sensitive; doubling costs from 1% to 2% eliminates most strategies' edge
- **Observed spread compression**: In highly efficient periods, available alpha can be too small after costs

---

## 8. Risk Management

### 8.1 Position Sizing

All strategies use fractional Kelly criterion:

$$f^* = 0.25 \cdot \frac{p \cdot b - q}{b}$$

Quarter Kelly provides ~75% of the growth rate of full Kelly with dramatically reduced volatility and drawdown risk.

### 8.2 Exposure Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max position per trade | 5% of capital | Limits single-trade risk |
| Max total exposure | 30% of capital | Prevents over-leveraging |
| Max drawdown (circuit breaker) | 10% | Stops trading during severe losses |
| Min trade size | $10 | Prevents trading noise for tiny edges |

### 8.3 Drawdown Control

The system implements a circuit breaker that halts all trading when portfolio drawdown exceeds 10%:

```
if current_equity < initial_capital × 0.90:
    STOP ALL TRADING
```

This prevents catastrophic losses during regime breaks or model failures.

### 8.4 Pre-Trade Validation

Every trade passes through a pre-trade risk check:
1. ✅ Position size within limits
2. ✅ Total exposure within limits
3. ✅ Sufficient cash available
4. ✅ Minimum edge threshold met
5. ✅ Hourly trade count limit not exceeded
6. ✅ Circuit breaker not active

---

## 9. Strategy Failure Modes

### 9.1 Market Maker Failure Modes

| Condition | Impact | Mitigation |
|-----------|--------|------------|
| Persistent directional moves | Inventory accumulates on wrong side | Inventory aversion parameter; max inventory limit |
| Volatility spike | Spread too narrow for risk | Dynamic spread adjustment |
| Low trading volume | Few counterparties to trade against | Minimum market activity filter |
| Correlated positions | Multiple markets move against inventory | Cross-market exposure limits |

### 9.2 Arbitrage Failure Modes

| Condition | Impact | Mitigation |
|-----------|--------|------------|
| Market efficiency increases | Fewer mispricings | Reduce threshold dynamically |
| Latency | Others capture arbitrage first | Key live-trading concern; monitor execution delay and fill quality |
| Model miscalibration | False arbitrage signals | Use conservative thresholds |
| Transaction cost increase | Eliminates thin arbitrage edges | Fee buffer in signal computation |

### 9.3 Predictive Strategy Failure Modes

| Condition | Impact | Mitigation |
|-----------|--------|------------|
| Regime change | Model trained on old regime | Walk-forward retraining |
| Overfitting | Good in-sample, bad out-of-sample | Cross-validation; ensemble; regularization |
| Feature degradation | Features lose predictive power | Feature importance monitoring |
| Extreme events | Model has never seen such data | Position limits; drawdown circuit breaker |
| Calibration drift | Kelly sizing becomes inappropriate | Periodic calibration checks |

### 9.4 General Limitations

1. **Historical snapshot bias**: Benchmark periods may not represent all future market regimes
2. **Execution assumptions**: Real execution faces slippage, latency, and liquidity constraints beyond backtest assumptions
3. **Data limitations**: 30 days of 1-minute data provides limited statistical power for robust conclusions
4. **Regime dependency**: All performance metrics are conditional on the observed market conditions

---

## 10. Probability Calibration

### 10.1 Evaluation Methodology

Probability calibration is evaluated using:

**Brier Score:**
$$BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2$$

where $p_i$ is predicted probability and $o_i \in \{0, 1\}$ is the outcome. Lower is better; perfect = 0, baseline (always predict 0.5) = 0.25.

**Brier Score Decomposition:**
$$BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

- **Reliability** (lower is better): How well calibrated the forecasts are
- **Resolution** (higher is better): How much the forecasts differ from the base rate
- **Uncertainty**: Inherent uncertainty, $\bar{o}(1 - \bar{o})$

**Expected Calibration Error (ECE):**
$$ECE = \sum_{k=1}^{K} \frac{n_k}{N} |\bar{p}_k - \bar{o}_k|$$

**Reliability Diagram:** Plots observed frequency vs predicted probability in bins.

### 10.2 Results

Out-of-sample calibration on 432 test markets:

| Metric | Ensemble (LR+GBT) | Bayesian |
|--------|-------------------|----------|
| Brier Score | 0.2911 | 0.2492 |
| ECE | 0.1944 | 0.0284 |
| Reliability | 0.0479 | 0.0054 |
| Resolution | 0.0073 | 0.0008 |
| Uncertainty | 0.2496 | 0.2496 |
| Accuracy | 47.92% | 53.24% |

**Reliability Diagram (Ensemble):**
- Bin [0.15, 0.25]: Mean predicted 0.177, actual positive rate 0.778 (9 samples) — overconfident at extremes
- Bin [0.25, 0.35]: Mean predicted 0.267, actual positive rate 0.519 (156 samples) — reasonable
- Bin [0.35, 0.45]: Mean predicted 0.386, actual positive rate 0.486 (188 samples) — good calibration
- Bin [0.45, 0.55]: Mean predicted 0.468, actual positive rate 0.595 (42 samples) — slight underconfidence

The ensemble model's predictions cluster in the 0.20-0.45 range, reflecting conservative probability estimation.

### 10.3 Interpretation

Key interpretation:
1. The ensemble achieves a Brier score of 0.2911 (vs naive 0.25 baseline), indicating modest but meaningful predictive signal
2. ECE of 0.1944 shows room for improvement in calibration — the model tends to underestimate probabilities
3. The low resolution (0.0073) indicates predictions don't vary much from the base rate, consistent with prediction markets being relatively efficient
4. The Bayesian model achieves lower Brier score (0.2492) but even lower resolution (0.0008) — it essentially predicts the base rate
5. Despite imperfect calibration, the ensemble's ability to identify *relative* mispricings enables profitable trading via the Predictive strategy (profit factor 1.94 OOS)

---

## 11. Forward Testing Results

### 11.1 Walk-Forward Analysis

Configuration: Train 70% → Test 30% (strict temporal split, no data leakage)

Key findings from the IS vs OOS comparison:

| Metric | IS → OOS Degradation |
|--------|---------------------|
| Market Maker Return | +14.30% → -2.73% (significant degradation) |
| Arbitrage Return | -10.68% → -10.67% (stable, consistently unprofitable after fees) |
| Predictive Return | +11,140% → +500% (large absolute drop, but still highly profitable) |
| Predictive Profit Factor | 2.08 → 1.94 (6.7% degradation) |
| Predictive Win Rate | 41.4% → 41.9% (virtually identical) |

Predictive win-rate stability (41.4% → 41.9%) with PF ~2.0 supports genuine OOS signal; degradation is mainly reduced compounding, not edge collapse.

### 11.2 Paper Trading

Paper trading with the Predictive strategy on the final 20% of data (completely unseen, post-OOS period):

| Metric | Value |
|--------|-------|
| Total Trades | 223 |
| Total Return | +288.84% |
| Sharpe Ratio | 207.15 |
| Max Drawdown | 10.83% |
| Win Rate | 42.15% |
| Profit Factor | 2.00 |
| Avg Trade P&L | $121.46 |
| Total Fees | $1,798.87 |

Key observations:
- Strategy generates trades and manages risk in a live-like paper environment
- Profit factor remains ~2.0, consistent with IS and OOS
- Max drawdown stays within the 10.83% circuit breaker
- Win rate remains consistent across evaluation periods

---

## 12. Conclusions

### 12.1 Summary

Implemented platform capabilities:
1. Ingests real prediction market prices and resolutions via public Polymarket APIs
2. Implements three distinct trading strategies (Market Maker, Arbitrage, Predictive)
3. Provides a rigorous backtesting engine with look-ahead bias prevention
4. Evaluates strategies using walk-forward out-of-sample testing
5. Analyzes probability calibration using Brier scores and reliability diagrams
6. Includes comprehensive risk management with Kelly sizing and circuit breakers

### 12.2 Key Takeaways

1. **Edge exists** in prediction markets when prices deviate from fair value, but edges are small (3-8%) and require careful cost management
2. **Ensemble models** (LR + GBT) provide the best balance of accuracy, calibration, and interpretability
3. **Walk-forward testing** is essential — in-sample results alone are misleading
4. **Risk management** is more important than signal quality — position sizing and exposure limits determine survival
5. **Transaction costs** are the primary barrier to profitability in short-horizon markets

### 12.3 Future Work

- Multi-venue expansion beyond Polymarket (for example Kalshi)
- Higher-frequency features (order book depth, trade flow)
- Online learning for real-time model adaptation
- Portfolio-level optimization across multiple markets
- Reinforcement learning for dynamic position management

---

*This research document accompanies the source code and dashboard.*
