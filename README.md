# Prediction Market Trading System

A complete research and trading platform for short-horizon crypto prediction markets, inspired by Polymarket's 15-minute markets.

## Features

- **Data Layer** — Fetches crypto OHLCV data (Binance / synthetic fallback), simulates binary prediction markets using Black-Scholes digital option pricing
- **Backtesting Engine** — Event-driven, bar-by-bar execution with look-ahead bias prevention, transaction costs, slippage, and a drawdown circuit breaker
- **3 Trading Strategies**
  - *Market Maker* — Dynamic bid/ask spread with inventory control
  - *Arbitrage* — Detects YES/NO imbalance and fair-value deviations
  - *Predictive* — Logistic Regression + Gradient Boosted Trees ensemble with Kelly sizing
- **Walk-Forward Testing** — Rolling train/test splits for out-of-sample evaluation
- **Probability Calibration** — Brier score, ECE, reliability diagrams, post-hoc calibration
- **Risk Management** — Fractional Kelly criterion, position limits, exposure caps, drawdown breaker
- **Interactive Dashboard** — 5-page Streamlit app with Plotly charts

## Project Structure

```
├── config/
│   └── settings.py              # Global configuration
├── src/
│   ├── data/
│   │   ├── fetcher.py           # Crypto data fetcher (Binance + synthetic)
│   │   ├── market_simulator.py  # Binary market generator (Black-Scholes)
│   │   ├── features.py          # 35+ feature engineering pipeline
│   │   └── dataset.py           # Time-series dataset with temporal splits
│   ├── backtesting/
│   │   ├── engine.py            # Event-driven backtest engine
│   │   ├── position.py          # Position & portfolio manager
│   │   ├── trade_log.py         # Trade record logging
│   │   └── metrics.py           # Performance metrics (Sharpe, drawdown, etc.)
│   ├── strategies/
│   │   ├── base.py              # Abstract strategy interface
│   │   ├── risk_manager.py      # Pre-trade risk checks + Kelly sizing
│   │   ├── market_maker.py      # Market-making strategy
│   │   ├── arbitrage.py         # Arbitrage detection strategy
│   │   └── predictive.py        # ML ensemble strategy
│   ├── models/
│   │   ├── bayesian_model.py    # Beta-Binomial Bayesian model (rejected alt.)
│   │   ├── logistic_model.py    # Logistic Regression wrapper
│   │   └── calibration.py       # Probability calibration tools
│   ├── forward_testing/
│   │   ├── paper_trader.py      # Paper trading simulator
│   │   └── rolling_simulator.py # Walk-forward analysis
│   └── visualization/
│       └── charts.py            # Plotly chart functions
├── frontend/
│   ├── app.py                   # Streamlit main entry
│   └── pages/
│       ├── 01_data_explorer.py
│       ├── 02_backtesting.py
│       ├── 03_forward_testing.py
│       ├── 04_strategy_comparison.py
│       └── 05_research_analysis.py
├── research/
│   └── RESEARCH_DOCUMENT.md     # Full research write-up
├── tests/
│   ├── test_data.py
│   ├── test_backtesting.py
│   └── test_strategies.py
├── results/                     # All generated result files
├── generate_results.py          # Reproduces all results from scratch
├── PROJECT_REPORT.md            # Comprehensive project report
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run frontend/app.py
```

The app opens in your browser. Use the sidebar to navigate between pages:

| Page | Description |
|------|-------------|
| **Overview** | System summary and key settings |
| **Data Explorer** | Candlestick charts, market visualization, feature statistics |
| **Backtesting** | Run any strategy, view equity curves, trade logs, metrics |
| **Forward Testing** | Walk-forward analysis and paper trading |
| **Strategy Comparison** | Side-by-side comparison of all 3 strategies |
| **Research Analysis** | Calibration, feature importance, regime analysis, model comparison |

### 3. Run tests

```bash
pytest tests/ -v
```

## Configuration

All key parameters are in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL` | $10,000 | Starting capital |
| `MARKET_DURATION_MINUTES` | 15 | Market lifetime |
| `FEE_RATE` | 1% | Transaction fee per trade |
| `MAX_POSITION_PCT` | 5% | Max single position as % of capital |
| `MAX_EXPOSURE_PCT` | 30% | Max total exposure |
| `MAX_DRAWDOWN_PCT` | 10% | Circuit breaker threshold |
| `KELLY_FRACTION` | 0.25 | Fractional Kelly multiplier |

## Strategies

### Market Maker
Quotes dynamic bid/ask spreads around Black-Scholes fair value. Spread widens with volatility; inventory skew prevents accumulation.

### Arbitrage
Detects two types of inefficiency: (1) YES + NO prices sum to less than 1, (2) market price deviates significantly from fair value.

### Predictive (ML Ensemble)
Trains Logistic Regression + Gradient Boosted Trees on 35+ features. Ensemble combines both predictions. Trades when estimated edge exceeds 5%.

## Research Document

See [`research/RESEARCH_DOCUMENT.md`](research/RESEARCH_DOCUMENT.md) for the full write-up covering:
- Mathematical model formulations
- Strategy development process (with 2 rejected alternatives)
- Experimental results (in-sample and out-of-sample)
- Probability calibration analysis
- Robustness testing across market regimes
- Risk management framework
- Strategy failure modes

## Tech Stack

- **Python 3.11+**
- **pandas / numpy / scipy** — Data processing & statistics
- **scikit-learn** — ML models
- **ccxt** — Exchange data (with synthetic fallback)
- **Plotly** — Interactive charts
- **Streamlit** — Dashboard framework
- **pytest** — Testing
