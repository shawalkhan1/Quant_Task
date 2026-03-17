# Prediction Market Trading System

A complete research and trading platform for short-horizon crypto prediction markets, inspired by Polymarket's 15-minute markets.

## Required vs Built

Required items were: live market data ingestion, 3 strategies, event-driven backtesting, forward testing, calibration analysis, dashboard, reproducible outputs, tests, and Docker runtime.

Built system covers all of these with live Polymarket APIs as the runtime market source.

For a precise requirement-to-implementation map, see [DELIVERABLE_STATUS.md](DELIVERABLE_STATUS.md).

## Current Working State

- Runtime data path is live Polymarket (Gamma, CLOB, Data API).
- Frontend pages load through [src/data/live_market_loader.py](src/data/live_market_loader.py).
- Results generation runs through [generate_results.py](generate_results.py) and writes outputs to [results](results).
- Automated tests are passing.
- Docker web service runs with health checks; one-shot jobs service runs without health checks to avoid false unhealthy flags.

## Features

- **Data Layer** — Fetches live prediction market data from Polymarket public APIs (Gamma, CLOB, Data API), with local caching
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
│   │   ├── polymarket_fetcher.py # Live Polymarket market/timeseries fetcher
│   │   ├── live_market_loader.py # Shared loader used by frontend pages
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

## Docker

This project includes a production-style Docker setup for both the Streamlit UI
and batch result generation.

### Included files

- `Dockerfile` — single runtime image (web + jobs)
- `docker-compose.yml` — orchestrates `web` and `jobs` services
- `.env.example` — runtime configuration template
- `docker/entrypoint.sh` — startup DNS/HTTP checks for Polymarket endpoints
- `docker/healthcheck.py` — container health probe for Streamlit
- `docker/smoke_test.py` — in-container live-data smoke test

### 1. Prepare environment

```bash
cp .env.example .env
```

Optional: set `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` in `.env` if your
network requires a proxy.

### 2. Build image

```bash
docker compose build
```

### 3. Run dashboard container

```bash
docker compose up -d web
```

App will be available at `http://localhost:8501`.

### 4. Run batch job container (generate results)

```bash
docker compose run --rm --profile jobs jobs
```

### 5. Run container smoke test

```bash
docker compose run --rm web python -m docker.smoke_test
```

### 6. Logs and shutdown

```bash
docker compose logs -f web
docker compose down
```

### Notes

- `./data` and `./results` are mounted into the container, so cache and result
  files persist across restarts.
- The image runs as a non-root user.
- Startup checks validate DNS and HTTPS connectivity to Polymarket APIs.
- To bypass startup checks temporarily, set `SKIP_STARTUP_CHECKS=true` in `.env`.

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
Quotes dynamic bid/ask spreads around observed market-implied fair value. Spread widens with volatility; inventory skew prevents accumulation.

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
- **requests** — Polymarket public API integration
- **Plotly** — Interactive charts
- **Streamlit** — Dashboard framework
- **pytest** — Testing

## Data Access Note

Polymarket market data is publicly accessible via no-auth endpoints.
Reference: https://docs.polymarket.com/api-reference/introduction
