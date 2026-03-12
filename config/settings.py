# Global configuration for the Prediction Market Trading Platform

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MARKET_DATA_DIR = os.path.join(DATA_DIR, "markets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "backtest_results")
FORWARD_RESULTS_DIR = os.path.join(RESULTS_DIR, "forward_test_results")
TRADE_LOGS_DIR = os.path.join(RESULTS_DIR, "trade_logs")

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MARKET_DATA_DIR,
          BACKTEST_RESULTS_DIR, FORWARD_RESULTS_DIR, TRADE_LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Data Settings
# ──────────────────────────────────────────────
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
DEFAULT_TIMEFRAME = "1m"           # 1-minute candles
MARKET_DURATION_MINUTES = 15       # 15-minute prediction markets
FETCH_DAYS = 30                    # Days of historical data to fetch
EXCHANGE_ID = "binance"

# ──────────────────────────────────────────────
# Prediction Market Settings
# ──────────────────────────────────────────────
MARKET_NOISE_STD = 0.05            # σ_noise for market price simulation
MARKET_FEE_RATE = 0.02             # 2% round-trip transaction cost
MIN_MARKET_PRICE = 0.01            # Floor for market prices
MAX_MARKET_PRICE = 0.99            # Ceiling for market prices
RISK_FREE_RATE = 0.05              # Annualized risk-free rate

# ──────────────────────────────────────────────
# Backtesting Settings
# ──────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0         # Starting capital in USD
TRANSACTION_COST_PCT = 0.01        # 1% per trade (each way)
SLIPPAGE_PCT = 0.005               # 0.5% slippage

# ──────────────────────────────────────────────
# Risk Management Defaults
# ──────────────────────────────────────────────
@dataclass
class RiskConfig:
    max_position_pct: float = 0.05         # Max 5% of capital per position
    max_total_exposure_pct: float = 0.30   # Max 30% total exposure
    max_drawdown_pct: float = 0.10         # 10% drawdown circuit breaker
    kelly_fraction: float = 0.25           # Quarter-Kelly for safety
    stop_loss_pct: float = 0.03            # 3% stop loss per position

DEFAULT_RISK_CONFIG = RiskConfig()

# ──────────────────────────────────────────────
# Strategy Settings
# ──────────────────────────────────────────────
@dataclass
class MarketMakerConfig:
    base_spread: float = 0.03
    vol_sensitivity: float = 0.5
    inventory_aversion: float = 0.1
    max_inventory: int = 10

@dataclass
class ArbitrageConfig:
    threshold: float = 0.02
    fee_buffer: float = 0.01
    max_position_size: float = 500.0

@dataclass
class PredictiveConfig:
    min_edge: float = 0.05          # Minimum edge to trade
    model_type: str = "ensemble"     # logistic, gbt, ensemble
    retrain_interval: int = 1440     # Retrain every 1440 bars (1 day of 1-min)
    lookback_window: int = 4320      # 3 days of 1-min bars for training

# ──────────────────────────────────────────────
# Forward Testing Settings
# ──────────────────────────────────────────────
WALK_FORWARD_TRAIN_DAYS = 7        # Train on 7 days
WALK_FORWARD_TEST_DAYS = 2         # Test on 2 days
WALK_FORWARD_STEP_DAYS = 1         # Roll forward by 1 day

# ──────────────────────────────────────────────
# Visualization Settings
# ──────────────────────────────────────────────
CHART_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 500
CHART_COLORS = {
    "profit": "#00d4aa",
    "loss": "#ff6b6b",
    "neutral": "#808080",
    "primary": "#00b4d8",
    "secondary": "#e77f67",
    "accent": "#ffd166",
}
