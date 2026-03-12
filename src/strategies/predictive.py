"""
Predictive Strategy — ML-based probability estimation.

Uses an ensemble of Logistic Regression and Gradient Boosted Trees
to predict the probability of a binary market outcome, then trades
when the predicted probability diverges from market prices.

Mathematical Formulation:
    P(Y=1|X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)  [Logistic Regression]

    P_ensemble = α * P_logistic + (1-α) * P_gbt   [Ensemble]

    Edge = P_model - P_market  (for YES direction)
    Trade when |Edge| > min_edge

    Position size via fractional Kelly:
        f* = kelly_frac * (p*b - q) / b
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from src.strategies.base import BaseStrategy
from src.strategies.risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class PredictiveStrategy(BaseStrategy):
    """
    ML-based predictive strategy for prediction markets.

    Uses Logistic Regression + Gradient Boosted Trees ensemble
    to estimate event probabilities and trade mispricings.
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        ensemble_alpha: float = 0.5,
        retrain_interval: int = 1440,
        lookback_window: int = 4320,
        model_type: str = "ensemble",
        risk_config=None,
    ):
        super().__init__(name="Predictive", risk_config=risk_config)
        self.min_edge = min_edge
        self.ensemble_alpha = ensemble_alpha
        self.retrain_interval = retrain_interval
        self.lookback_window = lookback_window
        self.model_type = model_type

        # Models
        self.logistic_model = None
        self.gbt_model = None
        self.is_trained = False
        self._bars_since_train = 0
        self._feature_columns: List[str] = []

        # Risk
        self.risk_mgr = RiskManager(RiskLimits(
            max_position_pct=0.05,
            max_total_exposure_pct=0.30,
            min_edge_to_trade=min_edge,
            kelly_fraction=0.25,
        ))

    def train(
        self,
        features_df: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> dict:
        """
        Train the predictive models on historical data.

        Parameters:
            features_df: DataFrame with feature values
            market_data: DataFrame with 'resolution' column (0 or 1)
            feature_columns: List of column names to use as features

        Returns:
            Training metrics dict
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

        # Prepare training data
        X, y = self._prepare_training_data(features_df, market_data, feature_columns)
        if X is None or len(X) < 100:
            logger.warning(f"Insufficient training data: {len(X) if X is not None else 0} samples")
            return {"status": "insufficient_data"}

        logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train Logistic Regression
        self.logistic_model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        )
        self.logistic_model.fit(X_scaled, y)

        # Train Gradient Boosted Trees
        self.gbt_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.gbt_model.fit(X_scaled, y)

        # Training metrics
        lr_probs = self.logistic_model.predict_proba(X_scaled)[:, 1]
        gbt_probs = self.gbt_model.predict_proba(X_scaled)[:, 1]
        ensemble_probs = self.ensemble_alpha * lr_probs + (1 - self.ensemble_alpha) * gbt_probs

        metrics = {
            "status": "trained",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "lr_brier": brier_score_loss(y, lr_probs),
            "gbt_brier": brier_score_loss(y, gbt_probs),
            "ensemble_brier": brier_score_loss(y, ensemble_probs),
            "lr_accuracy": accuracy_score(y, (lr_probs > 0.5).astype(int)),
            "gbt_accuracy": accuracy_score(y, (gbt_probs > 0.5).astype(int)),
            "ensemble_accuracy": accuracy_score(y, (ensemble_probs > 0.5).astype(int)),
            "positive_rate": y.mean(),
        }

        self.is_trained = True
        self._bars_since_train = 0
        logger.info(
            f"Models trained — Brier scores: LR={metrics['lr_brier']:.4f}, "
            f"GBT={metrics['gbt_brier']:.4f}, Ensemble={metrics['ensemble_brier']:.4f}"
        )
        return metrics

    def predict_probability(self, features: pd.Series) -> float:
        """
        Predict the probability of YES outcome.

        Returns ensemble prediction: α * LR + (1-α) * GBT
        """
        if not self.is_trained:
            return 0.5  # Default when untrained

        try:
            # Extract features
            X = self._extract_features(features)
            if X is None:
                return 0.5

            X_scaled = self._scaler.transform(X.reshape(1, -1))

            if self.model_type == "logistic":
                return float(self.logistic_model.predict_proba(X_scaled)[0, 1])
            elif self.model_type == "gbt":
                return float(self.gbt_model.predict_proba(X_scaled)[0, 1])
            else:  # ensemble
                lr_prob = self.logistic_model.predict_proba(X_scaled)[0, 1]
                gbt_prob = self.gbt_model.predict_proba(X_scaled)[0, 1]
                return float(self.ensemble_alpha * lr_prob + (1 - self.ensemble_alpha) * gbt_prob)
        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return 0.5

    def generate_signal(
        self,
        market_row: pd.Series,
        features: Optional[pd.Series],
        timestamp: datetime,
        portfolio_value: float,
        open_positions: Dict[str, list],
    ) -> Optional[dict]:
        """
        Generate predictive trading signal.

        1. Predict probability using ML ensemble
        2. Compare to market price
        3. Trade if edge exceeds minimum threshold
        """
        if not self.is_trained or features is None:
            return None

        market_id = market_row.get("market_id", "")
        time_to_expiry = market_row.get("time_to_expiry_min", 0)
        market_price_yes = market_row.get("market_price_yes", 0.5)
        market_price_no = market_row.get("market_price_no", 0.5)

        # Don't trade too close to expiry
        if time_to_expiry <= 2 or time_to_expiry > 13:
            return None

        # Skip if already positioned
        if self._already_in_market(market_id, open_positions):
            return None

        # Predict probability
        predicted_prob = self.predict_probability(features)
        self._bars_since_train += 1

        # Compute edge
        yes_edge = predicted_prob - market_price_yes
        no_edge = (1.0 - predicted_prob) - market_price_no

        direction = None
        edge = 0.0

        if yes_edge > self.min_edge:
            direction = "YES"
            edge = yes_edge
        elif no_edge > self.min_edge:
            direction = "NO"
            edge = no_edge
        else:
            return None

        # Position sizing
        if direction == "YES":
            size = self.risk_mgr.compute_kelly_size(
                predicted_prob=predicted_prob,
                market_price=market_price_yes,
                portfolio_value=portfolio_value,
                direction="YES",
            )
        else:
            size = self.risk_mgr.compute_kelly_size(
                predicted_prob=1.0 - predicted_prob,
                market_price=market_price_no,
                portfolio_value=portfolio_value,
                direction="NO",
            )

        if size < 10.0:
            return None

        self._trade_count += 1
        return {
            "action": "trade",
            "direction": direction,
            "size": size,
            "predicted_prob": predicted_prob,
            "confidence": min(abs(edge) / 0.15, 1.0),
            "edge": edge,
            "reason": (
                f"PRED: p_model={predicted_prob:.4f}, "
                f"p_mkt_yes={market_price_yes:.4f}, edge={edge:.4f}"
            ),
        }

    def _prepare_training_data(
        self,
        features_df: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare aligned X, y arrays for model training."""
        if feature_columns is None:
            # Auto-detect feature columns
            exclude = {
                "open", "high", "low", "close", "volume",
                "market_id", "symbol", "resolution", "strike",
                "fair_price", "market_price_yes", "market_price_no",
                "time_to_expiry_min", "minutes_elapsed", "close_price",
                "implied_vol",
            }
            feature_columns = [c for c in features_df.columns if c not in exclude]

        self._feature_columns = feature_columns

        # Only use rows where we have market resolution data
        # And only one sample per market (at the signal generation point)
        if "market_id" not in market_data.columns:
            return None, None

        # Use observations at specific minutes_elapsed (e.g., minute 3-5)
        # to have enough time info but not too close to expiry
        if "minutes_elapsed" in market_data.columns:
            signal_data = market_data[
                (market_data["minutes_elapsed"] >= 3) &
                (market_data["minutes_elapsed"] <= 5)
            ].copy()
        else:
            signal_data = market_data.copy()

        if len(signal_data) == 0:
            return None, None

        # Get corresponding features
        common_idx = signal_data.index.intersection(features_df.index)
        if len(common_idx) < 50:
            return None, None

        X_df = features_df.loc[common_idx, feature_columns].copy()
        y_series = signal_data.loc[common_idx, "resolution"]

        # Drop rows with NaN
        valid_mask = X_df.notna().all(axis=1) & y_series.notna()
        X_df = X_df[valid_mask]
        y_series = y_series[valid_mask]

        if len(X_df) < 50:
            return None, None

        return X_df.values, y_series.values.astype(int)

    def _extract_features(self, features: pd.Series) -> Optional[np.ndarray]:
        """Extract feature vector from a feature row."""
        if len(self._feature_columns) == 0:
            return None
        try:
            values = []
            for col in self._feature_columns:
                val = features.get(col, np.nan) if hasattr(features, "get") else np.nan
                if pd.isna(val):
                    val = 0.0
                values.append(val)
            return np.array(values, dtype=float)
        except Exception:
            return None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the GBT model."""
        if self.gbt_model is None or not self._feature_columns:
            return None
        importance = self.gbt_model.feature_importances_
        df = pd.DataFrame({
            "feature": self._feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return df

    def reset(self):
        super().reset()
        self.logistic_model = None
        self.gbt_model = None
        self.is_trained = False
        self._bars_since_train = 0
        self.risk_mgr.reset()
