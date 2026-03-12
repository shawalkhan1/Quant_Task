"""
Logistic Regression Model — standalone version.

Provides a clean interface for logistic regression probability estimation,
used as a baseline and component of the ensemble.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LogisticModel:
    """
    Logistic Regression model for binary outcome prediction.

    P(Y=1|X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
    where σ(z) = 1 / (1 + e^{-z})

    Uses L2 regularization to prevent overfitting.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        """Train the logistic regression model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import brier_score_loss, accuracy_score

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs", random_state=42
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        if feature_names:
            self.feature_names = feature_names

        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs > 0.5).astype(int)

        return {
            "brier_score": brier_score_loss(y, probs),
            "accuracy": accuracy_score(y, preds),
            "n_samples": len(y),
            "coefficients": dict(zip(self.feature_names, self.model.coef_[0])) if self.feature_names else {},
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            return np.full(len(X), 0.5)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_coefficients(self) -> Optional[pd.DataFrame]:
        """Return model coefficients as a DataFrame."""
        if not self.is_trained or not self.feature_names:
            return None
        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_[0],
            "abs_coefficient": np.abs(self.model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)
