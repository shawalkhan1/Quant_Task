"""
Probability Calibration Tools.

Evaluates whether predicted probabilities are well-calibrated
(i.e., when the model says 70%, the event occurs ~70% of the time).

Methods:
- Reliability diagrams
- Brier score decomposition
- Calibration curves
- Expected Calibration Error (ECE)
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """
    Analyzes probability calibration of predictive models.
    """

    @staticmethod
    def reliability_diagram(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compute reliability diagram data.

        For each bin of predicted probabilities, compute the actual
        fraction of positive outcomes.

        Returns DataFrame with columns:
            bin_center, mean_predicted, fraction_positive, count
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        records = []
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:  # Include right edge
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

            count = mask.sum()
            if count > 0:
                mean_pred = y_prob[mask].mean()
                frac_pos = y_true[mask].mean()
            else:
                mean_pred = bin_centers[i]
                frac_pos = np.nan

            records.append({
                "bin_center": bin_centers[i],
                "mean_predicted": mean_pred,
                "fraction_positive": frac_pos,
                "count": int(count),
            })

        return pd.DataFrame(records)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Compute Brier score: mean((y_prob - y_true)^2)

        Lower is better. Perfect = 0, baseline (0.5) = 0.25
        """
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def brier_decomposition(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Decompose Brier score into reliability, resolution, and uncertainty.

        BS = Reliability - Resolution + Uncertainty

        - Reliability: How well calibrated the forecasts are (lower = better)
        - Resolution: How much the forecasts differ from climatology (higher = better)
        - Uncertainty: Inherent uncertainty in the outcomes
        """
        bins = np.linspace(0, 1, n_bins + 1)
        n = len(y_true)
        base_rate = y_true.mean()

        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

            n_k = mask.sum()
            if n_k == 0:
                continue

            mean_pred = y_prob[mask].mean()
            frac_pos = y_true[mask].mean()

            reliability += n_k * (mean_pred - frac_pos) ** 2
            resolution += n_k * (frac_pos - base_rate) ** 2

        reliability /= n
        resolution /= n
        uncertainty = base_rate * (1 - base_rate)

        return {
            "brier_score": float(np.mean((y_prob - y_true) ** 2)),
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "skill_score": 1 - np.mean((y_prob - y_true) ** 2) / uncertainty if uncertainty > 0 else 0,
        }

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE).

        ECE = Σ (n_k / N) * |mean_pred_k - frac_pos_k|
        """
        bins = np.linspace(0, 1, n_bins + 1)
        n = len(y_true)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

            n_k = mask.sum()
            if n_k == 0:
                continue

            mean_pred = y_prob[mask].mean()
            frac_pos = y_true[mask].mean()
            ece += (n_k / n) * abs(mean_pred - frac_pos)

        return float(ece)

    @staticmethod
    def calibrate_probabilities(
        y_true_cal: np.ndarray,
        y_prob_cal: np.ndarray,
        y_prob_test: np.ndarray,
        method: str = "isotonic",
    ) -> np.ndarray:
        """
        Post-hoc calibration using isotonic regression or Platt scaling.
        """
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(y_prob_cal, y_true_cal)
            return iso.predict(y_prob_test)
        elif method == "platt":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(y_prob_cal.reshape(-1, 1), y_true_cal)
            return lr.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]
        else:
            raise ValueError(f"Unknown calibration method: {method}")
