"""
Bayesian Inference Model.

Alternative approach using Beta-Binomial model with Bayesian updating
for probability estimation.

This was one of the alternative approaches explored — documented
in the research as a rejected alternative due to slower adaptation
to rapid regime changes in 15-minute windows.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

logger = logging.getLogger(__name__)


class BayesianModel:
    """
    Beta-Binomial Bayesian model for prediction market probability estimation.

    Prior: Beta(α₀, β₀) — representing belief about resolution probability
    Likelihood: Binomial (market resolves YES or NO)
    Posterior: Beta(α₀ + successes, β₀ + failures)

    The model maintains a running posterior that updates with each observed
    market resolution, incorporating historical base rates.
    """

    def __init__(self, alpha_prior: float = 2.0, beta_prior: float = 2.0):
        """
        Initialize with Beta prior.

        Alpha=Beta=2 gives a weakly informative prior centered at 0.5.
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.observations = 0

    def update(self, outcome: int):
        """
        Bayesian update with a new observation.

        Parameters:
            outcome: 1 (YES resolved) or 0 (NO resolved)
        """
        if outcome == 1:
            self.alpha += 1
        else:
            self.beta += 1
        self.observations += 1

    def predict_probability(self) -> float:
        """
        Return posterior mean: E[p] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)

    def predict_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Return credible interval.
        """
        lower = beta_dist.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = beta_dist.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta)
        return (lower, upper)

    def uncertainty(self) -> float:
        """
        Return posterior standard deviation as a measure of uncertainty.
        """
        return beta_dist.std(self.alpha, self.beta)

    def reset(self):
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior
        self.observations = 0


class ContextualBayesianModel:
    """
    Enhanced Bayesian model that conditions on price features.

    Maintains separate Beta posteriors for different market regimes:
    - High volatility vs low volatility
    - Price above vs below SMA
    - Different times of day

    This provides more nuanced probability estimates than a single global prior.
    """

    def __init__(self):
        # Regime-specific priors
        self.models = {
            "high_vol": BayesianModel(2, 2),
            "low_vol": BayesianModel(2, 2),
            "trending_up": BayesianModel(3, 2),  # Slight bullish prior
            "trending_down": BayesianModel(2, 3),  # Slight bearish prior
            "neutral": BayesianModel(2, 2),
        }
        self.global_model = BayesianModel(2, 2)

    def get_regime(self, features: pd.Series) -> str:
        """Determine current market regime from features."""
        if features is None:
            return "neutral"

        vol = features.get("volatility_15m", 0.6)
        if isinstance(vol, (float, int)) and not np.isnan(vol):
            if vol > 0.8:
                return "high_vol"
            elif vol < 0.4:
                return "low_vol"

        ret = features.get("return_15m", 0)
        if isinstance(ret, (float, int)) and not np.isnan(ret):
            if ret > 0.005:
                return "trending_up"
            elif ret < -0.005:
                return "trending_down"

        return "neutral"

    def update(self, outcome: int, features: Optional[pd.Series] = None):
        """Update global and regime-specific models."""
        self.global_model.update(outcome)
        regime = self.get_regime(features)
        if regime in self.models:
            self.models[regime].update(outcome)

    def predict_probability(
        self,
        features: Optional[pd.Series] = None,
        global_weight: float = 0.3,
    ) -> float:
        """
        Weighted prediction combining global and regime-specific models.
        """
        global_prob = self.global_model.predict_probability()
        regime = self.get_regime(features)
        regime_prob = self.models[regime].predict_probability()

        return global_weight * global_prob + (1 - global_weight) * regime_prob

    def uncertainty(self, features: Optional[pd.Series] = None) -> float:
        """Combined uncertainty estimate."""
        regime = self.get_regime(features)
        return (
            0.3 * self.global_model.uncertainty()
            + 0.7 * self.models[regime].uncertainty()
        )

    def reset(self):
        self.global_model.reset()
        for model in self.models.values():
            model.reset()
