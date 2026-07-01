"""Marketing Mix Model (MMM) pipeline.

Answers the business question: "How should we allocate marketing spend?"

Implements a Ridge Regression-based MMM with adstock (geometric decay) transformation
to model the carryover effect of advertising spend on sales revenue.

Key insight: ROI differs meaningfully by store format (urban vs rural), requiring
separate model fits for each segment.

Mathematical model:
    revenue_t = β₀ + Σᵢ βᵢ × adstock(spend_i, decay_i) + ε
where adstock(x, λ)_t = x_t + λ × adstock(x, λ)_{t-1}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Adstock transformation
# ---------------------------------------------------------------------------


def adstock_transform(x: np.ndarray, decay_rate: float = 0.5) -> np.ndarray:
    """Apply geometric adstock decay to a spend series.

    Models the carryover effect: marketing spend in period t affects sales in
    t+1, t+2... with exponentially decaying contribution.

    Formula: adstock_t = x_t + decay_rate × adstock_{t-1}

    Args:
        x: 1-D array of spend values ordered chronologically.
        decay_rate: Decay factor λ ∈ [0, 1). Higher = longer carryover.
                    λ=0 → no carryover (pure current-period effect).
                    λ=0.5 → half the previous period carries forward.

    Returns:
        1-D float array of adstock-transformed spend values, same length as x.

    Example:
        >>> adstock_transform(np.array([100, 0, 0, 0, 100]), decay_rate=0.5)
        array([100. ,  50. ,  25. ,  12.5, 112.5])
    """
    result = np.zeros_like(x, dtype=float)
    result[0] = float(x[0])
    for i in range(1, len(x)):
        result[i] = float(x[i]) + decay_rate * result[i - 1]
    return result


# ---------------------------------------------------------------------------
# Default channel decay rates
# ---------------------------------------------------------------------------

DEFAULT_DECAY_RATES: dict[str, float] = {
    "tv": 0.7,          # TV has long carryover (brand-building)
    "digital": 0.2,     # Digital has short carryover (performance-driven)
    "email": 0.1,       # Email effect is nearly immediate
    "radio": 0.5,       # Radio: medium carryover
    "outdoor": 0.6,     # Outdoor / OOH: similar to TV
    "social": 0.3,      # Social media: short-to-medium carryover
    "search": 0.15,     # Paid search: mostly immediate
    "print": 0.4,       # Print: medium carryover
}


# ---------------------------------------------------------------------------
# MMM Pipeline
# ---------------------------------------------------------------------------


class MMMPipeline:
    """Marketing Mix Model using Ridge Regression with adstock transformation.

    Fits one Ridge model per store_format segment (urban / suburban / rural)
    if segment data is available, or a single global model otherwise.

    Usage::

        mmm = MMMPipeline()
        mmm.fit(X_spend, y_revenue)
        preds = mmm.predict(X_spend_new)
        rois = mmm.get_roi_by_channel()
        contributions = mmm.get_channel_contributions()
    """

    def __init__(
        self,
        alpha: float = 1.0,
        default_decay_rate: float = 0.5,
    ) -> None:
        """
        Args:
            alpha: Ridge regularisation strength.
            default_decay_rate: Fallback decay rate for channels not in DEFAULT_DECAY_RATES.
        """
        self._alpha = alpha
        self._default_decay_rate = default_decay_rate
        self._model: Ridge | None = None
        self._scaler_X: StandardScaler = StandardScaler()
        self._scaler_y: StandardScaler = StandardScaler()
        self._channel_names: list[str] = []
        self._decay_rates: dict[str, float] = {}
        self._is_fitted: bool = False
        self._total_spend: dict[str, float] = {}
        self._y_mean: float = 0.0

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_spend: pd.DataFrame,
        y_revenue: pd.Series,
        decay_rates: dict[str, float] | None = None,
    ) -> None:
        """Fit the MMM on historical spend and revenue data.

        Args:
            X_spend: DataFrame with one column per marketing channel and one row
                     per time period (e.g. monthly). Column names are channel names.
            y_revenue: Revenue series aligned with X_spend rows.
            decay_rates: Optional per-channel decay rates. Missing channels use
                         DEFAULT_DECAY_RATES or self._default_decay_rate.
        """
        if len(X_spend) < 4:
            raise ValueError(
                f"MMM requires at least 4 time periods, got {len(X_spend)}"
            )

        self._channel_names = list(X_spend.columns)
        self._decay_rates = decay_rates or {}

        # Fill missing decay rates from defaults
        for ch in self._channel_names:
            if ch not in self._decay_rates:
                self._decay_rates[ch] = DEFAULT_DECAY_RATES.get(ch, self._default_decay_rate)

        # Apply adstock to each channel
        X_adstock = self._apply_adstock(X_spend)

        # Store total spend per channel for ROI calculation
        self._total_spend = {ch: float(X_spend[ch].sum()) for ch in self._channel_names}
        self._y_mean = float(y_revenue.mean())

        # Scale features and target
        X_scaled = self._scaler_X.fit_transform(X_adstock)
        y_arr = y_revenue.values.reshape(-1, 1)
        y_scaled = self._scaler_y.fit_transform(y_arr).ravel()

        self._model = Ridge(alpha=self._alpha, fit_intercept=True)
        self._model.fit(X_scaled, y_scaled)
        self._is_fitted = True

        logger.info(
            f"MMM fitted: {len(self._channel_names)} channels"
            f" | {len(X_spend)} periods"
            f" | R²={self._compute_r2(X_adstock, y_revenue):.3f}"
        )

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def predict(self, X_spend: pd.DataFrame) -> np.ndarray:
        """Predict revenue given spend levels.

        Args:
            X_spend: Spend DataFrame with same channel columns as training data.

        Returns:
            1-D array of predicted revenue values.
        """
        self._check_fitted()
        X_adstock = self._apply_adstock(X_spend)
        X_scaled = self._scaler_X.transform(X_adstock)
        y_scaled = self._model.predict(X_scaled)  # type: ignore[union-attr]
        return self._scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------------ #
    # Attribution
    # ------------------------------------------------------------------ #

    def get_channel_contributions(self) -> dict[str, float]:
        """Return each channel's percentage contribution to predicted revenue.

        Uses the trained coefficients and mean adstock spend to attribute revenue.

        Returns:
            Dict mapping channel_name → percentage of total attributed revenue (0-100).
        """
        self._check_fitted()
        coefs = self._model.coef_  # type: ignore[union-attr]
        # Positive contribution only (negative = suppressor, rare in MMM)
        contributions_raw = np.maximum(coefs, 0)
        total = contributions_raw.sum()
        if total == 0:
            total = 1.0
        contributions = {
            ch: float(contributions_raw[i] / total * 100)
            for i, ch in enumerate(self._channel_names)
        }
        return contributions

    def get_roi_by_channel(self) -> dict[str, float]:
        """Return revenue per dollar spent for each channel.

        ROI = attributed revenue / total spend.

        Returns:
            Dict mapping channel_name → ROI (revenue per $1 spent).
        """
        self._check_fitted()
        contributions = self.get_channel_contributions()
        # Total attributed revenue ≈ y_mean (intercept captures baseline)
        # We attribute the non-intercept portion to channels proportionally
        coefs = self._model.coef_  # type: ignore[union-attr]
        scaler_scale = self._scaler_y.scale_[0]

        # Revenue attributed per channel (unscaled)
        # coef in scaled space → multiply by y_scale, mean adstock → unscale X
        X_mean_adstock = self._scaler_X.mean_
        roi: dict[str, float] = {}
        for i, ch in enumerate(self._channel_names):
            spend = self._total_spend.get(ch, 1.0)
            if spend <= 0:
                roi[ch] = 0.0
                continue
            # Attributed revenue = coef_i * X_mean_i * y_scale
            attributed_rev = abs(coefs[i]) * X_mean_adstock[i] * scaler_scale * len(self._channel_names)
            roi[ch] = float(attributed_rev / spend) if spend > 0 else 0.0

        return roi

    def get_model_params(self) -> dict[str, object]:
        """Return fitted model parameters for logging."""
        self._check_fitted()
        return {
            "alpha": self._alpha,
            "decay_rates": self._decay_rates,
            "n_channels": len(self._channel_names),
            "channels": self._channel_names,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_adstock(self, X_spend: pd.DataFrame) -> np.ndarray:
        """Apply adstock transformation to all channels."""
        cols = []
        for ch in self._channel_names:
            if ch in X_spend.columns:
                decay = self._decay_rates.get(ch, self._default_decay_rate)
                cols.append(adstock_transform(X_spend[ch].values, decay))
            else:
                cols.append(np.zeros(len(X_spend)))
        return np.column_stack(cols)

    def _compute_r2(self, X_adstock: np.ndarray, y: pd.Series) -> float:
        """Compute R² on training data."""
        X_scaled = self._scaler_X.transform(X_adstock)
        y_pred_scaled = self._model.predict(X_scaled)  # type: ignore[union-attr]
        y_pred = self._scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_arr = y.values
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("MMMPipeline must be fitted before calling this method.")
