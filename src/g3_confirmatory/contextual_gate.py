"""Bayesian linear gate for a baseline and a signal expert.

Sequential regression estimates the baseline's realized utility advantage.
Its probability of a positive advantage supplies the baseline mixture weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BayesianLinearGate:
    n_features: int
    prior_variance: float = 10.0
    observation_variance: float = 0.01
    minimum_weight: float = 0.05
    maximum_weight: float = 0.95

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")
        if self.prior_variance <= 0.0 or self.observation_variance <= 0.0:
            raise ValueError("Prior and observation variances must be positive.")
        if not 0.0 <= self.minimum_weight < self.maximum_weight <= 1.0:
            raise ValueError("Gate weight bounds must satisfy 0 <= min < max <= 1.")
        self.mean = np.zeros(self.n_features, dtype=float)
        self.covariance = self.prior_variance * np.eye(self.n_features)
        self.n_updates = 0

    def predict(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.size}.")
        mean = float(x @ self.mean)
        variance = float(x @ self.covariance @ x + self.observation_variance)
        sd = float(np.sqrt(max(variance, 1e-15)))
        probability_baseline_better = float(norm.cdf(mean / sd))
        weight = float(
            np.clip(
                probability_baseline_better,
                self.minimum_weight,
                self.maximum_weight,
            )
        )
        return {
            "predictive_advantage_mean": mean,
            "predictive_advantage_sd": sd,
            "probability_baseline_better": probability_baseline_better,
            "baseline_weight": weight,
            "signal_weight": 1.0 - weight,
            "n_updates": int(self.n_updates),
        }

    def update(self, x: np.ndarray, observed_advantage: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.size}.")
        y = float(observed_advantage)
        if not np.isfinite(y):
            return
        v = self.covariance
        noise = float(self.observation_variance)
        denom = float(noise + x @ v @ x)
        gain = (v @ x) / max(denom, 1e-15)
        prediction_error = y - float(x @ self.mean)
        self.mean = self.mean + gain * prediction_error
        self.covariance = v - np.outer(gain, x) @ v
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.n_updates += 1

    def state(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "n_updates": int(self.n_updates),
            "prior_variance": float(self.prior_variance),
            "observation_variance": float(self.observation_variance),
        }


def standardized_context_features(
    macro_factors: pd.DataFrame,
    formation_date: pd.Timestamp,
    columns: Sequence[str],
    *,
    lag_months: int = 1,
    minimum_history: int = 12,
    include_intercept: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Standardize each observation using strictly earlier source dates."""

    frame = macro_factors.copy()
    frame.index = pd.to_datetime(frame.index).to_period("M").to_timestamp("M")
    frame = frame.sort_index()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Contextual gate columns are missing: {missing}")
    formation = pd.Timestamp(formation_date).to_period("M").to_timestamp("M")
    source_date = formation - pd.offsets.MonthEnd(int(lag_months))
    current = pd.to_numeric(frame.loc[source_date, list(columns)], errors="coerce")
    if current.isna().any():
        raise ValueError(
            f"Missing contextual features at source date {source_date.date()}: "
            f"{current[current.isna()].index.tolist()}"
        )
    history = frame.loc[frame.index < source_date, list(columns)].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(history) < minimum_history:
        z = np.zeros(len(columns), dtype=float)
        mean = pd.Series(0.0, index=columns)
        sd = pd.Series(1.0, index=columns)
        standardized = False
    else:
        mean = history.mean(axis=0)
        sd = history.std(axis=0, ddof=1).replace(0.0, 1.0).fillna(1.0)
        z = ((current - mean) / sd).to_numpy(dtype=float)
        standardized = True
    x = np.concatenate([[1.0], z]) if include_intercept else z
    return x, {
        "formation_date": str(formation.date()),
        "source_date": str(source_date.date()),
        "columns": list(columns),
        "raw_values": current.to_dict(),
        "historical_mean": mean.to_dict(),
        "historical_sd": sd.to_dict(),
        "history_count": int(len(history)),
        "standardized": bool(standardized),
        "include_intercept": bool(include_intercept),
    }
