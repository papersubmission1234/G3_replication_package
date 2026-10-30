"""Full-information generalized posterior and decision-aligned losses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class GeneralizedPosterior:
    arm_names: tuple[str, ...]
    calibration_months: int = 24
    eta: float | None = None
    forgetting: float = 1.0
    eta_multiplier: float = 1.0
    initial_weights: Mapping[str, float] | None = None
    log_prior_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.arm_names:
            raise ValueError("At least one arm is required.")
        if self.calibration_months < 0:
            raise ValueError("calibration_months must be nonnegative.")
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1].")
        if self.eta_multiplier <= 0.0:
            raise ValueError("eta_multiplier must be positive.")
        if self.log_prior_scale < 0.0:
            raise ValueError("log_prior_scale must be nonnegative.")

        if self.initial_weights is None:
            prior = np.repeat(1.0 / len(self.arm_names), len(self.arm_names))
        else:
            raw = np.array(
                [max(float(self.initial_weights.get(name, 0.0)), 0.0) for name in self.arm_names],
                dtype=float,
            )
            if raw.sum() <= 0.0:
                raise ValueError("initial_weights must assign positive mass to at least one arm.")
            prior = raw / raw.sum()
        prior = np.maximum(prior, 1e-15)
        prior /= prior.sum()
        self._initial_weights = dict(zip(self.arm_names, prior.tolist()))
        self._log_weights = {
            name: float(self.log_prior_scale * np.log(self._initial_weights[name]))
            for name in self.arm_names
        }
        self._calibration_vectors: list[dict[str, float]] = []
        self._eta_effective = None if self.eta is None else float(self.eta)
        self.n_updates = 0

    @property
    def eta_effective(self) -> float | None:
        return self._eta_effective

    @property
    def calibrated(self) -> bool:
        return self._eta_effective is not None

    def weights(self) -> dict[str, float]:
        values = np.array([self._log_weights[name] for name in self.arm_names])
        values -= np.max(values)
        probs = np.exp(values)
        probs /= probs.sum()
        return dict(zip(self.arm_names, probs.tolist()))

    def entropy(self) -> float:
        p = np.array(list(self.weights().values()))
        return float(-np.sum(p * np.log(np.maximum(p, 1e-15))))

    def update(self, losses: Mapping[str, float]) -> dict[str, float]:
        clean = {
            name: float(losses.get(name, np.nan))
            for name in self.arm_names
        }
        if self._eta_effective is None:
            self._calibration_vectors.append(clean)
            if len(self._calibration_vectors) >= self.calibration_months:
                pooled = np.array(
                    [
                        value
                        for vector in self._calibration_vectors
                        for value in vector.values()
                        if np.isfinite(value)
                    ],
                    dtype=float,
                )
                scale = float(np.std(pooled, ddof=1)) if pooled.size > 1 else np.nan
                self._eta_effective = (
                    self.eta_multiplier / scale
                    if np.isfinite(scale) and scale > 1e-12
                    else self.eta_multiplier
                )
            self.n_updates += 1
            return self.weights()

        eta = float(self._eta_effective)
        for name in self.arm_names:
            self._log_weights[name] *= self.forgetting
            value = clean[name]
            if np.isfinite(value):
                self._log_weights[name] -= eta * value
        self.n_updates += 1
        return self.weights()

    def state(self) -> dict[str, object]:
        return {
            "arm_names": list(self.arm_names),
            "initial_weights": dict(self._initial_weights),
            "log_prior_scale": float(self.log_prior_scale),
            "weights": self.weights(),
            "log_weights": dict(self._log_weights),
            "eta_effective": self._eta_effective,
            "n_updates": self.n_updates,
            "n_calibration_vectors": len(self._calibration_vectors),
            "entropy": self.entropy(),
        }


def diagonal_predictive_log_loss(
    scenarios: pd.DataFrame,
    realized_excess_returns: pd.Series,
    *,
    min_assets: int = 5,
) -> float:
    """Negative normalized diagonal Gaussian log score of action scenarios."""

    mean = scenarios.mean(axis=0)
    var = scenarios.var(axis=0, ddof=1).clip(lower=1e-10)
    realized = realized_excess_returns.reindex(scenarios.columns)
    ok = realized.notna() & mean.notna() & var.notna() & (var > 0.0)
    if int(ok.sum()) < min_assets:
        return np.nan
    r = realized.loc[ok].to_numpy(dtype=float)
    mu = mean.loc[ok].to_numpy(dtype=float)
    v = var.loc[ok].to_numpy(dtype=float)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * v) + (r - mu) ** 2 / v))


def variance_only_loss(
    scenarios: pd.DataFrame,
    realized_excess_returns: pd.Series,
    *,
    min_assets: int = 5,
) -> float:
    var = scenarios.var(axis=0, ddof=1).clip(lower=1e-10)
    realized = realized_excess_returns.reindex(scenarios.columns)
    ok = realized.notna() & var.notna() & (var > 0.0)
    if int(ok.sum()) < min_assets:
        return np.nan
    r = realized.loc[ok].to_numpy(dtype=float)
    v = var.loc[ok].to_numpy(dtype=float)
    centered = r - r.mean()
    return float(0.5 * np.mean(np.log(v) + centered ** 2 / v))


def quadratic_utility_loss(net_excess_return: float, gamma: float = 1.0) -> float:
    value = float(net_excess_return)
    return float(-(value - 0.5 * gamma * value * value))


def cvar_proxy_loss(
    net_excess_return: float,
    historical_net_excess_returns: pd.Series | list[float],
    *,
    alpha: float = 0.95,
    initial_var_loss: float = 0.02,
) -> float:
    history = np.asarray(historical_net_excess_returns, dtype=float)
    history = history[np.isfinite(history)]
    var_loss = (
        float(-np.quantile(history, 1.0 - alpha))
        if history.size >= 10
        else float(initial_var_loss)
    )
    p = float(net_excess_return)
    return float(-p + (1.0 / (1.0 - alpha)) * max(0.0, -p - max(var_loss, 0.0)))
