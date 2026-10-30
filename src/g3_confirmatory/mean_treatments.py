"""Mean-only treatments of the same hierarchical scenarios.

Each treatment centers the asset columns and adds its target mean. Sample
covariance is unchanged up to floating-point precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .arms import ArmResult, shift_scenario_mean


MEAN_TREATMENTS: tuple[str, ...] = (
    "base_empirical",
    "zero_centered_factor_innovations",
    "posterior_ar1_mean",
    "sampled_factor_mean",
    "literal_zero_total_mean",
)


@dataclass(frozen=True)
class MeanTreatmentResult:
    scenarios: pd.DataFrame
    target_mean: pd.Series
    diagnostics: dict[str, Any]


def _ar1_forecast_moments(history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Match the model's no-intercept AR(1) forecast calculation."""

    means: dict[str, float] = {}
    shock_sd: dict[str, float] = {}
    for column in history.columns:
        series = history[column].dropna().astype(float)
        if len(series) < 3:
            means[column] = 0.0
            shock_sd[column] = 1.0
            continue
        x_prev = series.iloc[:-1].to_numpy(dtype=float)
        x_next = series.iloc[1:].to_numpy(dtype=float)
        denom = float(np.dot(x_prev, x_prev))
        phi = 0.0 if denom <= 1e-10 else float(np.dot(x_prev, x_next) / denom)
        phi = float(np.clip(phi, -0.99, 0.99))
        resid = x_next - phi * x_prev
        means[column] = phi * float(series.iloc[-1])
        shock_sd[column] = max(float(np.std(resid, ddof=0)), 0.25)
    return pd.Series(means, dtype=float), pd.Series(shock_sd, dtype=float)


def _target_mean_from_factor_vector(
    model_result: dict[str, Any],
    factor_vectors_by_class: dict[str, pd.Series],
    assets: pd.Index,
) -> pd.Series:
    loadings = model_result["factor_loadings"].reindex(index=assets)
    summary = model_result["asset_summary"].reindex(index=assets)
    target = pd.Series(index=assets, dtype=float)
    for asset in assets:
        row = loadings.loc[asset]
        class_name = str(summary.loc[asset, "asset_class"])
        vector = factor_vectors_by_class.get(class_name, pd.Series(dtype=float))
        value = float(row.get("alpha", 0.0))
        shared = row.index.intersection(vector.index)
        if len(shared):
            beta = row.reindex(shared).fillna(0.0).to_numpy(dtype=float)
            x = vector.reindex(shared).fillna(0.0).to_numpy(dtype=float)
            value += float(beta @ x)
        target.loc[asset] = value
    return target


def hierarchical_target_mean(
    arm_result: ArmResult,
    treatment: str,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Return the ablation target mean for one fitted hierarchical arm."""

    treatment = str(treatment)
    if treatment not in MEAN_TREATMENTS:
        raise ValueError(
            f"Unknown hierarchical mean treatment {treatment!r}; expected one of "
            f"{MEAN_TREATMENTS}."
        )
    scenarios = arm_result.scenarios
    assets = scenarios.columns
    if treatment == "base_empirical":
        return scenarios.mean(axis=0), {"mean_treatment": treatment}
    if treatment == "literal_zero_total_mean":
        return pd.Series(0.0, index=assets), {"mean_treatment": treatment}

    if arm_result.model_result is None:
        raise ValueError("Hierarchical mean treatments require the fitted model result.")
    model = arm_result.model_result
    histories: dict[str, pd.DataFrame] = model["factor_history_z_by_class"]

    factor_vectors: dict[str, pd.Series] = {}
    forecast_meta: dict[str, Any] = {}
    for class_name, history in histories.items():
        ar1_mean, shock_sd = _ar1_forecast_moments(history)
        if treatment == "zero_centered_factor_innovations":
            vector = pd.Series(0.0, index=history.columns)
        elif treatment == "posterior_ar1_mean":
            vector = ar1_mean
        elif treatment == "sampled_factor_mean":
            if rng is None:
                raise ValueError("sampled_factor_mean requires an RNG.")
            vector = pd.Series(
                rng.normal(ar1_mean.to_numpy(), shock_sd.to_numpy()),
                index=history.columns,
                dtype=float,
            )
        else:
            raise AssertionError(treatment)
        factor_vectors[class_name] = vector
        forecast_meta[class_name] = {
            "n_factors": int(len(vector)),
            "factor_vector_l2": float(np.linalg.norm(vector.to_numpy(dtype=float))),
        }

    target = _target_mean_from_factor_vector(model, factor_vectors, assets)
    return target, {
        "mean_treatment": treatment,
        "factor_forecast_by_class": forecast_meta,
    }


def apply_hierarchical_mean_treatment(
    arm_result: ArmResult,
    treatment: str,
    *,
    rng: np.random.Generator | None = None,
) -> MeanTreatmentResult:
    """Apply a mean-only intervention and verify covariance invariance."""

    target, diagnostics = hierarchical_target_mean(
        arm_result, treatment, rng=rng
    )
    before_cov = arm_result.scenarios.cov()
    transformed = shift_scenario_mean(arm_result.scenarios, target)
    after_cov = transformed.cov()
    max_cov_error = float(
        np.max(np.abs(before_cov.to_numpy(dtype=float) - after_cov.to_numpy(dtype=float)))
    )
    mean_error = float(
        np.max(np.abs(transformed.mean(axis=0).to_numpy() - target.to_numpy()))
    )
    diagnostics = {
        **diagnostics,
        "max_abs_covariance_change": max_cov_error,
        "max_abs_target_mean_error": mean_error,
        "covariance_invariant": bool(max_cov_error <= 1e-10),
    }
    return MeanTreatmentResult(
        scenarios=transformed,
        target_mean=target,
        diagnostics=diagnostics,
    )
