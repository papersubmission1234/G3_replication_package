"""Predictive arms used by the confirmatory full-information aggregator."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .hierarchical_model import (
    HierarchicalBayesianModelConfig,
    fit_hierarchical_bayesian_etf_model,
)
from .joint_factor import JointFactorPosterior


@dataclass
class ArmResult:
    name: str
    scenarios: pd.DataFrame
    diagnostics: dict[str, Any]
    model_result: dict[str, Any] | None = None

    @property
    def mean(self) -> pd.Series:
        return self.scenarios.mean(axis=0)

    @property
    def variance(self) -> pd.Series:
        return self.scenarios.var(axis=0, ddof=1).clip(lower=1e-10)


class PredictiveArm(Protocol):
    name: str

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        ...


def _ensure_psd(cov: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, floor)
    return (vecs * vals) @ vecs.T


class JointFactorArm:
    """FF5+momentum Bayesian factor arm.
    
    Scoring and optimization use the same generated scenario matrix.
    """

    def __init__(self, name: str = "M_FF5MOM", factor_columns: tuple[str, ...] = (
        "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"
    )) -> None:
        self.name = name
        self.factor_columns = factor_columns

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        factor_df = factors["equity"].reindex(excess_returns.index)
        cols = [col for col in self.factor_columns if col in factor_df.columns]
        if len(cols) != len(self.factor_columns):
            missing = sorted(set(self.factor_columns).difference(cols))
            raise ValueError(f"Missing JF factors: {missing}")
        joined_index = excess_returns.index.intersection(factor_df.dropna().index)
        R = excess_returns.loc[joined_index].to_numpy(dtype=float)
        F = factor_df.loc[joined_index, cols].to_numpy(dtype=float)
        if len(joined_index) < max(12, len(cols) + 2):
            raise ValueError("Insufficient observations for JointFactorArm.")

        posterior = JointFactorPosterior(R.shape[1], F.shape[1])
        posterior.fit(F, R)
        gamma_draws = rng.standard_gamma(
            shape=posterior.a_post,
            size=(n_scenarios, posterior.n),
        )
        psi = np.clip(posterior.b_post[None, :] / gamma_draws, 1e-10, None)
        z = rng.standard_normal((n_scenarios, posterior.n, posterior.k))
        B = posterior.B_post[None, :, :] + np.sqrt(psi)[:, :, None] * (
            z @ posterior._L_inv_chol.T
        )
        factor_draws = rng.multivariate_normal(
            posterior.f_mean,
            _ensure_psd(posterior.f_cov),
            size=n_scenarios,
        )
        conditional_mean = np.einsum("snk,sk->sn", B, factor_draws)
        residual = rng.standard_normal((n_scenarios, posterior.n)) * np.sqrt(psi)
        scenarios = pd.DataFrame(
            conditional_mean + residual,
            columns=excess_returns.columns,
        )
        return ArmResult(
            name=self.name,
            scenarios=scenarios,
            diagnostics={
                "n_train": int(len(joined_index)),
                "n_assets": int(scenarios.shape[1]),
                "factor_columns": cols,
                "score_distribution": "same empirical scenarios as action",
            },
            model_result={"posterior": posterior},
        )


class HierarchicalArm:
    def __init__(
        self,
        config: HierarchicalBayesianModelConfig,
        name: str = "M_HierMacro",
    ) -> None:
        self.name = name
        self.config = config

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        cfg = replace(
            self.config,
            n_draws=n_scenarios,
            random_state=int(rng.integers(0, 2_147_483_647)),
        )
        result = fit_hierarchical_bayesian_etf_model(
            monthly_excess_returns=excess_returns,
            asset_class_series=asset_classes,
            macro_factors_df=factors["macro"],
            equity_factors_df=factors["equity"],
            bond_factors_df=factors["bond"],
            config=cfg,
        )
        scenarios = result["posterior_predictive_draws"].reset_index(drop=True)
        return ArmResult(
            name=self.name,
            scenarios=scenarios,
            diagnostics=result["diagnostics"],
            model_result=result,
        )


class LedoitWolfArm:
    def __init__(self, name: str = "M_LW_MinVar") -> None:
        self.name = name

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        complete = excess_returns.dropna(axis=1, how="any")
        if complete.shape[1] < 5:
            raise ValueError("Too few complete assets for Ledoit-Wolf arm.")
        estimator = LedoitWolf(assume_centered=False).fit(complete.to_numpy(dtype=float))
        cov = _ensure_psd(estimator.covariance_)
        draws = rng.multivariate_normal(
            np.zeros(complete.shape[1]), cov, size=n_scenarios
        )
        scenarios = pd.DataFrame(draws, columns=complete.columns)
        return ArmResult(
            name=self.name,
            scenarios=scenarios,
            diagnostics={
                "n_train": int(len(complete)),
                "n_assets": int(complete.shape[1]),
                "shrinkage": float(estimator.shrinkage_),
                "predictive_mean": "zero",
            },
        )


class MomentumArm:
    """Conventional twelve-minus-one cross-sectional momentum arm."""

    def __init__(
        self,
        name: str = "M_Momentum",
        lookback_months: int = 12,
        skip_months: int = 1,
        mean_scale: float = 0.005,
        winsor_quantile: float | None = 0.01,
    ) -> None:
        self.name = name
        self.lookback_months = int(lookback_months)
        self.skip_months = int(skip_months)
        self.mean_scale = float(mean_scale)
        self.winsor_quantile = winsor_quantile

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        required = self.lookback_months + self.skip_months
        if len(excess_returns) < required:
            raise ValueError(
                f"Momentum requires {required} rows, found {len(excess_returns)}."
            )
        end = -self.skip_months if self.skip_months > 0 else None
        start = -(self.lookback_months + self.skip_months)
        formation = excess_returns.iloc[start:end]
        if len(formation) != self.lookback_months:
            raise AssertionError(
                f"Momentum slice has {len(formation)} observations, expected "
                f"{self.lookback_months}."
            )
        signal = (1.0 + formation).prod(axis=0, min_count=self.lookback_months) - 1.0
        signal = signal.dropna()
        if self.winsor_quantile is not None and len(signal) > 10:
            q = float(self.winsor_quantile)
            signal = signal.clip(signal.quantile(q), signal.quantile(1.0 - q))
        sd = float(signal.std(ddof=0))
        z = (signal - signal.mean()) / sd if sd > 1e-12 else signal * 0.0
        mu = self.mean_scale * z

        complete = excess_returns.loc[:, mu.index].dropna(axis=1, how="any")
        mu = mu.reindex(complete.columns)
        if complete.shape[1] < 5:
            raise ValueError("Too few complete assets for momentum arm.")
        cov = np.cov(complete.to_numpy(dtype=float), rowvar=False, ddof=1)
        cov = _ensure_psd(cov + 1e-8 * np.eye(complete.shape[1]))
        draws = rng.multivariate_normal(mu.to_numpy(dtype=float), cov, size=n_scenarios)
        scenarios = pd.DataFrame(draws, columns=complete.columns)
        return ArmResult(
            name=self.name,
            scenarios=scenarios,
            diagnostics={
                "n_train": int(len(excess_returns)),
                "n_assets": int(complete.shape[1]),
                "lookback_months": self.lookback_months,
                "skip_months": self.skip_months,
                "formation_observations": int(len(formation)),
                "mean_scale": self.mean_scale,
                "winsor_quantile": self.winsor_quantile,
            },
        )


def shift_scenario_mean(
    scenarios: pd.DataFrame,
    target_mean: pd.Series | np.ndarray | float,
) -> pd.DataFrame:
    """Change only column means, preserving sample covariance exactly."""

    current_mean = scenarios.mean(axis=0)
    if np.isscalar(target_mean):
        target = pd.Series(float(target_mean), index=scenarios.columns)
    elif isinstance(target_mean, pd.Series):
        target = target_mean.reindex(scenarios.columns)
    else:
        target = pd.Series(np.asarray(target_mean, dtype=float), index=scenarios.columns)
    if target.isna().any():
        raise ValueError("Target scenario mean is missing for one or more assets.")
    shifted = scenarios.subtract(current_mean, axis=1).add(target, axis=1)
    return shifted


def shrink_scenario_mean(scenarios: pd.DataFrame, fraction: float) -> pd.DataFrame:
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Mean-shrinkage fraction must lie in [0, 1].")
    return shift_scenario_mean(scenarios, (1.0 - fraction) * scenarios.mean(axis=0))


class EWMAShrinkageArm:
    """Responsive zero-mean covariance arm using EWMA plus linear shrinkage."""

    def __init__(
        self,
        name: str = "M_EWMA_LW",
        half_life_months: float = 12.0,
    ) -> None:
        if half_life_months <= 0:
            raise ValueError("half_life_months must be positive.")
        self.name = name
        self.half_life_months = float(half_life_months)

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        complete = excess_returns.dropna(axis=1, how="any")
        if complete.shape[1] < 5:
            raise ValueError("Too few complete assets for EWMA-shrinkage arm.")
        x = complete.to_numpy(dtype=float)
        n_obs = len(x)
        decay = np.exp(np.log(0.5) / self.half_life_months)
        ages = np.arange(n_obs - 1, -1, -1, dtype=float)
        weights = decay ** ages
        weights /= weights.sum()
        mean = weights @ x
        centered = x - mean
        weighted = centered * np.sqrt(weights[:, None] * n_obs)
        estimator = LedoitWolf(assume_centered=True).fit(weighted)
        cov = _ensure_psd(estimator.covariance_)
        draws = rng.multivariate_normal(
            np.zeros(complete.shape[1]), cov, size=n_scenarios
        )
        return ArmResult(
            name=self.name,
            scenarios=pd.DataFrame(draws, columns=complete.columns),
            diagnostics={
                "n_train": int(n_obs),
                "n_assets": int(complete.shape[1]),
                "half_life_months": self.half_life_months,
                "shrinkage": float(estimator.shrinkage_),
                "predictive_mean": "zero",
                "covariance": "EWMA weighted Ledoit-Wolf",
            },
            model_result={
                "mean": pd.Series(np.zeros(complete.shape[1]), index=complete.columns),
                "covariance": pd.DataFrame(cov, index=complete.columns, columns=complete.columns),
            },
        )


class StudentTTailArm:
    """Heavy-tailed multivariate-t scenario arm with shrinkage covariance."""

    def __init__(
        self,
        name: str = "M_StudentT",
        degrees_of_freedom: float = 5.0,
        mean_mode: str = "zero",
    ) -> None:
        if degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must exceed two for finite covariance.")
        if mean_mode not in {"zero", "sample"}:
            raise ValueError("mean_mode must be zero or sample.")
        self.name = name
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.mean_mode = mean_mode

    def fit_generate(
        self,
        *,
        excess_returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        asset_classes: pd.Series,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ArmResult:
        complete = excess_returns.dropna(axis=1, how="any")
        if complete.shape[1] < 5:
            raise ValueError("Too few complete assets for Student-t arm.")
        estimator = LedoitWolf(assume_centered=False).fit(
            complete.to_numpy(dtype=float)
        )
        target_cov = _ensure_psd(estimator.covariance_)
        mean = (
            complete.mean(axis=0).to_numpy(dtype=float)
            if self.mean_mode == "sample"
            else np.zeros(complete.shape[1], dtype=float)
        )
        df = self.degrees_of_freedom
        # rescale t shocks to preserve the target covariance
        scale_cov = target_cov * (df - 2.0) / df
        z = rng.multivariate_normal(
            np.zeros(complete.shape[1]), _ensure_psd(scale_cov), size=n_scenarios
        )
        u = rng.chisquare(df, size=n_scenarios) / df
        draws = mean[None, :] + z / np.sqrt(u[:, None])
        return ArmResult(
            name=self.name,
            scenarios=pd.DataFrame(draws, columns=complete.columns),
            diagnostics={
                "n_train": int(len(complete)),
                "n_assets": int(complete.shape[1]),
                "degrees_of_freedom": df,
                "mean_mode": self.mean_mode,
                "shrinkage": float(estimator.shrinkage_),
                "tail_model": "elliptical multivariate Student-t",
            },
            model_result={
                "mean": pd.Series(mean, index=complete.columns),
                "covariance": pd.DataFrame(
                    target_cov, index=complete.columns, columns=complete.columns
                ),
            },
        )


def plugin_gaussian_scenarios(
    result: ArmResult,
    *,
    n_scenarios: int,
    rng: np.random.Generator,
) -> ArmResult:
    """Use fitted point moments for a Gaussian predictive approximation.
    
    This removes posterior parameter sampling for the within-model versus
    between-model uncertainty comparison.
    """

    columns = pd.Index(result.scenarios.columns.astype(str))
    mu: pd.Series
    cov: np.ndarray
    source = "moment_matched_empirical_scenarios"

    if result.name == "M_FF5MOM" and result.model_result is not None:
        posterior = result.model_result.get("posterior")
        if posterior is not None:
            mu_arr, cov = posterior.posterior_mean(posterior.f_mean)
            mu = pd.Series(mu_arr, index=columns)
            source = "joint_factor_posterior_mean"
        else:
            mu = result.scenarios.mean(axis=0)
            cov = result.scenarios.cov().to_numpy(dtype=float)
    elif result.name == "M_HierMacro" and result.model_result is not None:
        model = result.model_result
        mu = result.scenarios.mean(axis=0).reindex(columns).astype(float)
        loadings = pd.DataFrame(model["factor_loadings"]).reindex(index=columns)
        residual = pd.Series(model["residual_variance"]).reindex(columns).astype(float)
        history = pd.DataFrame(model["factor_history_z"]).astype(float)
        factor_cols = [
            col for col in loadings.columns
            if col != "alpha" and col in history.columns
        ]
        if factor_cols:
            beta = loadings[factor_cols].fillna(0.0).to_numpy(dtype=float)
            factor_cov = history[factor_cols].cov().to_numpy(dtype=float)
            cov = beta @ _ensure_psd(factor_cov) @ beta.T
            cov += np.diag(residual.fillna(residual.median()).clip(lower=1e-10))
            source = "hierarchical_point_loadings_and_residual_variance"
        else:
            cov = result.scenarios.cov().to_numpy(dtype=float)
    elif result.model_result is not None and {
        "mean", "covariance"
    }.issubset(result.model_result):
        mu = pd.Series(result.model_result["mean"]).reindex(columns).astype(float)
        cov = pd.DataFrame(result.model_result["covariance"]).reindex(
            index=columns, columns=columns
        ).to_numpy(dtype=float)
        source = "arm_point_mean_and_covariance"
    else:
        mu = result.scenarios.mean(axis=0)
        cov = result.scenarios.cov().to_numpy(dtype=float)

    mu = mu.fillna(0.0)
    cov = _ensure_psd(cov)
    draws = rng.multivariate_normal(mu.to_numpy(dtype=float), cov, size=n_scenarios)
    return ArmResult(
        name=result.name,
        scenarios=pd.DataFrame(draws, columns=columns),
        diagnostics={
            **result.diagnostics,
            "scenario_mode": "plugin_gaussian",
            "plugin_source": source,
        },
        model_result=result.model_result,
    )
