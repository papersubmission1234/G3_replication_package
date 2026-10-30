"""Zero-mean arms for covariance-estimator sensitivity checks.

These arms share the same mean specification and differ only in covariance.
They are kept separate from arms.py to preserve its raw-arm cache fingerprint.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .arms import ArmResult


def _ensure_psd(cov: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, float(floor))
    return (vectors * values) @ vectors.T


def mp_clean_covariance(x: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Apply Marchenko-Pastur eigenvalue clipping to sample correlations.
    
    x contains T complete observations of N returns. Demeaning gives T_eff=T-1,
    so the upper noise edge is (1+sqrt(N/T_eff))**2. Eigenvalues at or below
    the edge are replaced by their average. Larger eigenvalues are retained.
    The result is renormalized to unit diagonal, rescaled by sample standard
    deviations, and given a numerical eigenvalue floor.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("x must be a T x N matrix with at least 3 rows and 2 columns")
    if not np.isfinite(x).all():
        raise ValueError("MP cleaning requires complete finite observations")

    t, n = x.shape
    centered = x - x.mean(axis=0, keepdims=True)
    sd = centered.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1e-12)
    z = centered / sd
    corr = (z.T @ z) / (t - 1)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    q = float(n) / float(t - 1)
    lambda_plus = float((1.0 + np.sqrt(q)) ** 2)
    noise = eigvals <= lambda_plus
    n_noise = int(noise.sum())
    n_signal = int((~noise).sum())
    if n_noise:
        replacement = float(eigvals[noise].mean())
        cleaned_vals = eigvals.copy()
        cleaned_vals[noise] = replacement
    else:
        replacement = float("nan")
        cleaned_vals = eigvals.copy()

    cleaned_corr = (eigvecs * cleaned_vals) @ eigvecs.T
    cleaned_corr = 0.5 * (cleaned_corr + cleaned_corr.T)
    d = np.sqrt(np.maximum(np.diag(cleaned_corr), 1e-12))
    cleaned_corr = cleaned_corr / np.outer(d, d)
    np.fill_diagonal(cleaned_corr, 1.0)

    cov = cleaned_corr * np.outer(sd, sd)
    cov = _ensure_psd(cov, floor=floor)
    diagnostics = {
        "mp_t": int(t),
        "mp_t_eff": int(t - 1),
        "mp_q_n_over_t_eff": q,
        "mp_lambda_plus": lambda_plus,
        "mp_noise_eigenvalues": n_noise,
        "mp_signal_eigenvalues": n_signal,
        "mp_noise_replacement": replacement,
    }
    return cov, diagnostics


@dataclass
class _MemoizedCovarianceArm:
    """Base class with in-process memoization across matched risk-map cells."""

    name: str = "M_Covariance"

    def __post_init__(self) -> None:
        self._memo: dict[tuple, ArmResult] = {}

    def _key(self, excess_returns: pd.DataFrame, n_scenarios: int) -> tuple:
        return (
            str(excess_returns.index[0]),
            str(excess_returns.index[-1]),
            tuple(map(str, excess_returns.columns)),
            int(n_scenarios),
        )

    def _cached_or_generate(self, key: tuple, compute) -> ArmResult:
        if key not in self._memo:
            self._memo[key] = compute()
        return copy.deepcopy(self._memo[key])


class SampleCovarianceArm(_MemoizedCovarianceArm):
    """Zero-mean Gaussian arm with sample covariance.
    
    A small eigenvalue floor permits scenario generation when N exceeds T.
    It serves as a numerical safeguard, rather than a shrinkage target.
    """

    def fit_generate(self, *, excess_returns, factors, asset_classes, n_scenarios, rng):
        complete = excess_returns.dropna(axis=1, how="any")
        if complete.shape[1] < 5:
            raise ValueError("Too few complete assets for sample-covariance arm.")
        key = self._key(complete, n_scenarios)

        def compute():
            x = complete.to_numpy(float)
            cov = np.cov(x, rowvar=False, ddof=1)
            cov = _ensure_psd(cov)
            draws = rng.multivariate_normal(np.zeros(complete.shape[1]), cov, size=n_scenarios)
            return ArmResult(
                name=self.name,
                scenarios=pd.DataFrame(draws, columns=complete.columns),
                diagnostics={
                    "n_train": int(len(complete)),
                    "n_assets": int(complete.shape[1]),
                    "predictive_mean": "zero",
                    "covariance_estimator": "raw_sample_with_numerical_psd_floor",
                },
                model_result={
                    "mean": pd.Series(np.zeros(complete.shape[1]), index=complete.columns),
                    "covariance": pd.DataFrame(cov, index=complete.columns, columns=complete.columns),
                },
            )

        return self._cached_or_generate(key, compute)


class MarchenkoPasturArm(_MemoizedCovarianceArm):
    """Zero-mean Gaussian arm using MP eigenvalue clipping."""

    def fit_generate(self, *, excess_returns, factors, asset_classes, n_scenarios, rng):
        complete = excess_returns.dropna(axis=1, how="any")
        if complete.shape[1] < 5:
            raise ValueError("Too few complete assets for Marchenko-Pastur arm.")
        key = self._key(complete, n_scenarios)

        def compute():
            cov, diag = mp_clean_covariance(complete.to_numpy(float))
            draws = rng.multivariate_normal(np.zeros(complete.shape[1]), cov, size=n_scenarios)
            return ArmResult(
                name=self.name,
                scenarios=pd.DataFrame(draws, columns=complete.columns),
                diagnostics={
                    "n_train": int(len(complete)),
                    "n_assets": int(complete.shape[1]),
                    "predictive_mean": "zero",
                    "covariance_estimator": "marchenko_pastur_eigenvalue_clipping",
                    **diag,
                },
                model_result={
                    "mean": pd.Series(np.zeros(complete.shape[1]), index=complete.columns),
                    "covariance": pd.DataFrame(cov, index=complete.columns, columns=complete.columns),
                },
            )

        return self._cached_or_generate(key, compute)
