from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hierarchical_bayesian_etf_model import (
    DEFAULT_BOND_STYLE_COLUMNS,
    DEFAULT_EQUITY_STYLE_COLUMNS,
    DEFAULT_MACRO_COLUMNS,
    HierarchicalBayesianModelConfig,
    SUPPORTED_ASSET_CLASSES,
    fit_hierarchical_bayesian_etf_model,
)

EPS = 1e-10

@dataclass
class HierarchicalBacktestConfig:
    warmup_months: int = 36
    window_type: str = "rolling"
    rolling_window_months: Optional[int] = 36

    min_obs: int = 24
    max_train_missing_frac: float = 0.25
    min_eligible_etfs: int = 5

    n_posterior_draws: int = 1000
    n_scenario_draws: Optional[int] = None
    credible_interval_level: float = 0.95

    cvar_alpha: float = 0.95
    cvar_budget: Optional[float] = None
    turnover_penalty: float = 0.005
    transaction_cost_bps: float = 10.0
    min_weight: float = 0.0
    active_min_weight: float = 0.0005
    max_weight: float = 0.20
    partial_adjustment_alpha: float = 0.75

    residual_ewma_decay: float = 0.94
    factor_cov_ewma_decay: float = 0.94
    residual_lookback_months: Optional[int] = 60
    factor_cov_lookback_months: Optional[int] = 60
    covariance_shrinkage: float = 0.20
    covariance_shrink_target: str = "constant_correlation"
    min_cov_history_obs: int = 24

    scenario_distribution: str = "gaussian"
    student_t_df: float = 7.0

    solver: Optional[str] = None
    random_state: Optional[int] = 42

    forecast_mode: str = "mean_zero"
    model_n_iter: int = 6
    alpha_prior_var: float = 1e-4
    macro_prior_var: float = 0.20
    style_prior_var: float = 0.25
    alpha_group_mean_var: float = 1e-4
    macro_group_mean_var: float = 0.20
    style_group_mean_var: float = 0.25
    coef_prior_spec: dict[str, Any] = field(default_factory=lambda: {"family": "gaussian"})
    residual_var_floor: float = 1e-6

    initial_portfolio_balance: float = 1000.0
    show_progress: bool = True
    print_monthly_log: bool = True
    print_skips: bool = True

    macro_columns: tuple[str, ...] | None = DEFAULT_MACRO_COLUMNS
    equity_style_columns: tuple[str, ...] | None = DEFAULT_EQUITY_STYLE_COLUMNS
    bond_style_columns: tuple[str, ...] | None = DEFAULT_BOND_STYLE_COLUMNS
    factor_toggles: dict[str, bool] | None = None
    factor_toggle_default: bool = True


def _sort_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _to_month_end_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = df.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _ensure_common_model_index(
    monthly_simple_returns: pd.DataFrame,
    monthly_excess_returns: pd.DataFrame,
    macro_factors_df: pd.DataFrame,
    equity_factors_df: pd.DataFrame,
    bond_factors_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    common_idx = (
        monthly_simple_returns.index
        .intersection(monthly_excess_returns.index)
        .intersection(macro_factors_df.index)
        .intersection(equity_factors_df.index)
        .intersection(bond_factors_df.index)
        .sort_values()
    )
    if len(common_idx) < 2:
        raise ValueError("Not enough common monthly dates across returns and factor blocks.")
    return common_idx


def _project_to_box_simplex(
    raw_weights: pd.Series,
    max_weight: float,
    solver: Optional[str] = None,
) -> pd.Series:
    w0 = raw_weights.astype(float).to_numpy()
    n_assets = len(w0)

    if n_assets == 0:
        return raw_weights.copy()
    if n_assets * max_weight < 1.0 - 1e-8:
        raise ValueError("Box-simplex projection is infeasible: n_assets * max_weight < 1.")

    w = cp.Variable(n_assets)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(w - w0)),
        [cp.sum(w) == 1.0, w >= 0.0, w <= max_weight],
    )

    installed = set(cp.installed_solvers())
    candidates = [solver] if solver else ["CLARABEL", "ECOS", "OSQP", "SCS"]

    for candidate in candidates:
        if candidate is None or candidate not in installed:
            continue
        try:
            problem.solve(solver=candidate, warm_start=True, verbose=False)
            if problem.status in {"optimal", "optimal_inaccurate"}:
                projected = pd.Series(np.asarray(w.value).ravel(), index=raw_weights.index, dtype=float)
                projected[projected.abs() < 1e-12] = 0.0
                return projected
        except Exception:
            continue

    raise RuntimeError(f"Projection solver failed. Last status: {problem.status}")


def _validate_weights(
    weights: pd.Series,
    max_weight: float,
    min_weight: float = 0.0,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    active = weights > tol
    return {
        "weight_sum": float(weights.sum()),
        "has_negative_weight": bool((weights < -tol).any()),
        "max_weight_breached": bool((weights > max_weight + tol).any()),
        "min_active_weight_breached": bool((weights[active] < (min_weight - tol)).any()) if min_weight > 0 else False,
        "n_active_positions": int(active.sum()),
    }


def get_eligible_etfs_for_rebalance(
    rebalance_date: pd.Timestamp,
    eligibility_matrix: pd.DataFrame,
    y_train_window: pd.DataFrame,
    asset_class_series: pd.Series,
    config: HierarchicalBacktestConfig,
) -> Tuple[List[str], Dict[str, Any]]:
    if rebalance_date not in eligibility_matrix.index:
        return [], {"skip_reason": "rebalance_date_missing_from_eligibility_matrix"}

    eligibility_row = eligibility_matrix.loc[rebalance_date].reindex(y_train_window.columns).fillna(False)
    eligible_initial = eligibility_row[eligibility_row.astype(bool)].index.tolist()
    eligible_initial = [asset for asset in eligible_initial if pd.notna(asset_class_series.get(asset))]

    if not eligible_initial:
        return [], {"skip_reason": "no_supported_etfs_pass_eligibility_matrix"}

    y_sub = y_train_window[eligible_initial]
    obs_ok = y_sub.notna().sum(axis=0) >= config.min_obs
    missing_ok = y_sub.isna().mean(axis=0) <= config.max_train_missing_frac
    eligible_final = obs_ok.index[obs_ok & missing_ok].tolist()

    if len(eligible_final) < config.min_eligible_etfs:
        return eligible_final, {
            "skip_reason": "too_few_etfs_after_history_and_missingness_filters",
            "n_eligible_initial": len(eligible_initial),
            "n_eligible_final": len(eligible_final),
        }

    if config.min_weight > config.max_weight + 1e-12:
        return eligible_final, {
            "skip_reason": "min_weight_exceeds_max_weight",
            "n_eligible_initial": len(eligible_initial),
            "n_eligible_final": len(eligible_final),
        }

    if len(eligible_final) * config.max_weight < 1.0 - 1e-8:
        return eligible_final, {
            "skip_reason": "universe_too_small_for_max_weight_constraint",
            "n_eligible_initial": len(eligible_initial),
            "n_eligible_final": len(eligible_final),
        }

    if config.min_weight > 0 and len(eligible_final) * config.min_weight > 1.0 + 1e-8:
        return eligible_final, {
            "skip_reason": "universe_too_large_for_min_weight_constraint",
            "n_eligible_initial": len(eligible_initial),
            "n_eligible_final": len(eligible_final),
        }

    return eligible_final, {
        "skip_reason": None,
        "n_eligible_initial": len(eligible_initial),
        "n_eligible_final": len(eligible_final),
    }


def build_train_window_dates(
    model_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    rebalance_idx: int,
    config: HierarchicalBacktestConfig,
) -> Dict[str, Any]:
    rebalance_date = rebalance_dates[rebalance_idx]
    prediction_date = rebalance_dates[rebalance_idx + 1]

    model_pos = model_dates.get_loc(rebalance_date)
    if isinstance(model_pos, slice):
        raise ValueError("Duplicate rebalance_date found in model_dates.")

    if config.window_type == "expanding":
        train_start_pos = 0
    elif config.window_type == "rolling":
        if config.rolling_window_months is None:
            raise ValueError("rolling_window_months must be set when window_type='rolling'.")
        train_start_pos = max(0, model_pos - config.rolling_window_months + 1)
    else:
        raise ValueError("window_type must be 'expanding' or 'rolling'.")

    train_dates = model_dates[train_start_pos : model_pos + 1]
    return {
        "rebalance_date": rebalance_date,
        "prediction_date": prediction_date,
        "train_dates": train_dates,
    }


def _compute_ewma_covariance(
    returns_df: pd.DataFrame,
    decay: float,
) -> pd.DataFrame:
    returns_df = returns_df.copy().astype(float)
    returns_df = returns_df.loc[returns_df.notna().any(axis=1)]

    if len(returns_df) < 2:
        raise ValueError("Not enough observations to estimate covariance.")

    x = returns_df.to_numpy(dtype=float)
    mask = np.isfinite(x)
    weights = decay ** np.arange(len(returns_df) - 1, -1, -1, dtype=float)
    weights = weights / weights.sum()

    denom = np.maximum((mask * weights[:, None]).sum(axis=0), EPS)
    weighted_mean = np.nansum(np.where(mask, x, 0.0) * weights[:, None], axis=0) / denom
    x_centered = np.where(mask, x - weighted_mean, 0.0)

    cov = (x_centered * weights[:, None]).T @ x_centered
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=returns_df.columns, columns=returns_df.columns)


def _compute_ewma_variance(series: pd.Series, decay: float, floor: float) -> float:
    x = series.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float(floor)
    weights = decay ** np.arange(x.size - 1, -1, -1, dtype=float)
    weights = weights / weights.sum()
    mean_ = float(np.dot(weights, x))
    var_ = float(np.dot(weights, (x - mean_) ** 2))
    return max(var_, floor)


def repair_psd_covariance(
    cov_matrix: pd.DataFrame,
    min_eigenvalue: float = 1e-8,
    jitter: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cov = cov_matrix.copy().astype(float)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov.to_numpy(dtype=float))
    eigvals_clipped = np.clip(eigvals, min_eigenvalue, None)
    repaired = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    repaired = 0.5 * (repaired + repaired.T)
    repaired[np.diag_indices_from(repaired)] += jitter

    repaired_df = pd.DataFrame(repaired, index=cov.index, columns=cov.columns)
    eigvals_after = np.linalg.eigvalsh(repaired_df.to_numpy(dtype=float))
    return repaired_df, {
        "min_eigenvalue_before": float(eigvals.min()),
        "min_eigenvalue_after": float(eigvals_after.min()),
        "trace_after": float(np.trace(repaired_df.to_numpy(dtype=float))),
    }


def _shrink_covariance(
    raw_cov: pd.DataFrame,
    shrinkage: float,
    target: str,
) -> pd.DataFrame:
    variances = np.diag(raw_cov.to_numpy(dtype=float)).copy()
    variances = np.maximum(variances, EPS)
    stds = np.sqrt(variances)

    if target == "diagonal":
        target_cov = np.diag(variances)
    elif target == "identity":
        avg_var = float(np.mean(variances))
        target_cov = np.eye(len(variances)) * avg_var
    elif target == "constant_correlation":
        corr = raw_cov.to_numpy(dtype=float) / np.outer(stds, stds)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        avg_corr = float(corr[np.triu_indices_from(corr, k=1)].mean()) if corr.shape[0] > 1 else 0.0
        target_corr = np.full_like(corr, avg_corr)
        np.fill_diagonal(target_corr, 1.0)
        target_cov = np.outer(stds, stds) * target_corr
    else:
        raise ValueError("covariance_shrink_target must be 'diagonal', 'identity', or 'constant_correlation'.")

    shrunk = ((1.0 - shrinkage) * raw_cov.to_numpy(dtype=float)) + (shrinkage * target_cov)
    return pd.DataFrame(shrunk, index=raw_cov.index, columns=raw_cov.columns)


def estimate_factor_risk_from_model(
    model_result: Dict[str, Any],
    config: HierarchicalBacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.Series]:
    factor_loadings = model_result["factor_loadings"].copy().astype(float)
    beta_matrix = factor_loadings.drop(columns=["alpha"], errors="ignore").fillna(0.0)
    factor_history = model_result["factor_history_z"].reindex(columns=beta_matrix.columns).copy()
    residual_history = model_result["residual_history"].reindex(columns=beta_matrix.index).copy()

    if config.factor_cov_lookback_months is not None:
        factor_history = factor_history.iloc[-config.factor_cov_lookback_months :]
    factor_history = factor_history.dropna(how="any")

    if len(factor_history) < config.min_cov_history_obs:
        raise ValueError("Insufficient monthly factor history for risk estimation.")

    raw_factor_cov = _compute_ewma_covariance(factor_history, decay=config.factor_cov_ewma_decay)
    shrunk_factor_cov = _shrink_covariance(
        raw_cov=raw_factor_cov,
        shrinkage=config.covariance_shrinkage,
        target=config.covariance_shrink_target,
    )
    factor_covariance, factor_cov_diag = repair_psd_covariance(shrunk_factor_cov)

    if config.residual_lookback_months is not None:
        residual_history = residual_history.iloc[-config.residual_lookback_months :]

    residual_variance = pd.Series(
        {
            asset: _compute_ewma_variance(
                residual_history[asset],
                decay=config.residual_ewma_decay,
                floor=config.residual_var_floor,
            )
            for asset in beta_matrix.index
        },
        dtype=float,
        name="ewma_residual_variance",
    )

    sigma = (
        beta_matrix.to_numpy(dtype=float)
        @ factor_covariance.to_numpy(dtype=float)
        @ beta_matrix.to_numpy(dtype=float).T
    ) + np.diag(residual_variance.reindex(beta_matrix.index).to_numpy(dtype=float))
    sigma_df = pd.DataFrame(sigma, index=beta_matrix.index, columns=beta_matrix.index)
    sigma_df, sigma_diag = repair_psd_covariance(sigma_df)

    diagnostics = {
        "risk_frequency": "monthly_factor_model",
        "n_cov_assets": int(sigma_df.shape[0]),
        "n_factor_observations": int(len(factor_history)),
        "n_factor_columns": int(factor_covariance.shape[0]),
        "covariance_shrinkage": float(config.covariance_shrinkage),
        "covariance_shrink_target": config.covariance_shrink_target,
    }
    diagnostics.update({f"factor_cov_{k}": v for k, v in factor_cov_diag.items()})
    diagnostics.update({f"sigma_{k}": v for k, v in sigma_diag.items()})

    return sigma_df, diagnostics, factor_covariance, residual_variance

def generate_predictive_scenarios(
    posterior_mean_draws: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    n_scenarios: Optional[int] = None,
    distribution: str = "gaussian",
    student_t_df: float = 7.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    cols = posterior_mean_draws.columns
    covariance_matrix = covariance_matrix.reindex(index=cols, columns=cols).astype(float)

    n_available = len(posterior_mean_draws)
    n_scenarios = n_available if n_scenarios is None else int(n_scenarios)

    draw_idx = rng.choice(n_available, size=n_scenarios, replace=(n_scenarios > n_available))
    mean_draws = posterior_mean_draws.iloc[draw_idx].reset_index(drop=True)

    sigma = covariance_matrix.to_numpy(dtype=float)
    chol = np.linalg.cholesky(sigma + 1e-10 * np.eye(len(cols)))

    z = rng.standard_normal(size=(n_scenarios, len(cols)))
    if distribution == "student_t":
        scales = np.sqrt(student_t_df / rng.chisquare(student_t_df, size=n_scenarios))[:, None]
        z = z * scales
    elif distribution != "gaussian":
        raise ValueError("distribution must be 'gaussian' or 'student_t'.")

    shocks = z @ chol.T
    scenarios = mean_draws.to_numpy(dtype=float) + shocks
    return pd.DataFrame(scenarios, columns=cols)


def sample_posterior_predictive_scenarios(
    posterior_predictive_draws: pd.DataFrame,
    n_scenarios: Optional[int] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    posterior_predictive_draws = posterior_predictive_draws.copy().astype(float)
    cols = posterior_predictive_draws.columns
    n_available = len(posterior_predictive_draws)
    if n_available == 0:
        raise ValueError("posterior_predictive_draws is empty.")

    n_scenarios = n_available if n_scenarios is None else int(n_scenarios)
    rng = np.random.default_rng(random_state)
    draw_idx = rng.choice(n_available, size=n_scenarios, replace=(n_scenarios > n_available))
    scenarios = posterior_predictive_draws.iloc[draw_idx].reset_index(drop=True)
    scenarios.columns = cols
    return scenarios


def _portfolio_cvar_from_scenarios(
    returns_arr: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    portfolio_returns = returns_arr @ weights
    losses = -portfolio_returns
    var_alpha = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_alpha]
    return float(tail_losses.mean())


def _solve_return_cvar_portfolio(
    returns_arr: np.ndarray,
    asset_index: pd.Index,
    current_weights: pd.Series,
    config: HierarchicalBacktestConfig,
    min_weight_floor: float,
    cvar_budget: float,
    budget_source: str,
) -> Tuple[pd.Series, Dict[str, Any]]:
    current_weights = current_weights.reindex(asset_index).fillna(0.0).astype(float)
    n_scenarios, n_assets = returns_arr.shape

    if n_assets == 0:
        raise ValueError("No assets available for optimization.")
    if min_weight_floor > config.max_weight + 1e-12:
        raise ValueError("Optimization infeasible: min_weight > max_weight.")
    if n_assets * config.max_weight < 1.0 - 1e-8:
        raise ValueError("Optimization infeasible: n_assets * max_weight < 1.")
    if min_weight_floor > 0 and n_assets * min_weight_floor > 1.0 + 1e-8:
        raise ValueError("Optimization infeasible: n_assets * min_weight > 1.")

    expected_returns = returns_arr.mean(axis=0)
    w_prev = current_weights.to_numpy(dtype=float)
    tcost_rate = float(config.transaction_cost_bps) / 10000.0

    w = cp.Variable(n_assets)
    eta = cp.Variable()
    u = cp.Variable(n_scenarios, nonneg=True)

    losses = -returns_arr @ w
    tail_scale = 1.0 / max((1.0 - config.cvar_alpha) * n_scenarios, EPS)
    cvar_expr = eta + tail_scale * cp.sum(u)

    objective = cp.Maximize(
        expected_returns @ w
        - config.turnover_penalty * cp.norm1(w - w_prev)
        - tcost_rate * cp.norm1(w - w_prev)
    )

    constraints = [
        cp.sum(w) == 1.0,
        w >= min_weight_floor,
        w <= config.max_weight,
        u >= losses - eta,
        cvar_expr <= cvar_budget,
    ]

    problem = cp.Problem(objective, constraints)
    installed = set(cp.installed_solvers())
    candidates = [config.solver] if config.solver else ["CLARABEL", "ECOS", "SCS", "OSQP"]

    solver_used = None
    for cand in candidates:
        if cand is None or cand not in installed:
            continue
        try:
            problem.solve(solver=cand, warm_start=True, verbose=False)
            if problem.status in {"optimal", "optimal_inaccurate"}:
                solver_used = cand
                break
        except Exception:
            continue

    if solver_used is None:
        raise RuntimeError(f"Return-max / CVaR-budget optimizer failed. Last status: {problem.status}")

    target_weights = pd.Series(np.asarray(w.value).ravel(), index=asset_index, dtype=float)
    target_weights[target_weights.abs() < 1e-12] = 0.0

    diagnostics = {
        "solver_status": problem.status,
        "solver_used": solver_used,
        "objective_value": float(problem.value),
        "n_scenarios": int(n_scenarios),
        "n_assets": int(n_assets),
        "scenario_expected_return": float(expected_returns @ target_weights.to_numpy(dtype=float)),
        "scenario_cvar": _portfolio_cvar_from_scenarios(
            returns_arr=returns_arr,
            weights=target_weights.to_numpy(dtype=float),
            alpha=config.cvar_alpha,
        ),
        "cvar_budget_used": float(cvar_budget),
        "cvar_budget_source": budget_source,
        "min_weight_floor_used": float(min_weight_floor),
    }
    diagnostics.update(_validate_weights(target_weights, config.max_weight, min_weight_floor))
    return target_weights, diagnostics


def optimize_return_cvar_portfolio(
    scenario_matrix: pd.DataFrame,
    current_weights: pd.Series,
    config: HierarchicalBacktestConfig,
) -> Tuple[pd.Series, Dict[str, Any]]:
    scenario_matrix = scenario_matrix.copy().astype(float)
    current_weights = current_weights.reindex(scenario_matrix.columns).fillna(0.0).astype(float)
    returns_arr = scenario_matrix.to_numpy(dtype=float)
    n_scenarios, n_assets = returns_arr.shape
    if n_assets == 0:
        raise ValueError("No assets available for optimization.")

    if config.cvar_budget is None:
        budget_source = "equal_weight_scenario_cvar"
        budget_weights = np.repeat(1.0 / n_assets, n_assets)
        cvar_budget = _portfolio_cvar_from_scenarios(
            returns_arr=returns_arr,
            weights=budget_weights,
            alpha=config.cvar_alpha,
        )
    else:
        budget_source = "config"
        cvar_budget = float(config.cvar_budget)

    stage1_weights, stage1_diag = _solve_return_cvar_portfolio(
        returns_arr=returns_arr,
        asset_index=scenario_matrix.columns,
        current_weights=current_weights,
        config=config,
        min_weight_floor=float(config.min_weight),
        cvar_budget=float(cvar_budget),
        budget_source=budget_source,
    )

    active_floor = float(max(config.active_min_weight, 0.0))
    if active_floor <= 0.0:
        stage1_diag["optimizer_stage"] = "single_stage"
        return stage1_weights, stage1_diag

    kept_assets = stage1_weights[stage1_weights >= active_floor].index.tolist()
    if not kept_assets:
        kept_assets = [stage1_weights.idxmax()]

    if len(kept_assets) * active_floor > 1.0 + 1e-8:
        kept_assets = stage1_weights.sort_values(ascending=False).index.tolist()
        max_keep = int(np.floor(1.0 / active_floor))
        max_keep = max(1, min(max_keep, len(kept_assets)))
        kept_assets = kept_assets[:max_keep]

    kept_returns_arr = scenario_matrix.loc[:, kept_assets].to_numpy(dtype=float)
    kept_current_weights = current_weights.reindex(kept_assets).fillna(0.0)
    stage2_weights, stage2_diag = _solve_return_cvar_portfolio(
        returns_arr=kept_returns_arr,
        asset_index=pd.Index(kept_assets),
        current_weights=kept_current_weights,
        config=config,
        min_weight_floor=active_floor,
        cvar_budget=float(cvar_budget),
        budget_source=budget_source,
    )

    target_weights = pd.Series(0.0, index=scenario_matrix.columns, dtype=float)
    target_weights.loc[stage2_weights.index] = stage2_weights.values

    diagnostics = dict(stage2_diag)
    diagnostics["optimizer_stage"] = "two_stage_active_floor"
    diagnostics["stage1_n_assets"] = int(stage1_diag["n_assets"])
    diagnostics["stage2_n_assets"] = int(stage2_diag["n_assets"])
    diagnostics["active_min_weight"] = active_floor
    diagnostics["assets_dropped_after_stage1"] = int(stage1_diag["n_assets"] - stage2_diag["n_assets"])
    diagnostics["stage1_objective_value"] = float(stage1_diag["objective_value"])
    diagnostics["stage1_solver_used"] = stage1_diag["solver_used"]
    diagnostics["stage1_scenario_expected_return"] = float(stage1_diag["scenario_expected_return"])
    diagnostics["stage1_scenario_cvar"] = float(stage1_diag["scenario_cvar"])
    diagnostics.update(_validate_weights(target_weights, config.max_weight, active_floor))
    return target_weights, diagnostics


def apply_execution_rule(
    current_weights: pd.Series,
    target_weights: pd.Series,
    partial_adjustment_alpha: float,
    max_weight: float,
    solver: Optional[str] = None,
) -> pd.Series:
    current_weights = current_weights.astype(float)
    target_weights = target_weights.reindex(current_weights.index).fillna(0.0).astype(float)
    raw_exec = current_weights + partial_adjustment_alpha * (target_weights - current_weights)
    raw_exec = raw_exec.clip(lower=0.0)
    if raw_exec.sum() <= EPS:
        return raw_exec
    return _project_to_box_simplex(raw_exec, max_weight=max_weight, solver=solver)


def compute_turnover_and_cost(
    current_weights: pd.Series,
    executed_weights: pd.Series,
    transaction_cost_bps: float,
) -> Tuple[float, float]:
    current_weights = current_weights.astype(float)
    executed_weights = executed_weights.reindex(current_weights.index).fillna(0.0).astype(float)
    turnover = float((executed_weights - current_weights).abs().sum())
    cost = float(transaction_cost_bps / 10000.0 * turnover)
    return turnover, cost


def _compute_realized_portfolio_return(
    executed_weights: pd.Series,
    realized_monthly_returns: pd.Series,
    return_label: str,
) -> float:
    realized_aligned = realized_monthly_returns.reindex(executed_weights.index)
    held_mask = executed_weights > EPS

    held_weights = executed_weights.loc[held_mask].copy()
    if held_weights.empty:
        return 0.0

    held_returns = realized_aligned.loc[held_mask].copy()
    missing_mask = held_returns.isna()
    if missing_mask.any():
        missing_etfs = held_returns.index[missing_mask].tolist()
        raise ValueError(f"Missing realized {return_label} returns for held ETFs: {missing_etfs}")

    return float(held_weights.to_numpy() @ held_returns.astype(float).to_numpy())


def compute_sample_crps(forecast_draws: np.ndarray, realized_value: float) -> float:
    x = np.asarray(forecast_draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(realized_value):
        return np.nan
    x_sorted = np.sort(x)
    n = x_sorted.size
    term1 = np.mean(np.abs(x_sorted - realized_value))
    weights = (2 * np.arange(1, n + 1) - n - 1).astype(float)
    term2 = np.dot(weights, x_sorted) / (n ** 2)
    return float(term1 - term2)


def _logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    max_value = np.max(values)
    return float(max_value + np.log(np.mean(np.exp(values - max_value))))


def _kde_loglik_draws(
    forecast_draws: np.ndarray,
    realized_value: float,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    x = np.asarray(forecast_draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or not np.isfinite(realized_value):
        return np.array([], dtype=float), np.nan

    sample_std = np.std(x, ddof=1)
    if bandwidth is None:
        bandwidth = 1.06 * max(sample_std, EPS) * (x.size ** (-1 / 5))
    bandwidth = max(float(bandwidth), EPS)

    z = (realized_value - x) / bandwidth
    log_kernel = -0.5 * z ** 2 - 0.5 * np.log(2.0 * np.pi) - np.log(bandwidth)
    return log_kernel, bandwidth


def score_predictive_distribution(
    forecast_draws: np.ndarray,
    realized_value: float,
    interval_level: float = 0.95,
) -> Dict[str, float]:
    x = np.asarray(forecast_draws, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0 or not np.isfinite(realized_value):
        return {
            "predictive_mean": np.nan,
            "predictive_std": np.nan,
            "realized_value": realized_value,
            "crps": np.nan,
            "log_score": np.nan,
            "waic_penalty": np.nan,
            "waic_contribution": np.nan,
            "interval_lower": np.nan,
            "interval_upper": np.nan,
            "interval_hit": np.nan,
            "abs_error": np.nan,
            "squared_error": np.nan,
            "n_draws_used": int(x.size),
        }

    alpha = 1.0 - float(interval_level)
    interval_lower = float(np.quantile(x, alpha / 2.0))
    interval_upper = float(np.quantile(x, 1.0 - alpha / 2.0))

    loglik_draws, bandwidth = _kde_loglik_draws(x, realized_value)
    log_score = _logmeanexp(loglik_draws) if loglik_draws.size else np.nan
    waic_penalty = float(np.var(loglik_draws, ddof=1)) if loglik_draws.size > 1 else 0.0
    waic_contribution = float(-2.0 * (log_score - waic_penalty)) if np.isfinite(log_score) else np.nan

    predictive_mean = float(np.mean(x))
    error = predictive_mean - float(realized_value)
    return {
        "predictive_mean": predictive_mean,
        "predictive_std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "realized_value": float(realized_value),
        "crps": compute_sample_crps(x, realized_value),
        "log_score": log_score,
        "kde_bandwidth": bandwidth,
        "waic_penalty": waic_penalty,
        "waic_contribution": waic_contribution,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "interval_hit": float(interval_lower <= realized_value <= interval_upper),
        "abs_error": float(abs(error)),
        "squared_error": float(error ** 2),
        "n_draws_used": int(x.size),
    }


def _cross_sectional_rank_ic(score_rows: pd.DataFrame) -> float:
    usable = score_rows[["predictive_mean", "realized_value"]].dropna()
    if len(usable) < 2:
        return np.nan
    return float(usable["predictive_mean"].corr(usable["realized_value"], method="spearman"))


def _portfolio_risk_stats_from_scenarios(
    scenario_matrix: pd.DataFrame,
    weights: pd.Series,
    alpha: float,
) -> Dict[str, float]:
    weights = weights.reindex(scenario_matrix.columns).fillna(0.0).astype(float)
    scenario_portfolio_returns = scenario_matrix.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    var_threshold = float(np.quantile(scenario_portfolio_returns, 1.0 - alpha))
    tail = scenario_portfolio_returns[scenario_portfolio_returns <= var_threshold]
    cvar_threshold = float(tail.mean()) if tail.size else np.nan
    return {
        "scenario_portfolio_expected_return": float(np.mean(scenario_portfolio_returns)),
        "scenario_portfolio_var": var_threshold,
        "scenario_portfolio_cvar": cvar_threshold,
    }


def _build_model_config(config: HierarchicalBacktestConfig) -> HierarchicalBayesianModelConfig:
    return HierarchicalBayesianModelConfig(
        min_obs=config.min_obs,
        n_draws=config.n_posterior_draws,
        n_iter=config.model_n_iter,
        random_state=config.random_state,
        forecast_mode=config.forecast_mode,
        alpha_prior_var=config.alpha_prior_var,
        macro_prior_var=config.macro_prior_var,
        style_prior_var=config.style_prior_var,
        alpha_group_mean_var=config.alpha_group_mean_var,
        macro_group_mean_var=config.macro_group_mean_var,
        style_group_mean_var=config.style_group_mean_var,
        coef_prior_spec=config.coef_prior_spec,
        residual_var_floor=config.residual_var_floor,
        macro_columns=config.macro_columns,
        equity_style_columns=(
            DEFAULT_EQUITY_STYLE_COLUMNS if config.equity_style_columns is None else config.equity_style_columns
        ),
        bond_style_columns=(
            DEFAULT_BOND_STYLE_COLUMNS if config.bond_style_columns is None else config.bond_style_columns
        ),
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
    )

def run_backtest(
    monthly_simple_returns: pd.DataFrame,
    monthly_excess_returns: pd.DataFrame,
    eligibility_matrix: pd.DataFrame,
    asset_class_series: pd.Series,
    macro_factors_df: pd.DataFrame,
    equity_factors_df: pd.DataFrame,
    bond_factors_df: pd.DataFrame,
    config: Optional[HierarchicalBacktestConfig] = None,
) -> Dict[str, Any]:
    config = HierarchicalBacktestConfig() if config is None else config
    rng = np.random.default_rng(config.random_state)

    monthly_simple_returns = _to_month_end_index(_sort_index(monthly_simple_returns))
    monthly_excess_returns = _to_month_end_index(_sort_index(monthly_excess_returns))
    eligibility_matrix = _to_month_end_index(_sort_index(eligibility_matrix))
    macro_factors_df = _to_month_end_index(_sort_index(macro_factors_df))
    equity_factors_df = _to_month_end_index(_sort_index(equity_factors_df))
    bond_factors_df = _to_month_end_index(_sort_index(bond_factors_df))
    asset_class_series = asset_class_series.copy()
    asset_class_series.index = asset_class_series.index.astype(str)

    model_dates = _ensure_common_model_index(
        monthly_simple_returns=monthly_simple_returns,
        monthly_excess_returns=monthly_excess_returns,
        macro_factors_df=macro_factors_df,
        equity_factors_df=equity_factors_df,
        bond_factors_df=bond_factors_df,
    )
    rebalance_dates = model_dates.intersection(eligibility_matrix.index).sort_values()
    supported_assets = asset_class_series[asset_class_series.isin(SUPPORTED_ASSET_CLASSES)].index
    all_etfs = (
        monthly_simple_returns.columns
        .intersection(monthly_excess_returns.columns)
        .intersection(eligibility_matrix.columns)
        .intersection(supported_assets)
        .sort_values()
    )

    monthly_simple_returns = monthly_simple_returns.reindex(index=model_dates, columns=all_etfs)
    monthly_excess_returns = monthly_excess_returns.reindex(index=model_dates, columns=all_etfs)
    eligibility_matrix = eligibility_matrix.reindex(index=rebalance_dates, columns=all_etfs).fillna(False)
    asset_class_series = asset_class_series.reindex(all_etfs)

    eligible_rebalance_positions = [
        idx
        for idx, date in enumerate(rebalance_dates[:-1])
        if int((model_dates <= date).sum()) >= config.warmup_months
    ]
    if not eligible_rebalance_positions:
        raise ValueError("No rebalance dates satisfy the warmup requirement.")

    start_loop_idx = eligible_rebalance_positions[0]
    balance_anchor_date = rebalance_dates[start_loop_idx]
    loop = range(start_loop_idx, len(rebalance_dates) - 1)

    portfolio_returns = pd.Series(dtype=float, name="portfolio_return_gross")
    portfolio_returns_net = pd.Series(dtype=float, name="portfolio_return_net")
    portfolio_balance = pd.Series(dtype=float, name="portfolio_balance_gross")
    portfolio_balance_net = pd.Series(dtype=float, name="portfolio_balance_net")
    turnover_history = pd.Series(dtype=float, name="turnover")
    cost_history = pd.Series(dtype=float, name="transaction_cost")

    target_weights_history = pd.DataFrame(index=rebalance_dates[:-1], columns=all_etfs, dtype=float)
    executed_weights_history = pd.DataFrame(index=rebalance_dates[:-1], columns=all_etfs, dtype=float)
    posterior_mean_history = pd.DataFrame(index=rebalance_dates[:-1], columns=all_etfs, dtype=float)
    posterior_interval_lower_history = pd.DataFrame(index=rebalance_dates[:-1], columns=all_etfs, dtype=float)
    posterior_interval_upper_history = pd.DataFrame(index=rebalance_dates[:-1], columns=all_etfs, dtype=float)

    diagnostics_records: List[Dict[str, Any]] = []
    forecast_score_records: List[Dict[str, Any]] = []

    posterior_draws_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    posterior_predictive_draws_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    scenario_matrix_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    factor_loadings_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    residual_variance_by_date: Dict[pd.Timestamp, pd.Series] = {}
    factor_covariance_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    sigma_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    predictive_covariance_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    eligible_universe_by_date: Dict[pd.Timestamp, List[str]] = {}
    group_mean_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}

    current_weights = pd.Series(0.0, index=all_etfs, dtype=float)
    current_balance_gross = float(config.initial_portfolio_balance)
    current_balance_net = float(config.initial_portfolio_balance)

    progress = tqdm(loop, desc="Hierarchical ETF backtest", disable=not config.show_progress)

    for rebalance_idx in progress:
        slices = build_train_window_dates(
            model_dates=model_dates,
            rebalance_dates=rebalance_dates,
            rebalance_idx=rebalance_idx,
            config=config,
        )
        rebalance_date = slices["rebalance_date"]
        prediction_date = slices["prediction_date"]
        train_dates = slices["train_dates"]

        y_train_full = monthly_excess_returns.loc[train_dates, all_etfs]
        diag_row: Dict[str, Any] = {
            "rebalance_date": rebalance_date,
            "prediction_date": prediction_date,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "window_type": config.window_type,
            "skipped": False,
            "skip_reason": None,
            "monthly_log_score": np.nan,
            "monthly_waic": np.nan,
            "monthly_rank_ic": np.nan,
        }

        eligible_etfs, eligibility_diag = get_eligible_etfs_for_rebalance(
            rebalance_date=rebalance_date,
            eligibility_matrix=eligibility_matrix,
            y_train_window=y_train_full,
            asset_class_series=asset_class_series,
            config=config,
        )
        diag_row.update(eligibility_diag)
        diag_row["eligible_etfs"] = eligible_etfs
        eligible_universe_by_date[rebalance_date] = eligible_etfs

        if eligibility_diag.get("skip_reason") is not None:
            diag_row["skipped"] = True
            target_weights = current_weights.copy()
            executed_weights = current_weights.copy()
            turnover = 0.0
            cost = 0.0
        else:
            try:
                model_result = fit_hierarchical_bayesian_etf_model(
                    monthly_excess_returns=y_train_full[eligible_etfs],
                    asset_class_series=asset_class_series.reindex(eligible_etfs),
                    macro_factors_df=macro_factors_df.loc[train_dates],
                    equity_factors_df=equity_factors_df.loc[train_dates],
                    bond_factors_df=bond_factors_df.loc[train_dates],
                    config=_build_model_config(config),
                )

                posterior_draws = model_result["posterior_draws"].copy()
                posterior_predictive_draws = model_result.get("posterior_predictive_draws", posterior_draws).copy()
                posterior_mean = model_result["posterior_mean"].copy()
                alpha = 1.0 - float(config.credible_interval_level)
                posterior_interval_lower = posterior_draws.quantile(alpha / 2.0, axis=0)
                posterior_interval_upper = posterior_draws.quantile(1.0 - alpha / 2.0, axis=0)

                diag_row["model_n_etfs_used"] = int(len(posterior_mean))
                diag_row["model_asset_count_by_class"] = model_result["diagnostics"]["asset_count_by_class"]

                if len(posterior_mean) < config.min_eligible_etfs:
                    raise ValueError("Too few ETFs remain after hierarchical model estimation.")

                common_model_etfs = posterior_mean.index.intersection(posterior_predictive_draws.columns)
                posterior_draws = posterior_draws.reindex(columns=common_model_etfs)
                posterior_predictive_draws = posterior_predictive_draws.reindex(columns=common_model_etfs)
                posterior_mean = posterior_mean.reindex(common_model_etfs)
                posterior_interval_lower = posterior_interval_lower.reindex(common_model_etfs)
                posterior_interval_upper = posterior_interval_upper.reindex(common_model_etfs)

                if len(common_model_etfs) < config.min_eligible_etfs:
                    raise ValueError("Too few ETFs remain after intersecting model and predictive draw universes.")

                scenario_matrix = sample_posterior_predictive_scenarios(
                    posterior_predictive_draws=posterior_predictive_draws,
                    n_scenarios=config.n_scenario_draws,
                    random_state=int(rng.integers(0, 1_000_000_000)),
                )
                predictive_covariance = scenario_matrix.cov().astype(float)
                predictive_covariance, predictive_cov_diag = repair_psd_covariance(predictive_covariance)
                residual_variance = model_result["residual_variance"].reindex(common_model_etfs).copy()
                diag_row.update(
                    {
                        "risk_frequency": "posterior_predictive_draws",
                        "n_cov_assets": int(predictive_covariance.shape[0]),
                        "n_factor_observations": int(len(model_result["factor_history_z"])),
                        "n_factor_columns": int(model_result["factor_history_z"].shape[1]),
                        "scenario_source": "posterior_predictive_draws",
                    }
                )
                diag_row.update({f"sigma_{k}": v for k, v in predictive_cov_diag.items()})

                realized_next_excess = monthly_excess_returns.loc[prediction_date].reindex(common_model_etfs)
                monthly_score_rows: List[Dict[str, Any]] = []
                for etf in common_model_etfs:
                    realized_value = realized_next_excess.loc[etf]
                    if not np.isfinite(realized_value):
                        continue
                    score_row = score_predictive_distribution(
                        forecast_draws=scenario_matrix[etf].to_numpy(),
                        realized_value=realized_value,
                        interval_level=config.credible_interval_level,
                    )
                    score_row.update(
                        {"rebalance_date": rebalance_date, "prediction_date": prediction_date, "etf": etf}
                    )
                    monthly_score_rows.append(score_row)
                    forecast_score_records.append(score_row)

                if monthly_score_rows:
                    monthly_scores = pd.DataFrame(monthly_score_rows)
                    diag_row["monthly_log_score"] = float(monthly_scores["log_score"].sum())
                    diag_row["monthly_waic"] = float(monthly_scores["waic_contribution"].sum())
                    diag_row["monthly_rank_ic"] = _cross_sectional_rank_ic(monthly_scores)
                    diag_row["monthly_coverage"] = float(monthly_scores["interval_hit"].mean())
                    diag_row["monthly_mae"] = float(monthly_scores["abs_error"].mean())

                current_subset = current_weights.reindex(common_model_etfs).fillna(0.0)
                target_subset, opt_diag = optimize_return_cvar_portfolio(
                    scenario_matrix=scenario_matrix,
                    current_weights=current_subset,
                    config=config,
                )
                diag_row.update(opt_diag)
                diag_row["scenario_rows"] = int(scenario_matrix.shape[0])
                diag_row["scenario_cols"] = int(scenario_matrix.shape[1])

                target_weights = pd.Series(0.0, index=all_etfs, dtype=float)
                target_weights.loc[target_subset.index] = target_subset

                executed_weights = apply_execution_rule(
                    current_weights=current_weights,
                    target_weights=target_weights,
                    partial_adjustment_alpha=config.partial_adjustment_alpha,
                    max_weight=config.max_weight,
                    solver=config.solver,
                )
                eligible_now = (
                    eligibility_matrix.loc[rebalance_date]
                    .reindex(executed_weights.index)
                    .fillna(False)
                    .astype(bool)
                )
                executed_weights.loc[~eligible_now] = 0.0
                if executed_weights.sum() > 0:
                    executed_weights = executed_weights / executed_weights.sum()

                turnover, cost = compute_turnover_and_cost(
                    current_weights=current_weights,
                    executed_weights=executed_weights,
                    transaction_cost_bps=config.transaction_cost_bps,
                )

                executed_subset = executed_weights.reindex(common_model_etfs).fillna(0.0)
                portfolio_scenario_stats = _portfolio_risk_stats_from_scenarios(
                    scenario_matrix=scenario_matrix,
                    weights=executed_subset,
                    alpha=config.cvar_alpha,
                )
                diag_row.update(portfolio_scenario_stats)
                diag_row["portfolio_var_net"] = diag_row["scenario_portfolio_var"] - cost
                diag_row["portfolio_cvar_net"] = diag_row["scenario_portfolio_cvar"] - cost

                posterior_mean_history.loc[rebalance_date, posterior_mean.index] = posterior_mean.values
                posterior_interval_lower_history.loc[rebalance_date, posterior_interval_lower.index] = posterior_interval_lower.values
                posterior_interval_upper_history.loc[rebalance_date, posterior_interval_upper.index] = posterior_interval_upper.values

                posterior_draws_by_date[rebalance_date] = posterior_draws
                posterior_predictive_draws_by_date[rebalance_date] = posterior_predictive_draws
                scenario_matrix_by_date[rebalance_date] = scenario_matrix
                factor_loadings_by_date[rebalance_date] = model_result["factor_loadings"].copy()
                residual_variance_by_date[rebalance_date] = residual_variance.copy()
                factor_covariance_by_date[rebalance_date] = pd.DataFrame()
                sigma_by_date[rebalance_date] = predictive_covariance.copy()
                predictive_covariance_by_date[rebalance_date] = predictive_covariance.copy()
                group_mean_by_date[rebalance_date] = model_result["group_mean_by_class"].copy()

            except Exception as exc:
                diag_row["skipped"] = True
                diag_row["skip_reason"] = f"model_or_optimization_error: {exc}"
                target_weights = current_weights.copy()
                executed_weights = current_weights.copy()
                turnover = 0.0
                cost = 0.0

        realized_gross_return = _compute_realized_portfolio_return(
            executed_weights=executed_weights,
            realized_monthly_returns=monthly_simple_returns.loc[prediction_date],
            return_label="simple",
        )
        realized_net_return = realized_gross_return - cost
        realized_gross_excess_return = _compute_realized_portfolio_return(
            executed_weights=executed_weights,
            realized_monthly_returns=monthly_excess_returns.loc[prediction_date],
            return_label="excess",
        )
        realized_net_excess_return = realized_gross_excess_return - cost

        current_balance_gross *= (1.0 + realized_gross_return)
        current_balance_net *= (1.0 + realized_net_return)
        total_return_gross = current_balance_gross / config.initial_portfolio_balance - 1.0
        total_return_net = current_balance_net / config.initial_portfolio_balance - 1.0

        diag_row["turnover"] = turnover
        diag_row["transaction_cost"] = cost
        diag_row["realized_gross_return"] = realized_gross_return
        diag_row["realized_net_return"] = realized_net_return
        diag_row["realized_gross_excess_return"] = realized_gross_excess_return
        diag_row["realized_net_excess_return"] = realized_net_excess_return
        diag_row["portfolio_total_return_gross"] = total_return_gross
        diag_row["portfolio_total_return_net"] = total_return_net
        diag_row["portfolio_balance_gross"] = current_balance_gross
        diag_row["portfolio_balance_net"] = current_balance_net

        if np.isfinite(diag_row.get("portfolio_var_net", np.nan)):
            diag_row["var_breach"] = float(realized_net_excess_return <= diag_row["portfolio_var_net"])
        else:
            diag_row["var_breach"] = np.nan
        if np.isfinite(diag_row.get("portfolio_cvar_net", np.nan)):
            diag_row["cvar_breach"] = float(realized_net_excess_return <= diag_row["portfolio_cvar_net"])
        else:
            diag_row["cvar_breach"] = np.nan

        diag_row.update({f"exec_{k}": v for k, v in _validate_weights(executed_weights, config.max_weight, config.min_weight).items()})

        target_weights_history.loc[rebalance_date] = target_weights.values
        executed_weights_history.loc[rebalance_date] = executed_weights.values
        portfolio_returns.loc[prediction_date] = realized_gross_return
        portfolio_returns_net.loc[prediction_date] = realized_net_return
        portfolio_balance.loc[prediction_date] = current_balance_gross
        portfolio_balance_net.loc[prediction_date] = current_balance_net
        turnover_history.loc[rebalance_date] = turnover
        cost_history.loc[rebalance_date] = cost

        diagnostics_records.append(diag_row)
        current_weights = executed_weights.copy()

        progress.set_postfix(
            month=str(rebalance_date.date()),
            month_net=f"{realized_net_return:.2%}",
            total_net=f"{total_return_net:.2%}",
            nav_net=f"{current_balance_net:,.2f}",
            turnover=f"{turnover:.2%}",
            cost=f"{cost:.2%}",
        )

        if config.print_monthly_log:
            print(
                f"[{rebalance_date.date()} -> {prediction_date.date()}] "
                f"month_net={realized_net_return:.2%}, "
                f"total_net={total_return_net:.2%}, "
                f"nav_net={current_balance_net:,.2f}, "
                f"turnover={turnover:.2%}, "
                f"cost_drag={cost:.2%}"
            )
            if diag_row.get("skip_reason") and config.print_skips:
                print(f"  skip_reason: {diag_row['skip_reason']}")

    diagnostics_table = pd.DataFrame(diagnostics_records).set_index("rebalance_date").sort_index()
    forecast_scores_table = pd.DataFrame(forecast_score_records)
    if not forecast_scores_table.empty:
        forecast_scores_table = forecast_scores_table.sort_values(["prediction_date", "etf"]).reset_index(drop=True)

    return {
        "portfolio_returns": portfolio_returns.sort_index(),
        "portfolio_returns_net": portfolio_returns_net.sort_index(),
        "portfolio_balance": portfolio_balance.sort_index(),
        "portfolio_balance_net": portfolio_balance_net.sort_index(),
        "target_weights_history": target_weights_history.sort_index(),
        "executed_weights_history": executed_weights_history.sort_index(),
        "turnover_history": turnover_history.sort_index(),
        "cost_history": cost_history.sort_index(),
        "diagnostics_table": diagnostics_table,
        "posterior_mean_history": posterior_mean_history.sort_index(),
        "posterior_interval_lower_history": posterior_interval_lower_history.sort_index(),
        "posterior_interval_upper_history": posterior_interval_upper_history.sort_index(),
        "posterior_draws_by_date": posterior_draws_by_date,
        "posterior_predictive_draws_by_date": posterior_predictive_draws_by_date,
        "scenario_matrix_by_date": scenario_matrix_by_date,
        "factor_loadings_by_date": factor_loadings_by_date,
        "residual_variance_by_date": residual_variance_by_date,
        "factor_covariance_by_date": factor_covariance_by_date,
        "sigma_by_date": sigma_by_date,
        "predictive_covariance_by_date": predictive_covariance_by_date,
        "group_mean_by_date": group_mean_by_date,
        "eligible_universe_by_date": eligible_universe_by_date,
        "forecast_scores_table": forecast_scores_table,
        "balance_anchor_date": balance_anchor_date,
        "initial_portfolio_balance": config.initial_portfolio_balance,
        "config": config,
    }


def compute_portfolio_performance_summary(
    backtest_result: Dict[str, Any],
    rf_series: Optional[pd.Series] = None,
    benchmark_excess_series: Optional[pd.Series] = None,
    alpha: float = 0.95,
) -> pd.DataFrame:
    returns_net = backtest_result["portfolio_returns_net"].dropna().astype(float)
    if returns_net.empty:
        raise ValueError("portfolio_returns_net is empty.")

    periods_per_year = 12
    initial_balance = float(backtest_result["initial_portfolio_balance"])
    ending_balance = float(backtest_result["portfolio_balance_net"].dropna().iloc[-1])
    cumulative_return = ending_balance / initial_balance - 1.0
    annualized_return = (ending_balance / initial_balance) ** (periods_per_year / len(returns_net)) - 1.0
    annualized_volatility = returns_net.std(ddof=1) * np.sqrt(periods_per_year)

    if rf_series is None:
        rf_series = pd.Series(0.0, index=returns_net.index)
    else:
        rf_series = rf_series.reindex(returns_net.index).fillna(0.0).astype(float)

    excess_returns = returns_net - rf_series
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns_net.std(ddof=1) if returns_net.std(ddof=1) > 0 else np.nan

    downside = np.minimum(excess_returns.to_numpy(dtype=float), 0.0)
    downside_rms = np.sqrt(np.mean(downside ** 2))
    downside_deviation = float(downside_rms * np.sqrt(periods_per_year))
    sortino = float(np.sqrt(periods_per_year) * excess_returns.mean() / downside_rms) if downside_rms > 0 else np.nan

    balance_plot = pd.concat(
        [
            pd.Series([initial_balance], index=pd.DatetimeIndex([backtest_result["balance_anchor_date"]])),
            backtest_result["portfolio_balance_net"].dropna(),
        ]
    ).sort_index()
    drawdown = balance_plot / balance_plot.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

    var_threshold = float(returns_net.quantile(1.0 - alpha))
    cvar_threshold = float(returns_net[returns_net <= var_threshold].mean()) if (returns_net <= var_threshold).any() else np.nan

    omega_num = float(np.clip((returns_net - rf_series).to_numpy(dtype=float), 0.0, None).sum())
    omega_den = float(np.clip((rf_series - returns_net).to_numpy(dtype=float), 0.0, None).sum())
    omega_ratio = omega_num / omega_den if omega_den > 0 else np.nan

    pain_index = float(drawdown.abs().mean())
    pain_ratio = annualized_return / pain_index if pain_index > 0 else np.nan

    treynor = np.nan
    if benchmark_excess_series is not None:
        benchmark = benchmark_excess_series.reindex(returns_net.index).astype(float)
        common = pd.concat([excess_returns.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
        if len(common) >= 2 and common["benchmark"].var(ddof=1) > 0:
            beta = float(common["portfolio"].cov(common["benchmark"]) / common["benchmark"].var(ddof=1))
            treynor = float((common["portfolio"].mean() * periods_per_year) / beta) if abs(beta) > EPS else np.nan

    diagnostics = backtest_result["diagnostics_table"]
    var_breach_rate = float(diagnostics["var_breach"].dropna().mean()) if "var_breach" in diagnostics else np.nan
    cvar_breach_rate = float(diagnostics["cvar_breach"].dropna().mean()) if "cvar_breach" in diagnostics else np.nan

    return pd.DataFrame(
        {
            "Metric": [
                "Annualized Return",
                "Annualized Volatility",
                "Cumulative Return",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Treynor Ratio",
                "Calmar Ratio",
                f"VaR {int(alpha * 100)}%",
                f"CVaR {int(alpha * 100)}%",
                "Downside Deviation",
                "Omega Ratio",
                "Pain Ratio",
                "VaR Breach Rate",
                "CVaR Breach Rate",
                "Max Drawdown",
                "Ending Balance",
            ],
            "Value": [
                annualized_return,
                annualized_volatility,
                cumulative_return,
                sharpe,
                sortino,
                treynor,
                calmar,
                var_threshold,
                cvar_threshold,
                downside_deviation,
                omega_ratio,
                pain_ratio,
                var_breach_rate,
                cvar_breach_rate,
                max_drawdown,
                ending_balance,
            ],
        }
    )


def summarize_model_performance(backtest_result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores = backtest_result["forecast_scores_table"].copy()
    if scores.empty:
        raise ValueError("forecast_scores_table is empty.")

    monthly_rank_ic = scores.groupby("prediction_date").apply(lambda frame: _cross_sectional_rank_ic(frame)).rename("rank_ic")
    summary = pd.DataFrame(
        {
            "Metric": [
                "Predictive Log Score (Mean)",
                "Predictive Log Score (Total)",
                "WAIC",
                "Rank IC (Mean Spearman)",
                "Rank IC (Median Spearman)",
                "MAE",
                "RMSE",
                "95% Credible Interval Coverage",
                "Forecast Count",
            ],
            "Value": [
                float(scores["log_score"].mean()),
                float(scores["log_score"].sum()),
                float(scores["waic_contribution"].sum()),
                float(monthly_rank_ic.mean()),
                float(monthly_rank_ic.median()),
                float(scores["abs_error"].mean()),
                float(np.sqrt(scores["squared_error"].mean())),
                float(scores["interval_hit"].mean()),
                int(len(scores)),
            ],
        }
    )

    by_month = (
        scores.groupby("prediction_date")
        .agg(
            mean_log_score=("log_score", "mean"),
            waic=("waic_contribution", "sum"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            coverage=("interval_hit", "mean"),
            n_forecasts=("etf", "size"),
        )
        .join(monthly_rank_ic, how="left")
        .sort_index()
    )
    return summary, by_month


def format_metric_table(metric_table: pd.DataFrame, percent_metrics: set[str]) -> pd.DataFrame:
    formatted = metric_table.copy()
    values = []
    for metric, value in zip(formatted["Metric"], formatted["Value"]):
        if pd.isna(value):
            values.append("NaN")
        elif metric in percent_metrics:
            values.append(f"{value:.2%}")
        elif metric == "Ending Balance":
            values.append(f"{value:,.2f}")
        elif metric == "Forecast Count":
            values.append(f"{int(value)}")
        else:
            values.append(f"{value:.4f}")
    formatted["Value"] = values
    return formatted


def build_live_diagnostics_table(backtest_result: Dict[str, Any]) -> pd.DataFrame:
    diagnostics = backtest_result["diagnostics_table"].copy()
    keep = [
        "prediction_date",
        "model_n_etfs_used",
        "realized_net_return",
        "portfolio_total_return_net",
        "turnover",
        "transaction_cost",
        "monthly_waic",
        "monthly_rank_ic",
        "monthly_log_score",
        "scenario_expected_return",
        "scenario_cvar",
        "scenario_portfolio_var",
        "scenario_portfolio_cvar",
        "var_breach",
        "cvar_breach",
        "skipped",
        "skip_reason",
    ]
    existing = [col for col in keep if col in diagnostics.columns]
    return diagnostics[existing]

def plot_balance_and_drawdown(backtest_result: Dict[str, Any]) -> None:
    initial_balance = float(backtest_result["initial_portfolio_balance"])
    anchor_date = backtest_result["balance_anchor_date"]
    balance_net = pd.concat(
        [
            pd.Series([initial_balance], index=pd.DatetimeIndex([anchor_date])),
            backtest_result["portfolio_balance_net"].dropna(),
        ]
    ).sort_index()
    drawdown = balance_net / balance_net.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(balance_net.index, balance_net.values, color="navy", linewidth=2.2)
    axes[0].set_title("Portfolio Balance Over Time")
    axes[0].set_ylabel("Balance")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(drawdown.index, drawdown.values, 0.0, alpha=0.35, color="firebrick")
    axes[1].plot(drawdown.index, drawdown.values, color="firebrick", linewidth=1.8)
    axes[1].set_title("Portfolio Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
