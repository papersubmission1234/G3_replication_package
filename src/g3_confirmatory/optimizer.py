"""Deterministic mean-return maximization under a CVaR budget using HiGHS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-12


@dataclass(frozen=True)
class OptimizerConfig:
    cvar_alpha: float = 0.95
    max_weight: float = 0.20
    active_min_weight: float = 0.0005
    adjustment_penalty: float = 0.10
    transaction_cost_bps: float = 10.0
    cvar_budget_mode: Literal["scenario_equal_weight", "explicit"] = "scenario_equal_weight"
    cvar_budget: float | None = None
    highs_options: dict[str, Any] | None = None

    def validate(self, n_assets: int) -> None:
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between zero and one.")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must lie in (0, 1].")
        if n_assets * self.max_weight < 1.0 - 1e-10:
            raise ValueError(
                f"Infeasible cap: {n_assets} assets * {self.max_weight} < 1."
            )
        if self.active_min_weight < 0.0 or self.active_min_weight > self.max_weight:
            raise ValueError("active_min_weight must be in [0, max_weight].")
        if self.adjustment_penalty < 0.0 or self.transaction_cost_bps < 0.0:
            raise ValueError("penalties and costs must be nonnegative.")
        if self.cvar_budget_mode == "explicit" and self.cvar_budget is None:
            raise ValueError("explicit cvar_budget_mode requires cvar_budget.")


@dataclass
class OptimizationResult:
    weights: pd.Series
    diagnostics: dict[str, Any]


@dataclass
class TurnoverResult:
    one_way_turnover: float
    full_l1_change: float
    forced_sale_weight: float
    initial_funding: bool
    transaction_cost: float


def empirical_var_cvar(
    portfolio_returns: np.ndarray,
    alpha: float = 0.95,
) -> dict[str, float | int]:
    """Compute equal-probability empirical VaR and CVaR.
    
    CVaR follows the Rockafellar--Uryasev LP formulation. A fractional tail count
    gives the boundary loss its fractional probability mass.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    returns = np.asarray(portfolio_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = int(returns.size)
    if n == 0:
        return {
            "var": np.nan,
            "cvar": np.nan,
            "n_observations": 0,
            "effective_tail_count": np.nan,
            "full_tail_observations": 0,
            "fractional_boundary_weight": np.nan,
        }

    losses_ascending = np.sort(-returns)
    # finite-scenario VaR uses the left-continuous quantile
    var_index = min(n - 1, max(0, int(np.ceil(float(alpha) * n)) - 1))
    var = float(losses_ascending[var_index])

    losses_descending = losses_ascending[::-1]
    tail_mass = float((1.0 - float(alpha)) * n)
    full = int(np.floor(tail_mass + 1e-14))
    fraction = float(tail_mass - full)
    weighted_sum = float(losses_descending[:full].sum()) if full else 0.0
    if fraction > 1e-14:
        boundary_index = min(full, n - 1)
        weighted_sum += fraction * float(losses_descending[boundary_index])
    cvar = float(weighted_sum / tail_mass)
    return {
        "var": var,
        "cvar": cvar,
        "n_observations": n,
        "effective_tail_count": tail_mass,
        "full_tail_observations": full,
        "fractional_boundary_weight": fraction,
    }


def empirical_cvar(portfolio_returns: np.ndarray, alpha: float = 0.95) -> float:
    return float(empirical_var_cvar(portfolio_returns, alpha)["cvar"])


def equal_weight_cvar(scenarios: pd.DataFrame, alpha: float = 0.95) -> float:
    n = scenarios.shape[1]
    if n == 0:
        raise ValueError("No assets in scenario matrix.")
    return empirical_cvar(
        scenarios.to_numpy(dtype=float) @ np.repeat(1.0 / n, n),
        alpha=alpha,
    )


def _solve_lp(
    scenarios: pd.DataFrame,
    pretrade_weights: pd.Series,
    config: OptimizerConfig,
    *,
    min_weight_floor: float,
    cvar_budget: float,
    initial_funding: bool,
    objective_mean: pd.Series | None = None,
    risk_scenarios: pd.DataFrame | None = None,
) -> OptimizationResult:
    objective_scenarios = scenarios.astype(float)
    if objective_scenarios.isna().any().any():
        raise ValueError("Objective scenario matrix contains missing values.")
    assets = pd.Index(objective_scenarios.columns.astype(str))
    objective_scenarios.columns = assets

    if risk_scenarios is None:
        risk_matrix = objective_scenarios
        risk_scenario_source = "objective_scenarios"
    else:
        risk_matrix = risk_scenarios.astype(float).copy()
        risk_matrix.columns = risk_matrix.columns.astype(str)
        missing = assets.difference(risk_matrix.columns)
        extra = risk_matrix.columns.difference(assets)
        if len(missing) or len(extra):
            raise ValueError(
                "Risk scenarios must contain exactly the objective assets; "
                f"missing={missing.tolist()[:10]}, extra={extra.tolist()[:10]}."
            )
        risk_matrix = risk_matrix.reindex(columns=assets)
        if risk_matrix.isna().any().any():
            raise ValueError("Risk scenario matrix contains missing values.")
        risk_scenario_source = "caller_override"

    R_objective = np.ascontiguousarray(
        objective_scenarios.to_numpy(dtype=float)
    )
    R_risk = np.ascontiguousarray(risk_matrix.to_numpy(dtype=float))
    S_objective, N = R_objective.shape
    S_risk = R_risk.shape[0]
    config.validate(N)
    if min_weight_floor * N > 1.0 + 1e-10:
        raise ValueError("Active floor is infeasible for this asset count.")

    prev = pretrade_weights.copy()
    prev.index = prev.index.astype(str)
    prev = prev.reindex(assets).fillna(0.0).astype(float)

    w0 = 0
    eta_idx = N
    u0 = N + 1
    d0 = N + 1 + S_risk
    n_vars = N + 1 + S_risk + N

    c = np.zeros(n_vars, dtype=float)
    if objective_mean is None:
        objective_mean_vec = R_objective.mean(axis=0)
        objective_mean_source = "scenario_mean"
    else:
        mean_series = objective_mean.copy()
        mean_series.index = mean_series.index.astype(str)
        mean_series = mean_series.reindex(assets)
        if mean_series.isna().any():
            missing = mean_series[mean_series.isna()].index.tolist()
            raise ValueError(
                "objective_mean is missing assets: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        objective_mean_vec = mean_series.to_numpy(dtype=float)
        objective_mean_source = "caller_override"
    c[w0:w0 + N] = -objective_mean_vec
    one_way_multiplier = 1.0 if initial_funding else 0.5
    cost_rate = config.transaction_cost_bps / 10_000.0
    c[d0:d0 + N] = config.adjustment_penalty + cost_rate * one_way_multiplier

    A_eq = np.zeros((1, n_vars), dtype=float)
    A_eq[0, w0:w0 + N] = 1.0
    b_eq = np.array([1.0], dtype=float)

    n_ub = 1 + S_risk + 2 * N
    A_ub = np.zeros((n_ub, n_vars), dtype=float)
    b_ub = np.zeros(n_ub, dtype=float)

    tail_scale = 1.0 / ((1.0 - config.cvar_alpha) * S_risk)
    A_ub[0, eta_idx] = 1.0
    A_ub[0, u0:u0 + S_risk] = tail_scale
    b_ub[0] = float(cvar_budget)

    for s in range(S_risk):
        row = 1 + s
        A_ub[row, w0:w0 + N] = -R_risk[s]
        A_ub[row, eta_idx] = -1.0
        A_ub[row, u0 + s] = -1.0

    abs_start = 1 + S_risk
    for i in range(N):
        row_pos = abs_start + 2 * i
        row_neg = row_pos + 1
        A_ub[row_pos, w0 + i] = 1.0
        A_ub[row_pos, d0 + i] = -1.0
        b_ub[row_pos] = prev.iloc[i]

        A_ub[row_neg, w0 + i] = -1.0
        A_ub[row_neg, d0 + i] = -1.0
        b_ub[row_neg] = -prev.iloc[i]

    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(float(min_weight_floor), float(config.max_weight))] * N)
    bounds.append((None, None))
    bounds.extend([(0.0, None)] * S_risk)
    bounds.extend([(0.0, None)] * N)

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={} if config.highs_options is None else dict(config.highs_options),
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"HiGHS optimization failed: status={result.status}, message={result.message}"
        )

    weights = pd.Series(result.x[w0:w0 + N], index=assets, dtype=float)
    weights[weights.abs() < 1e-12] = 0.0
    weights /= weights.sum()

    objective_scenario_returns = (
        R_objective @ weights.to_numpy(dtype=float)
    )
    risk_scenario_returns = R_risk @ weights.to_numpy(dtype=float)
    objective_tail = empirical_var_cvar(
        objective_scenario_returns, config.cvar_alpha
    )
    risk_tail = empirical_var_cvar(
        risk_scenario_returns, config.cvar_alpha
    )
    full_l1_inside = float(np.abs(weights.to_numpy() - prev.to_numpy()).sum())
    ineqlin = getattr(result, "ineqlin", None)
    marginals = getattr(ineqlin, "marginals", None)
    residuals = getattr(ineqlin, "residual", None)
    cvar_budget_marginal_min_objective = (
        float(marginals[0])
        if marginals is not None and len(marginals) > 0
        else np.nan
    )
    cvar_budget_slack = (
        float(residuals[0])
        if residuals is not None and len(residuals) > 0
        else np.nan
    )
    diagnostics = {
        "solver": "scipy.optimize.linprog/highs",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(getattr(result, "nit", -1)),
        "objective_minimized": float(result.fun),
        "n_scenarios": int(S_objective),
        "n_objective_scenarios": int(S_objective),
        "n_risk_scenarios": int(S_risk),
        "n_assets": int(N),
        "cvar_alpha": float(config.cvar_alpha),
        "cvar_budget": float(cvar_budget),
        # negate the HiGHS marginal because it minimizes negative return
        "cvar_budget_marginal_min_objective": cvar_budget_marginal_min_objective,
        "cvar_budget_shadow_value_max_objective": (
            -cvar_budget_marginal_min_objective
            if np.isfinite(cvar_budget_marginal_min_objective)
            else np.nan
        ),
        "cvar_budget_slack": cvar_budget_slack,
        "cvar_budget_binding": bool(
            np.isfinite(cvar_budget_slack) and cvar_budget_slack <= 1e-8
        ),
        "scenario_expected_return": float(objective_scenario_returns.mean()),
        "objective_scenario_expected_return": float(
            objective_scenario_returns.mean()
        ),
        "objective_expected_return": float(objective_mean_vec @ weights.to_numpy(dtype=float)),
        "objective_mean_source": objective_mean_source,
        "scenario_var": float(risk_tail["var"]),
        "scenario_cvar": float(risk_tail["cvar"]),
        "risk_scenario_var": float(risk_tail["var"]),
        "risk_scenario_cvar": float(risk_tail["cvar"]),
        "objective_scenario_var": float(objective_tail["var"]),
        "objective_scenario_cvar": float(objective_tail["cvar"]),
        "cvar_effective_tail_count": float(
            (1.0 - config.cvar_alpha) * S_risk
        ),
        "cvar_full_tail_observations": int(
            np.floor((1.0 - config.cvar_alpha) * S_risk + 1e-14)
        ),
        "cvar_horizon": "one_month_predictive_return",
        "risk_scenario_source": risk_scenario_source,
        "full_l1_change_inside_current_universe": full_l1_inside,
        "min_weight_floor": float(min_weight_floor),
        "max_weight": float(weights.max()),
        "weight_sum": float(weights.sum()),
        "config": asdict(config),
    }
    return OptimizationResult(weights=weights, diagnostics=diagnostics)


def optimize_portfolio(
    scenarios: pd.DataFrame,
    pretrade_weights: pd.Series,
    config: OptimizerConfig,
    *,
    explicit_cvar_budget: float | None = None,
    initial_funding: bool = False,
    objective_mean: pd.Series | None = None,
    risk_scenarios: pd.DataFrame | None = None,
) -> OptimizationResult:
    """Solve the two-stage long-only mean-return/CVaR problem.
    
    scenarios supply the default objective mean. If risk_scenarios is provided,
    that separate distribution supplies the CVaR constraint, freezing the risk map
    as well as the numerical budget.
    """

    scenarios = scenarios.copy()
    scenarios.columns = scenarios.columns.astype(str)
    if risk_scenarios is not None:
        risk_scenarios = risk_scenarios.copy()
        risk_scenarios.columns = risk_scenarios.columns.astype(str)
    if explicit_cvar_budget is not None:
        cvar_budget = float(explicit_cvar_budget)
        budget_source = "caller_explicit"
    elif config.cvar_budget_mode == "explicit":
        cvar_budget = float(config.cvar_budget)
        budget_source = "config_explicit"
    else:
        budget_scenarios = scenarios if risk_scenarios is None else risk_scenarios
        cvar_budget = equal_weight_cvar(budget_scenarios, config.cvar_alpha)
        budget_source = (
            "same_scenario_equal_weight"
            if risk_scenarios is None
            else "risk_scenario_equal_weight"
        )

    stage1 = _solve_lp(
        scenarios,
        pretrade_weights,
        config,
        min_weight_floor=0.0,
        cvar_budget=cvar_budget,
        initial_funding=initial_funding,
        objective_mean=objective_mean,
        risk_scenarios=risk_scenarios,
    )
    floor = float(config.active_min_weight)
    if floor <= 0.0:
        stage1.diagnostics.update(
            {"optimizer_stage": "single_stage", "cvar_budget_source": budget_source}
        )
        return stage1

    kept = stage1.weights[stage1.weights >= floor - 1e-12].index.tolist()
    if not kept:
        kept = [str(stage1.weights.idxmax())]
    max_keep = max(1, int(np.floor(1.0 / floor)))
    if len(kept) > max_keep:
        kept = stage1.weights.sort_values(ascending=False).index[:max_keep].tolist()
    if len(kept) * config.max_weight < 1.0 - 1e-10:
        for asset in stage1.weights.sort_values(ascending=False).index:
            if asset not in kept:
                kept.append(str(asset))
            if len(kept) * config.max_weight >= 1.0 - 1e-10:
                break

    initial_stage2_n_assets = len(kept)
    stage2_expansion_attempts = 0
    ranked_assets = [str(x) for x in stage1.weights.sort_values(ascending=False).index]
    remaining = [x for x in ranked_assets if x not in kept]

    while True:
        try:
            stage2 = _solve_lp(
                scenarios.loc[:, kept],
                pretrade_weights,
                config,
                min_weight_floor=floor,
                cvar_budget=cvar_budget,
                initial_funding=initial_funding,
                objective_mean=(
                    None if objective_mean is None
                    else objective_mean.reindex(kept)
                ),
                risk_scenarios=(
                    None
                    if risk_scenarios is None
                    else risk_scenarios.loc[:, kept]
                ),
            )
            break
        except RuntimeError as exc:
            if "infeasible" not in str(exc).lower():
                raise
            available_slots = max_keep - len(kept)
            if not remaining or available_slots <= 0:
                raise
            n_add = min(5, len(remaining), available_slots)
            kept.extend(remaining[:n_add])
            remaining = remaining[n_add:]
            stage2_expansion_attempts += 1

    full = pd.Series(0.0, index=scenarios.columns, dtype=float)
    full.loc[stage2.weights.index] = stage2.weights
    stage2.weights = full
    stage2.diagnostics.update(
        {
            "optimizer_stage": "two_stage_active_floor",
            "cvar_budget_source": budget_source,
            "stage1_n_assets": int(len(stage1.weights)),
            "stage2_n_assets": int(len(kept)),
            "stage2_initial_n_assets": int(initial_stage2_n_assets),
            "stage2_expansion_attempts": int(stage2_expansion_attempts),
            "stage2_expanded_for_feasibility": bool(stage2_expansion_attempts > 0),
            "stage1_objective_minimized": stage1.diagnostics["objective_minimized"],
            "assets_dropped_after_stage1": int(len(stage1.weights) - len(kept)),
        }
    )
    return stage2


def drift_weights(
    previous_target_weights: pd.Series,
    realized_simple_returns: pd.Series,
) -> pd.Series:
    """Drift prior target weights to current pretrade weights."""

    prev = previous_target_weights.astype(float).copy()
    prev.index = prev.index.astype(str)
    returns = realized_simple_returns.copy()
    returns.index = returns.index.astype(str)
    returns = returns.reindex(prev.index)
    held = prev > EPS
    missing = held & returns.isna()
    if missing.any():
        raise ValueError(
            "Missing realized returns for previously held assets: "
            f"{returns.index[missing].tolist()}"
        )
    gross_values = prev * (1.0 + returns.fillna(0.0))
    total = float(gross_values.sum())
    if total <= EPS:
        raise ValueError("Portfolio value is nonpositive after drifting weights.")
    drifted = gross_values / total
    drifted.name = "pretrade_weight"
    return drifted


def turnover_and_cost(
    pretrade_weights: pd.Series,
    target_weights: pd.Series,
    transaction_cost_bps: float,
    *,
    initial_funding: bool = False,
) -> TurnoverResult:
    """Compute one-way turnover over the union of assets.
    
    Ongoing turnover is half the L1 change. Initial funding is a full one-way
    purchase and is flagged separately for turnover summaries.
    """

    pre = pretrade_weights.astype(float).copy()
    target = target_weights.astype(float).copy()
    pre.index = pre.index.astype(str)
    target.index = target.index.astype(str)
    union = pre.index.union(target.index)
    pre = pre.reindex(union).fillna(0.0)
    target = target.reindex(union).fillna(0.0)
    full_l1 = float((target - pre).abs().sum())
    one_way = full_l1 if initial_funding else 0.5 * full_l1
    forced_sale = float(pre.loc[pre.index.difference(target_weights.index)].sum())
    cost = float(transaction_cost_bps / 10_000.0 * one_way)
    return TurnoverResult(
        one_way_turnover=one_way,
        full_l1_change=full_l1,
        forced_sale_weight=forced_sale,
        initial_funding=bool(initial_funding),
        transaction_cost=cost,
    )


def realized_portfolio_return(
    weights: pd.Series,
    realized_returns: pd.Series,
    *,
    label: str = "simple",
) -> float:
    weights = weights.astype(float).copy()
    weights.index = weights.index.astype(str)
    realized = realized_returns.copy()
    realized.index = realized.index.astype(str)
    realized = realized.reindex(weights.index)
    held = weights > EPS
    missing = held & realized.isna()
    if missing.any():
        raise ValueError(
            f"Missing realized {label} returns for held assets: "
            f"{realized.index[missing].tolist()}"
        )
    return float(weights.loc[held].to_numpy() @ realized.loc[held].to_numpy(dtype=float))
