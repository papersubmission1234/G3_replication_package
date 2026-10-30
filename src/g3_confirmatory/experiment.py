"""Common-random-number experiments for full-information expert aggregation.

Each arm is fitted once per seed and formation date. Its scenarios are reused
across scores and actions, and all arm losses are observed counterfactually.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import joblib
import threadpoolctl

from . import __version__
from .actions import (
    anchored_baseline_weights,
    baseline_preserving_mean_mix_scenarios,
    bounded_baseline_weights,
    complementarity_sparse_weights,
    hard_arm_from_uniform,
    mean_only_overlay_scenarios,
    mix_scenarios_crn,
    posterior_calibrated_mean_box,
    row_overlay_scenarios_crn,
)
from .arms import (
    ArmResult,
    HierarchicalArm,
    JointFactorArm,
    EWMAShrinkageArm,
    LedoitWolfArm,
    MomentumArm,
    PredictiveArm,
    StudentTTailArm,
    plugin_gaussian_scenarios,
    shrink_scenario_mean,
)
from .asset_classes import style_policy
from .data_bundle import DataBundle, training_dates
from .decision_shapley import (
    all_coalitions,
    coalition_label,
    exact_shapley_pair_interactions,
    exact_shapley_values,
    grand_coalition_loo_values,
)
from .hashing import iter_tree_files, sha256_dataframe, sha256_tree
from .hierarchical_model import HierarchicalBayesianModelConfig
from .optimizer import (
    OptimizerConfig,
    drift_weights,
    empirical_cvar,
    equal_weight_cvar,
    optimize_portfolio,
    realized_portfolio_return,
    turnover_and_cost,
)
from .posterior import (
    GeneralizedPosterior,
    cvar_proxy_loss,
    diagonal_predictive_log_loss,
    quadratic_utility_loss,
    variance_only_loss,
)
from .randomness import NamedRNGFactory
from .mean_treatments import MEAN_TREATMENTS, apply_hierarchical_mean_treatment
from .contextual_gate import BayesianLinearGate, standardized_context_features
from .serialization import write_json
from .arm_cache import ArmResultCache, arm_generation_config


@dataclass(frozen=True)
class ExperimentConfig:
    training_months: int = 36
    information_lag_months: int = 1
    min_observations: int = 36
    max_missing_fraction: float = 0.0
    min_assets: int = 5
    n_scenarios: int = 500
    posterior_calibration_months: int = 24
    posterior_forgetting: float = 1.0
    posterior_eta_multiplier: float = 1.0
    posterior_initial_weights: dict[str, float] | None = None
    posterior_log_prior_scale: float = 1.0
    utility_gamma: float = 1.0
    score_names: tuple[str, ...] = ("predictive_log", "post_cost_utility")
    action_names: tuple[str, ...] = ("hard", "mix")
    include_equal_mix: bool = True
    arm_names: tuple[str, ...] = ("M_FF5MOM", "M_HierMacro")
    class_style_policy: str = "default"
    included_asset_classes: tuple[str, ...] | None = None
    hierarchical_forecast_mode: str = "mean_zero"
    hierarchical_mean_treatment: str = "base_empirical"
    arm_scenario_mode: str = "posterior_predictive"
    mean_shrinkage_fraction: float = 0.0

    overlay_baseline_arm: str = "M_HierMacro"
    overlay_signal_arm: str = "M_FF5MOM"
    overlay_fraction_mode: str = "posterior"
    overlay_fixed_fraction: float = 0.10
    overlay_max_fraction: float = 0.25
    overlay_gate_threshold: float = 0.70
    overlay_safe_cvar: bool = True
    overlay_cvar_tolerance: float = 0.0

    safe_mix_baseline_arm: str = "M_HierMacro"
    safe_mix_max_nonbaseline_weight: float = 0.20
    include_fixed_safe_mix: bool = False
    include_fixed_safe_singletons: bool = False

    anchored_shapley_action: str = "safe_mean_mix"
    anchored_shapley_max_players: int = 4
    anchored_shapley_cvar_budget_source: str = "baseline"

    sparse_redundancy_penalty: float = 1.0
    sparse_concentration_reward: float = 0.05
    sparse_minimum_arm_weight: float = 0.01

    dro_kappa: float = 1.0
    dro_entropy_weight: float = 1.0
    dro_disagreement_weight: float = 1.0
    dro_max_sigma_fraction: float = 0.50

    include_contextual_gate: bool = False
    contextual_baseline_arm: str = "M_HierMacro"
    contextual_signal_arm: str = "M_FF5MOM"
    contextual_feature_columns: tuple[str, ...] = ("VIX", "HY OAS")
    contextual_lag_months: int = 1
    contextual_prior_variance: float = 10.0
    contextual_observation_variance: float = 0.01
    contextual_minimum_weight: float = 0.05
    contextual_maximum_weight: float = 0.95
    posterior_partition: str = "pooled"
    state_calibration_months: int = 8
    state_pooling_weight: float = 0.50
    common_cvar_budget_source: str = "none"
    # separate loss and action budgets; standalone experts stay endogenous
    loss_cvar_budget_source: str = "inherit"
    action_cvar_budget_source: str = "inherit"
    # a common map fixes risk scenarios as well as the CVaR threshold
    loss_cvar_scenario_source: str = "inherit"
    action_cvar_scenario_source: str = "inherit"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initial_wealth: float = 1000.0
    apply_initial_funding_cost: bool = True
    fallback_policy: str = "fail"
    save_weights: bool = True

    def validate(self) -> None:
        if self.training_months <= 0 or self.min_observations <= 0:
            raise ValueError("Training and minimum-observation counts must be positive.")
        if self.min_observations > self.training_months:
            raise ValueError("min_observations cannot exceed training_months.")
        if not 0.0 <= self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must lie in [0, 1].")
        if self.n_scenarios < 10:
            raise ValueError("n_scenarios is too small for CVaR estimation.")
        if self.fallback_policy not in {"fail", "equal_weight"}:
            raise ValueError("fallback_policy must be fail or equal_weight.")
        supported_scores = {
            "predictive_log",
            "post_cost_utility",
            "variance_only",
            "cvar_proxy",
            "marginal_mix_utility",
            "anchored_loo_utility",
            "anchored_shapley_utility",
        }
        unknown = set(self.score_names).difference(supported_scores)
        if unknown:
            raise ValueError(f"Unknown scores: {sorted(unknown)}")
        supported_actions = {
            "hard",
            "mix",
            "mean_overlay",
            "row_overlay",
            "sparse_mix",
            "robust_mix",
            "safe_mix",
            "safe_mean_mix",
        }
        if not set(self.action_names).issubset(supported_actions):
            raise ValueError(
                f"Unknown actions: {sorted(set(self.action_names).difference(supported_actions))}"
            )
        if "marginal_mix_utility" in self.score_names and "mix" not in self.action_names:
            raise ValueError(
                "marginal_mix_utility requires the mix action because the score "
                "is defined from the implemented full mixture and leave-one-out mixtures."
            )
        if not 0.0 <= self.safe_mix_max_nonbaseline_weight <= 1.0:
            raise ValueError("safe_mix_max_nonbaseline_weight must lie in [0, 1].")
        if self.anchored_shapley_max_players < 1:
            raise ValueError("anchored_shapley_max_players must be positive.")
        safe_actions = {"safe_mix", "safe_mean_mix"}
        if safe_actions.intersection(self.action_names):
            if self.safe_mix_baseline_arm not in self.arm_names:
                raise ValueError("safe_mix_baseline_arm must be included in arm_names.")
        anchored_scores = {
            "anchored_loo_utility",
            "anchored_shapley_utility",
        }.intersection(self.score_names)
        if anchored_scores:
            if len(self.arm_names) < 2:
                raise ValueError(
                    "Anchored contribution scoring requires a baseline and at "
                    "least one optional expert."
                )
            if self.safe_mix_baseline_arm not in self.arm_names:
                raise ValueError(
                    "Anchored contribution scoring requires "
                    "safe_mix_baseline_arm in arm_names."
                )
            if self.anchored_shapley_action not in safe_actions:
                raise ValueError(
                    "anchored_shapley_action must be safe_mix or safe_mean_mix."
                )
            if self.anchored_shapley_action not in self.action_names:
                raise ValueError(
                    "anchored_shapley_action must also appear in action_names so the "
                    "score is aligned with an implemented action."
                )
            n_players = len(self.arm_names) - 1
            if (
                "anchored_shapley_utility" in anchored_scores
                and n_players > self.anchored_shapley_max_players
            ):
                raise ValueError(
                    "Exact Shapley scoring is exponential in the number of optional "
                    f"experts; got {n_players} players with maximum "
                    f"{self.anchored_shapley_max_players}."
                )
            allowed_budget_sources = {
                "baseline",
                "action_map",
                "none",
                "fixed_equal_mix",
                *self.arm_names,
            }
            if self.anchored_shapley_cvar_budget_source not in allowed_budget_sources:
                raise ValueError(
                    "anchored_shapley_cvar_budget_source must be baseline, "
                    "action_map, none, fixed_equal_mix, or an arm name."
                )
        if self.posterior_eta_multiplier <= 0.0:
            raise ValueError("posterior_eta_multiplier must be positive.")
        if self.posterior_log_prior_scale < 0.0:
            raise ValueError("posterior_log_prior_scale must be nonnegative.")
        if self.posterior_initial_weights is not None:
            unknown_prior_arms = set(self.posterior_initial_weights).difference(self.arm_names)
            if unknown_prior_arms:
                raise ValueError(
                    f"posterior_initial_weights contains unknown arms: {sorted(unknown_prior_arms)}"
                )
            if sum(max(float(v), 0.0) for v in self.posterior_initial_weights.values()) <= 0.0:
                raise ValueError("posterior_initial_weights must contain positive mass.")
        if self.arm_scenario_mode not in {"posterior_predictive", "plugin_gaussian"}:
            raise ValueError("arm_scenario_mode must be posterior_predictive or plugin_gaussian.")
        if self.overlay_fraction_mode not in {"posterior", "fixed", "threshold"}:
            raise ValueError("overlay_fraction_mode must be posterior, fixed, or threshold.")
        for value, label in (
            (self.overlay_fixed_fraction, "overlay_fixed_fraction"),
            (self.overlay_max_fraction, "overlay_max_fraction"),
            (self.overlay_gate_threshold, "overlay_gate_threshold"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1].")
        if self.overlay_cvar_tolerance < 0.0:
            raise ValueError("overlay_cvar_tolerance must be nonnegative.")
        if self.sparse_redundancy_penalty < 0.0 or self.sparse_concentration_reward < 0.0:
            raise ValueError("Sparse action penalties must be nonnegative.")
        if not 0.0 <= self.sparse_minimum_arm_weight <= 1.0:
            raise ValueError("sparse_minimum_arm_weight must lie in [0, 1].")
        if self.dro_kappa < 0.0 or self.dro_entropy_weight < 0.0 or self.dro_disagreement_weight < 0.0:
            raise ValueError("DRO calibration parameters must be nonnegative.")
        if not 0.0 <= self.dro_max_sigma_fraction <= 10.0:
            raise ValueError("dro_max_sigma_fraction must lie in [0, 10].")
        if self.include_contextual_gate:
            if self.contextual_baseline_arm not in self.arm_names or self.contextual_signal_arm not in self.arm_names:
                raise ValueError("Contextual baseline and signal arms must be in arm_names.")
            if self.contextual_baseline_arm == self.contextual_signal_arm:
                raise ValueError("Contextual baseline and signal arms must differ.")
            if not self.contextual_feature_columns:
                raise ValueError("At least one contextual_feature_column is required.")
        if self.included_asset_classes is not None:
            unknown_classes = set(self.included_asset_classes).difference(
                {"equity", "bond", "mixed", "real_estate", "commodities", "crypto", "metals", "cash_fx"}
            )
            if unknown_classes:
                raise ValueError(f"Unknown included_asset_classes: {sorted(unknown_classes)}")
        if self.hierarchical_mean_treatment not in MEAN_TREATMENTS:
            raise ValueError(
                f"Unknown hierarchical_mean_treatment: {self.hierarchical_mean_treatment}"
            )
        if not 0.0 <= self.mean_shrinkage_fraction <= 1.0:
            raise ValueError("mean_shrinkage_fraction must lie in [0, 1].")
        if self.posterior_partition not in {
            "pooled", "regime", "partially_pooled_regime"
        }:
            raise ValueError(
                "posterior_partition must be pooled, regime, or "
                "partially_pooled_regime."
            )
        if self.state_calibration_months < 0:
            raise ValueError("state_calibration_months must be nonnegative.")
        if not 0.0 <= self.state_pooling_weight <= 1.0:
            raise ValueError("state_pooling_weight must lie in [0, 1].")
        valid_budget_sources = {"inherit", "none", "fixed_equal_mix", *self.arm_names}
        for source, label in (
            (self.common_cvar_budget_source, "common_cvar_budget_source"),
            (self.loss_cvar_budget_source, "loss_cvar_budget_source"),
            (self.action_cvar_budget_source, "action_cvar_budget_source"),
            (self.loss_cvar_scenario_source, "loss_cvar_scenario_source"),
            (self.action_cvar_scenario_source, "action_cvar_scenario_source"),
        ):
            if source not in valid_budget_sources:
                raise ValueError(
                    f"{label} must be inherit, none, fixed_equal_mix, or an arm name; "
                    f"got {source!r}."
                )
        clean_map_requested = (
            self.loss_cvar_budget_source != "inherit"
            or self.action_cvar_budget_source != "inherit"
            or self.loss_cvar_scenario_source != "inherit"
            or self.action_cvar_scenario_source != "inherit"
        )
        if clean_map_requested and self.common_cvar_budget_source != "none":
            raise ValueError(
                "Do not combine common_cvar_budget_source with explicit "
                "loss_cvar_budget_source/action_cvar_budget_source."
            )


@dataclass
class StrategyState:
    name: str
    target_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    wealth: float = 1000.0
    net_excess_history: list[float] = field(default_factory=list)


@dataclass
class SeedRunResult:
    returns: pd.DataFrame
    weights: pd.DataFrame
    posterior: pd.DataFrame
    arm_losses: pd.DataFrame
    scenario_hashes: pd.DataFrame
    optimizer_diagnostics: pd.DataFrame
    arm_diagnostics: pd.DataFrame
    manifest: dict[str, Any]


def build_arms(config: ExperimentConfig) -> dict[str, PredictiveArm]:
    hier_cfg = HierarchicalBayesianModelConfig(
        min_obs=config.min_observations,
        n_draws=config.n_scenarios,
        n_iter=6,
        forecast_mode=config.hierarchical_forecast_mode,
        alpha_prior_var=1e-4,
        macro_prior_var=0.20,
        style_prior_var=0.25,
        alpha_group_mean_var=1e-4,
        macro_group_mean_var=0.20,
        style_group_mean_var=0.25,
        coef_prior_spec={"family": "gaussian"},
        class_style_families=style_policy(config.class_style_policy),
    )
    known: dict[str, PredictiveArm] = {
        "M_FF5MOM": JointFactorArm(),
        "M_HierMacro": HierarchicalArm(hier_cfg),
        "M_LW_MinVar": LedoitWolfArm(),
        "M_Momentum": MomentumArm(),
        "M_EWMA_LW": EWMAShrinkageArm(),
        "M_StudentT": StudentTTailArm(),
    }
    missing = set(config.arm_names).difference(known)
    if missing:
        raise ValueError(f"Unknown arm names: {sorted(missing)}")
    return {name: known[name] for name in config.arm_names}


def _eligible_training_panel(
    data: DataBundle,
    formation_date: pd.Timestamp,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    dates = training_dates(
        formation_date,
        training_months=config.training_months,
        information_lag_months=config.information_lag_months,
    )
    if formation_date not in data.eligibility.index:
        raise KeyError(f"Formation date missing from eligibility matrix: {formation_date}")
    declared = data.eligibility.loc[formation_date]
    tickers = declared[declared.astype(bool)].index
    if config.included_asset_classes is not None:
        allowed = set(config.included_asset_classes)
        tickers = pd.Index([
            ticker for ticker in tickers
            if data.asset_classes.get(ticker) in allowed
        ])
    panel = data.monthly_excess_returns.reindex(index=dates, columns=tickers)
    obs = panel.notna().sum(axis=0)
    missing_frac = panel.isna().mean(axis=0)
    keep = (obs >= config.min_observations) & (
        missing_frac <= config.max_missing_fraction
    )
    eligible = keep[keep].index
    if len(eligible) < config.min_assets:
        raise ValueError(
            f"Only {len(eligible)} assets pass history filters at {formation_date}; "
            f"minimum is {config.min_assets}."
        )
    return panel.loc[:, eligible], data.asset_classes.reindex(eligible), dates


def _factor_inputs(
    data: DataBundle,
    return_dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    # need one extra macro observation for the first difference
    macro_start = return_dates.min() - pd.offsets.MonthEnd(1)
    macro_dates = pd.date_range(macro_start, return_dates.max(), freq="ME")
    return {
        "equity": data.equity_factors.reindex(return_dates),
        "bond": data.bond_factors.reindex(return_dates),
        "macro": data.macro_factors.reindex(macro_dates),
    }


def _intersect_arm_scenarios(
    results: Mapping[str, ArmResult],
    min_assets: int,
) -> tuple[dict[str, pd.DataFrame], pd.Index]:
    names = list(results)
    common = pd.Index(results[names[0]].scenarios.columns.astype(str))
    for name in names[1:]:
        common = common.intersection(results[name].scenarios.columns.astype(str))
    common = common.sort_values()
    if len(common) < min_assets:
        raise ValueError(
            f"Only {len(common)} assets are common across arms; minimum is {min_assets}."
        )
    return {
        name: result.scenarios.reindex(columns=common)
        for name, result in results.items()
    }, common


def _pretrade_for_state(
    state: StrategyState,
    formation_date: pd.Timestamp,
    data: DataBundle,
    *,
    first_period: bool,
) -> pd.Series:
    if first_period or state.target_weights.empty:
        return pd.Series(dtype=float, name="pretrade_weight")
    realized_to_formation = data.outcome_simple_returns.loc[formation_date]
    return drift_weights(state.target_weights, realized_to_formation)


def _equal_weight_target(assets: pd.Index) -> pd.Series:
    if len(assets) == 0:
        raise ValueError("Cannot build equal weight over an empty universe.")
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def _optimize_or_fallback(
    scenarios: pd.DataFrame,
    pretrade: pd.Series,
    config: ExperimentConfig,
    *,
    first_period: bool,
    explicit_cvar_budget: float | None = None,
    objective_mean: pd.Series | None = None,
    risk_scenarios: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    try:
        result = optimize_portfolio(
            scenarios,
            pretrade,
            config.optimizer,
            explicit_cvar_budget=explicit_cvar_budget,
            initial_funding=first_period,
            objective_mean=objective_mean,
            risk_scenarios=risk_scenarios,
        )
        return result.weights, result.diagnostics
    except Exception as exc:
        if config.fallback_policy == "fail":
            raise
        weights = _equal_weight_target(scenarios.columns)
        return weights, {
            "fallback": True,
            "fallback_policy": "equal_weight",
            "optimization_error": repr(exc),
            "n_assets": int(len(weights)),
            "n_scenarios": int(len(scenarios)),
        }


def _execute_strategy(
    *,
    state: StrategyState,
    target: pd.Series,
    pretrade: pd.Series,
    formation_date: pd.Timestamp,
    outcome_date: pd.Timestamp,
    data: DataBundle,
    config: ExperimentConfig,
    first_period: bool,
) -> dict[str, Any]:
    turnover = turnover_and_cost(
        pretrade,
        target,
        config.optimizer.transaction_cost_bps,
        initial_funding=first_period,
    )
    if first_period and not config.apply_initial_funding_cost:
        turnover.transaction_cost = 0.0

    gross_simple = realized_portfolio_return(
        target, data.outcome_simple_returns.loc[outcome_date], label="simple"
    )
    gross_excess = realized_portfolio_return(
        target, data.outcome_excess_returns.loc[outcome_date], label="excess"
    )
    net_simple = gross_simple - turnover.transaction_cost
    net_excess = gross_excess - turnover.transaction_cost
    state.wealth *= 1.0 + net_simple
    state.target_weights = target.copy()
    state.net_excess_history.append(net_excess)
    return {
        "strategy": state.name,
        "formation_date": formation_date,
        "outcome_date": outcome_date,
        "gross_simple_return": gross_simple,
        "net_simple_return": net_simple,
        "gross_excess_return": gross_excess,
        "net_excess_return": net_excess,
        "transaction_cost": turnover.transaction_cost,
        "one_way_turnover": turnover.one_way_turnover,
        "full_l1_change": turnover.full_l1_change,
        "forced_sale_weight": turnover.forced_sale_weight,
        "initial_funding": turnover.initial_funding,
        "wealth": state.wealth,
        "n_positions": int((target > 1e-12).sum()),
        "max_weight": float(target.max()),
    }


def _strategy_names(
    config: ExperimentConfig, *, include_fixed_state_mix: bool = False
) -> list[str]:
    names = [f"standalone__{name}" for name in config.arm_names]
    names.append("benchmark__1N")
    for score in config.score_names:
        for action in config.action_names:
            names.append(f"{score}__{action}")
    if config.include_equal_mix:
        names.append("fixed_equal__mix")
    if config.include_fixed_safe_mix:
        for action in config.action_names:
            if action in {"safe_mix", "safe_mean_mix"}:
                names.append(f"fixed_safe_equal__{action}")
    if config.include_fixed_safe_singletons:
        optional = [
            arm for arm in config.arm_names
            if arm != config.safe_mix_baseline_arm
        ]
        for action in config.action_names:
            if action in {"safe_mix", "safe_mean_mix"}:
                for arm in optional:
                    names.append(f"fixed_safe_singleton__{arm}__{action}")
    if config.include_contextual_gate:
        names.append("contextual_bayes__mix")
    if "marginal_mix_utility" in config.score_names:
        for arm in config.arm_names:
            names.append(f"diagnostic__marginal_mix_utility__without__{arm}")
    if include_fixed_state_mix:
        names.append("fixed_state__mix")
    return names


def _overlay_fraction(
    posterior_weights: Mapping[str, float],
    config: ExperimentConfig,
) -> float:
    signal_weight = float(posterior_weights.get(config.overlay_signal_arm, 0.0))
    if config.overlay_fraction_mode == "fixed":
        return float(config.overlay_fixed_fraction)
    if config.overlay_fraction_mode == "threshold":
        return (
            float(config.overlay_max_fraction)
            if signal_weight >= config.overlay_gate_threshold
            else 0.0
        )
    return min(max(signal_weight, 0.0), float(config.overlay_max_fraction))


def _baseline_safe_budget(
    action_scenarios: pd.DataFrame,
    baseline_target: pd.Series,
    config: ExperimentConfig,
) -> float:
    aligned = baseline_target.reindex(action_scenarios.columns).fillna(0.0)
    total = float(aligned.sum())
    if total <= 1e-12:
        raise ValueError("Baseline target has no weight in the action universe.")
    aligned /= total
    returns = action_scenarios.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float)
    return float(
        empirical_cvar(returns, config.optimizer.cvar_alpha)
        + config.overlay_cvar_tolerance
    )


def _cvar_scenarios_from_source(
    source: str,
    *,
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: Iterable[str],
    n_scenarios: int,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
) -> pd.DataFrame | None:
    """Resolve a declared reference distribution for CVaR evaluation."""

    if source in {"none", "inherit"}:
        return None
    names = list(arm_names)
    if source == "fixed_equal_mix":
        reference_scenarios, _ = mix_scenarios_crn(
            arm_scenarios,
            {name: 1.0 / len(names) for name in names},
            n_rows=n_scenarios,
            arm_uniforms=arm_uniforms,
            row_uniforms=row_uniforms,
        )
    elif source in arm_scenarios:
        reference_scenarios = arm_scenarios[source]
    else:
        raise ValueError(
            "CVaR scenario source must be none, fixed_equal_mix, or one of "
            f"{names}; got {source!r}."
        )
    return reference_scenarios


def _cvar_budget_from_source(
    source: str,
    *,
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: Iterable[str],
    n_scenarios: int,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    alpha: float,
) -> float | None:
    """Resolve an explicit CVaR budget from a declared reference distribution."""

    reference_scenarios = _cvar_scenarios_from_source(
        source,
        arm_scenarios=arm_scenarios,
        arm_names=arm_names,
        n_scenarios=n_scenarios,
        arm_uniforms=arm_uniforms,
        row_uniforms=row_uniforms,
    )
    if reference_scenarios is None:
        return None
    return equal_weight_cvar(reference_scenarios, alpha)


def _anchored_shapley_cvar_budget(
    *,
    config: ExperimentConfig,
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: list[str],
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    action_explicit_cvar_budget: float | None,
) -> float | None:
    """Resolve the risk map shared by the Shapley action and coalitions."""

    source = config.anchored_shapley_cvar_budget_source
    if source == "baseline":
        source = config.safe_mix_baseline_arm
    if source == "action_map":
        return action_explicit_cvar_budget
    return _cvar_budget_from_source(
        source,
        arm_scenarios=arm_scenarios,
        arm_names=arm_names,
        n_scenarios=config.n_scenarios,
        arm_uniforms=arm_uniforms,
        row_uniforms=row_uniforms,
        alpha=config.optimizer.cvar_alpha,
    )


def _anchored_shapley_cvar_scenarios(
    *,
    config: ExperimentConfig,
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: list[str],
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    action_cvar_scenarios: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Resolve the full CVaR reference distribution shared by coalitions."""

    source = config.anchored_shapley_cvar_budget_source
    if source == "baseline":
        source = config.safe_mix_baseline_arm
    if source == "action_map":
        return action_cvar_scenarios
    return _cvar_scenarios_from_source(
        source,
        arm_scenarios=arm_scenarios,
        arm_names=arm_names,
        n_scenarios=config.n_scenarios,
        arm_uniforms=arm_uniforms,
        row_uniforms=row_uniforms,
    )


def _safe_action_scenarios(
    action: str,
    *,
    arm_scenarios: Mapping[str, pd.DataFrame],
    posterior_weights: Mapping[str, float],
    arm_names: list[str],
    config: ExperimentConfig,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    active_nonbaseline: set[str] | frozenset[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    """Build a HierMacro-anchored action with optional experts capped ex ante."""

    baseline = config.safe_mix_baseline_arm
    adjusted = anchored_baseline_weights(
        posterior_weights,
        arm_names,
        baseline_arm=baseline,
        max_nonbaseline_weight=config.safe_mix_max_nonbaseline_weight,
        active_nonbaseline=active_nonbaseline,
    )
    if action == "safe_mix":
        action_scenarios, action_diag = mix_scenarios_crn(
            arm_scenarios,
            adjusted,
            n_rows=config.n_scenarios,
            arm_uniforms=arm_uniforms,
            row_uniforms=row_uniforms,
        )
    elif action == "safe_mean_mix":
        action_scenarios, action_diag = baseline_preserving_mean_mix_scenarios(
            arm_scenarios,
            adjusted,
            baseline_arm=baseline,
        )
    else:
        raise ValueError(f"Unsupported safe action: {action!r}")
    action_diag.update(
        {
            "action": action,
            "safe_baseline_arm": baseline,
            "safe_max_nonbaseline_weight": config.safe_mix_max_nonbaseline_weight,
            "safe_weights_json": json.dumps(adjusted, sort_keys=True),
            "safe_active_nonbaseline_json": json.dumps(
                sorted(active_nonbaseline) if active_nonbaseline is not None else None
            ),
        }
    )
    return action_scenarios, action_diag, adjusted


def _counterfactual_net_excess_return(
    *,
    target: pd.Series,
    pretrade: pd.Series,
    outcome_date: pd.Timestamp,
    data: DataBundle,
    config: ExperimentConfig,
    first_period: bool,
) -> tuple[float, dict[str, float]]:
    """Evaluate an action without changing the portfolio state.
    
    Coalitions share the implemented strategy's pretrade holdings.
    """

    turnover = turnover_and_cost(
        pretrade,
        target,
        config.optimizer.transaction_cost_bps,
        initial_funding=first_period,
    )
    transaction_cost = float(turnover.transaction_cost)
    if first_period and not config.apply_initial_funding_cost:
        transaction_cost = 0.0
    gross_excess = realized_portfolio_return(
        target,
        data.outcome_excess_returns.loc[outcome_date],
        label="excess",
    )
    net_excess = float(gross_excess - transaction_cost)
    return net_excess, {
        "counterfactual_transaction_cost": transaction_cost,
        "counterfactual_one_way_turnover": float(turnover.one_way_turnover),
        "counterfactual_full_l1_change": float(turnover.full_l1_change),
        "counterfactual_forced_sale_weight": float(turnover.forced_sale_weight),
    }


def _anchored_coalition_values(
    *,
    coalitions: Iterable[frozenset[str]],
    posterior_weights: Mapping[str, float],
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: list[str],
    pretrade: pd.Series,
    outcome_date: pd.Timestamp,
    data: DataBundle,
    config: ExperimentConfig,
    first_period: bool,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    explicit_cvar_budget: float | None,
    risk_scenarios: pd.DataFrame | None,
) -> tuple[dict[frozenset[str], float], list[dict[str, Any]]]:
    """Evaluate coalition actions from common holdings and one risk map."""

    coalition_values: dict[frozenset[str], float] = {}
    diagnostics: list[dict[str, Any]] = []
    ordered_coalitions = sorted(
        {frozenset(coalition) for coalition in coalitions},
        key=lambda coalition: (len(coalition), sorted(coalition)),
    )
    for coalition in ordered_coalitions:
        action_scenarios, action_diag, adjusted = _safe_action_scenarios(
            config.anchored_shapley_action,
            arm_scenarios=arm_scenarios,
            posterior_weights=posterior_weights,
            arm_names=arm_names,
            config=config,
            arm_uniforms=arm_uniforms,
            row_uniforms=row_uniforms,
            active_nonbaseline=coalition,
        )
        target, opt_diag = _optimize_or_fallback(
            action_scenarios,
            pretrade,
            config,
            first_period=first_period,
            explicit_cvar_budget=explicit_cvar_budget,
            risk_scenarios=risk_scenarios,
        )
        net_excess, return_diag = _counterfactual_net_excess_return(
            target=target,
            pretrade=pretrade,
            outcome_date=outcome_date,
            data=data,
            config=config,
            first_period=first_period,
        )
        utility = float(
            -quadratic_utility_loss(net_excess, gamma=config.utility_gamma)
        )
        coalition_values[coalition] = utility
        diagnostics.append(
            {
                "coalition": coalition_label(coalition),
                "coalition_size": len(coalition),
                "coalition_utility": utility,
                "coalition_net_excess_return": net_excess,
                "coalition_weights_json": json.dumps(adjusted, sort_keys=True),
                **action_diag,
                **opt_diag,
                **return_diag,
            }
        )
    return coalition_values, diagnostics


def _anchored_shapley_losses(
    *,
    posterior_weights: Mapping[str, float],
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: list[str],
    pretrade: pd.Series,
    outcome_date: pd.Timestamp,
    data: DataBundle,
    config: ExperimentConfig,
    first_period: bool,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    explicit_cvar_budget: float | None,
    risk_scenarios: pd.DataFrame | None,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    dict[str, float],
    dict[tuple[str, str], float],
]:
    """Compute optional experts' Shapley credit around a mandatory baseline.
    
    Values are post-cost quadratic utility. Every coalition starts from the same
    holdings and uses the same numerical CVaR budget.
    """

    baseline = config.safe_mix_baseline_arm
    players = tuple(name for name in arm_names if name != baseline)
    coalition_values, diagnostics = _anchored_coalition_values(
        coalitions=all_coalitions(players),
        posterior_weights=posterior_weights,
        arm_scenarios=arm_scenarios,
        arm_names=arm_names,
        pretrade=pretrade,
        outcome_date=outcome_date,
        data=data,
        config=config,
        first_period=first_period,
        arm_uniforms=arm_uniforms,
        row_uniforms=row_uniforms,
        explicit_cvar_budget=explicit_cvar_budget,
        risk_scenarios=risk_scenarios,
    )

    shapley = exact_shapley_values(players, coalition_values)
    interactions = exact_shapley_pair_interactions(players, coalition_values)
    losses = {baseline: 0.0}
    losses.update({player: -float(value) for player, value in shapley.items()})
    total_increment = float(
        coalition_values[frozenset(players)] - coalition_values[frozenset()]
    )
    efficiency_gap = float(sum(shapley.values()) - total_increment)
    summary = {
        "coalition_value_empty": float(coalition_values[frozenset()]),
        "coalition_value_full": float(coalition_values[frozenset(players)]),
        "shapley_total_increment": total_increment,
        "shapley_efficiency_gap": efficiency_gap,
    }
    return losses, diagnostics, summary, interactions


def _anchored_loo_losses(
    *,
    posterior_weights: Mapping[str, float],
    arm_scenarios: Mapping[str, pd.DataFrame],
    arm_names: list[str],
    pretrade: pd.Series,
    outcome_date: pd.Timestamp,
    data: DataBundle,
    config: ExperimentConfig,
    first_period: bool,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
    explicit_cvar_budget: float | None,
    risk_scenarios: pd.DataFrame | None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float]]:
    """Grand-coalition leave-one-out credit under the anchored action."""

    baseline = config.safe_mix_baseline_arm
    players = tuple(name for name in arm_names if name != baseline)
    full = frozenset(players)
    coalitions = {frozenset(), full}
    coalitions.update(full.difference({player}) for player in players)
    coalition_values, diagnostics = _anchored_coalition_values(
        coalitions=coalitions,
        posterior_weights=posterior_weights,
        arm_scenarios=arm_scenarios,
        arm_names=arm_names,
        pretrade=pretrade,
        outcome_date=outcome_date,
        data=data,
        config=config,
        first_period=first_period,
        arm_uniforms=arm_uniforms,
        row_uniforms=row_uniforms,
        explicit_cvar_budget=explicit_cvar_budget,
        risk_scenarios=risk_scenarios,
    )
    contributions = grand_coalition_loo_values(players, coalition_values)
    losses = {baseline: 0.0}
    losses.update(
        {player: -float(value) for player, value in contributions.items()}
    )
    total_increment = float(
        coalition_values[full] - coalition_values[frozenset()]
    )
    contribution_sum = float(sum(contributions.values()))
    summary = {
        "coalition_value_empty": float(coalition_values[frozenset()]),
        "coalition_value_full": float(coalition_values[full]),
        "loo_total_increment": total_increment,
        "loo_sum_contributions": contribution_sum,
        "loo_allocation_gap": contribution_sum - total_increment,
    }
    return losses, diagnostics, summary


def run_seed(
    data: DataBundle,
    *,
    master_seed: int,
    config: ExperimentConfig | None = None,
    formation_states: pd.Series | None = None,
    fixed_state_weights: Mapping[str, Mapping[str, float]] | None = None,
    arm_cache: ArmResultCache | None = None,
    source_tree_sha256: str | None = None,
    source_tree_file_count: int | None = None,
) -> SeedRunResult:
    config = ExperimentConfig() if config is None else config
    config.validate()
    factory = NamedRNGFactory(master_seed)
    arms = build_arms(config)
    arm_names = list(arms)

    if formation_states is not None:
        formation_states = formation_states.copy()
        formation_states.index = (
            pd.to_datetime(formation_states.index).to_period("M").to_timestamp("M")
        )
        formation_states = formation_states.astype(str).reindex(data.formation_dates)
        if formation_states.isna().any():
            missing = formation_states[formation_states.isna()].index.tolist()
            raise ValueError(f"Missing formation-state labels: {missing[:10]}")
    if config.posterior_partition != "pooled" and formation_states is None:
        raise ValueError(
            "Regime posterior partitions require formation_states indexed by "
            "formation month-end."
        )

    include_fixed_state_mix = fixed_state_weights is not None
    states = {
        name: StrategyState(name=name, wealth=config.initial_wealth)
        for name in _strategy_names(
            config, include_fixed_state_mix=include_fixed_state_mix
        )
    }
    clean_risk_map = (
        config.loss_cvar_budget_source != "inherit"
        or config.action_cvar_budget_source != "inherit"
        or config.loss_cvar_scenario_source != "inherit"
        or config.action_cvar_scenario_source != "inherit"
    )
    decision_loss_needed = bool(
        {"post_cost_utility", "cvar_proxy"}.intersection(config.score_names)
        or config.include_contextual_gate
    )
    # counterfactual paths update scores; they are not investable portfolios
    loss_states = (
        {
            arm: StrategyState(
                name=f"diagnostic__loss_map__{arm}",
                wealth=config.initial_wealth,
            )
            for arm in arm_names
        }
        if clean_risk_map and decision_loss_needed
        else {}
    )
    pooled_posteriors = {
        score: GeneralizedPosterior(
            tuple(arm_names),
            calibration_months=config.posterior_calibration_months,
            forgetting=config.posterior_forgetting,
            eta_multiplier=config.posterior_eta_multiplier,
            initial_weights=config.posterior_initial_weights,
            log_prior_scale=config.posterior_log_prior_scale,
        )
        for score in config.score_names
    }
    state_posteriors: dict[tuple[str, str], GeneralizedPosterior] = {}
    contextual_gate = (
        BayesianLinearGate(
            n_features=1 + len(config.contextual_feature_columns),
            prior_variance=config.contextual_prior_variance,
            observation_variance=config.contextual_observation_variance,
            minimum_weight=config.contextual_minimum_weight,
            maximum_weight=config.contextual_maximum_weight,
        )
        if config.include_contextual_gate
        else None
    )
    if formation_states is not None:
        for regime in sorted(formation_states.unique()):
            for score in config.score_names:
                state_posteriors[(score, regime)] = GeneralizedPosterior(
                    tuple(arm_names),
                    calibration_months=config.state_calibration_months,
                    forgetting=config.posterior_forgetting,
                    eta_multiplier=config.posterior_eta_multiplier,
                    initial_weights=config.posterior_initial_weights,
                    log_prior_scale=config.posterior_log_prior_scale,
                )

    def action_posterior(score: str, regime: str | None) -> tuple[dict[str, float], dict[str, Any]]:
        if config.posterior_partition == "pooled":
            posterior = pooled_posteriors[score]
            return posterior.weights(), {
                "posterior_scope": "pooled",
                "posterior_state": posterior.state(),
            }
        if regime is None:
            raise AssertionError("A state is required for a regime posterior.")
        state_posterior = state_posteriors[(score, regime)]
        if config.posterior_partition == "regime":
            return state_posterior.weights(), {
                "posterior_scope": "regime",
                "posterior_state": state_posterior.state(),
            }
        pooled = pooled_posteriors[score].weights()
        local = state_posterior.weights()
        lam = config.state_pooling_weight
        blended = {
            arm: (1.0 - lam) * pooled[arm] + lam * local[arm]
            for arm in arm_names
        }
        total = sum(blended.values())
        blended = {arm: value / total for arm, value in blended.items()}
        return blended, {
            "posterior_scope": "partially_pooled_regime",
            "posterior_state": state_posterior.state(),
            "pooled_posterior_state": pooled_posteriors[score].state(),
        }

    return_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    posterior_records: list[dict[str, Any]] = []
    loss_records: list[dict[str, Any]] = []
    scenario_hash_records: list[dict[str, Any]] = []
    optimizer_records: list[dict[str, Any]] = []
    arm_diag_records: list[dict[str, Any]] = []
    start_time = time.time()

    for period_idx, row in enumerate(data.calendar.itertuples(index=False)):
        formation = pd.Timestamp(row.formation_date)
        outcome = pd.Timestamp(row.outcome_date)
        first_period = period_idx == 0
        regime = (
            str(formation_states.loc[formation])
            if formation_states is not None
            else None
        )
        training_panel, classes, return_dates = _eligible_training_panel(
            data, formation, config
        )
        factor_inputs = _factor_inputs(data, return_dates)

        arm_results: dict[str, ArmResult] = {}
        raw_generation_config = arm_generation_config(config)
        for arm_name, arm in arms.items():
            def _compute_raw_arm_result(
                *, _arm=arm, _arm_name=arm_name
            ) -> ArmResult:
                return _arm.fit_generate(
                    excess_returns=training_panel,
                    factors=factor_inputs,
                    asset_classes=classes,
                    n_scenarios=config.n_scenarios,
                    rng=factory.generator(
                        "arm", _arm_name, formation.date()
                    ),
                )

            if arm_cache is None:
                result = _compute_raw_arm_result()
                cache_hit = False
                cache_digest = None
            else:
                cache_payload = arm_cache.key_payload(
                    master_seed=master_seed,
                    formation_date=formation,
                    arm_name=arm_name,
                    n_scenarios=config.n_scenarios,
                    generation_config=raw_generation_config,
                )
                result, cache_hit, cache_digest = arm_cache.get_or_compute(
                    cache_payload,
                    _compute_raw_arm_result,
                )
            result.diagnostics = {
                **result.diagnostics,
                "raw_arm_cache_hit": bool(cache_hit),
                "raw_arm_cache_digest": cache_digest,
            }
            if arm_name == "M_HierMacro":
                treatment = apply_hierarchical_mean_treatment(
                    result,
                    config.hierarchical_mean_treatment,
                    rng=factory.generator(
                        "mean_treatment",
                        config.hierarchical_mean_treatment,
                        formation.date(),
                    ),
                )
                result.scenarios = treatment.scenarios
                result.diagnostics = {**result.diagnostics, **treatment.diagnostics}
            if config.mean_shrinkage_fraction > 0.0:
                result.scenarios = shrink_scenario_mean(
                    result.scenarios, config.mean_shrinkage_fraction
                )
                result.diagnostics = {
                    **result.diagnostics,
                    "mean_shrinkage_fraction": config.mean_shrinkage_fraction,
                }
            if config.arm_scenario_mode == "plugin_gaussian":
                result = plugin_gaussian_scenarios(
                    result,
                    n_scenarios=config.n_scenarios,
                    rng=factory.generator(
                        "plugin_scenarios", arm_name, formation.date()
                    ),
                )
            arm_results[arm_name] = result
            arm_diag_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "arm": arm_name,
                    **{
                        key: value
                        for key, value in result.diagnostics.items()
                        if np.isscalar(value) or value is None
                    },
                }
            )

        arm_scenarios, common_assets = _intersect_arm_scenarios(
            arm_results, config.min_assets
        )
        for arm_name, scenarios in arm_scenarios.items():
            scenario_hash_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "arm": arm_name,
                    "scenario_sha256": sha256_dataframe(scenarios),
                    "n_rows": scenarios.shape[0],
                    "n_assets": scenarios.shape[1],
                }
            )

        # form the action posterior before observing the outcome
        action_weights_by_score: dict[str, dict[str, float]] = {}
        for score in config.score_names:
            action_weights, posterior_meta = action_posterior(score, regime)
            action_weights_by_score[score] = action_weights
            state = posterior_meta["posterior_state"]
            for arm_name, weight in action_weights.items():
                posterior_records.append(
                    {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "regime": regime,
                        "score": score,
                        "posterior_scope": posterior_meta["posterior_scope"],
                        "arm": arm_name,
                        "weight": weight,
                        "entropy": float(
                            -sum(
                                p * np.log(max(p, 1e-15))
                                for p in action_weights.values()
                            )
                        ),
                        "eta_effective": state["eta_effective"],
                        "n_updates": state["n_updates"],
                    }
                )

        pretrade_by_strategy = {
            name: _pretrade_for_state(
                state, formation, data, first_period=first_period
            )
            for name, state in states.items()
        }
        loss_pretrade_by_arm = {
            arm: _pretrade_for_state(
                state, formation, data, first_period=first_period
            )
            for arm, state in loss_states.items()
        }

        hard_uniform = float(
            factory.generator("hard_action", formation.date()).random()
        )
        arm_uniforms = factory.generator(
            "mix_labels", formation.date()
        ).random(config.n_scenarios)
        row_uniforms = factory.generator(
            "mix_row_resample", formation.date()
        ).random(config.n_scenarios)

        shared_explicit_cvar_budget = _cvar_budget_from_source(
            config.common_cvar_budget_source,
            arm_scenarios=arm_scenarios,
            arm_names=arm_names,
            n_scenarios=config.n_scenarios,
            arm_uniforms=arm_uniforms,
            row_uniforms=row_uniforms,
            alpha=config.optimizer.cvar_alpha,
        )
        if clean_risk_map:
            standalone_explicit_cvar_budget = None
            loss_explicit_cvar_budget = _cvar_budget_from_source(
                config.loss_cvar_budget_source,
                arm_scenarios=arm_scenarios,
                arm_names=arm_names,
                n_scenarios=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
                alpha=config.optimizer.cvar_alpha,
            )
            action_explicit_cvar_budget = _cvar_budget_from_source(
                config.action_cvar_budget_source,
                arm_scenarios=arm_scenarios,
                arm_names=arm_names,
                n_scenarios=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
                alpha=config.optimizer.cvar_alpha,
            )
            loss_cvar_scenarios = _cvar_scenarios_from_source(
                config.loss_cvar_scenario_source,
                arm_scenarios=arm_scenarios,
                arm_names=arm_names,
                n_scenarios=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
            )
            action_cvar_scenarios = _cvar_scenarios_from_source(
                config.action_cvar_scenario_source,
                arm_scenarios=arm_scenarios,
                arm_names=arm_names,
                n_scenarios=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
            )
        else:
            standalone_explicit_cvar_budget = shared_explicit_cvar_budget
            loss_explicit_cvar_budget = shared_explicit_cvar_budget
            action_explicit_cvar_budget = shared_explicit_cvar_budget
            loss_cvar_scenarios = None
            action_cvar_scenarios = None
        anchored_credit_scores = {
            "anchored_loo_utility",
            "anchored_shapley_utility",
        }.intersection(config.score_names)
        anchored_credit_explicit_cvar_budget = (
            _anchored_shapley_cvar_budget(
                config=config,
                arm_scenarios=arm_scenarios,
                arm_names=arm_names,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
                action_explicit_cvar_budget=action_explicit_cvar_budget,
            )
            if anchored_credit_scores
            else None
        )
        anchored_credit_cvar_scenarios = (
            _anchored_shapley_cvar_scenarios(
                config=config,
                arm_scenarios=arm_scenarios,
                arm_names=arm_names,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
                action_cvar_scenarios=action_cvar_scenarios,
            )
            if anchored_credit_scores
            else None
        )

        arm_period_performance: dict[str, dict[str, Any]] = {}
        arm_period_targets: dict[str, pd.Series] = {}
        for arm_name in arm_names:
            strategy_name = f"standalone__{arm_name}"
            target, diag = _optimize_or_fallback(
                arm_scenarios[arm_name],
                pretrade_by_strategy[strategy_name],
                config,
                first_period=first_period,
                explicit_cvar_budget=standalone_explicit_cvar_budget,
            )
            perf = _execute_strategy(
                state=states[strategy_name],
                target=target,
                pretrade=pretrade_by_strategy[strategy_name],
                formation_date=formation,
                outcome_date=outcome,
                data=data,
                config=config,
                first_period=first_period,
            )
            arm_period_performance[arm_name] = perf
            arm_period_targets[arm_name] = target.copy()
            return_records.append({"master_seed": master_seed, "regime": regime, **perf})
            optimizer_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "strategy": strategy_name,
                    **diag,
                }
            )
            if config.save_weights:
                for ticker, weight in target[target > 0.0].items():
                    weight_records.append(
                        {
                            "master_seed": master_seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": strategy_name,
                            "ticker": ticker,
                            "weight": weight,
                        }
                    )

        # freeze the loss-risk map while varying action risk
        loss_arm_period_performance = arm_period_performance
        if loss_states:
            loss_arm_period_performance = {}
            for arm_name in arm_names:
                target, diag = _optimize_or_fallback(
                    arm_scenarios[arm_name],
                    loss_pretrade_by_arm[arm_name],
                    config,
                    first_period=first_period,
                    explicit_cvar_budget=loss_explicit_cvar_budget,
                    risk_scenarios=loss_cvar_scenarios,
                )
                perf = _execute_strategy(
                    state=loss_states[arm_name],
                    target=target,
                    pretrade=loss_pretrade_by_arm[arm_name],
                    formation_date=formation,
                    outcome_date=outcome,
                    data=data,
                    config=config,
                    first_period=first_period,
                )
                loss_arm_period_performance[arm_name] = perf
                optimizer_records.append(
                    {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "strategy": f"diagnostic__loss_map__{arm_name}",
                        **diag,
                        "diagnostic_only": True,
                        "loss_cvar_budget_source_config": config.loss_cvar_budget_source,
                        "action_cvar_budget_source_config": config.action_cvar_budget_source,
                        "loss_cvar_scenario_source_config": config.loss_cvar_scenario_source,
                        "action_cvar_scenario_source_config": config.action_cvar_scenario_source,
                    }
                )

        benchmark_name = "benchmark__1N"
        benchmark_target = _equal_weight_target(common_assets)
        benchmark_perf = _execute_strategy(
            state=states[benchmark_name],
            target=benchmark_target,
            pretrade=pretrade_by_strategy[benchmark_name],
            formation_date=formation,
            outcome_date=outcome,
            data=data,
            config=config,
            first_period=first_period,
        )
        return_records.append({"master_seed": master_seed, "regime": regime, **benchmark_perf})
        if config.save_weights:
            for ticker, weight in benchmark_target.items():
                weight_records.append(
                    {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "outcome_date": outcome,
                        "strategy": benchmark_name,
                        "ticker": ticker,
                        "weight": weight,
                    }
                )

        strategy_period_performance: dict[str, dict[str, Any]] = {}
        for score in config.score_names:
            weights = action_weights_by_score[score]
            for action in config.action_names:
                strategy_name = f"{score}__{action}"
                action_diag: dict[str, Any]
                objective_mean: pd.Series | None = None
                action_budget = action_explicit_cvar_budget
                if score in anchored_credit_scores and action == config.anchored_shapley_action:
                    action_budget = anchored_credit_explicit_cvar_budget

                if action == "hard":
                    selected = hard_arm_from_uniform(weights, arm_names, hard_uniform)
                    action_scenarios = arm_scenarios[selected]
                    action_diag = {
                        "action": "hard",
                        "selected_arm": selected,
                        "hard_uniform": hard_uniform,
                    }
                elif action == "mix":
                    action_scenarios, action_diag = mix_scenarios_crn(
                        arm_scenarios,
                        weights,
                        n_rows=config.n_scenarios,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                    )
                    action_diag["action"] = "mix"
                elif action in {"safe_mix", "safe_mean_mix"}:
                    action_scenarios, action_diag, _ = _safe_action_scenarios(
                        action,
                        arm_scenarios=arm_scenarios,
                        posterior_weights=weights,
                        arm_names=arm_names,
                        config=config,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                    )
                elif action in {"mean_overlay", "row_overlay"}:
                    baseline = config.overlay_baseline_arm
                    signal = config.overlay_signal_arm
                    if baseline not in arm_scenarios or signal not in arm_scenarios:
                        raise ValueError(
                            "Overlay actions require baseline and signal arms in arm_names: "
                            f"{baseline!r}, {signal!r}."
                        )
                    fraction = _overlay_fraction(weights, config)
                    if action == "mean_overlay":
                        action_scenarios, action_diag = mean_only_overlay_scenarios(
                            arm_scenarios[baseline], arm_scenarios[signal], fraction
                        )
                    else:
                        action_scenarios, action_diag = row_overlay_scenarios_crn(
                            arm_scenarios,
                            baseline_arm=baseline,
                            signal_arm=signal,
                            fraction=fraction,
                            n_rows=config.n_scenarios,
                            arm_uniforms=arm_uniforms,
                            row_uniforms=row_uniforms,
                        )
                    action_diag.update(
                        {
                            "action": action,
                            "overlay_fraction_mode": config.overlay_fraction_mode,
                            "overlay_posterior_signal_weight": float(weights.get(signal, 0.0)),
                        }
                    )
                    if config.overlay_safe_cvar:
                        action_budget = _baseline_safe_budget(
                            action_scenarios,
                            arm_period_targets[baseline],
                            config,
                        )
                        action_diag["safe_cvar_budget"] = action_budget
                        action_diag["safe_cvar_baseline"] = baseline
                elif action == "sparse_mix":
                    adjusted, sparse_diag = complementarity_sparse_weights(
                        weights,
                        arm_scenarios,
                        redundancy_penalty=config.sparse_redundancy_penalty,
                        concentration_reward=config.sparse_concentration_reward,
                        minimum_reported_weight=config.sparse_minimum_arm_weight,
                    )
                    action_scenarios, action_diag = mix_scenarios_crn(
                        arm_scenarios,
                        adjusted,
                        n_rows=config.n_scenarios,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                    )
                    action_diag.update(
                        {
                            "action": "sparse_mix",
                            "sparse_effective_arm_count": sparse_diag["effective_arm_count"],
                            "sparse_optimizer_success": sparse_diag["optimizer_success"],
                            "sparse_weights_json": json.dumps(adjusted, sort_keys=True),
                        }
                    )
                elif action == "robust_mix":
                    action_scenarios, action_diag = mix_scenarios_crn(
                        arm_scenarios,
                        weights,
                        n_rows=config.n_scenarios,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                    )
                    objective_mean, radius, dro_diag = posterior_calibrated_mean_box(
                        action_scenarios,
                        arm_scenarios,
                        weights,
                        kappa=config.dro_kappa,
                        entropy_weight=config.dro_entropy_weight,
                        disagreement_weight=config.dro_disagreement_weight,
                        max_sigma_fraction=config.dro_max_sigma_fraction,
                    )
                    action_diag.update(
                        {
                            "action": "robust_mix",
                            "dro_mean_radius_average": dro_diag["mean_radius_average"],
                            "dro_mean_radius_max": dro_diag["mean_radius_max"],
                            "dro_posterior_entropy_normalized": dro_diag[
                                "posterior_entropy_normalized"
                            ],
                            "dro_radius_sha256": sha256_dataframe(
                                radius.to_frame("mean_radius")
                            ),
                        }
                    )
                else:
                    raise AssertionError(action)

                target, opt_diag = _optimize_or_fallback(
                    action_scenarios,
                    pretrade_by_strategy[strategy_name],
                    config,
                    first_period=first_period,
                    explicit_cvar_budget=action_budget,
                    objective_mean=objective_mean,
                    risk_scenarios=action_cvar_scenarios,
                )
                perf = _execute_strategy(
                    state=states[strategy_name],
                    target=target,
                    pretrade=pretrade_by_strategy[strategy_name],
                    formation_date=formation,
                    outcome_date=outcome,
                    data=data,
                    config=config,
                    first_period=first_period,
                )
                strategy_period_performance[strategy_name] = perf
                return_records.append({"master_seed": master_seed, "regime": regime, **perf})
                optimizer_record = {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "strategy": strategy_name,
                    **opt_diag,
                }
                for key, value in action_diag.items():
                    if np.isscalar(value) or value is None or isinstance(value, str):
                        optimizer_record[key] = value
                optimizer_records.append(optimizer_record)
                if config.save_weights:
                    for ticker, weight in target[target > 0.0].items():
                        weight_records.append(
                            {
                                "master_seed": master_seed,
                                "formation_date": formation,
                                "outcome_date": outcome,
                                "strategy": strategy_name,
                                "ticker": ticker,
                                "weight": weight,
                            }
                        )


        if config.include_fixed_safe_mix or config.include_fixed_safe_singletons:
            baseline = config.safe_mix_baseline_arm
            optional = [name for name in arm_names if name != baseline]
            for action in config.action_names:
                if action not in {"safe_mix", "safe_mean_mix"}:
                    continue
                fixed_budget = action_explicit_cvar_budget
                if anchored_credit_scores and action == config.anchored_shapley_action:
                    fixed_budget = anchored_credit_explicit_cvar_budget

                controls: list[
                    tuple[str, dict[str, float], set[str] | None, str | None]
                ] = []
                if config.include_fixed_safe_mix:
                    controls.append(
                        (
                            f"fixed_safe_equal__{action}",
                            {name: 1.0 / len(arm_names) for name in arm_names},
                            None,
                            None,
                        )
                    )
                if config.include_fixed_safe_singletons:
                    for arm in optional:
                        raw = {name: 0.0 for name in arm_names}
                        raw[arm] = 1.0
                        controls.append(
                            (
                                f"fixed_safe_singleton__{arm}__{action}",
                                raw,
                                {arm},
                                arm,
                            )
                        )

                for strategy_name, raw_weights, active, singleton_arm in controls:
                    action_scenarios, action_diag, _ = _safe_action_scenarios(
                        action,
                        arm_scenarios=arm_scenarios,
                        posterior_weights=raw_weights,
                        arm_names=arm_names,
                        config=config,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                        active_nonbaseline=active,
                    )
                    target, opt_diag = _optimize_or_fallback(
                        action_scenarios,
                        pretrade_by_strategy[strategy_name],
                        config,
                        first_period=first_period,
                        explicit_cvar_budget=fixed_budget,
                        risk_scenarios=action_cvar_scenarios,
                    )
                    perf = _execute_strategy(
                        state=states[strategy_name],
                        target=target,
                        pretrade=pretrade_by_strategy[strategy_name],
                        formation_date=formation,
                        outcome_date=outcome,
                        data=data,
                        config=config,
                        first_period=first_period,
                    )
                    return_records.append(
                        {"master_seed": master_seed, "regime": regime, **perf}
                    )
                    record = {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "strategy": strategy_name,
                        **opt_diag,
                        "fixed_safe_control": True,
                        "fixed_safe_equal": singleton_arm is None,
                        "fixed_safe_singleton_arm": singleton_arm,
                        "fixed_safe_cvar_budget": fixed_budget,
                    }
                    for key, value in action_diag.items():
                        if (
                            np.isscalar(value)
                            or value is None
                            or isinstance(value, str)
                        ):
                            record[key] = value
                    optimizer_records.append(record)
                    if config.save_weights:
                        for ticker, weight in target[target > 0.0].items():
                            weight_records.append(
                                {
                                    "master_seed": master_seed,
                                    "formation_date": formation,
                                    "outcome_date": outcome,
                                    "strategy": strategy_name,
                                    "ticker": ticker,
                                    "weight": weight,
                                }
                            )

        contextual_x: np.ndarray | None = None
        contextual_prediction: dict[str, float] | None = None
        if contextual_gate is not None:
            contextual_x, context_meta = standardized_context_features(
                data.macro_factors,
                formation,
                config.contextual_feature_columns,
                lag_months=config.contextual_lag_months,
            )
            contextual_prediction = contextual_gate.predict(contextual_x)
            baseline = config.contextual_baseline_arm
            signal = config.contextual_signal_arm
            contextual_weights = {
                baseline: contextual_prediction["baseline_weight"],
                signal: contextual_prediction["signal_weight"],
            }
            action_scenarios, gate_mix_diag = mix_scenarios_crn(
                arm_scenarios,
                contextual_weights,
                n_rows=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
            )
            strategy_name = "contextual_bayes__mix"
            target, opt_diag = _optimize_or_fallback(
                action_scenarios,
                pretrade_by_strategy[strategy_name],
                config,
                first_period=first_period,
                explicit_cvar_budget=action_explicit_cvar_budget,
                risk_scenarios=action_cvar_scenarios,
            )
            perf = _execute_strategy(
                state=states[strategy_name],
                target=target,
                pretrade=pretrade_by_strategy[strategy_name],
                formation_date=formation,
                outcome_date=outcome,
                data=data,
                config=config,
                first_period=first_period,
            )
            strategy_period_performance[strategy_name] = perf
            return_records.append({"master_seed": master_seed, "regime": regime, **perf})
            optimizer_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "strategy": strategy_name,
                    **opt_diag,
                    "action": "contextual_mix",
                    "context_source_date": context_meta["source_date"],
                    "context_history_count": context_meta["history_count"],
                    "context_probability_baseline_better": contextual_prediction[
                        "probability_baseline_better"
                    ],
                    "context_baseline_weight": contextual_prediction["baseline_weight"],
                    "context_signal_weight": contextual_prediction["signal_weight"],
                }
            )
            gate_entropy = float(
                -sum(
                    p * np.log(max(p, 1e-15))
                    for p in contextual_weights.values()
                )
            )
            for arm_name, weight in contextual_weights.items():
                posterior_records.append(
                    {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "regime": regime,
                        "score": "contextual_bayes_gate",
                        "posterior_scope": "bayesian_linear_context",
                        "arm": arm_name,
                        "weight": weight,
                        "entropy": gate_entropy,
                        "eta_effective": np.nan,
                        "n_updates": contextual_gate.n_updates,
                    }
                )
            if config.save_weights:
                for ticker, weight in target[target > 0.0].items():
                    weight_records.append(
                        {
                            "master_seed": master_seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": strategy_name,
                            "ticker": ticker,
                            "weight": weight,
                        }
                    )

        # coalition paths have separate holdings and shared row draws
        marginal_period_performance: dict[str, dict[str, Any]] = {}
        if "marginal_mix_utility" in config.score_names:
            score = "marginal_mix_utility"
            weights = action_weights_by_score[score]
            for omitted_arm in arm_names:
                loo_weights = {
                    arm: value for arm, value in weights.items() if arm != omitted_arm
                }
                if sum(loo_weights.values()) <= 1e-15:
                    marginal_period_performance[omitted_arm] = {
                        "net_excess_return": np.nan
                    }
                    continue
                strategy_name = f"diagnostic__marginal_mix_utility__without__{omitted_arm}"
                loo_scenarios, loo_diag = mix_scenarios_crn(
                    arm_scenarios,
                    loo_weights,
                    n_rows=config.n_scenarios,
                    arm_uniforms=arm_uniforms,
                    row_uniforms=row_uniforms,
                )
                target, opt_diag = _optimize_or_fallback(
                    loo_scenarios,
                    pretrade_by_strategy[strategy_name],
                    config,
                    first_period=first_period,
                    explicit_cvar_budget=action_explicit_cvar_budget,
                    risk_scenarios=action_cvar_scenarios,
                )
                perf = _execute_strategy(
                    state=states[strategy_name],
                    target=target,
                    pretrade=pretrade_by_strategy[strategy_name],
                    formation_date=formation,
                    outcome_date=outcome,
                    data=data,
                    config=config,
                    first_period=first_period,
                )
                marginal_period_performance[omitted_arm] = perf
                return_records.append({"master_seed": master_seed, "regime": regime, **perf})
                optimizer_records.append(
                    {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "strategy": strategy_name,
                        **opt_diag,
                        "action": "leave_one_out_mix",
                        "omitted_arm": omitted_arm,
                        "realized_row_counts": json.dumps(
                            loo_diag["realized_row_counts"], sort_keys=True
                        ),
                    }
                )
                if config.save_weights:
                    for ticker, weight in target[target > 0.0].items():
                        weight_records.append(
                            {
                                "master_seed": master_seed,
                                "formation_date": formation,
                                "outcome_date": outcome,
                                "strategy": strategy_name,
                                "ticker": ticker,
                                "weight": weight,
                            }
                        )

        if config.include_equal_mix:
            strategy_name = "fixed_equal__mix"
            equal_weights = {name: 1.0 / len(arm_names) for name in arm_names}
            action_scenarios, mix_diag = mix_scenarios_crn(
                arm_scenarios,
                equal_weights,
                n_rows=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
            )
            target, opt_diag = _optimize_or_fallback(
                action_scenarios,
                pretrade_by_strategy[strategy_name],
                config,
                first_period=first_period,
                explicit_cvar_budget=action_explicit_cvar_budget,
                risk_scenarios=action_cvar_scenarios,
            )
            perf = _execute_strategy(
                state=states[strategy_name],
                target=target,
                pretrade=pretrade_by_strategy[strategy_name],
                formation_date=formation,
                outcome_date=outcome,
                data=data,
                config=config,
                first_period=first_period,
            )
            return_records.append({"master_seed": master_seed, "regime": regime, **perf})
            optimizer_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "strategy": strategy_name,
                    **opt_diag,
                    "action": "mix",
                    "fixed_equal": True,
                    "realized_row_counts": json.dumps(mix_diag["realized_row_counts"], sort_keys=True),
                }
            )
            if config.save_weights:
                for ticker, weight in target[target > 0.0].items():
                    weight_records.append(
                        {
                            "master_seed": master_seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": strategy_name,
                            "ticker": ticker,
                            "weight": weight,
                        }
                    )

        if fixed_state_weights is not None:
            if regime is None:
                raise ValueError("fixed_state_weights requires formation_states.")
            if regime not in fixed_state_weights:
                raise ValueError(f"No fixed weights were supplied for state {regime!r}.")
            strategy_name = "fixed_state__mix"
            action_scenarios, mix_diag = mix_scenarios_crn(
                arm_scenarios,
                fixed_state_weights[regime],
                n_rows=config.n_scenarios,
                arm_uniforms=arm_uniforms,
                row_uniforms=row_uniforms,
            )
            target, opt_diag = _optimize_or_fallback(
                action_scenarios,
                pretrade_by_strategy[strategy_name],
                config,
                first_period=first_period,
                explicit_cvar_budget=action_explicit_cvar_budget,
                risk_scenarios=action_cvar_scenarios,
            )
            perf = _execute_strategy(
                state=states[strategy_name],
                target=target,
                pretrade=pretrade_by_strategy[strategy_name],
                formation_date=formation,
                outcome_date=outcome,
                data=data,
                config=config,
                first_period=first_period,
            )
            return_records.append(
                {"master_seed": master_seed, "regime": regime, **perf}
            )
            optimizer_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "strategy": strategy_name,
                    **opt_diag,
                    "action": "mix",
                    "fixed_state": True,
                    "fixed_state_weights": json.dumps(
                        dict(fixed_state_weights[regime]), sort_keys=True
                    ),
                    "realized_row_counts": json.dumps(
                        mix_diag["realized_row_counts"], sort_keys=True
                    ),
                }
            )
            if config.save_weights:
                for ticker, weight in target[target > 0.0].items():
                    weight_records.append(
                        {
                            "master_seed": master_seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": strategy_name,
                            "ticker": ticker,
                            "weight": weight,
                        }
                    )


        # use one risk budget and pretrade state for coalition credit
        anchored_period_losses: dict[str, dict[str, float]] = {}
        if anchored_credit_scores:
            if config.anchored_shapley_cvar_budget_source == "baseline":
                credit_budget_source = config.safe_mix_baseline_arm
            elif config.anchored_shapley_cvar_budget_source == "action_map":
                credit_budget_source = (
                    config.action_cvar_budget_source
                    if clean_risk_map
                    else config.common_cvar_budget_source
                )
            else:
                credit_budget_source = config.anchored_shapley_cvar_budget_source

            for score in sorted(anchored_credit_scores):
                credit_action = config.anchored_shapley_action
                strategy_name = f"{score}__{credit_action}"
                if score == "anchored_shapley_utility":
                    (
                        period_losses,
                        coalition_diagnostics,
                        credit_summary,
                        shapley_interactions,
                    ) = _anchored_shapley_losses(
                        posterior_weights=action_weights_by_score[score],
                        arm_scenarios=arm_scenarios,
                        arm_names=arm_names,
                        pretrade=pretrade_by_strategy[strategy_name],
                        outcome_date=outcome,
                        data=data,
                        config=config,
                        first_period=first_period,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                        explicit_cvar_budget=anchored_credit_explicit_cvar_budget,
                        risk_scenarios=anchored_credit_cvar_scenarios,
                    )
                    diagnostic_strategy = (
                        "diagnostic__anchored_shapley_coalition"
                    )
                elif score == "anchored_loo_utility":
                    (
                        period_losses,
                        coalition_diagnostics,
                        credit_summary,
                    ) = _anchored_loo_losses(
                        posterior_weights=action_weights_by_score[score],
                        arm_scenarios=arm_scenarios,
                        arm_names=arm_names,
                        pretrade=pretrade_by_strategy[strategy_name],
                        outcome_date=outcome,
                        data=data,
                        config=config,
                        first_period=first_period,
                        arm_uniforms=arm_uniforms,
                        row_uniforms=row_uniforms,
                        explicit_cvar_budget=anchored_credit_explicit_cvar_budget,
                        risk_scenarios=anchored_credit_cvar_scenarios,
                    )
                    shapley_interactions = {}
                    diagnostic_strategy = "diagnostic__anchored_loo_coalition"
                else:  # pragma: no cover - guarded by anchored_credit_scores
                    raise AssertionError(score)

                anchored_period_losses[score] = period_losses
                for diagnostic in coalition_diagnostics:
                    optimizer_records.append(
                        {
                            "master_seed": master_seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": diagnostic_strategy,
                            "score": score,
                            "credit_action": credit_action,
                            "credit_budget_source": credit_budget_source,
                            "credit_budget_source_config": (
                                config.anchored_shapley_cvar_budget_source
                            ),
                            "credit_common_cvar_budget": (
                                anchored_credit_explicit_cvar_budget
                            ),
                            **credit_summary,
                            **diagnostic,
                        }
                    )
                for (left, right), interaction in shapley_interactions.items():
                    optimizer_records.append(
                        {
                            "master_seed": master_seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": (
                                "diagnostic__anchored_shapley_interaction"
                            ),
                            "score": score,
                            "credit_action": credit_action,
                            "credit_budget_source": credit_budget_source,
                            "credit_budget_source_config": (
                                config.anchored_shapley_cvar_budget_source
                            ),
                            "credit_common_cvar_budget": (
                                anchored_credit_explicit_cvar_budget
                            ),
                            "interaction_left": left,
                            "interaction_right": right,
                            "shapley_pair_interaction": interaction,
                            **credit_summary,
                        }
                    )

        realized_excess = data.outcome_excess_returns.loc[outcome]
        losses_by_score: dict[str, dict[str, float]] = {
            score: {} for score in config.score_names
        }
        for arm_name in arm_names:
            for score in config.score_names:
                if score == "predictive_log":
                    loss = diagonal_predictive_log_loss(
                        arm_scenarios[arm_name], realized_excess
                    )
                elif score == "variance_only":
                    loss = variance_only_loss(arm_scenarios[arm_name], realized_excess)
                elif score == "post_cost_utility":
                    loss = quadratic_utility_loss(
                        loss_arm_period_performance[arm_name]["net_excess_return"],
                        gamma=config.utility_gamma,
                    )
                elif score == "cvar_proxy":
                    state = (
                        loss_states[arm_name]
                        if loss_states
                        else states[f"standalone__{arm_name}"]
                    )
                    loss = cvar_proxy_loss(
                        loss_arm_period_performance[arm_name]["net_excess_return"],
                        state.net_excess_history[:-1],
                    )
                elif score == "marginal_mix_utility":
                    full_perf = strategy_period_performance[
                        "marginal_mix_utility__mix"
                    ]
                    loo_perf = marginal_period_performance.get(arm_name, {})
                    full_loss = quadratic_utility_loss(
                        full_perf["net_excess_return"],
                        gamma=config.utility_gamma,
                    )
                    loo_return = loo_perf.get("net_excess_return", np.nan)
                    loo_loss = (
                        quadratic_utility_loss(
                            loo_return, gamma=config.utility_gamma
                        )
                        if np.isfinite(loo_return)
                        else np.nan
                    )
                    loss = (
                        float(full_loss - loo_loss)
                        if np.isfinite(loo_loss)
                        else np.nan
                    )
                elif score in {
                    "anchored_loo_utility",
                    "anchored_shapley_utility",
                }:
                    loss = float(anchored_period_losses[score][arm_name])
                else:
                    raise AssertionError(score)
                losses_by_score[score][arm_name] = loss
                loss_records.append(
                    {
                        "master_seed": master_seed,
                        "formation_date": formation,
                        "outcome_date": outcome,
                        "regime": regime,
                        "score": score,
                        "arm": arm_name,
                        "loss": loss,
                    }
                )
        if contextual_gate is not None and contextual_x is not None:
            baseline = config.contextual_baseline_arm
            signal = config.contextual_signal_arm
            baseline_return = loss_arm_period_performance[baseline]["net_excess_return"]
            signal_return = loss_arm_period_performance[signal]["net_excess_return"]
            baseline_utility = -quadratic_utility_loss(
                baseline_return, gamma=config.utility_gamma
            )
            signal_utility = -quadratic_utility_loss(
                signal_return, gamma=config.utility_gamma
            )
            observed_advantage = float(baseline_utility - signal_utility)
            contextual_gate.update(contextual_x, observed_advantage)
            loss_records.append(
                {
                    "master_seed": master_seed,
                    "formation_date": formation,
                    "outcome_date": outcome,
                    "regime": regime,
                    "score": "contextual_gate_observation",
                    "arm": f"{baseline}_minus_{signal}",
                    "loss": -observed_advantage,
                }
            )

        for score in config.score_names:
            if config.posterior_partition in {"pooled", "partially_pooled_regime"}:
                pooled_posteriors[score].update(losses_by_score[score])
            if config.posterior_partition in {"regime", "partially_pooled_regime"}:
                if regime is None:
                    raise AssertionError("Missing regime at posterior update.")
                # record the formation posterior before updating it with realized losses
                state_posteriors[(score, regime)].update(losses_by_score[score])

    runtime = time.time() - start_time
    if source_tree_sha256 is None or source_tree_file_count is None:
        source_root = Path(__file__).resolve().parents[2]
        source_tree_sha256 = sha256_tree(source_root)
        source_tree_file_count = len(iter_tree_files(source_root))
    manifest = {
        "package_version": __version__,
        "source_tree_sha256": source_tree_sha256,
        "source_tree_file_count": int(source_tree_file_count),
        "master_seed": int(master_seed),
        "experiment_config": asdict(config),
        "calendar": {
            "n_outcomes": int(len(data.calendar)),
            "outcome_start": str(data.outcome_dates.min().date()),
            "outcome_end": str(data.outcome_dates.max().date()),
            "formation_start": str(data.formation_dates.min().date()),
            "formation_end": str(data.formation_dates.max().date()),
        },
        "data": data.source_metadata,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "runtime_seconds": runtime,
        "state_design": {
            "posterior_partition": config.posterior_partition,
            "state_calibration_months": config.state_calibration_months,
            "state_pooling_weight": config.state_pooling_weight,
            "state_counts": (
                formation_states.value_counts().to_dict()
                if formation_states is not None else {}
            ),
            "fixed_state_weights": (
                {key: dict(value) for key, value in fixed_state_weights.items()}
                if fixed_state_weights is not None else None
            ),
        },
        "randomness": {
            "scheme": "SHA256 named streams",
            "namespace": factory.namespace,
            "streams": [
                "arm/<arm>/<formation>",
                "hard_action/<formation>",
                "mix_labels/<formation>",
                "mix_row_resample/<formation>",
                "mean_treatment/<treatment>/<formation>",
            ],
        },
        "arm_cache": (
            {
                "enabled": arm_cache.enabled,
                "mode": arm_cache.mode,
                "root": str(arm_cache.root) if arm_cache.root is not None else None,
                "stats_at_seed_end": arm_cache.stats.as_dict(),
            }
            if arm_cache is not None
            else {"enabled": False}
        ),
    }
    return SeedRunResult(
        returns=pd.DataFrame(return_records),
        weights=pd.DataFrame(weight_records),
        posterior=pd.DataFrame(posterior_records),
        arm_losses=pd.DataFrame(loss_records),
        scenario_hashes=pd.DataFrame(scenario_hash_records),
        optimizer_diagnostics=pd.DataFrame(optimizer_records),
        arm_diagnostics=pd.DataFrame(arm_diag_records),
        manifest=manifest,
    )


def write_seed_result(result: SeedRunResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(output_dir / "returns.csv", index=False)
    result.posterior.to_csv(output_dir / "posterior.csv", index=False)
    result.arm_losses.to_csv(output_dir / "arm_losses.csv", index=False)
    result.scenario_hashes.to_csv(output_dir / "scenario_hashes.csv", index=False)
    result.optimizer_diagnostics.to_csv(
        output_dir / "optimizer_diagnostics.csv", index=False
    )
    result.arm_diagnostics.to_csv(output_dir / "arm_diagnostics.csv", index=False)
    if not result.weights.empty:
        result.weights.to_csv(output_dir / "weights.csv.gz", index=False, compression="gzip")
    write_json(output_dir / "run_manifest.json", result.manifest)


def load_seed_results(root: str | Path) -> pd.DataFrame:
    paths = sorted(Path(root).glob("seed_*/returns.csv"))
    if not paths:
        raise FileNotFoundError(f"No seed_*/returns.csv files under {root}")
    return pd.concat([pd.read_csv(path, parse_dates=["formation_date", "outcome_date"]) for path in paths])
