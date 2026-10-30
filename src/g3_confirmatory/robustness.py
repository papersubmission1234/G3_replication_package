"""Prespecified robustness-grid construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import pandas as pd

from .experiment import ExperimentConfig
from .run_checks import named_setting_errors


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    family: str
    config: ExperimentConfig
    purpose: str


def build_robustness_specs(base: ExperimentConfig) -> list[ExperimentSpec]:
    """Build the prespecified sensitivity grid without repeating the base run.
    
    Each cell changes one setting, except for the named arm-pool and state designs.
    """
    specs: list[ExperimentSpec] = []

    def add(identifier: str, family: str, config: ExperimentConfig, purpose: str) -> None:
        specs.append(ExperimentSpec(identifier, family, config, purpose))


    for scenarios in (250, 1000):
        add(
            f"scenario_count_{scenarios}",
            "scenario_count",
            replace(base, n_scenarios=scenarios),
            "Monte Carlo scenario-count sensitivity.",
        )
    for scenarios in (500, ):
        add(
            f"scenario_count_{scenarios}",
            "scenario_resolution_additions",
            replace(base, n_scenarios=scenarios),
            "Additional endogenous scenario-resolution check.",
        )
    for scenarios in (250, 500, 1000):
        add(
            f"scenario_count_{scenarios}__common_risk",
            "scenario_resolution_common_risk",
            replace(
                base,
                n_scenarios=scenarios,
                common_cvar_budget_source="none",
                loss_cvar_budget_source="none",
                loss_cvar_scenario_source="none",
                action_cvar_budget_source="M_HierMacro",
                action_cvar_scenario_source="M_HierMacro",
            ),
            "Common HierMacro risk scenarios and budget; endogenous scoring losses.",
        )
    for bps in (0.0, 20.0, 50.0):
        add(
            f"cost_{int(bps):02d}bps",
            "transaction_cost",
            replace(
                base,
                optimizer=replace(base.optimizer, transaction_cost_bps=bps),
            ),
            "Net-performance sensitivity to proportional transaction costs.",
        )
    for cap in (0.10, 0.30):
        add(
            f"position_cap_{int(cap*100):02d}pct",
            "position_cap",
            replace(base, optimizer=replace(base.optimizer, max_weight=cap)),
            "Concentration-constraint sensitivity.",
        )
    for alpha in (0.90, 0.975):
        add(
            f"cvar_alpha_{str(alpha).replace('.', '_')}",
            "cvar_alpha",
            replace(base, optimizer=replace(base.optimizer, cvar_alpha=alpha)),
            "Tail-confidence-level sensitivity at the primary scenario count.",
        )
    add(
        "cvar_alpha_0_975_scenarios_1000",
        "cvar_tail_resolution",
        replace(
            base,
            n_scenarios=1000,
            optimizer=replace(base.optimizer, cvar_alpha=0.975),
        ),
        (
            "97.5% CVaR with exactly 1,000 scenarios (25 effective tail observations). "
            "Compare with the same confidence level at the primary scenario count."
        ),
    )
    add(
        "common_cvar_budget_hier",
        "cvar_budget",
        replace(base, common_cvar_budget_source="M_HierMacro"),
        "Hold the numerical CVaR budget common across actions within a date.",
    )
    add(
        "common_cvar_budget_equal_pool",
        "cvar_budget",
        replace(base, common_cvar_budget_source="fixed_equal_mix"),
        "Use the fixed equal predictive pool as the common risk-budget reference.",
    )
    for penalty in (0.0, 0.05, 0.20):
        add(
            f"adjustment_penalty_{str(penalty).replace('.', '_')}",
            "adjustment_penalty",
            replace(
                base,
                optimizer=replace(base.optimizer, adjustment_penalty=penalty),
            ),
            "Sensitivity to the separate L1 allocation-adjustment penalty.",
        )
    add(
        "history_24_of_36",
        "history_filter",
        replace(base, min_observations=24, max_missing_fraction=1.0 / 3.0),
        "Incomplete-history sensitivity within the 36-month window.",
    )
    for training in (24,):
        min_obs = min(training, base.min_observations)
        add(
            f"estimation_window_{training}m",
            "estimation_window",
            replace(
                base,
                training_months=training,
                min_observations=min_obs,
                max_missing_fraction=0.0,
            ),
            "Shorter model-estimation-window sensitivity supported by the available history.",
        )
    add(
        "universe_equity_bond_only",
        "asset_universe",
        replace(base, included_asset_classes=("equity", "bond")),
        "Equity-and-bond universe versus the full multi-asset universe.",
    )
    if base.information_lag_months != 0:
        add(
            "information_lag_zero",
            "information_timing",
            replace(base, information_lag_months=0),
            "After-close market-data timing sensitivity.",
        )
    for policy in ("macro_only", "equity_bond_only", "all_equity_style"):
        add(
            f"class_style_{policy}",
            "asset_class_factor_policy",
            replace(base, class_style_policy=policy),
            "Alternative style-factor assignment for non-equity/non-bond classes.",
        )
    for fraction in (0.25, 0.50, 1.00):
        add(
            f"mean_shrinkage_{str(fraction).replace('.', '_')}",
            "mean_shrinkage",
            replace(base, mean_shrinkage_fraction=fraction),
            "Prespecified action-layer scenario-mean shrinkage response curve.",
        )
    incremental_pools = {
        "hier_only": ("M_HierMacro",),
        "hier_ff5": ("M_HierMacro", "M_FF5MOM"),
        "hier_lw": ("M_HierMacro", "M_LW_MinVar"),
        "hier_momentum": ("M_HierMacro", "M_Momentum"),
        "hier_ff5_lw": ("M_HierMacro", "M_FF5MOM", "M_LW_MinVar"),
        "hier_ff5_momentum": ("M_HierMacro", "M_FF5MOM", "M_Momentum"),
        "full_four_arm": (
            "M_HierMacro", "M_FF5MOM", "M_LW_MinVar", "M_Momentum"
        ),
    }
    for label, pool in incremental_pools.items():
        if tuple(base.arm_names) == tuple(pool):
            continue
        add(
            f"arm_pool_{label}",
            "arm_pool",
            replace(base, arm_names=pool),
            "Incremental arm-complementarity test with equal-pool control.",
        )
    for calibration in (12, 36):
        add(
            f"posterior_calibration_{calibration}m",
            "posterior_calibration",
            replace(base, posterior_calibration_months=calibration),
            "Learning-rate calibration-window sensitivity.",
        )
    for forgetting in (0.95, 0.98):
        add(
            f"posterior_forgetting_{str(forgetting).replace('.', '_')}",
            "posterior_forgetting",
            replace(base, posterior_forgetting=forgetting),
            "Discounted generalized-posterior sensitivity.",
        )
    for spec in specs:
        errors = named_setting_errors(spec.experiment_id, asdict(spec.config))
        if errors:
            raise ValueError("; ".join(errors))
    return specs


def specs_to_frame(specs: Iterable[ExperimentSpec]) -> pd.DataFrame:
    records = []
    for spec in specs:
        payload = asdict(spec.config)
        records.append(
            {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "purpose": spec.purpose,
                "config": payload,
            }
        )
    return pd.DataFrame(records)


def select_specs(
    specs: Iterable[ExperimentSpec],
    families: Iterable[str] | None = None,
) -> list[ExperimentSpec]:
    specs = list(specs)
    if families is None:
        return specs
    wanted = set(families)
    unknown = wanted.difference({spec.family for spec in specs})
    if unknown:
        raise ValueError(f"Unknown robustness families: {sorted(unknown)}")
    return [spec for spec in specs if spec.family in wanted]
