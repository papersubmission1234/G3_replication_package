#!/usr/bin/env python3
"""Run the score, action, and CVaR-map decomposition with common random numbers.

Compare three action maps:
- endogenous: the action supplies its mean, CVaR scenarios, and budget;
- Hier budget only: the action supplies the mean and scenarios, HierMacro the budget;
- full Hier map: the action supplies the mean, HierMacro the scenarios and budget.

Predictive-log posteriors are independent of decision losses. The utility block
first varies the action map with endogenous losses, then varies the loss map
while holding the action map fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace

from _common import (
    ROOT,
    add_execution_arguments,
    execution_kwargs,
    load_context,
    select_seeds,
    write_json,
)
from g3_confirmatory.configuration import (
    configured_seeds,
    experiment_config_from_mapping,
    load_yaml,
)
from g3_confirmatory.runner import ExperimentJob, preflight_existing_jobs, run_experiment_jobs
from g3_confirmatory.run_checks import merge_trial_manifest


ACTION_MAPS = (
    (
        "endogenous",
        "none",
        "none",
        "Action-specific CVaR distribution and action-specific equal-weight budget.",
    ),
    (
        "hier_budget_only",
        "M_HierMacro",
        "none",
        "Action-specific CVaR distribution with only the scalar budget fixed to M_HierMacro.",
    ),
    (
        "hier_full_risk",
        "M_HierMacro",
        "M_HierMacro",
        "Both the CVaR distribution and scalar budget are fixed to M_HierMacro.",
    ),
)

LOSS_MAPS_FOR_FIXED_ACTION = (
    (
        "hier_budget_only",
        "M_HierMacro",
        "none",
        "Counterfactual arm portfolios share only the HierMacro scalar budget.",
    ),
    (
        "hier_full_risk",
        "M_HierMacro",
        "M_HierMacro",
        "Counterfactual arm portfolios share the full HierMacro CVaR map.",
    ),
)


def experiment_specs(base):
    common = dict(
        arm_names=("M_FF5MOM", "M_HierMacro"),
        action_names=("hard", "mix"),
        include_equal_mix=True,
        posterior_partition="pooled",
        common_cvar_budget_source="none",
    )
    specs = []

    # reuse score posteriors across action maps
    for action_label, budget_source, scenario_source, description in ACTION_MAPS:
        experiment_id = f"predictive_log__action_{action_label}"
        specs.append(
            (
                experiment_id,
                replace(
                    base,
                    **common,
                    score_names=("predictive_log",),
                    loss_cvar_budget_source="none",
                    loss_cvar_scenario_source="none",
                    action_cvar_budget_source=budget_source,
                    action_cvar_scenario_source=scenario_source,
                ),
                f"Predictive-log action decomposition. {description}",
            )
        )

    for action_label, budget_source, scenario_source, description in ACTION_MAPS:
        experiment_id = (
            "post_cost_utility__loss_endogenous__action_" + action_label
        )
        specs.append(
            (
                experiment_id,
                replace(
                    base,
                    **common,
                    score_names=("post_cost_utility",),
                    loss_cvar_budget_source="none",
                    loss_cvar_scenario_source="none",
                    action_cvar_budget_source=budget_source,
                    action_cvar_scenario_source=scenario_source,
                ),
                f"Utility-score action decomposition with endogenous loss map. {description}",
            )
        )

    # hold the action map fixed while varying the loss map
    for loss_label, budget_source, scenario_source, description in LOSS_MAPS_FOR_FIXED_ACTION:
        experiment_id = (
            f"post_cost_utility__loss_{loss_label}__action_hier_full_risk"
        )
        specs.append(
            (
                experiment_id,
                replace(
                    base,
                    **common,
                    score_names=("post_cost_utility",),
                    loss_cvar_budget_source=budget_source,
                    loss_cvar_scenario_source=scenario_source,
                    action_cvar_budget_source="M_HierMacro",
                    action_cvar_scenario_source="M_HierMacro",
                ),
                "Utility-score loss-map decomposition under a fully fixed HierMacro action map. "
                + description,
            )
        )

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "confirmatory.yaml")
    )
    parser.add_argument("--seed", type=int, help="Run one smoke-test seed.")
    parser.add_argument(
        "--seed-start",
        type=int,
        help="Override the first seed when --n-seeds is supplied.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        help="Override the configured seed count without editing YAML.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument(
        "--experiment",
        help="Optional comma-separated experiment IDs. Use --list first.",
    )
    parser.add_argument("--list", action="store_true")
    add_execution_arguments(parser)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    base = experiment_config_from_mapping(raw_cfg.get("experiment"))
    configured_seeds(raw_cfg)
    if args.no_weights:
        base = replace(base, save_weights=False)

    specs = experiment_specs(base)
    requested = (
        {item.strip() for item in args.experiment.split(",") if item.strip()}
        if args.experiment
        else None
    )
    if requested is not None:
        known = {experiment_id for experiment_id, _, _ in specs}
        missing = requested.difference(known)
        if missing:
            raise ValueError(f"Unknown experiment IDs: {sorted(missing)}")
        specs = [item for item in specs if item[0] in requested]

    if args.list:
        for experiment_id, config, _ in specs:
            print(
                experiment_id,
                f"score={config.score_names[0]}",
                f"loss_budget={config.loss_cvar_budget_source}",
                f"loss_risk={config.loss_cvar_scenario_source}",
                f"action_budget={config.action_cvar_budget_source}",
                f"action_risk={config.action_cvar_scenario_source}",
            )
        return

    cfg, paths, data, loaded_base, _, loaded_seeds = load_context(args.config)
    execution = execution_kwargs(args, cfg, paths)
    if asdict(loaded_base) != asdict(
        experiment_config_from_mapping(raw_cfg.get("experiment"))
    ):
        raise AssertionError("Experiment config changed while loading execution context.")

    if args.seed is not None:
        seeds = select_seeds(loaded_seeds, args.seed)
    elif args.n_seeds is not None:
        if args.n_seeds <= 0:
            raise ValueError("--n-seeds must be positive.")
        start = int(args.seed_start if args.seed_start is not None else loaded_seeds[0])
        seeds = list(range(start, start + int(args.n_seeds)))
    else:
        seeds = loaded_seeds

    root = paths["output_root"] / "decompositions" / "full_risk_map"
    preflight_existing_jobs(
        [ExperimentJob(experiment_id, config, root / experiment_id)
         for experiment_id, config, purpose in specs], seeds,
        overwrite=args.overwrite, data=data,
    )
    root.mkdir(parents=True, exist_ok=True)
    merge_trial_manifest(
        root / "frozen_trial_manifest.json",
        {
            "frozen_before_execution": True,
            "requested_seeds": seeds,
            "n_configurations": len(specs),
            "configurations": [
                {
                    "experiment_id": experiment_id,
                    "purpose": purpose,
                    "config": asdict(config),
                }
                for experiment_id, config, purpose in specs
            ],
            "primary_estimands": [
                "mix minus hard under endogenous action CVaR map",
                "mix minus hard under common HierMacro budget only",
                "mix minus hard under fully fixed HierMacro CVaR map",
                "budget-threshold channel: endogenous effect minus budget-only effect",
                "risk-distribution channel: budget-only effect minus fully fixed-risk effect",
                "objective/pooling channel: fully fixed-risk mix-minus-hard effect",
                "final HierMacro posterior across endogenous, budget-only, and fully fixed utility-loss maps",
            ],
        },
    )

    jobs = []
    for experiment_id, config, purpose in specs:
        output = root / experiment_id
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "configuration.json",
            {
                "experiment_id": experiment_id,
                "purpose": purpose,
                "config": asdict(config),
            },
        )
        jobs.append(
            ExperimentJob(
                job_id=f"full_risk_map/{experiment_id}",
                config=config,
                output_dir=output,
                metadata={"family": "full_risk_map", "purpose": purpose},
            )
        )

    run_experiment_jobs(
        data,
        jobs=jobs,
        seeds=seeds,
        overwrite=args.overwrite,
        **execution,
    )
    for job in jobs:
        print(f"completed: {job.job_id}")


if __name__ == "__main__":
    main()
