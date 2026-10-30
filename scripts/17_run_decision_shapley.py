#!/usr/bin/env python3
"""Run exploratory Shapley attribution with HierMacro as the baseline.

In the mean family, optional experts contribute means while centered HierMacro
scenarios stay fixed. In the row family, they contribute scenario rows with
capped posterior mass. The default pools have two and three optional experts.

The leave-one-out score compares each expert with the grand coalition; the
Shapley score uses all coalitions. Every coalition starts from the same holdings
and uses the same HierMacro CVaR budget. The baseline loss is zero.

Exact Shapley requires 2**k coalitions per month for k optional experts.
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


def _label(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def experiment_specs(base, *, mean_caps: list[float], row_caps: list[float]):
    common = dict(
        score_names=(
            "post_cost_utility",
            "anchored_loo_utility",
            "anchored_shapley_utility",
        ),
        include_equal_mix=False,
        include_fixed_safe_mix=True,
        include_fixed_safe_singletons=True,
        posterior_partition="pooled",
        safe_mix_baseline_arm="M_HierMacro",
        anchored_shapley_max_players=4,
        anchored_shapley_cvar_budget_source="baseline",
        common_cvar_budget_source="none",
        # share the HierMacro budget across loss and action paths
        loss_cvar_budget_source="M_HierMacro",
        action_cvar_budget_source="M_HierMacro",
    )

    specs = []
    for cap in mean_caps:
        specs.append(
            (
                "mean",
                f"directional_mean_cap_{_label(cap)}",
                replace(
                    base,
                    **common,
                    arm_names=("M_HierMacro", "M_FF5MOM", "M_Momentum"),
                    action_names=("safe_mean_mix",),
                    anchored_shapley_action="safe_mean_mix",
                    safe_mix_max_nonbaseline_weight=float(cap),
                ),
                "HierMacro covariance retained exactly; FF5MOM and momentum can contribute only capped predictive means.",
            )
        )

    for cap in row_caps:
        specs.append(
            (
                "row",
                f"directional_tail_row_cap_{_label(cap)}",
                replace(
                    base,
                    **common,
                    arm_names=(
                        "M_HierMacro",
                        "M_FF5MOM",
                        "M_Momentum",
                        "M_StudentT",
                    ),
                    action_names=("safe_mix",),
                    anchored_shapley_action="safe_mix",
                    safe_mix_max_nonbaseline_weight=float(cap),
                ),
                "Capped full-scenario overlay adds directional and heavy-tail experts without allowing them to displace HierMacro.",
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
        "--family", choices=("all", "mean", "row"), default="all"
    )
    parser.add_argument(
        "--mean-caps",
        default="0.05,0.10,0.20",
        help="Comma-separated total non-HierMacro caps for mean-only modules.",
    )
    parser.add_argument(
        "--row-caps",
        default="0.10,0.20",
        help="Comma-separated total non-HierMacro caps for full-row mixtures.",
    )
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

    mean_caps = [float(x) for x in args.mean_caps.split(",") if x.strip()]
    row_caps = [float(x) for x in args.row_caps.split(",") if x.strip()]
    for cap in (*mean_caps, *row_caps):
        if not 0.0 <= cap <= 1.0:
            raise ValueError("All safe-action caps must lie in [0, 1].")

    specs = experiment_specs(base, mean_caps=mean_caps, row_caps=row_caps)
    if args.family != "all":
        specs = [item for item in specs if item[0] == args.family]

    requested = (
        {x.strip() for x in args.experiment.split(",") if x.strip()}
        if args.experiment
        else None
    )
    if requested is not None:
        known = {experiment_id for _, experiment_id, _, _ in specs}
        missing = requested.difference(known)
        if missing:
            raise ValueError(f"Unknown experiment IDs: {sorted(missing)}")
        specs = [item for item in specs if item[1] in requested]

    if args.list:
        for family, experiment_id, config, _ in specs:
            print(
                family,
                experiment_id,
                f"arms={','.join(config.arm_names)}",
                f"action={config.anchored_shapley_action}",
                f"cap={config.safe_mix_max_nonbaseline_weight:.3f}",
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

    root = paths["output_root"] / "decompositions" / "decision_shapley"
    preflight_existing_jobs(
        [ExperimentJob(experiment_id, config, root / family / experiment_id)
         for family, experiment_id, config, purpose in specs], seeds,
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
                    "family": family,
                    "experiment_id": experiment_id,
                    "purpose": purpose,
                    "config": asdict(config),
                }
                for family, experiment_id, config, purpose in specs
            ],
            "primary_estimands": [
                "anchored_shapley_utility safe action minus M_HierMacro standalone",
                "anchored_shapley_utility safe action minus fixed_safe_equal control",
                "anchored_shapley_utility safe action minus each fixed singleton overlay",
                "anchored_shapley_utility safe action minus post_cost_utility safe action",
                "anchored_shapley_utility safe action minus anchored_loo_utility safe action",
                "anchored_loo_utility safe action minus M_HierMacro and matched fixed controls",
                "posterior Shapley contribution sign and persistence for each optional expert",
                "cap sensitivity without selecting a cap ex post",
            ],
            "credit_assignment": {
                "coalition_value": "realized post-cost quadratic utility",
                "counterfactual_start": "same pretrade holdings as the implemented score-matched strategy",
                "risk_map": "common M_HierMacro equal-weight CVaR budget",
                "baseline": "M_HierMacro mandatory; zero contribution loss",
                "benchmarks": [
                    "grand-coalition leave-one-out contribution",
                    "exact Shapley contribution over all coalition contexts",
                ],
            },
        },
    )

    jobs = []
    for family, experiment_id, config, purpose in specs:
        output = root / family / experiment_id
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "configuration.json",
            {
                "family": family,
                "experiment_id": experiment_id,
                "purpose": purpose,
                "config": asdict(config),
            },
        )
        jobs.append(
            ExperimentJob(
                job_id=f"decision_shapley/{family}/{experiment_id}",
                config=config,
                output_dir=output,
                metadata={"family": f"decision_shapley_{family}", "purpose": purpose},
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
