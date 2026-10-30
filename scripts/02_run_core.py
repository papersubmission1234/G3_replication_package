#!/usr/bin/env python3
"""Run the preregistered two-arm score-by-action factorial with CRN."""
from __future__ import annotations

import argparse
from dataclasses import asdict

from _common import (
    ROOT, add_execution_arguments, execution_kwargs, load_context,
    select_seeds, write_json,
)
from g3_confirmatory.runner import ExperimentJob, preflight_existing_jobs, run_experiment_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    add_execution_arguments(parser)
    args = parser.parse_args()

    cfg, paths, data, experiment, _, seeds = load_context(args.config)
    execution = execution_kwargs(args, cfg, paths)
    selected = select_seeds(seeds, args.seed)
    output = paths["output_root"] / "core"
    preflight_existing_jobs(
        [ExperimentJob("core", experiment, output)], selected,
        overwrite=args.overwrite, data=data,
    )
    output.mkdir(parents=True, exist_ok=True)

    declaration = {
        "family": "confirmatory_core",
        "frozen_before_execution": True,
        "experiment_config": asdict(experiment),
        "seeds": selected,
        "primary_comparisons": [
            "post_cost_utility__hard - predictive_log__hard",
            "post_cost_utility__mix - post_cost_utility__hard",
            "post_cost_utility__mix - fixed_equal__mix",
            "post_cost_utility__mix - standalone__M_HierMacro",
            "optimized strategies - benchmark__1N",
        ],
        "primary_outcomes": [
            "annualized_excess_mean",
            "excess_sharpe",
            "certainty_equivalent_annualized",
            "maximum_drawdown",
            "mean_one_way_turnover_ex_initial",
            "terminal_wealth_from_1000",
            "posterior weight on M_HierMacro",
        ],
    }
    write_json(output / "preregistered_core_declaration.json", declaration)
    run_experiment_seeds(
        data,
        seeds=selected,
        config=experiment,
        output_dir=output,
        overwrite=args.overwrite,
        **execution,
    )
    print(f"Core experiment written to {output}")


if __name__ == "__main__":
    main()
