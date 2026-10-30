#!/usr/bin/env python3
"""Run exploratory small-pool and mean-shrinkage sensitivity experiments.

These post-review specifications are recorded separately from the core study.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

from _common import ROOT, load_context, add_execution_arguments, execution_kwargs
from g3_confirmatory.configuration import experiment_config_from_mapping
from g3_confirmatory.runner import ExperimentJob, preflight_existing_jobs, run_experiment_jobs
from g3_confirmatory.run_checks import checked_core_config
from g3_confirmatory.displacement_diagnostics import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--family", choices=("small_pool", "shrinkage"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    add_execution_arguments(parser)
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg, paths, data, base, _, all_seeds = load_context(args.config)
    seeds = [args.seed] if args.seed is not None else all_seeds
    base = checked_core_config(paths["output_root"], base, seeds, required=True)
    base = replace(base, score_names=("post_cost_utility",), action_names=("hard", "mix"),
                   include_equal_mix=True, save_weights=False, posterior_partition="pooled",
                   common_cvar_budget_source="none", loss_cvar_budget_source="none", loss_cvar_scenario_source="none",
                   action_cvar_budget_source="none", action_cvar_scenario_source="none")
    specs = []
    if args.family == "small_pool":
        for label, arms in (
            ("hier_lw", ("M_HierMacro", "M_LW_MinVar")),
            ("hier_momentum", ("M_HierMacro", "M_Momentum")),
            ("four_models", ("M_FF5MOM", "M_HierMacro", "M_LW_MinVar", "M_Momentum")),
        ):
            specs.append((label, replace(base, arm_names=arms)))
    else:
        for fraction in (0.0, 0.5, 1.0):
            for map_name, source in (("endogenous", "none"), ("full_hier", "M_HierMacro")):
                label = f"shrink_{fraction:g}_{map_name}".replace(".", "p")
                specs.append((label, replace(base, mean_shrinkage_fraction=fraction,
                    action_cvar_budget_source=source, action_cvar_scenario_source=source)))
        # each shrinkage treatment has its own HierMacro risk map
    if args.list:
        for name, config in specs:
            print(name, config.arm_names, config.mean_shrinkage_fraction, config.action_cvar_scenario_source)
        return
    seeds = [args.seed] if args.seed is not None else all_seeds
    root = paths["output_root"] / "diagnostics/extensions" / args.family
    jobs = [ExperimentJob(job_id=f"diagnostic_{args.family}/{name}", config=c, output_dir=root / name,
                metadata={"family": "diagnostic_extensions", "exploratory": True}) for name, c in specs]
    preflight_existing_jobs(jobs, seeds, overwrite=args.overwrite, data=data)
    atomic_json(root / "extension_protocol.json", {"post_review_exploratory": True,
        "not_retroactively_preregistered": True, "no_inherited_performance_claims": True,
        "specifications": {name: asdict(c) for name, c in specs}, "seeds": all_seeds})
    execution = execution_kwargs(args, cfg, paths)
    execution["result_cache_dir"] = paths["output_root"] / "diagnostics/cache/seed_runs"
    execution["cache_mode"] = "read"
    run_experiment_jobs(data=data, jobs=jobs, seeds=seeds, overwrite=args.overwrite, **execution)
    print(f"New extension results: {root}")


if __name__ == "__main__":
    main()
