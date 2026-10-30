#!/usr/bin/env python3
"""Run the covariance-invariant hierarchical mean-suppression ablation."""
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
from g3_confirmatory.mean_treatments import MEAN_TREATMENTS
from g3_confirmatory.runner import ExperimentJob, preflight_existing_jobs, run_experiment_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--treatments",
        help="Comma-separated subset of mean treatments; default is the full prespecified set.",
    )
    add_execution_arguments(parser)
    args = parser.parse_args()

    cfg, paths, data, base, _, seeds = load_context(args.config)
    execution = execution_kwargs(args, cfg, paths)
    selected_seeds = select_seeds(seeds, args.seed)
    treatments = (
        [item.strip() for item in args.treatments.split(",") if item.strip()]
        if args.treatments else list(MEAN_TREATMENTS)
    )
    unknown = set(treatments).difference(MEAN_TREATMENTS)
    if unknown:
        raise ValueError(f"Unknown mean treatments: {sorted(unknown)}")

    root = paths["output_root"] / "mean_ablation"
    preflight_existing_jobs(
        [ExperimentJob(t, replace(base, hierarchical_mean_treatment=t), root / t) for t in treatments],
        selected_seeds, overwrite=args.overwrite, data=data,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "ablation_declaration.json",
        {
            "treatments": treatments,
            "seeds": selected_seeds,
            "identification": (
                "Within a master seed and formation date, the raw hierarchical "
                "fit/scenarios are Sd and reused; only the scenario column "
                "means change across treatments."
            ),
            "required_diagnostics": [
                "max_abs_covariance_change",
                "max_abs_target_mean_error",
                "scenario hashes",
                "portfolio risk budget",
                "turnover",
                "realized returns",
            ],
        },
    )

    jobs: list[ExperimentJob] = []
    for treatment in treatments:
        config = replace(base, hierarchical_mean_treatment=treatment)
        out = root / treatment
        write_json(out / "configuration.json", asdict(config))
        jobs.append(
            ExperimentJob(
                job_id=treatment,
                config=config,
                output_dir=out,
                metadata={"mean_treatment": treatment},
            )
        )

    run_experiment_jobs(
        data,
        jobs=jobs,
        seeds=selected_seeds,
        overwrite=args.overwrite,
        **execution,
    )
    for treatment in treatments:
        print(f"Completed mean treatment: {treatment}")


if __name__ == "__main__":
    main()
