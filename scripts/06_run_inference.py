#!/usr/bin/env python3
"""Compute paired effects, uncertainty intervals, and multiple-testing results.

Seed variation measures algorithmic uncertainty; time-series resampling measures
historical-path uncertainty. Multiplicity tests use completed trials with the
full core calendar. Date-misaligned and holdout-only trials are excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, load_context, read_seed_table, write_json
from g3_confirmatory.inference import (
    multiple_testing_by_seed,
    newey_west_mean_test,
    paired_block_bootstrap_many,
    probabilistic_sharpe_ratio,
    summarize_algorithmic_seed_effects,
)
from g3_confirmatory.run_checks import scan_run_settings
from g3_confirmatory.metrics import (
    default_subperiod_metrics,
    paired_seed_differences,
    summarize_metrics_across_seeds,
)

PRIMARY_PAIRS = (
    ("post_cost_utility__hard", "predictive_log__hard", "score_under_hard"),
    ("post_cost_utility__mix", "post_cost_utility__hard", "action_under_utility"),
    ("post_cost_utility__mix", "fixed_equal__mix", "adaptive_vs_equal_pool"),
    ("post_cost_utility__mix", "standalone__M_HierMacro", "mix_vs_hier"),
    ("post_cost_utility__mix", "benchmark__1N", "mix_vs_1N"),
    ("standalone__M_HierMacro", "benchmark__1N", "hier_vs_1N"),
)


def _newey_west_core(returns: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for treatment, control, label in PRIMARY_PAIRS:
        for seed, group in returns.groupby("master_seed"):
            panel = group[group["strategy"].isin([treatment, control])].pivot(
                index="outcome_date", columns="strategy", values="net_excess_return"
            ).dropna()
            if treatment not in panel or control not in panel or len(panel) < 3:
                continue
            test = newey_west_mean_test(panel[treatment] - panel[control])
            records.append(
                {
                    "comparison": label,
                    "treatment": treatment,
                    "control": control,
                    "master_seed": int(seed),
                    **test,
                }
            )
    return pd.DataFrame(records)


def _psr_core(returns: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (seed, strategy), group in returns.groupby(["master_seed", "strategy"]):
        values = pd.to_numeric(group["net_excess_return"], errors="coerce").dropna()
        if len(values) < 4 or values.std(ddof=1) <= 0:
            continue
        result = probabilistic_sharpe_ratio(values)
        records.append(
            {"master_seed": int(seed), "strategy": str(strategy), **result}
        )
    return pd.DataFrame(records)


def _posterior_summary(core_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    posterior = read_seed_table(core_root, "posterior.csv")
    posterior["formation_date"] = pd.to_datetime(posterior["formation_date"])
    last = (
        posterior.sort_values("formation_date")
        .groupby(["master_seed", "score", "arm"], as_index=False)
        .tail(1)
    )
    summary = (
        last.groupby(["score", "arm"])["weight"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    return last, summary


def _discover_full_calendar_trials(
    output_root: Path,
    core_returns: pd.DataFrame,
    experiment_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_dates = {
        int(seed): tuple(pd.to_datetime(group["outcome_date"]).sort_values().dt.strftime("%Y-%m-%d"))
        for seed, group in core_returns[
            core_returns["strategy"] == "benchmark__1N"
        ].groupby("master_seed")
    }
    expected_seeds = sorted(core_dates)
    trial_frames: list[pd.DataFrame] = []
    audit_records: list[dict[str, object]] = []

    if experiment_ids is None:
        return_paths = sorted(output_root.rglob("seed_*/returns.csv"))
        return_paths = [p for p in return_paths if not any(
            part in {"cache", "raw_arms", "seed_runs"} for part in p.relative_to(output_root).parts
        )]
    else:
        missing = [e for e in experiment_ids if not list((output_root / e).glob("seed_*/returns.csv"))]
        if missing:
            raise ValueError("Declared inference experiments are missing: " + ", ".join(missing))
        return_paths = sorted(p for e in experiment_ids for p in (output_root / e).glob("seed_*/returns.csv"))
    for path in return_paths:
        try:
            experiment_dir = path.parent.parent
            rel = experiment_dir.relative_to(output_root).as_posix()
        except ValueError:
            continue
        if rel.startswith("inference"):
            continue
        frame = pd.read_csv(path, parse_dates=["formation_date", "outcome_date"])
        if frame.empty:
            continue
        seed_values = pd.unique(frame["master_seed"])
        if len(seed_values) != 1:
            raise ValueError(f"Expected one seed in {path}; found {seed_values}")
        seed = int(seed_values[0])
        for strategy, group in frame.groupby("strategy"):
            dates = tuple(group.sort_values("outcome_date")["outcome_date"].dt.strftime("%Y-%m-%d"))
            calendar_match = seed in core_dates and dates == core_dates[seed]
            trial_id = f"{rel}::{strategy}"
            audit_records.append(
                {
                    "trial_id": trial_id,
                    "experiment_id": rel,
                    "strategy": strategy,
                    "master_seed": seed,
                    "n_months": len(dates),
                    "calendar_matches_core": calendar_match,
                    "return_path": str(path),
                }
            )
            if not calendar_match:
                continue
            copy = group.copy()
            copy["original_strategy"] = copy["strategy"]
            copy["strategy"] = trial_id
            copy["experiment_id"] = rel
            trial_frames.append(copy)

    if not trial_frames:
        return pd.DataFrame(), pd.DataFrame(audit_records)
    trials = pd.concat(trial_frames, ignore_index=True)
    audit = pd.DataFrame(audit_records)

    # require every core seed for the multiple-testing family
    counts = trials.groupby("strategy")["master_seed"].nunique()
    complete_trials = counts[counts == len(expected_seeds)].index
    trials = trials[trials["strategy"].isin(complete_trials)].copy()
    if not audit.empty:
        audit["complete_across_core_seeds"] = audit["trial_id"].isin(complete_trials)
    if experiment_ids is not None:
        incomplete = audit[~audit["calendar_matches_core"] | ~audit["complete_across_core_seeds"]]
        absent = set(experiment_ids) - set(trials["experiment_id"])
        if not incomplete.empty or absent:
            names = sorted(set(incomplete["experiment_id"]) | absent)
            raise ValueError("Declared inference experiments have incomplete or misaligned seeds: " + ", ".join(names))
    return trials, audit


def _read_family_manifest(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "experiment_id" not in frame or frame.empty:
        raise ValueError(f"Expected a nonempty experiment_id column in {path}")
    values = frame["experiment_id"].astype(str).tolist()
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate experiments in {path}")
    for value in values:
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"Experiment paths must be relative to output_root: {value}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument("--quick", action="store_true", help="Small bootstrap counts for smoke testing.")
    parser.add_argument("--skip-multiple-testing", action="store_true")
    family_group = parser.add_mutually_exclusive_group()
    family_group.add_argument(
        "--family-manifest", type=Path,
        default=ROOT / "docs" / "expected_37_experiments.csv",
        help="CSV declaring the main inference experiments; diagnostic folders are excluded.",
    )
    family_group.add_argument(
        "--discover-all", action="store_true",
        help="Explicitly use all completed experiments, including later diagnostics, as a different family.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel bootstrap workers; 0 chooses a conservative automatic count.")
    parser.add_argument("--mp-start-method", choices=("spawn", "fork", "forkserver"), default=None)
    args = parser.parse_args()

    cfg, paths, _, experiment, _, configured_seeds = load_context(args.config)
    parallel_cfg = dict(cfg.get("parallel", {}) or {})
    workers = int(args.workers) if args.workers is not None else int(parallel_cfg.get("workers", 1))
    mp_start_method = args.mp_start_method or str(parallel_cfg.get("start_method", "spawn"))
    inference_cfg = cfg.get("inference", {})
    block_bootstrap = 200 if args.quick else int(inference_cfg.get("primary_bootstrap_replications", 5000))
    multiple_bootstrap = 300 if args.quick else int(inference_cfg.get("multiple_testing_bootstrap_replications", 5000))
    mean_block = float(inference_cfg.get("mean_block_length_months", 6.0))
    bootstrap_seed = int(inference_cfg.get("bootstrap_seed", 20260322))

    core_root = paths["output_root"] / "core"
    returns = read_seed_table(core_root, "returns.csv")
    returns["outcome_date"] = pd.to_datetime(returns["outcome_date"])
    if set(map(int, returns["master_seed"].unique())) != set(configured_seeds):
        raise ValueError("Core return seeds differ from the configured seed set.")
    experiment_ids = None if args.discover_all else _read_family_manifest(args.family_manifest)
    checks = scan_run_settings(paths["output_root"], expected_core=experiment, experiments=experiment_ids)
    bad = [r for r in checks if not r["passed"]]
    if bad:
        raise ValueError("Saved run settings need correction before inference:\n" + "\n".join(
            r["manifest"] + ": " + "; ".join(r["issues"]) for r in bad[:12]
        ))
    trial_returns, trial_audit = _discover_full_calendar_trials(paths["output_root"], returns, experiment_ids)
    output = paths["output_root"] / "inference"
    output.mkdir(parents=True, exist_ok=True)

    write_json(
        output / "inference_declaration.json",
        {
            "return_convention": "net excess return for Sharpe/inference; net simple return for wealth/drawdown",
            "primary_pairs": [
                {"treatment": t, "control": c, "label": label}
                for t, c, label in PRIMARY_PAIRS
            ],
            "uncertainty_layers": [
                "paired algorithmic effects across master seeds",
                "stationary block bootstrap over aligned monthly histories",
            ],
            "primary_bootstrap_replications": block_bootstrap,
            "multiple_testing_bootstrap_replications": multiple_bootstrap,
            "mean_block_length_months": mean_block,
            "parallel_workers": workers,
            "multiprocessing_start_method": mp_start_method,
            "multiple_testing_note": (
                "Only declared trials completed for all core seeds with the exact core outcome calendar are included. "
                "--discover-all explicitly selects a different, recursively discovered family."
            ),
            "family_manifest": str(args.family_manifest) if experiment_ids is not None else None,
            "discover_all": bool(args.discover_all),
        },
    )

    metrics = default_subperiod_metrics(returns, utility_gamma=experiment.utility_gamma)
    metrics.to_csv(output / "core_metrics_by_seed_strategy.csv", index=False)
    summarize_metrics_across_seeds(metrics).to_csv(
        output / "core_metrics_seed_summary.csv", index=False
    )

    all_differences = []
    available = set(returns["strategy"])
    available_pairs = [
        (treatment, control, label)
        for treatment, control, label in PRIMARY_PAIRS
        if treatment in available and control in available
    ]
    for treatment, control, label in available_pairs:
        diff = paired_seed_differences(
            metrics, treatment=treatment, control=control
        )
        diff.insert(0, "comparison", label)
        all_differences.append(diff)
    paired = pd.concat(all_differences, ignore_index=True) if all_differences else pd.DataFrame()
    paired.to_csv(output / "primary_paired_seed_differences.csv", index=False)
    if not paired.empty:
        summarize_algorithmic_seed_effects(paired).to_csv(
            output / "primary_algorithmic_effect_summary.csv", index=False
        )
    bootstrap = paired_block_bootstrap_many(
        returns,
        comparisons=available_pairs,
        n_bootstrap=block_bootstrap,
        mean_block_length=mean_block,
        master_seed=bootstrap_seed,
        workers=workers,
        start_method=mp_start_method,
    )
    bootstrap.to_csv(output / "primary_historical_block_bootstrap.csv", index=False)

    _newey_west_core(returns).to_csv(output / "primary_newey_west_by_seed.csv", index=False)
    _psr_core(returns).to_csv(output / "core_probabilistic_sharpe.csv", index=False)
    final_posterior, posterior_summary = _posterior_summary(core_root)
    final_posterior.to_csv(output / "core_final_posterior_by_seed.csv", index=False)
    posterior_summary.to_csv(output / "core_final_posterior_summary.csv", index=False)

    trial_audit.to_csv(output / "multiple_testing_trial_inclusion_audit.csv", index=False)
    if trial_returns.empty:
        print("No complete full-calendar trial family found; core inference is complete.")
        return
    family = sorted(trial_returns["strategy"].unique())
    write_json(
        output / "multiple_testing_frozen_family.json",
        {
            "n_trials": len(family),
            "family_manifest": str(args.family_manifest) if experiment_ids is not None else None,
            "discover_all": bool(args.discover_all),
            "trials": family,
            "benchmark": "core::benchmark__1N",
            "return_column": "net_excess_return",
        },
    )
    trial_metrics = default_subperiod_metrics(
        trial_returns, utility_gamma=experiment.utility_gamma
    )
    trial_metrics.to_csv(output / "all_trial_metrics_by_seed.csv", index=False)

    if args.skip_multiple_testing:
        print("Skipped DSR/SPA/Romano-Wolf by request.")
        return
    benchmark = "core::benchmark__1N"
    if benchmark not in family:
        raise ValueError(
            f"Primary benchmark {benchmark!r} is absent from the complete trial family."
        )
    candidates = [trial for trial in family if trial != benchmark]
    dsr, spa, rw = multiple_testing_by_seed(
        trial_returns,
        strategies=candidates,
        benchmark=benchmark,
        return_column="net_excess_return",
        n_bootstrap=multiple_bootstrap,
        mean_block_length=mean_block,
        random_seed=bootstrap_seed,
        workers=workers,
        start_method=mp_start_method,
    )
    dsr.to_csv(output / "all_trials_deflated_sharpe_approx.csv", index=False)
    spa.to_csv(output / "all_trials_spa_by_seed.csv", index=False)
    rw.to_csv(output / "all_trials_romano_wolf_by_seed.csv", index=False)
    print(f"Inference outputs written to {output}")


if __name__ == "__main__":
    main()
