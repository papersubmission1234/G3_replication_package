#!/usr/bin/env python3
"""Summarize the baseline-anchored Decision-Shapley experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import ROOT
from g3_confirmatory.configuration import (
    experiment_config_from_mapping,
    load_yaml,
    study_paths,
)
from g3_confirmatory.metrics import (
    metrics_by_seed_strategy,
    paired_seed_differences,
    summarize_metrics_across_seeds,
)


def _read_seed_files(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"seed_*/{filename}"))
    if not paths:
        return pd.DataFrame()
    parse_dates = [
        column
        for column in ("formation_date", "outcome_date")
        if column in pd.read_csv(paths[0], nrows=0).columns
    ]
    return pd.concat(
        [pd.read_csv(path, parse_dates=parse_dates) for path in paths],
        ignore_index=True,
    )


def _mean_ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    if n > 1 and np.isfinite(sd):
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        half = 0.0
    return n, mean, sd, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    paths = study_paths(raw_cfg)
    base = experiment_config_from_mapping(raw_cfg.get("experiment"))
    root = paths["output_root"] / "decompositions" / "decision_shapley"
    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    experiment_roots = sorted(
        {path.parent.parent for path in root.glob("**/seed_*/returns.csv")}
    )
    if not experiment_roots:
        raise FileNotFoundError(f"No Decision-Shapley seed outputs under {root}")

    all_metrics = []
    all_metric_summaries = []
    all_paired = []
    all_paired_summary = []
    all_credit_seed = []
    all_credit_summary = []
    all_interaction_seed = []
    all_interaction_summary = []
    all_final_posterior = []
    all_effective_seed = []
    all_effective_summary = []
    all_efficiency = []

    for experiment_root in experiment_roots:
        rel = experiment_root.relative_to(root)
        family = rel.parts[0]
        experiment_id = "/".join(rel.parts[1:])
        returns = _read_seed_files(experiment_root, "returns.csv")
        losses = _read_seed_files(experiment_root, "arm_losses.csv")
        posterior = _read_seed_files(experiment_root, "posterior.csv")
        diagnostics = _read_seed_files(experiment_root, "optimizer_diagnostics.csv")

        metrics = metrics_by_seed_strategy(
            returns,
            utility_gamma=base.utility_gamma,
            period_label="full",
        )
        metrics.insert(0, "experiment_id", experiment_id)
        metrics.insert(0, "family", family)
        all_metrics.append(metrics)

        metric_summary = summarize_metrics_across_seeds(
            metrics.drop(columns=["family", "experiment_id"])
        )
        metric_summary.insert(0, "experiment_id", experiment_id)
        metric_summary.insert(0, "family", family)
        all_metric_summaries.append(metric_summary)

        strategies = set(metrics["strategy"])
        action = (
            "safe_mean_mix"
            if any("safe_mean_mix" in name for name in strategies)
            else "safe_mix"
        )
        matched_controls = [
            "standalone__M_HierMacro",
            f"fixed_safe_equal__{action}",
            f"post_cost_utility__{action}",
            *sorted(
                strategy
                for strategy in strategies
                if strategy.startswith("fixed_safe_singleton__")
                and strategy.endswith(f"__{action}")
            ),
        ]
        base_metrics = metrics.drop(columns=["family", "experiment_id"])
        treatments = [
            f"anchored_loo_utility__{action}",
            f"anchored_shapley_utility__{action}",
        ]
        for treatment in treatments:
            controls = list(matched_controls)
            if treatment.startswith("anchored_shapley_utility"):
                controls.append(f"anchored_loo_utility__{action}")
            for control in dict.fromkeys(controls):
                if {treatment, control}.issubset(strategies):
                    paired = paired_seed_differences(
                        base_metrics,
                        treatment=treatment,
                        control=control,
                        metric_columns=(
                            "excess_sharpe",
                            "annualized_excess_mean",
                            "certainty_equivalent_annualized",
                            "maximum_drawdown",
                            "mean_one_way_turnover_ex_initial",
                            "terminal_wealth_from_1000",
                        ),
                    )
                    paired.insert(0, "experiment_id", experiment_id)
                    paired.insert(0, "family", family)
                    all_paired.append(paired)
                    for metric, group in paired.groupby("metric", sort=True):
                        n, mean, sd, lo, hi = _mean_ci(group["difference"])
                        all_paired_summary.append(
                            {
                                "family": family,
                                "experiment_id": experiment_id,
                                "treatment": treatment,
                                "control": control,
                                "metric": metric,
                                "n_seeds": n,
                                "mean_difference": mean,
                                "seed_sd": sd,
                                "ci_low": lo,
                                "ci_high": hi,
                                "fraction_positive": float(
                                    (
                                        pd.to_numeric(
                                            group["difference"], errors="coerce"
                                        )
                                        > 0
                                    ).mean()
                                ),
                            }
                        )

        if not losses.empty:
            for credit_score in (
                "anchored_loo_utility",
                "anchored_shapley_utility",
            ):
                credit = losses[losses["score"] == credit_score].copy()
                if credit.empty:
                    continue
                credit["contribution"] = -pd.to_numeric(
                    credit["loss"], errors="coerce"
                )
                seed_arm = (
                    credit.groupby(["master_seed", "arm"], as_index=False)
                    .agg(
                        mean_contribution=("contribution", "mean"),
                        median_contribution=("contribution", "median"),
                        fraction_positive=(
                            "contribution",
                            lambda values: float((values > 0).mean()),
                        ),
                        n_months=("contribution", "count"),
                    )
                )
                seed_arm.insert(0, "credit_score", credit_score)
                seed_arm.insert(0, "experiment_id", experiment_id)
                seed_arm.insert(0, "family", family)
                all_credit_seed.append(seed_arm)
                for arm, group in seed_arm.groupby("arm", sort=True):
                    n, mean, sd, lo, hi = _mean_ci(group["mean_contribution"])
                    all_credit_summary.append(
                        {
                            "family": family,
                            "experiment_id": experiment_id,
                            "credit_score": credit_score,
                            "arm": arm,
                            "n_seeds": n,
                            "mean_monthly_contribution_utility": mean,
                            "seed_sd": sd,
                            "ci_low": lo,
                            "ci_high": hi,
                            "fraction_seed_means_positive": float(
                                (group["mean_contribution"] > 0).mean()
                            ),
                        }
                    )

        if not posterior.empty:
            for credit_score in (
                "anchored_loo_utility",
                "anchored_shapley_utility",
            ):
                credit_post = posterior[posterior["score"] == credit_score].copy()
                if credit_post.empty:
                    continue
                credit_post["formation_date"] = pd.to_datetime(
                    credit_post["formation_date"]
                )
                final = (
                    credit_post.sort_values("formation_date")
                    .groupby(["master_seed", "arm"], as_index=False)
                    .tail(1)
                )
                final = final.groupby("arm", as_index=False).agg(
                    mean=("weight", "mean"),
                    std=("weight", "std"),
                    min=("weight", "min"),
                    max=("weight", "max"),
                )
                final.insert(0, "credit_score", credit_score)
                final.insert(0, "experiment_id", experiment_id)
                final.insert(0, "family", family)
                all_final_posterior.append(final)

        if not diagnostics.empty:
            interaction = diagnostics[
                diagnostics["strategy"]
                == "diagnostic__anchored_shapley_interaction"
            ].copy()
            if not interaction.empty:
                interaction["shapley_pair_interaction"] = pd.to_numeric(
                    interaction["shapley_pair_interaction"], errors="coerce"
                )
                seed_pair = (
                    interaction.groupby(
                        ["master_seed", "interaction_left", "interaction_right"],
                        as_index=False,
                    )
                    .agg(
                        mean_interaction=("shapley_pair_interaction", "mean"),
                        median_interaction=("shapley_pair_interaction", "median"),
                        fraction_positive=(
                            "shapley_pair_interaction",
                            lambda s: float((s > 0).mean()),
                        ),
                        n_months=("shapley_pair_interaction", "count"),
                    )
                )
                seed_pair.insert(0, "experiment_id", experiment_id)
                seed_pair.insert(0, "family", family)
                all_interaction_seed.append(seed_pair)
                for (left, right), group in seed_pair.groupby(
                    ["interaction_left", "interaction_right"], sort=True
                ):
                    n, mean, sd, lo, hi = _mean_ci(group["mean_interaction"])
                    all_interaction_summary.append(
                        {
                            "family": family,
                            "experiment_id": experiment_id,
                            "interaction_left": left,
                            "interaction_right": right,
                            "n_seeds": n,
                            "mean_monthly_interaction": mean,
                            "seed_sd": sd,
                            "ci_low": lo,
                            "ci_high": hi,
                            "fraction_seed_means_positive": float(
                                (group["mean_interaction"] > 0).mean()
                            ),
                        }
                    )

            safe_weights_json = (
                diagnostics["safe_weights_json"]
                if "safe_weights_json" in diagnostics.columns
                else pd.Series(np.nan, index=diagnostics.index, dtype=object)
            )
            for credit_score in (
                "anchored_loo_utility",
                "anchored_shapley_utility",
            ):
                treatment = f"{credit_score}__{action}"
                action_rows = diagnostics[
                    (diagnostics["strategy"] == treatment)
                    & safe_weights_json.notna()
                ].copy()
                if action_rows.empty:
                    continue
                action_rows["formation_date"] = pd.to_datetime(
                    action_rows["formation_date"]
                )
                weight_records = []
                for row in action_rows.itertuples(index=False):
                    payload = json.loads(str(row.safe_weights_json))
                    for arm, weight in payload.items():
                        weight_records.append(
                            {
                                "master_seed": int(row.master_seed),
                                "formation_date": row.formation_date,
                                "arm": str(arm),
                                "effective_action_weight": float(weight),
                            }
                        )
                effective = pd.DataFrame(weight_records)
                final_effective = (
                    effective.sort_values("formation_date")
                    .groupby(["master_seed", "arm"], as_index=False)
                    .tail(1)
                    .rename(
                        columns={
                            "effective_action_weight": (
                                "final_effective_action_weight"
                            )
                        }
                    )
                    [["master_seed", "arm", "final_effective_action_weight"]]
                )
                seed_effective = (
                    effective.groupby(["master_seed", "arm"], as_index=False)
                    .agg(
                        mean_effective_action_weight=(
                            "effective_action_weight",
                            "mean",
                        ),
                        maximum_effective_action_weight=(
                            "effective_action_weight",
                            "max",
                        ),
                    )
                    .merge(final_effective, on=["master_seed", "arm"], how="left")
                )
                seed_effective.insert(0, "credit_score", credit_score)
                seed_effective.insert(0, "experiment_id", experiment_id)
                seed_effective.insert(0, "family", family)
                all_effective_seed.append(seed_effective)
                for arm, group in seed_effective.groupby("arm", sort=True):
                    for measure in (
                        "mean_effective_action_weight",
                        "final_effective_action_weight",
                    ):
                        n, mean, sd, lo, hi = _mean_ci(group[measure])
                        all_effective_summary.append(
                            {
                                "family": family,
                                "experiment_id": experiment_id,
                                "credit_score": credit_score,
                                "arm": arm,
                                "measure": measure,
                                "n_seeds": n,
                                "mean": mean,
                                "seed_sd": sd,
                                "ci_low": lo,
                                "ci_high": hi,
                            }
                        )

        if not diagnostics.empty and "shapley_efficiency_gap" in diagnostics.columns:
            diag = diagnostics[
                diagnostics["strategy"] == "diagnostic__anchored_shapley_coalition"
            ].copy()
            if not diag.empty:
                all_efficiency.append(
                    {
                        "family": family,
                        "experiment_id": experiment_id,
                        "max_abs_shapley_efficiency_gap": float(
                            pd.to_numeric(diag["shapley_efficiency_gap"], errors="coerce")
                            .abs()
                            .max()
                        ),
                        "n_coalition_rows": int(len(diag)),
                    }
                )

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summaries_all = pd.concat(all_metric_summaries, ignore_index=True)
    paired_all = pd.concat(all_paired, ignore_index=True) if all_paired else pd.DataFrame()
    paired_summary_all = pd.DataFrame(all_paired_summary)
    credit_seed_all = (
        pd.concat(all_credit_seed, ignore_index=True)
        if all_credit_seed
        else pd.DataFrame()
    )
    credit_summary_all = pd.DataFrame(all_credit_summary)
    interaction_seed_all = (
        pd.concat(all_interaction_seed, ignore_index=True)
        if all_interaction_seed
        else pd.DataFrame()
    )
    interaction_summary_all = pd.DataFrame(all_interaction_summary)
    final_posterior_all = (
        pd.concat(all_final_posterior, ignore_index=True)
        if all_final_posterior
        else pd.DataFrame()
    )
    effective_seed_all = (
        pd.concat(all_effective_seed, ignore_index=True)
        if all_effective_seed
        else pd.DataFrame()
    )
    effective_summary_all = pd.DataFrame(all_effective_summary)
    efficiency_all = pd.DataFrame(all_efficiency)

    metrics_all.to_csv(summary_root / "metrics_by_seed_strategy.csv", index=False)
    summaries_all.to_csv(summary_root / "metric_summary_across_seeds.csv", index=False)
    paired_all.to_csv(summary_root / "paired_seed_differences.csv", index=False)
    paired_summary_all.to_csv(
        summary_root / "paired_seed_difference_summary.csv", index=False
    )
    credit_seed_all.to_csv(
        summary_root / "action_consistent_contribution_by_seed_arm.csv",
        index=False,
    )
    credit_summary_all.to_csv(
        summary_root / "action_consistent_contribution_summary.csv",
        index=False,
    )
    for credit_score, prefix in (
        ("anchored_loo_utility", "loo"),
        ("anchored_shapley_utility", "shapley"),
    ):
        seed_subset = (
            credit_seed_all[credit_seed_all["credit_score"] == credit_score]
            if not credit_seed_all.empty
            else pd.DataFrame()
        )
        summary_subset = (
            credit_summary_all[
                credit_summary_all["credit_score"] == credit_score
            ]
            if not credit_summary_all.empty
            else pd.DataFrame()
        )
        seed_subset.to_csv(
            summary_root / f"{prefix}_contribution_by_seed_arm.csv", index=False
        )
        summary_subset.to_csv(
            summary_root / f"{prefix}_contribution_summary.csv", index=False
        )
    interaction_seed_all.to_csv(
        summary_root / "shapley_pair_interaction_by_seed.csv", index=False
    )
    interaction_summary_all.to_csv(
        summary_root / "shapley_pair_interaction_summary.csv", index=False
    )
    final_posterior_all.to_csv(
        summary_root / "final_action_consistent_posterior_summary.csv",
        index=False,
    )
    for credit_score, prefix in (
        ("anchored_loo_utility", "loo"),
        ("anchored_shapley_utility", "shapley"),
    ):
        subset = (
            final_posterior_all[
                final_posterior_all["credit_score"] == credit_score
            ]
            if not final_posterior_all.empty
            else pd.DataFrame()
        )
        subset.to_csv(
            summary_root / f"final_{prefix}_posterior_summary.csv", index=False
        )
    effective_seed_all.to_csv(
        summary_root / "effective_action_weights_by_seed_arm.csv", index=False
    )
    effective_summary_all.to_csv(
        summary_root / "effective_action_weight_summary.csv", index=False
    )
    efficiency_all.to_csv(summary_root / "shapley_efficiency_check.csv", index=False)

    sharpe = summaries_all[
        (summaries_all["metric"] == "excess_sharpe")
        & (summaries_all["period"] == "full")
    ].sort_values("mean", ascending=False)
    sharpe.to_csv(summary_root / "sharpe_ranking.csv", index=False)

    print("\nSharpe ranking")
    print(
        sharpe[
            ["family", "experiment_id", "strategy", "n_seeds", "mean", "ci_low", "ci_high"]
        ].head(40).to_string(index=False)
    )
    if not credit_summary_all.empty:
        print("\nMean action-consistent utility contributions (seed means)")
        print(credit_summary_all.to_string(index=False))
    if not interaction_summary_all.empty:
        print("\nMean pairwise Shapley interactions (seed means)")
        print(interaction_summary_all.to_string(index=False))
    print(f"\nSummary files written to {summary_root}")


if __name__ == "__main__":
    main()
