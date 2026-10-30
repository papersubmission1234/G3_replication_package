#!/usr/bin/env python3
"""Summarize the ordered CVaR-map decomposition.

For each seed, the endogenous mix-minus-hard effect is the sum of:
- objective/pooling: the effect under the full Hier map;
- risk distribution: the budget-only effect minus the full-map effect;
- budget threshold: the endogenous effect minus the budget-only effect.

These components depend on the chosen order and are descriptive, not causal.
Tables are saved under decompositions/full_risk_map/summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from _common import ROOT
from g3_confirmatory.configuration import (
    experiment_config_from_mapping,
    load_yaml,
    study_paths,
)
from g3_confirmatory.metrics import metrics_by_seed_strategy


ACTION_EXPERIMENTS = {
    "predictive_log": {
        "endogenous": "predictive_log__action_endogenous",
        "hier_budget_only": "predictive_log__action_hier_budget_only",
        "hier_full_risk": "predictive_log__action_hier_full_risk",
    },
    "post_cost_utility": {
        "endogenous": "post_cost_utility__loss_endogenous__action_endogenous",
        "hier_budget_only": "post_cost_utility__loss_endogenous__action_hier_budget_only",
        "hier_full_risk": "post_cost_utility__loss_endogenous__action_hier_full_risk",
    },
}

UTILITY_LOSS_EXPERIMENTS = {
    "endogenous": "post_cost_utility__loss_endogenous__action_hier_full_risk",
    "hier_budget_only": "post_cost_utility__loss_hier_budget_only__action_hier_full_risk",
    "hier_full_risk": "post_cost_utility__loss_hier_full_risk__action_hier_full_risk",
}

METRICS = (
    "excess_sharpe",
    "annualized_excess_mean",
    "certainty_equivalent_annualized",
    "maximum_drawdown",
    "mean_one_way_turnover_ex_initial",
    "terminal_wealth_from_1000",
)


def _read_seed_files(experiment_root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(experiment_root.glob(f"seed_*/{filename}"))
    if not paths:
        return pd.DataFrame()
    header = pd.read_csv(paths[0], nrows=0).columns
    parse_dates = [
        name for name in ("formation_date", "outcome_date") if name in header
    ]
    return pd.concat(
        [pd.read_csv(path, parse_dates=parse_dates) for path in paths],
        ignore_index=True,
    )


def _mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    n = int(x.size)
    if n == 0:
        return {
            "n_seeds": 0,
            "mean": np.nan,
            "sd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "fraction_positive": np.nan,
        }
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        half = np.nan
    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half if np.isfinite(half) else np.nan,
        "ci_high": mean + half if np.isfinite(half) else np.nan,
        "fraction_positive": float(np.mean(x > 0.0)),
    }


def _load_metrics(root: Path, utility_gamma: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_root in sorted(path for path in root.iterdir() if path.is_dir()):
        returns = _read_seed_files(experiment_root, "returns.csv")
        if returns.empty:
            continue
        metrics = metrics_by_seed_strategy(
            returns,
            utility_gamma=utility_gamma,
            period_label="full",
        )
        metrics.insert(0, "experiment_id", experiment_root.name)
        frames.append(metrics)
    if not frames:
        raise FileNotFoundError(
            f"No completed seed_*/returns.csv files found under {root}."
        )
    return pd.concat(frames, ignore_index=True)


def _cell_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = metrics[metrics["strategy"].str.contains(
        "__hard|__mix|standalone__M_HierMacro|fixed_equal__mix", regex=True
    )]
    for keys, group in subset.groupby(["experiment_id", "strategy"], sort=True):
        experiment_id, strategy = keys
        for metric in METRICS:
            summary = _mean_ci(group[metric])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "strategy": strategy,
                    "metric": metric,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _action_deltas(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for score, experiments in ACTION_EXPERIMENTS.items():
        hard = f"{score}__hard"
        mix = f"{score}__mix"
        for map_label, experiment_id in experiments.items():
            block = metrics[
                (metrics["experiment_id"] == experiment_id)
                & (metrics["strategy"].isin([hard, mix]))
            ]
            for metric in METRICS:
                pivot = block.pivot(
                    index="master_seed", columns="strategy", values=metric
                )
                if not {hard, mix}.issubset(pivot.columns):
                    continue
                delta = pivot[mix] - pivot[hard]
                for seed, value in delta.items():
                    rows.append(
                        {
                            "score": score,
                            "action_map": map_label,
                            "master_seed": int(seed),
                            "metric": metric,
                            "mix_minus_hard": float(value),
                        }
                    )
    by_seed = pd.DataFrame(rows)
    summaries = []
    for keys, group in by_seed.groupby(["score", "action_map", "metric"]):
        score, action_map, metric = keys
        summaries.append(
            {
                "score": score,
                "action_map": action_map,
                "metric": metric,
                **_mean_ci(group["mix_minus_hard"]),
            }
        )
    return by_seed, pd.DataFrame(summaries)


def _channel_decomposition(
    action_deltas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (score, metric), group in action_deltas.groupby(["score", "metric"]):
        pivot = group.pivot(
            index="master_seed", columns="action_map", values="mix_minus_hard"
        ).dropna(
            subset=["endogenous", "hier_budget_only", "hier_full_risk"]
        )
        for seed, values in pivot.iterrows():
            full_risk = float(values["hier_full_risk"])
            budget_only = float(values["hier_budget_only"])
            endogenous = float(values["endogenous"])
            components = {
                "objective_pooling_channel": full_risk,
                "risk_distribution_channel": budget_only - full_risk,
                "budget_threshold_channel": endogenous - budget_only,
                "total_endogenous_mix_minus_hard": endogenous,
            }
            reconstruction = (
                components["objective_pooling_channel"]
                + components["risk_distribution_channel"]
                + components["budget_threshold_channel"]
            )
            for component, value in components.items():
                rows.append(
                    {
                        "score": score,
                        "master_seed": int(seed),
                        "metric": metric,
                        "component": component,
                        "value": value,
                        "reconstruction_gap": reconstruction - endogenous,
                    }
                )
    by_seed = pd.DataFrame(rows)
    summaries = []
    for keys, group in by_seed.groupby(["score", "metric", "component"]):
        score, metric, component = keys
        summaries.append(
            {
                "score": score,
                "metric": metric,
                "component": component,
                **_mean_ci(group["value"]),
                "max_abs_reconstruction_gap": float(
                    group["reconstruction_gap"].abs().max()
                ),
            }
        )
    return by_seed, pd.DataFrame(summaries)


def _loss_map_posterior(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for loss_map, experiment_id in UTILITY_LOSS_EXPERIMENTS.items():
        posterior = _read_seed_files(root / experiment_id, "posterior.csv")
        if posterior.empty:
            continue
        posterior["formation_date"] = pd.to_datetime(posterior["formation_date"])
        final = (
            posterior.sort_values("formation_date")
            .groupby(["master_seed", "score", "arm"], as_index=False)
            .tail(1)
        )
        final = final[
            (final["score"] == "post_cost_utility")
            & (final["arm"] == "M_HierMacro")
        ]
        for record in final.itertuples(index=False):
            rows.append(
                {
                    "loss_map": loss_map,
                    "master_seed": int(record.master_seed),
                    "final_hier_weight": float(record.weight),
                }
            )
    by_seed = pd.DataFrame(rows)
    summaries = []
    for loss_map, group in by_seed.groupby("loss_map"):
        summaries.append(
            {
                "loss_map": loss_map,
                **_mean_ci(group["final_hier_weight"]),
            }
        )
    return by_seed, pd.DataFrame(summaries)


def _loss_map_performance(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    maps = list(UTILITY_LOSS_EXPERIMENTS)
    for strategy in ("post_cost_utility__hard", "post_cost_utility__mix"):
        for metric in METRICS:
            pieces = []
            for loss_map, experiment_id in UTILITY_LOSS_EXPERIMENTS.items():
                block = metrics[
                    (metrics["experiment_id"] == experiment_id)
                    & (metrics["strategy"] == strategy)
                ][["master_seed", metric]].rename(columns={metric: loss_map})
                pieces.append(block.set_index("master_seed"))
            if not pieces:
                continue
            panel = pd.concat(pieces, axis=1).dropna(subset=maps)
            for seed, values in panel.iterrows():
                for treatment, control in (
                    ("hier_budget_only", "endogenous"),
                    ("hier_full_risk", "hier_budget_only"),
                    ("hier_full_risk", "endogenous"),
                ):
                    rows.append(
                        {
                            "strategy": strategy,
                            "metric": metric,
                            "treatment_loss_map": treatment,
                            "control_loss_map": control,
                            "master_seed": int(seed),
                            "difference": float(values[treatment] - values[control]),
                        }
                    )
    by_seed = pd.DataFrame(rows)
    summaries = []
    for keys, group in by_seed.groupby(
        ["strategy", "metric", "treatment_loss_map", "control_loss_map"]
    ):
        strategy, metric, treatment, control = keys
        summaries.append(
            {
                "strategy": strategy,
                "metric": metric,
                "treatment_loss_map": treatment,
                "control_loss_map": control,
                **_mean_ci(group["difference"]),
            }
        )
    return by_seed, pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "confirmatory.yaml")
    )
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    paths = study_paths(raw_cfg)
    base = experiment_config_from_mapping(raw_cfg.get("experiment"))
    root = paths["output_root"] / "decompositions" / "full_risk_map"
    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(root, base.utility_gamma)
    cells = _cell_summary(metrics)
    action_by_seed, action_summary = _action_deltas(metrics)
    channels_by_seed, channels_summary = _channel_decomposition(action_by_seed)
    posterior_by_seed, posterior_summary = _loss_map_posterior(root)
    loss_perf_by_seed, loss_perf_summary = _loss_map_performance(metrics)

    metrics.to_csv(summary_root / "metrics_by_seed_strategy.csv", index=False)
    cells.to_csv(summary_root / "cell_metric_summary.csv", index=False)
    action_by_seed.to_csv(summary_root / "action_mix_minus_hard_by_seed.csv", index=False)
    action_summary.to_csv(summary_root / "action_mix_minus_hard_summary.csv", index=False)
    channels_by_seed.to_csv(
        summary_root / "action_channel_decomposition_by_seed.csv", index=False
    )
    channels_summary.to_csv(
        summary_root / "action_channel_decomposition_summary.csv", index=False
    )
    posterior_by_seed.to_csv(
        summary_root / "utility_loss_map_final_hier_posterior_by_seed.csv",
        index=False,
    )
    posterior_summary.to_csv(
        summary_root / "utility_loss_map_final_hier_posterior_summary.csv",
        index=False,
    )
    loss_perf_by_seed.to_csv(
        summary_root / "utility_loss_map_performance_by_seed.csv", index=False
    )
    loss_perf_summary.to_csv(
        summary_root / "utility_loss_map_performance_summary.csv", index=False
    )

    print("\nMix minus hard (excess Sharpe):")
    print(
        action_summary[action_summary["metric"] == "excess_sharpe"]
        [["score", "action_map", "n_seeds", "mean", "ci_low", "ci_high"]]
        .sort_values(["score", "action_map"])
        .to_string(index=False)
    )
    print("\nPathwise action-channel decomposition (excess Sharpe):")
    print(
        channels_summary[channels_summary["metric"] == "excess_sharpe"]
        [["score", "component", "n_seeds", "mean", "ci_low", "ci_high"]]
        .sort_values(["score", "component"])
        .to_string(index=False)
    )
    if not posterior_summary.empty:
        print("\nFinal HierMacro posterior under utility loss maps:")
        print(
            posterior_summary[
                ["loss_map", "n_seeds", "mean", "ci_low", "ci_high"]
            ]
            .sort_values("loss_map")
            .to_string(index=False)
        )
    print(f"\nSummary files written to {summary_root}")


if __name__ == "__main__":
    main()
