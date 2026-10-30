#!/usr/bin/env python3
"""Build the complete main-text and electronic-companion figure set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT, read_seed_table
from g3_confirmatory.configuration import load_yaml, study_paths
from g3_confirmatory.metrics import metrics_by_seed_strategy
from g3_confirmatory.reporting import safe_read_csv

SELECTED_WEALTH = [
    "standalone__M_HierMacro",
    "post_cost_utility__mix",
    "fixed_equal__mix",
    "post_cost_utility__hard",
    "predictive_log__hard",
    "benchmark__1N",
]


def _save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _posterior_figure(output_root: Path, figures: Path) -> bool:
    core = output_root / "core"
    try:
        posterior = read_seed_table(core, "posterior.csv")
    except FileNotFoundError:
        return False
    posterior["formation_date"] = pd.to_datetime(posterior["formation_date"])
    frame = posterior[posterior["arm"] == "M_HierMacro"].copy()
    if frame.empty:
        return False
    summary = frame.groupby(["formation_date", "score"])["weight"].agg(
        mean="mean",
        low=lambda x: np.quantile(x, 0.025),
        high=lambda x: np.quantile(x, 0.975),
    ).reset_index()
    summary["score"] = summary["score"].replace({
        "post_cost_utility": "Utility score",
        "predictive_log": "Log score"
    })
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for score, group in summary.groupby("score"):
        line, = ax.plot(group["formation_date"], group["mean"], label=score)
        ax.fill_between(
            group["formation_date"], group["low"], group["high"],
            alpha=0.15, color=line.get_color(),
        )
    ax.axhline(0.5, linewidth=0.8, linestyle="--", color="black")
    ax.set_ylabel("Posterior weight on M_HierMacro")
    ax.set_xlabel("Formation date")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    _save(fig, figures, "figure_01_posterior_paths")
    return True


def _wealth_figure(output_root: Path, figures: Path) -> bool:
    core = output_root / "core"
    try:
        returns = read_seed_table(core, "returns.csv")
    except FileNotFoundError:
        return False
    returns["outcome_date"] = pd.to_datetime(returns["outcome_date"])
    frame = returns[returns["strategy"].isin(SELECTED_WEALTH)].copy()
    if frame.empty:
        return False
    summary = frame.groupby(["outcome_date", "strategy"])["wealth"].agg(
        mean="mean",
        low=lambda x: np.quantile(x, 0.025),
        high=lambda x: np.quantile(x, 0.975),
    ).reset_index()
    summary["strategy"] = summary["strategy"].replace({
        "benchmark__1N": "1/N",
        "fixed_equal__mix": "Fixed pool",
        "post_cost_utility__hard": "Utility weights + model sampling",
        "post_cost_utility__mix": "Utility weights + linear pool",
        "predictive_log__hard": "Log weights + model sampling",
        "standalone__M_HierMacro": "HierMacro expert"
    })
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for strategy, group in summary.groupby("strategy"):
        line, = ax.plot(group["outcome_date"], group["mean"], label=strategy)
        ax.fill_between(
            group["outcome_date"], group["low"], group["high"],
            alpha=0.08, color=line.get_color(),
        )
    ax.axhline(1000, linewidth=0.8, linestyle="--", color="black")
    ax.set_ylabel("Net wealth (initial 1,000)")
    ax.set_xlabel("Outcome date")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, figures, "figure_02_wealth_paths")
    return True


def _seed_effect_figure(output_root: Path, figures: Path) -> bool:
    path = output_root / "inference" / "primary_paired_seed_differences.csv"
    frame = safe_read_csv(path)
    if frame.empty:
        return False
    frame = frame[(frame["period"] == "full") & (frame["metric"] == "excess_sharpe")]
    if frame.empty:
        return False
    frame["comparison"] = frame["comparison"].replace({
        "hier_vs_1N": "HierMacro - 1/N",
        "mix_vs_1N": "Utility pool - 1/N",
        "mix_vs_hier": "Utility pool - HierMacro",
        "adaptive_vs_equal_pool": "Utility pool - fixed pool",
        "action_under_utility": "Pool - sampling (utility)",
        "score_under_hard": "Utility - log (sampling)",
    })
    labels = list(dict.fromkeys(frame["comparison"]))
    values = [frame.loc[frame["comparison"] == label, "difference"].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.55 * len(labels))))
    ax.boxplot(values, vert=False, tick_labels=labels, showmeans=True)
    ax.axvline(0.0, linewidth=0.8, linestyle="--", color="black")
    ax.set_xlabel("Paired excess-Sharpe difference across seeds")
    _save(fig, figures, "figure_03_score_action_seed_effects")
    return True


def _mean_ablation_figure(output_root: Path, figures: Path) -> bool:
    root = output_root / "mean_ablation"
    if not root.exists():
        return False
    records = []
    for treatment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = sorted(treatment_dir.glob("seed_*/returns.csv"))
        if not paths:
            continue
        returns = pd.concat(
            [pd.read_csv(path, parse_dates=["formation_date", "outcome_date"]) for path in paths],
            ignore_index=True,
        )
        metrics = metrics_by_seed_strategy(returns)
        selected = metrics[metrics["strategy"] == "post_cost_utility__mix"]
        for row in selected.itertuples(index=False):
            records.append(
                {
                    "treatment": treatment_dir.name,
                    "excess_sharpe": row.excess_sharpe,
                    "master_seed": row.master_seed,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return False
    order = list(dict.fromkeys(frame["treatment"]))
    values = [frame.loc[frame["treatment"] == name, "excess_sharpe"].to_numpy() for name in order]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(values, tick_labels=order, showmeans=True)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Excess Sharpe under utility scenario mixing")
    ax.set_xlabel("Hierarchical mean treatment")
    _save(fig, figures, "figure_04_mean_suppression_ablation")
    return True


def _riskmap_channels_figure(output_root: Path, figures: Path) -> bool:
    path = (
        output_root / "decompositions" / "full_risk_map" / "summary"
        / "action_channel_decomposition_summary.csv"
    )
    frame = safe_read_csv(path)
    if frame.empty:
        return False
    frame = frame[
        (frame["metric"] == "excess_sharpe")
        & (frame["component"].isin([
            "objective_pooling_channel",
            "risk_distribution_channel",
            "budget_threshold_channel",
        ]))
    ].copy()
    if frame.empty:
        return False
    scores = list(dict.fromkeys(frame["score"]))
    components = [
        "objective_pooling_channel",
        "risk_distribution_channel",
        "budget_threshold_channel",
    ]
    labels = [component.replace("_channel", "").replace("_", " ") for component in components]
    x = np.arange(len(components))
    width = 0.8 / max(1, len(scores))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for index, score in enumerate(scores):
        group = frame[frame["score"] == score].set_index("component").reindex(components)
        means = group["mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        errors = np.vstack([means - low, high - means])
        ax.bar(x + (index - (len(scores) - 1) / 2) * width, means, width=width,
               yerr=errors, capsize=3, label=score)
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Paired excess-Sharpe contribution")
    ax.set_xlabel("Action-effect channel")
    ax.legend(frameon=False)
    _save(fig, figures, "figure_05_score_action_riskmap_channels")
    return True


def _riskmap_action_figure(output_root: Path, figures: Path) -> bool:
    path = (
        output_root / "decompositions" / "full_risk_map" / "summary"
        / "action_mix_minus_hard_summary.csv"
    )
    frame = safe_read_csv(path)
    if frame.empty:
        return False
    frame = frame[frame["metric"] == "excess_sharpe"].copy()
    if frame.empty:
        return False
    frame["label"] = frame["score"].astype(str) + " | " + frame["action_map"].astype(str)
    frame = frame.sort_values(["score", "action_map"])
    means = frame["mean"].to_numpy(dtype=float)
    errors = np.vstack([
        means - frame["ci_low"].to_numpy(dtype=float),
        frame["ci_high"].to_numpy(dtype=float) - means,
    ])
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.48 * len(frame))))
    ax.errorbar(means, y, xerr=errors, fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("Scenario mix minus hard sampling: excess Sharpe")
    _save(fig, figures, "figure_06_action_effect_by_risk_map")
    return True


def _shapley_contribution_figure(output_root: Path, figures: Path) -> bool:
    root = output_root / "decompositions" / "decision_shapley" / "summary"
    frame = safe_read_csv(root / "shapley_contribution_summary.csv")
    if frame.empty:
        return False
    primary = frame[
        frame["experiment_id"].astype(str).str.contains("0p1", regex=False)
    ].copy()
    if primary.empty:
        primary = frame.copy()
    if "mean_monthly_contribution_utility" not in primary:
        return False
    primary = primary.sort_values(["credit_score", "arm"])
    labels = primary["credit_score"].astype(str) + " | " + primary["arm"].astype(str)
    means = primary["mean_monthly_contribution_utility"].to_numpy(dtype=float)
    errors = np.vstack([
        means - primary["ci_low"].to_numpy(dtype=float),
        primary["ci_high"].to_numpy(dtype=float) - means,
    ])
    y = np.arange(len(primary))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.48 * len(primary))))
    ax.errorbar(means, y, xerr=errors, fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean monthly decision-Shapley utility contribution")
    _save(fig, figures, "figure_07_decision_shapley_contributions")
    return True


def _shapley_weight_figure(output_root: Path, figures: Path) -> bool:
    root = output_root / "decompositions" / "decision_shapley" / "summary"
    frame = safe_read_csv(root / "effective_action_weight_summary.csv")
    if frame.empty:
        return False
    frame = frame[
        (frame["measure"] == "mean_effective_action_weight")
        & frame["experiment_id"].astype(str).str.contains("0p1", regex=False)
        & frame["credit_score"].astype(str).str.contains("shapley", case=False, regex=False)
    ].copy()
    if frame.empty:
        return False
    frame = frame.sort_values("arm")
    means = frame["mean"].to_numpy(dtype=float)
    errors = np.vstack([
        means - frame["ci_low"].to_numpy(dtype=float),
        frame["ci_high"].to_numpy(dtype=float) - means,
    ])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(frame["arm"], means, yerr=errors, capsize=3)
    ax.set_ylabel("Mean effective action weight")
    ax.set_ylim(0.0, min(1.05, max(1.0, float(np.nanmax(means + errors[1])) * 1.1)))
    ax.tick_params(axis="x", rotation=15)
    _save(fig, figures, "figure_08_shapley_effective_action_weights")
    return True


def _eligibility_figure(output_root: Path, figures: Path) -> bool:
    path = output_root.parent / "data" / "eligibility_primary" / "eligibility_audit_by_date.csv"
    frame = safe_read_csv(path, parse_dates=["formation_date", "outcome_date"])
    if frame.empty or "n_eligible" not in frame:
        return False
    date_column = "formation_date" if "formation_date" in frame else frame.columns[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(pd.to_datetime(frame[date_column]), frame["n_eligible"])
    ax.set_ylabel("Eligible ETFs")
    ax.set_xlabel("Formation date")
    _save(fig, figures, "figure_09_point_in_time_universe_size")
    return True


def _cvar_calibration_figure(output_root: Path, figures: Path) -> bool:
    path = output_root / "risk_diagnostics" / "monthly_cvar_calibration_summary.csv"
    frame = safe_read_csv(path)
    if frame.empty:
        return False
    candidate = next(
        (column for column in [
            "net_var_calibration_gap",
            "gross_var_calibration_gap",
            "net_tail_severity_gap",
            "gross_tail_severity_gap",
            "cvar_budget_binding_rate",
        ] if column in frame.columns),
        None,
    )
    if candidate is None:
        return False
    label_cols = [column for column in ["experiment_id", "strategy"] if column in frame]
    if not label_cols:
        return False
    frame["label"] = frame[label_cols].astype(str).agg(" | ".join, axis=1)
    frame = frame.sort_values(candidate).tail(30)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.35 * len(frame))))
    ax.barh(y, frame[candidate])
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, frame["label"], fontsize=7)
    ax.set_xlabel(candidate.replace("_", " "))
    if "var_calibration_gap" in candidate:
        ax.set_title("Monthly VaR exceedance-rate gap; CVaR severity is reported in the table")
    _save(fig, figures, "figure_10_monthly_cvar_calibration")
    return True


def _robustness_figure(output_root: Path, figures: Path) -> bool:
    table_path = output_root / "paper_tables" / "main_13_selected_robustness.csv"
    frame = safe_read_csv(table_path)
    if frame.empty:
        return False
    frame = frame[
        (frame["metric"] == "excess_sharpe")
        & (frame["strategy"] == "post_cost_utility__mix")
    ].copy()
    if frame.empty:
        return False
    frame = frame.sort_values("mean")
    y = np.arange(len(frame))
    means = frame["mean"].to_numpy(dtype=float)
    errors = np.vstack([
        means - frame["ci_low"].to_numpy(dtype=float),
        frame["ci_high"].to_numpy(dtype=float) - means,
    ])
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.38 * len(frame))))
    ax.errorbar(means, y, xerr=errors, fmt="o", capsize=3)
    ax.set_yticks(y, frame["robustness_configuration"], fontsize=7)
    ax.set_xlabel("Utility-mix excess Sharpe across robustness configurations")
    _save(fig, figures, "figure_11_robustness_forest")
    return True


def _daily_risk_figure(output_root: Path, figures: Path) -> bool:
    path = output_root / "risk_diagnostics" / "daily_risk_metric_summary.csv"
    frame = safe_read_csv(path)
    if frame.empty or "metric" not in frame:
        return False
    frame = frame[frame["metric"].astype(str).str.contains("daily_es_95_net", regex=False)].copy()
    if frame.empty:
        return False
    label_cols = [column for column in ["experiment_id", "strategy"] if column in frame]
    frame["label"] = frame[label_cols].astype(str).agg(" | ".join, axis=1)
    frame = frame.sort_values("mean").tail(25)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.36 * len(frame))))
    ax.barh(y, frame["mean"])
    ax.set_yticks(y, frame["label"], fontsize=7)
    ax.set_xlabel("Ex-post one-day 95% expected shortfall (diagnostic)")
    _save(fig, figures, "figure_12_daily_es_diagnostic")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    figures = paths["output_root"] / "paper_figures"
    figures.mkdir(parents=True, exist_ok=True)

    builders: list[tuple[str, Callable[[Path, Path], bool], str]] = [
        ("figure_01_posterior_paths", _posterior_figure, "main"),
        ("figure_02_wealth_paths", _wealth_figure, "main"),
        ("figure_03_score_action_seed_effects", _seed_effect_figure, "main"),
        ("figure_04_mean_suppression_ablation", _mean_ablation_figure, "main"),
        ("figure_05_score_action_riskmap_channels", _riskmap_channels_figure, "main"),
        ("figure_06_action_effect_by_risk_map", _riskmap_action_figure, "main"),
        ("figure_07_decision_shapley_contributions", _shapley_contribution_figure, "main_or_appendix"),
        ("figure_08_shapley_effective_action_weights", _shapley_weight_figure, "appendix"),
        ("figure_09_point_in_time_universe_size", _eligibility_figure, "appendix"),
        ("figure_10_monthly_cvar_calibration", _cvar_calibration_figure, "appendix"),
        ("figure_11_robustness_forest", _robustness_figure, "appendix"),
        ("figure_12_daily_es_diagnostic", _daily_risk_figure, "appendix"),
    ]
    records = []
    for figure_id, builder, classification in builders:
        try:
            available = bool(builder(paths["output_root"], figures))
            error = ""
        except Exception as exc:
            available = False
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "figure_id": figure_id,
                "classification": classification,
                "available": available,
                "png": f"{figure_id}.png" if available else "",
                "pdf": f"{figure_id}.pdf" if available else "",
                "error": error,
            }
        )
    catalog = pd.DataFrame(records)
    catalog.to_csv(figures / "figure_catalog.csv", index=False)
    status = {
        "config": str(Path(args.config).resolve()),
        "n_figures_requested": int(len(catalog)),
        "n_available": int(catalog["available"].sum()),
        "missing": catalog.loc[~catalog["available"], "figure_id"].tolist(),
    }
    (figures / "figure_build_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    print(f"Figures written to {figures}")


if __name__ == "__main__":
    main()
