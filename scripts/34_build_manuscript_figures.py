#!/usr/bin/env python3
"""Draw manuscript Figure 6 and Figures S1, S2, S5 from saved summaries."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, load_context
from g3_confirmatory.reporting import (
    checked_read, reporting_context, resolve, select_one, sources,
    validate_interval, write_csv, write_json,
)

NAMES = {
    "6": "figure_06_ordered_action_channels",
    "S1": "figure_S01_mean_treatments",
    "S2": "figure_S02_shapley_attribution",
    "S5": "figure_S05_daily_es_horizons",
}


def selected_plot_row(row, label, panel, factor=1.0, mean_column="mean"):
    return {"panel": panel, "label": label,
            "mean": float(row[mean_column]) * factor,
            "ci_low": float(row["ci_low"]) * factor,
            "ci_high": float(row["ci_high"]) * factor,
            "n_seeds": int(row["n_seeds"])}


def load_channels(root: Path, expected_seeds: int):
    path = root / "decompositions/full_risk_map/summary/action_channel_decomposition_summary.csv"
    frame = checked_read(path, ["score", "metric", "component", "mean", "ci_low", "ci_high", "n_seeds"])
    rows = []
    for metric, factor, panel in [("annualized_excess_mean", 100.0, "Annualized excess return (pp)"),
                                  ("excess_sharpe", 1.0, "Excess Sharpe ratio")]:
        for component, label in [("objective_pooling_channel", "Return forecast"),
                                 ("risk_distribution_channel", "Risk distribution"),
                                 ("budget_threshold_channel", "Budget")]:
            row = select_one(frame, score="post_cost_utility", metric=metric, component=component)
            validate_interval(row, expected_seeds)
            rows.append({"component": component, "metric": metric,
                         **selected_plot_row(row, label, panel, factor)})
    return pd.DataFrame(rows), [path]


def load_mean_treatments(root: Path, expected_seeds: int):
    path = root / "paper_tables/manuscript_support/table_09_mean_ablation_all_strategies.csv"
    frame = checked_read(path, ["experiment", "portfolio_strategy", "mean", "ci_low", "ci_high", "n_seeds"])
    treatments = [
        ("zero_centered_factor_innovations", "Zero-centered\ninnovations"),
        ("base_empirical", "Base"), ("posterior_ar1_mean", "Posterior\nAR(1) mean"),
        ("literal_zero_total_mean", "Zero total\nmean"),
        ("sampled_factor_mean", "Sampled\nfactor mean"),
    ]
    rows = []
    for treatment, label in treatments:
        row = select_one(frame, experiment=f"mean_ablation/{treatment}",
                         portfolio_strategy="post_cost_utility__mix")
        validate_interval(row, expected_seeds)
        rows.append({"treatment": treatment, **selected_plot_row(row, label, "Excess Sharpe ratio")})
    return pd.DataFrame(rows), [path]


def load_shapley(root: Path, expected_seeds: int):
    directory = root / "decompositions/decision_shapley/summary"
    contribution_path = directory / "decision_shapley_contribution_summary.csv"
    if not contribution_path.exists():
        contribution_path = directory / "shapley_contribution_summary.csv"
    weight_path = directory / "effective_action_weight_summary.csv"
    keys = ["family", "experiment_id", "credit_score", "arm", "n_seeds", "ci_low", "ci_high"]
    contribution = checked_read(contribution_path, keys + ["mean_monthly_contribution_utility"])
    weights = checked_read(weight_path, keys + ["mean", "measure"])
    select = dict(family="mean", experiment_id="directional_mean_cap_0p1",
                  credit_score="anchored_shapley_utility")
    rows = []
    for arm, label in [("M_FF5MOM", "FF5MOM"), ("M_Momentum", "Momentum")]:
        row = select_one(contribution, **select, arm=arm)
        validate_interval(row, expected_seeds, "mean_monthly_contribution_utility")
        rows.append({"arm": arm, **selected_plot_row(row, label, "Monthly utility contribution", 1e5,
                                                     "mean_monthly_contribution_utility")})
    baseline = select_one(contribution, **select, arm="M_HierMacro")
    validate_interval(baseline, expected_seeds, "mean_monthly_contribution_utility")
    if abs(float(baseline["mean_monthly_contribution_utility"])) > 1e-14:
        raise ValueError("The mandatory baseline must have zero Shapley attribution")
    total = 0.0
    for arm, label in [("M_HierMacro", "HierMacro"), ("M_FF5MOM", "FF5MOM"), ("M_Momentum", "Momentum")]:
        row = select_one(weights, **select, arm=arm, measure="mean_effective_action_weight")
        validate_interval(row, expected_seeds)
        total += float(row["mean"])
        rows.append({"arm": arm, **selected_plot_row(row, label, "Effective mean-input weight (%)", 100.0)})
    if not np.isclose(total, 1.0, atol=1e-10, rtol=0):
        raise ValueError("Mean effective model weights do not sum to one")
    return pd.DataFrame(rows), [contribution_path, weight_path]


def load_daily_es(root: Path, expected_seeds: int):
    path = root / "risk_diagnostics/daily_risk_metric_summary.csv"
    frame = checked_read(path, ["experiment_id", "strategy", "metric", "mean", "ci_low", "ci_high", "n_seeds"])
    strategies = [
        ("standalone__M_HierMacro", "HierMacro expert"),
        ("post_cost_utility__mix", "Utility weights + linear pool"),
        ("fixed_equal__mix", "Fixed pool"),
        ("post_cost_utility__hard", "Utility weights + model sampling"),
        ("standalone__M_FF5MOM", "FF5MOM expert"), ("benchmark__1N", "1/N"),
    ]
    rows = []
    for metric, panel in [("daily_es_95_net", "Daily ES 95%"),
                          ("daily_es_975_net", "Daily ES 97.5%"),
                          ("overlapping_21d_es_95_net", "Overlapping 21-day ES 95%")]:
        for strategy, label in strategies:
            row = select_one(frame, experiment_id="core", strategy=strategy, metric=metric)
            validate_interval(row, expected_seeds)
            rows.append({"strategy": strategy, "metric": metric,
                         **selected_plot_row(row, label, panel, 100.0)})
    return pd.DataFrame(rows), [path]


LOADERS = {"6": load_channels, "S1": load_mean_treatments, "S2": load_shapley, "S5": load_daily_es}


def errors(group: pd.DataFrame):
    mean = group["mean"].to_numpy(float)
    return np.vstack([mean - group["ci_low"].to_numpy(float),
                      group["ci_high"].to_numpy(float) - mean])


def plot_ordered_channels(frame: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), layout="constrained")
    labels = ["Return forecast\n$C_{\\mathrm{obj}}$", "Risk distribution\n$C_Q$", "Budget\n$C_B$"]
    for ax, (panel, group) in zip(axes, frame.groupby("panel", sort=False)):
        x = np.arange(len(group))
        ax.bar(x, group["mean"], yerr=errors(group), width=0.55, capsize=4,
               color=["0.30", "0.55", "0.78"], edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x, labels)
        ax.set_ylabel(panel)
        ax.grid(axis="y", alpha=0.20)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)
    return fig


def plot_mean_treatment_intervals(frame: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.3, 3.8), layout="constrained")
    x = np.arange(len(frame))
    ax.errorbar(x, frame["mean"], yerr=errors(frame), fmt="o", color="black", capsize=4)
    span = float(frame.ci_high.max() - frame.ci_low.min())
    for index, row in frame.iterrows():
        ax.text(index + 0.10, row["mean"], f"{row['mean']:.3f}", va="center", fontsize=9)
    ax.set_xticks(x, frame.label)
    ax.set_xlim(-0.4, len(frame) - 0.35)
    ax.set_ylim(frame.ci_low.min() - 0.08 * span, frame.ci_high.max() + 0.12 * span)
    ax.set_ylabel("Linear-pool excess Sharpe ratio")
    ax.set_xlabel("HierMacro mean treatment")
    ax.grid(axis="y", alpha=0.20)
    return fig


def plot_shapley_attribution(frame: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), layout="constrained")
    panels = list(frame.groupby("panel", sort=False))
    ax = axes[0]
    panel, group = panels[0]
    y = np.arange(len(group))
    ax.errorbar(group["mean"], y, xerr=errors(group), fmt="o",
                color="black", ecolor="black", capsize=3.5, markersize=5)
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_yticks(y, group.label)
    ax.set_ylim(len(group) - 0.5, -0.5)
    ax.set_xlabel("Monthly utility contribution ($\\times 10^{-5}$)")
    ax.grid(axis="x", alpha=0.20)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    ax = axes[1]
    panel, group = panels[1]
    colors = ["#4D4D4D", "#8C8C8C", "#C8C8C8"]
    if len(group) != len(colors):
        raise ValueError(
            f"Expected {len(colors)} weight arms, got {len(group)}: {list(group.label)}"
        )
    y = np.arange(len(group))
    ax.barh(y, group["mean"], xerr=errors(group), height=0.55, capsize=2.5,
            color=colors, edgecolor="black", linewidth=0.6)
    for yi, mean in zip(y, group["mean"]):
        ax.text(mean + 0.01 * float(group["ci_high"].max()), yi,
                f"{mean:.1f}%", va="center", fontsize=8.5)
    ax.set_yticks(y, group.label)
    ax.set_ylim(len(group) - 0.5, -0.5)
    ax.set_xlim(0, float(group["ci_high"].max()) * 1.12)
    ax.set_xlabel(panel)
    ax.grid(axis="x", alpha=0.20)
    ax.set_axisbelow(True)
    return fig


def plot_daily_es_horizons(frame: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharey=True, layout="constrained")
    for index, (ax, (panel, group)) in enumerate(zip(axes, frame.groupby("panel", sort=False))):
        y = np.arange(len(group))
        ax.barh(y, group["mean"], xerr=errors(group), height=0.58, capsize=2.5,
                color="#8C8C8C", edgecolor="black", linewidth=0.5)
        ax.set_title(panel, fontsize=10)
        ax.set_xlabel("Loss (%)")
        ax.set_yticks(y, group.label)
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.20)
        ax.set_axisbelow(True)
        if index:
            ax.tick_params(axis="y", left=False)
    axes[0].invert_yaxis()
    return fig


PLOTTERS = {"6": plot_ordered_channels, "S1": plot_mean_treatment_intervals,
            "S2": plot_shapley_attribution, "S5": plot_daily_es_horizons}


def build_figures(output_root: Path, destination: Path, requested: list[str],
                  expected_seeds: int = 30, dpi: int = 300,
                  formats=("png", "pdf"), validate_only=False) -> dict:
    loaded = {key: LOADERS[key](output_root, expected_seeds) for key in requested}
    used = [path for _, paths in loaded.values() for path in paths]
    report = {"figures": requested, "expected_seeds": expected_seeds,
              "sources": sources(used), "outputs": [], "complete": True,
              "selected_values": {key: json_safe_frame(frame) for key, (frame, _) in loaded.items()}}
    if validate_only:
        print(f"Validated {len(loaded)} figures. No files written.")
        return report
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"font.family": "serif", "font.size": 10, "axes.spines.top": False,
             "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
    outputs = []
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        for key, (frame, _) in loaded.items():
            fig = PLOTTERS[key](frame)
            for extension in formats:
                path = destination / f"{NAMES[key]}.{extension}"
                temporary = path.with_name(path.name + ".tmp")
                fig.savefig(temporary, format=extension, dpi=dpi, bbox_inches="tight", facecolor="white")
                os.replace(temporary, path)
                outputs.append(path)
            plt.close(fig)
            path = destination / "plot_data" / f"{NAMES[key]}.csv"
            write_csv(path, frame)
            outputs.append(path)
    report["outputs"] = sources(outputs)
    selection = "_".join(requested)
    write_json(destination / f"manuscript_additions_{selection}_manifest.json", report)
    print(f"Generated {len(loaded)} figures and their plotted-value CSVs: {destination}")
    return report


def json_safe_frame(frame):
    import json
    return json.loads(frame.to_json(orient="records", double_precision=15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/confirmatory.yaml")
    parser.add_argument("--output-root", type=Path, help="Use saved summaries without loading a configuration")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures", default="6,S1,S2,S5")
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    requested = [item.strip().upper() for item in args.figures.split(",") if item.strip()]
    if not requested or len(set(requested)) != len(requested) or set(requested).difference(LOADERS):
        parser.error("--figures must contain unique entries from 6,S1,S2,S5")
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    if not formats or set(formats).difference({"png", "pdf", "svg"}):
        parser.error("--formats must contain png,pdf,svg")
    if args.output_root:
        output_root = resolve(args.output_root)
        expected = args.expected_seeds if args.expected_seeds is not None else 30
    else:
        _, _, paths, seeds, _ = reporting_context(args.config)
        output_root = paths["output_root"]
        expected = args.expected_seeds if args.expected_seeds is not None else len(seeds)
    if expected < 2 or args.dpi < 1:
        parser.error("Invalid expected seed count or dpi")
    destination = resolve(args.output_dir) if args.output_dir else output_root / "paper_figures"
    build_figures(output_root, destination, requested, expected, args.dpi, formats, args.validate_only)


if __name__ == "__main__":
    main()
