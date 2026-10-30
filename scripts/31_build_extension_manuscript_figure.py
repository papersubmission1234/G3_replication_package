#!/usr/bin/env python3
"""Build Figure 11 from saved extension metrics.

Reads diagnostics/summary/extensions/extension_metrics_by_seed.csv.
Use --validate-only to check the inputs before drawing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
ESTIMATORS = ("sample", "mp", "ledoit_wolf", "ewma_ledoit_wolf")
RISK_MAPS = ("endogenous", "common_budget", "common_full")
LABELS = {
    "sample": "Sample covariance",
    "mp": "Marchenko-Pastur",
    "ledoit_wolf": "Ledoit-Wolf",
    "ewma_ledoit_wolf": "EWMA + Ledoit-Wolf",
}
MAP_LABELS = {
    "endogenous": "Endogenous",
    "common_budget": "Common budget",
    "common_full": "Common risk",
}
EXPECTED_SEEDS = set(range(20260322, 20260352))


def resolve_from_root(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def paired_effects(metrics_csv: Path) -> pd.DataFrame:
    """Validate complete covariance cells and calculate paired seed intervals.

    Use the per-seed metric table, not an action summary with an ambiguous
    'metric' column. Pair hard and mixed actions before calculating intervals.
    """
    frame = pd.read_csv(metrics_csv)
    required = {
        "family", "experiment", "master_seed", "strategy", "n_months",
        "excess_sharpe",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")
    frame = frame.loc[
        (frame["family"] == "covariance_robustness")
        & frame["strategy"].isin(["post_cost_utility__hard", "post_cost_utility__mix"])
    ].copy()
    if frame.empty:
        raise ValueError("No covariance hard/mix rows were found.")
    keys = ["experiment", "master_seed", "strategy"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate experiment/seed/strategy rows in the metric table.")
    values = pd.to_numeric(frame["excess_sharpe"], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite covariance Sharpe values.")
    if not (pd.to_numeric(frame["n_months"], errors="raise") == 83).all():
        raise ValueError("Every plotted strategy must have 83 monthly outcomes.")
    expected_cells = {f"{est}__{risk}" for est in ESTIMATORS for risk in RISK_MAPS}
    actual_cells = set(frame["experiment"])
    if actual_cells != expected_cells:
        raise ValueError(
            f"Covariance cell coverage differs: missing={sorted(expected_cells-actual_cells)}, "
            f"unexpected={sorted(actual_cells-expected_cells)}"
        )
    records = []
    for estimator in ESTIMATORS:
        for risk_map in RISK_MAPS:
            cell = f"{estimator}__{risk_map}"
            group = frame.loc[frame["experiment"] == cell]
            pivot = group.pivot(index="master_seed", columns="strategy", values="excess_sharpe")
            if set(pivot.index.astype(int)) != EXPECTED_SEEDS or pivot.isna().any().any():
                raise ValueError(f"{cell}: both actions must contain the same 30 master seeds.")
            difference = pivot["post_cost_utility__mix"] - pivot["post_cost_utility__hard"]
            n = len(difference)
            mean = float(difference.mean())
            half = float(student_t.ppf(0.975, n - 1) * difference.std(ddof=1) / np.sqrt(n))
            records.append({
                "estimator": estimator,
                "risk_map": risk_map,
                "metric": "excess_sharpe",
                "contrast": "post_cost_utility__mix_minus_hard",
                "n_seeds": n,
                "mean": mean,
                "ci_low": mean - half,
                "ci_high": mean + half,
            })
    return pd.DataFrame(records)


def draw_figure(effects: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4D4D4D", "#8C8C8C", "#C8C8C8"]
    if len(colors) != len(RISK_MAPS):
        raise ValueError(f"Expected {len(RISK_MAPS)} colors, got {len(colors)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "font.weight": "normal", "axes.labelweight": "normal",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(6.9, 4.0))
        y = np.arange(len(ESTIMATORS), dtype=float)
        for offset, risk_map, marker, color in zip(
                (-0.20, 0.0, 0.20), RISK_MAPS, ("o", "s", "^"), colors):
            group = effects.loc[effects["risk_map"] == risk_map].set_index("estimator").loc[list(ESTIMATORS)]
            mean = group["mean"].to_numpy(float)
            low = group["ci_low"].to_numpy(float)
            high = group["ci_high"].to_numpy(float)
            ax.errorbar(
                mean, y + offset, xerr=np.vstack((mean - low, high - mean)),
                fmt=marker, markersize=4.5, capsize=3, elinewidth=0.9,
                capthick=0.9, linestyle="none", color=color,
                markeredgecolor="black", markeredgewidth=0.4,
                label=MAP_LABELS[risk_map],
            )
        ax.axvline(0.0, linewidth=0.7, linestyle="--", alpha=0.65, color="black")
        ax.set_yticks(y, [LABELS[est] for est in ESTIMATORS])
        ax.invert_yaxis()
        ax.set_xlabel("Linear pool minus model sampling (excess Sharpe ratio)", labelpad=7)
        span = float(effects["ci_high"].max() - effects["ci_low"].min())
        ax.set_xlim(float(effects["ci_low"].min()) - 0.08 * span,
                    float(effects["ci_high"].max()) + 0.08 * span)
        ax.set_ylim(len(ESTIMATORS) - 0.5, -0.5)
        ax.grid(axis="x", linewidth=0.4, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", width=0.6, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=3, frameon=False, fontsize=8.4, handletextpad=0.5,
                  columnspacing=1.4)
        fig.subplots_adjust(left=0.27, right=0.985, top=0.84, bottom=0.19)
        png = output_dir / "figure_11_covariance_riskmap_effects.png"
        pdf = output_dir / "figure_11_covariance_riskmap_effects.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--metrics", type=Path, help="Explicit extension_metrics_by_seed.csv override.")
    parser.add_argument("--output-dir", type=Path, help="Explicit figure directory override.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check coverage and calculate paired effects without writing figures.")
    args = parser.parse_args()
    output_root = ROOT / "work/outputs"
    config_path = resolve_from_root(args.config)
    if config_path.exists():
        import yaml
        with config_path.open(encoding="utf-8-sig") as handle:
            cfg = yaml.safe_load(handle) or {}
        configured_output = cfg.get("paths", {}).get("output_root")
        if configured_output:
            output_root = resolve_from_root(configured_output)
    elif args.metrics is None:
        raise FileNotFoundError(f"Config not found: {config_path}. Pass --metrics explicitly for an exported bundle.")
    metrics_csv = resolve_from_root(args.metrics) if args.metrics else (
        output_root / "diagnostics/summary/extensions/extension_metrics_by_seed.csv"
    )
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Metric file not found: {metrics_csv}. Run script 30 or use --metrics.")
    effects = paired_effects(metrics_csv)
    print(effects.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Validated: 12 covariance cells, paired hard/mix actions, 30 seeds, 83 outcomes.")
    if args.validate_only:
        print("Validation only: no plots or result files written.")
        return
    output_dir = resolve_from_root(args.output_dir) if args.output_dir else output_root / "paper_figures"
    png, pdf = draw_figure(effects, output_dir)
    source_csv = output_dir / "figure_11_covariance_riskmap_effects_data.csv"
    effects.to_csv(source_csv, index=False)
    print(f"Wrote: {png}\nWrote: {pdf}\nWrote: {source_csv}")


if __name__ == "__main__":
    main()
