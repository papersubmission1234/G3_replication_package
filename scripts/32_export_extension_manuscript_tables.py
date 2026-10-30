#!/usr/bin/env python3
"""Export numerical sources for the added manuscript tables, without backtests.

Reads extension_metrics_by_seed.csv and computes within-seed contrasts before
Student-t intervals. Single-seed groups receive no inferential interval.
Run from the repository root or scripts/; defaults are repository-relative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "excess_sharpe", "annualized_excess_mean", "certainty_equivalent_annualized",
    "maximum_drawdown", "mean_one_way_turnover_ex_initial", "cagr_net_simple",
)
ESTIMATORS = ("sample", "mp", "ledoit_wolf", "ewma_ledoit_wolf")
MAPS = ("endogenous", "common_budget", "common_full")
EXPECTED_SEEDS = set(range(20260322, 20260352))
POOLS = ("hier_lw", "hier_momentum", "four_models")


def rooted(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def summary(values: pd.Series) -> dict:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Empty or non-finite comparison.")
    n = int(x.size)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return {"n_seeds": n, "mean": mean, "seed_sd": sd,
            "ci_low": mean - half, "ci_high": mean + half}


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "experiment", "master_seed", "strategy"]
    required = set(keys) | set(METRICS) | {"n_months"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame["family"].isin(
        ["small_pool", "covariance_robustness", "frozen_shrinkage"])].copy()
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate family/experiment/seed/strategy records.")
    for col in (*METRICS, "n_months", "master_seed"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[list(METRICS)].to_numpy()).all():
        raise ValueError("Metric values must be finite.")
    if not frame["n_months"].eq(83).all():
        raise ValueError("Every strategy must contain 83 outcomes.")
    expected = {
        "small_pool": set(POOLS),
        "covariance_robustness": {f"{e}__{m}" for e in ESTIMATORS for m in MAPS},
        "frozen_shrinkage": {f"shrink_{s}_{m}" for s in ("0", "0p25", "0p5", "0p75", "1")
                             for m in ("endogenous", "common_full")},
    }
    frozen_sets = []
    for family, cells in expected.items():
        part = frame.loc[frame["family"] == family]
        if set(part["experiment"]) != cells:
            raise ValueError(f"Unexpected or missing cells in {family}.")
        for cell, group in part.groupby("experiment"):
            seed_sets = [set(g["master_seed"].astype(int)) for _, g in group.groupby("strategy")]
            if not seed_sets or any(s != seed_sets[0] for s in seed_sets):
                raise ValueError(f"Unmatched strategy seeds in {family}/{cell}.")
            if family != "frozen_shrinkage" and seed_sets[0] != EXPECTED_SEEDS:
                raise ValueError(f"{family}/{cell} requires the declared 30 seeds.")
            if family == "frozen_shrinkage":
                frozen_sets.append(seed_sets[0])
    if any(s != frozen_sets[0] for s in frozen_sets):
        raise ValueError("Shrinkage fractions/maps must have the same seed set.")
    return frame


def paired(group: pd.DataFrame, metric: str, treatment: str, control: str) -> pd.Series:
    p = group.pivot(index="master_seed", columns="strategy", values=metric)
    if treatment not in p or control not in p or p[[treatment, control]].isna().any().any():
        raise ValueError(f"Missing paired strategy: {treatment} or {control}")
    return p[treatment] - p[control]


def export(frame: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for (family, experiment, strategy), group in frame.groupby(["family", "experiment", "strategy"]):
        for metric in METRICS:
            records.append({"family": family, "experiment": experiment, "strategy": strategy,
                            "metric": metric, **summary(group[metric])})
    pd.DataFrame(records).to_csv(out / "extension_metric_summary.csv", index=False)
    rows = []
    pool = frame.loc[frame["family"] == "small_pool"]
    for cell in POOLS:
        group = pool.loc[pool["experiment"] == cell]
        for contrast, control in (
            ("mix_minus_hard", "post_cost_utility__hard"),
            ("mix_minus_fixed", "fixed_equal__mix"),
            ("mix_minus_HierMacro", "standalone__M_HierMacro"),
        ):
            for metric in METRICS:
                rows.append({"experiment": cell, "contrast": contrast, "metric": metric,
                             **summary(paired(group, metric, "post_cost_utility__mix", control))})
    pd.DataFrame(rows).to_csv(out / "small_pool_paired_summary.csv", index=False)
    effects, attenuation, channels = [], [], []
    cov = frame.loc[frame["family"] == "covariance_robustness"]
    for estimator in ESTIMATORS:
        for metric in METRICS:
            diffs = {}
            for risk_map in MAPS:
                group = cov.loc[cov["experiment"] == f"{estimator}__{risk_map}"]
                diffs[risk_map] = paired(group, metric, "post_cost_utility__mix", "post_cost_utility__hard")
                effects.append({"estimator": estimator, "risk_map": risk_map, "metric": metric,
                                **summary(diffs[risk_map])})
            attenuation.append({"estimator": estimator, "metric": metric,
                                **summary(diffs["endogenous"] - diffs["common_full"])})
            parts = {
                "objective_pooling": diffs["common_full"],
                "risk_distribution": diffs["common_budget"] - diffs["common_full"],
                "threshold": diffs["endogenous"] - diffs["common_budget"],
            }
            for name, value in parts.items():
                channels.append({"estimator": estimator, "metric": metric, "channel": name,
                                 **summary(value)})
    pd.DataFrame(effects).to_csv(out / "covariance_action_effects.csv", index=False)
    pd.DataFrame(attenuation).to_csv(out / "covariance_attenuation.csv", index=False)
    pd.DataFrame(channels).to_csv(out / "covariance_channels.csv", index=False)
    coverage = frame.groupby(["family", "experiment"]).agg(
        n_seeds=("master_seed", "nunique"), n_strategies=("strategy", "nunique"),
        min_outcomes=("n_months", "min"), max_outcomes=("n_months", "max"),
    ).reset_index()
    coverage.to_csv(out / "extension_cell_coverage.csv", index=False)
    frame.loc[frame["family"] == "frozen_shrinkage"].to_csv(
        out / "frozen_shrinkage_values_by_seed.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_root = ROOT / "work/outputs"
    config = rooted(args.config)
    if config.exists():
        import yaml
        cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig")) or {}
        if cfg.get("paths", {}).get("output_root"):
            output_root = rooted(cfg["paths"]["output_root"])
    elif args.metrics is None:
        raise FileNotFoundError(f"Config not found: {config}. Use --metrics for an exported bundle.")
    source = rooted(args.metrics) if args.metrics else (
        output_root / "diagnostics/summary/extensions/extension_metrics_by_seed.csv")
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = validate(pd.read_csv(source))
    out = rooted(args.output_dir) if args.output_dir else output_root / "paper_tables/extension_support"
    export(frame, out)
    info = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "strategy_histories": len(frame),
            "note": "Derived from recorded per-seed metrics. Intervals are conditional algorithmic-seed intervals; n=1 intervals are absent."}
    (out / "export_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Exported numerical table sources to {out}")
    print("No models, portfolios, or bootstrap procedures were rerun.")


if __name__ == "__main__":
    main()
