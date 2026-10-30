#!/usr/bin/env python3
"""Summarize displacement results and collect a reproduction archive."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import zipfile

from _common import ROOT
from g3_confirmatory.configuration import load_yaml, study_paths
from g3_confirmatory.displacement_diagnostics import atomic_csv, atomic_json, digest, seed_summary, numeric_binding
from g3_confirmatory.metrics import compute_performance_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def stationary_indices(n, rng, mean_block=6):
    result = np.empty(n, dtype=int)
    result[0] = rng.integers(n)
    for j in range(1, n):
        result[j] = rng.integers(n) if rng.random() < 1.0 / mean_block else (result[j - 1] + 1) % n
    return result


def slope(x, y):
    a, b = x - np.mean(x), y - np.mean(y)
    denom = float(a @ a)
    return float(a @ b / denom) if denom > 1e-20 else np.nan


def associations(d, n_boot):
    keys = ["score", "action", "left_map", "right_map", "scope"]
    covariates = ["allocation_displacement", "boundary_displacement_at_left", "opportunity_displacement"]
    ys = ["net_excess_return_difference", "quadratic_utility_difference"]
    # average seeds first; historical inference uses 83 dates
    monthly = d.groupby(keys + ["outcome_date"], dropna=False)[covariates + ys].mean().reset_index()
    rows = []
    for key, g in monthly.groupby(keys, dropna=False):
        g = g.sort_values("outcome_date")
        for xcol in covariates:
            for ycol in ys:
                pair = g[[xcol, ycol]].dropna()
                x, y = pair[xcol].to_numpy(float), pair[ycol].to_numpy(float)
                if len(x) < 12 or np.std(x) < 1e-10:
                    continue
                ols = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True})
                rng = np.random.default_rng(20260913)
                boots = np.array([slope(x[ix], y[ix]) for ix in (stationary_indices(len(x), rng) for _ in range(n_boot))])
                boots = boots[np.isfinite(boots)]
                lo, hi = np.quantile(boots, [.025, .975]) if len(boots) else (np.nan, np.nan)
                rows.append({**dict(zip(keys, key)), "predictor": xcol, "outcome": ycol, "n_historical_months": len(x),
                             "slope": float(ols.params[1]), "hac_standard_error": float(ols.bse[1]),
                             "hac_p_value": float(ols.pvalues[1]), "stationary_bootstrap_slope_low": lo,
                             "stationary_bootstrap_slope_high": hi, "bootstrap_replications": n_boot,
                             "mean_block_length": 6, "interpretation": "descriptive association, not causal; conditional on saved paths"})
    return monthly, pd.DataFrame(rows)


def summarize_extensions(root: Path, output: Path):
    rows, binding = [], []
    for f in sorted((root / "extensions").glob("*/*/seed_*/returns.csv")):
        family, experiment = f.parents[2].name, f.parents[1].name
        data = pd.read_csv(f)
        manifest = json.loads((f.parent / "run_manifest.json").read_text())
        gamma = manifest["experiment_config"]["utility_gamma"]
        for (seed, strategy), group in data.groupby(["master_seed", "strategy"]):
            rows.append({"family": family, "experiment": experiment, "master_seed": seed, "strategy": strategy,
                         **compute_performance_metrics(group, utility_gamma=gamma)})
        diagnostics = f.parent / "optimizer_diagnostics.csv"
        if diagnostics.exists():
            fixed = numeric_binding(pd.read_csv(diagnostics))
            fixed["family"], fixed["experiment"] = family, experiment
            by = fixed.groupby(["family", "experiment", "master_seed", "strategy"])[["true_cvar_binding", "true_cvar_slack"]].mean().reset_index()
            binding.append(by)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    atomic_csv(output / "extension_metrics_by_seed.csv", frame)
    _, summary = seed_summary(frame, ["family", "experiment", "strategy"],
                             ["excess_sharpe", "annualized_excess_mean", "certainty_equivalent_annualized", "maximum_drawdown", "mean_one_way_turnover_ex_initial"])
    atomic_csv(output / "extension_metric_summary.csv", summary)
    effects = []
    for (family, experiment, seed), g in frame.groupby(["family", "experiment", "master_seed"]):
        g = g.set_index("strategy")
        for treatment, control in (("post_cost_utility__mix", "post_cost_utility__hard"),
                                   ("post_cost_utility__mix", "fixed_equal__mix"),
                                   ("post_cost_utility__mix", "standalone__M_HierMacro")):
            for metric in ("excess_sharpe", "annualized_excess_mean", "certainty_equivalent_annualized"):
                effects.append({"family": family, "experiment": experiment, "master_seed": seed,
                                "treatment": treatment, "control": control, "metric_name": metric,
                                "value": float(g.loc[treatment, metric] - g.loc[control, metric])})
    e = pd.DataFrame(effects)
    atomic_csv(output / "extension_paired_effects_by_seed.csv", e)
    _, summary = seed_summary(e, ["family", "experiment", "treatment", "control", "metric_name"], ["value"])
    atomic_csv(output / "extension_paired_effects_summary.csv", summary)
    if binding:
        atomic_csv(output / "extension_true_binding_by_seed.csv", pd.concat(binding, ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--tag", default="primary")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true", help="Label smoke/partial outputs, never submission-ready.")
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = load_yaml(args.config)
    root = study_paths(cfg)["output_root"] / "diagnostics"
    src = root / "displacement" / args.tag
    output = root / "summary" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    manifests = sorted(src.glob("seed_*/completed.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed displacement seeds in {src}")
    expected = cfg.get("seeds", list(range(int(cfg.get("seed_start", 20260322)), int(cfg.get("seed_start", 20260322)) + int(cfg.get("n_seeds", 30)))))
    records = [json.loads(p.read_text()) for p in manifests]
    full = set(int(r["master_seed"]) for r in records) == set(map(int, expected)) and all(r["full_calendar"] for r in records)
    if not full and not args.allow_partial:
        raise ValueError("Not all configured seeds/full calendars are completed. Finish replay or use --allow-partial for smoke outputs.")
    signatures = {(r["settings"]["data_hash"], r["settings"]["environment_hash"], r["settings"]["code_hash"], tuple(r["scores"])) for r in records}
    if len(signatures) != 1:
        raise ValueError("Cannot combine seeds from different data, code, environments or score sets.")
    for path, record in zip(manifests, records):
        for name, h in record["artifacts"].items():
            if digest(path.parent / name) != h:
                raise ValueError(f"Damaged output: {path.parent / name}")
    def collect(name):
        return pd.concat([pd.read_csv(p.parent / f"{name}.csv") for p in manifests], ignore_index=True)
    risk, disp, returns = collect("risk_evaluations"), collect("displacement"), collect("replay_returns")
    for name, frame, keys, metrics in [
        ("cross_risk_feasibility", risk, ["score", "action", "source_map", "evaluation_map"], ["infeasible", "positive_cvar_breach", "signed_cvar_gap", "boundary_displacement"]),
        ("displacement", disp, ["score", "action", "left_map", "right_map", "scope"], ["allocation_displacement", "boundary_displacement_at_left", "left_target_infeasible_under_right", "opportunity_displacement", "net_excess_return_difference", "quadratic_utility_difference"]),
        ("replayed_binding", returns, ["score", "action", "risk_map"], ["true_cvar_binding", "true_cvar_slack", "lp_auxiliary_binding", "cvar_shadow"]),
    ]:
        seeds, summary = seed_summary(frame, keys, metrics)
        atomic_csv(output / f"{name}_by_seed.csv", seeds)
        atomic_csv(output / f"{name}_summary.csv", summary)
    metric_rows = []
    for key, group in returns.groupby(["master_seed", "score", "action", "risk_map"]):
        metric_rows.append({**dict(zip(["master_seed", "score", "action", "risk_map"], key)),
                           **compute_performance_metrics(group)})
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(output / "replayed_performance_by_seed.csv", metrics)
    effects = []
    for (seed, score), group in metrics.groupby(["master_seed", "score"]):
        for metric in ("excess_sharpe", "annualized_excess_mean", "certainty_equivalent_annualized"):
            cells = group.pivot(index="risk_map", columns="action", values=metric)
            delta = cells["mix"] - cells["hard"]
            for comp, val in {"objective_pooling": delta["hier_full_risk"],
                              "risk_distribution": delta["hier_budget_only"] - delta["hier_full_risk"],
                              "budget_threshold": delta["endogenous"] - delta["hier_budget_only"],
                              "total": delta["endogenous"]}.items():
                effects.append({"master_seed": seed, "score": score, "metric_name": metric, "component": comp, "value": val})
    effects = pd.DataFrame(effects)
    atomic_csv(output / "reconstructed_channels_by_seed.csv", effects)
    _, summaries = seed_summary(effects, ["score", "metric_name", "component"], ["value"])
    atomic_csv(output / "reconstructed_channels_summary.csv", summaries)
    monthly, assoc = associations(disp, args.bootstrap)
    atomic_csv(output / "monthly_displacement_outcomes.csv", monthly)
    atomic_csv(output / "displacement_outcome_associations.csv", assoc)
    target = monthly[(monthly.score == "post_cost_utility") & (monthly.action == "mix") &
                     (monthly.left_map == "endogenous") & (monthly.right_map == "hier_full_risk") &
                     (monthly.scope == "one_step_convex_parent")]
    if not target.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(target.allocation_displacement, 100 * target.net_excess_return_difference, s=18)
        ax.axhline(0, linewidth=.7)
        ax.set_xlabel("Allocation displacement (half-L1 distance)")
        ax.set_ylabel("Next-month excess-return difference (percentage points)")
        ax.set_title("Common full map minus endogenous map\nSame objective and pretrade holdings; convex-parent probe", fontsize=10)
        fig.tight_layout()
        fig.savefig(output / "displacement_outcome_association.png", dpi=250)
        plt.close(fig)
    summarize_extensions(root, output)
    atomic_json(output / "summary_status.json", {"full_configured_seed_coverage": full, "tag": args.tag,
                "n_seeds": len(records), "n_months": sorted({r["n_months"] for r in records}),
                "scores": records[0]["scores"], "algorithmic_ci_not_market_ci": True,
                "association_is_descriptive": True, "original_outputs_modified": False})
    bundle = root / f"displacement_bundle_{args.tag}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (src, output, root / "audit", root / "extensions"):
            if base.exists():
                for path in sorted(base.rglob("*")):
                    if path.is_file() and path.suffix in {".csv", ".json", ".png", ".md"} and "cache" not in path.parts:
                        z.write(path, path.relative_to(root).as_posix())
        for path in (ROOT / "scripts/22_audit_diagnostics.py", ROOT / "scripts/25_summarize_displacement.py"):
            z.write(path, "reproduction/scripts/" + path.name)
        p = ROOT / "scripts/24_run_diagnostic_extensions.py"
        z.write(p, "reproduction/scripts/" + p.name)
        p = ROOT / "scripts/23_run_feasible_set_displacement.py"
        z.write(p, "reproduction/scripts/" + p.name)
        p = ROOT / "src/g3_confirmatory/displacement_diagnostics.py"
        z.write(p, "reproduction/src/g3_confirmatory/" + p.name)
    print(f"Summary: {output}\nReturn this file: {bundle}")


if __name__ == "__main__":
    main()
