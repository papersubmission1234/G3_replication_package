#!/usr/bin/env python3
"""Audit CVaR binding flags and inputs needed for portfolio replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import zipfile

from _common import ROOT
from g3_confirmatory.configuration import load_yaml, study_paths, configured_seeds
from g3_confirmatory.displacement_diagnostics import (
    MAPS, SCORES, atomic_csv, atomic_json, numeric_binding, matched_tail_panel, seed_dir, seed_summary,
)
import pandas as pd


def audit_chunks(chunks, destination: Path):
    reduced = []
    panel = []
    for source, chunk in chunks:
        if "experiment_id" not in chunk:
            chunk["experiment_id"] = source
        if "cvar_budget" not in chunk:
            continue
        d = numeric_binding(chunk)
        keys = ["experiment_id", "master_seed", "strategy"]
        metrics = ["true_cvar_slack", "true_cvar_binding", "true_cvar_violation", "reported_lp_binding", "positive_cvar_shadow"]
        metrics = [m for m in metrics if m in d]
        grouped = d.groupby(keys, dropna=False)[metrics].agg(["sum", "count"])
        grouped.columns = [f"{x}__{y}" for x, y in grouped.columns]
        reduced.append(grouped.reset_index())
        core = d.loc[d["experiment_id"].eq("core")]
        cols = keys + [c for c in ("formation_date", "outcome_date", "cvar_budget", "cvar_alpha", "risk_scenario_var", "scenario_var", "risk_scenario_cvar", "scenario_cvar", *metrics) if c in d]
        if not core.empty:
            panel.append(core[cols])
    if not reduced:
        raise ValueError("No usable optimizer diagnostic rows found.")
    combined = pd.concat(reduced, ignore_index=True)
    keys = ["experiment_id", "master_seed", "strategy"]
    sums = combined.groupby(keys, dropna=False).sum(numeric_only=True).reset_index()
    metric_names = [c[:-5] for c in sums if c.endswith("__sum")]
    per_seed = sums[keys].copy()
    for m in metric_names:
        per_seed[m] = sums[m + "__sum"] / sums[m + "__count"].replace(0, float("nan"))
    per_seed["n_valid"] = sums["true_cvar_binding__count"]
    atomic_csv(destination / "binding_by_seed_strategy.csv", per_seed)
    _, summary = seed_summary(per_seed, ["experiment_id", "strategy"], metric_names)
    atomic_csv(destination / "binding_summary.csv", summary)
    if panel:
        atomic_csv(destination / "core_true_binding_panel.csv", pd.concat(panel, ignore_index=True))
    core = per_seed[per_seed.experiment_id.eq("core")].groupby("strategy")[metric_names].mean()
    atomic_csv(destination / "core_binding_correction.csv", core.reset_index())
    print(core.to_string())


def corrected_core_tables(destination: Path, read_table):
    correction = pd.read_csv(destination / "core_binding_correction.csv").set_index("strategy")
    if correction.empty:
        return
    for source, target, old_col in (
        ("main_12_cvar_calibration.csv", "main_12_cvar_calibration_core_corrected.csv", "cvar_budget_binding_rate"),
        ("appendix/app_A07_optimizer_summary.csv", "app_optimizer_core_corrected.csv", "cvar_binding_fraction"),
    ):
        try:
            original = read_table(source)
        except (FileNotFoundError, KeyError):
            continue
        original = original[original.experiment_id.eq("core")].copy()
        if original.empty:
            continue
        original["old_lp_auxiliary_binding_rate"] = original[old_col]
        original[old_col] = original.strategy.map(correction.true_cvar_binding)
        original["true_mean_cvar_slack"] = original.strategy.map(correction.true_cvar_slack)
        original["true_cvar_binding_tolerance"] = 1e-8
        if original[old_col].isna().any():
            raise ValueError("A core strategy lacks a corrected binding rate.")
        atomic_csv(destination / target, original)


def matched_calibration(destination: Path, return_chunks):
    core_returns = []
    for frame in return_chunks:
        if "experiment_id" in frame:
            frame = frame.loc[frame.experiment_id.eq("core")]
        if not frame.empty:
            core_returns.append(frame)
    if not core_returns:
        raise ValueError("No core realized-return records for matched calibration.")
    diagnostics = pd.read_csv(destination / "core_true_binding_panel.csv")
    # loss-map probes have no investable return series
    diagnostic_only = diagnostics.strategy.astype(str).str.startswith("diagnostic__")
    if diagnostic_only.any():
        atomic_csv(destination / "nonimplemented_calibration_rows_omitted.csv", diagnostics.loc[diagnostic_only])
    diagnostics = diagnostics.loc[~diagnostic_only].copy()
    panel = matched_tail_panel(diagnostics, pd.concat(core_returns, ignore_index=True))
    atomic_csv(destination / "core_matched_tail_panel.csv", panel)
    rows = []
    for strategy, g in panel.groupby("strategy"):
        valid = g.loc[g.calibration_valid]
        tail = valid.loc[valid.gross_excess_var_exceedance.eq(1.)]
        old_tail = valid.loc[valid.old_simple_against_excess_var_exceedance.eq(1.)]
        rows.append({"strategy": strategy, "n_seed_months": len(g), "n_valid": len(valid),
            "n_seeds": g.master_seed.nunique(),
            "gross_excess_var_exceedance_rate": valid.gross_excess_var_exceedance.mean(),
            "matched_net_excess_var_exceedance_rate": valid.matched_net_excess_var_exceedance.mean(),
            "old_simple_against_excess_var_exceedance_rate": valid.old_simple_against_excess_var_exceedance.mean(),
            "gross_excess_tail_severity_gap": tail.gross_excess_loss_minus_cvar.mean(),
            "old_simple_against_excess_tail_severity_gap": (-old_tail.gross_simple_return - old_tail.risk_scenario_cvar).mean(),
            "n_gross_excess_var_exceedances": len(tail),
            "actual_cvar_binding_rate": valid.true_cvar_binding.mean(),
            "mean_predicted_excess_cvar": valid.risk_scenario_cvar.mean(),
            "mean_cvar_budget": valid.cvar_budget.mean(),
            "interpretation": "Pooled descriptive frequency on one historical path; not independent seed-month evidence"})
    atomic_csv(destination / "core_tail_calibration_excess_corrected.csv", pd.DataFrame(rows))


def joint_risk_effect(frame: pd.DataFrame, destination: Path):
    from scipy import stats
    import numpy as np
    rows = []
    for (score, metric), g in frame.groupby(["score", "metric"]):
        wide = g.pivot(index="master_seed", columns="action_map", values="mix_minus_hard")
        value = (wide["endogenous"] - wide["hier_full_risk"]).dropna()
        n = len(value)
        mean = float(value.mean())
        half = float(stats.t.ppf(.975, n - 1) * value.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append({"score": score, "metric": metric, "n_seeds": n, "mean": mean,
                     "ci_low": mean - half, "ci_high": mean + half,
                     "two_sided_p": float(stats.ttest_1samp(value, 0).pvalue) if n > 1 else np.nan})
    atomic_csv(destination / "joint_risk_map_effect_from_saved_results.csv", pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--bundle", type=Path, help="Optional paper_bundle.zip; audit needs no raw input data.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.bundle:
        destination = (args.output_dir or ROOT / "work/outputs/diagnostics/audit").resolve()
        with zipfile.ZipFile(args.bundle) as z:
            member = next(n for n in z.namelist() if n.endswith("paper_data/all_optimizer_diagnostics.csv.gz"))
            import gzip
            with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as stream:
                audit_chunks((("bundle", c) for c in pd.read_csv(stream, chunksize=40000, low_memory=False)), destination)
            prefix = member.split("paper_data/")[0]
            returns_member = prefix + "paper_data/all_returns.csv.gz"
            with z.open(returns_member) as raw, gzip.GzipFile(fileobj=raw) as stream:
                matched_calibration(destination, pd.read_csv(stream, chunksize=40000, low_memory=False))
            corrected_core_tables(destination, lambda name: pd.read_csv(z.open(prefix + "outputs/paper_tables/" + name)))
            risk_member = prefix + "outputs/decompositions/full_risk_map/summary/action_mix_minus_hard_by_seed.csv"
            if risk_member in z.namelist():
                joint_risk_effect(pd.read_csv(z.open(risk_member)), destination)
        atomic_json(destination / "audit_status.json", {"status": "completed", "source": str(args.bundle.resolve()),
                    "binding_tolerance": 1e-8, "original_outputs_modified": False})
        return
    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    root = paths["output_root"]
    destination = (args.output_dir or root / "diagnostics/audit").resolve()
    sources = sorted((root / "core").glob("seed_*/optimizer_diagnostics.csv"))
    sources += sorted((root / "decompositions/full_risk_map").glob("*/seed_*/optimizer_diagnostics.csv"))
    if not sources:
        raise FileNotFoundError("No original optimizer_diagnostics.csv files. Use --bundle for an exported bundle audit.")
    def chunks():
        for f in sources:
            experiment = "core" if f.parent.parent.name == "core" else f.parent.parent.name
            for chunk in pd.read_csv(f, chunksize=40000, low_memory=False):
                yield experiment, chunk
    audit_chunks(chunks(), destination)
    matched_calibration(destination, (pd.read_csv(f) for f in sorted((root / "core").glob("seed_*/returns.csv"))))
    corrected_core_tables(destination, lambda name: pd.read_csv(root / "paper_tables" / name))
    risk_source = root / "decompositions/full_risk_map/summary/action_mix_minus_hard_by_seed.csv"
    if risk_source.exists():
        joint_risk_effect(pd.read_csv(risk_source), destination)
    inventory = []
    for seed in configured_seeds(cfg):
        for score in SCORES:
            for label in MAPS:
                folder = seed_dir(root, score, label, seed)
                missing = [name for name in ("returns.csv", "posterior.csv", "scenario_hashes.csv", "optimizer_diagnostics.csv", "run_manifest.json") if not (folder / name).exists()]
                inventory.append({"master_seed": seed, "score": score, "risk_map": label,
                                  "available": not missing, "missing": ";".join(missing), "path": str(folder)})
    atomic_csv(destination / "replay_input_inventory.csv", pd.DataFrame(inventory))
    atomic_json(destination / "audit_status.json", {"status": "completed", "binding_tolerance": 1e-8,
                "n_source_files": len(sources), "all_replay_inputs_present": all(r["available"] for r in inventory),
                "original_outputs_modified": False})
    print(f"Audit written to {destination}. No model fit or portfolio replay was run.")


if __name__ == "__main__":
    main()
