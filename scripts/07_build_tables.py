#!/usr/bin/env python3
"""Build manuscript and appendix tables from saved results.

Missing inputs produce an empty CSV and a note in the table catalog.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import ROOT
from g3_confirmatory.configuration import (
    experiment_config_from_mapping, load_yaml, study_paths,
)
from g3_confirmatory.reporting import (
    TableArtifact,
    all_experiment_metrics,
    failure_index,
    holdings_diagnostics,
    optimizer_diagnostics_summary,
    posterior_diagnostics,
    read_seed_tree,
    run_manifest_index,
    safe_read_csv,
    safe_read_json,
    table_sheet_name,
    write_table_catalog,
    write_workbook,
)

STRATEGY_ORDER = [
    "standalone__M_HierMacro",
    "post_cost_utility__mix",
    "fixed_equal__mix",
    "post_cost_utility__hard",
    "predictive_log__mix",
    "predictive_log__hard",
    "standalone__M_FF5MOM",
    "standalone__M_LW_MinVar",
    "standalone__M_Momentum",
    "benchmark__1N",
]

MAIN_METRICS = [
    "annualized_excess_mean",
    "annualized_excess_volatility",
    "excess_sharpe",
    "cagr_net_simple",
    "maximum_drawdown",
    "certainty_equivalent_annualized",
    "historical_es_95_monthly",
    "mean_one_way_turnover_ex_initial",
    "terminal_wealth_from_1000",
]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _external_panel_reference(path: Path, relative_name: str, note: str) -> pd.DataFrame:
    """Reference a large diagnostic panel without copying it into the workbook."""

    return pd.DataFrame([
        {
            "canonical_file": relative_name,
            "available": bool(path.exists()),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "note": note,
        }
    ])


def _format_mean_ci(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if {"mean", "ci_low", "ci_high"}.issubset(out.columns):
        out["estimate_95ci"] = out.apply(
            lambda row: (
                f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                if pd.notna(row["mean"])
                else ""
            ),
            axis=1,
        )
    return out


def _core_performance(summary: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = summary[
        (summary["experiment_id"] == "core")
        & (summary["period"] == period)
        & (summary["metric"].isin(MAIN_METRICS))
    ].copy()
    if frame.empty:
        return frame
    frame["strategy_order"] = frame["strategy"].map(
        {name: i for i, name in enumerate(STRATEGY_ORDER)}
    ).fillna(999)
    frame = frame.sort_values(["strategy_order", "strategy", "metric"])
    frame["estimate_95ci"] = frame.apply(
        lambda row: f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
        axis=1,
    )
    wide = frame.pivot(index="strategy", columns="metric", values="estimate_95ci")
    order = [strategy for strategy in STRATEGY_ORDER if strategy in wide.index]
    order += [strategy for strategy in wide.index if strategy not in order]
    return wide.reindex(order).reset_index()


def _sample_and_universe(work_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = work_root / "data"
    manifest = safe_read_json(data_root / "study_build_manifest.json")
    audit = safe_read_csv(
        data_root / "eligibility_primary" / "eligibility_audit_by_date.csv",
        parse_dates=["formation_date", "outcome_date"],
    )
    universe = safe_read_csv(data_root / "universe_derived.csv")
    records: list[dict[str, Any]] = []

    def add(item: str, value: Any, note: str = "") -> None:
        records.append({"item": item, "value": value, "note": note})

    add("candidate_common_tickers", manifest.get("candidate_common_tickers"))
    add("derived_universe_size", manifest.get("derived_universe_size"))
    add("formation_count", manifest.get("formation_count"))
    add("formation_start", manifest.get("formation_start"))
    add("formation_end", manifest.get("formation_end"))
    add("outcome_count", manifest.get("outcome_count"))
    add("outcome_start", manifest.get("outcome_start"))
    add("outcome_end", manifest.get("outcome_end"))
    add("timing_rule", manifest.get("timing_rule"))
    add("first_training_window", manifest.get("first_training_window"))
    add("liquidity_rule", manifest.get("liquidity_rule"))
    add("delisting_rule", manifest.get("delisting_rule"))
    add("missing_outcome_return_rule", manifest.get("missing_outcome_return_rule"))
    if not audit.empty and "n_eligible" in audit:
        eligible = pd.to_numeric(audit["n_eligible"], errors="coerce")
        add("mean_monthly_eligible", float(eligible.mean()))
        add("minimum_monthly_eligible", float(eligible.min()))
        add("maximum_monthly_eligible", float(eligible.max()))
    if not universe.empty and "asset_class" in universe:
        counts = (
            universe["asset_class"].fillna("unknown").value_counts().rename_axis("asset_class")
            .reset_index(name="n_tickers")
        )
    else:
        counts = pd.DataFrame(columns=["asset_class", "n_tickers"])
    return pd.DataFrame(records), counts


def _mean_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[
        summary["experiment_id"].str.startswith("mean_ablation/", na=False)
        & (summary["period"] == "full")
        & (summary["strategy"] == "post_cost_utility__mix")
        & (summary["metric"].isin(MAIN_METRICS))
    ].copy()
    if frame.empty:
        return frame
    frame["mean_treatment"] = frame["experiment_id"].str.split("/").str[-1]
    return _format_mean_ci(frame)[
        [
            "mean_treatment", "metric", "n_seeds", "mean", "sd", "ci_low",
            "ci_high", "estimate_95ci",
        ]
    ]


def _robustness(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[
        summary["experiment_id"].str.startswith("robustness/", na=False)
        & (summary["period"] == "full")
        & (summary["metric"].isin(MAIN_METRICS))
        & (summary["strategy"].isin([
            "standalone__M_HierMacro",
            "post_cost_utility__mix",
            "fixed_equal__mix",
            "post_cost_utility__hard",
            "benchmark__1N",
        ]))
    ].copy()
    if frame.empty:
        return frame
    frame["robustness_configuration"] = frame["experiment_id"].str.split("/").str[-1]
    return _format_mean_ci(frame)[
        [
            "robustness_configuration", "strategy", "metric", "n_seeds",
            "mean", "sd", "ci_low", "ci_high", "estimate_95ci",
        ]
    ]


def _extension_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[
        summary["experiment_id"].str.startswith("extensions/", na=False)
        & (summary["period"] == "full")
        & (summary["metric"].isin(MAIN_METRICS))
    ].copy()
    if frame.empty:
        return frame
    return _format_mean_ci(frame)[
        [
            "experiment_id", "strategy", "metric", "n_seeds", "mean", "sd",
            "ci_low", "ci_high", "estimate_95ci",
        ]
    ]


def _loss_and_selection_diagnostics(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    losses = read_seed_tree(
        output_root,
        "arm_losses.csv",
        parse_dates=("formation_date", "outcome_date"),
    )
    if losses.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_records: list[dict[str, Any]] = []
    for keys, group in losses.groupby(["experiment_id", "score", "arm"], sort=True):
        experiment_id, score, arm = keys
        values = pd.to_numeric(group["loss"], errors="coerce")
        summary_records.append(
            {
                "experiment_id": experiment_id,
                "score": score,
                "arm": arm,
                "n_observations": int(values.notna().sum()),
                "mean_loss": float(values.mean()),
                "sd_loss": float(values.std(ddof=1)),
                "median_loss": float(values.median()),
            }
        )
    return losses, pd.DataFrame(summary_records)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument(
        "--write-xlsx",
        action="store_true",
        help=(
            "Also create paper_tables.xlsx. The workbook is optional and contains "
            "only the main-text tables plus the catalog; CSV files remain authoritative."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    experiment = experiment_config_from_mapping(cfg.get("experiment"))
    output_root = paths["output_root"]
    work_root = output_root.parent
    tables_root = output_root / "paper_tables"
    appendix_root = tables_root / "appendix"
    tables_root.mkdir(parents=True, exist_ok=True)
    appendix_root.mkdir(parents=True, exist_ok=True)

    metrics_by_seed, metric_summary = all_experiment_metrics(
        output_root,
        utility_gamma=experiment.utility_gamma,
    )
    holdings_by_date, holdings_summary = holdings_diagnostics(output_root)
    optimizer_all, optimizer_summary = optimizer_diagnostics_summary(output_root)
    final_posterior, posterior_summary = posterior_diagnostics(output_root)
    manifest_index, return_fills = run_manifest_index(output_root)
    failures = failure_index(output_root)
    losses_all, losses_summary = _loss_and_selection_diagnostics(output_root)

    inference_root = output_root / "inference"
    risk_root = output_root / "risk_diagnostics"
    riskmap_root = output_root / "decompositions" / "full_risk_map" / "summary"
    shapley_root = output_root / "decompositions" / "decision_shapley" / "summary"

    sample_table, class_counts = _sample_and_universe(work_root)
    core_full = _core_performance(metric_summary, "full")
    core_holdout = _core_performance(metric_summary, "holdout_2024_2025")
    primary_effects = _format_mean_ci(
        safe_read_csv(inference_root / "primary_algorithmic_effect_summary.csv")
    )
    riskmap_actions = _format_mean_ci(
        safe_read_csv(riskmap_root / "action_mix_minus_hard_summary.csv")
    )
    riskmap_channels = _format_mean_ci(
        safe_read_csv(riskmap_root / "action_channel_decomposition_summary.csv")
    )
    shapley_performance = _format_mean_ci(
        safe_read_csv(shapley_root / "metric_summary_across_seeds.csv")
    )
    shapley_contributions = safe_read_csv(
        shapley_root / "decision_shapley_contribution_summary.csv"
    )
    if shapley_contributions.empty:
        shapley_contributions = safe_read_csv(
            shapley_root / "shapley_contribution_summary.csv"
        )
    shapley_effective_weights = safe_read_csv(
        shapley_root / "effective_action_weight_summary.csv"
    )
    mean_ablation = _mean_ablation(metric_summary)
    robustness = _robustness(metric_summary)
    extensions = _extension_summary(metric_summary)
    cvar_calibration = safe_read_csv(risk_root / "monthly_cvar_calibration_summary.csv")
    daily_risk = safe_read_csv(risk_root / "daily_risk_metric_summary.csv")
    newey_west = safe_read_csv(inference_root / "primary_newey_west_by_seed.csv")
    block_bootstrap = safe_read_csv(
        inference_root / "primary_historical_block_bootstrap.csv"
    )
    psr = safe_read_csv(inference_root / "core_probabilistic_sharpe.csv")

    main_tables: dict[str, pd.DataFrame] = {
        "main_01_sample_universe": sample_table,
        "main_02_asset_class_counts": class_counts,
        "main_03_core_full_performance": core_full,
        "main_04_core_holdout_performance": core_holdout,
        "main_05_score_action_effects": primary_effects,
        "main_06_riskmap_action_effects": riskmap_actions,
        "main_07_riskmap_channels": riskmap_channels,
        "main_08_mean_ablation": mean_ablation,
        "main_09_shapley_performance": shapley_performance,
        "main_10_shapley_contributions": shapley_contributions,
        "main_11_shapley_effective_weights": shapley_effective_weights,
        "main_12_cvar_calibration": cvar_calibration,
        "main_13_selected_robustness": robustness,
    }

    appendix_tables: dict[str, pd.DataFrame] = {
        "app_A01_all_metrics_by_seed": metrics_by_seed,
        "app_A02_all_metrics_summary": metric_summary,
        "app_A03_extensions_summary": extensions,
        "app_A04_holdings_by_date": holdings_by_date,
        "app_A05_holdings_summary": holdings_summary,
        "app_A06_optimizer_all": optimizer_all,
        "app_A07_optimizer_summary": optimizer_summary,
        "app_A08_final_posterior_by_seed": final_posterior,
        "app_A09_final_posterior_summary": posterior_summary,
        "app_A10_arm_losses_all": losses_all,
        "app_A11_arm_loss_summary": losses_summary,
        "app_A12_daily_risk_summary": daily_risk,
        "app_A13_newey_west": newey_west,
        "app_A14_block_bootstrap": block_bootstrap,
        "app_A15_probabilistic_sharpe": psr,
        "app_A16_trial_inclusion_audit": safe_read_csv(
            inference_root / "multiple_testing_trial_inclusion_audit.csv"
        ),
        "app_A17_dsr_approx": safe_read_csv(
            inference_root / "all_trials_deflated_sharpe_approx.csv"
        ),
        "app_A18_spa": safe_read_csv(inference_root / "all_trials_spa_by_seed.csv"),
        "app_A19_romano_wolf": safe_read_csv(
            inference_root / "all_trials_romano_wolf_by_seed.csv"
        ),
        "app_A20_run_manifest_index": manifest_index,
        "app_A21_missing_return_fills": return_fills,
        "app_A22_failures": failures,
        "app_A23_eligibility_by_date": safe_read_csv(
            work_root / "data" / "eligibility_primary" / "eligibility_audit_by_date.csv"
        ),
        "app_A24_eligibility_by_ticker": safe_read_csv(
            work_root / "data" / "eligibility_primary" / "eligibility_audit_by_ticker.csv"
        ),
        "app_A25_data_dictionary": safe_read_csv(work_root / "data" / "data_dictionary.csv"),
        "app_A26_universe": safe_read_csv(work_root / "data" / "universe_derived.csv"),
        "app_A27_riskmap_cells": safe_read_csv(riskmap_root / "cell_metric_summary.csv"),
        "app_A28_riskmap_by_seed": safe_read_csv(
            riskmap_root / "action_channel_decomposition_by_seed.csv"
        ),
        "app_A29_shapley_paired_effects": safe_read_csv(
            shapley_root / "paired_seed_difference_summary.csv"
        ),
        # large diagnostic panels stay in their source files
        "app_A30_monthly_cvar_panel": _external_panel_reference(
            risk_root / "monthly_cvar_calibration_panel.csv",
            "risk_diagnostics/monthly_cvar_calibration_panel.csv",
            "Canonical raw monthly CVaR calibration panel; not duplicated in paper_tables.",
        ),
        "app_A31_daily_path_audit": _external_panel_reference(
            risk_root / "daily_path_reconciliation_audit.csv",
            "risk_diagnostics/daily_path_reconciliation_audit.csv",
            "Canonical daily-path reconciliation audit; not duplicated in paper_tables.",
        ),
        "app_A32_missing_outcome_return_audit": safe_read_csv(
            output_root / "validation" / "missing_outcome_return_audit.csv"
        ),
    }

    all_tables = {**main_tables, **appendix_tables}
    artifacts: list[TableArtifact] = []
    required_main = {
        "main_01_sample_universe",
        "main_03_core_full_performance",
        "main_05_score_action_effects",
        "main_06_riskmap_action_effects",
        "main_07_riskmap_channels",
        "main_08_mean_ablation",
        "main_09_shapley_performance",
        "main_12_cvar_calibration",
        "main_13_selected_robustness",
    }
    for table_id, frame in all_tables.items():
        is_main = table_id.startswith("main_")
        filename = f"{table_id}.csv"
        target = tables_root / filename if is_main else appendix_root / filename
        _write_csv(target, frame)
        artifacts.append(
            TableArtifact(
                table_id=table_id,
                classification="main text" if is_main else "electronic companion",
                title=table_id.replace("_", " ").title(),
                purpose=(
                    "Primary score-action-risk-map evidence"
                    if table_id in {
                        "main_05_score_action_effects",
                        "main_06_riskmap_action_effects",
                        "main_07_riskmap_channels",
                    }
                    else "Reproducible manuscript or appendix result"
                ),
                filename=str(target.relative_to(tables_root)),
                workbook_sheet=table_sheet_name(table_id),
                required=table_id in required_main,
                available=not frame.empty,
                n_rows=int(len(frame)),
                n_columns=int(len(frame.columns)),
                source_files=(
                    "frozen seed outputs and generated input audits"
                    if not frame.empty
                    else "upstream output not yet completed"
                ),
                notes=(
                    "Required headline table is empty; complete its upstream run."
                    if table_id in required_main and frame.empty
                    else ""
                ),
            )
        )

    catalog = write_table_catalog(tables_root / "table_catalog.csv", artifacts)

    # export small tables to Excel; keep full panels in CSV
    workbook_path = tables_root / "paper_tables.xlsx"
    if args.write_xlsx:
        workbook_tables = {table_id: frame for table_id, frame in main_tables.items()}
        workbook_tables["table_catalog"] = catalog
        write_workbook(workbook_tables, workbook_path)
    elif workbook_path.exists():
        workbook_path.unlink()

    markdown = [
        "# Machine-generated paper tables",
        "",
        (
            "The individual CSV files are authoritative. "
            "paper_tables.xlsx is an optional presentation copy containing only main tables."
        ),
        "",
        "## Required-table completion status",
        catalog[catalog["required"]][
            ["table_id", "available", "n_rows", "notes"]
        ].to_markdown(index=False),
    ]
    for table_id in [
        "main_01_sample_universe",
        "main_03_core_full_performance",
        "main_05_score_action_effects",
        "main_06_riskmap_action_effects",
        "main_07_riskmap_channels",
        "main_08_mean_ablation",
        "main_09_shapley_performance",
        "main_12_cvar_calibration",
    ]:
        frame = main_tables[table_id]
        markdown.extend([
            "",
            f"## {table_id}",
            frame.head(80).to_markdown(index=False) if not frame.empty else "*Pending upstream run.*",
        ])
    (tables_root / "paper_tables.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    required_missing = catalog[catalog["required"] & ~catalog["available"]]
    status = {
        "config": str(Path(args.config).resolve()),
        "output_root": str(output_root),
        "n_tables": int(len(catalog)),
        "n_available": int(catalog["available"].sum()),
        "required_missing": required_missing["table_id"].tolist(),
        "paper_ready_tables": bool(required_missing.empty),
        "xlsx_requested": bool(args.write_xlsx),
        "xlsx_available": bool(workbook_path.exists()),
        "authoritative_format": "csv",
    }
    (tables_root / "table_build_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    print(f"Tables written to {tables_root}")


if __name__ == "__main__":
    _main()
