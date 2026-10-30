#!/usr/bin/env python3
"""Export manuscript Tables 2 and 10 without running portfolio experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from _common import ROOT, load_context
from g3_confirmatory.reporting import (
    ci_text, mean_ci, reporting_context, resolve, sources, write_csv, write_json,
)

SAMPLING = "post_cost_utility__hard"
POOL = "post_cost_utility__mix"
FIXED = "fixed_equal__mix"
HIER = "standalone__M_HierMacro"
FULL_RISK = (
    "decompositions/full_risk_map/"
    "post_cost_utility__loss_endogenous__action_hier_full_risk"
)
CLASSES = [
    ("equity", "Equity"), ("bond", "Bond"), ("real_estate", "Real estate"),
    ("metals", "Metals"), ("commodities", "Commodities"), ("mixed", "Mixed"),
    ("cash_fx", "Cash/FX"), ("crypto", "Crypto"),
]


class IncompleteCellError(ValueError):
    """A required result cell has not finished or is absent."""


def normalize_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("save_weights", None)
    result.pop("score_names", None)
    if result.get("common_cvar_budget_source", "none") == "none":
        for field in ("loss_cvar_budget_source", "loss_cvar_scenario_source",
                      "action_cvar_budget_source", "action_cvar_scenario_source"):
            if result.get(field, "inherit") == "inherit":
                result[field] = "none"
    return result


def flattened(mapping: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flattened(value, name))
        else:
            result[name] = value
    return result


def check_config(actual: dict, expected: dict, location: str) -> None:
    if "post_cost_utility" not in actual.get("score_names", []):
        raise ValueError(f"{location}: utility scoring is absent")
    if not {"hard", "mix"}.issubset(actual.get("action_names", [])):
        raise ValueError(f"{location}: both actions are required")
    a, b = flattened(normalize_config(actual)), flattened(normalize_config(expected))
    differences = {key: {"expected": b.get(key), "actual": a.get(key)}
                   for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}
    if differences:
        raise ValueError(f"{location}: configuration mismatch: {differences}")


def check_data_hashes(actual: dict, reference: dict, *, eligibility_variant=False) -> None:
    excluded = {"eligibility", "realized_excess_returns", "realized_simple_returns"} \
        if eligibility_variant else set()
    expected = reference.get("panel_hashes", {})
    observed = actual.get("panel_hashes", {})
    if not expected or not observed:
        raise ValueError("Prepared-data hashes are missing; comparability cannot be checked")
    mismatched = [key for key, value in expected.items()
                  if key not in excluded and observed.get(key) != value]
    if mismatched:
        raise ValueError(f"Prepared-data hashes differ from core: {mismatched}")
    raw_expected = reference.get("source_file_hashes", {})
    raw_observed = actual.get("source_file_hashes", {})
    raw_diff = [key for key, value in raw_expected.items()
                if key != "eligibility" and raw_observed.get(key) != value]
    if raw_diff:
        raise ValueError(f"Original input-file hashes differ from core: {raw_diff}")


def table10_cells() -> list[dict]:
    cells = []
    for scenarios in (250, 500, 1000, 2000):
        for risk in ("endogenous", "common_risk"):
            if scenarios == 2000:
                experiment_id = "core" if risk == "endogenous" else FULL_RISK
            else:
                experiment_id = f"robustness/scenario_count_{scenarios}"
                if risk == "common_risk":
                    experiment_id += "__common_risk"
            cells.append(dict(panel="A", label=f"{scenarios:,} scenarios",
                              scenarios=scenarios, risk=risk,
                              experiment_id=experiment_id, overrides={}))
    for label, name, changes in [
        ("CVaR 90%", "cvar_alpha_0_9", {"optimizer.cvar_alpha": 0.90}),
        ("CVaR 97.5%", "cvar_alpha_0_975", {"optimizer.cvar_alpha": 0.975}),
        ("10% position cap", "position_cap_10pct", {"optimizer.max_weight": 0.10}),
        ("50 bp transaction cost", "cost_50bps", {"optimizer.transaction_cost_bps": 50.0}),
        ("No adjustment penalty", "adjustment_penalty_0_0", {"optimizer.adjustment_penalty": 0.0}),
        ("Trading in 24 of 36 months", "eligibility_history_24_of_36",
         {"min_observations": 24, "max_missing_fraction": 1.0 / 3.0}),
    ]:
        cells.append(dict(panel="B", label=label, scenarios=2000, risk="endogenous",
                          experiment_id="robustness/" + name, overrides=changes))
    return cells


def expected_cell_config(base: dict, cell: dict) -> dict:
    expected = copy.deepcopy(base)
    expected["n_scenarios"] = cell["scenarios"]
    for key, value in cell["overrides"].items():
        if key.startswith("optimizer."):
            expected["optimizer"][key.split(".", 1)[1]] = value
        else:
            expected[key] = value
    if cell["risk"] == "common_risk":
        expected.update(common_cvar_budget_source="none", loss_cvar_budget_source="none",
                        loss_cvar_scenario_source="none", action_cvar_budget_source="M_HierMacro",
                        action_cvar_scenario_source="M_HierMacro")
    return expected


def read_table10_cell(output_root: Path, cell: dict, seeds: list[int],
                      dates: pd.DatetimeIndex, reference: dict):
    from g3_confirmatory.metrics import compute_performance_metrics

    directory = output_root / cell["experiment_id"]
    expected = expected_cell_config(reference["experiment_config"], cell)
    strategies = [SAMPLING, POOL, FIXED, HIER] if cell["risk"] == "endogenous" else [SAMPLING, POOL]
    present = {int(p.parent.name.removeprefix("seed_")) for p in directory.glob("seed_*/returns.csv")}
    extra = sorted(present.difference(seeds))
    if extra:
        raise ValueError(f"{directory}: unexpected seeds {extra}")
    missing = [seed for seed in seeds if seed not in present or not
               (directory / f"seed_{seed}/run_manifest.json").is_file()]
    if missing:
        raise IncompleteCellError(f"{cell['experiment_id']}: missing complete seed outputs {missing}")
    rows, used = [], []
    eligibility_variant = cell["experiment_id"].endswith("eligibility_history_24_of_36")
    for seed in seeds:
        folder = directory / f"seed_{seed}"
        manifest_path, return_path = folder / "run_manifest.json", folder / "returns.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("master_seed", -1)) != seed:
            raise ValueError(f"{manifest_path}: wrong master seed")
        if manifest.get("calendar") != reference.get("calendar"):
            raise ValueError(f"{manifest_path}: calendar differs from the core experiment")
        check_config(manifest.get("experiment_config", {}), expected, str(manifest_path))
        check_data_hashes(manifest.get("data", {}), reference["data"],
                          eligibility_variant=eligibility_variant)
        frame = pd.read_csv(return_path)
        required = {"strategy", "master_seed", "formation_date", "outcome_date",
                    "net_simple_return", "net_excess_return"}
        if not required.issubset(frame):
            raise ValueError(f"{return_path}: missing {sorted(required.difference(frame))}")
        if not pd.to_numeric(frame.master_seed, errors="raise").eq(seed).all():
            raise ValueError(f"{return_path}: wrong/mixed seed identifiers")
        frame = frame.loc[frame.strategy.isin(strategies)].copy()
        frame["outcome_date"] = pd.to_datetime(frame.outcome_date, errors="raise")
        frame["formation_date"] = pd.to_datetime(frame.formation_date, errors="raise")
        if frame.duplicated(["strategy", "outcome_date"]).any():
            raise ValueError(f"{return_path}: duplicate strategy/outcome rows")
        observations = {}
        for strategy in strategies:
            group = frame.loc[frame.strategy.eq(strategy)].sort_values("outcome_date")
            if not pd.DatetimeIndex(group.outcome_date).equals(dates):
                raise IncompleteCellError(f"{return_path}: {strategy} lacks the exact configured outcome calendar")
            if not np.array_equal(group.formation_date.to_numpy(),
                                  (dates - pd.offsets.MonthEnd(1)).to_numpy()):
                raise ValueError(f"{return_path}: incorrect formation-to-outcome alignment")
            if not np.isfinite(group[["net_simple_return", "net_excess_return"]].to_numpy(float)).all():
                raise ValueError(f"{return_path}: nonfinite returns")
            metrics = compute_performance_metrics(group, utility_gamma=expected["utility_gamma"])
            observations[strategy] = float(metrics["excess_sharpe"])
        if not np.isfinite(list(observations.values())).all():
            raise ValueError(f"{return_path}: nonfinite Sharpe ratio")
        row = dict(panel=cell["panel"], specification=cell["label"],
                   n_scenarios=cell["scenarios"], risk_specification=cell["risk"],
                   experiment_id=cell["experiment_id"], master_seed=seed,
                   sampling_sharpe=observations[SAMPLING], pool_sharpe=observations[POOL],
                   delta_sharpe=observations[POOL] - observations[SAMPLING])
        if cell["risk"] == "endogenous":
            row.update(fixed_pool_sharpe=observations[FIXED], hiermacro_sharpe=observations[HIER])
        rows.append(row)
        used.extend([manifest_path, return_path])
    frame = pd.DataFrame(rows)
    summary = dict(panel=cell["panel"], specification=cell["label"],
                   n_scenarios=cell["scenarios"], risk_specification=cell["risk"],
                   experiment_id=cell["experiment_id"], n_seeds=len(seeds), n_months=len(dates),
                   cvar_alpha=expected["optimizer"]["cvar_alpha"],
                   effective_tail_count=(1.0 - expected["optimizer"]["cvar_alpha"]) * cell["scenarios"])
    for metric in ["sampling_sharpe", "pool_sharpe", "delta_sharpe", "fixed_pool_sharpe", "hiermacro_sharpe"]:
        if metric in frame:
            result = mean_ci(frame[metric])
            for name in ("mean", "ci_low", "ci_high"):
                summary[metric + "_" + name] = result[name]
    return frame, summary, used


def table10_wide(cells: list[dict], summaries: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows, display_rows = [], []
    for cell in cells:
        if cell["risk"] != "endogenous":
            continue
        end = summaries.get(cell["experiment_id"])
        common_id = FULL_RISK if cell["scenarios"] == 2000 else \
            f"robustness/scenario_count_{cell['scenarios']}__common_risk"
        common = summaries.get(common_id) if cell["panel"] == "A" else None
        row = dict(panel=cell["panel"], specification=cell["label"], n_scenarios=cell["scenarios"],
                   endogenous_status="complete" if end else "missing",
                   common_risk_status=("complete" if common else "missing") if cell["panel"] == "A" else "not_applicable")
        displayed = dict(Panel=cell["panel"], Specification=cell["label"])
        for metric, label in [("sampling_sharpe", "Utility sampling"), ("pool_sharpe", "Utility pool"),
                              ("fixed_pool_sharpe", "Fixed pool"), ("hiermacro_sharpe", "HierMacro")]:
            row[metric] = end[metric + "_mean"] if end else np.nan
            displayed[label] = f"{row[metric]:.3f}" if end else "MISSING"
        for name, result, label in [("delta_sharpe", end, "Pool - sampling [95% CI]"),
                                    ("common_risk_delta_sharpe", common, "Common-risk pool - sampling [95% CI]")]:
            for part in ("mean", "ci_low", "ci_high"):
                row[name + "_" + part] = result["delta_sharpe_" + part] if result else np.nan
            displayed[label] = ci_text(*(row[name + "_" + p] for p in ("mean", "ci_low", "ci_high"))) \
                if result else ("MISSING" if cell["panel"] == "A" or name == "delta_sharpe" else "")
        row["n_paired_seeds"] = end["n_seeds"] if end else 0
        row["cvar_alpha"] = end["cvar_alpha"] if end else np.nan
        row["effective_tail_count"] = end["effective_tail_count"] if end else np.nan
        numeric_rows.append(row)
        display_rows.append(displayed)
    return pd.DataFrame(numeric_rows), pd.DataFrame(display_rows)[[
        "Panel", "Specification", "Utility sampling", "Utility pool", "Pool - sampling [95% CI]",
        "Common-risk pool - sampling [95% CI]", "Fixed pool", "HierMacro",
    ]]


def export_table10(output_root: Path, destination: Path, seeds: list[int],
                   dates: pd.DatetimeIndex, config_path: Path, *, allow_incomplete=False) -> dict:
    reference_path = output_root / f"core/seed_{seeds[0]}/run_manifest.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    base = reference["experiment_config"]
    if base["n_scenarios"] != 2000 or base["optimizer"]["cvar_alpha"] != 0.95:
        raise ValueError("Table 10 requires the primary 2,000-scenario, 95%-CVaR core run")
    expected_dates = pd.period_range(reference["calendar"]["outcome_start"],
                                     reference["calendar"]["outcome_end"], freq="M").to_timestamp("M")
    if not dates.equals(expected_dates) or reference["calendar"]["n_outcomes"] != len(dates):
        raise ValueError("Configured reporting dates disagree with archived core dates")
    cells, summaries, frames, used, missing = table10_cells(), {}, [], [config_path], []
    for cell in cells:
        try:
            frame, summary, paths = read_table10_cell(output_root, cell, seeds, dates, reference)
        except IncompleteCellError as exc:
            missing.append({"experiment_id": cell["experiment_id"], "reason": str(exc)})
            continue
        summaries[cell["experiment_id"]] = summary
        frames.append(frame)
        used.extend(paths)
    if missing and not allow_incomplete:
        message = "\n".join(item["reason"] for item in missing)
        raise IncompleteCellError("Table 10 is incomplete; no table files were written.\n" + message +
                                  "\nAfter the runs finish, rerun this command; --allow-incomplete writes a labelled preview only.")
    numeric, display = table10_wide(cells, summaries)
    suffix = ".partial" if missing else ""
    outputs = []
    for name, frame in [("table_10_numeric", numeric), ("table_10_formatted", display),
                        ("table_10_cells", pd.DataFrame(list(summaries.values()))),
                        ("table_10_by_seed", pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())]:
        target = destination / f"{name}{suffix}.csv"
        write_csv(target, frame)
        outputs.append(target)
    report = {"table": 10, "complete": not missing, "required_cells": len(cells),
              "complete_cells": len(summaries), "missing_cells": missing,
              "seeds": seeds, "outcome_start": str(dates.min().date()),
              "outcome_end": str(dates.max().date()), "n_outcomes": len(dates),
              "method": "Recompute each seed's Sharpe from saved net excess returns; pool minus sampling within seed; Student-t CI across paired seeds.",
              "uncertainty_scope": "Algorithmic seeds conditional on one market history; not historical-path inference.",
              "sources": sources(used), "outputs": sources(outputs)}
    write_json(destination / f"table_10_manifest{suffix}.json", report)
    print(f"Table 10: {'PARTIAL PREVIEW' if missing else 'complete'}; {len(summaries)}/{len(cells)} cells; {destination}")
    return report


def calculate_table2(data, min_months: int = 12):
    calendar = data.calendar[["formation_date", "outcome_date"]].copy()
    for name in calendar:
        calendar[name] = pd.to_datetime(calendar[name], errors="raise")
        if calendar[name].duplicated().any():
            raise ValueError(f"Duplicate {name} in the calendar")
    calendar = calendar.sort_values("formation_date")
    formations = pd.DatetimeIndex(calendar.formation_date)
    outcomes = pd.DatetimeIndex(calendar.outcome_date)
    if not (outcomes == formations + pd.offsets.MonthEnd(1)).all():
        raise ValueError("Formation dates must map to the following outcome month")
    if data.eligibility.index.has_duplicates or data.eligibility.columns.has_duplicates:
        raise ValueError("Duplicate eligibility dates or tickers")
    eligibility = data.eligibility.reindex(formations)
    if eligibility.isna().any().any():
        raise ValueError("Missing formation-date eligibility flags")
    if not eligibility.isin([True, False, 0, 1]).all().all():
        raise ValueError("Eligibility must contain boolean/0/1 values")
    eligibility = eligibility.astype(bool)
    eligibility = eligibility.loc[:, eligibility.any(axis=0)]
    classes = data.asset_classes.reindex(eligibility.columns)
    if classes.isna().any() or not classes.isin(dict(CLASSES)).all():
        raise ValueError("Missing or unsupported asset-class mapping")
    raw_returns = data.outcome_excess_returns
    if raw_returns.index.has_duplicates or raw_returns.columns.has_duplicates:
        raise ValueError("Duplicate outcome-return dates or tickers")
    returns = raw_returns.reindex(index=outcomes, columns=eligibility.columns).astype(float)
    flags = eligibility.to_numpy(bool)
    values = returns.to_numpy(float)
    if not np.isfinite(values[flags]).all():
        raise ValueError("Nonfinite returns for eligible holding-period observations; do not silently drop them")
    records, panel_rows = [], []
    for index, ticker in enumerate(eligibility.columns):
        mask = flags[:, index]
        x = values[mask, index]
        n = len(x)
        constant = n < 2 or np.all(x == x[0])
        records.append({"ticker": ticker, "asset_class": classes[ticker],
                        "eligible_months": n, "used_return_months": n,
                        "included_in_moments": n >= min_months,
                        "annualized_mean_pct": 100 * 12 * float(x.mean()),
                        "annualized_volatility_pct": 100 * np.sqrt(12) * float(x.std(ddof=1)) if n >= 2 else np.nan,
                        "skewness": float(skew(x, bias=False)) if n >= 3 and not constant else np.nan,
                        "excess_kurtosis": float(kurtosis(x, fisher=True, bias=False)) if n >= 4 and not constant else np.nan})
        for formation, outcome, value in zip(formations[mask], outcomes[mask], x):
            panel_rows.append({"ticker": ticker, "asset_class": classes[ticker],
                               "formation_date": formation.date().isoformat(),
                               "outcome_date": outcome.date().isoformat(), "excess_return": float(value)})
    per_etf = pd.DataFrame(records)
    included = per_etf.loc[per_etf.included_in_moments]
    moment_columns = ["annualized_mean_pct", "annualized_volatility_pct", "skewness", "excess_kurtosis"]
    if included.empty:
        raise ValueError("No ETF has at least 12 eligible observations")
    bad = ~np.isfinite(included[moment_columns].to_numpy(float)).all(axis=1)
    if bad.any():
        raise ValueError("Undefined moments among qualifying ETFs (constant returns or inadequate samples): "
                         + ", ".join(included.loc[bad, "ticker"].astype(str)))
    summary = []
    for key, label in CLASSES + [("all", "All")]:
        group = per_etf if key == "all" else per_etf.loc[per_etf.asset_class.eq(key)]
        qualified = group.loc[group.included_in_moments]
        summary.append({"asset_class": label, "n_etfs": len(group),
                        "n_etfs_at_least_12_months": len(qualified),
                        "average_eligible_months": qualified.eligible_months.mean(),
                        "mean_pct_pa": qualified.annualized_mean_pct.mean(),
                        "volatility_pct_pa": qualified.annualized_volatility_pct.mean(),
                        "median_skewness": qualified.skewness.median(),
                        "median_excess_kurtosis": qualified.excess_kurtosis.median()})
    return pd.DataFrame(summary), per_etf, pd.DataFrame(panel_rows)


def export_table2(config_path: Path, output_root: Path, destination: Path,
                  seeds: list[int], dates: pd.DatetimeIndex) -> dict:
    from _common import load_context

    cfg, paths, data, _, _, _ = load_context(config_path)
    if not pd.DatetimeIndex(data.outcome_dates).equals(dates):
        raise ValueError("Prepared-data calendar differs from configured outcomes")
    reference_path = output_root / f"core/seed_{seeds[0]}/run_manifest.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    check_data_hashes(data.source_metadata, reference["data"])
    summary, per_etf, panel = calculate_table2(data)
    expected = int(cfg.get("raw_data", {}).get("expected_universe_size", 470))
    if len(per_etf) != expected:
        raise ValueError(f"Expected {expected} ever-eligible ETFs, found {len(per_etf)}")
    universe_path = paths["eligibility_path"].parent / "universe_derived.csv"
    used = [config_path, reference_path, paths["eligibility_path"]]
    if universe_path.exists():
        universe = pd.read_csv(universe_path).set_index("ticker")
        actual = per_etf.set_index("ticker")
        if set(universe.index) != set(actual.index):
            raise ValueError("Ever-eligible tickers disagree with universe_derived.csv")
        if not np.array_equal(universe.reindex(actual.index).eligible_months.to_numpy(), actual.eligible_months.to_numpy()):
            raise ValueError("Eligible-month counts disagree with universe_derived.csv")
        used.append(universe_path)
    display = summary.copy()
    for col in ["average_eligible_months", "mean_pct_pa", "volatility_pct_pa",
                "median_skewness", "median_excess_kurtosis"]:
        digits = 1 if col == "average_eligible_months" else 2
        display[col] = display[col].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    outputs = []
    for name, frame in [("table_02_numeric.csv", summary), ("table_02_formatted.csv", display),
                        ("table_02_per_etf.csv", per_etf), ("table_02_eligible_outcomes.csv.gz", panel)]:
        path = destination / name
        write_csv(path, frame)
        outputs.append(path)
    report = {"table": 2, "complete": True, "n_etfs": len(per_etf),
              "n_qualifying_etfs": int(per_etf.included_in_moments.sum()),
              "n_eligible_outcomes": len(panel), "sources": sources(used), "outputs": sources(outputs),
              "source_metadata": data.source_metadata,
              "definitions": {"eligibility": "Formation month t matched to outcome month t+1; no outcome-month eligibility selection.",
                              "returns": "DataBundle.outcome_excess_returns: asset returns net of the risk-free rate, with the primary missing-outcome accounting; no portfolio transaction costs.",
                              "annualization": "12 * monthly mean; sqrt(12) * sample standard deviation (ddof=1); percentages.",
                              "higher_moments": "scipy.stats.skew(bias=False); kurtosis(fisher=True, bias=False). Explicit convention, not inferred from unretained original Table 2 code.",
                              "aggregation": "Means of per-ETF annualized means/volatilities; medians of per-ETF higher moments; all among ETFs with at least 12 eligible months.",
                              "average_months": "Mean eligible months among that same >=12-month subset, not all ever-eligible ETFs."}}
    write_json(destination / "table_02_manifest.json", report)
    print(f"Table 2: {len(per_etf)} ETFs, {report['n_qualifying_etfs']} qualifying; {destination}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/confirmatory.yaml")
    parser.add_argument("--tables", default="2,10", help="Comma-separated 2,10; choose 10 to avoid loading raw inputs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Table 10 only: write .partial.csv previews, never replace final files")
    args = parser.parse_args()
    requested = [item.strip() for item in args.tables.split(",") if item.strip()]
    if not requested or set(requested).difference({"2", "10"}):
        parser.error("--tables must be 2, 10, or 2,10")
    config_path, cfg, paths, seeds, dates = reporting_context(args.config)
    if len(seeds) < 2:
        parser.error("At least two configured seeds are required for Table 10 intervals")
    destination = resolve(args.output_dir) if args.output_dir else paths["output_root"] / "paper_tables/manuscript_additions"
    if "2" in requested:
        export_table2(config_path, paths["output_root"], destination, seeds, dates)
    if "10" in requested:
        export_table10(paths["output_root"], destination, seeds, dates, config_path,
                       allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
