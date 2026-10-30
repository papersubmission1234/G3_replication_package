#!/usr/bin/env python3
"""Prepare eligibility and study configuration from the five original input files.

Reads Meta.csv, Portfolio Macro Data.xlsx, adjusted_close_prices.csv, Volume.csv,
and ff5_plus_momentum_monthly_2016_2025.csv. Derived files go under work/data;
source files stay in the input folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from _common import ROOT
from g3_confirmatory.configuration import load_yaml, experiment_config_from_mapping, configured_seeds
from g3_confirmatory.asset_classes import normalize_asset_classes
from g3_confirmatory.eligibility import (
    EligibilityConfig,
    build_eligibility_matrix,
    build_formation_outcome_map,
    infer_terminal_delisting_events,
    write_eligibility_result,
)
from g3_confirmatory.hashing import sha256_file
from g3_confirmatory.serialization import write_json

REQUIRED_FILES = (
    "Meta.csv",
    "Portfolio Macro Data.xlsx",
    "adjusted_close_prices.csv",
    "Volume.csv",
    "ff5_plus_momentum_monthly_2016_2025.csv",
)

RAW_START = pd.Timestamp("2016-01-04")
RAW_END = pd.Timestamp("2025-12-31")
OUTCOME_START = "2019-02-28"
OUTCOME_END = "2025-12-31"


def _read_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, dtype=str, quotechar='"')
    def conv(x):
        if pd.isna(x) or x.strip() == '' or x.strip().lower() == 'no data':
            return np.nan
        s = x.replace(',', '.')
        try:
            return float(s)
        except ValueError:
            return np.nan
    frame = frame.map(conv)
    frame.index = pd.to_datetime(frame.index, errors='coerce')
    if frame.index.isna().any():
        raise ValueError(f"Unparseable dates in {path}")
    frame = frame.sort_index().apply(pd.to_numeric, errors='coerce')
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).last()
    return frame.loc[(frame.index >= RAW_START) & (frame.index <= RAW_END)]


def _meta_columns(meta: pd.DataFrame) -> tuple[str, str]:
    ticker = "Ticker"
    object_col = "Investment Object"
    if ticker not in meta.columns or object_col not in meta.columns:
        raise ValueError(
            "Meta.csv must contain a ticker column and an investment-object/asset-class column. "
            f"Found columns: {meta.columns.tolist()}"
        )
    return ticker, object_col


def _build_data_dictionary(
    input_dir: Path,
    meta: pd.DataFrame,
    price: pd.DataFrame,
    volume: pd.DataFrame,
    factor: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_file(
        filename: str,
        role: str,
        frequency: str,
        start: str | None,
        end: str | None,
        columns: int | None,
        notes: str,
    ) -> None:
        path = input_dir / filename
        rows.append(
            {
                "source_file": filename,
                "field": "*",
                "role": role,
                "frequency": frequency,
                "date_start": start,
                "date_end": end,
                "n_columns": columns,
                "sha256": sha256_file(path),
                "derivation_or_units": notes,
            }
        )

    add_file(
        "Meta.csv",
        "ETF metadata and asset-class mapping",
        "static",
        None,
        None,
        int(meta.shape[1]),
        "Ticker and investment-object labels are used; all supported asset classes are retained.",
    )
    add_file(
        "Portfolio Macro Data.xlsx",
        "Macro, rate, regime, and transaction-cost predictors",
        "daily workbook",
        "2016-01-04",
        "2025-12-31 study cutoff",
        None,
        "Existing workbook sheet/field transformations in bayesian_factor_inputs2.py are used.",
    )
    add_file(
        "adjusted_close_prices.csv",
        "Adjusted prices; realized simple returns and current-tradability price check",
        "daily",
        str(price.index.min().date()),
        str(price.index.max().date()),
        int(price.shape[1]),
        "Monthly realized simple returns use month-end adjusted prices. Prices are not multiplied into Volume.csv because that file already contains dollar trading amount.",
    )
    add_file(
        "Volume.csv",
        "Trading activity/liquidity",
        "daily",
        str(volume.index.min().date()),
        str(volume.index.max().date()),
        int(volume.shape[1]),
        "The supplied values are already dollar trading amounts. 'No data' is parsed as missing; monthly trading amount is the sum of observed daily dollar amounts.",
    )
    add_file(
        "ff5_plus_momentum_monthly_2016_2025.csv",
        "Fama-French 5 + momentum + RF",
        "monthly",
        str(factor.index.min().date()),
        str(factor.index.max().date()),
        int(factor.shape[1]),
        "Loaded locally first; no network download is required.",
    )
    return pd.DataFrame(rows)


def _derive_notebook_universe(
    *,
    volume: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    formation_dates: pd.DatetimeIndex,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Notebook comparison: 37 active months, lagged relative liquidity, and future activity.

    This diagnostic uses future information. The universe includes every ticker
    eligible in any supported month of the raw sample.
    """
    vol = volume.copy()
    px = adjusted_close.copy()
    vol.index = pd.to_datetime(vol.index)
    px.index = pd.to_datetime(px.index)
    cols = vol.columns.intersection(px.columns).sort_values()
    vol = vol.reindex(columns=cols).apply(pd.to_numeric, errors='coerce').clip(lower=0)
    px = px.reindex(columns=cols).apply(pd.to_numeric, errors='coerce')

    monthly_vol = vol.resample('ME').sum(min_count=1).fillna(0.0)
    monthly_px = px.resample('ME').last()
    # current month plus the preceding 36 months
    history_ok = monthly_vol.replace(0.0, np.nan).rolling(
        37, min_periods=0
    ).count().gt(36)

    rolling36 = monthly_vol.rolling(36).mean()
    relative = rolling36.div(rolling36.sum(axis=1), axis=0)
    liquidity_ok = relative.shift(1).gt(1.0 / 10_000.0)

    # diagnostic only: this screen uses future activity
    future_any = (
        monthly_vol.gt(0.0)
        .shift(-1)
        .iloc[::-1]
        .cummax()
        .iloc[::-1]
        .fillna(False)
    )
    # carry the preceding oracle value into the final month
    if len(future_any.index) >= 2:
        future_any.iloc[-1] = future_any.iloc[-1] | future_any.iloc[-2]

    notebook_eligibility = history_ok & liquidity_ok & future_any
    ever = notebook_eligibility.any(axis=0)
    tickers = ever[ever].index.astype(str).tolist()

    audit = pd.DataFrame({
        'ticker': notebook_eligibility.columns.astype(str),
        'notebook_eligible_months': notebook_eligibility.sum(axis=0).astype(int).to_numpy(),
        'notebook_ever_eligible': ever.to_numpy(dtype=bool),
        'first_notebook_eligible': [
            notebook_eligibility.index[notebook_eligibility[c].to_numpy()].min()
            if notebook_eligibility[c].any() else pd.NaT
            for c in notebook_eligibility.columns
        ],
        'last_notebook_eligible': [
            notebook_eligibility.index[notebook_eligibility[c].to_numpy()].max()
            if notebook_eligibility[c].any() else pd.NaT
            for c in notebook_eligibility.columns
        ],
    })
    return tickers, notebook_eligibility.astype(np.int8), audit


def _generated_config(
    *,
    work_root: Path,
    input_dir: Path,
    universe_path: Path,
    universe_size: int,
    eligibility_path: Path,
    delisting_path: Path,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "paths": {
            "input_dir": str(input_dir.resolve()),
            "eligibility_path": str(eligibility_path.resolve()),
            "output_root": str((work_root / "outputs").resolve()),
            # future inactivity is not an announcement date
            "delisting_events_csv": None,
        },
        "raw_data": {
            "meta_csv": str((input_dir / "Meta.csv").resolve()),
            "macro_workbook": str((input_dir / "Portfolio Macro Data.xlsx").resolve()),
            "volume_csv": str((input_dir / "Volume.csv").resolve()),
            "adjusted_close_csv": str((input_dir / "adjusted_close_prices.csv").resolve()),
            "ff5_plus_momentum_csv": str(
                (input_dir / "ff5_plus_momentum_monthly_2016_2025.csv").resolve()
            ),
            "universe_tickers_file": str(universe_path.resolve()),
            "expected_universe_size": int(universe_size),
            "oracle_delisting_diagnostic_csv": str(delisting_path.resolve()),
        },
        "calendar": {
            "outcome_start": OUTCOME_START,
            "outcome_end": OUTCOME_END,
            "expected_outcomes": 83,
        },
        "eligibility": {
            "history_months": 36,
            "liquidity_window_months": 36,
            # history excludes the formation month; tradability is checked separately
            "signal_lag_months": 1,
            "formation_after_month_end_close": True,
            "relative_liquidity_threshold": 0.0001,
            "liquidity_measure": "dollar_volume",
            "require_every_history_month_active": True,
            "min_active_history_months": None,
            "volume_is_dollar": True,
            "price_proxy": "adjusted_close",
            "max_current_price_staleness_days": 10,
            "delisting_mode": "current_only",
        },
        "realized_returns": {
            "policy": "last_trade_to_cash",
            "fixed_haircut": 0.30,
            "accrue_cash_after_last_quote": True,
            "max_formation_price_staleness_days": 10,
        },
        "experiment": {
            "training_months": 36,
            "information_lag_months": 0,
            "min_observations": 36,
            "max_missing_fraction": 0.0,
            "min_assets": 5,
            "n_scenarios": 500,
            "posterior_calibration_months": 24,
            "posterior_forgetting": 1.0,
            "utility_gamma": 1.0,
            "score_names": ["predictive_log", "post_cost_utility"],
            "action_names": ["hard", "mix"],
            "include_equal_mix": True,
            "arm_names": ["M_FF5MOM", "M_HierMacro"],
            "class_style_policy": "default",
            "included_asset_classes": None,
            "hierarchical_forecast_mode": "mean_zero",
            "hierarchical_mean_treatment": "base_empirical",
            "mean_shrinkage_fraction": 0.0,
            "posterior_partition": "pooled",
            "state_calibration_months": 8,
            "state_pooling_weight": 0.5,
            "common_cvar_budget_source": "none",
            "initial_wealth": 1000.0,
            "apply_initial_funding_cost": True,
            "fallback_policy": "fail",
            "save_weights": True,
            "optimizer": {
                "cvar_alpha": 0.95,
                "max_weight": 0.20,
                "active_min_weight": 0.0005,
                "adjustment_penalty": 0.10,
                "transaction_cost_bps": 10.0,
                "cvar_budget_mode": "scenario_equal_weight",
                "cvar_budget": None,
                "highs_options": None,
            },
        },
        "regime": {
            "vix_column": "VIX",
            "hy_oas_column": "HY OAS",
            "source_units": "decimal",
            "lag_months": 1,
            "stress_vix": 18.0,
            "crisis_vix": 25.0,
            "stress_hy_oas_bps": 350.0,
            "crisis_hy_oas_bps": 500.0,
        },
        "seed_start": 20260322,
        "n_seeds": 30,
        "inference": {
            "primary_bootstrap_replications": 5000,
            "multiple_testing_bootstrap_replications": 5000,
            "mean_block_length_months": 6.0,
            "bootstrap_seed": 20260322,
        },
    }
    template = load_yaml(ROOT / "config" / "confirmatory.yaml") if template is None else template
    experiment_config_from_mapping(template.get("experiment")).validate()
    configured_seeds(template)
    for key in ("experiment", "regime", "inference", "parallel"):
        if key in template:
            result[key] = template[key]
    for key in ("seed_start", "n_seeds", "seeds"):
        result.pop(key, None)
    if "seeds" in template:
        result["seeds"] = template["seeds"]
    else:
        result["seed_start"] = template.get("seed_start", 20260322)
        result["n_seeds"] = template.get("n_seeds", 30)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "input_data",
        help="Folder containing exactly the five required source files.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "work",
        help="Folder for derived data, configuration, and outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated data and configuration if they already exist.",
    )
    parser.add_argument(
        "--base-config", type=Path, default=ROOT / "config" / "confirmatory.yaml",
        help="Inherit experiment and seed settings from this config; default is the main study config.",
    )
    args = parser.parse_args()
    template = load_yaml(args.base_config)
    experiment_config_from_mapping(template.get("experiment")).validate()
    configured_seeds(template)

    input_dir = args.input_dir.resolve()
    work_root = args.work_root.resolve()
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise SystemExit(
            "Missing required input file(s): " + ", ".join(missing) +
            f"\nPut all five files in: {input_dir}"
        )

    generated_data = work_root / "data"
    config_dir = work_root / "config"
    for path in (generated_data, config_dir, work_root / "outputs"):
        path.mkdir(parents=True, exist_ok=True)

    marker = generated_data / "study_build_manifest.json"
    if marker.exists() and not args.overwrite:
        raise SystemExit(
            f"Generated inputs already exist at {generated_data}. "
            "Use --overwrite to regenerate them."
        )

    meta = pd.read_csv(input_dir / "Meta.csv")
    ticker_col, object_col = _meta_columns(meta)
    meta = meta.dropna(subset=[ticker_col]).drop_duplicates(ticker_col, keep="first")
    meta[ticker_col] = meta[ticker_col].astype(str)
    meta_index = pd.Index(meta[ticker_col])

    adjusted = _read_panel(input_dir / "adjusted_close_prices.csv")
    volume = _read_panel(input_dir / "Volume.csv")
    factor = pd.read_csv(
        input_dir / "ff5_plus_momentum_monthly_2016_2025.csv",
        index_col=0,
        parse_dates=True,
    )
    factor.index = pd.to_datetime(factor.index).to_period("M").to_timestamp("M")
    factor = factor.sort_index().apply(pd.to_numeric, errors="coerce")

    # tickers must occur in all three input panels
    candidate = (
        adjusted.columns.intersection(volume.columns)
        .intersection(meta_index)
        .sort_values()
    )
    if len(candidate) == 0:
        raise SystemExit("The five files have no common ETF tickers.")

    adjusted = adjusted.reindex(columns=candidate)
    volume = volume.reindex(columns=candidate)

    raw_classes = meta.set_index(ticker_col)[object_col].reindex(candidate)
    normalized_classes = normalize_asset_classes(raw_classes, unknown_policy="error")

    timing = build_formation_outcome_map(
        outcome_start=OUTCOME_START,
        outcome_end=OUTCOME_END,
    )
    formation_dates = pd.DatetimeIndex(timing["formation_date"])
    settings = EligibilityConfig(
        history_months=36,
        liquidity_window_months=36,
        signal_lag_months=1,
        relative_liquidity_threshold=1.0 / 10_000.0,
        require_every_history_month_active=True,
        # Volume.csv already contains dollar trading amounts
        liquidity_measure="dollar_volume",
        volume_is_dollar=True,
        price_proxy="adjusted_close",
        max_current_price_staleness_days=10,
        delisting_mode="current_only",
        formation_after_month_end_close=True,
    )

    full_result = build_eligibility_matrix(
        volume=volume,
        adjusted_close=adjusted,
        formation_dates=formation_dates,
        config=settings,
    )
    ever_eligible = full_result.eligibility.any(axis=0)
    derived_tickers = ever_eligible[ever_eligible].index.astype(str).tolist()
    if not derived_tickers:
        raise SystemExit("The primary eligibility rule produced an empty universe.")

    # keep the full-panel liquidity denominator
    result = build_eligibility_matrix(
        volume=volume,
        adjusted_close=adjusted,
        formation_dates=formation_dates,
        universe_tickers=derived_tickers,
        config=settings,
    )

    eligibility_dir = generated_data / "eligibility_primary"
    if eligibility_dir.exists() and args.overwrite:
        shutil.rmtree(eligibility_dir)
    write_eligibility_result(result, eligibility_dir)
    eligibility_path = generated_data / "eligibility.csv"
    result.eligibility.to_csv(eligibility_path)

    eligibility_variants = {
        "eligibility_current_only": settings,
        "eligibility_oracle_future_activity": replace(
            settings, delisting_mode="oracle_future_activity"
        ),
        "eligibility_oracle_next_month_activity": replace(
            settings, delisting_mode="oracle_next_month_activity"
        ),
        "eligibility_notebook_raw_volume": replace(
            settings, liquidity_measure="raw_volume"
        ),
        "eligibility_dollar_volume": settings,
        "eligibility_history_24_of_36": replace(
            settings,
            require_every_history_month_active=False,
            min_active_history_months=24,
        ),
    }
    for variant_name, variant_settings in eligibility_variants.items():
        variant = build_eligibility_matrix(
            volume=volume,
            adjusted_close=adjusted,
            formation_dates=formation_dates,
            universe_tickers=derived_tickers,
            config=variant_settings,
        )
        variant_dir = generated_data / variant_name
        if variant_dir.exists() and args.overwrite:
            shutil.rmtree(variant_dir)
        write_eligibility_result(variant, variant_dir)

    universe = result.audit_by_ticker.copy()
    universe.insert(0, "ticker", universe.index.astype(str))
    universe["asset_class"] = normalized_classes.reindex(universe["ticker"]).to_numpy()
    universe["raw_asset_class"] = raw_classes.reindex(universe["ticker"]).astype("string").to_numpy()
    universe["selected_by_rule"] = True
    universe["universe_selection_rule"] = (
        "ever eligible under the primary current-only supplied-dollar-volume rule; "
        "monthly eligibility remains point-in-time"
    )
    universe["liquidity_reference_universe_size"] = int(len(candidate))
    universe_path = generated_data / "universe_derived.csv"
    universe.to_csv(universe_path, index=False)

    n_universe = len(derived_tickers)
    alias = generated_data / f"universe_{n_universe}.csv"
    shutil.copy2(universe_path, alias)
    if n_universe == 470:
        shutil.copy2(universe_path, generated_data / "universe_470.csv")

    inferred_delisting = infer_terminal_delisting_events(volume, adjusted)
    delisting_path = generated_data / "delisting_events_oracle_diagnostic.csv"
    inferred_delisting.to_csv(delisting_path, index=False)

    dictionary = _build_data_dictionary(
        input_dir, meta, adjusted, volume, factor
    )
    dictionary.to_csv(generated_data / "data_dictionary.csv", index=False)

    classes_out = pd.DataFrame(
        {
            "ticker": candidate.astype(str),
            "raw_asset_class": raw_classes.reindex(candidate).astype("string").to_numpy(),
            "asset_class": normalized_classes.reindex(candidate).to_numpy(),
        }
    )
    classes_out.to_csv(generated_data / "asset_classes.csv", index=False)

    generated_cfg = _generated_config(
        work_root=work_root,
        input_dir=input_dir,
        universe_path=universe_path,
        universe_size=n_universe,
        eligibility_path=eligibility_path,
        delisting_path=delisting_path,
        template=template,
    )
    config_path = config_dir / "generated_study_config.yaml"
    config_path.write_text(
        yaml.safe_dump(generated_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    source_hashes = {name: sha256_file(input_dir / name) for name in REQUIRED_FILES}
    manifest = {
        "input_dir": str(input_dir),
        "work_root": str(work_root),
        "config_path": str(config_path),
        "settings_template": str(args.base_config.resolve()),
        "n_scenarios": generated_cfg["experiment"]["n_scenarios"],
        "source_hashes": source_hashes,
        "raw_date_limit": [str(RAW_START.date()), str(RAW_END.date())],
        "candidate_common_tickers": int(len(candidate)),
        "derived_universe_size": int(n_universe),
        "derived_universe_is_exactly_470": bool(n_universe == 470),
        "formation_count": int(len(formation_dates)),
        "formation_start": str(formation_dates.min().date()),
        "formation_end": str(formation_dates.max().date()),
        "outcome_count": int(len(timing)),
        "outcome_start": OUTCOME_START,
        "outcome_end": OUTCOME_END,
        "timing_rule": (
            "after-close formation; 36-month history/training window includes "
            "formation month (lag 0)"
        ),
        "first_training_window": "2016-02-29 through 2019-01-31",
        "delisting_rule": (
            "primary eligibility is current-only and uses no future disappearance; "
            "the inferred terminal future-activity file is diagnostic only"
        ),
        "missing_outcome_return_rule": (
            "last observed adjusted-price return during the holding month, followed "
            "by cash accrual to month-end; 30% and 55% haircut plus total-loss sensitivities "
            "are generated separately"
        ),
        "liquidity_rule": (
            "36-month rolling mean of supplied dollar trading amount over "
            "[t-36,t-1], divided by the rolling total over the full candidate "
            "universe; threshold 1/10,000"
        ),
        "generated_files": {
            "eligibility": str(eligibility_path),
            "universe": str(universe_path),
            "delisting_audit": str(delisting_path),
            "data_dictionary": str(generated_data / "data_dictionary.csv"),
            "asset_classes": str(generated_data / "asset_classes.csv"),
            "config": str(config_path),
        },
    }
    write_json(marker, manifest)

    next_steps = f"""Generated successfully.\n\nDerived universe: {n_universe} ETFs from {len(candidate)} common candidate tickers.\nConfig: {config_path}\n\nRun next from the repository root:\n\npython scripts/01_validate_inputs.py --config \"{config_path}\"\npython scripts/09_validate_theory.py --config \"{config_path}\"\npython scripts/02_run_core.py --config \"{config_path}\" --seed 20260322\n\nAfter the one-seed smoke test succeeds, run the full experiment families without --seed.\n"""
    (work_root / "RUN_NEXT.txt").write_text(next_steps, encoding="utf-8")

    print(next_steps)
    if n_universe != 470:
        print(
            "NOTE: the corrected automatic rules produced "
            f"{n_universe} ETFs, not 470. The script does not force the count; "
            "forcing exactly 470 would add an undocumented selection rule."
        )
    print(
        "NOTE: delisting_events_oracle_diagnostic.csv is an audit file inferred "
        "from future activity. It is never used by the primary current-only rule "
        "and is not evidence of actual announcement dates."
    )


if __name__ == "__main__":
    main()