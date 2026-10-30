#!/usr/bin/env python3
"""Check input coverage against the configured study calendar."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import ROOT, write_json
from g3_confirmatory.asset_classes import normalize_asset_classes
from g3_confirmatory.configuration import load_yaml, resolve_path, study_paths
from g3_confirmatory.eligibility import build_formation_outcome_map
from g3_confirmatory.hashing import sha256_file


def _panel_info(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    index = pd.to_datetime(frame.index)
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "date_start": str(index.min().date()) if len(index) else None,
        "date_end": str(index.max().date()) if len(index) else None,
        "nonmissing_cells": int(frame.notna().sum().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    root = paths["input_dir"]
    raw = cfg.get("raw_data", {})
    calendar_cfg = cfg.get("calendar", {})
    timing = build_formation_outcome_map(
        outcome_start=calendar_cfg.get("outcome_start", "2019-02-28"),
        outcome_end=calendar_cfg.get("outcome_end", "2025-12-31"),
    )
    first_formation = pd.Timestamp(timing["formation_date"].min())
    last_formation = pd.Timestamp(timing["formation_date"].max())
    experiment_cfg = cfg.get("experiment", {}) or {}
    information_lag = int(experiment_cfg.get("information_lag_months", 1))
    training_months = int(experiment_cfg.get("training_months", 36))
    first_training_end = first_formation - pd.offsets.MonthEnd(information_lag)
    first_training_start = pd.date_range(
        end=first_training_end, periods=training_months, freq="ME"
    ).min()
    prior_price_required = first_training_start - pd.offsets.MonthEnd(1)

    adjusted = resolve_path(raw.get("adjusted_close_csv") or root / "adjusted_close_prices.csv")
    prepared_eligibility = paths["eligibility_path"]
    meta_path = resolve_path(raw.get("meta_csv") or root / "Meta.csv")
    supplied_factors = resolve_path(
        raw.get("ff5_plus_momentum_csv")
        or root / "ff5_plus_momentum_monthly_2016_2025.csv"
    )
    downloaded_factors = paths.get(
        "factor_cache_dir", paths["output_root"].parent / "data" / "factor_cache"
    ) / "ff5_plus_momentum_monthly_2015_2025.csv"

    reports: dict[str, object] = {
        "requested_calendar": {
            "n_outcomes": int(len(timing)),
            "outcome_start": str(pd.Timestamp(timing["outcome_date"].min()).date()),
            "outcome_end": str(pd.Timestamp(timing["outcome_date"].max()).date()),
            "formation_start": str(first_formation.date()),
            "formation_end": str(last_formation.date()),
        },
        "minimum_raw_history": {
            "configured_information_lag_months": information_lag,
            "configured_training_months": training_months,
            "eligibility_monthly_activity_start_for_36_lagged_months": str(
                (first_formation - pd.offsets.MonthEnd(36)).date()
            ),
            "configured_first_training_return": str(first_training_start.date()),
            "configured_prior_price_required": str(prior_price_required.date()),
            "lag0_first_training_return": "2016-01-31",
            "lag0_prior_price_required": "2015-12-31",
            "lag1_first_training_return": "2015-12-31",
            "lag1_prior_price_required": "2015-11-30",
            "explanation": (
                "A monthly return needs the previous month-end price. The exact "
                "required start depends on whether model estimation uses the "
                "formation-month return (lag 0) or stops one month earlier (lag 1)."
            ),
        },
        "panels": {
            "adjusted_close": _panel_info(adjusted),
            "prepared_eligibility": _panel_info(prepared_eligibility),
            "supplied_factors": _panel_info(supplied_factors),
            "downloaded_factors": _panel_info(downloaded_factors),
        },
        "raw_notebook_inputs": {},
        "blocking_issues": [],
        "warnings": [],
    }

    date_panel_keys = {
        "volume_csv",
        "open_csv",
        "adjusted_close_csv",
        "ff5_plus_momentum_csv",
    }
    for key in (
        "volume_csv", "open_csv", "adjusted_close_csv",
        "ff5_plus_momentum_csv", "meta_csv", "universe_tickers_file"
    ):
        if key in raw and raw[key]:
            path = resolve_path(raw[key])
            if key in date_panel_keys:
                info = _panel_info(path)
            elif path.exists() and path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                info = {
                    "path": str(path),
                    "exists": True,
                    "sha256": sha256_file(path),
                    "rows": int(frame.shape[0]),
                    "columns": int(frame.shape[1]),
                }
            else:
                info = {
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                }
            reports["raw_notebook_inputs"][key] = info
        else:
            reports["raw_notebook_inputs"][key] = {"configured": False}

    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        raw_col = "Investment Object"
        ticker_col = "Ticker"
        if {raw_col, ticker_col}.issubset(meta.columns):
            series = (
                meta.dropna(subset=[ticker_col])
                .drop_duplicates(ticker_col)
                .set_index(ticker_col)[raw_col]
            )
            classes = normalize_asset_classes(series, unknown_policy="error")
            reports["asset_classes"] = {
                "meta_rows": int(len(meta)),
                "unique_tickers": int(series.index.nunique()),
                "counts": classes.value_counts(dropna=False).to_dict(),
            }
    else:
        reports["blocking_issues"].append(f"Missing metadata file: {meta_path}")

    adjusted_info = reports["panels"]["adjusted_close"]
    if not adjusted_info.get("exists"):
        reports["blocking_issues"].append(f"Missing adjusted-close panel: {adjusted}")
    else:
        start = pd.Timestamp(adjusted_info["date_start"])
        if start > prior_price_required:
            reports["blocking_issues"].append(
                "The adjusted-close panel starts after the prior price required by "
                f"the configured lag-{information_lag}, {training_months}-month "
                f"first training window ({prior_price_required.date()})."
            )

    selected_factor_info = (
        reports["panels"]["supplied_factors"]
        if reports["panels"]["supplied_factors"].get("exists")
        else reports["panels"]["downloaded_factors"]
    )
    if not selected_factor_info.get("exists"):
        reports["blocking_issues"].append(
            "No local Fama-French factor cache is available and network access "
            "cannot be assumed for a frozen rerun."
        )
    else:
        factor_start = pd.Timestamp(selected_factor_info["date_start"])
        factor_end = pd.Timestamp(selected_factor_info["date_end"])
        if factor_start > first_training_start or factor_end < last_formation:
            reports["blocking_issues"].append(
                "The available factor cache does not cover the configured training/"
                f"formation calendar: {factor_start.date()}--{factor_end.date()}, "
                f"required from {first_training_start.date()} through at least "
                f"{last_formation.date()}."
            )

    if not reports["raw_notebook_inputs"].get("volume_csv", {}).get("exists", False):
        reports["blocking_issues"].append(
            "Volume.csv is required to reproduce the notebook history/liquidity and "
            "tradability rules; it was not present in the supplied code archive."
        )
    if not reports["raw_notebook_inputs"].get("adjusted_close_csv", {}).get("exists", False):
        reports["blocking_issues"].append(
            "The raw adjusted-close panel configured for eligibility generation is missing."
        )
    delisting_path = paths.get("delisting_events_csv")
    if (cfg.get("eligibility", {}) or {}).get("delisting_mode", "current_only") == "point_in_time":
        if delisting_path is None or not delisting_path.exists():
            reports["blocking_issues"].append(
                "Point-in-time delisting treatment requires an author-verified "
                "delisting_events.csv with announcement and effective dates."
            )

    universe_info = reports["raw_notebook_inputs"].get("universe_tickers_file", {})
    if not universe_info.get("exists", False):
        reports["blocking_issues"].append(
            "The exact frozen 470-ticker point-in-time universe file is not supplied."
        )

    output = paths["output_root"] / "archive_audit"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "archive_support_audit.json", reports)
    flat = []
    for category in ("blocking_issues", "warnings"):
        for message in reports[category]:
            flat.append({"severity": category, "message": message})
    pd.DataFrame(flat).to_csv(output / "archive_support_issues.csv", index=False)
    print(f"Archive audit written to {output}")
    if reports["blocking_issues"]:
        print("Blocking issues:")
        for issue in reports["blocking_issues"]:
            print(" -", issue)


if __name__ == "__main__":
    main()
