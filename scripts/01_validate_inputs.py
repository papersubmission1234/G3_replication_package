#!/usr/bin/env python3
"""Validate the frozen universe, no-look-ahead timing, and study calendar."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from _common import ROOT, load_context, write_json
from g3_confirmatory.data_bundle import validate_data_bundle

ALL_CLASSES = (
    "equity",
    "bond",
    "mixed",
    "real_estate",
    "commodities",
    "crypto",
    "metals",
    "cash_fx",
)


def _eligible_class_counts(data) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for formation in data.formation_dates:
        flags = data.eligibility.loc[formation].astype(bool)
        classes = data.asset_classes.reindex(flags[flags].index)
        counts = classes.value_counts()
        row: dict[str, object] = {
            "formation_date": formation,
            "n_eligible": int(flags.sum()),
        }
        for class_name in ALL_CLASSES:
            row[f"n_{class_name}"] = int(counts.get(class_name, 0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("formation_date")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Write the audit but return success even if confirmatory requirements fail.",
    )
    args = parser.parse_args()

    output_dir: Path | None = None
    report: dict[str, object] = {
        "config_path": str(Path(args.config).resolve()),
        "status": "started",
        "errors": [],
        "warnings": [],
    }
    try:
        cfg, paths, data, experiment, _, seeds = load_context(args.config)
        output_dir = paths["output_root"] / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        audit = validate_data_bundle(
            data,
            training_months=experiment.training_months,
            information_lag_months=experiment.information_lag_months,
            require_complete_training_history=(
                experiment.min_observations == experiment.training_months
                and experiment.max_missing_fraction == 0.0
            ),
        )
        audit.to_csv(output_dir / "data_timing_validation.csv")
        class_by_date = _eligible_class_counts(data)
        class_by_date.to_csv(output_dir / "eligible_asset_classes_by_formation.csv")
        missing_return_audit = pd.DataFrame(
            data.source_metadata.get("missing_outcome_return_audit", [])
        )
        missing_return_audit.to_csv(
            output_dir / "missing_outcome_return_audit.csv", index=False
        )

        expected_outcomes = int(cfg.get("calendar", {}).get("expected_outcomes", 83))
        raw_cfg = cfg.get("raw_data", {}) or {}
        expected_universe = raw_cfg.get("expected_universe_size")
        errors: list[str] = []
        warnings: list[str] = []

        if len(data.calendar) != expected_outcomes:
            errors.append(
                f"Calendar has {len(data.calendar)} outcomes; expected {expected_outcomes}."
            )
        calendar_cfg = cfg.get("calendar", {}) or {}
        expected_outcome_start = pd.Timestamp(
            calendar_cfg.get("outcome_start", "2019-02-28")
        ).to_period("M").to_timestamp("M")
        expected_outcome_end = pd.Timestamp(
            calendar_cfg.get("outcome_end", "2025-12-31")
        ).to_period("M").to_timestamp("M")
        expected_formation_start = expected_outcome_start - pd.offsets.MonthEnd(1)
        expected_formation_end = expected_outcome_end - pd.offsets.MonthEnd(1)

        if data.outcome_dates.min() != expected_outcome_start:
            errors.append(
                f"First outcome is {data.outcome_dates.min().date()}, expected "
                f"{expected_outcome_start.date()}."
            )
        if data.outcome_dates.max() != expected_outcome_end:
            errors.append(
                f"Last outcome is {data.outcome_dates.max().date()}, expected "
                f"{expected_outcome_end.date()}."
            )
        if data.formation_dates.min() != expected_formation_start:
            errors.append(
                f"The first formation must be one month before the first outcome: "
                f"expected {expected_formation_start.date()}, found "
                f"{data.formation_dates.min().date()}."
            )
        if data.formation_dates.max() != expected_formation_end:
            errors.append(
                f"The final formation must be one month before the final outcome: "
                f"expected {expected_formation_end.date()}, found "
                f"{data.formation_dates.max().date()}."
            )
        if expected_universe is not None and data.source_metadata["n_tickers"] != int(expected_universe):
            errors.append(
                "The frozen input universe does not match the declared size: "
                f"{data.source_metadata['n_tickers']} versus {expected_universe}."
            )

        observed_classes = set(data.asset_classes.dropna().astype(str).unique())
        missing_classes = set(ALL_CLASSES).difference(observed_classes)
        if missing_classes:
            warnings.append(
                "No ticker is mapped to these supported classes in the frozen universe: "
                + ", ".join(sorted(missing_classes))
            )
        ever_eligible_classes = set()
        for formation in data.formation_dates:
            flags = data.eligibility.loc[formation].astype(bool)
            ever_eligible_classes.update(
                data.asset_classes.reindex(flags[flags].index).dropna().astype(str)
            )
        missing_eligible_classes = set(ALL_CLASSES).difference(ever_eligible_classes)
        if missing_eligible_classes:
            warnings.append(
                "These supported classes are never eligible in the requested window: "
                + ", ".join(sorted(missing_eligible_classes))
            )

        eligibility_cfg = cfg.get("eligibility", {}) or {}
        delisting_mode = str(eligibility_cfg.get("delisting_mode", "current_only"))
        if delisting_mode in {
            "oracle_future_activity",
            "oracle_next_month_activity",
        }:
            errors.append(
                "Primary eligibility uses future activity and therefore has look-ahead. "
                "Use delisting_mode=current_only unless a genuine point-in-time "
                "announcement/effective-date file is available."
            )
        if delisting_mode == "point_in_time" and not paths.get("delisting_events_csv"):
            warnings.append(
                "point_in_time mode has no event table; it is operationally equivalent "
                "to current_only for delisting exclusions."
            )

        rr_meta = data.source_metadata.get("missing_outcome_return_policy", {}) or {}
        if int(rr_meta.get("n_unresolved", 0)) > 0:
            errors.append(
                "The realized-return policy leaves eligible holding-period returns "
                f"unresolved: {rr_meta.get('n_unresolved')}."
            )

        effective_tail = (
            (1.0 - float(experiment.optimizer.cvar_alpha))
            * int(experiment.n_scenarios)
        )
        if effective_tail < 20.0:
            warnings.append(
                "The empirical CVaR tail contains fewer than 20 effective scenario "
                f"observations ({effective_tail:.2f}). Increase n_scenarios for this "
                "alpha in robustness runs."
            )

        if experiment.included_asset_classes is not None:
            errors.append(
                "Confirmatory experiment restricts included_asset_classes. Leave this null "
                "to use the complete multi-asset universe; equity/bond-only belongs in robustness."
            )
        if experiment.information_lag_months == 0:
            if not bool(eligibility_cfg.get("formation_after_month_end_close", False)):
                errors.append(
                    "information_lag_months=0 requires formation_after_month_end_close=true "
                    "so formation-month data are observable before the action."
                )
            else:
                warnings.append(
                    "Lag-zero estimation is being used with explicit after-close formation. "
                    "This is the executable 83-outcome convention using a clean February-2016 "
                    "through January-2019 first training window; it must be "
                    "stated in the manuscript."
                )
        elif experiment.information_lag_months != 1:
            warnings.append(
                f"Nonstandard information lag: {experiment.information_lag_months} months."
            )

        report.update(
            {
                "status": "failed" if errors else "passed",
                "errors": errors,
                "warnings": warnings,
                "calendar": {
                    "n_outcomes": int(len(data.calendar)),
                    "formation_start": str(data.formation_dates.min().date()),
                    "formation_end": str(data.formation_dates.max().date()),
                    "outcome_start": str(data.outcome_dates.min().date()),
                    "outcome_end": str(data.outcome_dates.max().date()),
                },
                "data": data.source_metadata,
                "experiment": {
                    "training_months": experiment.training_months,
                    "information_lag_months": experiment.information_lag_months,
                    "min_observations": experiment.min_observations,
                    "max_missing_fraction": experiment.max_missing_fraction,
                    "included_asset_classes": experiment.included_asset_classes,
                    "n_scenarios": experiment.n_scenarios,
                    "n_seeds": len(seeds),
                },
                "asset_classes_supported": list(ALL_CLASSES),
                "asset_classes_observed": sorted(observed_classes),
                "asset_classes_ever_eligible": sorted(ever_eligible_classes),
                "eligibility_semantics": (
                    "eligibility[t]=1 means selectable at formation month-end t and "
                    "held for the return ending at t+1 month-end"
                ),
                "delisting_mode": delisting_mode,
                "missing_outcome_return_policy": rr_meta,
                "cvar_horizon": "one_month_predictive_return",
                "cvar_effective_tail_scenarios": effective_tail,
            }
        )
        write_json(output_dir / "validation_report.json", report)
        if errors and not args.no_fail:
            raise SystemExit("Input validation failed; see validation_report.json")
        print(f"Validation {report['status']}: {output_dir}")
    except Exception as exc:
        report["status"] = "failed"
        report.setdefault("errors", []).append(repr(exc))
        if output_dir is None:
            output_dir = ROOT / "outputs" / "validation"
            output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "validation_report.json", report)
        if args.no_fail:
            print(f"Validation failed but --no-fail was set: {exc}", file=sys.stderr)
            return
        raise


if __name__ == "__main__":
    main()
