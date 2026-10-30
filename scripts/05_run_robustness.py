#!/usr/bin/env python3
"""Run the one-change-at-a-time robustness grid.

Shared inputs reuse one worker pool and raw-arm cache. Eligibility variants
run separately because their data fingerprints differ.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from _common import (
    ROOT,
    add_execution_arguments,
    execution_kwargs,
    load_context,
    select_seeds,
    write_json,
)
from g3_confirmatory.data_bundle import DataBundle
from g3_confirmatory.configuration import (
    resolve_path, load_yaml, study_paths, experiment_config_from_mapping, configured_seeds,
)
from g3_confirmatory.realized_returns import (
    RealizedReturnConfig, config_from_mapping, read_daily_adjusted_close,
    resolve_realized_returns,
)
from g3_confirmatory.hashing import sha256_dataframe, sha256_file
from g3_confirmatory.robustness import build_robustness_specs, select_specs
from g3_confirmatory.runner import ExperimentJob, preflight_existing_jobs, run_experiment_jobs
from g3_confirmatory.run_checks import merge_trial_manifest

REALIZED_RETURN_VARIANTS = {
    "missing_return_last_trade_to_cash": RealizedReturnConfig(
        policy="last_trade_to_cash"
    ),
    "missing_return_haircut_30pct": RealizedReturnConfig(
        policy="fixed_haircut_to_cash", fixed_haircut=0.30
    ),
    # adverse sensitivity; 55% is not an assumed ETF recovery rate
    "missing_return_stock_haircut_55pct": RealizedReturnConfig(
        policy="fixed_haircut_to_cash", fixed_haircut=0.55
    ),
    "missing_return_total_loss": RealizedReturnConfig(policy="total_loss"),
}

ELIGIBILITY_VARIANTS = {
    "eligibility_current_only": "eligibility_current_only/eligibility.csv",
    "eligibility_oracle_next_month_activity": "eligibility_oracle_next_month_activity/eligibility.csv",
    "eligibility_oracle_future_activity": "eligibility_oracle_future_activity/eligibility.csv",
    "eligibility_notebook_raw_volume": "eligibility_notebook_raw_volume/eligibility.csv",
    "eligibility_history_24_of_36": "eligibility_history_24_of_36/eligibility.csv",
}


def _read_eligibility(path: Path, columns: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index).to_period("M").to_timestamp("M")
    frame = frame.replace(
        {"True": True, "False": False, "1": True, "0": False, 1: True, 0: False}
    ).astype("boolean").fillna(False).astype(bool)
    frame.columns = frame.columns.astype(str)
    return frame.reindex(columns=columns, fill_value=False).sort_index()


def _with_return_accounting(
    data: DataBundle,
    *,
    eligibility: pd.DataFrame,
    daily_adjusted: pd.DataFrame,
    return_config: RealizedReturnConfig,
    metadata_updates: dict[str, object] | None = None,
) -> DataBundle:
    realized_simple, realized_excess, audit, rr_meta = resolve_realized_returns(
        monthly_simple_returns=data.monthly_simple_returns,
        risk_free=data.risk_free,
        eligibility=eligibility,
        calendar=data.calendar,
        daily_adjusted_close=daily_adjusted,
        config=return_config,
    )
    metadata = dict(data.source_metadata)
    metadata.update({} if metadata_updates is None else metadata_updates)
    metadata["missing_outcome_return_policy"] = rr_meta
    metadata["missing_outcome_return_audit"] = audit.to_dict(orient="records")
    metadata["panel_hashes"] = dict(metadata.get("panel_hashes", {}))
    metadata["panel_hashes"]["realized_simple_returns"] = sha256_dataframe(
        realized_simple
    )
    metadata["panel_hashes"]["realized_excess_returns"] = sha256_dataframe(
        realized_excess
    )
    return replace(
        data,
        eligibility=eligibility,
        realized_simple_returns=realized_simple,
        realized_excess_returns=realized_excess,
        source_metadata=metadata,
    )


def _with_eligibility(
    data: DataBundle,
    path: Path,
    *,
    daily_adjusted: pd.DataFrame,
    return_config: RealizedReturnConfig,
) -> DataBundle:
    eligibility = _read_eligibility(path, data.eligibility.columns)
    missing = data.formation_dates.difference(eligibility.index)
    if len(missing):
        raise ValueError(
            f"Eligibility variant {path} misses formation dates: {missing.tolist()[:8]}"
        )
    return _with_return_accounting(
        data,
        eligibility=eligibility,
        daily_adjusted=daily_adjusted,
        return_config=return_config,
        metadata_updates={
            "eligibility_variant": str(path),
            "eligibility_variant_sha256": sha256_file(path),
            "panel_hashes": {
                **dict(data.source_metadata.get("panel_hashes", {})),
                "eligibility": sha256_dataframe(eligibility.astype(float)),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--families",
        help=(
            "Comma-separated robustness families. Default: all. Examples: "
            "transaction_cost,scenario_count,cvar_alpha"
        ),
    )
    parser.add_argument("--list", action="store_true", help="Print the frozen grid and exit.")
    parser.add_argument(
        "--skip-missing-eligibility",
        action="store_true",
        help="Skip missing eligibility variants instead of failing.",
    )
    add_execution_arguments(parser)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    base = experiment_config_from_mapping(cfg.get("experiment"))
    base.validate()
    all_seeds = configured_seeds(cfg)
    seeds = select_seeds(all_seeds, args.seed)
    specs = build_robustness_specs(base)
    families = (
        [item.strip() for item in args.families.split(",") if item.strip()]
        if args.families
        else None
    )
    custom_families = {"eligibility_rule", "missing_outcome_return"}
    if families is not None:
        built_in = {spec.family for spec in specs}
        unknown = set(families).difference(built_in | custom_families)
        if unknown:
            raise ValueError(f"Unknown robustness families: {sorted(unknown)}")
        selected_built_in = [family for family in families if family in built_in]
        specs = select_specs(specs, selected_built_in) if selected_built_in else []

    eligibility_root = paths["eligibility_path"].parent
    eligibility_records = []
    for experiment_id, relative in ELIGIBILITY_VARIANTS.items():
        path = eligibility_root / relative
        variant_config = (
            replace(base, min_observations=24, max_missing_fraction=1.0 / 3.0)
            if experiment_id == "eligibility_history_24_of_36"
            else base
        )
        eligibility_records.append(
            {
                "experiment_id": experiment_id,
                "family": "eligibility_rule",
                "purpose": (
                    "Point-in-time/delisting/history construction sensitivity. "
                    "Oracle variants are diagnostic only and cannot support a primary claim."
                ),
                "eligibility_path": str(path),
                "exists": path.exists(),
                "experiment_config": asdict(variant_config),
            }
        )

    realized_return_records = [
        {
            "experiment_id": experiment_id,
            "family": "missing_outcome_return",
            "purpose": (
                "No-look-ahead accounting sensitivity for an already-held asset "
                "whose ordinary outcome-month adjusted-price return is missing."
            ),
            "return_config": asdict(return_config),
            "exists": True,
            "experiment_config": asdict(base),
        }
        for experiment_id, return_config in REALIZED_RETURN_VARIANTS.items()
    ]

    include_eligibility = families is None or "eligibility_rule" in set(families)
    include_missing_returns = (
        families is None or "missing_outcome_return" in set(families)
    )
    selected_eligibility_records = eligibility_records if include_eligibility else []
    selected_return_records = realized_return_records if include_missing_returns else []
    manifest_records = [
        {
            "experiment_id": spec.experiment_id,
            "family": spec.family,
            "purpose": spec.purpose,
            "eligibility_path": str(paths["eligibility_path"]),
            "exists": True,
            "experiment_config": asdict(spec.config),
        }
        for spec in specs
    ] + selected_eligibility_records + selected_return_records

    if args.list:
        for record in manifest_records:
            config = record["experiment_config"]
            n = config["n_scenarios"]
            alpha = config["optimizer"]["cvar_alpha"]
            print(record["family"], record["experiment_id"],
                  f"scenarios={n}", f"alpha={alpha}", f"tail={n * (1 - alpha):g}")
        return

    root = paths["output_root"] / "robustness"
    preflight_existing_jobs(
        [ExperimentJob(r["experiment_id"], experiment_config_from_mapping(r["experiment_config"]),
                       root / r["experiment_id"]) for r in manifest_records],
        seeds, overwrite=args.overwrite,
    )
    cfg, paths, data, loaded_base, _, _ = load_context(args.config)
    if asdict(loaded_base) != asdict(base):
        raise ValueError("Configuration changed while loading the data.")
    execution = execution_kwargs(args, cfg, paths)
    preflight_existing_jobs(
        [ExperimentJob(spec.experiment_id, spec.config, root / spec.experiment_id) for spec in specs],
        seeds, overwrite=args.overwrite, data=data,
    )
    root.mkdir(parents=True, exist_ok=True)
    merge_trial_manifest(
        root / "frozen_trial_manifest.json",
        {
            "frozen_before_execution": True,
            "base_config": asdict(base),
            "seeds": seeds,
            "n_configurations": len(manifest_records),
            "configurations": manifest_records,
            "selection_warning": (
                "All completed, failed, and discarded configurations remain in this "
                "manifest. Do not delete unfavorable trials before DSR/SPA/stepdown analysis."
            ),
        },
    )

    jobs: list[ExperimentJob] = []
    for spec in specs:
        out = root / spec.experiment_id
        write_json(
            out / "configuration.json",
            {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "purpose": spec.purpose,
                "config": asdict(spec.config),
            },
        )
        jobs.append(
            ExperimentJob(
                job_id=spec.experiment_id,
                config=spec.config,
                output_dir=out,
                metadata={"family": spec.family, "purpose": spec.purpose},
            )
        )
    if jobs:
        run_experiment_jobs(
            data,
            jobs=jobs,
            seeds=seeds,
            overwrite=args.overwrite,
            **execution,
        )
        for spec in specs:
            print(f"completed robustness configuration: {spec.experiment_id}")

    if not include_missing_returns and not include_eligibility:
        return

    raw_data_cfg = dict(cfg.get("raw_data", {}) or {})
    adjusted_raw = raw_data_cfg.get("adjusted_close_csv")
    if not adjusted_raw:
        raise ValueError("raw_data.adjusted_close_csv is required for return accounting.")
    daily_adjusted = read_daily_adjusted_close(resolve_path(adjusted_raw))
    primary_return_config = config_from_mapping(cfg.get("realized_returns"))

    # reuse predictive scenarios across return-accounting rules
    if include_missing_returns:
        for record in selected_return_records:
            return_config = REALIZED_RETURN_VARIANTS[str(record["experiment_id"])]
            variant_data = _with_return_accounting(
                data,
                eligibility=data.eligibility,
                daily_adjusted=daily_adjusted,
                return_config=return_config,
                metadata_updates={
                    "realized_return_variant": str(record["experiment_id"]),
                },
            )
            out = root / str(record["experiment_id"])
            write_json(out / "configuration.json", record)
            run_experiment_jobs(
                variant_data,
                jobs=[
                    ExperimentJob(
                        job_id=str(record["experiment_id"]),
                        config=base,
                        output_dir=out,
                        metadata={"family": "missing_outcome_return"},
                    )
                ],
                seeds=seeds,
                overwrite=args.overwrite,
                **execution,
            )
            print(f"completed return-accounting robustness: {record['experiment_id']}")

    for record in selected_eligibility_records:
        path = Path(record["eligibility_path"])
        if not path.exists():
            message = (
                f"Missing {path}. Re-run scripts\\00_build_from_five_files.py --overwrite first."
            )
            if args.skip_missing_eligibility:
                print("skipped:", message)
                continue
            raise FileNotFoundError(message)
        variant_data = _with_eligibility(
            data,
            path,
            daily_adjusted=daily_adjusted,
            return_config=primary_return_config,
        )
        variant_config = (
            replace(base, min_observations=24, max_missing_fraction=1.0 / 3.0)
            if record["experiment_id"] == "eligibility_history_24_of_36"
            else base
        )
        out = root / str(record["experiment_id"])
        write_json(out / "configuration.json", record)
        run_experiment_jobs(
            variant_data,
            jobs=[
                ExperimentJob(
                    job_id=str(record["experiment_id"]),
                    config=variant_config,
                    output_dir=out,
                    metadata={"family": "eligibility_rule"},
                )
            ],
            seeds=seeds,
            overwrite=args.overwrite,
            **execution,
        )
        print(f"completed eligibility robustness: {record['experiment_id']}")


if __name__ == "__main__":
    main()
