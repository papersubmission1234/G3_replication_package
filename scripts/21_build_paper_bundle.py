#!/usr/bin/env python3
"""Collect tables, figures, summaries, manifests, and input audits.

The archive omits large weights, predictive scenarios, and caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from _common import ROOT
from g3_confirmatory.configuration import load_yaml, study_paths
from g3_confirmatory.reporting import read_seed_tree


INCLUDE_OUTPUT_DIRS = (
    "paper_tables",
    "paper_figures",
    "inference",
    "risk_diagnostics",
    "validation",
    "output_validation",
)

INCLUDE_DECOMPOSITION_SUMMARIES = (
    Path("decompositions/full_risk_map/summary"),
    Path("decompositions/decision_shapley/summary"),
    Path("decompositions/summary"),
)


PANEL_SPECS = (
    ("all_returns.csv.gz", "returns.csv", ("formation_date", "outcome_date")),
    ("all_posterior.csv.gz", "posterior.csv", ("formation_date", "outcome_date")),
    ("all_arm_losses.csv.gz", "arm_losses.csv", ("formation_date", "outcome_date")),
    (
        "all_optimizer_diagnostics.csv.gz",
        "optimizer_diagnostics.csv",
        ("formation_date", "outcome_date"),
    ),
    ("all_arm_diagnostics.csv.gz", "arm_diagnostics.csv", ("formation_date",)),
    ("all_scenario_hashes.csv.gz", "scenario_hashes.csv", ("formation_date",)),
)

INCLUDE_DATA_FILES = (
    "study_build_manifest.json",
    "data_dictionary.csv",
    "asset_classes.csv",
    "universe_derived.csv",
    "eligibility.csv",
    "delisting_events_oracle_diagnostic.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path, *, skip_large_daily: bool = False) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if skip_large_daily and path.name == "portfolio_daily_returns.csv.gz":
            continue
        _copy_file(path, destination / path.relative_to(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument(
        "--destination",
        help="Destination directory. Default: <output_root>/paper_bundle.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    output_root = paths["output_root"]
    work_root = output_root.parent
    destination = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else output_root / "paper_bundle"
    )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    _copy_file(Path(args.config).resolve(), destination / "config" / Path(args.config).name)
    for directory in INCLUDE_OUTPUT_DIRS:
        _copy_tree(
            output_root / directory,
            destination / "outputs" / directory,
            skip_large_daily=(directory == "risk_diagnostics"),
        )
    for rel in INCLUDE_DECOMPOSITION_SUMMARIES:
        _copy_tree(output_root / rel, destination / "outputs" / rel)

    panel_root = destination / "paper_data"
    panel_root.mkdir(parents=True, exist_ok=True)
    panel_status = []
    for output_name, seed_filename, parse_dates in PANEL_SPECS:
        frame = read_seed_tree(
            output_root, seed_filename, parse_dates=parse_dates
        )
        target = panel_root / output_name
        if frame.empty:
            panel_status.append(
                {
                    "panel": output_name,
                    "source_filename": seed_filename,
                    "available": False,
                    "n_rows": 0,
                    "n_columns": 0,
                }
            )
            continue
        frame.to_csv(target, index=False, compression="gzip")
        panel_status.append(
            {
                "panel": output_name,
                "source_filename": seed_filename,
                "available": True,
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
            }
        )
    pd.DataFrame(panel_status).to_csv(
        panel_root / "paper_data_catalog.csv", index=False
    )

    for pattern in (
        "**/frozen_trial_manifest.json",
        "**/preregistered_core_declaration.json",
        "**/ablation_declaration.json",
        "**/configuration.json",
        "**/experiment_manifest.json",
        "**/failures.json",
    ):
        for path in sorted(output_root.glob(pattern)):
            if not path.is_file() or "paper_bundle" in path.parts:
                continue
            _copy_file(path, destination / "outputs" / path.relative_to(output_root))

    for path in sorted(output_root.rglob("seed_*/run_manifest.json")):
        if "paper_bundle" in path.parts:
            continue
        _copy_file(path, destination / "outputs" / path.relative_to(output_root))

    data_root = work_root / "data"
    for name in INCLUDE_DATA_FILES:
        path = data_root / name
        if path.exists():
            _copy_file(path, destination / "data" / name)
    for subdir in (
        "eligibility_primary",
        "eligibility_current_only",
        "eligibility_oracle_next_month_activity",
        "eligibility_oracle_future_activity",
        "eligibility_notebook_raw_volume",
        "eligibility_history_24_of_36",
    ):
        source = data_root / subdir
        if source.exists():
            for name in (
                "eligibility_audit_by_date.csv",
                "eligibility_audit_by_ticker.csv",
                "eligibility_metadata.json",
                "formation_outcome_map.csv",
            ):
                path = source / name
                if path.exists():
                    _copy_file(path, destination / "data" / subdir / name)

    records = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        records.append(
            {
                "relative_path": path.relative_to(destination).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(destination / "bundle_manifest.csv", index=False)
    status = {
        "config": str(Path(args.config).resolve()),
        "output_root": str(output_root),
        "destination": str(destination),
        "n_files": int(len(manifest)),
        "total_bytes": int(manifest["size_bytes"].sum()) if not manifest.empty else 0,
        "exclusions": [
            "weights.csv.gz",
            "raw predictive-arm caches",
            "exact seed-run caches",
            "portfolio_daily_returns.csv.gz (daily summary remains included)",
        ],
    }
    (destination / "README_BUNDLE.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    archive = destination.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(destination.parent))
    print(json.dumps({**status, "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
