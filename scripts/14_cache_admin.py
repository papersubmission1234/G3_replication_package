#!/usr/bin/env python3
"""Inspect or clear the multiprocessing computation caches."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from _common import ROOT
from g3_confirmatory.configuration import load_yaml, study_paths


def _size(path: Path) -> tuple[int, int]:
    count = 0
    total = 0
    if not path.exists():
        return count, total
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return count, total


def _fmt_size(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument("--action", choices=("status", "clear"), default="status")
    parser.add_argument("--cache", choices=("all", "raw_arms", "seed_runs"), default="all")
    parser.add_argument("--yes", action="store_true", help="Required for destructive clear.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    parallel = dict(cfg.get("parallel", {}) or {})
    work_root = paths["output_root"].parent
    caches = {
        "raw_arms": Path(parallel.get("cache_dir", work_root / "cache" / "raw_arms")),
        "seed_runs": Path(
            parallel.get("result_cache_dir", work_root / "cache" / "seed_runs")
        ),
    }
    selected = caches if args.cache == "all" else {args.cache: caches[args.cache]}

    if args.action == "status":
        for name, path in selected.items():
            count, total = _size(path)
            locks = len(list(path.rglob("*.lock"))) if path.exists() else 0
            print(f"{name}: {path}")
            print(f"  files={count:,} size={_fmt_size(total)} locks={locks}")
        return

    if not args.yes:
        raise SystemExit("Refusing to clear caches without --yes.")
    for name, path in selected.items():
        if path.exists():
            shutil.rmtree(path)
            print(f"cleared {name}: {path}")
        else:
            print(f"already absent {name}: {path}")


if __name__ == "__main__":
    main()
