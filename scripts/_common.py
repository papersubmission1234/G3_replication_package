from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from g3_confirmatory.configuration import (
    configured_seeds,
    experiment_config_from_mapping,
    load_yaml,
    regime_config_from_mapping,
    resolve_path,
    study_paths,
)
from g3_confirmatory.runner import load_configured_data
from g3_confirmatory.realized_returns import config_from_mapping
from g3_confirmatory.serialization import write_json as _write_json


def load_context(config_path: str | Path):
    cfg = load_yaml(config_path)
    paths = study_paths(cfg)
    calendar = cfg.get("calendar", {})
    outcome_start = str(calendar.get("outcome_start", "2019-02-28"))
    outcome_end = str(calendar.get("outcome_end", "2025-12-31"))
    delisting = paths.get("delisting_events_csv")
    realized_cfg = config_from_mapping(cfg.get("realized_returns"))
    raw_data = dict(cfg.get("raw_data", {}) or {})
    for key in ("meta_csv", "macro_workbook", "adjusted_close_csv", "ff5_plus_momentum_csv"):
        if raw_data.get(key):
            raw_data[key] = resolve_path(raw_data[key])
    adjusted_path = raw_data.get("adjusted_close_csv")
    data = load_configured_data(
        input_dir=paths["input_dir"],
        eligibility_path=paths["eligibility_path"],
        outcome_start=outcome_start,
        outcome_end=outcome_end,
        delisting_events_path=delisting,
        realized_return_config=realized_cfg,
        daily_adjusted_close_path=adjusted_path,
        raw_data=raw_data,
        factor_cache_dir=paths.get(
            "factor_cache_dir", paths["output_root"].parent / "data" / "factor_cache"
        ),
    )
    experiment = experiment_config_from_mapping(cfg.get("experiment"))
    regime = regime_config_from_mapping(cfg.get("regime"))
    seeds = configured_seeds(cfg)
    return cfg, paths, data, experiment, regime, seeds


def select_seeds(default_seeds: list[int], seed: int | None) -> list[int]:
    return [int(seed)] if seed is not None else default_seeds


def read_seed_table(root: str | Path, filename: str) -> pd.DataFrame:
    paths = sorted(Path(root).glob(f"seed_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No seed_*/{filename} files found under {root}")
    parse_dates = [
        column for column in ("formation_date", "outcome_date")
        if column in pd.read_csv(paths[0], nrows=0).columns
    ]
    return pd.concat(
        [pd.read_csv(path, parse_dates=parse_dates) for path in paths],
        ignore_index=True,
    )


def write_json(path: str | Path, payload) -> None:
    _write_json(path, payload)


def add_execution_arguments(parser) -> None:
    """Add common multiprocessing/cache switches to an argparse parser."""

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Worker processes. 1 is sequential; 0 chooses a conservative "
            "automatic count (at most 4 unless the config overrides it)."
        ),
    )
    parser.add_argument(
        "--schedule",
        choices=("auto", "seed", "task"),
        default=None,
        help=(
            "Parallel scheduling. seed maximizes cache locality; task also "
            "parallelizes independent configurations for one-seed smoke tests."
        ),
    )
    parser.add_argument(
        "--mp-start-method",
        choices=("spawn", "fork", "forkserver"),
        default=None,
        help="Multiprocessing start method. Windows must use spawn.",
    )
    parser.add_argument(
        "--numeric-threads-per-worker",
        type=int,
        default=None,
        help="BLAS/OpenMP threads inside each worker; normally keep this at 1.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Raw predictive-arm cache directory. Default: <work>/cache/raw_arms.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("off", "read", "readwrite", "refresh"),
        default=None,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable predictive-arm caching for this command.",
    )
    parser.add_argument(
        "--result-cache-dir",
        help="Exact seed-run cache directory. Default: <work>/cache/seed_runs.",
    )
    parser.add_argument(
        "--result-cache-mode",
        choices=("off", "read", "readwrite", "refresh"),
        default=None,
    )
    parser.add_argument(
        "--no-result-cache",
        action="store_true",
        help="Disable exact seed-run reuse while retaining the raw-arm cache.",
    )
    parser.add_argument(
        "--cache-compression",
        type=int,
        default=None,
        help="joblib compression level for cache entries (0-9; default 1).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Finish independent tasks after a failure, write all failure records, "
            "then exit nonzero."
        ),
    )


def execution_kwargs(args, cfg: dict, paths: dict[str, Path]) -> dict:
    """Resolve CLI/config parallel settings for runner functions."""

    parallel = dict(cfg.get("parallel", {}) or {})
    workers = (
        int(args.workers)
        if getattr(args, "workers", None) is not None
        else int(parallel.get("workers", 1))
    )
    schedule = (
        str(args.schedule)
        if getattr(args, "schedule", None) is not None
        else str(parallel.get("schedule", "auto"))
    )
    start_method = (
        str(args.mp_start_method)
        if getattr(args, "mp_start_method", None) is not None
        else str(parallel.get("start_method", "spawn"))
    )
    numeric_threads = (
        int(args.numeric_threads_per_worker)
        if getattr(args, "numeric_threads_per_worker", None) is not None
        else int(parallel.get("numeric_threads_per_worker", 1))
    )
    compression = (
        int(args.cache_compression)
        if getattr(args, "cache_compression", None) is not None
        else int(parallel.get("cache_compression", 1))
    )
    if not 0 <= compression <= 9:
        raise ValueError("cache_compression must lie in [0, 9].")

    if getattr(args, "no_cache", False):
        cache_dir = None
        cache_mode = "off"
    else:
        raw_cache = getattr(args, "cache_dir", None) or parallel.get("cache_dir")
        cache_dir = (
            Path(raw_cache).expanduser().resolve()
            if raw_cache
            else (paths["output_root"].parent / "cache" / "raw_arms").resolve()
        )
        cache_mode = (
            str(args.cache_mode)
            if getattr(args, "cache_mode", None) is not None
            else str(parallel.get("cache_mode", "readwrite"))
        )

    if getattr(args, "no_result_cache", False):
        result_cache_dir = None
        result_cache_mode = "off"
    else:
        raw_result_cache = (
            getattr(args, "result_cache_dir", None)
            or parallel.get("result_cache_dir")
        )
        result_cache_dir = (
            Path(raw_result_cache).expanduser().resolve()
            if raw_result_cache
            else (paths["output_root"].parent / "cache" / "seed_runs").resolve()
        )
        result_cache_mode = (
            str(args.result_cache_mode)
            if getattr(args, "result_cache_mode", None) is not None
            else str(parallel.get("result_cache_mode", "readwrite"))
        )

    fail_fast = bool(parallel.get("fail_fast", True))
    if getattr(args, "keep_going", False):
        fail_fast = False
    return {
        "workers": workers,
        "schedule": schedule,
        "start_method": start_method,
        "numeric_threads_per_worker": numeric_threads,
        "cache_dir": cache_dir,
        "cache_mode": cache_mode,
        "result_cache_dir": result_cache_dir,
        "result_cache_mode": result_cache_mode,
        "cache_compression": compression,
        "fail_fast": fail_fast,
    }
