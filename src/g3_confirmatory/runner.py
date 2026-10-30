"""Run independent seeds or configuration grids in a shared process pool.

Raw predictive fits can be cached across scores, actions, priors, and risk maps.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any
import os
import json
import time
import traceback

import pandas as pd

from . import __version__
from .arm_cache import (
    ArmResultCache,
    arm_code_fingerprint,
    arm_data_fingerprint,
    computational_environment_fingerprint,
    data_bundle_fingerprint,
)
from .data_bundle import DataBundle, load_study_data, validate_data_bundle
from .realized_returns import RealizedReturnConfig
from .experiment import ExperimentConfig, run_seed, write_seed_result
from .hashing import combine_hashes, iter_tree_files, sha256_tree
from .serialization import write_json
from .seed_result_cache import SeedResultCache, seed_run_signature
from .run_checks import config_differences, named_setting_errors


_SEED_OUTPUT_FILES = {
    "returns.csv",
    "posterior.csv",
    "arm_losses.csv",
    "scenario_hashes.csv",
    "optimizer_diagnostics.csv",
    "arm_diagnostics.csv",
    "weights.csv.gz",
    "run_manifest.json",
}


def _clear_seed_run_files(seed_dir: Path) -> None:
    """Remove stale run outputs while preserving seed-specific input files."""

    if not seed_dir.exists():
        return
    for name in _SEED_OUTPUT_FILES:
        path = seed_dir / name
        path.unlink(missing_ok=True)

@dataclass
class ExperimentJob:
    """One experiment configuration sharing a common prepared data bundle."""

    job_id: str
    config: ExperimentConfig
    output_dir: str | Path
    formation_states: pd.Series | None = None
    fixed_state_weights: Mapping[str, Mapping[str, float]] | None = None
    fixed_state_weights_by_seed: Mapping[
        int, Mapping[str, Mapping[str, float]]
    ] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_output_dir(self) -> Path:
        return Path(self.output_dir).resolve()


_WORKER_DATA: DataBundle | None = None
_WORKER_CACHE: ArmResultCache | None = None
_WORKER_RESULT_CACHE: SeedResultCache | None = None
_WORKER_DATA_FINGERPRINT: str | None = None
_WORKER_RUN_CODE_FINGERPRINT: str | None = None
_WORKER_SOURCE_TREE_SHA256: str | None = None
_WORKER_SOURCE_TREE_FILE_COUNT: int | None = None


def _limit_numeric_threads(limit: int) -> None:
    """Prevent BLAS/OpenMP oversubscription inside each worker process."""

    limit = max(1, int(limit))
    value = str(limit)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = value
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=limit)
    except Exception:
        pass


def _worker_initializer(
    data: DataBundle,
    cache_dir: str | None,
    cache_mode: str,
    result_cache_dir: str | None,
    result_cache_mode: str,
    result_data_fingerprint: str,
    arm_data_fingerprint_value: str,
    code_fingerprint: str,
    run_code_fingerprint: str,
    source_tree_sha256: str,
    source_tree_file_count: int,
    cache_compression: int,
    numeric_threads: int,
) -> None:
    global _WORKER_DATA, _WORKER_CACHE, _WORKER_RESULT_CACHE
    global _WORKER_DATA_FINGERPRINT, _WORKER_RUN_CODE_FINGERPRINT
    global _WORKER_SOURCE_TREE_SHA256, _WORKER_SOURCE_TREE_FILE_COUNT
    _limit_numeric_threads(numeric_threads)
    _WORKER_DATA = data
    _WORKER_DATA_FINGERPRINT = result_data_fingerprint
    _WORKER_RUN_CODE_FINGERPRINT = run_code_fingerprint
    _WORKER_SOURCE_TREE_SHA256 = source_tree_sha256
    _WORKER_SOURCE_TREE_FILE_COUNT = int(source_tree_file_count)
    _WORKER_CACHE = ArmResultCache(
        cache_dir,
        mode=cache_mode,
        data_fingerprint=arm_data_fingerprint_value,
        code_fingerprint=code_fingerprint,
        compression=cache_compression,
    )
    _WORKER_RESULT_CACHE = SeedResultCache(
        result_cache_dir,
        mode=result_cache_mode,
    )


def _weights_for_seed(
    job: ExperimentJob,
    seed: int,
) -> Mapping[str, Mapping[str, float]] | None:
    if job.fixed_state_weights_by_seed is not None:
        try:
            return job.fixed_state_weights_by_seed[int(seed)]
        except KeyError as exc:
            raise KeyError(
                f"Job {job.job_id!r} has no fixed_state_weights for seed {seed}."
            ) from exc
    return job.fixed_state_weights


def _execute_job_seed(job: ExperimentJob, seed: int) -> dict[str, Any]:
    if _WORKER_DATA is None:
        raise RuntimeError("Experiment worker was not initialized with data.")
    output = job.normalized_output_dir() / f"seed_{int(seed)}"
    before = _WORKER_CACHE.snapshot() if _WORKER_CACHE is not None else None
    result_before = (
        _WORKER_RESULT_CACHE.snapshot()
        if _WORKER_RESULT_CACHE is not None
        else None
    )
    started = time.time()
    try:
        fixed_weights = _weights_for_seed(job, int(seed))

        def _run_and_write() -> None:
            result = run_seed(
                _WORKER_DATA,
                master_seed=int(seed),
                config=job.config,
                formation_states=job.formation_states,
                fixed_state_weights=fixed_weights,
                arm_cache=_WORKER_CACHE,
                source_tree_sha256=_WORKER_SOURCE_TREE_SHA256,
                source_tree_file_count=_WORKER_SOURCE_TREE_FILE_COUNT,
            )
            result.manifest["resume_signature"] = seed_run_signature(
                data_fingerprint=_WORKER_DATA_FINGERPRINT,
                run_code_fingerprint="resume-inputs-v1",
                master_seed=int(seed), config=job.config,
                formation_states=job.formation_states,
                fixed_state_weights=fixed_weights,
            )
            write_seed_result(result, output)

        result_cache_hit = False
        if (
            _WORKER_RESULT_CACHE is not None
            and _WORKER_DATA_FINGERPRINT is not None
            and _WORKER_RUN_CODE_FINGERPRINT is not None
        ):
            signature = seed_run_signature(
                data_fingerprint=_WORKER_DATA_FINGERPRINT,
                run_code_fingerprint=_WORKER_RUN_CODE_FINGERPRINT,
                master_seed=int(seed),
                config=job.config,
                formation_states=job.formation_states,
                fixed_state_weights=fixed_weights,
            )
            result_cache_hit = _WORKER_RESULT_CACHE.get_or_run(
                signature,
                output,
                _run_and_write,
            )
        else:
            _run_and_write()

        cache_delta = (
            _WORKER_CACHE.stats.delta(before)
            if _WORKER_CACHE is not None and before is not None
            else {}
        )
        result_cache_delta = (
            SeedResultCache.delta(_WORKER_RESULT_CACHE.snapshot(), result_before)
            if _WORKER_RESULT_CACHE is not None and result_before is not None
            else {}
        )
        return {
            "job_id": job.job_id,
            "seed": int(seed),
            "status": "completed",
            "runtime_seconds": time.time() - started,
            "cache": cache_delta,
            "result_cache": result_cache_delta,
            "result_cache_hit": bool(result_cache_hit),
        }
    except Exception as exc:
        cache_delta = (
            _WORKER_CACHE.stats.delta(before)
            if _WORKER_CACHE is not None and before is not None
            else {}
        )
        result_cache_delta = (
            SeedResultCache.delta(_WORKER_RESULT_CACHE.snapshot(), result_before)
            if _WORKER_RESULT_CACHE is not None and result_before is not None
            else {}
        )
        return {
            "job_id": job.job_id,
            "seed": int(seed),
            "status": "failed",
            "runtime_seconds": time.time() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "cache": cache_delta,
            "result_cache": result_cache_delta,
            "result_cache_hit": False,
        }

def _execute_seed_batch(
    seed: int,
    jobs: Sequence[ExperimentJob],
    stop_on_error: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in jobs:
        record = _execute_job_seed(job, int(seed))
        records.append(record)
        if stop_on_error and record["status"] == "failed":
            break
    return records


def load_configured_data(
    *,
    input_dir: str | Path,
    eligibility_path: str | Path,
    outcome_start: str,
    outcome_end: str,
    delisting_events_path: str | Path | None = None,
    realized_return_config: RealizedReturnConfig | None = None,
    daily_adjusted_close_path: str | Path | None = None,
    raw_data: dict[str, Any] | None = None,
    factor_cache_dir: str | Path | None = None,
) -> DataBundle:
    events = None
    if delisting_events_path is not None and Path(delisting_events_path).exists():
        events = pd.read_csv(delisting_events_path)
        for column in ("announcement_date", "effective_date"):
            if column in events.columns:
                events[column] = pd.to_datetime(events[column])
    return load_study_data(
        input_dir=input_dir,
        eligibility_path=eligibility_path,
        outcome_start=outcome_start,
        outcome_end=outcome_end,
        delisting_events=events,
        realized_return_config=realized_return_config,
        daily_adjusted_close_path=daily_adjusted_close_path,
        raw_data=raw_data,
        factor_cache_dir=factor_cache_dir,
    )


def _validation_key(config: ExperimentConfig) -> tuple[int, int, bool]:
    return (
        int(config.training_months),
        int(config.information_lag_months),
        bool(
            config.min_observations == config.training_months
            and config.max_missing_fraction == 0.0
        ),
    )


def _cache_totals(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for key, value in dict(record.get("cache", {})).items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _safe_worker_count(workers: int, n_tasks: int) -> int:
    workers = int(workers)
    if workers < 0:
        raise ValueError("workers must be nonnegative (0 means automatic).")
    if workers == 0:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    return max(1, min(workers, max(1, int(n_tasks))))


def preflight_existing_jobs(
    jobs: Sequence[ExperimentJob], seeds: Iterable[int], *,
    overwrite: bool = False, data: DataBundle | None = None,
) -> None:
    """Check settings before writing declarations or clearing any seed files."""
    jobs = list(jobs)
    requested = {int(seed) for seed in seeds}
    outputs = [job.normalized_output_dir() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Different jobs cannot write to the same output directory.")
    errors = []
    data_fingerprint = None
    for job in jobs:
        errors.extend(named_setting_errors(job.job_id, asdict(job.config)))
        for seed_dir in sorted(job.normalized_output_dir().glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            marker = seed_dir / "run_manifest.json"
            if overwrite and seed in requested:
                continue
            if not marker.exists():
                if seed not in requested and (seed_dir / "returns.csv").exists():
                    errors.append(f"{seed_dir}: unfinished output outside the requested seeds")
                continue
            try:
                manifest = json.loads(marker.read_text(encoding="utf-8"))
                saved = manifest.get("experiment_config", manifest.get("config"))
                if not isinstance(saved, dict):
                    raise ValueError("missing experiment_config")
                differences = config_differences(saved, job.config)
                if manifest.get("master_seed") != seed:
                    differences.append("master_seed does not match the folder name")
                required = _SEED_OUTPUT_FILES - {"weights.csv.gz"}
                if job.config.save_weights:
                    required = required | {"weights.csv.gz"}
                missing = [name for name in sorted(required)
                           if not (seed_dir / name).is_file() or (seed_dir / name).stat().st_size == 0]
                if missing:
                    differences.append("missing output files: " + ", ".join(missing))
                if data is not None:
                    for group in ("panel_hashes", "source_file_hashes"):
                        previous = manifest.get("data", {}).get(group, {})
                        current = data.source_metadata.get(group, {})
                        for key in previous.keys() & current.keys():
                            if previous[key] != current[key]:
                                differences.append(f"input hash changed: {group}.{key}")
                    calendar = manifest.get("calendar", {})
                    expected_calendar = {
                        "n_outcomes": len(data.calendar),
                        "formation_start": str(data.formation_dates.min().date()),
                        "formation_end": str(data.formation_dates.max().date()),
                        "outcome_start": str(data.outcome_dates.min().date()),
                        "outcome_end": str(data.outcome_dates.max().date()),
                    }
                    if calendar != expected_calendar:
                        differences.append("formation/outcome calendar changed")
                    if manifest.get("resume_signature"):
                        if data_fingerprint is None:
                            data_fingerprint = data_bundle_fingerprint(data)
                        signature = seed_run_signature(
                            data_fingerprint=data_fingerprint,
                            run_code_fingerprint="resume-inputs-v1", master_seed=seed,
                            config=job.config, formation_states=job.formation_states,
                            fixed_state_weights=_weights_for_seed(job, seed),
                        )
                        if signature != manifest["resume_signature"]:
                            differences.append("prepared inputs or state design changed")
                if differences:
                    errors.append(str(seed_dir) + ": " + "; ".join(differences))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                errors.append(f"{marker}: {exc}")
    if errors:
        raise ValueError(
            "Existing outputs do not match the requested run. No seed files were changed.\n"
            + "\n".join(errors[:12])
            + (f"\n... {len(errors) - 12} further issue(s)." if len(errors) > 12 else "")
            + "\nArchive the affected folder outside the output root, or use --overwrite "
              "for ALL affected seeds. A one-seed rerun must not leave mixed configurations."
        )


def run_experiment_jobs(
    data: DataBundle,
    *,
    jobs: Sequence[ExperimentJob],
    seeds: Iterable[int],
    workers: int = 1,
    schedule: str = "auto",
    start_method: str = "spawn",
    numeric_threads_per_worker: int = 1,
    cache_dir: str | Path | None = None,
    cache_mode: str = "readwrite",
    result_cache_dir: str | Path | None = None,
    result_cache_mode: str = "readwrite",
    cache_compression: int = 1,
    overwrite: bool = False,
    fail_fast: bool = True,
    progress: bool = True,
) -> dict[str, dict[str, object]]:
    """Run the grid with deterministic seeds.

    The seed schedule groups configurations to reuse cached fits. The task
    schedule runs each configuration/seed pair separately. Auto selects seed
    grouping for multiple seeds and task scheduling for one seed.
    A missing cache_dir disables the raw-fit cache.
    """

    jobs = list(jobs)
    if not jobs:
        return {}
    job_ids = [str(job.job_id) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("ExperimentJob.job_id values must be unique.")
    requested_seeds = [int(seed) for seed in seeds]
    if not requested_seeds:
        raise ValueError("At least one seed is required.")
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("Seeds must be unique.")
    schedule = str(schedule).lower()
    if schedule not in {"auto", "seed", "task"}:
        raise ValueError("schedule must be auto, seed, or task.")
    if start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError("Unsupported multiprocessing start method.")

    preflight_existing_jobs(jobs, requested_seeds, overwrite=overwrite, data=data)

    validations: dict[tuple[int, int, bool], pd.DataFrame] = {}
    for job in jobs:
        job.config.validate()
        output = job.normalized_output_dir()
        output.mkdir(parents=True, exist_ok=True)
        key = _validation_key(job.config)
        if key not in validations:
            training_months, lag, strict = key
            validations[key] = validate_data_bundle(
                data,
                training_months=training_months,
                information_lag_months=lag,
                require_complete_training_history=strict,
            )
        validations[key].to_csv(output / "data_timing_validation.csv")

    skipped: dict[str, list[int]] = {job.job_id: [] for job in jobs}
    pending_by_seed: dict[int, list[ExperimentJob]] = {
        seed: [] for seed in requested_seeds
    }
    for seed in requested_seeds:
        for job in jobs:
            seed_dir = job.normalized_output_dir() / f"seed_{seed}"
            marker = seed_dir / "run_manifest.json"
            if marker.exists() and not overwrite:
                skipped[job.job_id].append(seed)
            else:
                # keep caller-supplied seed inputs
                _clear_seed_run_files(seed_dir)
                pending_by_seed[seed].append(job)

    pending_pairs = [
        (seed, job)
        for seed, seed_jobs in pending_by_seed.items()
        for job in seed_jobs
    ]
    records: list[dict[str, Any]] = []
    started = time.time()

    effective_schedule = schedule
    if schedule == "auto":
        effective_schedule = "seed" if len(requested_seeds) > 1 else "task"

    n_tasks = (
        len([seed for seed, seed_jobs in pending_by_seed.items() if seed_jobs])
        if effective_schedule == "seed"
        else len(pending_pairs)
    )
    actual_workers = _safe_worker_count(workers, n_tasks) if n_tasks else 1
    result_data_fp = data_bundle_fingerprint(data)
    arm_data_fp = arm_data_fingerprint(data)
    code_fp = arm_code_fingerprint()
    source_root = Path(__file__).resolve().parents[2]
    source_hash = sha256_tree(source_root)
    source_count = len(iter_tree_files(source_root))
    run_code_root = Path(__file__).resolve().parent
    environment_hash = computational_environment_fingerprint()
    run_code_hash = combine_hashes(
        [
            ("package_source", sha256_tree(run_code_root)),
            ("computational_environment", environment_hash),
        ]
    )
    resolved_cache = str(Path(cache_dir).resolve()) if cache_dir is not None else None
    resolved_result_cache = (
        str(Path(result_cache_dir).resolve())
        if result_cache_dir is not None
        else None
    )
    # overwrite may use the cache; refresh forces recomputation
    effective_result_cache_mode = result_cache_mode

    if progress:
        print(
            f"Running {len(pending_pairs)} seed/configuration task(s) "
            f"with workers={actual_workers}, schedule={effective_schedule}, "
            f"arm_cache={cache_mode if resolved_cache else 'off'}, "
            f"result_cache={effective_result_cache_mode if resolved_result_cache else 'off'}."
        )

    if pending_pairs:
        if actual_workers == 1:
            _worker_initializer(
                data,
                resolved_cache,
                cache_mode,
                resolved_result_cache,
                effective_result_cache_mode,
                result_data_fp,
                arm_data_fp,
                code_fp,
                run_code_hash,
                source_hash,
                source_count,
                cache_compression,
                numeric_threads_per_worker,
            )
            if effective_schedule == "seed":
                for seed in requested_seeds:
                    seed_jobs = pending_by_seed[seed]
                    if not seed_jobs:
                        continue
                    batch = _execute_seed_batch(seed, seed_jobs, fail_fast)
                    records.extend(batch)
                    if progress:
                        for record in batch:
                            print(
                                f"[{record['status']}] {record['job_id']} "
                                f"seed={record['seed']} "
                                f"({record['runtime_seconds']:.1f}s)"
                            )
                    if fail_fast and any(r["status"] == "failed" for r in batch):
                        break
            else:
                for seed, job in pending_pairs:
                    record = _execute_job_seed(job, seed)
                    records.append(record)
                    if progress:
                        print(
                            f"[{record['status']}] {record['job_id']} "
                            f"seed={record['seed']} "
                            f"({record['runtime_seconds']:.1f}s)"
                        )
                    if fail_fast and record["status"] == "failed":
                        break
        else:
            context = get_context(start_method)
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=context,
                initializer=_worker_initializer,
                initargs=(
                    data,
                    resolved_cache,
                    cache_mode,
                    resolved_result_cache,
                    effective_result_cache_mode,
                    result_data_fp,
                    arm_data_fp,
                    code_fp,
                    run_code_hash,
                    source_hash,
                    source_count,
                    cache_compression,
                    numeric_threads_per_worker,
                ),
            ) as executor:
                futures = {}
                if effective_schedule == "seed":
                    for seed, seed_jobs in pending_by_seed.items():
                        if seed_jobs:
                            future = executor.submit(
                                _execute_seed_batch,
                                seed,
                                tuple(seed_jobs),
                                fail_fast,
                            )
                            futures[future] = ("seed", seed)
                else:
                    for seed, job in pending_pairs:
                        future = executor.submit(_execute_job_seed, job, seed)
                        futures[future] = (job.job_id, seed)

                stop = False
                for future in as_completed(futures):
                    result = future.result()
                    batch = result if isinstance(result, list) else [result]
                    records.extend(batch)
                    if progress:
                        for record in batch:
                            print(
                                f"[{record['status']}] {record['job_id']} "
                                f"seed={record['seed']} "
                                f"({record['runtime_seconds']:.1f}s)"
                            )
                    if fail_fast and any(r["status"] == "failed" for r in batch):
                        stop = True
                        for other in futures:
                            if other is not future:
                                other.cancel()
                        break
                if stop:
                    executor.shutdown(wait=True, cancel_futures=True)

    manifests: dict[str, dict[str, object]] = {}
    for job in jobs:
        job_records = [record for record in records if record["job_id"] == job.job_id]
        completed = sorted(
            int(record["seed"])
            for record in job_records
            if record["status"] == "completed"
        )
        failures = [
            {
                "seed": int(record["seed"]),
                "error": record.get("error"),
                "traceback": record.get("traceback"),
            }
            for record in job_records
            if record["status"] == "failed"
        ]
        output = job.normalized_output_dir()
        write_json(output / "failures.json", failures)
        manifest: dict[str, object] = {
            "package_version": __version__,
            "source_tree_sha256": source_hash,
            "source_tree_file_count": source_count,
            "executable_code_sha256": run_code_hash,
            "computational_environment_sha256": environment_hash,
            "experiment_config": asdict(job.config),
            "requested_seeds": requested_seeds,
            "completed_seeds": completed,
            "skipped_existing_seeds": sorted(skipped[job.job_id]),
            "failures": failures,
            "runtime_seconds_parent_wall_clock": time.time() - started,
            "sum_completed_task_runtime_seconds": float(
                sum(
                    float(record["runtime_seconds"])
                    for record in job_records
                    if record["status"] == "completed"
                )
            ),
            "parallel_execution": {
                "workers_requested": int(workers),
                "workers_used": int(actual_workers),
                "schedule_requested": schedule,
                "schedule_used": effective_schedule,
                "start_method": start_method,
                "numeric_threads_per_worker": int(numeric_threads_per_worker),
                "fail_fast": bool(fail_fast),
            },
            "arm_cache": {
                "mode": cache_mode if resolved_cache else "off",
                "root": resolved_cache,
                "data_fingerprint": arm_data_fp,
                "code_fingerprint": code_fp,
                "task_totals": _cache_totals(job_records),
            },
            "seed_result_data_fingerprint": result_data_fp,
            "seed_result_cache": {
                "mode": effective_result_cache_mode if resolved_result_cache else "off",
                "root": resolved_result_cache,
                "task_totals": _cache_totals(
                    [{"cache": record.get("result_cache", {})} for record in job_records]
                ),
                "n_hits": int(sum(bool(record.get("result_cache_hit")) for record in job_records)),
            },
            "formation_state_counts": (
                job.formation_states.value_counts().to_dict()
                if job.formation_states is not None
                else None
            ),
            "fixed_state_weights": (
                {state: dict(weights) for state, weights in job.fixed_state_weights.items()}
                if job.fixed_state_weights is not None
                else None
            ),
            "job_metadata": dict(job.metadata),
        }
        write_json(output / "experiment_manifest.json", manifest)
        manifests[job.job_id] = manifest

    failures_all = [record for record in records if record["status"] == "failed"]
    if failures_all:
        summary = "; ".join(
            f"{record['job_id']} seed {record['seed']}: {record.get('error')}"
            for record in failures_all[:8]
        )
        raise RuntimeError(
            f"{len(failures_all)} experiment task(s) failed. {summary}"
        )
    return manifests


def run_experiment_seeds(
    data: DataBundle,
    *,
    seeds: Iterable[int],
    config: ExperimentConfig,
    output_dir: str | Path,
    formation_states: pd.Series | None = None,
    fixed_state_weights: Mapping[str, Mapping[str, float]] | None = None,
    overwrite: bool = False,
    workers: int = 1,
    schedule: str = "auto",
    start_method: str = "spawn",
    numeric_threads_per_worker: int = 1,
    cache_dir: str | Path | None = None,
    cache_mode: str = "readwrite",
    result_cache_dir: str | Path | None = None,
    result_cache_mode: str = "readwrite",
    cache_compression: int = 1,
    fail_fast: bool = True,
    progress: bool = True,
) -> dict[str, object]:
    """Run one experiment configuration."""

    job = ExperimentJob(
        job_id="experiment",
        config=config,
        output_dir=output_dir,
        formation_states=formation_states,
        fixed_state_weights=fixed_state_weights,
    )
    return run_experiment_jobs(
        data,
        jobs=[job],
        seeds=seeds,
        workers=workers,
        schedule=schedule,
        start_method=start_method,
        numeric_threads_per_worker=numeric_threads_per_worker,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
        result_cache_dir=result_cache_dir,
        result_cache_mode=result_cache_mode,
        cache_compression=cache_compression,
        overwrite=overwrite,
        fail_fast=fail_fast,
        progress=progress,
    )["experiment"]
