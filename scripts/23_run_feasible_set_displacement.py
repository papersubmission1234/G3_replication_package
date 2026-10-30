#!/usr/bin/env python3
"""Replay saved risk-map cells and measure feasible-set displacement.

Results go to a separate diagnostics tag. Existing seed CSVs are read only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import json
import os
from pathlib import Path
import time

# set thread limits before importing numerical libraries
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
from _common import ROOT, load_context
from g3_confirmatory.displacement_diagnostics import (
    MAPS, SCORES, VERSION, atomic_csv, atomic_json, digest, load_reference, replay_seed,
)
from g3_confirmatory.arm_cache import ArmResultCache, arm_data_fingerprint, data_bundle_fingerprint, computational_environment_fingerprint
from g3_confirmatory.hashing import sha256_tree

_DATA = None
_CACHE = None
_SETTINGS = None


def initialize(data, settings):
    global _DATA, _CACHE, _SETTINGS
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    _DATA, _SETTINGS = data, settings
    # replay reads the existing raw-arm cache
    _CACHE = ArmResultCache(settings["cache_dir"], mode="read", data_fingerprint=settings["arm_data_hash"])


def execute(seed):
    settings = _SETTINGS
    root, out = Path(settings["root"]), Path(settings["destination"])
    scores = tuple(settings["scores"])
    refs = {(score, label): load_reference(root, score, label, seed) for score in scores for label in MAPS}
    hashes = {path: h for ref in refs.values() for path, h in ref["source_hashes"].items()}
    signature_payload = {"version": VERSION, "seed": seed, "settings": settings, "sources": hashes}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, default=str).encode()).hexdigest()
    folder = out / f"seed_{seed}"
    done = folder / "completed.json"
    if done.exists():
        record = json.loads(done.read_text())
        intact = all((folder / name).exists() and digest(folder / name) == h for name, h in record.get("artifacts", {}).items())
        if record.get("signature") == signature and record.get("artifacts") and intact:
            print(f"[skip] seed {seed}: matching inputs and verified output checksums", flush=True)
            return {"master_seed": seed, "status": "skipped", "runtime_seconds": record["runtime_seconds"]}
        raise ValueError(f"{folder}: existing results have different inputs or damaged outputs. Use a NEW --tag; no files were overwritten.")
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Lock exists: {lock}. If no previous run is active, delete ONLY this RUNNING.lock and resume.")
    os.close(fd)
    start = time.perf_counter()
    try:
        result = replay_seed(_DATA, seed, refs, scores, cache=_CACHE, max_months=settings["max_months"], allow_refit=settings["allow_refit"])
        output_hashes = {}
        for name, frame in result.items():
            target = folder / (name + ".csv")
            atomic_csv(target, frame)
            output_hashes[target.name] = digest(target)
        seconds = time.perf_counter() - start
        atomic_json(done, {"version": VERSION, "signature": signature, "sources": hashes,
                    "runtime_seconds": seconds, "artifacts": output_hashes, "master_seed": seed,
                    "scores": scores, "n_months": int(result["replay_returns"]["formation_date"].nunique()),
                    "full_calendar": settings["max_months"] is None, "settings": settings,
                    "cache_stats": _CACHE.stats.as_dict()})
        print(f"[completed] seed {seed}: {seconds:.1f}s", flush=True)
        return {"master_seed": seed, "status": "completed", "runtime_seconds": seconds}
    except Exception as exc:
        atomic_json(folder / "failure.json", {"error": repr(exc), "signature": signature})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scores", default=",".join(SCORES))
    parser.add_argument("--max-months", type=int, help="Smoke only; use a separate --tag.")
    parser.add_argument("--tag", default="primary")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-refit", action="store_true", help="Explicitly permit raw-arm regeneration on a cache miss; original hashes must still match.")
    args = parser.parse_args()
    if args.workers < 1 or (args.max_months is not None and args.max_months < 1):
        parser.error("workers and max-months must be positive.")
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("tag must contain only letters, numbers, hyphens, or underscores.")
    os.chdir(ROOT)
    scores = tuple(x.strip() for x in args.scores.split(","))
    if not scores or set(scores) - set(SCORES) or len(scores) != len(set(scores)):
        parser.error("scores must be a unique subset of post_cost_utility,predictive_log")
    cfg, paths, data, base, _, configured = load_context(args.config)
    seeds = [args.seed] if args.seed is not None else configured
    destination = paths["output_root"] / "diagnostics/displacement" / args.tag
    # record the source tree for resume checks
    code_hash = sha256_tree(ROOT / "src/g3_confirmatory")
    settings = dict(root=str(paths["output_root"]), destination=str(destination), scores=scores,
                    max_months=args.max_months, allow_refit=args.allow_refit, data_hash=data_bundle_fingerprint(data),
                    arm_data_hash=arm_data_fingerprint(data), environment_hash=computational_environment_fingerprint(),
                    code_hash=code_hash, script_hash=digest(Path(__file__)),
                    cache_dir=str((args.cache_dir or paths["output_root"].parent / "cache/raw_arms").resolve()))
    for seed in seeds:
        for score in scores:
            for label in MAPS:
                load_reference(paths["output_root"], score, label, seed)
    begin = time.perf_counter()
    if args.workers == 1:
        initialize(data, settings)
        records = [execute(seed) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds)), mp_context=get_context("spawn"),
                initializer=initialize, initargs=(data, settings)) as pool:
            records = [future.result() for future in as_completed([pool.submit(execute, seed) for seed in seeds])]
    atomic_json(destination / "last_invocation.json", {"seeds": seeds, "workers": args.workers,
                "runtime_seconds": time.perf_counter() - begin, "records": records})
    print(f"Done. Targeted outputs: {destination}")


if __name__ == "__main__":
    main()
