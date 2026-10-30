#!/usr/bin/env python3
"""Compare covariance estimators under three CVaR maps.

Each pool contains M_FF5MOM and a zero-mean M_Covariance arm. The covariance
estimator is sample, Marchenko-Pastur, Ledoit-Wolf, or EWMA with Ledoit-Wolf.
The arm name and random stream are shared across estimators.

Utility scoring uses hard and row-mixture actions under endogenous, common
budget, and common full covariance-arm risk maps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from _common import ROOT, load_context
from g3_confirmatory import experiment as engine
from g3_confirmatory.covariance_estimators import SampleCovarianceArm, MarchenkoPasturArm
from g3_confirmatory.arm_cache import ArmResultCache, arm_data_fingerprint
from g3_confirmatory.arms import LedoitWolfArm, EWMAShrinkageArm
from g3_confirmatory.serialization import write_json
from g3_confirmatory.run_checks import checked_core_config

ESTIMATORS = ("sample", "mp", "ledoit_wolf", "ewma_ledoit_wolf")
MAPS = (
    ("endogenous", "none", "none"),
    ("common_budget", "M_Covariance", "none"),
    ("common_full", "M_Covariance", "M_Covariance"),
)

_DATA = None
_SETTINGS = None
_CACHE = None
_ORIGINAL_BUILD = None


class _MemoizedArm:
    def __init__(self, arm):
        self.arm = arm
        self.name = arm.name
        self.memo = {}

    def fit_generate(self, *, excess_returns, factors, asset_classes, n_scenarios, rng):
        key = (str(excess_returns.index[0]), str(excess_returns.index[-1]), tuple(map(str, excess_returns.columns)), int(n_scenarios))
        if key not in self.memo:
            self.memo[key] = self.arm.fit_generate(
                excess_returns=excess_returns, factors=factors, asset_classes=asset_classes,
                n_scenarios=n_scenarios, rng=rng,
            )
        return copy.deepcopy(self.memo[key])


def _arm_for(estimator: str):
    if estimator == "sample":
        arm = SampleCovarianceArm(name="M_Covariance")
    elif estimator == "mp":
        arm = MarchenkoPasturArm(name="M_Covariance")
    elif estimator == "ledoit_wolf":
        arm = LedoitWolfArm(name="M_Covariance")
    elif estimator == "ewma_ledoit_wolf":
        arm = EWMAShrinkageArm(name="M_Covariance")
    else:
        raise ValueError(estimator)
    return _MemoizedArm(arm)


def _initialize(data, settings):
    global _DATA, _SETTINGS, _CACHE, _ORIGINAL_BUILD
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    _DATA, _SETTINGS = data, settings
    _ORIGINAL_BUILD = engine.build_arms
    _CACHE = ArmResultCache(
        settings["cache_dir"], mode="read",
        data_fingerprint=settings["arm_data_hash"],
    )


def _run_one(seed: int, estimator: str, cov_arm, map_label: str, budget_source: str, scenario_source: str):
    global _ORIGINAL_BUILD
    original = _ORIGINAL_BUILD

    def patched_build(config):
        base_names = tuple(a for a in config.arm_names if a != "M_Covariance")
        base_cfg = replace(config, arm_names=base_names)
        built = original(base_cfg)
        built["M_Covariance"] = cov_arm
        return {name: built[name] for name in config.arm_names}

    engine.build_arms = patched_build
    try:
        cfg = replace(
            _SETTINGS["base"],
            arm_names=("M_FF5MOM", "M_Covariance"),
            score_names=("post_cost_utility",),
            action_names=("hard", "mix"),
            include_equal_mix=True,
            save_weights=False,
            posterior_partition="pooled",
            common_cvar_budget_source="none",
            loss_cvar_budget_source="none",
            loss_cvar_scenario_source="none",
            action_cvar_budget_source=budget_source,
            action_cvar_scenario_source=scenario_source,
            mean_shrinkage_fraction=0.0,
        )
        result = engine.run_seed(
            _DATA,
            master_seed=seed,
            config=cfg,
            arm_cache=_CACHE,
        )
        result.manifest["diagnostic_extension"] = "covariance_regularization"
        result.manifest["covariance_estimator"] = estimator
        result.manifest["risk_map"] = map_label
        return result
    finally:
        engine.build_arms = original


def _execute_seed(seed: int):
    root = Path(_SETTINGS["destination"])
    results = {}
    for estimator in _SETTINGS["estimators"]:
        # reuse covariance scenarios and streams across risk maps
        cov_arm = _arm_for(estimator)
        map_posteriors = []
        for map_label, budget_source, scenario_source in MAPS:
            exp = f"{estimator}__{map_label}"
            folder = root / exp / f"seed_{seed}"
            if folder.exists() and not _SETTINGS["overwrite"]:
                raise FileExistsError(f"{folder} exists. Use --overwrite or keep the existing completed extension.")
            result = _run_one(seed, estimator, cov_arm, map_label, budget_source, scenario_source)
            folder.mkdir(parents=True, exist_ok=True)
            engine.write_seed_result(result, folder)
            map_posteriors.append(result.posterior[["formation_date", "arm", "weight"]].sort_values(["formation_date", "arm"]).reset_index(drop=True))
            results[exp] = folder
        first = map_posteriors[0]
        for other in map_posteriors[1:]:
            if not first[["formation_date", "arm"]].equals(other[["formation_date", "arm"]]) or not np.allclose(first["weight"], other["weight"], atol=1e-12, rtol=0):
                raise ValueError(f"Posterior differs across risk-map cells for estimator={estimator}, seed={seed}")
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--estimator",
        choices=ESTIMATORS,
        help="Run only one covariance estimator; default runs all estimators.",
    )

    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for est in ESTIMATORS:
            for label, budget, scenario in MAPS:
                print(f"{est}__{label}: budget={budget}, risk_scenarios={scenario}")
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.chdir(ROOT)
    cfg, paths, data, base, _, configured = load_context(args.config)
    seeds = [args.seed] if args.seed is not None else configured
    base = checked_core_config(paths["output_root"], base, seeds, required=False)
    selected_estimators = (
        (args.estimator,) if args.estimator is not None else ESTIMATORS
    )

    destination = paths["output_root"] / "diagnostics/extensions/covariance_robustness"
    settings = {
        "destination": str(destination),
        "cache_dir": str((args.cache_dir or paths["output_root"].parent / "cache/raw_arms").resolve()),
        "arm_data_hash": arm_data_fingerprint(data),
        "overwrite": bool(args.overwrite),
        "base": base,
        "estimators": selected_estimators,
    }
    start = time.perf_counter()
    if args.workers == 1:
        _initialize(data, settings)
        done = [_execute_seed(s) for s in seeds]
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds)),
            mp_context=__import__("multiprocessing").get_context("spawn"),
            initializer=_initialize,
            initargs=(data, settings),
        ) as pool:
            done = [f.result() for f in as_completed([pool.submit(_execute_seed, s) for s in seeds])]
    write_json(destination / "extension_protocol.json", {
        "family": "covariance_robustness",
        "estimators": ESTIMATORS,
        "executed_estimators": list(selected_estimators),
        "risk_maps": [x[0] for x in MAPS],
        "matched_zero_mean_covariance_arm": True,
        "common_arm_name_and_rng_stream_across_estimators": True,
        "seeds": sorted(done),
        "runtime_seconds": time.perf_counter() - start,
    })
    print(f"Completed covariance robustness: {destination}")


if __name__ == "__main__":
    main()
