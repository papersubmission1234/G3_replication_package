#!/usr/bin/env python3
"""Test action-layer mean shrinkage with frozen posterior weights.

Posterior weights come from the unshrunk core run. Hard-action and row-mixture
random streams are reused at every shrinkage fraction. The common_full map
also fixes the unshrunk HierMacro CVaR scenarios and budget. This separates
mean shrinkage from posterior learning and risk-map changes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
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
from g3_confirmatory.actions import hard_arm_from_uniform, mix_scenarios_crn
from g3_confirmatory.displacement_diagnostics import atomic_csv, atomic_json, digest, load_reference
from g3_confirmatory.arm_cache import ArmResultCache, arm_data_fingerprint, arm_generation_config
from g3_confirmatory.arms import shrink_scenario_mean
from g3_confirmatory.hashing import sha256_dataframe, sha256_tree
from g3_confirmatory.optimizer import equal_weight_cvar
from g3_confirmatory.randomness import NamedRNGFactory

FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
RISK_MAPS = ("endogenous", "common_full")
STRATEGIES = (
    "post_cost_utility__hard",
    "post_cost_utility__mix",
    "fixed_equal__mix",
    "standalone__M_HierMacro",
    "standalone__M_FF5MOM",
)
VERSION = 4

# use saved zero-shrinkage paths to avoid solver replay drift
ZERO_CONTROL_MODE = "exact_source_reuse"


_DATA = None
_CACHE = None
_SETTINGS = None


def _safe_fraction_label(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _initialize(data, settings):
    global _DATA, _CACHE, _SETTINGS
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    _DATA, _SETTINGS = data, settings
    _CACHE = ArmResultCache(
        settings["cache_dir"],
        mode="read",
        data_fingerprint=settings["arm_data_hash"],
    )


def _generate_original_scenarios(data, base, seed, formation, cache, allow_refit=False):
    factory = NamedRNGFactory(seed)
    panel, classes, dates = engine._eligible_training_panel(data, formation, base)
    factors = engine._factor_inputs(data, dates)
    generated = {}
    for arm_name, arm in engine.build_arms(base).items():
        def compute(_arm=arm, _name=arm_name):
            if cache is not None and not allow_refit:
                raise FileNotFoundError(
                    f"Raw-arm cache miss for seed={seed}, date={formation.date()}, arm={_name}. "
                    "No refit started. Check --cache-dir or add --allow-refit deliberately."
                )
            return _arm.fit_generate(
                excess_returns=panel,
                factors=factors,
                asset_classes=classes,
                n_scenarios=base.n_scenarios,
                rng=factory.generator("arm", _name, formation.date()),
            )
        if cache is None:
            result = compute()
        else:
            payload = cache.key_payload(
                master_seed=seed,
                formation_date=formation,
                arm_name=arm_name,
                n_scenarios=base.n_scenarios,
                generation_config=arm_generation_config(base),
            )
            result, _, _ = cache.get_or_compute(payload, compute)
        if arm_name == "M_HierMacro":
            treatment = engine.apply_hierarchical_mean_treatment(
                result,
                base.hierarchical_mean_treatment,
                rng=factory.generator("mean_treatment", base.hierarchical_mean_treatment, formation.date()),
            )
            result.scenarios = treatment.scenarios
        generated[arm_name] = result
    return engine._intersect_arm_scenarios(generated, base.min_assets)[0]


def _run_seed(seed: int):
    settings = _SETTINGS
    out_root = Path(settings["destination"])
    source_root = Path(settings["source_root"])
    ref = load_reference(source_root, "post_cost_utility", "endogenous", seed)
    base = ref["config"]
    if tuple(base.arm_names) != ("M_FF5MOM", "M_HierMacro"):
        raise ValueError(f"Frozen shrinkage expects the two-arm core, got {base.arm_names}")
    posterior = ref["posterior"].set_index(["formation_date", "arm"])
    source_returns = ref["returns"].set_index(["formation_date", "strategy"])
    source_diagnostics = ref["optimizer_diagnostics"].set_index(["formation_date", "strategy"])
    source_hashes = ref["scenario_hashes"].set_index(["formation_date", "arm"])

    calendar = _DATA.calendar
    factory = NamedRNGFactory(seed)
    states = {
        (fraction, risk_map, strategy): engine.StrategyState(strategy, wealth=base.initial_wealth)
        for fraction in FRACTIONS for risk_map in RISK_MAPS for strategy in STRATEGIES
    }
    rows = {(fraction, risk_map): [] for fraction in FRACTIONS for risk_map in RISK_MAPS}
    diags = {(fraction, risk_map): [] for fraction in FRACTIONS for risk_map in RISK_MAPS}
    posterior_rows = []

    for t, row in enumerate(calendar.itertuples(index=False)):
        formation = pd.Timestamp(row.formation_date)
        outcome = pd.Timestamp(row.outcome_date)
        first = t == 0
        original = _generate_original_scenarios(
            _DATA, base, seed, formation, _CACHE, settings["allow_refit"]
        )
        for arm_name, matrix in original.items():
            h = sha256_dataframe(matrix)
            old = source_hashes.loc[(formation, arm_name), "scenario_sha256"]
            if h != old:
                raise ValueError(
                    f"Scenario hash mismatch for seed={seed} date={formation.date()} arm={arm_name}."
                )

        names = list(original)
        weights = {a: float(posterior.loc[(formation, a), "weight"]) for a in names}
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
            raise ValueError("Saved posterior does not sum to one.")
        for a, w in weights.items():
            posterior_rows.append({
                "master_seed": seed,
                "formation_date": formation,
                "score": "post_cost_utility",
                "arm": a,
                "weight": w,
                "source": "unshrunk_current_core",
            })

        hard_u = float(factory.generator("hard_action", formation.date()).random())
        arm_u = factory.generator("mix_labels", formation.date()).random(base.n_scenarios)
        row_u = factory.generator("mix_row_resample", formation.date()).random(base.n_scenarios)
        hard_arm = hard_arm_from_uniform(weights, names, hard_u)

        q_equal = {a: 1.0 / len(names) for a in names}
        unshrunk_hier = original["M_HierMacro"]
        common_budget = equal_weight_cvar(unshrunk_hier, base.optimizer.cvar_alpha)

        for fraction in FRACTIONS:
            transformed = {
                a: shrink_scenario_mean(m, fraction) for a, m in original.items()
            }
            hard = transformed[hard_arm]
            mixed, _ = mix_scenarios_crn(
                transformed, weights, n_rows=base.n_scenarios,
                arm_uniforms=arm_u, row_uniforms=row_u,
            )
            equal_mix, _ = mix_scenarios_crn(
                transformed, q_equal, n_rows=base.n_scenarios,
                arm_uniforms=arm_u, row_uniforms=row_u,
            )
            objectives = {
                "post_cost_utility__hard": hard,
                "post_cost_utility__mix": mixed,
                "fixed_equal__mix": equal_mix,
                "standalone__M_HierMacro": transformed["M_HierMacro"],
                "standalone__M_FF5MOM": transformed["M_FF5MOM"],
            }

            for risk_map in RISK_MAPS:
                for strategy, P in objectives.items():
                    if fraction == 0.0 and risk_map == "endogenous":
                        key = (formation, strategy)
                        if key not in source_returns.index:
                            raise KeyError(
                                f"Missing current-core control row for seed={seed} "
                                f"date={formation.date()} strategy={strategy}."
                            )
                        if key not in source_diagnostics.index:
                            raise KeyError(
                                f"Missing current-core optimizer diagnostics for seed={seed} "
                                f"date={formation.date()} strategy={strategy}."
                            )

                        ret_payload = source_returns.loc[key].to_dict()
                        ret_payload.update({
                            "master_seed": seed,
                            "formation_date": formation,
                            "outcome_date": outcome,
                            "strategy": strategy,
                            "shrinkage_fraction": fraction,
                            "risk_map": risk_map,
                            "posterior_source": "unshrunk_current_core",
                            "control_source": ZERO_CONTROL_MODE,
                        })
                        rows[(fraction, risk_map)].append(ret_payload)

                        diag_payload = source_diagnostics.loc[key].to_dict()
                        diag_payload.update({
                            "master_seed": seed,
                            "formation_date": formation,
                            "strategy": strategy,
                            "shrinkage_fraction": fraction,
                            "risk_map": risk_map,
                            "control_source": ZERO_CONTROL_MODE,
                        })
                        diags[(fraction, risk_map)].append(diag_payload)
                        continue

                    state = states[(fraction, risk_map, strategy)]
                    pretrade = engine._pretrade_for_state(state, formation, _DATA, first_period=first)
                    if risk_map == "endogenous":
                        Q = P
                        budget = equal_weight_cvar(P, base.optimizer.cvar_alpha)
                        explicit = None
                        risk_scenarios = None
                    else:
                        Q = unshrunk_hier
                        budget = common_budget
                        explicit = budget
                        risk_scenarios = Q
                    target, diag = engine._optimize_or_fallback(
                        P, pretrade, base, first_period=first,
                        explicit_cvar_budget=explicit,
                        risk_scenarios=risk_scenarios,
                    )
                    perf = engine._execute_strategy(
                        state=state,
                        target=target,
                        pretrade=pretrade,
                        formation_date=formation,
                        outcome_date=outcome,
                        data=_DATA,
                        config=base,
                        first_period=first,
                    )
                    rows[(fraction, risk_map)].append({
                        "master_seed": seed,
                        "formation_date": formation,
                        "outcome_date": outcome,
                        "strategy": strategy,
                        "shrinkage_fraction": fraction,
                        "risk_map": risk_map,
                        "posterior_source": "unshrunk_current_core",
                        **perf,
                    })
                    diags[(fraction, risk_map)].append({
                        "master_seed": seed,
                        "formation_date": formation,
                        "strategy": strategy,
                        "shrinkage_fraction": fraction,
                        "risk_map": risk_map,
                        **diag,
                    })


    source_digest = {k: v for k, v in ref["source_hashes"].items()}
    for fraction in FRACTIONS:
        for risk_map in RISK_MAPS:
            exp = f"shrink_{_safe_fraction_label(fraction)}_{risk_map}"
            folder = out_root / exp / f"seed_{seed}"
            if folder.exists() and not settings["overwrite"]:
                raise FileExistsError(f"{folder} exists. Use --overwrite or remove only this extension folder.")
            folder.mkdir(parents=True, exist_ok=True)
            ret = pd.DataFrame(rows[(fraction, risk_map)])
            opt = pd.DataFrame(diags[(fraction, risk_map)])
            atomic_csv(folder / "returns.csv", ret)
            atomic_csv(folder / "optimizer_diagnostics.csv", opt)
            atomic_csv(folder / "posterior.csv", pd.DataFrame(posterior_rows).drop_duplicates())
            manifest = {
                "diagnostic_extension": "frozen_posterior_shrinkage",
                "version": VERSION,
                "master_seed": seed,
                "shrinkage_fraction": fraction,
                "risk_map": risk_map,
                "posterior_frozen": True,
                "posterior_source": "post_cost_utility endogenous unshrunk core",
                "common_full_map_is_unshrunk_across_fractions": risk_map == "common_full",
                "source_hashes": source_digest,
                "fraction_zero_control": {
                    "mode": ZERO_CONTROL_MODE,
                    "status": "reused_authoritative_current_core_rows",
                    "scenario_hashes_verified_exactly": True,
                    "posterior_frozen_to_current_core": True,
                    "note": (
                        "The lambda=0 endogenous control is copied from the current core "
                        "rather than re-solved, avoiding platform-specific LP numerical noise."
                    ),
                },
                "experiment_config": asdict(base),
            }
            atomic_json(folder / "run_manifest.json", manifest)
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-refit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.chdir(ROOT)
    cfg, paths, data, base, _, configured = load_context(args.config)
    seeds = [args.seed] if args.seed is not None else configured
    destination = paths["output_root"] / "diagnostics/extensions/frozen_shrinkage"
    settings = {
        "destination": str(destination),
        "source_root": str(paths["output_root"]),
        "cache_dir": str((args.cache_dir or paths["output_root"].parent / "cache/raw_arms").resolve()),
        "arm_data_hash": arm_data_fingerprint(data),
        "allow_refit": bool(args.allow_refit),
        "overwrite": bool(args.overwrite),
        "script_hash": digest(Path(__file__)),
        "source_tree_hash": sha256_tree(ROOT / "src/g3_confirmatory"),
    }
    start = time.perf_counter()
    if args.workers == 1:
        _initialize(data, settings)
        done = [_run_seed(s) for s in seeds]
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds)),
            mp_context=__import__("multiprocessing").get_context("spawn"),
            initializer=_initialize,
            initargs=(data, settings),
        ) as pool:
            done = [f.result() for f in as_completed([pool.submit(_run_seed, s) for s in seeds])]
    atomic_json(destination / "extension_protocol.json", {
        "family": "frozen_posterior_shrinkage",
        "fractions": FRACTIONS,
        "risk_maps": RISK_MAPS,
        "posterior_frozen": True,
        "common_full_map_unshrunk_across_fractions": True,
        "seeds": sorted(done),
        "runtime_seconds": time.perf_counter() - start,
    })
    print(f"Completed frozen-posterior shrinkage: {destination}")


if __name__ == "__main__":
    main()
