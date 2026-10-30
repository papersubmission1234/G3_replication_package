"""Feasible-set diagnostics from saved predictions and pre-action posteriors.

Return and scenario hashes are checked against the three source risk-map runs.
Opportunity sets use the convex first-stage problem before sparse refitting.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import experiment as engine
from .actions import hard_arm_from_uniform, mix_scenarios_crn
from .arm_cache import arm_generation_config
from .configuration import experiment_config_from_mapping
from .hashing import sha256_dataframe
from .optimizer import empirical_cvar, equal_weight_cvar, _solve_lp
from .randomness import NamedRNGFactory

MAPS = ("endogenous", "hier_budget_only", "hier_full_risk")
SCORES = ("post_cost_utility", "predictive_log")
VERSION = "displacement-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def experiment_id(score: str, risk_map: str) -> str:
    if score not in SCORES or risk_map not in MAPS:
        raise ValueError("Unsupported score or risk map.")
    if score == "post_cost_utility":
        return f"{score}__loss_endogenous__action_{risk_map}"
    return f"{score}__action_{risk_map}"


def seed_dir(root: Path, score: str, risk_map: str, seed: int) -> Path:
    return root / "decompositions" / "full_risk_map" / experiment_id(score, risk_map) / f"seed_{seed}"


def load_reference(root: Path, score: str, risk_map: str, seed: int) -> dict:
    folder = seed_dir(root, score, risk_map, seed)
    names = ("run_manifest.json", "returns.csv", "posterior.csv", "scenario_hashes.csv", "optimizer_diagnostics.csv")
    missing = [str(folder / name) for name in names if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError("Replay requires the original per-seed outputs, not just paper tables. Missing:\n" + "\n".join(missing))
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    cfg_key = "config" if "config" in manifest else "experiment_config"
    if cfg_key not in manifest:
        raise ValueError(f"No experiment configuration in {folder / 'run_manifest.json'}")
    config = experiment_config_from_mapping(manifest[cfg_key])
    config.validate()
    tables = {}
    for name in names[1:]:
        table = pd.read_csv(folder / name, float_precision="round_trip")
        for col in ("formation_date", "outcome_date"):
            if col in table:
                table[col] = pd.to_datetime(table[col])
        tables[name.split(".")[0]] = table
    return {"folder": folder, "config": config, "manifest": manifest, **tables,
            "source_hashes": {str(folder / name): digest(folder / name) for name in names}}


def numeric_binding(frame: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    """Actual portfolio ES slack, not the slack of one auxiliary LP row."""
    out = frame.copy()
    if "cvar_budget" not in out or not ({"risk_scenario_cvar", "scenario_cvar"} & set(out.columns)):
        raise ValueError("Need cvar_budget and risk_scenario_cvar (or scenario_cvar).")
    risk = pd.to_numeric(out.get("risk_scenario_cvar", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "scenario_cvar" in out:
        risk = risk.fillna(pd.to_numeric(out["scenario_cvar"], errors="coerce"))
    budget = pd.to_numeric(out["cvar_budget"], errors="coerce")
    valid = np.isfinite(budget) & np.isfinite(risk)
    out["true_cvar_slack"] = budget - risk
    out["true_cvar_valid"] = valid
    out["true_cvar_binding"] = np.where(valid, np.abs(budget - risk) <= tolerance, np.nan)
    out["true_cvar_violation"] = np.where(valid, risk - budget > tolerance, np.nan)
    if "cvar_budget_binding" in out:
        out["reported_lp_binding"] = out["cvar_budget_binding"].map(
            lambda v: float(str(v).lower() == "true" or str(v) == "1") if pd.notna(v) else np.nan)
    if "cvar_budget_shadow_value_max_objective" in out:
        shadow = pd.to_numeric(out["cvar_budget_shadow_value_max_objective"], errors="coerce")
        out["positive_cvar_shadow"] = np.where(np.isfinite(shadow), shadow > tolerance, np.nan)
    return out


def matched_tail_panel(diagnostics: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Compare scenario risk with realized excess-return losses.
    
    Net risk thresholds include the observed implementation cost. Comparing net
    losses with gross forecasts is reported separately as a cost stress.
    """
    keys = ["master_seed", "strategy", "formation_date"]
    d, r = diagnostics.copy(), returns.copy()
    for frame in (d, r):
        frame["formation_date"] = pd.to_datetime(frame["formation_date"])
    required = keys + ["gross_excess_return", "net_excess_return", "gross_simple_return", "transaction_cost"]
    if set(required) - set(r):
        raise ValueError("Matched calibration requires saved gross/net excess returns and transaction costs.")
    keep = keys + ["gross_excess_return", "net_excess_return", "gross_simple_return", "transaction_cost"]
    z = d.merge(r[keep], on=keys, how="left", validate="one_to_one", indicator=True)
    if not z["_merge"].eq("both").all():
        raise ValueError("Some optimizer records have no matched realized return.")
    z = z.drop(columns="_merge")
    for risk_name in ("var", "cvar"):
        field = "risk_scenario_" + risk_name
        risk = pd.to_numeric(z.get(field, pd.Series(np.nan, index=z.index)), errors="coerce")
        if "scenario_" + risk_name in z:
            risk = risk.fillna(pd.to_numeric(z["scenario_" + risk_name], errors="coerce"))
        z[field] = risk
    gross_loss = -z["gross_excess_return"]
    net_loss = -z["net_excess_return"]
    cost = z["transaction_cost"]
    if not np.allclose(net_loss, gross_loss + cost, atol=1e-10, rtol=0, equal_nan=False):
        raise ValueError("Saved gross/net excess returns do not reconcile with costs.")
    valid = np.isfinite(gross_loss) & np.isfinite(z.risk_scenario_var) & np.isfinite(z.risk_scenario_cvar)
    z["calibration_valid"] = valid
    z["realized_gross_excess_loss"] = gross_loss
    z["realized_net_excess_loss"] = net_loss
    z["predicted_net_excess_var"] = z.risk_scenario_var + cost
    z["predicted_net_excess_cvar"] = z.risk_scenario_cvar + cost
    z["gross_excess_var_exceedance"] = np.where(valid, gross_loss > z.risk_scenario_var, np.nan)
    z["matched_net_excess_var_exceedance"] = np.where(valid, net_loss > z.predicted_net_excess_var, np.nan)
    z["old_simple_against_excess_var_exceedance"] = np.where(valid, -z.gross_simple_return > z.risk_scenario_var, np.nan)
    z["cost_stress_net_loss_above_gross_var"] = np.where(valid, net_loss > z.risk_scenario_var, np.nan)
    z["gross_excess_loss_minus_cvar"] = gross_loss - z.risk_scenario_cvar
    return z


def risk_gap(Q: pd.DataFrame, budget: float, w: pd.Series, alpha: float) -> float:
    if w.index.difference(Q.columns).size and abs(w.reindex(w.index.difference(Q.columns)).sum()) > 1e-10:
        raise ValueError("Portfolio has nonzero weights outside risk universe.")
    vector = w.reindex(Q.columns).fillna(0.0).to_numpy(float)
    return float(empirical_cvar(Q.to_numpy(float) @ vector, alpha) - budget)


def allocation_distance(a: pd.Series, b: pd.Series) -> float:
    universe = a.index.union(b.index)
    return float(0.5 * np.abs(a.reindex(universe, fill_value=0.0) - b.reindex(universe, fill_value=0.0)).sum())


def parent_solution(P, Q, budget, pretrade, config, first):
    """Same objective/holdings; full convex feasible set, before sparse refit."""
    solved = _solve_lp(P, pretrade, config.optimizer, min_weight_floor=0.0,
                      cvar_budget=budget, initial_funding=first, risk_scenarios=Q)
    if risk_gap(Q, budget, solved.weights, config.optimizer.cvar_alpha) > 1e-6:
        raise AssertionError("Convex-parent solution fails its own risk constraint.")
    return solved.weights, -float(solved.diagnostics["objective_minimized"])


def _validate_configs(refs: dict, scores: tuple[str, ...]):
    first = refs[scores[0], MAPS[0]]["config"]
    allowed = {"score_names", "action_cvar_budget_source", "action_cvar_scenario_source", "save_weights"}
    canonical = {k: v for k, v in asdict(first).items() if k not in allowed}
    if first.arm_names != ("M_FF5MOM", "M_HierMacro"):
        raise ValueError("This replay is specifically the current two-model SAR experiment.")
    if first.action_names != ("hard", "mix") or first.posterior_partition != "pooled":
        raise ValueError("Unsupported action set or state-dependent posterior.")
    if first.arm_scenario_mode != "posterior_predictive" or first.mean_shrinkage_fraction != 0:
        raise ValueError("Replay primary posterior-predictive unshrunk scenarios only.")
    for (score, label), ref in refs.items():
        c = ref["config"]
        if {k: v for k, v in asdict(c).items() if k not in allowed} != canonical:
            raise ValueError(f"Non-risk-map configuration mismatch: {score}/{label}")
        expected_b = "none" if label == "endogenous" else "M_HierMacro"
        expected_q = "M_HierMacro" if label == "hier_full_risk" else "none"
        if c.score_names != (score,) or c.action_cvar_budget_source != expected_b or c.action_cvar_scenario_source != expected_q:
            raise ValueError(f"Incorrect saved risk-map specification: {score}/{label}")
        if c.loss_cvar_budget_source != "none" or c.loss_cvar_scenario_source != "none":
            raise ValueError("Loss map must stay endogenous for the action decomposition.")
    return first


def replay_seed(data, seed: int, refs: dict, scores: tuple[str, ...], cache=None,
                max_months: int | None = None, progress: Callable[[str], None] = print, allow_refit: bool = True) -> dict[str, pd.DataFrame]:
    """Replay hard and mixture actions from saved model outputs.
    
    Accept a seed only when its hashes and monthly realized results agree with
    the source run. Diagnostic actions do not update the source posterior.
    """
    base = _validate_configs(refs, scores)
    factory = NamedRNGFactory(seed)
    arms = engine.build_arms(base)
    names = list(arms)
    calendar = data.calendar.iloc[:max_months] if max_months else data.calendar
    source_dates = pd.DatetimeIndex(refs[scores[0], MAPS[0]]["returns"]["formation_date"].unique()).sort_values()
    if not pd.DatetimeIndex(data.calendar["formation_date"]).equals(source_dates):
        raise ValueError("Current data calendar is not the original risk-map calendar.")
    states = {(score, action, label): engine.StrategyState(name=f"{score}__{action}", wealth=base.initial_wealth)
              for score in scores for action in ("hard", "mix") for label in MAPS}
    tables = {name: [] for name in ("replay_returns", "risk_evaluations", "displacement", "opportunity_values", "reconciliation", "scenario_hashes")}
    indexed = {}
    for key, ref in refs.items():
        indexed[key] = {
            "returns": ref["returns"].set_index(["formation_date", "strategy"], verify_integrity=True),
            "diagnostics": ref["optimizer_diagnostics"].set_index(["formation_date", "strategy"], verify_integrity=True),
            "hashes": ref["scenario_hashes"].set_index(["formation_date", "arm"], verify_integrity=True),
            "posterior": ref["posterior"].set_index(["formation_date", "arm"], verify_integrity=True),
        }
    for t, row in enumerate(calendar.itertuples(index=False)):
        formation, outcome = pd.Timestamp(row.formation_date), pd.Timestamp(row.outcome_date)
        first = t == 0
        panel, classes, dates = engine._eligible_training_panel(data, formation, base)
        factors = engine._factor_inputs(data, dates)
        generated = {}
        for arm_name, arm in arms.items():
            def compute(_arm=arm, _name=arm_name):
                if cache is not None and not allow_refit:
                    raise FileNotFoundError(f"Raw-arm cache miss for seed={seed}, date={formation.date()}, arm={_name}. "
                        "No refit started. Check --cache-dir and numerical environment; use --allow-refit only deliberately.")
                return _arm.fit_generate(excess_returns=panel, factors=factors, asset_classes=classes,
                                         n_scenarios=base.n_scenarios, rng=factory.generator("arm", _name, formation.date()))
            if cache is None:
                result = compute()
            else:
                payload = cache.key_payload(master_seed=seed, formation_date=formation, arm_name=arm_name,
                    n_scenarios=base.n_scenarios, generation_config=arm_generation_config(base))
                result, _, _ = cache.get_or_compute(payload, compute)
            if arm_name == "M_HierMacro":
                treatment = engine.apply_hierarchical_mean_treatment(result, base.hierarchical_mean_treatment,
                    rng=factory.generator("mean_treatment", base.hierarchical_mean_treatment, formation.date()))
                result.scenarios = treatment.scenarios
            generated[arm_name] = result
        matrices, assets = engine._intersect_arm_scenarios(generated, base.min_assets)
        for arm_name, matrix in matrices.items():
            h = sha256_dataframe(matrix)
            for key in refs:
                old = indexed[key]["hashes"].loc[(formation, arm_name), "scenario_sha256"]
                if h != old:
                    raise ValueError(f"Scenario hash mismatch: seed={seed}, date={formation.date()}, arm={arm_name}, cell={key}. "
                                     "Stop: check numerical environment, prepared data, code and original raw-arm cache.")
            tables["scenario_hashes"].append(dict(master_seed=seed, formation_date=formation, arm=arm_name, scenario_sha256=h))
        hu = float(factory.generator("hard_action", formation.date()).random())
        au = factory.generator("mix_labels", formation.date()).random(base.n_scenarios)
        ru = factory.generator("mix_row_resample", formation.date()).random(base.n_scenarios)
        HQ = matrices["M_HierMacro"]
        HB = equal_weight_cvar(HQ, base.optimizer.cvar_alpha)
        for score in scores:
            weights = {a: float(indexed[score, MAPS[0]]["posterior"].loc[(formation, a), "weight"]) for a in names}
            if not np.isclose(sum(weights.values()), 1.0, atol=1e-12) or min(weights.values()) < 0:
                raise ValueError("Invalid saved posterior.")
            for label in MAPS[1:]:
                check = np.array([indexed[score, label]["posterior"].loc[(formation, a), "weight"] for a in names], dtype=float)
                if not np.allclose(check, list(weights.values()), atol=1e-12, rtol=0):
                    raise ValueError("Posteriors differ across risk-map interventions; not a controlled action contrast.")
            hard = matrices[hard_arm_from_uniform(weights, names, hu)]
            mixed, _ = mix_scenarios_crn(matrices, weights, n_rows=base.n_scenarios, arm_uniforms=au, row_uniforms=ru)
            for action, P in (("hard", hard), ("mix", mixed)):
                maps = {"endogenous": (P, equal_weight_cvar(P, base.optimizer.cvar_alpha)),
                        "hier_budget_only": (P, HB), "hier_full_risk": (HQ, HB)}
                held = {label: engine._pretrade_for_state(states[score, action, label], formation, data, first_period=first) for label in MAPS}
                targets, outcomes = {}, {}
                ident = dict(master_seed=seed, formation_date=formation, outcome_date=outcome, score=score, action=action)
                for label, (Q, B) in maps.items():
                    cfg = refs[score, label]["config"]
                    target, diag = engine._optimize_or_fallback(P, held[label], cfg, first_period=first,
                        explicit_cvar_budget=None if label == "endogenous" else B,
                        risk_scenarios=Q if label == "hier_full_risk" else None)
                    perf = engine._execute_strategy(state=states[score, action, label], target=target, pretrade=held[label],
                        formation_date=formation, outcome_date=outcome, data=data, config=cfg, first_period=first)
                    key = (formation, f"{score}__{action}")
                    original = indexed[score, label]["returns"].loc[key]
                    errs = {col: abs(float(perf[col]) - float(original[col])) for col in
                            ("net_excess_return", "net_simple_return", "one_way_turnover", "transaction_cost", "wealth")}
                    old_diag = indexed[score, label]["diagnostics"].loc[key]
                    if "cvar_budget" in old_diag:
                        errs["cvar_budget"] = abs(float(diag["cvar_budget"]) - float(old_diag["cvar_budget"]))
                    for col, error in errs.items():
                        tol = 1e-5 if col == "wealth" else 1e-8
                        if not np.isfinite(error) or error > tol:
                            raise ValueError(f"Replay mismatch {score}/{action}/{label} seed={seed} {formation.date()}: {col} error={error:.3g}. Original outputs unchanged.")
                    gap = risk_gap(Q, B, target, base.optimizer.cvar_alpha)
                    if gap > 1e-6:
                        raise ValueError("Replayed action violates its own risk map.")
                    targets[label], outcomes[label] = target, perf
                    tables["reconciliation"].append({**ident, "risk_map": label, **{f"error_{k}": v for k, v in errs.items()}, "passed": True})
                    tables["replay_returns"].append({**ident, "risk_map": label, **perf, "true_cvar_slack": -gap,
                        "true_cvar_binding": abs(gap) <= 1e-8,
                        "lp_auxiliary_binding": bool(diag.get("cvar_budget_binding", False)),
                        "cvar_shadow": diag.get("cvar_budget_shadow_value_max_objective", np.nan)})
                for source, w in targets.items():
                    source_gap = risk_gap(*maps[source], w, base.optimizer.cvar_alpha)
                    for destination, (Q, B) in maps.items():
                        g = risk_gap(Q, B, w, base.optimizer.cvar_alpha)
                        tables["risk_evaluations"].append({**ident, "source_map": source, "evaluation_map": destination,
                            "signed_cvar_gap": g, "positive_cvar_breach": max(0.0, g), "infeasible": g > 1e-8,
                            "boundary_displacement": g - source_gap, "own_gap": source_gap, "evaluation_budget": B})
                reference_pretrade = held["hier_full_risk"]
                parents = {}
                for label, (Q, B) in maps.items():
                    w, value = parent_solution(P, Q, B, reference_pretrade, base, first)
                    net, _ = engine._counterfactual_net_excess_return(target=w, pretrade=reference_pretrade,
                        outcome_date=outcome, data=data, config=base, first_period=first)
                    parents[label] = (w, value, net)
                    tables["opportunity_values"].append({**ident, "risk_map": label, "parent_objective_value": value,
                        "counterfactual_net_excess_return": net, "common_pretrade_source": "hier_full_risk_same_action",
                        "parent_cvar_slack": -risk_gap(Q, B, w, base.optimizer.cvar_alpha)})
                gamma = base.utility_gamma
                for left, right in combinations(MAPS, 2):
                    for scope in ("recursive_implemented", "one_step_convex_parent"):
                        if scope == "recursive_implemented":
                            a, b = targets[left], targets[right]
                            ra, rb = outcomes[left]["net_excess_return"], outcomes[right]["net_excess_return"]
                            opportunity = np.nan
                        else:
                            a, va, ra = parents[left]
                            b, vb, rb = parents[right]
                            opportunity = vb - va
                        leftgap = risk_gap(*maps[left], a, base.optimizer.cvar_alpha)
                        rightgap = risk_gap(*maps[right], a, base.optimizer.cvar_alpha)
                        tables["displacement"].append({**ident, "left_map": left, "right_map": right, "scope": scope,
                            "allocation_displacement": allocation_distance(a, b),
                            "boundary_displacement_at_left": rightgap - leftgap,
                            "left_target_infeasible_under_right": rightgap > 1e-8,
                            "opportunity_displacement": opportunity,
                            "net_excess_return_difference": rb - ra,
                            "quadratic_utility_difference": (rb - gamma * rb * rb / 2) - (ra - gamma * ra * ra / 2)})
        if t == 0 or (t + 1) % 10 == 0 or t + 1 == len(calendar):
            progress(f"[seed {seed}] reconciled {t + 1}/{len(calendar)} formation months", flush=True)
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def seed_summary(frame: pd.DataFrame, group: list[str], metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average over dates within each seed, then form across-seed t intervals."""
    by_seed = frame.groupby(group + ["master_seed"], dropna=False)[metrics].mean().reset_index()
    rows = []
    for keys, sub in by_seed.groupby(group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for metric in metrics:
            v = sub[metric].dropna().astype(float)
            n = len(v)
            if not n:
                continue
            mean = float(v.mean())
            sd = float(v.std(ddof=1)) if n > 1 else np.nan
            half = float(stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            rows.append({**dict(zip(group, keys)), "metric": metric, "n_seeds": n, "mean": mean,
                         "seed_sd": sd, "ci_low": mean - half, "ci_high": mean + half})
    return by_seed, pd.DataFrame(rows)
