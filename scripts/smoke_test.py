#!/usr/bin/env python3
"""Bounded smoke checks for the documented G3 replication workflow.

mini: real numerical code on a small synthetic panel (default).
reuse: validate saved full-study outputs and export reports to a fresh folder.
inputs: run the actual input builder and validator in a fresh folder.
check: imports, README arguments, grids, theory, and reporting regression only.

This does not reproduce the paper. Existing study inputs/outputs are read only.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LIMITS = (
    "A smoke PASS covers only the checks listed in this report. It is not an "
    "83-month/30-seed rerun, a comparison with manuscript numbers or figures, "
    "or a test of every robustness, Shapley and displacement calculation."
)


def rooted(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def run_cli(filename, arguments):
    previous = sys.argv
    sys.argv = [str(ROOT / "scripts" / filename), *map(str, arguments)]
    try:
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise RuntimeError(f"{filename} exited with {exc.code}") from exc
    finally:
        sys.argv = previous


def family_ids(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["experiment_id"] for row in rows]
    if len(names) != 37 or len(names) != len(set(names)):
        raise ValueError("The declared main family must contain 37 unique experiment_id values.")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Family experiment paths must be relative to the output root.")
    return names


def readme_commands(path):
    commands = []
    in_block = False
    pending = ""
    guide = Path(path)
    if not guide.is_file():
        print(f"Skipping optional documentation check: {guide}")
        return []

    for line in guide.read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith(chr(96) * 3):
            in_block = not in_block
            continue
        if not in_block:
            continue
        line = line.strip()
        pending += line[:-1].strip() + " " if line.endswith("\\") else line
        if line.endswith("\\"):
            continue
        tokens = shlex.split(pending)
        pending = ""
        if (len(tokens) > 1 and tokens[0] in {"python", "python3", "py"}
                and tokens[1].startswith("scripts/") and tokens[1].endswith(".py")
                and Path(tokens[1]).name != Path(__file__).name):
            commands.append(tokens[1:])
    unique = list(dict.fromkeys(tuple(command) for command in commands))
    if not unique:
        raise ValueError(f"No Python script commands found in {path}")
    return unique


def common_checks(args, scratch):
    """Parse actual guide arguments, stopping before each script starts work."""
    import numpy as np
    import pandas as pd
    from g3_confirmatory.configuration import (
        configured_seeds, experiment_config_from_mapping, load_yaml, study_paths,
    )
    from g3_confirmatory.theory_checks import run_all_theory_checks
    from g3_confirmatory.reporting import resolve

    files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
    for path in files:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    cfg = load_yaml(args.config)
    experiment_config_from_mapping(cfg.get("experiment")).validate()
    seeds = configured_seeds(cfg)
    study_paths(cfg)
    family = family_ids(args.family_manifest)

    class ArgumentsParsed(BaseException):
        pass

    original_parse = argparse.ArgumentParser.parse_args

    def stop_after_parse(parser, values=None, namespace=None):
        original_parse(parser, values, namespace)
        raise ArgumentsParsed()

    commands = readme_commands(args.readme)
    checked = []
    for command in commands:
        path = rooted(command[0])
        if not path.is_file():
            raise FileNotFoundError(path)
        previous = sys.argv
        sys.argv = [str(path), *command[1:]]
        capture = io.StringIO()
        argparse.ArgumentParser.parse_args = stop_after_parse
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                try:
                    runpy.run_path(str(path), run_name="__main__")
                except ArgumentsParsed:
                    checked.append(list(command))
                except SystemExit as exc:
                    raise RuntimeError(f"Guide arguments failed: {command}\n{capture.getvalue()}") from exc
                else:
                    raise RuntimeError(f"No argument parser reached in {path}")
        finally:
            argparse.ArgumentParser.parse_args = original_parse
            sys.argv = previous
    print(f"Parsed {len(checked)} distinct guide commands; scientific execution stopped at argument parsing.")

    # Run the real data-free grid branches, including the guide's selected IDs.
    for command in commands:
        if Path(command[0]).name in {
            "05_run_robustness.py", "17_run_decision_shapley.py",
            "18_run_full_riskmap_decomposition.py",
        }:
            print("Grid:", " ".join(command))
            run_cli(Path(command[0]).name, [*command[1:], "--list"])

    checks = run_all_theory_checks()
    if checks.empty or "passed" not in checks or not checks["passed"].all():
        raise AssertionError("A supplied theory check failed.")
    checks.to_csv(scratch / "theory_checks.csv", index=False)

    summary = runpy.run_path(str(ROOT / "scripts/30_summarize_diagnostic_extensions.py"))
    mean, low, high = summary["ci"]([0.25])
    if mean != 0.25 or not np.isnan(low) or not np.isnan(high):
        raise AssertionError("Script 30: a single seed must retain its mean and have missing CI limits.")
    fixture = pd.DataFrame({
        "family": ["test"] * 4,
        "metric": ["excess_sharpe", "excess_sharpe", "annualized_excess_mean", "annualized_excess_mean"],
        "mix_minus_hard": [0.1, 0.3, 1.0, 3.0],
    })
    grouped = summary["summarize"](fixture, ["family", "metric"], ["mix_minus_hard"])
    if set(grouped.metric) != {"excess_sharpe", "annualized_excess_mean"}:
        raise AssertionError("Script 30 overwrote the original performance-metric labels.")
    found = grouped.set_index("metric")
    if not np.isclose(found.loc["excess_sharpe", "mean"], 0.2) or not np.isclose(found.loc["annualized_excess_mean", "mean"], 2.0):
        raise AssertionError("Script 30 returned incorrect grouped means.")
    if not (grouped.n_seeds.eq(2).all() and (grouped.ci_low < grouped["mean"]).all()
            and (grouped.ci_high > grouped["mean"]).all()):
        raise AssertionError("Script 30 did not retain ordinary multi-seed intervals.")
    ordinary = summary["summarize"](fixture, ["family"], ["mix_minus_hard"])
    if ordinary.metric.tolist() != ["mix_minus_hard"]:
        raise AssertionError("Script 30 lost its value-column label for ordinary summaries.")
    relative = Path("work/smoke_tests") / scratch.name / "not_created"
    if resolve(relative) != (ROOT / relative).resolve():
        raise AssertionError("The reporting relative-output-path helper failed.")
    details = {
        "source_files_compiled": len(files), "guide_commands_parsed": checked,
        "declared_experiments": len(family), "configured_seeds": seeds,
        "theory_checks_passed": int(len(checks)), "script30_regression": "passed",
        "reporting_relative_path": "passed",
        "raw_inputs_loaded": False, "full_experiments_executed": False,
    }
    print(f"PASS: syntax, imports/arguments, grids, {len(checks)} theory checks, script 30 regression.")
    return details


def saved_coverage(args):
    from g3_confirmatory.configuration import configured_seeds, load_yaml
    cfg = load_yaml(args.config)
    seeds = configured_seeds(cfg)
    names = family_ids(args.family_manifest)
    exporter = runpy.run_path(str(ROOT / "scripts/33_export_manuscript_tables.py"))
    # Use the real exporter to select its Table 10 experiment paths.
    cells = exporter["table10_cells"]()
    extra = [cell["experiment_id"] for cell in cells if cell["experiment_id"] not in names]
    experiments = list(dict.fromkeys([*names, *extra]))
    missing = []
    for name in experiments:
        folder = args.output_root / name
        actual = {p.name for p in folder.glob("seed_*") if p.is_dir()}
        expected = {f"seed_{seed}" for seed in seeds}
        if actual != expected:
            missing.append(f"{name}: missing seeds {sorted(expected-actual)}, unexpected {sorted(actual-expected)}")
        for seed in seeds:
            for filename in ("returns.csv", "run_manifest.json"):
                path = folder / f"seed_{seed}" / filename
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Saved coverage is incomplete:\n" + "\n".join(missing[:20])
                                + (f"\n... {len(missing)-20} more" if len(missing) > 20 else ""))
    print(f"PASS: {len(experiments)} experiment folders x {len(seeds)} seeds have returns and manifests.")
    return {"experiments": len(experiments), "seeds": len(seeds), "coverage_only": True}


def prepare_input_validation(args, scratch):
    """Keep the study's validation criteria; redirect only its input/work paths."""
    import copy
    import yaml
    original = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    work = scratch / "input_check"
    generated = yaml.safe_load((work / "config/generated_study_config.yaml").read_text(encoding="utf-8"))
    validation = copy.deepcopy(original)
    for name in ("input_dir", "eligibility_path", "output_root"):
        validation["paths"][name] = generated["paths"][name]
    validation["paths"]["factor_cache_dir"] = str(work / "data/factor_cache")
    for name in ("meta_csv", "macro_workbook", "volume_csv", "adjusted_close_csv",
                 "ff5_plus_momentum_csv", "universe_tickers_file", "oracle_delisting_diagnostic_csv"):
        if name in generated.get("raw_data", {}):
            validation.setdefault("raw_data", {})[name] = generated["raw_data"][name]
    # In particular, retain the original expected_universe_size (470). The
    # builder's generated config sets this to its observed size, which would
    # otherwise hide a wrong input universe from this smoke check.
    target = work / "config/smoke_validation.yaml"
    target.write_text(yaml.safe_dump(validation, sort_keys=False), encoding="utf-8")
    return {"validation_config": str(target), "scientific_settings": "retained from the supplied config"}


def stage_figure_inputs(args, scratch):
    """Feed freshly exported script-27 tables into script 34."""
    staged = scratch / "report_inputs"
    needed = [
        "decompositions/full_risk_map/summary/action_channel_decomposition_summary.csv",
        "decompositions/decision_shapley/summary/effective_action_weight_summary.csv",
        "risk_diagnostics/daily_risk_metric_summary.csv",
    ]
    shapley = "decompositions/decision_shapley/summary/decision_shapley_contribution_summary.csv"
    if not (args.output_root / shapley).is_file():
        shapley = "decompositions/decision_shapley/summary/shapley_contribution_summary.csv"
    needed.append(shapley)
    for relative in needed:
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output_root / relative, destination)
    return {"copied_summary_files": needed, "fresh_script27_table_used": True}


def computational_smoke(root: Path, scratch: Path, config_path: Path) -> dict:
    """Run production numerical code on synthetic data in a fresh directory.

    This is an execution smoke, not a replication test or a scientific result.
    The caller must run in a subprocess with a hard timeout and a __main__ guard.
    """
    import importlib.util
    import json
    import sys
    import time
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    root, scratch = root.resolve(), scratch.resolve()
    started = time.monotonic()
    destination = scratch / "synthetic_mini"
    if destination.exists():
        raise FileExistsError(f"Synthetic smoke directory already exists: {destination}")
    destination.mkdir(parents=True)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from g3_confirmatory.configuration import load_yaml, experiment_config_from_mapping
    from g3_confirmatory.data_bundle import DataBundle
    from g3_confirmatory.hierarchical_model import DEFAULT_MACRO_COLUMNS, DEFAULT_EQUITY_STYLE_COLUMNS
    from g3_confirmatory.runner import ExperimentJob, run_experiment_jobs
    from g3_confirmatory.arm_cache import arm_data_fingerprint
    from g3_confirmatory.experiment import write_seed_result
    from g3_confirmatory.metrics import metrics_by_seed_strategy

    def load_script(filename):
        name = "g3_smoke_" + Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / filename)
        if spec is None or spec.loader is None:
            raise ImportError(filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Keep the actual 36-month estimators and optimizer settings. Only the
    # fixture's cross-section, output calendar, scenario count and calibration
    # count are small. A shared macro shock makes a coherent synthetic panel.
    rng = np.random.default_rng(16427)
    n_assets = 10
    dates = pd.period_range("2015-12", periods=42, freq="M").to_timestamp("M")
    tickers = pd.Index([f"SMOKE{i:02d}" for i in range(n_assets)])
    equity = pd.DataFrame(rng.normal(.001, .015, (len(dates), 6)),
                          index=dates, columns=DEFAULT_EQUITY_STYLE_COLUMNS)
    bond = pd.DataFrame(rng.normal(.001, .005, (len(dates), 1)),
                        index=dates, columns=["duration"])
    macro = pd.DataFrame(1 + np.cumsum(rng.normal(0, .01, (len(dates), len(DEFAULT_MACRO_COLUMNS))), axis=0),
                         index=dates, columns=DEFAULT_MACRO_COLUMNS)
    macro["VIX"] = 20 * np.exp(rng.normal(0, .05, len(dates)))
    shared = 4 * macro["inflation"].diff().fillna(0).to_numpy()[:, None]
    values = (equity.to_numpy() @ rng.normal(0, .15, (6, n_assets))
              + shared * np.linspace(.7, 1.3, n_assets)[None, :]
              + rng.normal(.003, .008, (len(dates), n_assets)))
    excess = pd.DataFrame(values, index=dates, columns=tickers)
    risk_free = pd.Series(.001, index=dates, name="RF")
    simple = excess.add(risk_free, axis=0)
    formations, outcomes = dates[36:40], dates[37:41]
    calendar = pd.DataFrame({"formation_date": formations, "outcome_date": outcomes})
    eligible = pd.DataFrame(True, index=formations, columns=tickers)
    classes = pd.Series(["equity"] * 6 + ["bond"] * 2 + ["commodities"] * 2, index=tickers)
    data = DataBundle(simple, excess, risk_free, eligible, classes, equity, macro, bond,
                      calendar, {"fixture": "synthetic_smoke", "real_study_inputs": False})
    base = replace(experiment_config_from_mapping(load_yaml(config_path)["experiment"]),
                   training_months=36, information_lag_months=0, min_observations=36,
                   n_scenarios=100, posterior_calibration_months=2,
                   fallback_policy="fail", save_weights=True)
    if tuple(base.arm_names) != ("M_FF5MOM", "M_HierMacro"):
        raise ValueError("Expected the study's two primary expert names")
    if not {"predictive_log", "post_cost_utility"}.issubset(base.score_names):
        raise ValueError("Expected both primary scoring rules")
    if not {"hard", "mix"}.issubset(base.action_names):
        raise ValueError("Expected both primary actions")
    seed = 20260322
    risk = replace(base, score_names=("post_cost_utility",),
                   common_cvar_budget_source="none", loss_cvar_budget_source="none",
                   loss_cvar_scenario_source="none", action_cvar_budget_source="M_HierMacro",
                   action_cvar_scenario_source="M_HierMacro")
    jobs = [
        ExperimentJob("smoke_core", base, destination / "core"),
        ExperimentJob("smoke_mean_zero", replace(base, hierarchical_mean_treatment="literal_zero_total_mean"),
                       destination / "mean_zero"),
        ExperimentJob("smoke_common_risk", risk, destination / "common_risk"),
    ]
    # With two workers and at least two pending jobs, runner takes its real
    # ProcessPoolExecutor(spawn) path even though only one seed is requested.
    run_experiment_jobs(data, jobs=jobs, seeds=[seed], workers=2, schedule="task",
                         start_method="spawn", numeric_threads_per_worker=1,
                         cache_dir=None, cache_mode="off", result_cache_dir=None,
                         result_cache_mode="off", overwrite=False, fail_fast=True)

    def check_history(folder, label, expect_arms, expect_strategies, expect_learning=True):
        history = pd.read_csv(folder / "returns.csv", parse_dates=["formation_date", "outcome_date"])
        posterior = pd.read_csv(folder / "posterior.csv", parse_dates=["formation_date"])
        diagnostics = pd.read_csv(folder / "optimizer_diagnostics.csv")
        hashes = pd.read_csv(folder / "scenario_hashes.csv", parse_dates=["formation_date"])
        if diagnostics.empty:
            raise AssertionError(f"{label}: no optimizer diagnostics were produced")
        if not set(formations).issubset(set(pd.to_datetime(diagnostics.formation_date))):
            raise AssertionError(f"{label}: optimizer diagnostics miss formation dates")
        if history.duplicated(["master_seed", "strategy", "outcome_date"]).any():
            raise AssertionError(f"{label}: duplicate history rows")
        if set(history.strategy) != set(expect_strategies):
            raise AssertionError(f"{label}: unexpected strategies {set(history.strategy)}")
        if not history.master_seed.eq(seed).all():
            raise AssertionError(f"{label}: unexpected master seed")
        for strategy, group in history.groupby("strategy"):
            if not pd.DatetimeIndex(group.sort_values("outcome_date").outcome_date).equals(outcomes):
                raise AssertionError(f"{label}/{strategy}: incomplete outcome calendar")
        required_values = ["net_simple_return", "net_excess_return", "wealth", "transaction_cost", "one_way_turnover"]
        if not np.isfinite(history[required_values].to_numpy(float)).all():
            raise AssertionError(f"{label}: nonfinite return/accounting values")
        if not diagnostics.solver_status.eq(0).all():
            raise AssertionError(f"{label}: unsuccessful optimizer")
        if "fallback_used" in history and history.fallback_used.astype(bool).any():
            raise AssertionError(f"{label}: unexpected fallback")
        required_diag = ["cvar_budget", "scenario_cvar", "max_weight", "weight_sum"]
        if not np.isfinite(diagnostics[required_diag].to_numpy(float)).all():
            raise AssertionError(f"{label}: nonfinite optimizer diagnostics")
        if not np.allclose(diagnostics.weight_sum, 1., atol=1e-10, rtol=0):
            raise AssertionError(f"{label}: optimized weights do not sum to one")
        if (diagnostics.max_weight > base.optimizer.max_weight + 1e-8).any():
            raise AssertionError(f"{label}: position cap violation")
        if (diagnostics.scenario_cvar > diagnostics.cvar_budget + 1e-7).any():
            raise AssertionError(f"{label}: CVaR constraint violation")
        if posterior.duplicated(["formation_date", "score", "arm"]).any():
            raise AssertionError(f"{label}: duplicate posterior rows")
        if not np.isfinite(posterior.weight).all() or (posterior.weight < 0).any():
            raise AssertionError(f"{label}: invalid posterior weights")
        grouped = posterior.groupby(["formation_date", "score"])
        if not grouped.size().eq(len(expect_arms)).all() or not np.allclose(grouped.weight.sum(), 1., atol=1e-12, rtol=0):
            raise AssertionError(f"{label}: incomplete or non-normalized posterior")
        last = posterior.loc[posterior.formation_date.eq(formations[-1])]
        largest_learning_change = float(np.max(np.abs(last.weight.to_numpy() - 1. / len(expect_arms))))
        if expect_learning and largest_learning_change <= 1e-8:
            raise AssertionError(f"{label}: no nontrivial learning after calibration")
        if hashes.duplicated(["formation_date", "arm"]).any() or len(hashes) != len(formations) * len(expect_arms):
            raise AssertionError(f"{label}: incomplete scenario hashes")
        if set(hashes.arm) != set(expect_arms):
            raise AssertionError(f"{label}: wrong scenario arms")
        metrics = metrics_by_seed_strategy(history)
        if not metrics.n_months.eq(4).all() or not np.isfinite(metrics[["excess_sharpe", "annualized_excess_mean", "maximum_drawdown"]].to_numpy()).all():
            raise AssertionError(f"{label}: invalid derived performance metrics")
        return {"history_rows": int(len(history)), "strategies": int(history.strategy.nunique()),
                "optimizer_rows": int(len(diagnostics)), "scenario_hash_rows": int(len(hashes)),
                "largest_final_posterior_change": largest_learning_change}, hashes, posterior

    core_strategies = ["predictive_log__hard", "predictive_log__mix", "post_cost_utility__hard",
                       "post_cost_utility__mix", "fixed_equal__mix", "standalone__M_FF5MOM",
                       "standalone__M_HierMacro", "benchmark__1N"]
    utility_strategies = [s for s in core_strategies if not s.startswith("predictive_log")]
    reports = {}
    for label, folder, strategies in [("core", destination / "core", core_strategies),
                                       ("mean_zero", destination / "mean_zero", core_strategies),
                                       ("common_risk", destination / "common_risk", utility_strategies)]:
        report, _, _ = check_history(folder / f"seed_{seed}", label, base.arm_names, strategies)
        reports[label] = report

    # Use the real covariance runner's existing adapter, which restores
    # engine.build_arms in a finally block, rather than substitute stub experts.
    covariance = load_script("29_run_covariance_regularization.py")
    covariance._initialize(data, {"base": base, "cache_dir": str(destination / "empty_covariance_cache"),
                                  "arm_data_hash": arm_data_fingerprint(data)})
    cov_arm = covariance._arm_for("sample")
    cov_hashes, cov_posteriors = [], []
    for label, budget, scenario in [("endogenous", "none", "none"),
                                     ("common_full", "M_Covariance", "M_Covariance")]:
        result = covariance._run_one(seed, "sample", cov_arm, label, budget, scenario)
        folder = destination / ("covariance_" + label) / f"seed_{seed}"
        write_seed_result(result, folder)
        strategies = [s.replace("M_HierMacro", "M_Covariance") for s in utility_strategies]
        report, hashes, posterior = check_history(folder, "covariance_" + label,
                                                   ("M_FF5MOM", "M_Covariance"), strategies)
        reports["covariance_" + label] = report
        cov_hashes.append(hashes.sort_values(["formation_date", "arm"])[["formation_date", "arm", "scenario_sha256"]].reset_index(drop=True))
        cov_posteriors.append(posterior.sort_values(["formation_date", "arm"])[["formation_date", "arm", "weight"]].reset_index(drop=True))
    if not cov_hashes[0].equals(cov_hashes[1]):
        raise AssertionError("Covariance risk maps changed supposedly shared model scenarios")
    if not cov_posteriors[0].equals(cov_posteriors[1]):
        raise AssertionError("Covariance risk maps changed supposedly shared utility posteriors")
    summary = {"passed": True, "kind": "synthetic numerical smoke; not study replication",
               "assets": n_assets, "outcomes": 4, "scenarios_per_expert": 100,
               "training_months": 36, "posterior_calibration_months": 2,
               "seeds": [seed], "real_input_data_used": False, "saved_study_outputs_used": False,
               "multiprocessing_start_method": "spawn", "workers": 2,
               "exact_result_cache": "off", "production_optimizers_succeeded": True,
               "elapsed_seconds": time.monotonic() - started, "histories": reports,
               "scope_limitations": ["Does not test raw-data ingestion or eligibility reconstruction",
                                     "Does not reproduce paper numbers or figures",
                                     "Does not exercise every robustness, Shapley or displacement path",
                                     "Uses sample covariance as the representative covariance estimator"],
               "output_directory": str(destination)}
    (destination / "synthetic_smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def worker(args):
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    scratch = args.scratch
    if args.worker == "common":
        result = common_checks(args, scratch)
    elif args.worker == "mini":
        result = computational_smoke(ROOT, scratch, args.config)
    elif args.worker == "coverage":
        result = saved_coverage(args)
    elif args.worker == "input_validation_config":
        result = prepare_input_validation(args, scratch)
    elif args.worker == "stage_figures":
        result = stage_figure_inputs(args, scratch)
    else:
        raise ValueError(args.worker)
    (scratch / f"{args.worker}_details.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")


def stop_process_tree(process):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/confirmatory.yaml")
    parser.add_argument("--mode", choices=("check", "mini", "reuse", "inputs"), default="mini")
    parser.add_argument("--timeout", type=float, default=120, help="Total wall-clock budget in seconds (default: 120)")
    parser.add_argument("--readme", type=Path, default=ROOT / "docs/RUN_PAPER_REVIEWERS.md")
    parser.add_argument("--family-manifest", type=Path, default=ROOT / "docs/expected_37_experiments.csv")
    parser.add_argument("--output-root", type=Path, help="Saved outputs to inspect in reuse mode")
    parser.add_argument("--input-dir", type=Path, help="Five-file input directory for inputs mode")
    parser.add_argument("--scratch-parent", type=Path, default=ROOT / "work/smoke_tests")
    parser.add_argument("--worker", choices=("common", "mini", "coverage", "input_validation_config", "stage_figures"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    args.config, args.readme, args.family_manifest = map(rooted, (args.config, args.readme, args.family_manifest))
    if args.worker:
        worker(args)
        return 0

    started = time.monotonic()
    # No study imports in the supervisor: it can enforce the deadline and stop
    # numerical worker processes even if an import or optimizer stalls.
    try:
        import yaml
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8-sig")) or {}
        configured_outputs = rooted(cfg["paths"]["output_root"])
        configured_inputs = rooted(cfg["paths"]["input_dir"])
        outputs = rooted(args.output_root) if args.output_root else configured_outputs
        inputs = rooted(args.input_dir) if args.input_dir else configured_inputs
    except Exception as exc:
        print(f"FAIL: cannot load study config/dependencies: {exc}", file=sys.stderr)
        return 1
    args.output_root = outputs
    parent = rooted(args.scratch_parent)
    protected = (outputs, configured_outputs, inputs, configured_inputs)
    if any(parent == path or path in parent.parents for path in protected):
        parser.error("--scratch-parent must be outside the study input and output trees")
    parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=parent)).resolve()
    (scratch / "logs").mkdir()
    print(f"Mode: {args.mode}; total budget: {args.timeout:g} seconds", flush=True)
    print(f"Logs and isolated artifacts: {scratch}", flush=True)
    print(LIMITS, flush=True)
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(scratch / "matplotlib"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    own = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
           "--readme", str(args.readme), "--family-manifest", str(args.family_manifest),
           "--output-root", str(outputs), "--scratch", str(scratch)]
    stages = [("code_readme_theory", [*own, "--worker", "common"])]

    def script(name, *arguments):
        return [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)]

    if args.mode == "mini":
        stages.append(("synthetic_numerical_run", [*own, "--worker", "mini"]))
    elif args.mode == "inputs":
        staged_work = scratch / "input_check"
        stages += [
            ("build_real_inputs", script("00_build_from_five_files.py", "--input-dir", inputs,
                                        "--work-root", staged_work, "--base-config", args.config)),
            ("preserve_validation_criteria", [*own, "--worker", "input_validation_config"]),
            ("validate_real_inputs", script("01_validate_inputs.py", "--config",
                                           staged_work / "config/smoke_validation.yaml")),
        ]
    elif args.mode == "reuse":
        metrics = outputs / "diagnostics/summary/extensions/extension_metrics_by_seed.csv"
        stages += [
            ("saved_seed_coverage", [*own, "--worker", "coverage"]),
            ("saved_run_settings", script("12_audit_run_settings.py", "--config", args.config,
                                         "--output-root", outputs, "--report", scratch / "run_settings.json")),
            ("displacement_figure_inputs", script("26_build_manuscript_revision_figures.py",
                                                 "--diagnostics-root", outputs / "diagnostics", "--validate-only")),
            ("support_table_export", script("27_export_manuscript_support_tables.py",
                                           "--output-root", outputs, "--tables-dir",
                                           scratch / "report_inputs/paper_tables/manuscript_support")),
            ("stage_fresh_figure_inputs", [*own, "--worker", "stage_figures"]),
            ("extension_table_export", script("32_export_extension_manuscript_tables.py",
                                             "--metrics", metrics, "--output-dir", scratch / "extension_tables")),
            ("covariance_figure_render", script("31_build_extension_manuscript_figure.py",
                                               "--metrics", metrics, "--output-dir", scratch / "figures31")),
            ("manuscript_figure_render", script("34_build_manuscript_figures.py",
                                               "--output-root", scratch / "report_inputs",
                                               "--output-dir", scratch / "figures34")),
            ("table10_export", script("33_export_manuscript_tables.py", "--config", args.config,
                                      "--tables", "10", "--output-dir", scratch / "table10")),
        ]
        if args.output_root != rooted(cfg["paths"]["output_root"]):
            # Table10 has a config-based input selector rather than --output-root.
            altered = dict(cfg)
            altered["paths"] = {**cfg["paths"], "output_root": str(outputs)}
            staged_config = scratch / "reporting_config.yaml"
            staged_config.write_text(yaml.safe_dump(altered, sort_keys=False), encoding="utf-8")
            stages[-1][1][stages[-1][1].index("--config") + 1] = str(staged_config)

    records = []
    deadline = started + args.timeout
    exit_code = 0
    for index, (label, command) in enumerate(stages):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            records.append({"stage": label, "status": "TIMEOUT_BEFORE_START", "command": command})
            exit_code = 2
            break
        log = scratch / "logs" / f"{index+1:02d}_{label}.log"
        print(f"[{index+1}/{len(stages)}] {label}", flush=True)
        tick = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != "nt"))
            try:
                return_code = process.wait(timeout=max(0, deadline - time.monotonic()))
                status = "PASS" if return_code == 0 else "FAIL"
            except subprocess.TimeoutExpired:
                stop_process_tree(process)
                return_code, status = None, "TIMEOUT"
            except KeyboardInterrupt:
                stop_process_tree(process)
                return_code, status = None, "INTERRUPTED"
            except BaseException:
                stop_process_tree(process)
                raise
        elapsed = time.monotonic() - tick
        records.append({"stage": label, "status": status, "seconds": round(elapsed, 3),
                        "returncode": return_code, "command": command, "log": str(log)})
        print(f"  {status} ({elapsed:.1f}s)", flush=True)
        if status != "PASS":
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]), flush=True)
            exit_code = 2 if status in {"TIMEOUT", "INTERRUPTED"} else 1
            break
    performed = {r["stage"] for r in records}
    not_run = [label for label, _ in stages if label not in performed]
    status = "PASS" if exit_code == 0 else ("INCOMPLETE" if exit_code == 2 else "FAIL")
    report = {
        "status": status, "mode": args.mode, "scope_limitations": LIMITS,
        "elapsed_seconds": round(time.monotonic() - started, 3), "budget_seconds": args.timeout,
        "config": str(args.config), "output_root_read": str(outputs),
        "isolated_artifacts": str(scratch), "stages": records, "not_run": not_run,
        "raw_input_checks_requested": args.mode == "inputs",
        "raw_input_checks_completed": any(r["stage"] == "validate_real_inputs" and r["status"] == "PASS" for r in records),
        "synthetic_numerical_checks_requested": args.mode == "mini",
        "saved_output_checks_requested": args.mode == "reuse",
    }
    target = scratch / "smoke_report.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{status}: {len(records)-sum(r['status'] != 'PASS' for r in records)}/{len(stages)} stages passed.", flush=True)
    print(f"Report: {target}", flush=True)
    if not_run:
        print("Not run:", ", ".join(not_run), flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
