"""Check saved settings before resuming or summarizing runs."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .configuration import experiment_config_from_mapping
from .experiment import ExperimentConfig
from .serialization import to_jsonable, write_json


def config_payload(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, ExperimentConfig):
        config = experiment_config_from_mapping(dict(config))
    payload = asdict(config)
    if payload.get("posterior_initial_weights") is None:
        arms = payload["arm_names"]
        payload["posterior_initial_weights"] = {arm: 1.0 / len(arms) for arm in arms}
    return to_jsonable(payload)


def config_differences(saved, requested, *, ignore: Iterable[str] = ()) -> list[str]:
    left, right = config_payload(saved), config_payload(requested)
    for key in ignore:
        left.pop(key, None)
        right.pop(key, None)
    differences = []

    def compare(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                compare(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
        elif a != b:
            differences.append(f"{prefix}: saved={a!r}, requested={b!r}")

    compare(left, right)
    return differences


def named_setting_errors(experiment_id: str, config: Mapping[str, Any]) -> list[str]:
    """Only check names that state a numerical setting explicitly."""
    name = str(experiment_id).replace("\\", "/").rstrip("/").split("/")[-1]
    optimizer = dict(config.get("optimizer", {}) or {})
    expected = []
    match = re.fullmatch(
        r"scenario_count_(\d+)(?:__common_risk)?", name
    )
    if match:
        expected.append(("n_scenarios", int(match[1]), config.get("n_scenarios")))
    match = re.fullmatch(r"cvar_alpha_(\d+_\d+)(?:_scenarios_(\d+))?", name)
    if match:
        expected.append(("cvar_alpha", float(match[1].replace("_", ".")), optimizer.get("cvar_alpha")))
        if match[2]:
            expected.append(("n_scenarios", int(match[2]), config.get("n_scenarios")))
    match = re.fullmatch(r"cost_(\d+)bps", name)
    if match:
        expected.append(("transaction_cost_bps", float(match[1]), optimizer.get("transaction_cost_bps")))
    match = re.fullmatch(r"position_cap_(\d+)pct", name)
    if match:
        expected.append(("max_weight", int(match[1]) / 100.0, optimizer.get("max_weight")))
    for prefix, field in (("adjustment_penalty_", "adjustment_penalty"),
                          ("mean_shrinkage_", "mean_shrinkage_fraction"),
                          ("posterior_forgetting_", "posterior_forgetting")):
        if name.startswith(prefix):
            value = float(name[len(prefix):].replace("_", "."))
            actual = optimizer.get(field) if field == "adjustment_penalty" else config.get(field)
            expected.append((field, value, actual))
    for prefix, field in (("estimation_window_", "training_months"),
                          ("posterior_calibration_", "posterior_calibration_months")):
        match = re.fullmatch(re.escape(prefix) + r"(\d+)m", name)
        if match:
            expected.append((field, int(match[1]), config.get(field)))
    errors = []
    for field, value, actual in expected:
        if actual is None or abs(float(actual) - value) > 1e-12:
            errors.append(f"{experiment_id}: name requires {field}={value}, manifest records {actual}")
    return errors


def checked_core_config(root: Path, requested: ExperimentConfig, seeds: Iterable[int], *, required: bool) -> ExperimentConfig:
    """Do not silently replace the requested design with an older core run."""
    found = None
    for seed in seeds:
        path = Path(root) / "core" / f"seed_{int(seed)}" / "run_manifest.json"
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Missing core reference: {path}")
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("master_seed") != int(seed):
            raise ValueError(f"Seed mismatch in {path}")
        config = saved.get("experiment_config", saved.get("config"))
        if not isinstance(config, dict):
            raise ValueError(f"Missing experiment configuration in {path}")
        differences = config_differences(config, requested, ignore=("save_weights",))
        if differences:
            raise ValueError(
                f"Core reference does not match --config: {path}\n"
                + "\n".join(differences)
                + "\nUse the matching configuration or rerun the core in a separate output root."
            )
        found = config
    return requested if found is None else experiment_config_from_mapping(found)


def merge_trial_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Keep earlier trial declarations when a subset is rerun."""
    current = to_jsonable(payload)
    if not Path(path).is_file():
        current["last_selected_experiments"] = [r["experiment_id"] for r in current.get("configurations", [])]
        write_json(path, current)
        return
    previous = json.loads(Path(path).read_text(encoding="utf-8"))
    key = lambda r: (r.get("family", ""), r["experiment_id"])
    records = {key(r): r for r in previous.get("configurations", [])}
    history = list(previous.get("configuration_history", []))
    for record in current.get("configurations", []):
        earlier = records.get(key(record))
        if earlier is not None and earlier != record:
            history.append(earlier)
        records[key(record)] = record
    result = {**previous, **current}
    result["last_selected_experiments"] = [r["experiment_id"] for r in current.get("configurations", [])]
    result["configurations"] = list(records.values())
    result["n_configurations"] = len(records)
    result["configuration_history"] = history
    for field in ("seeds", "requested_seeds"):
        if field in current:
            result["last_" + field] = current[field]
            result[field] = sorted(set(previous.get(field, [])) | set(current[field]))
    write_json(path, result)


def scan_run_settings(output_root: Path, *, expected_core: ExperimentConfig | None = None,
                      experiments: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Inspect manifests without loading input data or recalculating results."""
    root = Path(output_root)
    paths = (sorted(root.rglob("seed_*/run_manifest.json")) if experiments is None else
             sorted(p for e in experiments for p in (root / e).glob("seed_*/run_manifest.json")))
    records = []
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for path in paths:
        relative = path.relative_to(root)
        if any(part in {"cache", "raw_arms", "seed_runs"} for part in relative.parts):
            continue
        experiment_id = relative.parent.parent.as_posix()
        issues = []
        config = {}
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            config = saved.get("experiment_config", saved.get("config", {}))
            if not isinstance(config, dict) or not config:
                raise ValueError("No valid experiment configuration")
            seed = int(path.parent.name.removeprefix("seed_"))
            if saved.get("master_seed") != seed:
                issues.append(f"seed label {seed} differs from manifest {saved.get('master_seed')}")
            issues.extend(named_setting_errors(experiment_id, config))
            if experiment_id == "core" and expected_core is not None:
                issues.extend(config_differences(config, expected_core))
            declaration_path = path.parent.parent / "configuration.json"
            if declaration_path.is_file():
                declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
                declared = declaration.get("experiment_config", declaration.get("config", declaration))
                if isinstance(declared, dict) and "n_scenarios" in declared:
                    issues.extend("configuration.json: " + s for s in config_differences(config, declared))
            normalized = config_payload(config)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            issues.append(str(exc))
            if not isinstance(config, dict):
                config = {}
            normalized = {}
        record = {"experiment_id": experiment_id, "manifest": str(path),
                  "n_scenarios": config.get("n_scenarios"),
                  "cvar_alpha": config.get("optimizer", {}).get("cvar_alpha"),
                  "issues": issues}
        records.append(record)
        groups.setdefault(experiment_id, []).append((record, normalized))
    for group in groups.values():
        signatures = {json.dumps(config, sort_keys=True) for _, config in group if config}
        if len(signatures) > 1:
            for record, _ in group:
                record["issues"].append("different configurations across seeds in this experiment")
    for record in records:
        record["passed"] = not record["issues"]
    return records
