"""YAML configuration helpers for command-line runners."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any
import os
import re

import yaml

from .experiment import ExperimentConfig
from .optimizer import OptimizerConfig
from .regimes import RegimeConfig

_UNRESOLVED_ENV = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    """Expand environment variables and resolve a path, rejecting unset variables."""

    raw = os.path.expanduser(str(value))
    expanded = os.path.expandvars(raw)
    unresolved = _UNRESOLVED_ENV.findall(expanded)
    if unresolved:
        raise ValueError(
            "Unresolved environment variable(s) in path "
            f"{value!r}: {sorted(set(unresolved))}"
        )
    path = Path(expanded)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()


def _expand(value: str | Path) -> Path:
    return resolve_path(value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Top-level YAML configuration must be a mapping.")
    return payload


def _known_kwargs(cls, mapping: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    unknown = set(mapping).difference(names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(mapping)


def experiment_config_from_mapping(mapping: dict[str, Any] | None) -> ExperimentConfig:
    payload = {} if mapping is None else dict(mapping)
    optimizer_payload = payload.pop("optimizer", {}) or {}
    optimizer = OptimizerConfig(**_known_kwargs(OptimizerConfig, optimizer_payload))
    for key in (
        "score_names",
        "action_names",
        "arm_names",
        "included_asset_classes",
        "contextual_feature_columns",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    return ExperimentConfig(
        optimizer=optimizer,
        **_known_kwargs(ExperimentConfig, payload),
    )


def regime_config_from_mapping(mapping: dict[str, Any] | None) -> RegimeConfig:
    payload = {} if mapping is None else dict(mapping)
    return RegimeConfig(**_known_kwargs(RegimeConfig, payload))


def study_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths", {})
    required = {"input_dir", "eligibility_path", "output_root"}
    missing = required.difference(paths)
    if missing:
        raise ValueError(f"Configuration paths missing: {sorted(missing)}")
    return {
        key: _expand(value)
        for key, value in paths.items()
        if value is not None and str(value).strip()
    }


def configured_seeds(config: dict[str, Any]) -> list[int]:
    if "seeds" in config:
        values = [int(value) for value in config["seeds"]]
    else:
        start = int(config.get("seed_start", 20260322))
        count = int(config.get("n_seeds", 30))
        values = list(range(start, start + count))
    if not values:
        raise ValueError("At least one master seed is required.")
    if len(values) != len(set(values)):
        raise ValueError("Master seeds must be unique.")
    return values
