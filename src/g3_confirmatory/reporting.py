"""Summaries and table catalogs from saved experiment outputs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
import os
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import default_subperiod_metrics, summarize_metrics_across_seeds

ROOT = Path(__file__).resolve().parents[2]

SKIP_OUTPUT_PARTS = {
    "inference",
    "tables",
    "paper_tables",
    "figures",
    "paper_figures",
    "output_validation",
    "risk_diagnostics",
    "paper_bundle",
}


@dataclass(frozen=True)
class TableArtifact:
    table_id: str
    classification: str
    title: str
    purpose: str
    filename: str
    workbook_sheet: str
    required: bool
    available: bool
    n_rows: int
    n_columns: int
    source_files: str
    notes: str = ""


def _is_experiment_file(path: Path, output_root: Path) -> bool:
    try:
        rel = path.relative_to(output_root)
    except ValueError:
        return False
    return not any(part in SKIP_OUTPUT_PARTS for part in rel.parts)


def discover_seed_files(output_root: Path, filename: str) -> list[Path]:
    """Find seed-level files while excluding generated reporting directories."""

    return sorted(
        path
        for path in output_root.rglob(f"seed_*/{filename}")
        if _is_experiment_file(path, output_root)
    )


def experiment_id_for_seed_file(path: Path, output_root: Path) -> str:
    return path.parent.parent.relative_to(output_root).as_posix()


def read_seed_tree(
    output_root: Path,
    filename: str,
    *,
    parse_dates: Iterable[str] = (),
) -> pd.DataFrame:
    """Read all matching seed files and attach experiment/source identifiers."""

    frames: list[pd.DataFrame] = []
    for path in discover_seed_files(output_root, filename):
        header = pd.read_csv(path, nrows=0).columns
        dates = [column for column in parse_dates if column in header]
        frame = pd.read_csv(path, parse_dates=dates)
        if frame.empty:
            continue
        frame.insert(0, "experiment_id", experiment_id_for_seed_file(path, output_root))
        frame.insert(1, "source_file", str(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def all_experiment_metrics(
    output_root: Path,
    *,
    utility_gamma: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long per-seed metrics and across-seed summaries for every run."""

    returns = read_seed_tree(
        output_root,
        "returns.csv",
        parse_dates=("formation_date", "outcome_date"),
    )
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_seed_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for experiment_id, group in returns.groupby("experiment_id", sort=True):
        metrics = default_subperiod_metrics(group, utility_gamma=utility_gamma)
        metrics.insert(0, "experiment_id", experiment_id)
        by_seed_frames.append(metrics)
        summary = summarize_metrics_across_seeds(metrics.drop(columns="experiment_id"))
        summary.insert(0, "experiment_id", experiment_id)
        summary_frames.append(summary)
    return (
        pd.concat(by_seed_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def mean_ci(
    values: pd.Series | Iterable[float],
    confidence: float = 0.95,
    *,
    n_key: str = "n",
    min_observations: int = 0,
    drop_nonfinite: bool = True,
) -> dict[str, float | int]:
    """Mean and confidence interval across observations.

    Parameters
    ----------
    values :
        Series or any iterable of numbers.
    confidence :
        Two-sided confidence level (default 0.95).
    n_key :
        Name of the sample-count key in the returned dict. Use ``"n_seeds"``
        when the observations are one-per-seed (matches ``validate_interval``).
    min_observations :
        If ``0`` (default), empty input yields a NaN-populated dict — the
        lenient behavior used by ``holdings_diagnostics`` / ``posterior_diagnostics``.
        If ``> 0``, raise when fewer than this many finite values remain
        (use ``2`` when a t-based CI is required, matching manuscript reporting).
    drop_nonfinite :
        If ``True`` (default), drop NaN/inf observations. Set to ``False`` to
        raise when *any* non-finite value is present (the previous manuscript
        behavior — safest for one-observation-per-seed data).
    """
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(list(values))
    raw = pd.to_numeric(series, errors="coerce")

    if not drop_nonfinite and not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite observation present")

    raw = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not drop_nonfinite and not np.isfinite(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite observation present")
    x = raw.dropna().to_numpy(dtype=float)
    n = int(len(x))
    empty: dict[str, float | int] = {
        n_key: 0, "mean": np.nan, "sd": np.nan,
        "ci_low": np.nan, "ci_high": np.nan,
        "median": np.nan, "min": np.nan, "max": np.nan,
    }
    if n < max(min_observations, 1):
        if min_observations and n < min_observations:
            raise ValueError(
                f"Need at least {min_observations} finite observations; got {n}"
            )
        return empty

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    if n > 1 and np.isfinite(sd):
        half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * sd / np.sqrt(n))
    else:
        half = 0.0
    return {
        n_key: n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

def ci_text(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"

def holdings_diagnostics(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute concentration diagnostics from saved nonzero target weights."""

    weights = read_seed_tree(
        output_root,
        "weights.csv.gz",
        parse_dates=("formation_date", "outcome_date"),
    )
    if weights.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"master_seed", "strategy", "formation_date", "ticker", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"weights files are missing columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    keys = ["experiment_id", "master_seed", "strategy", "formation_date"]
    if "outcome_date" in weights.columns:
        keys.append("outcome_date")
    for group_key, group in weights.groupby(keys, sort=True):
        values = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
        values = values[values > 0.0].sort_values(ascending=False)
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, key_values))
        hhi = float(np.square(values.to_numpy(dtype=float)).sum())
        row.update(
            {
                "n_positions": int(len(values)),
                "max_weight": float(values.max()) if not values.empty else np.nan,
                "minimum_nonzero_weight": float(values.min()) if not values.empty else np.nan,
                "hhi": hhi,
                "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0.0 else np.nan,
                "top_5_weight": float(values.iloc[:5].sum()),
                "top_10_weight": float(values.iloc[:10].sum()),
                "weight_sum": float(values.sum()),
            }
        )
        records.append(row)
    by_date = pd.DataFrame(records)

    summary_records: list[dict[str, Any]] = []
    metric_columns = [
        "n_positions",
        "max_weight",
        "minimum_nonzero_weight",
        "hhi",
        "effective_number_of_holdings",
        "top_5_weight",
        "top_10_weight",
        "weight_sum",
    ]
    for keys2, group in by_date.groupby(["experiment_id", "strategy"], sort=True):
        experiment_id, strategy = keys2
        n_seeds = int(group["master_seed"].nunique())
        for metric in metric_columns:
            summary_records.append(
                {
                    "experiment_id": experiment_id,
                    "strategy": strategy,
                    "metric": metric,
                    "n_seeds": n_seeds,
                    **mean_ci(group[metric]),
                }
            )
    return by_date, pd.DataFrame(summary_records)


def optimizer_diagnostics_summary(
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = read_seed_tree(
        output_root,
        "optimizer_diagnostics.csv",
        parse_dates=("formation_date", "outcome_date"),
    )
    if diagnostics.empty:
        return pd.DataFrame(), pd.DataFrame()

    aliases = {
        "risk_scenario_var": ("risk_scenario_var", "scenario_var"),
        "risk_scenario_cvar": ("risk_scenario_cvar", "scenario_cvar"),
        "objective_scenario_var": ("objective_scenario_var",),
        "objective_scenario_cvar": ("objective_scenario_cvar",),
        "cvar_budget_slack": ("cvar_budget_slack",),
        "cvar_budget_shadow_value_max_objective": (
            "cvar_budget_shadow_value_max_objective",
        ),
        "cvar_effective_tail_count": ("cvar_effective_tail_count",),
        "n_risk_scenarios": ("n_risk_scenarios", "n_scenarios"),
        "n_objective_scenarios": ("n_objective_scenarios", "n_scenarios"),
        "n_assets": ("n_assets",),
        "stage2_n_assets": ("stage2_n_assets",),
        "cvar_budget": ("cvar_budget",),
        "objective_expected_return": (
            "objective_expected_return",
            "scenario_expected_return",
        ),
    }
    standardized = diagnostics.copy()
    for target, candidates in aliases.items():
        if target in standardized.columns:
            continue
        source = next((name for name in candidates if name in standardized.columns), None)
        standardized[target] = (
            pd.to_numeric(standardized[source], errors="coerce")
            if source is not None
            else np.nan
        )

    if "cvar_budget_binding" not in standardized.columns:
        slack = pd.to_numeric(standardized["cvar_budget_slack"], errors="coerce")
        standardized["cvar_budget_binding"] = slack.notna() & (slack <= 1e-8)
    if "fallback" not in standardized.columns:
        standardized["fallback"] = False
    if "solver_status" not in standardized.columns:
        standardized["solver_status"] = 0

    records: list[dict[str, Any]] = []
    metrics = list(aliases)
    for keys, group in standardized.groupby(["experiment_id", "strategy"], sort=True):
        experiment_id, strategy = keys
        base = {
            "experiment_id": experiment_id,
            "strategy": strategy,
            "n_rows": int(len(group)),
            "n_seeds": int(group["master_seed"].nunique()) if "master_seed" in group else np.nan,
            "solver_failure_count": int(
                (pd.to_numeric(group["solver_status"], errors="coerce").fillna(0) != 0).sum()
            ),
            "fallback_count": int(
                group["fallback"].astype("boolean").fillna(False).sum()
            ),
            "cvar_binding_fraction": float(
                group["cvar_budget_binding"]
                .astype("boolean")
                .fillna(False)
                .mean()
            ),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            base[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else np.nan
            base[f"median_{metric}"] = float(values.median()) if values.notna().any() else np.nan
        records.append(base)
    return standardized, pd.DataFrame(records)


def posterior_diagnostics(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    posterior = read_seed_tree(
        output_root,
        "posterior.csv",
        parse_dates=("formation_date", "outcome_date"),
    )
    if posterior.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"master_seed", "score", "arm", "weight", "formation_date"}
    missing = required.difference(posterior.columns)
    if missing:
        raise ValueError(f"posterior files are missing columns: {sorted(missing)}")
    final = (
        posterior.sort_values("formation_date")
        .groupby(["experiment_id", "master_seed", "score", "arm"], as_index=False)
        .tail(1)
    )
    records: list[dict[str, Any]] = []
    for keys, group in final.groupby(["experiment_id", "score", "arm"], sort=True):
        experiment_id, score, arm = keys
        records.append(
            {
                "experiment_id": experiment_id,
                "score": score,
                "arm": arm,
                **mean_ci(group["weight"]),
            }
        )
    return final, pd.DataFrame(records)


def run_manifest_index(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index run metadata and extract outcome-return fills."""

    manifests: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("seed_*/run_manifest.json")):
        if not _is_experiment_file(path, output_root):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            manifests.append(
                {
                    "experiment_id": experiment_id_for_seed_file(path, output_root),
                    "source_file": str(path),
                    "manifest_read_error": str(exc),
                }
            )
            continue
        experiment_id = experiment_id_for_seed_file(path, output_root)
        data = dict(payload.get("data", {}) or {})
        config = dict(payload.get("experiment_config", {}) or {})
        optimizer = dict(config.get("optimizer", {}) or {})
        calendar = dict(payload.get("calendar", {}) or {})
        manifests.append(
            {
                "experiment_id": experiment_id,
                "master_seed": payload.get("master_seed"),
                "runtime_seconds": payload.get("runtime_seconds"),
                "source_tree_sha256": payload.get("source_tree_sha256"),
                "package_version": payload.get("package_version"),
                "outcome_start": calendar.get("outcome_start"),
                "outcome_end": calendar.get("outcome_end"),
                "n_outcomes": calendar.get("n_outcomes"),
                "n_tickers": data.get("n_tickers"),
                "n_ever_eligible": data.get("n_ever_eligible"),
                "n_scenarios": config.get("n_scenarios"),
                "cvar_alpha": optimizer.get("cvar_alpha"),
                "max_weight": optimizer.get("max_weight"),
                "transaction_cost_bps": optimizer.get("transaction_cost_bps"),
                "information_lag_months": config.get("information_lag_months"),
                "arm_names": "|".join(map(str, config.get("arm_names", []))),
                "score_names": "|".join(map(str, config.get("score_names", []))),
                "action_names": "|".join(map(str, config.get("action_names", []))),
                "source_file": str(path),
            }
        )
        for record in data.get("delisting_return_fills", []) or []:
            row = {
                "experiment_id": experiment_id,
                "master_seed": payload.get("master_seed"),
                "source_file": str(path),
            }
            if isinstance(record, dict):
                row.update(record)
            else:
                row["record"] = str(record)
            fills.append(row)
    return pd.DataFrame(manifests), pd.DataFrame(fills)


def failure_index(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("failures.json")):
        if not _is_experiment_file(path, output_root):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append({"source_file": str(path), "read_error": str(exc)})
            continue
        records = payload if isinstance(payload, list) else payload.get("failures", [])
        if not records:
            rows.append(
                {
                    "experiment_id": path.parent.relative_to(output_root).as_posix(),
                    "source_file": str(path),
                    "status": "empty_failure_file",
                }
            )
            continue
        for record in records:
            row = {
                "experiment_id": path.parent.relative_to(output_root).as_posix(),
                "source_file": str(path),
            }
            if isinstance(record, dict):
                row.update(record)
            else:
                row["record"] = str(record)
            rows.append(row)
    return pd.DataFrame(rows)


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def table_sheet_name(table_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in table_id)
    return cleaned[:31]


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Write a presentation workbook; the individual CSVs remain authoritative."""

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    note_fill = PatternFill("solid", fgColor="FCE4D6")

    used: set[str] = set()
    for table_id, frame in tables.items():
        base = table_sheet_name(table_id)
        name = base
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{base[:27]}_{suffix}"
        used.add(name)
        sheet = workbook.create_sheet(name)
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "A2"
        if frame.empty:
            sheet["A1"] = "No completed output was available for this table."
            sheet["A1"].fill = note_fill
            sheet["A1"].font = Font(bold=True)
            continue
        for col_index, column in enumerate(frame.columns, start=1):
            cell = sheet.cell(row=1, column=col_index, value=str(column))
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row_index, row in enumerate(frame.itertuples(index=False, name=None), start=2):
            for col_index, value in enumerate(row, start=1):
                if pd.isna(value):
                    value = None
                elif isinstance(value, pd.Timestamp):
                    value = value.to_pydatetime()
                elif isinstance(value, np.generic):
                    value = value.item()
                cell = sheet.cell(row=row_index, column=col_index, value=value)
                cell.alignment = Alignment(vertical="top", wrap_text=False)
                if isinstance(value, float):
                    cell.number_format = "0.0000;[Red](0.0000);-"
        sheet.auto_filter.ref = sheet.dimensions
        for col_index, column in enumerate(frame.columns, start=1):
            values = [str(column)] + [
                "" if pd.isna(value) else str(value)
                for value in frame[column].head(250)
            ]
            width = min(55, max(10, max(len(value) for value in values) + 2))
            sheet.column_dimensions[get_column_letter(col_index)].width = width
        sheet.row_dimensions[1].height = 32
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def write_table_catalog(path: Path, artifacts: Iterable[TableArtifact]) -> pd.DataFrame:
    frame = pd.DataFrame([artifact.__dict__ for artifact in artifacts])
    write_csv(path, frame)
    return frame

def resolve(path: str | Path) -> Path:
    p = Path(os.path.expandvars(str(path))).expanduser()
    if p.is_absolute():
        return p.resolve()
    return p.resolve() if p.exists() else (ROOT / p).resolve()


def reporting_context(config: str | Path):
    from .configuration import configured_seeds, load_yaml, study_paths
    path = resolve(config)
    cfg = load_yaml(path)
    paths = study_paths(cfg)
    seeds = configured_seeds(cfg)
    cal = cfg.get("calendar", {})
    dates = pd.period_range(
        cal.get("outcome_start", "2019-02-28"),
        cal.get("outcome_end", "2025-12-31"), freq="M",
    ).to_timestamp("M")
    if len(dates) != int(cal.get("expected_outcomes", 83)):
        raise ValueError("Configured outcome endpoints/count disagree")
    return path, cfg, paths, seeds, dates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [{"path": str(p.resolve()), "sha256": sha256(p)}
            for p in sorted(set(paths), key=str)]


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = {"method": "gzip", "mtime": 0} if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def checked_read(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source: {path}")
    frame = pd.read_csv(path)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Empty source: {path}")
    return frame


def select_one(frame: pd.DataFrame, **keys: Any) -> pd.Series:
    selected = frame
    for name, value in keys.items():
        if name not in selected:
            raise ValueError(f"Missing selector column {name}")
        selected = selected.loc[selected[name].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one row for {keys}; found {len(selected)}")
    return selected.iloc[0].copy()


def validate_interval(row: pd.Series, expected_seeds: int,
                      mean_column: str = "mean") -> None:
    if float(row["n_seeds"]) != expected_seeds:
        raise ValueError(f"Expected {expected_seeds} seeds, found {row['n_seeds']}")
    mean, low, high = row[[mean_column, "ci_low", "ci_high"]].to_numpy(float)
    if not np.isfinite([mean, low, high]).all() or not low <= mean <= high:
        raise ValueError("Invalid or missing mean / confidence limits")