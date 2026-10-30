"""Strict JSON for manifests and diagnostics.

Non-finite numbers become null, and scientific scalar types are converted to
standard Python values.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Recursively convert *value* to a strict-JSON-compatible object."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    return str(value)


def dumps_json(payload: Any, *, indent: int = 2, sort_keys: bool = True) -> str:
    """Return standards-compliant JSON text with no NaN/Infinity tokens."""

    return json.dumps(
        to_jsonable(payload),
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def write_json(path: str | Path, payload: Any) -> None:
    """Write strict JSON, creating the parent directory if needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_json(payload) + "\n", encoding="utf-8")
