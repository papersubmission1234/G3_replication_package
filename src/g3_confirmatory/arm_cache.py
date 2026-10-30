"""Process-safe cache for raw predictive fits and scenarios.

Entries are saved before mean treatments, shrinkage, plug-in conversion, and
optimization. Keys include data, seed, formation date, arm, window rules,
scenario count, class/style policy, and forecast mode. Posterior and optimizer
settings are excluded because they do not affect the raw fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import copy
import hashlib
import json
import os
import platform
import sys
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import threadpoolctl

from . import __version__
from .arms import ArmResult
from .data_bundle import DataBundle
from .hashing import (
    combine_hashes, sha256_dataframe, sha256_file, sha256_series,
)


CACHE_SCHEMA_VERSION = 1
CACHE_MODES = {"off", "read", "readwrite", "refresh"}


def _stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=True,
    ).encode("utf-8")


def _digest_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def arm_data_fingerprint(data: DataBundle) -> str:
    """Hash only prepared inputs that can change predictive-arm generation."""

    metadata = data.source_metadata or {}
    verified_panel_hashes = {
        "monthly_excess_returns": sha256_dataframe(data.monthly_excess_returns),
        "eligibility": sha256_dataframe(data.eligibility.astype(float)),
        "asset_classes": sha256_series(
            data.asset_classes.astype("category").cat.codes
        ),
        "equity_factors": sha256_dataframe(data.equity_factors),
        "macro_factors": sha256_dataframe(data.macro_factors),
        "bond_factors": sha256_dataframe(data.bond_factors),
    }
    declared = dict(metadata.get("panel_hashes", {}) or {})
    arm_declared = {
        key: value
        for key, value in declared.items()
        if key not in {"realized_simple_returns", "realized_excess_returns"}
    }
    payload = {
        "source_file_hashes": metadata.get("source_file_hashes", {}),
        "declared_panel_hashes": arm_declared,
        "verified_panel_hashes": verified_panel_hashes,
        "n_tickers": metadata.get("n_tickers"),
        "n_ever_eligible": metadata.get("n_ever_eligible"),
        "asset_class_counts": metadata.get("asset_class_counts", {}),
        "calendar": data.calendar.assign(
            formation_date=pd.to_datetime(data.calendar["formation_date"]).astype(str),
            outcome_date=pd.to_datetime(data.calendar["outcome_date"]).astype(str),
        ).to_dict(orient="list"),
    }
    return _digest_payload(payload)


def data_bundle_fingerprint(data: DataBundle) -> str:
    """Hash all prepared inputs needed for a complete seed run.
    
    This includes simple returns and risk-free rates, as well as the excess returns
    and factors used to fit the arms.
    """

    payload = {
        "arm_data_fingerprint": arm_data_fingerprint(data),
        "monthly_simple_returns": sha256_dataframe(data.monthly_simple_returns),
        "realized_simple_returns": sha256_dataframe(data.outcome_simple_returns),
        "realized_excess_returns": sha256_dataframe(data.outcome_excess_returns),
        "risk_free": sha256_series(data.risk_free),
        "source_metadata": data.source_metadata or {},
    }
    return _digest_payload(payload)


def computational_environment_fingerprint() -> str:
    """Fingerprint numerical-library versions that can change study outputs."""

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    return _digest_payload(payload)


def arm_code_fingerprint() -> str:
    """Hash the code files that determine raw predictive-arm generation."""

    root = Path(__file__).resolve().parent
    names = (
        "arms.py",
        "hierarchical_model.py",
        "joint_factor.py",
        "asset_classes.py",
        "data_bundle.py",
        "randomness.py",
    )
    code_items = [(name, sha256_file(root / name)) for name in names]
    code_items.append(("computational_environment", computational_environment_fingerprint()))
    return combine_hashes(code_items)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    waits: int = 0
    corrupt_entries: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
            "waits": int(self.waits),
            "corrupt_entries": int(self.corrupt_entries),
        }

    def copy(self) -> "CacheStats":
        return CacheStats(**self.as_dict())

    def delta(self, earlier: "CacheStats") -> dict[str, int]:
        current = self.as_dict()
        previous = earlier.as_dict()
        return {key: current[key] - previous[key] for key in current}


class ArmResultCache:
    """Atomic, process-safe cache for raw :class:`ArmResult` objects."""

    def __init__(
        self,
        cache_dir: str | Path | None,
        *,
        mode: str = "readwrite",
        data_fingerprint: str,
        code_fingerprint: str | None = None,
        compression: int = 1,
        lock_timeout_seconds: float = 3600.0,
        stale_lock_seconds: float = 7200.0,
    ) -> None:
        mode = str(mode).lower()
        if mode not in CACHE_MODES:
            raise ValueError(f"Unknown cache mode {mode!r}; expected {sorted(CACHE_MODES)}")
        if cache_dir is None:
            mode = "off"
        self.mode = mode
        self.root = Path(cache_dir).resolve() if cache_dir is not None else None
        self.data_fingerprint = str(data_fingerprint)
        self.code_fingerprint = code_fingerprint or arm_code_fingerprint()
        self.compression = int(compression)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.stats = CacheStats()
        if self.root is not None and self.mode != "off":
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_cache_info()

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and self.root is not None

    def snapshot(self) -> CacheStats:
        return self.stats.copy()

    def _write_cache_info(self) -> None:
        if self.root is None:
            return
        path = self.root / "CACHE_INFO.json"
        if path.exists():
            return
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "package_version": __version__,
            "code_fingerprint": self.code_fingerprint,
            "description": (
                "Raw predictive-arm results cached before mean treatments, "
                "scenario shrinkage, scoring, action construction, and optimization."
            ),
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def key_payload(
        self,
        *,
        master_seed: int,
        formation_date: pd.Timestamp,
        arm_name: str,
        n_scenarios: int,
        generation_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "package_version": __version__,
            "code_fingerprint": self.code_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "master_seed": int(master_seed),
            "formation_date": str(pd.Timestamp(formation_date).date()),
            "arm_name": str(arm_name),
            "n_scenarios": int(n_scenarios),
            "generation_config": dict(generation_config),
        }

    def _entry_path(self, digest: str) -> Path:
        if self.root is None:
            raise RuntimeError("Cache is disabled.")
        return self.root / digest[:2] / digest[2:4] / f"{digest}.joblib"

    @staticmethod
    def _lock_path(entry: Path) -> Path:
        return entry.with_suffix(entry.suffix + ".lock")

    def _load(self, entry: Path, expected_digest: str) -> ArmResult | None:
        try:
            wrapper = joblib.load(entry)
            if not isinstance(wrapper, dict):
                raise TypeError("Cache wrapper is not a mapping.")
            if wrapper.get("digest") != expected_digest:
                raise ValueError("Cache digest mismatch.")
            if int(wrapper.get("schema", -1)) != CACHE_SCHEMA_VERSION:
                raise ValueError("Cache schema mismatch.")
            result = wrapper.get("result")
            if not isinstance(result, ArmResult):
                raise TypeError("Cached object is not an ArmResult.")
            self.stats.hits += 1
            return copy.deepcopy(result)
        except Exception:
            self.stats.corrupt_entries += 1
            try:
                entry.rename(entry.with_suffix(entry.suffix + f".corrupt.{uuid.uuid4().hex}"))
            except OSError:
                pass
            return None

    def _acquire_lock(
        self, lock: Path, entry: Path, *, accept_existing: bool = True
    ) -> int:
        started = time.monotonic()
        lock.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
                return fd
            except FileExistsError:
                if accept_existing and entry.exists():
                    self.stats.waits += 1
                    return -1
                try:
                    age = time.time() - lock.stat().st_mtime
                    if age > self.stale_lock_seconds:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() - started > self.lock_timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for arm-cache lock {lock}")
                self.stats.waits += 1
                time.sleep(0.10)

    def get_or_compute(
        self,
        payload: Mapping[str, Any],
        compute: Callable[[], ArmResult],
    ) -> tuple[ArmResult, bool, str]:
        """Return ``(result, cache_hit, digest)`` for ``payload``."""

        digest = _digest_payload(payload)
        if not self.enabled:
            self.stats.misses += 1
            return compute(), False, digest

        assert self.root is not None
        entry = self._entry_path(digest)
        if self.mode != "refresh" and entry.exists():
            loaded = self._load(entry, digest)
            if loaded is not None:
                return loaded, True, digest

        self.stats.misses += 1
        if self.mode == "read":
            return compute(), False, digest

        lock = self._lock_path(entry)
        fd = self._acquire_lock(
            lock, entry, accept_existing=self.mode != "refresh"
        )
        if fd == -1:
            loaded = self._load(entry, digest)
            if loaded is not None:
                return loaded, True, digest
            fd = self._acquire_lock(
                lock, entry, accept_existing=self.mode != "refresh"
            )

        try:
            if self.mode != "refresh" and entry.exists():
                loaded = self._load(entry, digest)
                if loaded is not None:
                    return loaded, True, digest
            result = compute()
            entry.parent.mkdir(parents=True, exist_ok=True)
            temp = entry.with_name(
                f".{entry.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            wrapper = {
                "schema": CACHE_SCHEMA_VERSION,
                "digest": digest,
                "payload": dict(payload),
                "result": result,
            }
            joblib.dump(wrapper, temp, compress=self.compression)
            os.replace(temp, entry)
            self.stats.writes += 1
            return copy.deepcopy(result), False, digest
        finally:
            try:
                if fd >= 0:
                    os.close(fd)
            finally:
                try:
                    lock.unlink(missing_ok=True)
                except OSError:
                    pass


def arm_generation_config(config: Any) -> dict[str, Any]:
    """Extract only configuration fields that affect raw arm generation."""

    return {
        "training_months": int(config.training_months),
        "information_lag_months": int(config.information_lag_months),
        "min_observations": int(config.min_observations),
        "max_missing_fraction": float(config.max_missing_fraction),
        "included_asset_classes": (
            list(config.included_asset_classes)
            if config.included_asset_classes is not None
            else None
        ),
        "class_style_policy": str(config.class_style_policy),
        "hierarchical_forecast_mode": str(config.hierarchical_forecast_mode),
    }
