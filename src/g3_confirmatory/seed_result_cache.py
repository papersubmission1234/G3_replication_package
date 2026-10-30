"""Cache complete seed runs with matching data and configuration.

The key includes seed, formation states, and fixed-state weights. Result files
are materialized with hard links where possible, or copied across filesystems.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping
import hashlib
import json
import os
import shutil
import time
import uuid

import pandas as pd

from .experiment import ExperimentConfig


RESULT_CACHE_SCHEMA_VERSION = 1
RESULT_CACHE_MODES = {"off", "read", "readwrite", "refresh"}


def _stable_digest(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalized_config(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    if payload.get("posterior_initial_weights") is None:
        arms = list(payload["arm_names"])
        payload["posterior_initial_weights"] = {
            arm: 1.0 / len(arms) for arm in arms
        }
    return payload


def _series_payload(series: pd.Series | None) -> Any:
    if series is None:
        return None
    copy = series.copy()
    copy.index = pd.to_datetime(copy.index).astype(str)
    return [(str(index), str(value)) for index, value in copy.items()]


def seed_run_signature(
    *,
    data_fingerprint: str,
    run_code_fingerprint: str,
    master_seed: int,
    config: ExperimentConfig,
    formation_states: pd.Series | None,
    fixed_state_weights: Mapping[str, Mapping[str, float]] | None,
) -> str:
    payload = {
        "schema": RESULT_CACHE_SCHEMA_VERSION,
        "data_fingerprint": data_fingerprint,
        "run_code_fingerprint": run_code_fingerprint,
        "master_seed": int(master_seed),
        "config": _normalized_config(config),
        "formation_states": _series_payload(formation_states),
        "fixed_state_weights": fixed_state_weights,
    }
    return _stable_digest(payload)


class SeedResultCache:
    def __init__(
        self,
        root: str | Path | None,
        *,
        mode: str = "readwrite",
        lock_timeout_seconds: float = 7200.0,
        stale_lock_seconds: float = 14400.0,
    ) -> None:
        mode = str(mode).lower()
        if mode not in RESULT_CACHE_MODES:
            raise ValueError(
                f"Unknown result-cache mode {mode!r}; expected {sorted(RESULT_CACHE_MODES)}"
            )
        if root is None:
            mode = "off"
        self.root = Path(root).resolve() if root is not None else None
        self.mode = mode
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.waits = 0
        if self.root is not None and self.mode != "off":
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None and self.mode != "off"

    def snapshot(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "waits": self.waits,
        }

    @staticmethod
    def delta(current: Mapping[str, int], earlier: Mapping[str, int]) -> dict[str, int]:
        return {key: int(current.get(key, 0)) - int(earlier.get(key, 0)) for key in current}

    def _entry(self, signature: str) -> Path:
        if self.root is None:
            raise RuntimeError("Seed result cache is disabled.")
        return self.root / signature[:2] / signature[2:4] / signature

    @staticmethod
    def _complete(path: Path) -> bool:
        return (path / "run_manifest.json").exists() and (path / "returns.csv").exists()

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _materialize(self, entry: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        for source in entry.rglob("*"):
            if source.is_file():
                relative = source.relative_to(entry)
                self._link_or_copy(source, target / relative)

    def _publish(self, source: Path, entry: Path) -> None:
        entry.parent.mkdir(parents=True, exist_ok=True)
        temp = entry.parent / f".{entry.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp.mkdir(parents=True, exist_ok=False)
        try:
            for path in source.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(source)
                    self._link_or_copy(path, temp / relative)
            os.replace(temp, entry)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

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
                if accept_existing and self._complete(entry):
                    self.waits += 1
                    return -1
                try:
                    if time.time() - lock.stat().st_mtime > self.stale_lock_seconds:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() - started > self.lock_timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for seed-result cache lock {lock}")
                self.waits += 1
                time.sleep(0.15)

    def get_or_run(
        self,
        signature: str,
        target: str | Path,
        run_and_write: Callable[[], None],
    ) -> bool:
        """Materialize or compute one seed directory.

        Returns ``True`` when the complete result came from the cache.
        """

        target = Path(target)
        if not self.enabled:
            self.misses += 1
            run_and_write()
            return False
        entry = self._entry(signature)
        if self.mode != "refresh" and self._complete(entry):
            self._materialize(entry, target)
            self.hits += 1
            return True

        self.misses += 1
        if self.mode == "read":
            run_and_write()
            return False

        lock = entry.with_suffix(".lock")
        fd = self._acquire_lock(
            lock, entry, accept_existing=self.mode != "refresh"
        )
        if fd == -1:
            self._materialize(entry, target)
            self.hits += 1
            return True
        try:
            if self.mode != "refresh" and self._complete(entry):
                self._materialize(entry, target)
                self.hits += 1
                return True
            if entry.exists():
                shutil.rmtree(entry, ignore_errors=True)
            run_and_write()
            if not self._complete(target):
                raise RuntimeError(
                    f"Seed run did not produce a complete output directory: {target}"
                )
            self._publish(target, entry)
            self.writes += 1
            return False
        finally:
            try:
                if fd >= 0:
                    os.close(fd)
            finally:
                lock.unlink(missing_ok=True)
