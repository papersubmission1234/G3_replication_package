"""Hash helpers for data, source trees, scenarios, manifests, and outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TREE_EXCLUDE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "cache",
    ".cache",
    "dist",
    "outputs",
    "work",
    "input_data",
}
DEFAULT_TREE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
DEFAULT_TREE_EXCLUDE_NAMES = {"MANIFEST_SHA256.txt"}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_dataframe(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("\x1f".join(map(str, frame.index)).encode("utf-8"))
    h.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    arr = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    h.update(arr.tobytes())
    return h.hexdigest()


def sha256_series(series: pd.Series) -> str:
    return sha256_dataframe(series.to_frame())


def combine_hashes(items: Iterable[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name, digest in sorted(items):
        h.update(name.encode("utf-8"))
        h.update(digest.encode("ascii"))
    return h.hexdigest()


def iter_tree_files(
    root: str | Path,
    *,
    exclude_dirs: set[str] | None = None,
    exclude_suffixes: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    """List files in sorted order, excluding named directory components.
    
    Editable-install directories ending in .egg-info are also excluded.
    """

    base = Path(root).resolve()
    dirs = DEFAULT_TREE_EXCLUDE_DIRS | set(exclude_dirs or ())
    suffixes = DEFAULT_TREE_EXCLUDE_SUFFIXES | set(exclude_suffixes or ())
    names = DEFAULT_TREE_EXCLUDE_NAMES | set(exclude_names or ())
    files: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        parts = set(relative.parts[:-1])
        if parts & dirs or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name in names or path.suffix in suffixes:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(base).as_posix())


def sha256_tree(
    root: str | Path,
    *,
    exclude_dirs: set[str] | None = None,
    exclude_suffixes: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> str:
    """Hash a source tree using relative paths and each file's SHA-256 digest."""

    base = Path(root).resolve()
    items = [
        (path.relative_to(base).as_posix(), sha256_file(path))
        for path in iter_tree_files(
            base,
            exclude_dirs=exclude_dirs,
            exclude_suffixes=exclude_suffixes,
            exclude_names=exclude_names,
        )
    ]
    return combine_hashes(items)
