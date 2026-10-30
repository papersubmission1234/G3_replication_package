"""Asset-class labels and style-factor policies for the eight-category universe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

RAW_ASSET_CLASS_MAP: dict[str, str] = {
    "Equities": "equity",
    "Bonds": "bond",
    "Mixed Assets": "mixed",
    "Cryptocurrencies": "crypto",
    "Precious Metals": "metals",
    "Money Market and Currencies": "cash_fx",
    "Real Estate": "real_estate",
    "Commodities": "commodities",
}

SUPPORTED_ASSET_CLASSES: tuple[str, ...] = (
    "equity",
    "bond",
    "mixed",
    "real_estate",
    "commodities",
    "crypto",
    "metals",
    "cash_fx",
)

DEFAULT_CLASS_STYLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "equity": ("equity",),
    "bond": ("bond",),
    "mixed": ("equity", "bond"),
    "real_estate": ("equity",),
    "commodities": (),
    "crypto": ("equity",),
    "metals": (),
    "cash_fx": ("bond",),
}


@dataclass(frozen=True)
class AssetClassPolicy:
    """How raw labels are normalized and style blocks are assigned."""

    raw_map: Mapping[str, str]
    class_style_families: Mapping[str, Sequence[str]]
    unknown_policy: str = "error"


def normalize_asset_classes(
    raw: pd.Series,
    *,
    raw_map: Mapping[str, str] | None = None,
    unknown_policy: str = "error",
) -> pd.Series:
    """Map ticker-indexed metadata labels to asset classes.
    
    raw_map can replace the default mapping. Unknown labels raise an error unless
    unknown_policy is "drop" (missing) or "other".
    """

    mapping = dict(RAW_ASSET_CLASS_MAP if raw_map is None else raw_map)
    out = raw.astype("string").str.strip().map(mapping)
    unknown_mask = raw.notna() & out.isna()
    if unknown_mask.any():
        labels = sorted(raw.loc[unknown_mask].astype(str).unique().tolist())
        if unknown_policy == "error":
            raise ValueError(
                "Unmapped asset-class labels were found. Add them to the map "
                f"instead of silently dropping them: {labels}"
            )
        if unknown_policy == "other":
            out.loc[unknown_mask] = "other"
        elif unknown_policy != "drop":
            raise ValueError("unknown_policy must be 'error', 'drop', or 'other'.")
    out.name = "asset_class"
    return out


def style_policy(name: str) -> dict[str, tuple[str, ...]]:
    """Return a named factor-family policy for hierarchical-model robustness."""

    key = name.strip().lower()
    if key == "default":
        return dict(DEFAULT_CLASS_STYLE_FAMILIES)
    if key == "macro_only":
        return {asset_class: () for asset_class in SUPPORTED_ASSET_CLASSES}
    if key == "equity_bond_only":
        return {
            asset_class: (
                ("equity",) if asset_class == "equity" else
                ("bond",) if asset_class == "bond" else
                ()
            )
            for asset_class in SUPPORTED_ASSET_CLASSES
        }
    if key == "all_equity_style":
        return {asset_class: ("equity",) for asset_class in SUPPORTED_ASSET_CLASSES}
    raise ValueError(
        f"Unknown class style policy {name!r}. Expected default, macro_only, "
        "equity_bond_only, or all_equity_style."
    )
