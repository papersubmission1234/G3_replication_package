"""Point-in-time market-state classification and fixed state controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    vix_column: str = "VIX"
    hy_oas_column: str = "HY OAS"
    source_units: str = "decimal"
    lag_months: int = 1
    stress_vix: float = 18.0
    crisis_vix: float = 25.0
    stress_hy_oas_bps: float = 350.0
    crisis_hy_oas_bps: float = 500.0

    def validate(self) -> None:
        if self.source_units not in {"decimal", "percent", "bps", "auto"}:
            raise ValueError("source_units must be decimal, percent, bps, or auto.")
        if self.lag_months < 0:
            raise ValueError("lag_months must be nonnegative.")
        if self.stress_vix >= self.crisis_vix:
            raise ValueError("stress_vix must be below crisis_vix.")
        if self.stress_hy_oas_bps >= self.crisis_hy_oas_bps:
            raise ValueError("stress HY OAS must be below crisis HY OAS.")


@dataclass
class RegimeResult:
    labels: pd.Series
    audit: pd.DataFrame
    metadata: dict[str, Any]


def _to_month_end(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _oas_to_bps(series: pd.Series, source_units: str) -> tuple[pd.Series, str]:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    units = source_units
    if units == "auto":
        med = float(values.abs().median())
        if med < 1.0:
            units = "decimal"
        elif med < 100.0:
            units = "percent"
        else:
            units = "bps"
    if units == "decimal":
        return values * 10_000.0, units
    if units == "percent":
        return values * 100.0, units
    if units == "bps":
        return values, units
    raise AssertionError(units)


def classify_regimes(
    macro_factors: pd.DataFrame,
    formation_dates: pd.DatetimeIndex | list[pd.Timestamp],
    config: RegimeConfig | None = None,
) -> RegimeResult:
    """Classify formation dates from lagged VIX and HY-OAS.
    
    The default lag of one month uses February values for March-end formation.
    Zero lag is a sensitivity for decisions after all month-end fields are known.
    """

    config = RegimeConfig() if config is None else config
    config.validate()
    frame = _to_month_end(macro_factors)
    missing = [
        col for col in (config.vix_column, config.hy_oas_column)
        if col not in frame.columns
    ]
    if missing:
        raise KeyError(
            f"The exact regime columns are missing: {missing}. Do not select "
            "the first generic OAS column."
        )
    formations = pd.DatetimeIndex(formation_dates).to_period("M").to_timestamp("M")
    source_dates = formations - pd.offsets.MonthEnd(config.lag_months)
    vix = pd.to_numeric(frame[config.vix_column], errors="coerce").reindex(source_dates)
    hy_bps, resolved_units = _oas_to_bps(
        frame[config.hy_oas_column], config.source_units
    )
    hy = hy_bps.reindex(source_dates)
    vix.index = formations
    hy.index = formations
    if vix.isna().any() or hy.isna().any():
        bad = pd.DataFrame({"vix": vix, "hy_oas_bps": hy})
        raise ValueError(
            "Missing lagged regime observations for formation dates:\n"
            + bad[bad.isna().any(axis=1)].head(12).to_string()
        )

    crisis = (vix > config.crisis_vix) | (hy > config.crisis_hy_oas_bps)
    stress = (~crisis) & (
        (vix > config.stress_vix) | (hy > config.stress_hy_oas_bps)
    )
    labels = pd.Series(
        np.select([crisis, stress], ["crisis", "stress"], default="normal"),
        index=formations,
        name="formation_regime",
    )
    audit = pd.DataFrame(
        {
            "formation_date": formations,
            "source_date": source_dates,
            "vix": vix.to_numpy(dtype=float),
            "hy_oas_bps": hy.to_numpy(dtype=float),
            "formation_regime": labels.to_numpy(),
            # delayed feedback uses the regime label saved at prediction time
            "update_regime": labels.to_numpy(),
        }
    ).set_index("formation_date")
    metadata = {
        "config": asdict(config),
        "resolved_hy_oas_units": resolved_units,
        "counts": labels.value_counts().to_dict(),
        "timing": "state is fixed at portfolio formation and reused at feedback update",
    }
    return RegimeResult(labels=labels, audit=audit, metadata=metadata)


def estimate_fixed_state_weights(
    arm_losses: pd.DataFrame,
    *,
    score: str = "post_cost_utility",
    development_end: str | pd.Timestamp = "2023-12-31",
    temperature: float = 1.0,
    floor: float = 0.02,
) -> dict[str, dict[str, float]]:
    """Estimate control weights from development losses, then freeze them.
    
    Average arm losses through development_end are converted to softmax weights
    for evaluation and holdout.
    """

    required = {"outcome_date", "regime", "score", "arm", "loss"}
    missing = required.difference(arm_losses.columns)
    if missing:
        raise ValueError(f"arm_losses is missing columns: {sorted(missing)}")
    frame = arm_losses.copy()
    frame["outcome_date"] = pd.to_datetime(frame["outcome_date"])
    cutoff = pd.Timestamp(development_end).to_period("M").to_timestamp("M")
    frame = frame[(frame["score"] == score) & (frame["outcome_date"] <= cutoff)]
    if frame.empty:
        raise ValueError("No development losses are available for fixed state weights.")
    result: dict[str, dict[str, float]] = {}
    for state, group in frame.groupby("regime", dropna=False):
        if pd.isna(state):
            continue
        means = group.groupby("arm")["loss"].mean().dropna().sort_index()
        if means.empty:
            continue
        centered = means - means.min()
        raw = np.exp(-float(temperature) * centered.to_numpy(dtype=float))
        raw = np.maximum(raw, float(floor))
        raw /= raw.sum()
        result[str(state)] = dict(zip(means.index.astype(str), raw.tolist()))
    if not result:
        raise ValueError("No state-specific weights could be estimated.")
    return result
