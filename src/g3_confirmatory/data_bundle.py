"""Load and validate the prepared study data with explicit timing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .asset_classes import normalize_asset_classes
from . import bayesian_factor_inputs2 as factor_inputs
from .eligibility import build_formation_outcome_map
from .hashing import sha256_dataframe, sha256_file, sha256_series
from .realized_returns import (
    RealizedReturnConfig, read_daily_adjusted_close, resolve_realized_returns,
)


@dataclass
class DataBundle:
    monthly_simple_returns: pd.DataFrame
    monthly_excess_returns: pd.DataFrame
    risk_free: pd.Series
    eligibility: pd.DataFrame
    asset_classes: pd.Series
    equity_factors: pd.DataFrame
    macro_factors: pd.DataFrame
    bond_factors: pd.DataFrame
    calendar: pd.DataFrame
    source_metadata: dict[str, Any]
    realized_simple_returns: pd.DataFrame | None = None
    realized_excess_returns: pd.DataFrame | None = None

    @property
    def formation_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.calendar["formation_date"])

    @property
    def outcome_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.calendar["outcome_date"])

    @property
    def outcome_simple_returns(self) -> pd.DataFrame:
        return (
            self.monthly_simple_returns
            if self.realized_simple_returns is None
            else self.realized_simple_returns
        )

    @property
    def outcome_excess_returns(self) -> pd.DataFrame:
        return (
            self.monthly_excess_returns
            if self.realized_excess_returns is None
            else self.realized_excess_returns
        )


def _to_month_end(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _read_bool_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    def parse_bool(value: object) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(float(value) != 0.0)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
        raise ValueError(f"Unrecognized eligibility value: {value!r}")

    frame = frame.map(parse_bool)
    frame = _to_month_end(frame).astype(bool)
    frame.columns = frame.columns.astype(str)
    return frame


def _load_asset_classes(meta_path: Path, tickers: pd.Index) -> pd.Series:
    meta = pd.read_csv(meta_path)
    required = {"Ticker", "Investment Object"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Meta file missing columns: {sorted(missing)}")
    raw = (
        meta.dropna(subset=["Ticker"])
        .drop_duplicates("Ticker", keep="first")
        .set_index("Ticker")["Investment Object"]
    )
    raw.index = raw.index.astype(str)
    classes = normalize_asset_classes(raw, unknown_policy="error")
    return classes.reindex(tickers)


def apply_delisting_returns(
    simple_returns: pd.DataFrame,
    risk_free: pd.Series,
    events: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Fill missing effective-month returns from explicit delisting_return entries."""

    simple = simple_returns.copy()
    audit: list[dict[str, Any]] = []
    if events is not None and not events.empty and "delisting_return" in events.columns:
        for row in events.itertuples(index=False):
            ticker = str(getattr(row, "ticker"))
            date = pd.Timestamp(getattr(row, "effective_date")).to_period("M").to_timestamp("M")
            value = getattr(row, "delisting_return")
            if ticker not in simple.columns or date not in simple.index or not np.isfinite(value):
                continue
            old = simple.loc[date, ticker]
            if pd.isna(old):
                simple.loc[date, ticker] = float(value)
                audit.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "filled_simple_return": float(value),
                        "source": "delisting_events",
                    }
                )
    excess = simple.sub(risk_free.reindex(simple.index), axis=0)
    return simple, excess, audit


def load_study_data(
    *,
    input_dir: str | Path,
    eligibility_path: str | Path,
    outcome_start: str = "2019-02-28",
    outcome_end: str = "2025-12-31",
    delisting_events: pd.DataFrame | None = None,
    realized_return_config: RealizedReturnConfig | None = None,
    daily_adjusted_close_path: str | Path | None = None,
    raw_data: dict[str, Any] | None = None,
    factor_cache_dir: str | Path | None = None,
) -> DataBundle:
    """Read original inputs and the prepared eligibility matrix."""

    root = Path(input_dir).resolve()
    raw = dict(raw_data or {})
    meta_path = Path(raw.get("meta_csv") or root / "Meta.csv")
    workbook_path = Path(raw.get("macro_workbook") or root / "Portfolio Macro Data.xlsx")
    adjusted_path = Path(
        daily_adjusted_close_path or raw.get("adjusted_close_csv")
        or root / "adjusted_close_prices.csv"
    )
    raw["adjusted_close_csv"] = adjusted_path
    inputs = factor_inputs.build_bayesian_factor_inputs(
        input_dir=root, raw_data=raw, factor_cache_dir=factor_cache_dir
    )

    simple = _to_month_end(inputs["monthly_simple_returns"]).astype(float)
    excess = _to_month_end(inputs["monthly_excess_returns"]).astype(float)
    rf_df = _to_month_end(inputs["rf_df"]).astype(float)
    risk_free = rf_df[factor_inputs.RF_COL].rename("RF")
    equity = _to_month_end(inputs["equity_factors"]).astype(float)
    macro = _to_month_end(inputs["macro_factors_df"]).astype(float)
    bond = _to_month_end(inputs["bond_factors_df"]).astype(float)
    eligibility = _read_bool_matrix(Path(eligibility_path))

    common_tickers = (
        simple.columns.intersection(excess.columns)
        .intersection(eligibility.columns)
        .sort_values()
    )
    simple = simple.reindex(columns=common_tickers)
    excess = excess.reindex(columns=common_tickers)
    eligibility = eligibility.reindex(columns=common_tickers).fillna(False)
    asset_classes = _load_asset_classes(meta_path, common_tickers)
    missing_classes = asset_classes[asset_classes.isna()].index.tolist()
    if missing_classes:
        raise ValueError(
            "Tickers in the return/eligibility panels have no mapped asset class: "
            f"{missing_classes[:20]}{'...' if len(missing_classes) > 20 else ''}"
        )

    simple, excess, delisting_audit = apply_delisting_returns(
        simple, risk_free, delisting_events
    )
    calendar = build_formation_outcome_map(
        outcome_start=outcome_start,
        outcome_end=outcome_end,
    )

    realized_simple = simple.copy()
    realized_excess = excess.copy()
    missing_return_audit = pd.DataFrame()
    missing_return_metadata: dict[str, Any] = {
        "config": None,
        "n_missing_eligible_outcomes_detected": 0,
        "n_filled": 0,
        "n_unresolved": 0,
    }
    if realized_return_config is not None:
        daily_adjusted = read_daily_adjusted_close(adjusted_path)
        realized_simple, realized_excess, missing_return_audit, missing_return_metadata = (
            resolve_realized_returns(
                monthly_simple_returns=simple,
                risk_free=risk_free,
                eligibility=eligibility,
                calendar=calendar,
                daily_adjusted_close=daily_adjusted,
                config=realized_return_config,
            )
        )

    source_paths = {
        "meta": meta_path,
        "macro_workbook": workbook_path,
        "adjusted_close": adjusted_path,
        "eligibility": Path(eligibility_path),
    }
    file_hashes = {
        name: sha256_file(path)
        for name, path in source_paths.items()
        if path.exists()
    }
    source_metadata = {
        "input_dir": str(root),
        "source_file_hashes": file_hashes,
        "panel_hashes": {
            "monthly_simple_returns": sha256_dataframe(simple),
            "monthly_excess_returns": sha256_dataframe(excess),
            "realized_simple_returns": sha256_dataframe(realized_simple),
            "realized_excess_returns": sha256_dataframe(realized_excess),
            "risk_free": sha256_series(risk_free),
            "eligibility": sha256_dataframe(eligibility.astype(float)),
            "asset_classes": sha256_series(asset_classes.astype("category").cat.codes),
        },
        "delisting_return_fills": delisting_audit,
        "missing_outcome_return_policy": missing_return_metadata,
        "missing_outcome_return_audit": missing_return_audit.to_dict(orient="records"),
        "n_tickers": int(len(common_tickers)),
        "n_eligibility_columns": int(eligibility.shape[1]),
        "n_ever_eligible": int(eligibility.any(axis=0).sum()),
        "asset_class_counts": asset_classes.value_counts().to_dict(),
    }
    return DataBundle(
        monthly_simple_returns=simple,
        monthly_excess_returns=excess,
        risk_free=risk_free,
        eligibility=eligibility,
        asset_classes=asset_classes,
        equity_factors=equity,
        macro_factors=macro,
        bond_factors=bond,
        calendar=calendar,
        source_metadata=source_metadata,
        realized_simple_returns=realized_simple,
        realized_excess_returns=realized_excess,
    )


def training_dates(
    formation_date: pd.Timestamp,
    *,
    training_months: int,
    information_lag_months: int = 1,
) -> pd.DatetimeIndex:
    """Return exactly the completed monthly observations available at formation."""

    formation = pd.Timestamp(formation_date).to_period("M").to_timestamp("M")
    end = formation - pd.offsets.MonthEnd(information_lag_months)
    return pd.date_range(end=end, periods=training_months, freq="ME")


def validate_data_bundle(
    data: DataBundle,
    *,
    training_months: int = 36,
    information_lag_months: int = 1,
    require_complete_training_history: bool = True,
) -> pd.DataFrame:
    """Fail early on date, timing, class, and return-availability problems."""

    records: list[dict[str, Any]] = []
    for row in data.calendar.itertuples(index=False):
        formation = pd.Timestamp(row.formation_date)
        outcome = pd.Timestamp(row.outcome_date)
        dates = training_dates(
            formation,
            training_months=training_months,
            information_lag_months=information_lag_months,
        )
        eligible = (
            data.eligibility.loc[formation]
            if formation in data.eligibility.index
            else pd.Series(False, index=data.eligibility.columns)
        )
        tickers = eligible[eligible].index
        missing_training_dates = dates.difference(data.monthly_excess_returns.index)
        missing_outcome_date = outcome not in data.outcome_simple_returns.index
        equity_factor_panel = data.equity_factors.reindex(dates)
        bond_factor_panel = data.bond_factors.reindex(dates)
        macro_dates = pd.date_range(
            dates.min() - pd.offsets.MonthEnd(1), dates.max(), freq="ME"
        )
        macro_factor_panel = data.macro_factors.reindex(macro_dates)
        missing_equity_factor_rows = int(
            equity_factor_panel.isna().any(axis=1).sum()
        )
        missing_bond_factor_rows = int(
            bond_factor_panel.isna().any(axis=1).sum()
        )
        missing_macro_factor_rows = int(
            macro_factor_panel.isna().any(axis=1).sum()
        )
        complete_assets = 0
        if not missing_training_dates.empty:
            complete_assets = 0
        elif len(tickers):
            panel = data.monthly_excess_returns.reindex(index=dates, columns=tickers)
            complete_assets = int(panel.notna().all(axis=0).sum())
        missing_eligible_outcome_returns = 0
        if not missing_outcome_date and len(tickers):
            missing_eligible_outcome_returns = int(
                data.outcome_simple_returns.reindex(
                    index=[outcome], columns=tickers
                ).isna().sum(axis=1).iloc[0]
            )
        records.append(
            {
                "formation_date": formation,
                "outcome_date": outcome,
                "training_start": dates.min(),
                "training_end": dates.max(),
                "n_eligible": int(len(tickers)),
                "n_complete_training_assets": complete_assets,
                "n_missing_training_dates": int(len(missing_training_dates)),
                "n_missing_equity_factor_rows": missing_equity_factor_rows,
                "n_missing_bond_factor_rows": missing_bond_factor_rows,
                "n_missing_macro_factor_rows": missing_macro_factor_rows,
                "outcome_date_available": not missing_outcome_date,
                "n_missing_eligible_outcome_returns": missing_eligible_outcome_returns,
            }
        )

    audit = pd.DataFrame(records).set_index("formation_date")
    # the 35-row factor window allows one initial macro difference to be missing
    allowed_factor_missing_rows = 0 if require_complete_training_history else 1
    failures = audit[
        (audit["n_missing_training_dates"] > 0)
        | (audit["n_missing_equity_factor_rows"] > allowed_factor_missing_rows)
        | (audit["n_missing_bond_factor_rows"] > allowed_factor_missing_rows)
        | (audit["n_missing_macro_factor_rows"] > allowed_factor_missing_rows)
        | (~audit["outcome_date_available"])
        | (audit["n_missing_eligible_outcome_returns"] > 0)
        | ((audit["n_complete_training_assets"] == 0) if require_complete_training_history else False)
    ]
    if not failures.empty:
        first = failures.head(8).to_string()
        raise ValueError(
            "The requested evaluation calendar is not supported by the prepared "
            "data under the declared timing rule. Check the first training window, "
            "factor transformations, return availability, and configured minimum "
            "observations. First failing rows:\n" + first
        )
    return audit
