"""Account for missing returns on holdings selected at formation.

The default last_trade_to_cash rule uses the adjusted-price return through the
last observed quote, then cash through month-end. It does not use future
observations to remove assets from the formation universe. Alternative recovery
rules provide sensitivity checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

MissingOutcomePolicy = Literal[
    "last_trade_to_cash",
    "fixed_haircut_to_cash",
    "total_loss",
    "observed_only",
]


@dataclass(frozen=True)
class RealizedReturnConfig:
    policy: MissingOutcomePolicy = "last_trade_to_cash"
    fixed_haircut: float = 0.30
    accrue_cash_after_last_quote: bool = True
    max_formation_price_staleness_days: int = 10

    def validate(self) -> None:
        if self.policy not in {
            "last_trade_to_cash",
            "fixed_haircut_to_cash",
            "total_loss",
            "observed_only",
        }:
            raise ValueError(f"Unsupported missing-outcome policy: {self.policy}")
        if not 0.0 <= float(self.fixed_haircut) <= 1.0:
            raise ValueError("fixed_haircut must lie in [0, 1].")
        if int(self.max_formation_price_staleness_days) < 0:
            raise ValueError("max_formation_price_staleness_days must be nonnegative.")


def config_from_mapping(mapping: dict[str, Any] | None) -> RealizedReturnConfig:
    raw = {} if mapping is None else dict(mapping)
    config = RealizedReturnConfig(
        policy=str(raw.get("policy", "last_trade_to_cash")),
        fixed_haircut=float(raw.get("fixed_haircut", 0.30)),
        accrue_cash_after_last_quote=bool(
            raw.get("accrue_cash_after_last_quote", True)
        ),
        max_formation_price_staleness_days=int(
            raw.get("max_formation_price_staleness_days", 10)
        ),
    )
    config.validate()
    return config


def read_daily_adjusted_close(path: str | Path) -> pd.DataFrame:
    """Read the supplied wide adjusted-close panel, including decimal commas."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0, dtype=str, quotechar='"')

    def convert(value: object) -> float:
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip()
        if not text or text.lower() in {"no data", "nan", "none", "null"}:
            return np.nan
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return np.nan

    frame = frame.map(convert)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.isna().any():
        raise ValueError(f"Unparseable dates in {path}")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).last()
    frame.columns = frame.columns.astype(str)
    return frame.astype(float)


def _month_end_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _cash_growth(
    monthly_rf: float,
    *,
    formation_date: pd.Timestamp,
    cash_start_date: pd.Timestamp,
    outcome_date: pd.Timestamp,
    accrue: bool,
) -> float:
    if not accrue:
        return 1.0
    rf = float(monthly_rf) if np.isfinite(monthly_rf) else 0.0
    if rf <= -1.0:
        return 1.0
    total_days = max(1, int((outcome_date - formation_date).days))
    remaining_days = min(
        total_days,
        max(0, int((outcome_date - cash_start_date).days)),
    )
    return float((1.0 + rf) ** (remaining_days / total_days))


def resolve_realized_returns(
    *,
    monthly_simple_returns: pd.DataFrame,
    risk_free: pd.Series,
    eligibility: pd.DataFrame,
    calendar: pd.DataFrame,
    daily_adjusted_close: pd.DataFrame,
    config: RealizedReturnConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fill missing returns only for assets eligible at formation.
    
    Existing monthly returns are retained. Daily prices value holdings already
    selected; they do not change eligibility.
    """

    config = RealizedReturnConfig() if config is None else config
    config.validate()

    simple = _month_end_index(monthly_simple_returns).astype(float).copy()
    rf = _month_end_index(risk_free).astype(float)
    elig = _month_end_index(eligibility).fillna(False).astype(bool)
    daily = daily_adjusted_close.copy()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    daily.columns = daily.columns.astype(str)

    audit_records: list[dict[str, Any]] = []
    unresolved: list[tuple[pd.Timestamp, str]] = []

    for row in calendar.itertuples(index=False):
        formation = pd.Timestamp(row.formation_date).to_period("M").to_timestamp("M")
        outcome = pd.Timestamp(row.outcome_date).to_period("M").to_timestamp("M")
        if formation not in elig.index or outcome not in simple.index:
            continue
        eligible_tickers = elig.columns[elig.loc[formation].to_numpy(dtype=bool)]
        missing = [
            ticker
            for ticker in eligible_tickers
            if ticker in simple.columns and pd.isna(simple.at[outcome, ticker])
        ]
        for ticker in missing:
            record: dict[str, Any] = {
                "formation_date": formation,
                "outcome_date": outcome,
                "ticker": str(ticker),
                "policy": config.policy,
                "raw_monthly_return_missing": True,
            }
            if config.policy == "observed_only":
                unresolved.append((outcome, str(ticker)))
                record["status"] = "unresolved_observed_only"
                audit_records.append(record)
                continue
            if ticker not in daily.columns:
                unresolved.append((outcome, str(ticker)))
                record["status"] = "unresolved_ticker_absent_from_daily_panel"
                audit_records.append(record)
                continue

            series = pd.to_numeric(daily[ticker], errors="coerce").dropna()
            prior = series.loc[series.index <= formation]
            if prior.empty:
                unresolved.append((outcome, str(ticker)))
                record["status"] = "unresolved_no_formation_price"
                audit_records.append(record)
                continue
            start_date = pd.Timestamp(prior.index[-1])
            start_price = float(prior.iloc[-1])
            staleness = int((formation - start_date).days)
            record.update(
                {
                    "formation_quote_date": start_date,
                    "formation_price": start_price,
                    "formation_price_staleness_days": staleness,
                }
            )
            if (
                not np.isfinite(start_price)
                or start_price <= 0.0
                or staleness > config.max_formation_price_staleness_days
            ):
                unresolved.append((outcome, str(ticker)))
                record["status"] = "unresolved_stale_or_invalid_formation_price"
                audit_records.append(record)
                continue

            during = series.loc[(series.index > formation) & (series.index <= outcome)]
            if during.empty:
                last_date = formation
                last_price = start_price
                observed_gross = 1.0
                has_post_formation_quote = False
            else:
                last_date = pd.Timestamp(during.index[-1])
                last_price = float(during.iloc[-1])
                observed_gross = float(last_price / start_price)
                has_post_formation_quote = True

            month_rf = float(rf.get(outcome, 0.0))
            cash_growth = _cash_growth(
                month_rf,
                formation_date=formation,
                cash_start_date=last_date,
                outcome_date=outcome,
                accrue=config.accrue_cash_after_last_quote,
            )

            if config.policy == "total_loss":
                filled_return = -1.0
                recovery_fraction = 0.0
                cash_growth = 1.0
            else:
                recovery_fraction = (
                    1.0
                    if config.policy == "last_trade_to_cash"
                    else 1.0 - float(config.fixed_haircut)
                )
                filled_return = float(
                    observed_gross * recovery_fraction * cash_growth - 1.0
                )
                filled_return = max(-1.0, filled_return)

            later = series.loc[series.index > outcome]
            simple.at[outcome, ticker] = filled_return
            record.update(
                {
                    "status": "filled",
                    "last_quote_date_in_holding_period": last_date,
                    "last_quote_price": last_price,
                    "has_post_formation_quote": has_post_formation_quote,
                    "observed_gross_to_last_quote": observed_gross,
                    "recovery_fraction": recovery_fraction,
                    "cash_growth_after_last_quote": cash_growth,
                    "monthly_risk_free_return": month_rf,
                    "filled_simple_return": filled_return,
                    # quality flag only; leave eligibility and filled returns unchanged
                    "quotes_resume_after_outcome_ex_post": bool(not later.empty),
                    "first_quote_after_outcome_ex_post": (
                        pd.Timestamp(later.index[0]) if not later.empty else pd.NaT
                    ),
                }
            )
            audit_records.append(record)

    audit = pd.DataFrame(audit_records)
    excess = simple.sub(rf.reindex(simple.index), axis=0)
    metadata = {
        "config": asdict(config),
        "n_missing_eligible_outcomes_detected": int(len(audit_records)),
        "n_filled": int((audit.get("status") == "filled").sum()) if not audit.empty else 0,
        "n_unresolved": int(len(unresolved)),
        "unresolved_first_20": [
            {"outcome_date": str(date.date()), "ticker": ticker}
            for date, ticker in unresolved[:20]
        ],
        "lookahead_statement": (
            "No future activity is used to determine formation eligibility. "
            "Daily observations after formation are used only to measure the "
            "realized value of an already-held position."
        ),
    }
    return simple, excess, audit, metadata
