"""Formation-date eligibility from history, liquidity, and tradability.

With signal_lag_months=1, the history window is [t-36, t-1]. With zero lag
and formation_after_month_end_close=True, it is [t-35, t]. The primary design
uses the second convention: January 2019 formation, February 2019 first
outcome, and a February 2016--January 2019 initial window. This excludes the
potentially partial January 2016 return.

Volume.csv already contains dollar trading amounts, so volume_is_dollar=True
avoids multiplying them by prices again. The primary delisting screen uses
only announcements public at formation with an effective date in the coming
holding period. Full-future activity is an oracle sensitivity only.

An eligibility value at 2024-03-31 governs holdings for the April 2024 return.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
import hashlib
import json

import numpy as np
import pandas as pd

from .serialization import write_json

LiquidityMeasure = Literal["dollar_volume", "raw_volume"]

DelistingMode = Literal[
    "point_in_time",
    "current_only",
    "oracle_future_activity",
    "oracle_next_month_activity",
]


@dataclass(frozen=True)
class EligibilityConfig:
    history_months: int = 36
    liquidity_window_months: int = 36
    signal_lag_months: int = 1
    relative_liquidity_threshold: float = 1.0 / 10_000.0
    require_every_history_month_active: bool = True
    min_active_history_months: int | None = None
    require_current_activity: bool = True
    require_current_price: bool = True
    max_current_price_staleness_days: int = 10
    liquidity_measure: LiquidityMeasure = "dollar_volume"
    volume_is_dollar: bool = False
    price_proxy: str = "open_close_mean"
    delisting_mode: DelistingMode = "current_only"
    formation_after_month_end_close: bool = True

    def validate(self) -> None:
        if self.history_months <= 0 or self.liquidity_window_months <= 0:
            raise ValueError("history and liquidity windows must be positive.")
        if self.signal_lag_months < 0:
            raise ValueError("signal_lag_months must be nonnegative.")
        if self.signal_lag_months == 0 and not self.formation_after_month_end_close:
            raise ValueError(
                "signal_lag_months=0 is valid only when formation occurs after "
                "the month-end close, so month-t data are already observable."
            )
        if not 0.0 <= self.relative_liquidity_threshold <= 1.0:
            raise ValueError("relative_liquidity_threshold must lie in [0, 1].")
        if int(self.max_current_price_staleness_days) < 0:
            raise ValueError("max_current_price_staleness_days must be nonnegative.")
        if self.liquidity_measure not in {"dollar_volume", "raw_volume"}:
            raise ValueError("liquidity_measure must be dollar_volume or raw_volume.")
        if self.price_proxy not in {"open_close_mean", "adjusted_close"}:
            raise ValueError("price_proxy must be open_close_mean or adjusted_close.")
        if self.delisting_mode not in {
            "point_in_time",
            "current_only",
            "oracle_future_activity",
            "oracle_next_month_activity",
        }:
            raise ValueError(f"Unsupported delisting_mode: {self.delisting_mode}")


@dataclass
class EligibilityResult:
    eligibility: pd.DataFrame
    history_ok: pd.DataFrame
    liquidity_ok: pd.DataFrame
    current_activity_ok: pd.DataFrame
    delisting_ok: pd.DataFrame
    monthly_trading_amount: pd.DataFrame
    lagged_rolling_trading_amount: pd.DataFrame
    relative_liquidity_share: pd.DataFrame
    audit_by_date: pd.DataFrame
    audit_by_ticker: pd.DataFrame
    timing_map: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def monthly_dollar_volume(self) -> pd.DataFrame:
        """Backward-compatible alias; inspect metadata for the actual measure."""
        return self.monthly_trading_amount

    @property
    def lagged_rolling_dollar_volume(self) -> pd.DataFrame:
        """Backward-compatible alias; inspect metadata for the actual measure."""
        return self.lagged_rolling_trading_amount


def _month_end_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _daily_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.isna().any():
        raise ValueError(f"{label} contains unparseable dates.")
    out = out.sort_index().apply(pd.to_numeric, errors="coerce")
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    out.columns = out.columns.astype(str)
    return out


def month_end_after(formation_date: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(formation_date).to_period("M").to_timestamp("M")
    return date + pd.offsets.MonthEnd(1)


def build_formation_outcome_map(
    *,
    outcome_start: str | pd.Timestamp,
    outcome_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build formation and outcome dates for an inclusive outcome window.
    
    For February 2019--December 2025 outcomes, formation dates span
    January 2019--November 2025.
    """

    outcomes = pd.date_range(
        pd.Timestamp(outcome_start).to_period("M").to_timestamp("M"),
        pd.Timestamp(outcome_end).to_period("M").to_timestamp("M"),
        freq="ME",
    )
    formation = outcomes - pd.offsets.MonthEnd(1)
    result = pd.DataFrame(
        {
            "formation_date": formation,
            "outcome_date": outcomes,
            "holding_period_label": [
                f"{f:%Y-%m-%d} -> {o:%Y-%m-%d}" for f, o in zip(formation, outcomes)
            ],
        }
    )
    result.index = pd.RangeIndex(len(result), name="period_id")
    return result


def _align_daily_panels(
    volume: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    open_price: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    volume = _daily_index(volume, "volume")
    adjusted_close = _daily_index(adjusted_close, "adjusted_close")
    columns = volume.columns.intersection(adjusted_close.columns).sort_values()
    if columns.empty:
        raise ValueError("Volume and adjusted-close panels have no common tickers.")
    index = volume.index.union(adjusted_close.index).sort_values()
    volume = volume.reindex(index=index, columns=columns)
    adjusted_close = adjusted_close.reindex(index=index, columns=columns)

    open_aligned: pd.DataFrame | None = None
    if open_price is not None:
        open_aligned = _daily_index(open_price, "open_price")
        open_aligned = open_aligned.reindex(index=index, columns=columns)
    return volume, adjusted_close, open_aligned


def compute_monthly_trading_amount(
    volume: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    *,
    open_price: pd.DataFrame | None = None,
    liquidity_measure: LiquidityMeasure = "dollar_volume",
    volume_is_dollar: bool = False,
    price_proxy: str = "open_close_mean",
    max_current_price_staleness_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return monthly trading amounts and month-end tradability.
    
    raw_volume uses the values as supplied. For Volume.csv, use dollar_volume
    with volume_is_dollar=True because the file already contains dollar amounts.
    """

    volume, adjusted_close, open_price = _align_daily_panels(
        volume, adjusted_close, open_price
    )
    nonnegative_volume = volume.clip(lower=0.0)

    if liquidity_measure == "raw_volume":
        daily_amount = nonnegative_volume
        proxy_used = "raw input volume (literal notebook rule; not price-scaled)"
    elif volume_is_dollar:
        daily_amount = nonnegative_volume
        proxy_used = "input_declared_dollar_volume"
    else:
        if price_proxy == "open_close_mean" and open_price is not None:
            price = (open_price + adjusted_close) / 2.0
            price = price.where(price.notna(), adjusted_close)
            proxy_used = "mean(open, adjusted_close), adjusted_close fallback"
        else:
            price = adjusted_close
            proxy_used = "adjusted_close"
        daily_amount = nonnegative_volume * price.where(price > 0.0)

    monthly_amount = daily_amount.resample("ME").sum(min_count=1).fillna(0.0)
    monthly_share_volume = nonnegative_volume.resample("ME").sum(min_count=1).fillna(0.0)
    monthly_price = adjusted_close.resample("ME").last()

    # test the quote date so stale prices do not imply tradability
    valid_price = adjusted_close.notna() & (adjusted_close > 0.0)

    # timestamps avoid version-dependent datetime integer units
    quote_date_values = np.broadcast_to(
        adjusted_close.index.to_numpy(dtype="datetime64[ns]")[:, None],
        valid_price.shape,
    )
    quote_dates = pd.DataFrame(
        np.where(
            valid_price.to_numpy(dtype=bool),
            quote_date_values,
            np.datetime64("NaT", "ns"),
        ),
        index=adjusted_close.index,
        columns=adjusted_close.columns,
    )
    last_quote_date_monthly = quote_dates.resample("ME").max()
    month_end_values = np.broadcast_to(
        last_quote_date_monthly.index.to_numpy(dtype="datetime64[ns]")[:, None],
        last_quote_date_monthly.shape,
    )
    last_quote_values = last_quote_date_monthly.to_numpy(dtype="datetime64[ns]")
    has_quote = ~pd.isna(last_quote_values)
    staleness_days = np.full(last_quote_values.shape, np.inf, dtype=float)
    staleness_days[has_quote] = (
        month_end_values[has_quote] - last_quote_values[has_quote]
    ) / np.timedelta64(1, "D")
    fresh_price = pd.DataFrame(
        has_quote & (staleness_days <= int(max_current_price_staleness_days)),
        index=last_quote_date_monthly.index,
        columns=last_quote_date_monthly.columns,
    )
    current_activity = (
        (monthly_share_volume > 0.0)
        & monthly_price.notna()
        & (monthly_price > 0.0)
        & fresh_price
    )

    meta = {
        "liquidity_measure": liquidity_measure,
        "volume_is_dollar": bool(volume_is_dollar),
        "price_proxy_used": proxy_used,
        "daily_start": str(daily_amount.index.min().date()),
        "daily_end": str(daily_amount.index.max().date()),
        "n_tickers": int(len(daily_amount.columns)),
        "max_current_price_staleness_days": int(max_current_price_staleness_days),
    }
    return monthly_amount, current_activity.astype(bool), meta


def compute_monthly_dollar_volume(
    volume: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    *,
    open_price: pd.DataFrame | None = None,
    volume_is_dollar: bool = False,
    price_proxy: str = "open_close_mean",
    max_current_price_staleness_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compatibility wrapper for the confirmatory dollar-volume measure."""
    return compute_monthly_trading_amount(
        volume,
        adjusted_close,
        open_price=open_price,
        liquidity_measure="dollar_volume",
        volume_is_dollar=volume_is_dollar,
        price_proxy=price_proxy,
        max_current_price_staleness_days=max_current_price_staleness_days,
    )


def read_delisting_events(path: str | Path | None) -> pd.DataFrame:
    """Read ticker, announcement_date, and effective_date for each event.
    
    Extra columns, including delisting_return and source, are retained.
    Multiple events per ticker are allowed.
    """

    if path is None:
        return pd.DataFrame(columns=["ticker", "announcement_date", "effective_date"])
    frame = pd.read_csv(path)
    required = {"ticker", "announcement_date", "effective_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Delisting event file is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    for column in ["announcement_date", "effective_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if frame[["announcement_date", "effective_date"]].isna().any().any():
        raise ValueError("Delisting event dates contain missing/unparseable values.")
    invalid = frame["announcement_date"] > frame["effective_date"]
    if invalid.any():
        raise ValueError("A delisting announcement date occurs after its effective date.")
    return frame.sort_values(["ticker", "effective_date", "announcement_date"])


def infer_terminal_delisting_events(
    volume: pd.DataFrame,
    adjusted_close: pd.DataFrame,
) -> pd.DataFrame:
    """Infer terminal disappearances from the complete activity sample.
    
    This oracle rule flags a ticker's last active month when every later observed
    month has no valid activity. That last active month is itself ineligible.
    The inferred dates are not point-in-time announcements.
    """
    _, current_activity, _ = compute_monthly_trading_amount(
        volume,
        adjusted_close,
        liquidity_measure="raw_volume",
        price_proxy="adjusted_close",
    )
    current_activity = _month_end_index(current_activity).astype(bool)
    if current_activity.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "announcement_date", "effective_date",
                "last_active_month", "first_inactive_month",
                "months_observed_after_last_active", "oracle_approximation",
                "source", "note",
            ]
        )

    sample_end = current_activity.index.max()
    rows: list[dict[str, Any]] = []
    for ticker in current_activity.columns:
        active = current_activity.index[current_activity[ticker].to_numpy()]
        if len(active) == 0:
            continue
        last_active = pd.Timestamp(active.max())
        if last_active >= sample_end:
            continue
        later = current_activity.loc[current_activity.index > last_active, ticker]
        if later.any():
            continue
        rows.append(
            {
                "ticker": str(ticker),
                "announcement_date": pd.NaT,
                "effective_date": last_active,
                "last_active_month": last_active,
                "first_inactive_month": last_active + pd.offsets.MonthEnd(1),
                "months_observed_after_last_active": int(len(later)),
                "oracle_approximation": True,
                "source": "inferred_from_terminal_future_volume",
                "note": (
                    "No announcement data supplied. Inferred because every later "
                    "observed month has zero/no valid activity; uses future data."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["effective_date", "ticker"]
    ).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[
            "ticker", "announcement_date", "effective_date",
            "last_active_month", "first_inactive_month",
            "months_observed_after_last_active", "oracle_approximation",
            "source", "note",
        ]
    )


def _known_delisting_matrix(
    formation_dates: pd.DatetimeIndex,
    tickers: pd.Index,
    events: pd.DataFrame,
) -> pd.DataFrame:
    ok = pd.DataFrame(True, index=formation_dates, columns=tickers, dtype=bool)
    if events.empty:
        return ok
    for row in events.itertuples(index=False):
        ticker = str(getattr(row, "ticker"))
        if ticker not in ok.columns:
            continue
        announcement = pd.Timestamp(getattr(row, "announcement_date"))
        effective = pd.Timestamp(getattr(row, "effective_date"))
        known = formation_dates >= announcement
        inside_holding_period = pd.DatetimeIndex(
            [month_end_after(date) for date in formation_dates]
        ) >= effective
        ok.loc[known & inside_holding_period, ticker] = False
    return ok


def _future_activity_ok(
    current_activity: pd.DataFrame,
    formation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    # oracle sensitivity: this uses future activity
    future_any = (
        current_activity.astype("boolean")
        .shift(-1)
        .iloc[::-1]
        .cummax()
        .iloc[::-1]
        .fillna(False)
    )
    return future_any.reindex(formation_dates).fillna(False).astype(bool)


def build_eligibility_matrix(
    *,
    volume: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    formation_dates: pd.DatetimeIndex | list[pd.Timestamp],
    config: EligibilityConfig | None = None,
    open_price: pd.DataFrame | None = None,
    delisting_events: pd.DataFrame | None = None,
    universe_tickers: Iterable[str] | None = None,
) -> EligibilityResult:
    """Construct the point-in-time formation-date eligibility matrix."""

    config = EligibilityConfig() if config is None else config
    config.validate()
    formation_dates = pd.DatetimeIndex(formation_dates).to_period("M").to_timestamp("M")
    formation_dates = formation_dates.sort_values().unique()

    declared_universe: list[str] | None = None
    if universe_tickers is not None:
        declared_universe = list(
            dict.fromkeys(str(x).strip() for x in universe_tickers if str(x).strip())
        )
        if not declared_universe:
            raise ValueError("universe_tickers was supplied but is empty.")
        available = pd.Index(declared_universe).intersection(volume.columns.astype(str))
        available = available.intersection(adjusted_close.columns.astype(str))
        missing_declared = sorted(set(declared_universe).difference(available))
        if missing_declared:
            raise ValueError(
                "Declared universe tickers are missing from raw volume/price panels: "
                f"{missing_declared[:25]}{'...' if len(missing_declared) > 25 else ''}"
            )
        # measure relative liquidity before filtering the candidate panel

    monthly_amount, current_activity, volume_meta = compute_monthly_trading_amount(
        volume,
        adjusted_close,
        open_price=open_price,
        liquidity_measure=config.liquidity_measure,
        volume_is_dollar=config.volume_is_dollar,
        price_proxy=config.price_proxy,
        max_current_price_staleness_days=config.max_current_price_staleness_days,
    )
    monthly_amount = _month_end_index(monthly_amount)
    current_activity = _month_end_index(current_activity).astype(bool)
    tickers = monthly_amount.columns

    # liquidity uses trading amounts, independent of price availability
    monthly_activity = (monthly_amount > 0.0).astype(bool)

    lagged_activity = monthly_activity.shift(config.signal_lag_months)
    active_count = lagged_activity.rolling(
        config.history_months,
        min_periods=config.history_months,
    ).sum()
    required_active = (
        config.history_months
        if config.require_every_history_month_active
        else (
            config.min_active_history_months
            if config.min_active_history_months is not None
            else config.history_months
        )
    )
    history_ok_all = active_count >= int(required_active)

    lagged_rolling = monthly_amount.shift(config.signal_lag_months).rolling(
        config.liquidity_window_months,
        min_periods=config.liquidity_window_months,
    ).mean()
    universe_total = lagged_rolling.sum(axis=1, min_count=1).replace(0.0, np.nan)
    relative_share = lagged_rolling.div(universe_total, axis=0)
    liquidity_ok_all = relative_share > config.relative_liquidity_threshold

    if config.require_current_activity or config.require_current_price:
        current_ok_all = current_activity.copy()
    else:
        current_ok_all = pd.DataFrame(True, index=monthly_amount.index, columns=tickers)

    events = (
        pd.DataFrame(columns=["ticker", "announcement_date", "effective_date"])
        if delisting_events is None
        else delisting_events.copy()
    )
    if not events.empty:
        for col in ["announcement_date", "effective_date"]:
            events[col] = pd.to_datetime(events[col], errors="coerce")

    if config.delisting_mode == "point_in_time":
        delisting_ok = _known_delisting_matrix(formation_dates, tickers, events)
    elif config.delisting_mode == "current_only":
        delisting_ok = pd.DataFrame(True, index=formation_dates, columns=tickers)
    elif config.delisting_mode == "oracle_future_activity":
        delisting_ok = _future_activity_ok(monthly_activity, formation_dates)
    elif config.delisting_mode == "oracle_next_month_activity":
        delisting_ok = (
            monthly_activity.astype("boolean")
            .shift(-1)
            .reindex(formation_dates)
            .fillna(False)
            .astype(bool)
        )
    else:
        raise AssertionError(config.delisting_mode)

    history_ok = history_ok_all.reindex(formation_dates, columns=tickers).fillna(False).astype(bool)
    liquidity_ok = liquidity_ok_all.reindex(formation_dates, columns=tickers).fillna(False).astype(bool)
    current_ok = current_ok_all.reindex(formation_dates, columns=tickers).fillna(False).astype(bool)
    delisting_ok = delisting_ok.reindex(formation_dates, columns=tickers).fillna(False).astype(bool)
    eligibility = history_ok & liquidity_ok & current_ok & delisting_ok

    denominator_ticker_count = int(len(tickers))
    if declared_universe is not None:
        output_columns = pd.Index(declared_universe)
        history_ok = history_ok.reindex(columns=output_columns).fillna(False).astype(bool)
        liquidity_ok = liquidity_ok.reindex(columns=output_columns).fillna(False).astype(bool)
        current_ok = current_ok.reindex(columns=output_columns).fillna(False).astype(bool)
        delisting_ok = delisting_ok.reindex(columns=output_columns).fillna(False).astype(bool)
        eligibility = eligibility.reindex(columns=output_columns).fillna(False).astype(bool)
        monthly_amount = monthly_amount.reindex(columns=output_columns)
        lagged_rolling = lagged_rolling.reindex(columns=output_columns)
        relative_share = relative_share.reindex(columns=output_columns)
        current_activity = current_activity.reindex(columns=output_columns).fillna(False)
        tickers = output_columns

    eligibility.index.name = "formation_date"

    timing_map = pd.DataFrame(
        {
            "formation_date": formation_dates,
            "outcome_date": [month_end_after(date) for date in formation_dates],
            "history_window_start": [
                date - pd.offsets.MonthEnd(config.signal_lag_months + config.history_months - 1)
                for date in formation_dates
            ],
            "history_window_end": [
                date - pd.offsets.MonthEnd(config.signal_lag_months)
                for date in formation_dates
            ],
            "eligibility_semantics": [
                "eligible at formation; may be held through outcome month-end"
            ] * len(formation_dates),
        }
    ).set_index("formation_date")

    audit_by_date = pd.DataFrame(
        {
            "n_history_ok": history_ok.sum(axis=1),
            "n_liquidity_ok": liquidity_ok.sum(axis=1),
            "n_current_activity_ok": current_ok.sum(axis=1),
            "n_delisting_ok": delisting_ok.sum(axis=1),
            "n_eligible": eligibility.sum(axis=1),
        }
    )
    audit_by_date = audit_by_date.join(timing_map[["outcome_date"]])

    first_eligible = eligibility.apply(
        lambda s: s.index[s.to_numpy()].min() if s.any() else pd.NaT,
        axis=0,
    )
    last_eligible = eligibility.apply(
        lambda s: s.index[s.to_numpy()].max() if s.any() else pd.NaT,
        axis=0,
    )
    audit_by_ticker = pd.DataFrame(
        {
            "eligible_months": eligibility.sum(axis=0).astype(int),
            "first_eligible_formation": first_eligible,
            "last_eligible_formation": last_eligible,
            "ever_active": current_activity.any(axis=0).reindex(tickers).fillna(False),
        }
    )
    audit_by_ticker.index.name = "ticker"

    config_payload = asdict(config)
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    metadata = {
        "config": config_payload,
        "config_sha256": config_hash,
        "timing_interpretation": (
            "eligibility[t]=1 permits formation at t and holding until the "
            "next calendar month-end"
        ),
        "lookahead_status": (
            "point_in_time"
            if config.delisting_mode in {"point_in_time", "current_only"}
            else "oracle_approximation_uses_future_activity"
        ),
        "n_formations": int(len(formation_dates)),
        "formation_start": str(formation_dates.min().date()) if len(formation_dates) else None,
        "formation_end": str(formation_dates.max().date()) if len(formation_dates) else None,
        "n_tickers": int(len(tickers)),
        "liquidity_denominator_ticker_count": denominator_ticker_count,
        "declared_universe_size": (
            int(len(declared_universe)) if declared_universe is not None else None
        ),
        "declared_universe_required": declared_universe is not None,
        "history_activity_source": "positive monthly supplied trading amount",
        "current_tradability_source": (
            "positive current-month trading amount plus valid/fresh adjusted-close quote"
        ),
        "volume": volume_meta,
    }

    return EligibilityResult(
        eligibility=eligibility.astype(np.int8),
        history_ok=history_ok.astype(np.int8),
        liquidity_ok=liquidity_ok.astype(np.int8),
        current_activity_ok=current_ok.astype(np.int8),
        delisting_ok=delisting_ok.astype(np.int8),
        monthly_trading_amount=monthly_amount,
        lagged_rolling_trading_amount=lagged_rolling,
        relative_liquidity_share=relative_share,
        audit_by_date=audit_by_date,
        audit_by_ticker=audit_by_ticker,
        timing_map=timing_map,
        metadata=metadata,
    )


def write_eligibility_result(result: EligibilityResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.eligibility.to_csv(output_dir / "eligibility.csv")
    result.history_ok.to_csv(output_dir / "eligibility_component_history.csv")
    result.liquidity_ok.to_csv(output_dir / "eligibility_component_liquidity.csv")
    result.current_activity_ok.to_csv(output_dir / "eligibility_component_current_activity.csv")
    result.delisting_ok.to_csv(output_dir / "eligibility_component_delisting.csv")
    result.monthly_trading_amount.to_csv(output_dir / "monthly_trading_amount.csv")
    result.lagged_rolling_trading_amount.to_csv(output_dir / "lagged_rolling_trading_amount.csv")
    result.monthly_trading_amount.to_csv(output_dir / "monthly_dollar_volume.csv")
    result.lagged_rolling_trading_amount.to_csv(output_dir / "lagged_rolling_dollar_volume.csv")
    result.relative_liquidity_share.to_csv(output_dir / "relative_liquidity_share.csv")
    result.audit_by_date.to_csv(output_dir / "eligibility_audit_by_date.csv")
    result.audit_by_ticker.to_csv(output_dir / "eligibility_audit_by_ticker.csv")
    result.timing_map.to_csv(output_dir / "formation_outcome_map.csv")
    write_json(output_dir / "eligibility_metadata.json", result.metadata)


def explain_eligibility(
    result: EligibilityResult,
    ticker: str,
    formation_date: str | pd.Timestamp,
) -> dict[str, Any]:
    date = pd.Timestamp(formation_date).to_period("M").to_timestamp("M")
    ticker = str(ticker)
    if date not in result.eligibility.index or ticker not in result.eligibility.columns:
        raise KeyError(f"Unknown eligibility cell: {date=}, {ticker=}")
    return {
        "formation_date": date,
        "outcome_date": month_end_after(date),
        "ticker": ticker,
        "eligible": bool(result.eligibility.loc[date, ticker]),
        "history_ok": bool(result.history_ok.loc[date, ticker]),
        "liquidity_ok": bool(result.liquidity_ok.loc[date, ticker]),
        "current_activity_ok": bool(result.current_activity_ok.loc[date, ticker]),
        "delisting_ok": bool(result.delisting_ok.loc[date, ticker]),
        "relative_liquidity_share": float(
            result.relative_liquidity_share.reindex(index=[date], columns=[ticker]).iloc[0, 0]
        ),
    }
