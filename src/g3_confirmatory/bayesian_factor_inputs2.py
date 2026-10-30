from __future__ import annotations
import io
import re
import urllib.request
import zipfile
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

MACRO_SHEET = "5) Macro and rates predictors"
REGIME_SHEET = "6) Regime and stress indicators"
TCOST_SHEET = "7) Transaction costs and fees "

RF_COL = "RF"
ADJUSTED_CLOSE_FILENAME = "adjusted_close_prices.csv"
META_FILENAME = "Meta.csv"

FRENCH_START = "2015-11-30"
FRENCH_END = "2025-12-31"
FRENCH_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
FRENCH_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)
FRENCH_CACHE_FILENAME = "ff5_plus_momentum_monthly_2015_2025.csv"
ALTERNATE_FRENCH_FILENAMES = (
    "ff5_plus_momentum_monthly_2016_2025.csv",
    "ff5_plus_momentum_monthly.csv",
)

EQUITY_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

OBSERVED_FACTOR_NAME_MAP = {
    "Mkt-RF": "Mkt-RF (market)",
    "SMB": "SMB (size)",
    "HML": "HML (value)",
    "RMW": "RMW (profitability)",
    "CMA": "CMA (investment)",
    "Mom": "MOM (momentum)",
}

MACRO_FACTOR_COLS = [
    "inflation",
    "real interest rates",
    "term structure",
    "SOFR O/N",
    "MM_12M",
    "IG OAS",
    "HY OAS",
    "NY Fed recession",
    "VIX",
]

MODEL_MACRO_DIFF_COLS = [
    "inflation",
    "real interest rates",
    "term structure",
    "SOFR O/N",
    "MM_12M",
    "IG OAS",
    "HY OAS",
    "NY Fed recession",
    "VIX",
]

MACRO_PERCENT_COLS = [
    "UST 1m YTM",
    "UST 3m YTM",
    "UST 2Y YTM",
    "UST 5Y YTM",
    "UST 10Y YTM",
    "UST 30Y YTM",
    "T10Y-3M",
    "T10Y-2Y",
    "SOFR O/N",
    "SOFR 1m",
    "SOFR 3m",
    "SOFR 6m",
    "SOFR 12m",
    "EFFR O/N",
    "LIBOR O/N",
    "LIBOR 1m",
    "LIBOR 3m",
    "LIBOR 6m",
    "LIBOR 12m",
    "CPI inflation rate, monthly",
    "5Y BEI",
    "10Y BEI",
    "IG OAS",
    "HY OAS",
    "IG T-Spread",
    "HY T-Spread",
    "NY Fed Recession",
]

ASSET_CLASS_MAP = {
    "Equities": "equity",
    "Bonds": "bond",
    "Mixed Assets": "mixed",
    "Cryptocurrencies": "crypto",
    "Precious Metals": "metals",
    "Money Market and Currencies": "cash_fx",
    "Real Estate": "real_estate",
    "Commodities": "commodities",
}

def _standardize_column_name(column: Any) -> str:
    text = str(column).strip()
    text = (
        text.replace("ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“", "-")
        .replace("Ã¢â‚¬â€œ", "-")
        .replace("â€“", "-")
        .replace("\xa0", " ")
    )
    if text.startswith("T10Y") and "3M" in text:
        return "T10Y-3M"
    if text.startswith("T10Y") and "2Y" in text:
        return "T10Y-2Y"
    return text


def _first_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        col_str = str(col).lower()
        if "date" in col_str or "unnamed" in col_str or col == ",%":
            return col
    return str(df.columns[0])


def _select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = [col for col in columns if col in df.columns]
    return df[keep].copy()


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {label}: {missing}")


def clean_time_series_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_standardize_column_name(col) for col in out.columns]
    date_col = _first_date_column(out)
    out = out.rename(columns={date_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    out = (
        out.dropna(subset=["Date"])
        .sort_values("Date")
        .set_index("Date")
    )

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def convert_percent_like_to_decimal(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col] / 100.0
    return out


def align_daily_to_price_dates(df: pd.DataFrame, price_dates: pd.Index) -> pd.DataFrame:
    common_dates = df.index.intersection(price_dates)
    return df.loc[common_dates].sort_index()


def make_monthly_end_of_month(df_daily: pd.DataFrame) -> pd.DataFrame:
    return df_daily.resample("ME").last()


def _log_diff_positive(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0.0)
    return np.log(positive).diff()


def stitch_spread_adjusted_series(
    df_daily: pd.DataFrame,
    libor_col: str,
    sofr_col: str,
    output_col: str,
) -> tuple[pd.Series, dict[str, Any]]:
    _require_columns(df_daily, [libor_col, sofr_col], label=output_col)

    libor = df_daily[libor_col].copy()
    sofr = df_daily[sofr_col].copy()

    first_sofr_date = sofr.first_valid_index()
    overlap = pd.concat([libor.rename("libor"), sofr.rename("sofr")], axis=1).dropna()

    spread = np.nan
    spread_adjusted = False
    adjusted_libor = libor.copy()
    note = ""

    if not overlap.empty:
        spread = float((overlap["libor"] - overlap["sofr"]).mean())
        adjusted_libor = libor - spread
        spread_adjusted = True
        note = f"Spread-adjusted using mean({libor_col} - {sofr_col}) over overlap."
    else:
        note = (
            f"No overlap between {libor_col} and {sofr_col}; stitched without adjustment. "
            "This may introduce a level break."
        )

    stitched = pd.Series(np.nan, index=df_daily.index, name=output_col, dtype=float)

    if first_sofr_date is None:
        stitched.loc[:] = adjusted_libor
        note = f"{sofr_col} never becomes available; {output_col} stays on the {libor_col} leg."
    else:
        before_sofr_mask = stitched.index < first_sofr_date
        stitched.loc[before_sofr_mask] = adjusted_libor.loc[before_sofr_mask]
        stitched.loc[~before_sofr_mask] = sofr.loc[~before_sofr_mask]

    metadata = {
        "series": output_col,
        "libor_col": libor_col,
        "sofr_col": sofr_col,
        "first_sofr_date": first_sofr_date,
        "overlap_count": int(len(overlap)),
        "spread_adjusted": spread_adjusted,
        "spread": spread,
        "note": note,
    }

    return stitched, metadata


def summarize_missingness(df: pd.DataFrame, group_map: dict[str, str] | None = None) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100.0),
        }
    )

    if group_map is not None:
        summary.insert(0, "group", summary.index.map(group_map))

    return summary.sort_values(["missing_pct", "missing_count"], ascending=False)


def load_french_monthly(url: str, target_cols: list[str]) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin1")

    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        parts = [_standardize_column_name(part) for part in line.split(",")]
        if parts[: len(target_cols) + 1] == [""] + target_cols:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not locate the expected monthly header in {url}.")

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        parts = [_standardize_column_name(part) for part in line.split(",")]
        if parts and re.fullmatch(r"\d{6}", parts[0]):
            data_lines.append(line)
        elif len(data_lines) > 1:
            break

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [_standardize_column_name(c) for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Date"})
    _require_columns(df, ["Date", *target_cols], label=url)

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").apply(pd.to_numeric, errors="coerce").sort_index() / 100.0

    return df.loc[FRENCH_START:FRENCH_END]


def _read_french_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index).to_period("M").to_timestamp("M")
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    return frame


def _validate_french_panel(
    frame: pd.DataFrame, *, label: str, require_full_calendar: bool
) -> pd.DataFrame:
    _require_columns(frame, EQUITY_FACTOR_COLS + [RF_COL], label=label)
    target_start = pd.Timestamp(FRENCH_START).to_period("M").to_timestamp("M")
    target_end = pd.Timestamp(FRENCH_END).to_period("M").to_timestamp("M")
    frame = frame.loc[target_start:target_end, EQUITY_FACTOR_COLS + [RF_COL]]
    if frame.empty:
        raise ValueError(f"{label} is empty in the study date range.")
    if require_full_calendar:
        monthly = pd.date_range(target_start, target_end, freq="ME")
        missing_dates = monthly.difference(frame.index)
        if len(missing_dates):
            raise ValueError(
                f"{label} does not cover the requested download calendar. "
                f"Missing {len(missing_dates)} month(s), beginning "
                f"{missing_dates[:6].strftime('%Y-%m-%d').tolist()}."
            )
        frame = frame.reindex(monthly)
    missing_values = frame.isna().any(axis=1)
    if missing_values.any():
        bad = frame.index[missing_values][:6].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{label} contains missing required factors at {bad}.")
    return frame


def load_fama_french_monthly(
    input_dir: str | Path,
    *,
    factor_path: str | Path | None = None,
    factor_cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Read local factors; cache a download only when no local file is usable."""
    input_dir = Path(input_dir)
    cache_dir = (
        Path(factor_cache_dir)
        if factor_cache_dir is not None
        else Path(__file__).resolve().parents[2] / "work" / "data" / "factor_cache"
    )
    cache_path = cache_dir / FRENCH_CACHE_FILENAME

    explicit_cache = os.environ.get("G3_FRENCH_CACHE")
    cache_candidates: list[Path] = []
    if explicit_cache:
        cache_candidates.append(Path(os.path.expandvars(os.path.expanduser(explicit_cache))))
    if factor_path is not None:
        cache_candidates.append(Path(factor_path))
    cache_candidates.append(input_dir / FRENCH_CACHE_FILENAME)
    cache_candidates.extend(input_dir / name for name in ALTERNATE_FRENCH_FILENAMES)
    cache_candidates.append(cache_path)

    errors: list[str] = []
    for candidate in cache_candidates:
        if not candidate.exists():
            continue
        try:
            combined = _validate_french_panel(
                _read_french_cache(candidate),
                label=f"Fama-French cache {candidate}",
                require_full_calendar=False,
            )
            info = {
                "source": "local_cache",
                "cache_path": str(candidate),
                "required_start": FRENCH_START,
                "required_end": FRENCH_END,
                "actual_start": str(combined.index.min().date()),
                "actual_end": str(combined.index.max().date()),
            }
            return combined[EQUITY_FACTOR_COLS].copy(), combined[[RF_COL]].copy(), info
        except Exception as cache_exc:
            errors.append(f"{candidate}: {cache_exc}")

    try:
        ff5 = load_french_monthly(
            FRENCH_FF5_URL,
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        )
        mom = load_french_monthly(FRENCH_MOM_URL, ["Mom"])
        combined = _validate_french_panel(
            pd.concat([ff5, mom], axis=1),
            label="downloaded Fama-French data",
            require_full_calendar=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path)
        info = {
            "source": "download_no_local_cache",
            "cache_path": str(cache_path),
            "required_start": FRENCH_START,
            "required_end": FRENCH_END,
            "actual_start": str(combined.index.min().date()),
            "actual_end": str(combined.index.max().date()),
        }
        return combined[EQUITY_FACTOR_COLS].copy(), combined[[RF_COL]].copy(), info
    except Exception as download_exc:
        details = "\n - ".join(errors) if errors else "no local cache files were found"
        raise RuntimeError(
            "No readable local Fama-French cache was found and the fallback "
            "download failed. Put ff5_plus_momentum_monthly_2016_2025.csv in "
            "the input folder. Cache audit:\n - " + details
        ) from download_exc


def load_adjusted_close_prices(
    input_dir: str | Path,
    *,
    path: str | Path | None = None,
) -> pd.DataFrame:
    adjusted_close_path = Path(path) if path is not None else Path(input_dir) / ADJUSTED_CLOSE_FILENAME
    if not adjusted_close_path.exists():
        raise FileNotFoundError(f"Adjusted-close input not found: {adjusted_close_path}")
    adjusted_close = pd.read_csv(adjusted_close_path, index_col=0, parse_dates=True)
    adjusted_close.index = pd.to_datetime(adjusted_close.index)
    return adjusted_close.sort_index().apply(pd.to_numeric, errors="coerce")


def compute_monthly_simple_returns_from_adjusted_close(
    adjusted_close: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    if target_columns is not None:
        adjusted_close = adjusted_close.reindex(columns=target_columns)
    monthly_adjusted_close = adjusted_close.resample("ME").last()
    # leave missing prices missing when computing returns
    return monthly_adjusted_close.pct_change(fill_method=None)


def build_macro_factor_frame(
    macro_monthly: pd.DataFrame,
    regime_monthly: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        macro_monthly,
        [
            "CPI inflation rate, monthly",
            "UST 10Y YTM",
            "10Y BEI",
            "T10Y-3M",
            "SOFR O/N",
            "MM_12M",
            "IG OAS",
            "HY OAS",
            "NY Fed Recession",
        ],
        label="macro_monthly",
    )
    _require_columns(regime_monthly, ["VIX"], label="regime_monthly")

    macro_factors_df = pd.DataFrame(index=macro_monthly.index)
    macro_factors_df["inflation"] = macro_monthly["CPI inflation rate, monthly"]
    macro_factors_df["real interest rates"] = macro_monthly["UST 10Y YTM"] - macro_monthly["10Y BEI"]
    macro_factors_df["term structure"] = macro_monthly["T10Y-3M"]
    macro_factors_df["SOFR O/N"] = macro_monthly["SOFR O/N"]
    macro_factors_df["MM_12M"] = macro_monthly["MM_12M"]
    macro_factors_df["IG OAS"] = macro_monthly["IG OAS"]
    macro_factors_df["HY OAS"] = macro_monthly["HY OAS"]
    macro_factors_df["NY Fed recession"] = macro_monthly["NY Fed Recession"]
    macro_factors_df["VIX"] = regime_monthly["VIX"]

    return macro_factors_df.sort_index()


def build_transformed_macro_factor_frame(
    raw_macro_factors_df: pd.DataFrame,
    diff_columns: list[str] | tuple[str, ...] = MODEL_MACRO_DIFF_COLS,
) -> pd.DataFrame:
    transformed = raw_macro_factors_df.copy()
    diff_columns_set = set(diff_columns)
    for column in transformed.columns:
        if column == "VIX" and column in diff_columns_set:
            transformed[column] = _log_diff_positive(transformed[column])
        elif column in diff_columns_set:
            transformed[column] = transformed[column].diff()
    return transformed.sort_index()


def build_transformation_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "equity_factors",
                "variable": "Mkt-RF, SMB, HML, RMW, CMA, Mom",
                "transform": "No additional transform beyond monthly download and conversion to decimals.",
                "model_note": "Kept in simple-return space for an excess-return model.",
            },
            {
                "dataset": "macro_factors_df",
                "variable": "inflation",
                "transform": "Convert percent-like workbook series to decimals, sample at month-end, then take first differences.",
                "model_note": "Modeled as the monthly change in inflation rather than the inflation rate level.",
            },
            {
                "dataset": "macro_factors_df",
                "variable": "real interest rates, term structure, SOFR O/N, MM_12M, IG OAS, HY OAS, NY Fed recession",
                "transform": "Convert percent-like workbook series to decimals, stitch MM_12M, sample at month-end, then take first differences.",
                "model_note": "Modeled as monthly shocks rather than levels.",
            },
            {
                "dataset": "macro_factors_df",
                "variable": "VIX",
                "transform": "Sample VIX at month-end, then take log differences.",
                "model_note": "Uses volatility shocks in log space rather than end-of-month level differences.",
            },
            {
                "dataset": "monthly_excess_returns",
                "variable": "ETF excess returns",
                "transform": "Read adjusted close prices, compute month-end simple returns with pct_change(), then subtract monthly Fama-French RF.",
                "model_note": "Kept in simple-return space as requested.",
            },
            {
                "dataset": "bond_factors_df",
                "variable": "duration",
                "transform": "Use the negative monthly change in UST 10Y YTM.",
                "model_note": "This serves as the bond duration shock factor.",
            },
        ]
    )

def build_bond_factor_suggestions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "suggested_factor": "duration",
                "proxy_idea": "Use long-duration Treasury excess returns over cash, or a negative monthly change in the 10Y Treasury yield.",
                "why_it_matters": "Duration is usually the first-order style exposure across bond ETFs.",
            },
        ]
    )

def load_asset_class_series(
    input_dir: str | Path,
    target_columns: list[str] | None = None,
    *,
    meta_path: str | Path | None = None,
) -> pd.Series:
    meta_path = Path(meta_path) if meta_path is not None else Path(input_dir) / META_FILENAME
    meta = pd.read_csv(meta_path).drop_duplicates(subset="Ticker", keep="first")

    _require_columns(meta, ["Ticker", "Investment Object"], label=str(meta_path))

    class_frame = (
        meta.loc[:, ["Ticker", "Investment Object"]]
        .rename(columns={"Ticker": "ticker", "Investment Object": "raw_asset_class"})
        .dropna(subset=["ticker"])
        .drop_duplicates(subset="ticker", keep="first")
    )
    class_frame["asset_class"] = class_frame["raw_asset_class"].map(ASSET_CLASS_MAP)
    unknown = class_frame["raw_asset_class"].notna() & class_frame["asset_class"].isna()
    if unknown.any():
        labels = sorted(class_frame.loc[unknown, "raw_asset_class"].astype(str).unique())
        raise ValueError(
            "Unmapped asset classes found; update ASSET_CLASS_MAP instead of "
            f"silently dropping them: {labels}"
        )
    asset_class = class_frame.set_index("ticker")["asset_class"]

    if target_columns is not None:
        asset_class = asset_class.reindex(target_columns)

    return asset_class


def build_bond_factor_frame(macro_monthly: pd.DataFrame) -> pd.DataFrame:
    _require_columns(macro_monthly, ["UST 10Y YTM"], label="macro_monthly")

    bond_factors_df = pd.DataFrame(index=macro_monthly.index)
    bond_factors_df["duration"] = -macro_monthly["UST 10Y YTM"].diff()

    return bond_factors_df.sort_index()


def build_raw_bond_factor_frame(macro_monthly: pd.DataFrame) -> pd.DataFrame:
    _require_columns(macro_monthly, ["UST 10Y YTM"], label="macro_monthly")

    raw_bond_factors_df = pd.DataFrame(index=macro_monthly.index)
    raw_bond_factors_df["duration"] = macro_monthly["UST 10Y YTM"]

    return raw_bond_factors_df.sort_index()


def build_bayesian_factor_inputs(
    input_dir: str | Path,
    *,
    raw_data: dict[str, Any] | None = None,
    factor_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    raw = dict(raw_data or {})
    workbook_path = Path(raw.get("macro_workbook") or input_dir / "Portfolio Macro Data.xlsx")
    adjusted_close_prices = load_adjusted_close_prices(
        input_dir, path=raw.get("adjusted_close_csv")
    )

    macro_daily = pd.read_excel(workbook_path, sheet_name=MACRO_SHEET)
    regime_daily = pd.read_excel(workbook_path, sheet_name=REGIME_SHEET)
    tcost_daily = pd.read_excel(workbook_path, sheet_name=TCOST_SHEET)

    macro_daily_clean = clean_time_series_sheet(macro_daily)
    regime_daily_clean = clean_time_series_sheet(regime_daily)
    tcost_daily_clean = clean_time_series_sheet(tcost_daily)

    macro_daily_clean = convert_percent_like_to_decimal(
        macro_daily_clean,
        [col for col in MACRO_PERCENT_COLS if col in macro_daily_clean.columns],
    )

    price_dates = pd.DatetimeIndex(adjusted_close_prices.index)
    macro_daily_clean = align_daily_to_price_dates(macro_daily_clean, price_dates)
    regime_daily_clean = align_daily_to_price_dates(regime_daily_clean, price_dates)
    tcost_daily_clean = align_daily_to_price_dates(tcost_daily_clean, price_dates)

    mm_12m, mm_12m_meta = stitch_spread_adjusted_series(
        macro_daily_clean,
        libor_col="LIBOR 12m",
        sofr_col="SOFR 12m",
        output_col="MM_12M",
    )
    macro_daily_clean["MM_12M"] = mm_12m

    macro_monthly = make_monthly_end_of_month(macro_daily_clean)
    regime_monthly = make_monthly_end_of_month(regime_daily_clean)
    tcost_monthly = make_monthly_end_of_month(tcost_daily_clean)

    raw_macro_factors_df = build_macro_factor_frame(macro_monthly=macro_monthly, regime_monthly=regime_monthly)
    transformed_macro_factors_df = build_transformed_macro_factor_frame(raw_macro_factors_df=raw_macro_factors_df)
    raw_bond_factors_df = build_raw_bond_factor_frame(macro_monthly=macro_monthly)
    bond_factors_df = build_bond_factor_frame(macro_monthly=macro_monthly)
    equity_factors, rf_df, fama_french_download_info = load_fama_french_monthly(
        input_dir=input_dir,
        factor_path=raw.get("ff5_plus_momentum_csv"),
        factor_cache_dir=factor_cache_dir,
    )
    raw_equity_factors_df = equity_factors.copy()
    transformed_equity_factors_df = equity_factors.copy()
    observed_factor_df = equity_factors.rename(columns=OBSERVED_FACTOR_NAME_MAP)

    monthly_simple_returns = compute_monthly_simple_returns_from_adjusted_close(
        adjusted_close=adjusted_close_prices,
    )
    monthly_excess_returns = monthly_simple_returns.sub(rf_df[RF_COL], axis=0)
    monthly_excess_returns = monthly_excess_returns.dropna(how="all")

    tcost_cols = list(tcost_monthly.columns)
    tcost_df = _select_existing(tcost_monthly, tcost_cols)

    common_months = pd.DatetimeIndex(monthly_excess_returns.index)
    for df in [equity_factors, rf_df, raw_macro_factors_df, bond_factors_df, tcost_df]:
        common_months = common_months.intersection(df.index)
    common_months = common_months.sort_values()

    monthly_excess_returns = monthly_excess_returns.loc[common_months].sort_index()
    monthly_simple_returns = monthly_simple_returns.loc[common_months].sort_index()
    equity_factors = equity_factors.loc[common_months].sort_index()
    raw_equity_factors_df = raw_equity_factors_df.loc[common_months].sort_index()
    transformed_equity_factors_df = transformed_equity_factors_df.loc[common_months].sort_index()
    observed_factor_df = observed_factor_df.loc[common_months].sort_index()
    rf_df = rf_df.loc[common_months].sort_index()
    # January is needed for the February 2016 macro difference
    macro_months = common_months.union(
        pd.DatetimeIndex([common_months.min() - pd.offsets.MonthEnd(1)])
    )
    raw_macro_factors_df = raw_macro_factors_df.loc[macro_months].sort_index()
    transformed_macro_factors_df = transformed_macro_factors_df.loc[macro_months].sort_index()
    macro_factors_df = raw_macro_factors_df.copy()
    raw_bond_factors_df = raw_bond_factors_df.loc[common_months].sort_index()
    bond_factors_df = bond_factors_df.loc[common_months].sort_index()
    tcost_df = tcost_df.loc[common_months].sort_index()


    observed_features_monthly = pd.concat(
        [rf_df, equity_factors, macro_factors_df, bond_factors_df, tcost_df],
        axis=1,
    ).sort_index()

    X_monthly = pd.concat(
        [equity_factors, macro_factors_df, bond_factors_df, tcost_df],
        axis=1,
    ).sort_index()

    raw_factor_blocks_df = pd.concat(
        [raw_equity_factors_df, raw_macro_factors_df, raw_bond_factors_df],
        axis=1,
    ).sort_index()

    transformed_factor_blocks_df = pd.concat(
        [transformed_equity_factors_df, transformed_macro_factors_df, bond_factors_df],
        axis=1,
    ).sort_index()

    group_map = {RF_COL: "rf"}
    group_map.update({col: "equity_factor" for col in equity_factors.columns})
    group_map.update({col: "macro_factor" for col in macro_factors_df.columns})
    group_map.update({col: "bond_factor" for col in bond_factors_df.columns})
    group_map.update({col: "transaction_cost" for col in tcost_df.columns})

    missing_pct_by_factor = summarize_missingness(observed_features_monthly, group_map=group_map)
    money_market_stitch_info = pd.DataFrame([mm_12m_meta]).set_index("series")
    transformation_notes = build_transformation_notes()
    bond_factor_suggestions = build_bond_factor_suggestions()
    asset_class_series = load_asset_class_series(
        input_dir=input_dir,
        target_columns=list(adjusted_close_prices.columns),
        meta_path=raw.get("meta_csv"),
    )

    return {
        "macro_daily_clean": macro_daily_clean,
        "regime_daily_clean": regime_daily_clean,
        "tcost_daily_clean": tcost_daily_clean,
        "monthly_log_returns": None,
        "monthly_simple_returns": monthly_simple_returns,
        "monthly_excess_returns": monthly_excess_returns,
        "rf_df": rf_df,
        "equity_factors": equity_factors,
        "raw_equity_factors_df": raw_equity_factors_df,
        "transformed_equity_factors_df": transformed_equity_factors_df,
        "observed_factor_df": observed_factor_df,
        "macro_factors_df": macro_factors_df,
        "raw_macro_factors_df": raw_macro_factors_df,
        "transformed_macro_factors_df": transformed_macro_factors_df,
        "bond_factors_df": bond_factors_df,
        "raw_bond_factors_df": raw_bond_factors_df,
        "transformed_bond_factors_df": bond_factors_df,
        "tcost_df": tcost_df,
        "observed_features_monthly": observed_features_monthly,
        "X_monthly": X_monthly,
        "raw_factor_blocks_df": raw_factor_blocks_df,
        "transformed_factor_blocks_df": transformed_factor_blocks_df,
        "asset_class_series": asset_class_series,
        "missing_pct_by_factor": missing_pct_by_factor,
        "money_market_stitch_info": money_market_stitch_info,
        "transformation_notes": transformation_notes,
        "bond_factor_suggestions": bond_factor_suggestions,
        "fama_french_download_info": fama_french_download_info,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build factor inputs and optionally prepare monthly eligibility."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input-dir", type=Path, default=repo_root / "input_data")
    parser.add_argument("--build-eligibility", action="store_true")
    parser.add_argument("--volume-csv", type=Path)
    parser.add_argument("--open-csv", type=Path)
    parser.add_argument("--adjusted-close-csv", type=Path)
    parser.add_argument("--delisting-events-csv", type=Path)
    parser.add_argument(
        "--universe-tickers-file", type=Path,
        help=(
            "Text file (one ticker per line) or CSV with a ticker/Ticker column. "
            "Use this to freeze the declared 470-fund universe; otherwise every "
            "ticker common to Volume and adjusted-close panels is considered."
        ),
    )
    parser.add_argument("--expected-universe-size", type=int)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "work" / "data")
    parser.add_argument("--factor-cache-dir", type=Path)
    parser.add_argument("--outcome-start", default="2019-01-31")
    parser.add_argument("--outcome-end", default="2025-12-31")
    parser.add_argument(
        "--delisting-mode",
        default="point_in_time",
        choices=[
            "point_in_time",
            "current_only",
            "oracle_future_activity",
            "oracle_next_month_activity",
        ],
    )
    parser.add_argument("--relative-liquidity-threshold", type=float, default=1.0 / 10_000.0)
    parser.add_argument("--history-months", type=int, default=36)
    parser.add_argument(
        "--liquidity-measure",
        choices=["dollar_volume", "raw_volume"],
        default="dollar_volume",
        help=(
            "dollar_volume is the corrected primary screen; raw_volume exactly "
            "reproduces the notebook final relative-volume rule."
        ),
    )
    parser.add_argument("--volume-is-dollar", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.build_eligibility:
        try:
            from g3_confirmatory.eligibility import (
                EligibilityConfig,
                build_eligibility_matrix,
                build_formation_outcome_map,
                read_delisting_events,
                write_eligibility_result,
            )
        except ImportError:
            repo_src = Path(__file__).resolve().parents[2] / "src"
            sys.path.insert(0, str(repo_src))
            from g3_confirmatory.eligibility import (
                EligibilityConfig,
                build_eligibility_matrix,
                build_formation_outcome_map,
                read_delisting_events,
                write_eligibility_result,
            )

        if args.volume_csv is None:
            raise SystemExit(
                "--volume-csv is required to reproduce the notebook liquidity rule. "
                "Volume.csv was not included in the supplied code archive."
            )
        adjusted_path = args.adjusted_close_csv or (input_dir / ADJUSTED_CLOSE_FILENAME)
        volume = pd.read_csv(args.volume_csv, index_col=0, parse_dates=True)
        adjusted = pd.read_csv(adjusted_path, index_col=0, parse_dates=True)
        open_price = (
            pd.read_csv(args.open_csv, index_col=0, parse_dates=True)
            if args.open_csv is not None
            else None
        )
        universe_tickers = None
        if args.universe_tickers_file is not None:
            upath = args.universe_tickers_file
            if upath.suffix.lower() == ".csv":
                udf = pd.read_csv(upath)
                ticker_col = next(
                    (c for c in ("ticker", "Ticker", "TICKER") if c in udf.columns),
                    None,
                )
                if ticker_col is None:
                    if udf.shape[1] != 1:
                        raise SystemExit(
                            "Universe CSV must contain ticker/Ticker or exactly one column."
                        )
                    ticker_col = udf.columns[0]
                universe_tickers = udf[ticker_col].dropna().astype(str).tolist()
            else:
                universe_tickers = [
                    line.strip()
                    for line in upath.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            universe_tickers = list(dict.fromkeys(universe_tickers))
            if args.expected_universe_size is not None and len(universe_tickers) != args.expected_universe_size:
                raise SystemExit(
                    f"Declared universe has {len(universe_tickers)} tickers, expected "
                    f"{args.expected_universe_size}."
                )

        timing = build_formation_outcome_map(
            outcome_start=args.outcome_start, outcome_end=args.outcome_end
        )
        result = build_eligibility_matrix(
            volume=volume,
            adjusted_close=adjusted,
            open_price=open_price,
            formation_dates=pd.DatetimeIndex(timing["formation_date"]),
            delisting_events=read_delisting_events(args.delisting_events_csv),
            universe_tickers=universe_tickers,
            config=EligibilityConfig(
                history_months=args.history_months,
                liquidity_window_months=args.history_months,
                relative_liquidity_threshold=args.relative_liquidity_threshold,
                liquidity_measure=args.liquidity_measure,
                volume_is_dollar=args.volume_is_dollar,
                delisting_mode=args.delisting_mode,
            ),
        )
        write_eligibility_result(result, output_dir)
        print(json.dumps(result.metadata, indent=2, default=str))

    eligibility_path = output_dir / "eligibility.csv"
    if not eligibility_path.exists():
        eligibility_path = output_dir / "rebalance_universe_matrix.csv"
    if eligibility_path.exists():
        outputs = build_bayesian_factor_inputs(
            input_dir=input_dir,
            raw_data={"adjusted_close_csv": args.adjusted_close_csv},
            factor_cache_dir=args.factor_cache_dir or output_dir / "factor_cache",
        )
        print(
            f"Built factor inputs for {len(outputs['monthly_excess_returns'].columns)} "
            f"assets and {len(outputs['monthly_excess_returns'])} months."
        )
