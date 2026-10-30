"""Performance measurement with one consistent excess-return convention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .optimizer import empirical_var_cvar

MONTHS_PER_YEAR = 12.0


def maximum_drawdown(net_simple_returns: pd.Series) -> float:
    r = pd.to_numeric(net_simple_returns, errors="coerce").dropna().astype(float)
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def downside_deviation(excess_returns: pd.Series, target: float = 0.0) -> float:
    r = pd.to_numeric(excess_returns, errors="coerce").dropna().astype(float)
    if r.empty:
        return np.nan
    downside = np.minimum(r - float(target), 0.0)
    return float(np.sqrt(np.mean(downside * downside)))


def historical_var_es(net_simple_returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    r = pd.to_numeric(net_simple_returns, errors="coerce").dropna().astype(float)
    if r.empty:
        return np.nan, np.nan
    risk = empirical_var_cvar(r.to_numpy(dtype=float), alpha=alpha)
    return float(risk["var"]), float(risk["cvar"])


def compute_performance_metrics(
    frame: pd.DataFrame,
    *,
    utility_gamma: float = 1.0,
    exclude_initial_funding_from_turnover: bool = True,
) -> dict[str, float]:
    """Compute metrics for one aligned strategy return history.
    
    Mean return, Sharpe, and certainty equivalent use net_excess_return and the
    factor file's Treasury-bill rate. CAGR and wealth use net simple returns.
    """

    data = frame.sort_values("outcome_date").copy()
    simple = pd.to_numeric(data["net_simple_return"], errors="coerce").dropna()
    excess = pd.to_numeric(data["net_excess_return"], errors="coerce").dropna()
    n = int(min(len(simple), len(excess)))
    if n == 0:
        raise ValueError("No valid returns are available for metrics.")
    simple = simple.iloc[:n]
    excess = excess.iloc[:n]

    mean_m = float(excess.mean())
    vol_m = float(excess.std(ddof=1)) if n > 1 else np.nan
    sharpe = (
        float(np.sqrt(MONTHS_PER_YEAR) * mean_m / vol_m)
        if np.isfinite(vol_m) and vol_m > 0.0 else np.nan
    )
    annual_mean = MONTHS_PER_YEAR * mean_m
    annual_vol = np.sqrt(MONTHS_PER_YEAR) * vol_m if np.isfinite(vol_m) else np.nan
    terminal_growth = float(np.prod(1.0 + simple.to_numpy(dtype=float)))
    cagr = (
        terminal_growth ** (MONTHS_PER_YEAR / n) - 1.0
        if terminal_growth > 0.0 else np.nan
    )
    max_dd = maximum_drawdown(simple)
    downside_m = downside_deviation(excess)
    sortino = (
        np.sqrt(MONTHS_PER_YEAR) * mean_m / downside_m
        if np.isfinite(downside_m) and downside_m > 0.0 else np.nan
    )
    calmar = cagr / abs(max_dd) if np.isfinite(cagr) and max_dd < 0.0 else np.nan
    var95, es95 = historical_var_es(simple, alpha=0.95)
    monthly_ce = mean_m - 0.5 * float(utility_gamma) * float(excess.var(ddof=1))

    turnover = pd.to_numeric(data.get("one_way_turnover"), errors="coerce")
    if exclude_initial_funding_from_turnover and "initial_funding" in data.columns:
        turnover = turnover.loc[~data["initial_funding"].astype(bool)]
    cost = pd.to_numeric(data.get("transaction_cost"), errors="coerce")
    gross_simple = pd.to_numeric(data.get("gross_simple_return"), errors="coerce")

    return {
        "n_months": n,
        "annualized_excess_mean": annual_mean,
        "annualized_excess_volatility": annual_vol,
        "excess_sharpe": sharpe,
        "cagr_net_simple": float(cagr),
        "maximum_drawdown": max_dd,
        "calmar": float(calmar),
        "sortino_excess": float(sortino),
        "certainty_equivalent_annualized": MONTHS_PER_YEAR * monthly_ce,
        "historical_var_95_monthly": var95,
        "historical_es_95_monthly": es95,
        "terminal_growth_multiple": terminal_growth,
        "terminal_wealth_from_1000": 1000.0 * terminal_growth,
        "mean_one_way_turnover_ex_initial": float(turnover.mean()) if turnover.notna().any() else np.nan,
        "median_one_way_turnover_ex_initial": float(turnover.median()) if turnover.notna().any() else np.nan,
        "total_transaction_cost_return_units": float(cost.sum()) if cost.notna().any() else np.nan,
        "annualized_gross_simple_mean": float(MONTHS_PER_YEAR * gross_simple.mean()) if gross_simple.notna().any() else np.nan,
        "skewness_excess": float(stats.skew(excess, bias=False)) if n > 2 else np.nan,
        "kurtosis_excess_pearson": float(stats.kurtosis(excess, fisher=False, bias=False)) if n > 3 else np.nan,
    }


def metrics_by_seed_strategy(
    returns: pd.DataFrame,
    *,
    utility_gamma: float = 1.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    period_label: str = "full",
) -> pd.DataFrame:
    required = {
        "master_seed", "strategy", "outcome_date", "net_simple_return",
        "net_excess_return",
    }
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"returns is missing columns: {sorted(missing)}")
    frame = returns.copy()
    frame["outcome_date"] = pd.to_datetime(frame["outcome_date"])
    if start is not None:
        frame = frame[frame["outcome_date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["outcome_date"] <= pd.Timestamp(end)]
    records: list[dict[str, float | int | str]] = []
    for (seed, strategy), group in frame.groupby(["master_seed", "strategy"], sort=True):
        metrics = compute_performance_metrics(group, utility_gamma=utility_gamma)
        records.append(
            {
                "master_seed": int(seed),
                "strategy": str(strategy),
                "period": period_label,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def default_subperiod_metrics(
    returns: pd.DataFrame,
    *,
    utility_gamma: float = 1.0,
) -> pd.DataFrame:
    """Full sample, development sample, final two-year holdout, and rate era."""

    blocks = [
        (None, None, "full"),
        (None, "2023-12-31", "development_through_2023"),
        ("2024-01-31", "2025-12-31", "holdout_2024_2025"),
        (None, "2021-12-31", "pre_2022_rate_shock"),
        ("2022-01-31", None, "2022_2025"),
    ]
    out = [
        metrics_by_seed_strategy(
            returns,
            utility_gamma=utility_gamma,
            start=start,
            end=end,
            period_label=label,
        )
        for start, end, label in blocks
    ]
    return pd.concat(out, ignore_index=True)


def summarize_metrics_across_seeds(
    metrics: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    id_cols = {"master_seed", "strategy", "period"}
    metric_cols = [
        column for column in metrics.columns
        if column not in id_cols and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    alpha = 1.0 - confidence
    records: list[dict[str, float | int | str]] = []
    for (strategy, period), group in metrics.groupby(["strategy", "period"], sort=True):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            if n == 0:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else np.nan
            if n > 1 and np.isfinite(sd):
                half = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1) * sd / np.sqrt(n))
                low, high = mean - half, mean + half
            else:
                low = high = mean
            records.append(
                {
                    "strategy": strategy,
                    "period": period,
                    "metric": metric,
                    "n_seeds": n,
                    "mean": mean,
                    "sd": sd,
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_positive": float((values > 0.0).mean()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(records)


def paired_seed_differences(
    metrics: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    metric_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected = metrics[metrics["strategy"].isin([treatment, control])].copy()
    if metric_columns is None:
        metric_columns = [
            column for column in selected.columns
            if column not in {"master_seed", "strategy", "period"}
            and pd.api.types.is_numeric_dtype(selected[column])
        ]
    records: list[dict[str, float | int | str]] = []
    for period, group in selected.groupby("period"):
        left = group[group["strategy"] == treatment].set_index("master_seed")
        right = group[group["strategy"] == control].set_index("master_seed")
        common = left.index.intersection(right.index)
        for metric in metric_columns:
            diff = left.loc[common, metric] - right.loc[common, metric]
            for seed, value in diff.items():
                records.append(
                    {
                        "master_seed": int(seed),
                        "period": period,
                        "treatment": treatment,
                        "control": control,
                        "metric": metric,
                        "difference": float(value),
                    }
                )
    return pd.DataFrame(records)
