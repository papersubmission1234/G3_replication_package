#!/usr/bin/env python3
"""Build monthly forecast calibration and daily realized-risk diagnostics.

The optimization horizon remains one month. Daily and 21-day statistics are
reported separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from _common import ROOT, load_context
from g3_confirmatory.metrics import maximum_drawdown
from g3_confirmatory.optimizer import empirical_var_cvar
from g3_confirmatory.realized_returns import read_daily_adjusted_close
from g3_confirmatory.reporting import mean_ci


def _experiment_id(seed_dir: Path, output_root: Path) -> str:
    return seed_dir.parent.relative_to(output_root).as_posix()


def _discover_seed_dirs(output_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in output_root.rglob("seed_*/returns.csv")
        if (path.parent / "weights.csv.gz").exists()
    )


def _daily_portfolio_path(
    *,
    daily_prices: pd.DataFrame,
    target_weights: pd.Series,
    formation: pd.Timestamp,
    outcome: pd.Timestamp,
    monthly_rf: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    weights = pd.to_numeric(target_weights, errors="coerce").dropna().astype(float)
    weights = weights[weights > 0.0]
    if weights.empty:
        return pd.DataFrame(), {"covered_weight": 0.0}

    tickers = weights.index.intersection(daily_prices.columns)
    before = daily_prices.loc[daily_prices.index <= formation, tickers].tail(15)
    if before.empty:
        return pd.DataFrame(), {"covered_weight": 0.0}
    start = before.ffill().iloc[-1]
    valid = start.notna() & (start > 0.0)
    tickers = tickers[valid]
    covered_weight = float(weights.reindex(tickers).sum())
    if tickers.empty or covered_weight <= 0.0:
        return pd.DataFrame(), {"covered_weight": covered_weight}

    during_raw = daily_prices.loc[
        (daily_prices.index > formation) & (daily_prices.index <= outcome), tickers
    ].copy()
    if during_raw.empty:
        return pd.DataFrame(), {"covered_weight": covered_weight}

    start_row = pd.DataFrame(
        [start.reindex(tickers).to_numpy(dtype=float)],
        index=pd.DatetimeIndex([formation]),
        columns=tickers,
    )
    combined = pd.concat([start_row, during_raw]).sort_index()
    filled = combined.ffill()
    asset_returns = filled.pct_change(fill_method=None).iloc[1:]
    n_days = len(asset_returns)
    rf_daily = (
        float((1.0 + monthly_rf) ** (1.0 / max(1, n_days)) - 1.0)
        if np.isfinite(monthly_rf) and monthly_rf > -1.0
        else 0.0
    )

    # carry proceeds as cash after the final holding-period quote
    for ticker in tickers:
        observed = during_raw[ticker].dropna()
        if observed.empty:
            asset_returns.loc[:, ticker] = rf_daily
        else:
            last_observed = observed.index[-1]
            asset_returns.loc[asset_returns.index > last_observed, ticker] = rf_daily

    current_weights = weights.reindex(tickers).fillna(0.0).astype(float)
    cash_weight = max(0.0, 1.0 - float(current_weights.sum()))
    records: list[dict[str, float | pd.Timestamp]] = []
    for date, row in asset_returns.iterrows():
        values = row.fillna(0.0).to_numpy(dtype=float)
        w = current_weights.to_numpy(dtype=float)
        gross = float(w @ values + cash_weight * rf_daily)
        records.append({"date": pd.Timestamp(date), "gross_daily_return": gross})
        asset_values = w * (1.0 + values)
        cash_value = cash_weight * (1.0 + rf_daily)
        total = float(asset_values.sum() + cash_value)
        if total > 0.0:
            current_weights = pd.Series(asset_values / total, index=tickers)
            cash_weight = cash_value / total

    frame = pd.DataFrame(records)
    return frame, {
        "covered_weight": covered_weight,
        "n_daily_observations": float(n_days),
        "daily_rf": rf_daily,
    }


def _apply_monthly_net_reconciliation(
    daily: pd.DataFrame,
    *,
    recorded_gross: float,
    recorded_net: float,
) -> pd.DataFrame:
    out = daily.copy()
    out["net_daily_return"] = out["gross_daily_return"]
    if out.empty:
        return out
    gross_factor = float(np.prod(1.0 + out["gross_daily_return"].to_numpy(dtype=float)))
    if (
        np.isfinite(recorded_net)
        and recorded_net > -1.0
        and gross_factor > 0.0
    ):
        adjustment = (1.0 + float(recorded_net)) / gross_factor
        first = out.index[0]
        out.loc[first, "net_daily_return"] = (
            (1.0 + out.loc[first, "gross_daily_return"]) * adjustment - 1.0
        )
    elif np.isfinite(recorded_gross) and np.isfinite(recorded_net):
        first = out.index[0]
        out.loc[first, "net_daily_return"] -= float(recorded_gross - recorded_net)
    return out


def _daily_metrics(group: pd.DataFrame) -> dict[str, float]:
    gross = pd.to_numeric(group["gross_daily_return"], errors="coerce").dropna()
    net = pd.to_numeric(group["net_daily_return"], errors="coerce").dropna()
    out: dict[str, float] = {"n_daily_observations": float(len(net))}
    for label, series in (("gross", gross), ("net", net)):
        if series.empty:
            continue
        r95 = empirical_var_cvar(series.to_numpy(dtype=float), alpha=0.95)
        r975 = empirical_var_cvar(series.to_numpy(dtype=float), alpha=0.975)
        vol = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        out.update(
            {
                f"daily_var_95_{label}": float(r95["var"]),
                f"daily_es_95_{label}": float(r95["cvar"]),
                f"daily_es_975_{label}": float(r975["cvar"]),
                f"annualized_mean_{label}": 252.0 * float(series.mean()),
                f"annualized_volatility_{label}": np.sqrt(252.0) * vol,
                f"annualized_sharpe_zero_reference_{label}": (
                    np.sqrt(252.0) * float(series.mean()) / vol
                    if np.isfinite(vol) and vol > 0.0
                    else np.nan
                ),
                f"maximum_drawdown_daily_{label}": maximum_drawdown(series),
            }
        )
        rolling = (1.0 + series).rolling(21).apply(np.prod, raw=True) - 1.0
        rolling = rolling.dropna()
        if not rolling.empty:
            risk21 = empirical_var_cvar(rolling.to_numpy(dtype=float), alpha=0.95)
            out[f"overlapping_21d_es_95_{label}"] = float(risk21["cvar"])
    return out


def _build_daily_paths(
    *,
    output_root: Path,
    daily_prices: pd.DataFrame,
    risk_free: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[pd.DataFrame] = []
    audit_records: list[dict[str, object]] = []
    for seed_dir in _discover_seed_dirs(output_root):
        experiment_id = _experiment_id(seed_dir, output_root)
        returns = pd.read_csv(
            seed_dir / "returns.csv",
            parse_dates=["formation_date", "outcome_date"],
        )
        weights = pd.read_csv(
            seed_dir / "weights.csv.gz",
            parse_dates=["formation_date", "outcome_date"],
        )
        if returns.empty or weights.empty:
            continue
        seed = int(returns["master_seed"].iloc[0])
        indexed_returns = returns.set_index(["strategy", "formation_date", "outcome_date"])
        for (strategy, formation, outcome), wgroup in weights.groupby(
            ["strategy", "formation_date", "outcome_date"], sort=False
        ):
            key = (strategy, pd.Timestamp(formation), pd.Timestamp(outcome))
            if key not in indexed_returns.index:
                continue
            row = indexed_returns.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            target = wgroup.set_index("ticker")["weight"]
            monthly_rf = float(risk_free.get(pd.Timestamp(outcome), 0.0))
            path, audit = _daily_portfolio_path(
                daily_prices=daily_prices,
                target_weights=target,
                formation=pd.Timestamp(formation),
                outcome=pd.Timestamp(outcome),
                monthly_rf=monthly_rf,
            )
            audit_record = {
                "experiment_id": experiment_id,
                "master_seed": seed,
                "strategy": strategy,
                "formation_date": formation,
                "outcome_date": outcome,
                **audit,
            }
            if path.empty:
                audit_record["status"] = "no_daily_path"
                audit_records.append(audit_record)
                continue
            path = _apply_monthly_net_reconciliation(
                path,
                recorded_gross=float(row.get("gross_simple_return", np.nan)),
                recorded_net=float(row.get("net_simple_return", np.nan)),
            )
            gross_compounded = float(np.prod(1.0 + path["gross_daily_return"]) - 1.0)
            net_compounded = float(np.prod(1.0 + path["net_daily_return"]) - 1.0)
            audit_record.update(
                {
                    "status": "built",
                    "daily_gross_compounded": gross_compounded,
                    "recorded_gross_simple_return": float(
                        row.get("gross_simple_return", np.nan)
                    ),
                    "daily_net_compounded": net_compounded,
                    "recorded_net_simple_return": float(
                        row.get("net_simple_return", np.nan)
                    ),
                    "gross_reconciliation_error": gross_compounded
                    - float(row.get("gross_simple_return", np.nan)),
                    "net_reconciliation_error": net_compounded
                    - float(row.get("net_simple_return", np.nan)),
                }
            )
            path.insert(0, "experiment_id", experiment_id)
            path.insert(1, "master_seed", seed)
            path.insert(2, "strategy", strategy)
            path.insert(3, "formation_date", formation)
            path.insert(4, "outcome_date", outcome)
            daily_records.append(path)
            audit_records.append(audit_record)
    daily = pd.concat(daily_records, ignore_index=True) if daily_records else pd.DataFrame()
    audit = pd.DataFrame(audit_records)
    return daily, audit


def _risk_map_calibration(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare monthly VaR/CVaR forecasts with realized losses.

    VaR uses exceedance frequency at 1-alpha. CVaR uses tail-severity gaps
    because a conditional tail mean is not a quantile.
    """

    records: list[pd.DataFrame] = []
    for return_path in sorted(output_root.rglob("seed_*/returns.csv")):
        seed_dir = return_path.parent
        diagnostics_path = seed_dir / "optimizer_diagnostics.csv"
        if not diagnostics_path.exists():
            continue
        returns = pd.read_csv(
            return_path, parse_dates=["formation_date", "outcome_date"]
        )
        diagnostics = pd.read_csv(
            diagnostics_path, parse_dates=["formation_date"]
        )
        if diagnostics.empty:
            continue

        if "risk_scenario_var" not in diagnostics.columns and "scenario_var" in diagnostics:
            diagnostics["risk_scenario_var"] = diagnostics["scenario_var"]
        if "risk_scenario_cvar" not in diagnostics.columns and "scenario_cvar" in diagnostics:
            diagnostics["risk_scenario_cvar"] = diagnostics["scenario_cvar"]
        if "risk_scenario_var" not in diagnostics.columns:
            diagnostics["risk_scenario_var"] = np.nan
        if "risk_scenario_cvar" not in diagnostics.columns:
            diagnostics["risk_scenario_cvar"] = np.nan
        if "cvar_budget" not in diagnostics.columns:
            diagnostics["cvar_budget"] = np.nan

        experiment_id = _experiment_id(seed_dir, output_root)
        merged = diagnostics.merge(
            returns[
                [
                    "master_seed",
                    "strategy",
                    "formation_date",
                    "outcome_date",
                    "gross_simple_return",
                    "net_simple_return",
                ]
            ],
            on=["master_seed", "strategy", "formation_date"],
            how="inner",
        )
        if merged.empty:
            continue
        merged.insert(0, "experiment_id", experiment_id)
        for label in ("gross", "net"):
            loss = -pd.to_numeric(
                merged[f"{label}_simple_return"], errors="coerce"
            )
            var = pd.to_numeric(merged["risk_scenario_var"], errors="coerce")
            cvar = pd.to_numeric(merged["risk_scenario_cvar"], errors="coerce")
            merged[f"realized_{label}_loss"] = loss
            merged[f"{label}_var_exceedance"] = (
                loss.notna() & var.notna() & (loss > var)
            )
            merged[f"{label}_cvar_exceedance"] = (
                loss.notna() & cvar.notna() & (loss > cvar)
            )
            merged[f"{label}_loss_minus_var"] = loss - var
            merged[f"{label}_loss_minus_cvar"] = loss - cvar
        records.append(merged)

    panel = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if panel.empty:
        return panel, pd.DataFrame()

    summary_records: list[dict[str, object]] = []
    group_cols = ["experiment_id", "strategy"]
    for keys, group in panel.groupby(group_cols, sort=True):
        alpha = pd.to_numeric(group.get("cvar_alpha"), errors="coerce").mean()
        predicted_var = pd.to_numeric(group["risk_scenario_var"], errors="coerce")
        predicted_cvar = pd.to_numeric(group["risk_scenario_cvar"], errors="coerce")
        row: dict[str, object] = {
            "experiment_id": keys[0],
            "strategy": keys[1],
            "n_strategy_months": int(len(group)),
            "n_seeds": int(group["master_seed"].nunique())
            if "master_seed" in group.columns else np.nan,
            "cvar_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
            "nominal_var_exceedance_rate": (
                float(1.0 - alpha) if np.isfinite(alpha) else np.nan
            ),
            "mean_predicted_var": float(predicted_var.mean()),
            "mean_predicted_cvar": float(predicted_cvar.mean()),
            "mean_cvar_budget": float(
                pd.to_numeric(group["cvar_budget"], errors="coerce").mean()
            ),
            "cvar_budget_binding_rate": float(
                group.get("cvar_budget_binding", pd.Series(False, index=group.index))
                .astype("boolean")
                .fillna(False)
                .mean()
            ),
            "mean_cvar_shadow_value": float(
                pd.to_numeric(
                    group.get(
                        "cvar_budget_shadow_value_max_objective",
                        pd.Series(np.nan, index=group.index),
                    ),
                    errors="coerce",
                ).mean()
            ),
        }
        for label in ("gross", "net"):
            var_exceed = group[f"{label}_var_exceedance"].astype(bool)
            cvar_exceed = group[f"{label}_cvar_exceedance"].astype(bool)
            var_rate = float(var_exceed.mean())
            row[f"{label}_var_exceedance_rate"] = var_rate
            row[f"{label}_var_calibration_gap"] = (
                var_rate - float(1.0 - alpha) if np.isfinite(alpha) else np.nan
            )
            row[f"{label}_cvar_exceedance_rate"] = float(cvar_exceed.mean())
            tail = group.loc[var_exceed]
            if tail.empty:
                row[f"mean_realized_{label}_loss_when_var_exceeded"] = np.nan
                row[f"mean_predicted_cvar_when_var_exceeded"] = np.nan
                row[f"{label}_tail_severity_gap"] = np.nan
            else:
                realized_tail = pd.to_numeric(
                    tail[f"realized_{label}_loss"], errors="coerce"
                ).mean()
                predicted_tail_cvar = pd.to_numeric(
                    tail["risk_scenario_cvar"], errors="coerce"
                ).mean()
                row[f"mean_realized_{label}_loss_when_var_exceeded"] = float(
                    realized_tail
                )
                row[f"mean_predicted_cvar_when_var_exceeded"] = float(
                    predicted_tail_cvar
                )
                row[f"{label}_tail_severity_gap"] = float(
                    realized_tail - predicted_tail_cvar
                )
        summary_records.append(row)
    return panel, pd.DataFrame(summary_records)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    args = parser.parse_args()

    cfg, paths, data, _, _, _ = load_context(args.config)
    raw = dict(cfg.get("raw_data", {}) or {})
    adjusted = raw.get("adjusted_close_csv")
    if not adjusted:
        raise ValueError("raw_data.adjusted_close_csv is required.")
    daily_prices = read_daily_adjusted_close(adjusted)
    output_root = paths["output_root"]
    out = output_root / "risk_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    daily, audit = _build_daily_paths(
        output_root=output_root,
        daily_prices=daily_prices,
        risk_free=data.risk_free,
    )
    if not daily.empty:
        daily.to_csv(out / "portfolio_daily_returns.csv.gz", index=False, compression="gzip")
        metric_records: list[dict[str, object]] = []
        for keys, group in daily.groupby(
            ["experiment_id", "master_seed", "strategy"], sort=True
        ):
            metric_records.append(
                {
                    "experiment_id": keys[0],
                    "master_seed": int(keys[1]),
                    "strategy": keys[2],
                    **_daily_metrics(group),
                }
            )
        metrics = pd.DataFrame(metric_records)
        metrics.to_csv(out / "daily_risk_metrics_by_seed_strategy.csv", index=False)
        summary_records: list[dict[str, object]] = []
        metric_columns = [
            column for column in metrics.columns
            if column not in {"experiment_id", "master_seed", "strategy"}
        ]
        for (experiment_id, strategy), group in metrics.groupby(
            ["experiment_id", "strategy"], sort=True
        ):
            for metric in metric_columns:
                summary_records.append(
                    {
                        "experiment_id": experiment_id,
                        "strategy": strategy,
                        "metric": metric,
                        "n_seeds": int(group["master_seed"].nunique()),
                        **mean_ci(group[metric]),
                    }
                )
        summary = pd.DataFrame(summary_records)
        summary.to_csv(out / "daily_risk_metric_summary.csv", index=False)
    audit.to_csv(out / "daily_path_reconciliation_audit.csv", index=False)

    calibration, calibration_summary = _risk_map_calibration(output_root)
    calibration.to_csv(out / "monthly_cvar_calibration_panel.csv", index=False)
    calibration_summary.to_csv(out / "monthly_cvar_calibration_summary.csv", index=False)
    print(f"Risk diagnostics written to {out}")


if __name__ == "__main__":
    main()
