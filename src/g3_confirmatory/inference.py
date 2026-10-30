"""Paired inference over master seeds and monthly return histories.

Seed variation measures algorithmic uncertainty; dependent monthly observations
measure historical-path uncertainty. Comparisons share seed and outcome dates.
Multiple tests use the strategy family supplied by the caller.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Callable, Iterable, Any
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .metrics import compute_performance_metrics


def automatic_hac_lags(n: int) -> int:
    """Andrews/Newey-West style monthly lag rule, at least one lag."""
    return max(1, int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))))


def newey_west_mean_test(
    differences: pd.Series | np.ndarray,
    *,
    maxlags: int | None = None,
) -> dict[str, float | int]:
    x = pd.Series(differences, dtype=float).dropna()
    if len(x) < 3:
        raise ValueError("At least three paired observations are required.")
    lags = automatic_hac_lags(len(x)) if maxlags is None else int(maxlags)
    model = sm.OLS(x.to_numpy(), np.ones((len(x), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    estimate = float(model.params[0])
    se = float(model.bse[0])
    t_value = estimate / se if se > 0.0 else np.nan
    p_value = float(2.0 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
    return {
        "n": int(len(x)),
        "maxlags": lags,
        "mean_monthly_difference": estimate,
        "annualized_arithmetic_difference": 12.0 * estimate,
        "hac_standard_error": se,
        "t_statistic": t_value,
        "p_value_two_sided": p_value,
    }


def stationary_bootstrap_indices(
    n: int,
    *,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if mean_block_length <= 1.0:
        raise ValueError("mean_block_length must exceed one.")
    restart_probability = 1.0 / float(mean_block_length)
    indices = np.empty(n, dtype=int)
    indices[0] = int(rng.integers(0, n))
    for t in range(1, n):
        if rng.random() < restart_probability:
            indices[t] = int(rng.integers(0, n))
        else:
            indices[t] = (indices[t - 1] + 1) % n
    return indices


def moving_block_bootstrap_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0 or block_length <= 0:
        raise ValueError("n and block_length must be positive.")
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    pieces = [
        (start + np.arange(block_length, dtype=int)) % n
        for start in starts
    ]
    return np.concatenate(pieces)[:n]


def _aligned_strategy_pair(
    returns: pd.DataFrame,
    treatment: str,
    control: str,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = returns[returns["master_seed"] == seed].copy()
    left = frame[frame["strategy"] == treatment].set_index("outcome_date").sort_index()
    right = frame[frame["strategy"] == control].set_index("outcome_date").sort_index()
    common = left.index.intersection(right.index)
    if len(common) < 12:
        raise ValueError(
            f"Only {len(common)} aligned months for seed {seed}, {treatment}, {control}."
        )
    return left.loc[common].reset_index(), right.loc[common].reset_index()


def _paired_block_bootstrap_one_seed(payload: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        seed_frame,
        treatment,
        control,
        n_bootstrap,
        mean_block_length,
        master_seed,
        metric_names,
    ) = payload
    seed = int(pd.unique(seed_frame["master_seed"])[0])
    left, right = _aligned_strategy_pair(
        seed_frame, treatment, control, seed=seed
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([int(master_seed), int(seed)])
    )
    n = len(left)
    observed_left = compute_performance_metrics(left)
    observed_right = compute_performance_metrics(right)
    draws = {metric: [] for metric in metric_names}
    for _ in range(int(n_bootstrap)):
        idx = stationary_bootstrap_indices(
            n, mean_block_length=mean_block_length, rng=rng
        )
        lm = compute_performance_metrics(left.iloc[idx].copy())
        rm = compute_performance_metrics(right.iloc[idx].copy())
        for metric in metric_names:
            draws[metric].append(float(lm[metric] - rm[metric]))
    records: list[dict[str, Any]] = []
    for metric in metric_names:
        values = np.asarray(draws[metric], dtype=float)
        values = values[np.isfinite(values)]
        observed = float(observed_left[metric] - observed_right[metric])
        records.append(
            {
                "master_seed": seed,
                "treatment": treatment,
                "control": control,
                "metric": metric,
                "observed_difference": observed,
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "p_value_two_sided": float(
                    2.0 * min((values <= 0.0).mean(), (values >= 0.0).mean())
                ),
                "n_bootstrap": int(len(values)),
                "mean_block_length": float(mean_block_length),
            }
        )
    return records


def paired_block_bootstrap_metrics(
    returns: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    n_bootstrap: int = 2000,
    mean_block_length: float = 6.0,
    master_seed: int = 20260322,
    metrics: Iterable[str] = (
        "annualized_excess_mean",
        "excess_sharpe",
        "certainty_equivalent_annualized",
        "maximum_drawdown",
    ),
    workers: int = 1,
    start_method: str = "spawn",
) -> pd.DataFrame:
    """Run a paired historical-path bootstrap within each master seed.
    
    Seed-derived random streams make results independent of worker count and order.
    """

    metric_names = tuple(metrics)
    payloads = [
        (
            group.copy(),
            treatment,
            control,
            int(n_bootstrap),
            float(mean_block_length),
            int(master_seed),
            metric_names,
        )
        for _, group in returns.groupby("master_seed", sort=True)
    ]
    if not payloads:
        return pd.DataFrame()
    workers = int(workers)
    if workers == 0:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    workers = max(1, min(workers, len(payloads)))
    if workers == 1:
        chunks = [_paired_block_bootstrap_one_seed(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context(start_method),
        ) as executor:
            chunks = list(executor.map(_paired_block_bootstrap_one_seed, payloads))
    return pd.DataFrame([record for chunk in chunks for record in chunk])

def _paired_block_bootstrap_many_one_seed(
    payload: tuple[Any, ...],
) -> list[dict[str, Any]]:
    (
        seed_frame,
        comparisons,
        n_bootstrap,
        mean_block_length,
        master_seed,
        metric_names,
    ) = payload
    records: list[dict[str, Any]] = []
    for treatment, control, label in comparisons:
        chunk = _paired_block_bootstrap_one_seed(
            (
                seed_frame,
                treatment,
                control,
                n_bootstrap,
                mean_block_length,
                master_seed,
                metric_names,
            )
        )
        for record in chunk:
            record["comparison"] = str(label)
        records.extend(chunk)
    return records


def paired_block_bootstrap_many(
    returns: pd.DataFrame,
    *,
    comparisons: Iterable[tuple[str, str, str]],
    n_bootstrap: int = 2000,
    mean_block_length: float = 6.0,
    master_seed: int = 20260322,
    metrics: Iterable[str] = (
        "annualized_excess_mean",
        "excess_sharpe",
        "certainty_equivalent_annualized",
        "maximum_drawdown",
    ),
    workers: int = 1,
    start_method: str = "spawn",
) -> pd.DataFrame:
    """Bootstrap several paired comparisons with one worker task per seed.
    
    The worker reuses its seed panel for all requested comparisons.
    """

    comparison_list = tuple(
        (str(treatment), str(control), str(label))
        for treatment, control, label in comparisons
    )
    if not comparison_list:
        return pd.DataFrame()
    metric_names = tuple(metrics)
    payloads = [
        (
            group.copy(),
            comparison_list,
            int(n_bootstrap),
            float(mean_block_length),
            int(master_seed),
            metric_names,
        )
        for _, group in returns.groupby("master_seed", sort=True)
    ]
    if not payloads:
        return pd.DataFrame()
    workers = int(workers)
    if workers == 0:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    workers = max(1, min(workers, len(payloads)))
    if workers == 1:
        chunks = [
            _paired_block_bootstrap_many_one_seed(payload)
            for payload in payloads
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context(start_method),
        ) as executor:
            chunks = list(
                executor.map(_paired_block_bootstrap_many_one_seed, payloads)
            )
    records = [record for chunk in chunks for record in chunk]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    first = ["comparison"]
    remaining = [column for column in frame.columns if column not in first]
    return frame[first + remaining]


def summarize_algorithmic_seed_effects(
    paired_differences: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Across-seed paired effects with t and sign/Wilcoxon summaries."""
    required = {"metric", "difference"}
    missing = required.difference(paired_differences.columns)
    if missing:
        raise ValueError(f"paired_differences missing columns: {sorted(missing)}")
    group_cols = [
        column for column in ("period", "treatment", "control", "metric")
        if column in paired_differences.columns
    ]
    alpha = 1.0 - confidence
    records: list[dict[str, float | int | str]] = []
    for keys, group in paired_differences.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        x = pd.to_numeric(group["difference"], errors="coerce").dropna()
        n = len(x)
        if n == 0:
            continue
        mean = float(x.mean())
        sd = float(x.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        t_stat = mean / se if np.isfinite(se) and se > 0.0 else np.nan
        t_p = float(2.0 * stats.t.sf(abs(t_stat), n - 1)) if n > 1 and np.isfinite(t_stat) else np.nan
        half = (
            float(stats.t.ppf(1.0 - alpha / 2.0, n - 1) * se)
            if n > 1 and np.isfinite(se) else 0.0
        )
        try:
            wilcoxon_p = float(stats.wilcoxon(x, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        records.append(
            {
                **key_map,
                "n_seeds": n,
                "mean_difference": mean,
                "sd_difference": sd,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "fraction_positive": float((x > 0.0).mean()),
                "t_statistic": t_stat,
                "t_p_value_two_sided": t_p,
                "wilcoxon_p_value_two_sided": wilcoxon_p,
            }
        )
    return pd.DataFrame(records)


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    benchmark_annualized_sharpe: float = 0.0,
    periods_per_year: float = 12.0,
) -> dict[str, float | int]:
    """Bailey-Lopez de Prado probabilistic Sharpe ratio approximation."""
    x = pd.Series(returns, dtype=float).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("At least four returns are required for PSR.")
    sd = float(np.std(x, ddof=1))
    if sd <= 0.0:
        raise ValueError("Return standard deviation is zero.")
    sr_monthly = float(np.mean(x) / sd)
    benchmark_monthly = float(benchmark_annualized_sharpe / np.sqrt(periods_per_year))
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=False, bias=False))
    denominator_sq = 1.0 - skew * sr_monthly + ((kurt - 1.0) / 4.0) * sr_monthly ** 2
    denominator_sq = max(denominator_sq, 1e-12)
    z = (sr_monthly - benchmark_monthly) * np.sqrt(n - 1.0) / np.sqrt(denominator_sq)
    return {
        "n": n,
        "annualized_sharpe": float(sr_monthly * np.sqrt(periods_per_year)),
        "benchmark_annualized_sharpe": benchmark_annualized_sharpe,
        "skewness": skew,
        "kurtosis_pearson": kurt,
        "psr_z": float(z),
        "psr_probability": float(stats.norm.cdf(z)),
    }


def effective_number_of_trials(return_panel: pd.DataFrame) -> float:
    """Eigenvalue participation ratio of the strategy correlation matrix."""
    corr = return_panel.corr().fillna(0.0).to_numpy(dtype=float)
    if corr.size == 0:
        return 1.0
    vals = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    vals = np.clip(vals, 0.0, None)
    denom = float(np.sum(vals * vals))
    return float((vals.sum() ** 2) / denom) if denom > 0.0 else 1.0


def expected_max_sharpe_benchmark(
    sharpe_standard_deviation: float,
    n_trials: float,
) -> float:
    """Expected maximum of approximately Gaussian trial Sharpe estimates."""
    n = max(float(n_trials), 1.000001)
    gamma = 0.5772156649015329
    first = stats.norm.ppf(1.0 - 1.0 / n)
    second = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sharpe_standard_deviation * ((1.0 - gamma) * first + gamma * second))


def deflated_sharpe_ratio_table(
    return_panel: pd.DataFrame,
    *,
    periods_per_year: float = 12.0,
    use_effective_trials: bool = True,
) -> pd.DataFrame:
    """Approximate DSR for a frozen strategy family.
    
    The benchmark uses the observed monthly Sharpe dispersion and, by default,
    the eigenvalue-based effective trial count. Trial dependence and selection
    still need to be documented in the run manifest.
    """
    panel = return_panel.dropna(how="all").copy()
    monthly_sharpes = panel.mean() / panel.std(ddof=1)
    trial_sd = float(monthly_sharpes.std(ddof=1))
    n_eff = effective_number_of_trials(panel) if use_effective_trials else float(panel.shape[1])
    benchmark_monthly = expected_max_sharpe_benchmark(trial_sd, n_eff)
    records = []
    for strategy in panel.columns:
        psr = probabilistic_sharpe_ratio(
            panel[strategy],
            benchmark_annualized_sharpe=benchmark_monthly * np.sqrt(periods_per_year),
            periods_per_year=periods_per_year,
        )
        records.append(
            {
                "strategy": strategy,
                "n_declared_trials": int(panel.shape[1]),
                "effective_trials": n_eff,
                "trial_monthly_sharpe_sd": trial_sd,
                "expected_max_monthly_sharpe": benchmark_monthly,
                "dsr_probability": psr["psr_probability"],
                "annualized_sharpe": psr["annualized_sharpe"],
                "approximation_note": "complete frozen family required; eigenvalue effective-trial approximation",
            }
        )
    return pd.DataFrame(records)


def _return_panel_for_seed(
    returns: pd.DataFrame,
    *,
    seed: int,
    strategies: Iterable[str],
    return_column: str = "net_excess_return",
) -> pd.DataFrame:
    frame = returns[
        (returns["master_seed"] == seed)
        & (returns["strategy"].isin(list(strategies)))
    ].copy()
    panel = frame.pivot(index="outcome_date", columns="strategy", values=return_column)
    return panel.sort_index().dropna()


def spa_test(
    return_panel: pd.DataFrame,
    *,
    benchmark: str,
    n_bootstrap: int = 5000,
    mean_block_length: float = 6.0,
    random_seed: int = 20260322,
) -> dict[str, object]:
    """Run a stationary-bootstrap SPA-style test against one benchmark.
    
    Positive differentials favor the candidate. Uses Hansen's consistent recentering
    for clearly inferior alternatives and a max studentized statistic.
    """
    if benchmark not in return_panel.columns:
        raise KeyError(f"Benchmark {benchmark!r} is not in the return panel.")
    candidates = [column for column in return_panel.columns if column != benchmark]
    if not candidates:
        raise ValueError("SPA requires at least one candidate strategy.")
    panel = return_panel[[benchmark, *candidates]].dropna()
    d = panel[candidates].sub(panel[benchmark], axis=0).to_numpy(dtype=float)
    n, m = d.shape
    means = d.mean(axis=0)
    sd = d.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.inf)
    observed_t = np.sqrt(n) * means / sd
    observed_max = float(np.max(observed_t))

    threshold = -np.sqrt(2.0 * np.log(np.log(max(n, 3))))
    consistent_mean = np.where(observed_t <= threshold, means, 0.0)
    rng = np.random.default_rng(random_seed)
    boot_max = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(
            n, mean_block_length=mean_block_length, rng=rng
        )
        boot_mean = d[idx].mean(axis=0) - means + consistent_mean
        boot_max[b] = float(np.max(np.sqrt(n) * boot_mean / sd))
    p_value = float((1.0 + np.sum(boot_max >= observed_max)) / (n_bootstrap + 1.0))
    return {
        "benchmark": benchmark,
        "candidates": candidates,
        "n_months": n,
        "n_candidates": m,
        "observed_max_t": observed_max,
        "spa_p_value": p_value,
        "mean_block_length": mean_block_length,
        "n_bootstrap": n_bootstrap,
        "candidate_statistics": pd.DataFrame(
            {
                "strategy": candidates,
                "mean_return_advantage": means,
                "studentized_statistic": observed_t,
            }
        ).sort_values("studentized_statistic", ascending=False),
    }


def romano_wolf_stepdown(
    return_panel: pd.DataFrame,
    *,
    benchmark: str,
    n_bootstrap: int = 5000,
    mean_block_length: float = 6.0,
    random_seed: int = 20260322,
) -> pd.DataFrame:
    """One-sided Romano-Wolf stepdown p-values for return superiority."""
    if benchmark not in return_panel.columns:
        raise KeyError(f"Benchmark {benchmark!r} is not in the return panel.")
    candidates = [column for column in return_panel.columns if column != benchmark]
    panel = return_panel[[benchmark, *candidates]].dropna()
    d = panel[candidates].sub(panel[benchmark], axis=0).to_numpy(dtype=float)
    n, m = d.shape
    means = d.mean(axis=0)
    sd = d.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.inf)
    observed = np.sqrt(n) * means / sd

    rng = np.random.default_rng(random_seed)
    boot_t = np.empty((n_bootstrap, m), dtype=float)
    centered = d - means
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(
            n, mean_block_length=mean_block_length, rng=rng
        )
        boot_t[b] = np.sqrt(n) * centered[idx].mean(axis=0) / sd

    order = np.argsort(-observed)
    adjusted = np.zeros(m, dtype=float)
    running = 0.0
    remaining = list(order)
    for idx in order:
        max_boot = np.max(boot_t[:, remaining], axis=1)
        raw = float((1.0 + np.sum(max_boot >= observed[idx])) / (n_bootstrap + 1.0))
        running = max(running, raw)
        adjusted[idx] = running
        remaining.remove(idx)
    return pd.DataFrame(
        {
            "strategy": candidates,
            "mean_return_advantage": means,
            "studentized_statistic": observed,
            "romano_wolf_p_value_one_sided": adjusted,
        }
    ).sort_values("studentized_statistic", ascending=False)


def _multiple_testing_one_seed(payload: tuple[Any, ...]):
    (
        seed_frame,
        strategy_list,
        benchmark,
        return_column,
        n_bootstrap,
        mean_block_length,
        random_seed,
    ) = payload
    seed = int(pd.unique(seed_frame["master_seed"])[0])
    panel = _return_panel_for_seed(
        seed_frame,
        seed=seed,
        strategies=strategy_list,
        return_column=return_column,
    )
    dsr = deflated_sharpe_ratio_table(panel)
    dsr.insert(0, "master_seed", seed)
    spa = spa_test(
        panel,
        benchmark=benchmark,
        n_bootstrap=n_bootstrap,
        mean_block_length=mean_block_length,
        random_seed=random_seed + seed,
    )
    spa_record = {
        "master_seed": seed,
        "benchmark": benchmark,
        "n_months": spa["n_months"],
        "n_candidates": spa["n_candidates"],
        "observed_max_t": spa["observed_max_t"],
        "spa_p_value": spa["spa_p_value"],
    }
    rw = romano_wolf_stepdown(
        panel,
        benchmark=benchmark,
        n_bootstrap=n_bootstrap,
        mean_block_length=mean_block_length,
        random_seed=random_seed + 100_000 + seed,
    )
    rw.insert(0, "master_seed", seed)
    return dsr, spa_record, rw


def multiple_testing_by_seed(
    returns: pd.DataFrame,
    *,
    strategies: Iterable[str],
    benchmark: str = "benchmark__1N",
    return_column: str = "net_excess_return",
    n_bootstrap: int = 5000,
    mean_block_length: float = 6.0,
    random_seed: int = 20260322,
    workers: int = 1,
    start_method: str = "spawn",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run PSR/DSR, SPA, and Romano-Wolf independently by master seed."""

    strategy_list = list(dict.fromkeys([*strategies, benchmark]))
    payloads = [
        (
            group.copy(),
            strategy_list,
            benchmark,
            return_column,
            int(n_bootstrap),
            float(mean_block_length),
            int(random_seed),
        )
        for _, group in returns.groupby("master_seed", sort=True)
    ]
    if not payloads:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    workers = int(workers)
    if workers == 0:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    workers = max(1, min(workers, len(payloads)))
    if workers == 1:
        results = [_multiple_testing_one_seed(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context(start_method),
        ) as executor:
            results = list(executor.map(_multiple_testing_one_seed, payloads))
    dsr_frames = [result[0] for result in results]
    spa_records = [result[1] for result in results]
    rw_frames = [result[2] for result in results]
    return (
        pd.concat(dsr_frames, ignore_index=True),
        pd.DataFrame(spa_records),
        pd.concat(rw_frames, ignore_index=True),
    )

