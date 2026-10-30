"""Hard whole-model sampling and row-level predictive scenario mixing."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _common_columns(scenarios: Mapping[str, pd.DataFrame], arm_names: list[str]) -> list[str]:
    if not arm_names:
        return []
    common = list(scenarios[arm_names[0]].columns.astype(str))
    for name in arm_names[1:]:
        available = set(scenarios[name].columns.astype(str))
        common = [column for column in common if column in available]
    return common


def normalize_weights(
    weights: Mapping[str, float],
    available_arms: list[str],
) -> np.ndarray:
    raw = np.array([max(float(weights.get(name, 0.0)), 0.0) for name in available_arms])
    if raw.sum() <= 0.0:
        raw[:] = 1.0
    return raw / raw.sum()


def hard_arm_from_uniform(
    weights: Mapping[str, float],
    available_arms: list[str],
    uniform: float,
) -> str:
    probs = normalize_weights(weights, available_arms)
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, min(max(float(uniform), 0.0), 1.0 - 1e-15), side="right"))
    return available_arms[min(idx, len(available_arms) - 1)]


def mix_scenarios_crn(
    scenarios: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    n_rows: int,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Mix scenario rows using shared arm and row uniforms.
    
    Reusing these uniforms across posterior weights gives a common-random-number comparison.
    """

    available = [name for name, frame in scenarios.items() if frame is not None and not frame.empty]
    if not available:
        raise ValueError("No successful arm scenarios are available.")
    common = _common_columns(scenarios, available)
    if len(common) < 5:
        raise ValueError("Fewer than five common assets across arm scenarios.")
    if len(arm_uniforms) < n_rows or len(row_uniforms) < n_rows:
        raise ValueError("Not enough common random uniforms for scenario mixing.")

    probs = normalize_weights(weights, available)
    cdf = np.cumsum(probs)
    assignments = np.searchsorted(cdf, np.clip(arm_uniforms[:n_rows], 0.0, 1.0 - 1e-15), side="right")
    assignments = np.minimum(assignments, len(available) - 1)

    out = np.empty((n_rows, len(common)), dtype=float)
    counts: dict[str, int] = {name: 0 for name in available}
    for row in range(n_rows):
        arm_idx = int(assignments[row])
        name = available[arm_idx]
        frame = scenarios[name].loc[:, common]
        source_idx = min(int(np.floor(row_uniforms[row] * len(frame))), len(frame) - 1)
        out[row] = frame.iloc[source_idx].to_numpy(dtype=float)
        counts[name] += 1

    mixed = pd.DataFrame(out, columns=common)
    diagnostics = {
        "available_arms": available,
        "normalized_weights": dict(zip(available, probs.tolist())),
        "realized_row_counts": counts,
        "realized_row_shares": {name: count / n_rows for name, count in counts.items()},
        "n_rows": int(n_rows),
        "n_assets": int(len(common)),
    }
    return mixed, diagnostics


def bounded_baseline_weights(
    weights: Mapping[str, float],
    available_arms: list[str],
    *,
    baseline_arm: str,
    max_nonbaseline_weight: float,
) -> dict[str, float]:
    """Cap total nonbaseline probability while preserving its relative shares."""

    if baseline_arm not in available_arms:
        raise ValueError(f"Baseline arm {baseline_arm!r} is not available.")
    cap = float(max_nonbaseline_weight)
    if not 0.0 <= cap <= 1.0:
        raise ValueError("max_nonbaseline_weight must lie in [0, 1].")
    p = normalize_weights(weights, available_arms)
    raw = dict(zip(available_arms, p.tolist()))
    nonbaseline = [name for name in available_arms if name != baseline_arm]
    total_nonbaseline = sum(raw[name] for name in nonbaseline)
    if total_nonbaseline <= cap + 1e-15:
        return raw
    if total_nonbaseline <= 1e-15:
        return {name: 1.0 if name == baseline_arm else 0.0 for name in available_arms}
    scale = cap / total_nonbaseline
    out = {name: raw[name] * scale for name in nonbaseline}
    out[baseline_arm] = 1.0 - cap
    return out


def anchored_baseline_weights(
    weights: Mapping[str, float],
    available_arms: list[str],
    *,
    baseline_arm: str,
    max_nonbaseline_weight: float,
    active_nonbaseline: set[str] | frozenset[str] | None = None,
) -> dict[str, float]:
    """Return excluded experts' probability mass to the baseline.
    
    The surviving experts keep their original weights. This avoids increasing their
    exposure when a coalition member is removed.
    """

    if baseline_arm not in available_arms:
        raise ValueError(f"Baseline arm {baseline_arm!r} is not available.")
    cap = float(max_nonbaseline_weight)
    if not 0.0 <= cap <= 1.0:
        raise ValueError("max_nonbaseline_weight must lie in [0, 1].")

    p = normalize_weights(weights, available_arms)
    raw = dict(zip(available_arms, p.tolist()))
    optional = [name for name in available_arms if name != baseline_arm]
    active = set(optional) if active_nonbaseline is None else set(active_nonbaseline)
    unknown = active.difference(optional)
    if unknown:
        raise ValueError(f"Unknown active nonbaseline arms: {sorted(unknown)}")

    # keep surviving weights; return excluded mass to the baseline
    raw_optional_total = float(sum(raw[name] for name in optional))
    scale = (
        min(1.0, cap / raw_optional_total)
        if raw_optional_total > 1e-15
        else 0.0
    )
    full_optional = {name: raw[name] * scale for name in optional}
    out = {
        name: (full_optional[name] if name in active else 0.0)
        for name in optional
    }
    total_nonbaseline = float(sum(out.values()))
    out[baseline_arm] = 1.0 - total_nonbaseline
    return {name: float(out.get(name, 0.0)) for name in available_arms}


def baseline_preserving_mean_mix_scenarios(
    scenarios: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    baseline_arm: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pool expert means with centered draws from the baseline.
    
    The sample covariance stays equal to the baseline covariance, up to rounding.
    """

    if baseline_arm not in scenarios:
        raise ValueError(f"Baseline arm {baseline_arm!r} is not available.")
    names = [
        name for name in scenarios
        if float(weights.get(name, 0.0)) > 1e-15
    ]
    if baseline_arm not in names:
        names.insert(0, baseline_arm)
    common = _common_columns(scenarios, names)
    if len(common) < 5:
        raise ValueError("Fewer than five common assets for baseline-preserving mean mix.")

    probs = normalize_weights(weights, names)
    normalized = dict(zip(names, probs.tolist()))
    base = scenarios[baseline_arm].reindex(columns=common).astype(float)
    base_mean = base.mean(axis=0)
    target_mean = pd.Series(0.0, index=common, dtype=float)
    for name, probability in normalized.items():
        frame = scenarios[name].reindex(columns=common).astype(float)
        target_mean = target_mean.add(probability * frame.mean(axis=0), fill_value=0.0)
    mixed = base.subtract(base_mean, axis=1).add(target_mean, axis=1)
    max_cov_error = float(
        np.max(
            np.abs(
                mixed.cov().to_numpy(dtype=float)
                - base.cov().to_numpy(dtype=float)
            )
        )
    )
    return mixed, {
        "overlay_type": "baseline_preserving_multi_mean",
        "baseline_arm": baseline_arm,
        "normalized_weights": normalized,
        "total_nonbaseline_weight": float(
            1.0 - normalized.get(baseline_arm, 0.0)
        ),
        "baseline_mean_norm": float(np.linalg.norm(base_mean.to_numpy(dtype=float))),
        "target_mean_norm": float(np.linalg.norm(target_mean.to_numpy(dtype=float))),
        "max_covariance_error": max_cov_error,
        "n_assets": int(len(common)),
    }


def mean_only_overlay_scenarios(
    baseline_scenarios: pd.DataFrame,
    signal_scenarios: pd.DataFrame,
    fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Blend predictive means while keeping the baseline sample covariance.
    
    At fraction=0 the baseline scenarios are unchanged. At fraction=1 their means
    are replaced by the signal means; the centered draws stay the same.
    """

    lam = float(fraction)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("Overlay fraction must lie in [0, 1].")
    common = baseline_scenarios.columns.astype(str).intersection(
        signal_scenarios.columns.astype(str)
    ).sort_values()
    if len(common) < 5:
        raise ValueError("Fewer than five common assets for mean overlay.")
    base = baseline_scenarios.reindex(columns=common).astype(float)
    signal = signal_scenarios.reindex(columns=common).astype(float)
    base_mean = base.mean(axis=0)
    signal_mean = signal.mean(axis=0)
    target_mean = (1.0 - lam) * base_mean + lam * signal_mean
    overlaid = base.subtract(base_mean, axis=1).add(target_mean, axis=1)
    max_cov_error = float(
        np.max(
            np.abs(
                overlaid.cov().to_numpy(dtype=float)
                - base.cov().to_numpy(dtype=float)
            )
        )
    )
    return overlaid, {
        "overlay_type": "mean_only",
        "overlay_fraction": lam,
        "baseline_mean_norm": float(np.linalg.norm(base_mean.to_numpy(dtype=float))),
        "signal_mean_norm": float(np.linalg.norm(signal_mean.to_numpy(dtype=float))),
        "target_mean_norm": float(np.linalg.norm(target_mean.to_numpy(dtype=float))),
        "max_covariance_error": max_cov_error,
        "n_assets": int(len(common)),
    }


def row_overlay_scenarios_crn(
    scenarios: Mapping[str, pd.DataFrame],
    *,
    baseline_arm: str,
    signal_arm: str,
    fraction: float,
    n_rows: int,
    arm_uniforms: np.ndarray,
    row_uniforms: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    lam = float(fraction)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("Overlay fraction must lie in [0, 1].")
    if baseline_arm == signal_arm:
        raise ValueError("baseline_arm and signal_arm must differ.")
    mixed, diagnostics = mix_scenarios_crn(
        scenarios,
        {baseline_arm: 1.0 - lam, signal_arm: lam},
        n_rows=n_rows,
        arm_uniforms=arm_uniforms,
        row_uniforms=row_uniforms,
    )
    diagnostics.update(
        {
            "overlay_type": "row_mix",
            "overlay_fraction": lam,
            "baseline_arm": baseline_arm,
            "signal_arm": signal_arm,
        }
    )
    return mixed, diagnostics


def scenario_similarity_matrix(
    scenarios: Mapping[str, pd.DataFrame],
    arm_names: list[str] | None = None,
    *,
    mean_weight: float = 0.5,
) -> pd.DataFrame:
    """Operational redundancy from expected-return and covariance similarity."""

    names = list(scenarios) if arm_names is None else list(arm_names)
    if not names:
        raise ValueError("At least one arm is required.")
    common = _common_columns(scenarios, names)
    if len(common) < 5:
        raise ValueError("Fewer than five common assets for similarity calculation.")
    w = float(mean_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError("mean_weight must lie in [0, 1].")

    means = {
        name: scenarios[name].reindex(columns=common).mean(axis=0).to_numpy(dtype=float)
        for name in names
    }
    covs = {
        name: scenarios[name].reindex(columns=common).cov().to_numpy(dtype=float)
        for name in names
    }
    out = np.eye(len(names), dtype=float)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            a, b = means[left], means[right]
            a0, b0 = a - a.mean(), b - b.mean()
            denom = np.linalg.norm(a0) * np.linalg.norm(b0)
            mean_sim = float(a0 @ b0 / denom) if denom > 1e-15 else 0.0
            ca, cb = covs[left], covs[right]
            cov_denom = np.linalg.norm(ca) * np.linalg.norm(cb)
            cov_sim = float(np.sum(ca * cb) / cov_denom) if cov_denom > 1e-15 else 0.0
            sim = w * max(mean_sim, 0.0) + (1.0 - w) * max(cov_sim, 0.0)
            out[i, j] = out[j, i] = min(max(sim, 0.0), 1.0)
    return pd.DataFrame(out, index=names, columns=names)


def complementarity_sparse_weights(
    posterior_weights: Mapping[str, float],
    scenarios: Mapping[str, pd.DataFrame],
    *,
    redundancy_penalty: float = 1.0,
    concentration_reward: float = 0.05,
    minimum_reported_weight: float = 0.01,
) -> tuple[dict[str, float], dict[str, object]]:
    """Adjust weights for redundancy and sparsity using deterministic SLSQP.
    
    The objective combines KL divergence, a pairwise redundancy penalty, and a
    concentration reward. The optimization need not be globally convex.
    """

    from scipy.optimize import minimize

    names = [name for name, frame in scenarios.items() if frame is not None and not frame.empty]
    p = normalize_weights(posterior_weights, names)
    sim = scenario_similarity_matrix(scenarios, names).to_numpy(dtype=float)
    lam = max(float(redundancy_penalty), 0.0)
    tau = max(float(concentration_reward), 0.0)
    eps = 1e-12

    def objective(q: np.ndarray) -> float:
        q = np.maximum(q, eps)
        q = q / q.sum()
        kl = float(np.sum(q * (np.log(q) - np.log(np.maximum(p, eps)))))
        redundancy = float(q @ sim @ q)
        concentration = float(np.sum(q * q))
        return kl + lam * redundancy - tau * concentration

    result = minimize(
        objective,
        x0=p,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=[{"type": "eq", "fun": lambda q: float(np.sum(q) - 1.0)}],
        options={"maxiter": 250, "ftol": 1e-12, "disp": False},
    )
    q = result.x if result.success and result.x is not None else p
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    q /= q.sum()
    threshold = max(float(minimum_reported_weight), 0.0)
    if threshold > 0.0:
        q[q < threshold] = 0.0
        if q.sum() <= 0.0:
            q[int(np.argmax(p))] = 1.0
        q /= q.sum()
    weights = dict(zip(names, q.tolist()))
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "posterior_weights": dict(zip(names, p.tolist())),
        "adjusted_weights": weights,
        "redundancy_penalty": lam,
        "concentration_reward": tau,
        "minimum_reported_weight": threshold,
        "effective_arm_count": float(1.0 / np.sum(q * q)),
        "similarity_matrix": sim.tolist(),
        "arm_names": names,
    }
    return weights, diagnostics


def posterior_calibrated_mean_box(
    mixed_scenarios: pd.DataFrame,
    arm_scenarios: Mapping[str, pd.DataFrame],
    posterior_weights: Mapping[str, float],
    *,
    kappa: float = 1.0,
    entropy_weight: float = 1.0,
    disagreement_weight: float = 1.0,
    max_sigma_fraction: float = 0.50,
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """Return lower-bound means for an asset-wise box uncertainty set.
    
    With long-only weights, the worst-case objective is (mu_hat-radius)'w.
    The radius combines scenario standard errors, posterior entropy, and
    disagreement between expert means.
    """

    nominal = mixed_scenarios.mean(axis=0)
    sigma = mixed_scenarios.std(axis=0, ddof=1).clip(lower=1e-10)
    standard_error = sigma / np.sqrt(max(len(mixed_scenarios), 1))
    names = [name for name, frame in arm_scenarios.items() if frame is not None and not frame.empty]
    probs = normalize_weights(posterior_weights, names)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-15))))
    entropy_norm = entropy / np.log(len(names)) if len(names) > 1 else 0.0

    means = []
    for name in names:
        means.append(
            arm_scenarios[name].reindex(columns=nominal.index).mean(axis=0).to_numpy(dtype=float)
        )
    mean_matrix = np.vstack(means)
    weighted_mean = probs @ mean_matrix
    disagreement = np.sqrt(
        np.maximum(probs @ ((mean_matrix - weighted_mean) ** 2), 0.0)
    )
    disagreement_series = pd.Series(disagreement, index=nominal.index)

    radius = float(kappa) * (
        standard_error * (1.0 + float(entropy_weight) * entropy_norm)
        + float(disagreement_weight) * disagreement_series
    )
    radius = pd.concat(
        [radius, float(max_sigma_fraction) * sigma], axis=1
    ).min(axis=1)
    radius = radius.clip(lower=0.0)
    robust_mean = nominal - radius
    diagnostics = {
        "ambiguity_set": "assetwise_box_on_predictive_mean",
        "kappa": float(kappa),
        "entropy_weight": float(entropy_weight),
        "disagreement_weight": float(disagreement_weight),
        "posterior_entropy": entropy,
        "posterior_entropy_normalized": entropy_norm,
        "mean_radius_average": float(radius.mean()),
        "mean_radius_max": float(radius.max()),
        "mean_disagreement_average": float(disagreement_series.mean()),
        "max_sigma_fraction": float(max_sigma_fraction),
    }
    return robust_mean, radius, diagnostics
