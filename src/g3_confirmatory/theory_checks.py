"""Numerical checks and examples for the manuscript's analytical propositions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .posterior import GeneralizedPosterior
from .arms import shrink_scenario_mean


def check_exact_posterior_odds() -> dict[str, Any]:
    losses = [
        {"A": 0.10, "B": 0.20},
        {"A": 0.30, "B": 0.10},
        {"A": 0.05, "B": 0.25},
    ]
    eta = 2.5
    posterior = GeneralizedPosterior(("A", "B"), calibration_months=0, eta=eta)
    for vector in losses:
        posterior.update(vector)
    weights = posterior.weights()
    numerical = np.log(weights["A"] / weights["B"])
    analytical = -eta * sum(v["A"] - v["B"] for v in losses)
    return {
        "proposition": "exact_posterior_odds",
        "numerical_log_odds": float(numerical),
        "analytical_log_odds": float(analytical),
        "absolute_error": float(abs(numerical - analytical)),
        "passed": bool(abs(numerical - analytical) < 1e-12),
    }


def check_exponential_weights_regret(
    *,
    n_rounds: int = 500,
    n_arms: int = 4,
    eta: float = 0.10,
    random_seed: int = 20260322,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    losses = rng.uniform(0.0, 1.0, size=(n_rounds, n_arms))
    log_w = np.zeros(n_arms)
    aggregate_loss = 0.0
    for t in range(n_rounds):
        p = np.exp(log_w - np.max(log_w))
        p /= p.sum()
        aggregate_loss += float(p @ losses[t])
        log_w -= eta * losses[t]
    best = float(losses.sum(axis=0).min())
    regret = aggregate_loss - best
    bound = np.log(n_arms) / eta + eta * n_rounds / 8.0
    return {
        "proposition": "full_information_regret",
        "aggregate_loss": aggregate_loss,
        "best_arm_loss": best,
        "regret": regret,
        "bound": float(bound),
        "passed": bool(regret <= bound + 1e-10),
    }


def gaussian_kl(
    true_mean: np.ndarray,
    true_cov: np.ndarray,
    model_mean: np.ndarray,
    model_cov: np.ndarray,
) -> float:
    p = len(true_mean)
    inv = np.linalg.inv(model_cov)
    delta = model_mean - true_mean
    return float(
        0.5
        * (
            np.trace(inv @ true_cov)
            + delta.T @ inv @ delta
            - p
            + np.log(np.linalg.det(model_cov) / np.linalg.det(true_cov))
        )
    )


def check_score_decision_rank_reversal() -> dict[str, Any]:
    true_mean = np.array([0.04, 0.0])
    true_cov = np.diag([1.10, 1.00])
    model_a_mean = true_mean.copy()
    model_a_cov = np.diag([1.07, 1.00])
    model_b_mean = np.zeros(2)
    model_b_cov = true_cov.copy()
    utility_a = float(true_mean[0] - 0.5 * true_cov[0, 0])
    utility_b = float(true_mean[1] - 0.5 * true_cov[1, 1])
    kl_a = gaussian_kl(true_mean, true_cov, model_a_mean, model_a_cov)
    kl_b = gaussian_kl(true_mean, true_cov, model_b_mean, model_b_cov)
    return {
        "proposition": "score_decision_rank_reversal",
        "model_a_true_utility": utility_a,
        "model_b_true_utility": utility_b,
        "model_a_kl": kl_a,
        "model_b_kl": kl_b,
        "predictive_score_prefers_a": bool(kl_a < kl_b),
        "decision_prefers_b": bool(utility_b > utility_a),
        "passed": bool(kl_a < kl_b and utility_b > utility_a),
    }


def check_arm_composition_variance(
    *,
    probabilities: tuple[float, ...] = (0.3, 0.7),
    n_scenarios: int = 500,
    n_replications: int = 100_000,
    random_seed: int = 20260322,
) -> dict[str, Any]:
    q = np.asarray(probabilities, dtype=float)
    q /= q.sum()
    rng = np.random.default_rng(random_seed)
    hard = np.eye(len(q))[rng.choice(len(q), size=n_replications, p=q)]
    counts = rng.multinomial(n_scenarios, q, size=n_replications) / n_scenarios
    hard_cov = np.cov(hard, rowvar=False, ddof=1)
    mix_cov = np.cov(counts, rowvar=False, ddof=1)
    theoretical = np.diag(q) - np.outer(q, q)
    target_mix = theoretical / n_scenarios
    ratio = float(np.trace(hard_cov) / np.trace(mix_cov))
    return {
        "proposition": "arm_composition_variance_reduction",
        "n_scenarios": n_scenarios,
        "empirical_trace_ratio": ratio,
        "theoretical_ratio": float(n_scenarios),
        "hard_covariance_max_error": float(np.max(np.abs(hard_cov - theoretical))),
        "mix_covariance_max_error": float(np.max(np.abs(mix_cov - target_mix))),
        "passed": bool(abs(ratio / n_scenarios - 1.0) < 0.03),
    }


def check_no_universal_action_dominance() -> dict[str, Any]:
    hard_loss_q06 = 0.6
    mix_loss_q06 = 1.0
    hard_loss_q04 = 0.4
    mix_loss_q04 = 0.0
    return {
        "proposition": "no_universal_action_dominance",
        "q_0_6_hard_loss": hard_loss_q06,
        "q_0_6_mix_loss": mix_loss_q06,
        "q_0_4_hard_loss": hard_loss_q04,
        "q_0_4_mix_loss": mix_loss_q04,
        "mix_worse_at_q_0_6": mix_loss_q06 > hard_loss_q06,
        "mix_better_at_q_0_4": mix_loss_q04 < hard_loss_q04,
        "passed": bool(
            mix_loss_q06 > hard_loss_q06 and mix_loss_q04 < hard_loss_q04
        ),
    }


def check_mean_shrinkage_covariance_invariance(
    *,
    n_rows: int = 500,
    n_assets: int = 20,
    fraction: float = 0.75,
    random_seed: int = 20260322,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    frame = pd.DataFrame(rng.normal(size=(n_rows, n_assets)))
    shrunk = shrink_scenario_mean(frame, fraction)
    cov_error = float(np.max(np.abs(frame.cov().to_numpy() - shrunk.cov().to_numpy())))
    expected_mean = (1.0 - fraction) * frame.mean(axis=0)
    mean_error = float(np.max(np.abs(shrunk.mean(axis=0) - expected_mean)))
    return {
        "proposition": "mean_shrinkage_covariance_invariance",
        "fraction": fraction,
        "max_covariance_error": cov_error,
        "max_mean_error": mean_error,
        "passed": bool(cov_error < 1e-12 and mean_error < 1e-12),
    }


def check_state_fragmentation_rate(
    *,
    loss_gap: float = 0.05,
    eta: float = 2.0,
    state_probability: float = 0.25,
    n_periods: int = 100_000,
    random_seed: int = 20260322,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    state = rng.random(n_periods) < state_probability
    pooled_log_odds = eta * loss_gap * n_periods
    state_log_odds = eta * loss_gap * int(state.sum())
    empirical_ratio = state_log_odds / pooled_log_odds
    return {
        "proposition": "state_fragmentation_learning_rate",
        "state_probability": state_probability,
        "empirical_calendar_drift_ratio": float(empirical_ratio),
        "theoretical_calendar_drift_ratio": state_probability,
        "passed": bool(abs(empirical_ratio - state_probability) < 0.01),
    }


def run_all_theory_checks() -> pd.DataFrame:
    checks = [
        check_exact_posterior_odds(),
        check_exponential_weights_regret(),
        check_score_decision_rank_reversal(),
        check_arm_composition_variance(),
        check_no_universal_action_dominance(),
        check_mean_shrinkage_covariance_invariance(),
        check_state_fragmentation_rate(),
    ]
    return pd.DataFrame(checks)
