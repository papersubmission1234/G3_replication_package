"""Coalition credit and pairwise interactions for expert aggregation.

The caller supplies every coalition value; a mandatory baseline is excluded
from the player set. Shapley credit is measured relative to the empty coalition.
The leave-one-out benchmark uses grand-coalition deletions.

In the portfolio study, coalition values are post-cost utility from a common
optimizer, CVaR budget, asset universe, and pretrade portfolio.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Iterable, Mapping

import numpy as np


Coalition = frozenset[str]


def _ordered_players(players: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(str(player) for player in players))
    if any(not player for player in ordered):
        raise ValueError("Player names must be nonempty strings.")
    return ordered


def all_coalitions(players: Iterable[str]) -> tuple[Coalition, ...]:
    """Return all coalitions in deterministic size/lexicographic order."""

    ordered = _ordered_players(players)
    coalitions: list[Coalition] = []
    for size in range(len(ordered) + 1):
        for subset in combinations(ordered, size):
            coalitions.append(frozenset(subset))
    return tuple(coalitions)


def coalition_label(coalition: Iterable[str]) -> str:
    members = sorted(str(member) for member in coalition)
    return "EMPTY" if not members else "+".join(members)


def _validated_values(
    players: tuple[str, ...],
    coalition_values: Mapping[Coalition, float],
) -> dict[Coalition, float]:
    expected = set(all_coalitions(players))
    supplied = {frozenset(key) for key in coalition_values}
    missing = expected.difference(supplied)
    extra = supplied.difference(expected)
    if missing:
        labels = [
            coalition_label(item)
            for item in sorted(missing, key=lambda x: (len(x), sorted(x)))
        ]
        raise ValueError(f"Missing coalition values: {labels}")
    if extra:
        labels = [
            coalition_label(item)
            for item in sorted(extra, key=lambda x: (len(x), sorted(x)))
        ]
        raise ValueError(f"Coalition values contain unknown players: {labels}")

    clean = {
        coalition: float(coalition_values[coalition])
        for coalition in expected
    }
    nonfinite = [
        coalition_label(coalition)
        for coalition, value in clean.items()
        if not np.isfinite(value)
    ]
    if nonfinite:
        raise ValueError(f"Coalition values must be finite: {sorted(nonfinite)}")
    return clean


def exact_shapley_values(
    players: Iterable[str],
    coalition_values: Mapping[Coalition, float],
) -> dict[str, float]:
    """Compute Shapley credit from a complete coalition-value map.
    
    Any mandatory baseline is already included in each value and is not a player.
    """

    ordered = _ordered_players(players)
    n_players = len(ordered)
    if n_players == 0:
        return {}
    values_by_coalition = _validated_values(ordered, coalition_values)

    denominator = factorial(n_players)
    values: dict[str, float] = {}
    for player in ordered:
        others = tuple(name for name in ordered if name != player)
        contribution = 0.0
        for subset in all_coalitions(others):
            size = len(subset)
            weight = (
                factorial(size)
                * factorial(n_players - size - 1)
                / denominator
            )
            with_player = frozenset((*subset, player))
            contribution += weight * (
                values_by_coalition[with_player]
                - values_by_coalition[subset]
            )
        values[player] = float(contribution)
    return values


def exact_shapley_pair_interactions(
    players: Iterable[str],
    coalition_values: Mapping[Coalition, float],
) -> dict[tuple[str, str], float]:
    """Compute the pairwise Shapley interaction index.
    
    Positive values indicate complementarity across coalition contexts; negative
    values indicate redundancy or crowding through the cap and optimizer.
    """

    ordered = _ordered_players(players)
    n_players = len(ordered)
    if n_players < 2:
        return {}
    values_by_coalition = _validated_values(ordered, coalition_values)

    denominator = factorial(n_players - 1)
    interactions: dict[tuple[str, str], float] = {}
    for left_idx, left in enumerate(ordered):
        for right in ordered[left_idx + 1 :]:
            others = tuple(
                name for name in ordered if name not in {left, right}
            )
            interaction = 0.0
            for subset in all_coalitions(others):
                size = len(subset)
                weight = (
                    factorial(size)
                    * factorial(n_players - size - 2)
                    / denominator
                )
                with_left = frozenset((*subset, left))
                with_right = frozenset((*subset, right))
                with_both = frozenset((*subset, left, right))
                second_difference = (
                    values_by_coalition[with_both]
                    - values_by_coalition[with_left]
                    - values_by_coalition[with_right]
                    + values_by_coalition[subset]
                )
                interaction += weight * second_difference
            interactions[(left, right)] = float(interaction)
    return interactions


def grand_coalition_loo_values(
    players: Iterable[str],
    coalition_values: Mapping[Coalition, float],
) -> dict[str, float]:
    """Return v(N) - v(N without k) for each player k.
    
    Leave-one-out credits need not sum to v(N)-v(empty): interactions may be counted
    more than once or omitted.
    """

    ordered = _ordered_players(players)
    if not ordered:
        return {}
    full = frozenset(ordered)
    required = {full, *(full.difference({player}) for player in ordered)}
    supplied = {frozenset(key): float(value) for key, value in coalition_values.items()}
    missing = required.difference(supplied)
    if missing:
        labels = [
            coalition_label(item)
            for item in sorted(missing, key=lambda x: (len(x), sorted(x)))
        ]
        raise ValueError(f"Missing leave-one-out coalition values: {labels}")
    nonfinite = [
        coalition_label(coalition)
        for coalition in required
        if not np.isfinite(supplied[coalition])
    ]
    if nonfinite:
        raise ValueError(
            f"Coalition values must be finite: {sorted(nonfinite)}"
        )
    return {
        player: float(supplied[full] - supplied[full.difference({player})])
        for player in ordered
    }
