"""Monte Carlo tournament simulation.

Given a trained :class:`~worldcup.models.MatchPredictor`, simulate knockout
brackets and full group+knockout tournaments thousands of times to estimate
each team's probability of advancing and of lifting the trophy -- a
FiveThirtyEight-style forecast built on the project's own Elo + Poisson models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import MatchPredictor


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """Neutral-venue probability that A beats B (Elo expected score)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _largest_power_of_two(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def simulate_knockout(
    predictor: MatchPredictor, teams: list[str], n_sims: int = 20000, seed: int = 42
) -> pd.DataFrame:
    """Simulate a single-elimination bracket with a random draw each run.

    The team list is trimmed to the largest power of two by Elo. Draws within a
    knockout match are resolved by the Elo win probability (i.e. folded in).
    Returns per-team probabilities of reaching each stage and winning.
    """
    ranked = sorted(teams, key=predictor.rating, reverse=True)
    n = _largest_power_of_two(len(ranked))
    teams = ranked[:n]
    ratings = np.array([predictor.rating(t) for t in teams])

    # P[i, j] = probability team i beats team j
    diff = ratings[:, None] - ratings[None, :]
    P = 1.0 / (1.0 + 10 ** (-diff / 400.0))

    rng = np.random.default_rng(seed)
    champion = np.zeros(n)
    reached = {size: np.zeros(n) for size in _stage_sizes(n)}

    for _ in range(n_sims):
        alive = rng.permutation(n)
        for size in _stage_sizes(n):
            for t in alive:
                reached[size][t] += 1
            nxt = []
            for k in range(0, len(alive), 2):
                a, b = alive[k], alive[k + 1]
                nxt.append(a if rng.random() < P[a, b] else b)
            alive = np.array(nxt)
        champion[alive[0]] += 1

    out = pd.DataFrame({"team": teams, "elo": ratings.round(0), "p_champion": champion / n_sims})
    for size, col in _stage_columns(n).items():
        out[col] = reached[size] / n_sims
    return out.sort_values("p_champion", ascending=False).reset_index(drop=True)


def _stage_sizes(n: int) -> list[int]:
    """Bracket sizes from the round of ``n`` down to the final (2)."""
    sizes, s = [], n
    while s >= 2:
        sizes.append(s)
        s //= 2
    return sizes


def _stage_columns(n: int) -> dict[int, str]:
    names = {2: "p_final", 4: "p_semifinal", 8: "p_quarterfinal", 16: "p_round16", 32: "p_round32"}
    return {size: names.get(size, f"p_last{size}") for size in _stage_sizes(n) if size >= 2}


def simulate_tournament(
    predictor: MatchPredictor,
    teams: list[str],
    n_groups: int = 8,
    n_sims: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a full World-Cup-style tournament (groups + knockout).

    Teams are snake-drafted into ``n_groups`` balanced groups by Elo. Group
    games are simulated as independent Poisson scorelines (neutral venue);
    the top two of each group advance to a single-elimination knockout.
    """
    ranked = sorted(teams, key=predictor.rating, reverse=True)
    group_size = len(ranked) // n_groups
    ranked = ranked[: group_size * n_groups]
    ratings = {t: predictor.rating(t) for t in ranked}

    # snake draft into groups
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, t in enumerate(ranked):
        rnd = i // n_groups
        g = i % n_groups if rnd % 2 == 0 else n_groups - 1 - (i % n_groups)
        groups[g].append(t)

    rng = np.random.default_rng(seed)
    advanced = {t: 0 for t in ranked}
    champion = {t: 0 for t in ranked}

    lam_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def lambdas(a: str, b: str) -> tuple[float, float]:
        if (a, b) not in lam_cache:
            lam_cache[(a, b)] = predictor.score.expected_goals_neutral(ratings[a], ratings[b])
        return lam_cache[(a, b)]

    for _ in range(n_sims):
        qualifiers: list[str] = []
        for group in groups:
            pts = dict.fromkeys(group, 0)
            gd = dict.fromkeys(group, 0)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    la, lb = lambdas(a, b)
                    ga, gb = rng.poisson(la), rng.poisson(lb)
                    gd[a] += ga - gb
                    gd[b] += gb - ga
                    if ga > gb:
                        pts[a] += 3
                    elif gb > ga:
                        pts[b] += 3
                    else:
                        pts[a] += 1
                        pts[b] += 1
            ordered = sorted(group, key=lambda t: (pts[t], gd[t], ratings[t]), reverse=True)
            qualifiers += ordered[:2]

        for t in qualifiers:
            advanced[t] += 1

        # knockout among qualifiers (random bracket)
        alive = list(rng.permutation(qualifiers))
        while len(alive) > 1:
            nxt = []
            for k in range(0, len(alive), 2):
                a, b = alive[k], alive[k + 1]
                nxt.append(a if rng.random() < elo_win_prob(ratings[a], ratings[b]) else b)
            alive = nxt
        champion[alive[0]] += 1

    out = pd.DataFrame({
        "team": ranked,
        "elo": [round(ratings[t]) for t in ranked],
        "p_advance": [advanced[t] / n_sims for t in ranked],
        "p_champion": [champion[t] / n_sims for t in ranked],
    })
    return out.sort_values("p_champion", ascending=False).reset_index(drop=True)
