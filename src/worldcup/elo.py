"""Chronological Elo ratings for national teams across World Cup history.

Implements a World-Football-style Elo where the rating update is scaled by the
margin of victory. Ratings are computed match-by-match in date order, so they
can be used as a *leak-free* feature for downstream prediction models (the
rating before a match never depends on the match itself).
"""
from __future__ import annotations

import pandas as pd

from . import config


def _margin_multiplier(goal_diff: int) -> float:
    """Goal-difference weight (a 3-0 win moves the rating more than a 1-0)."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    k: float = config.ELO_K,
    start: float = config.ELO_START,
) -> pd.DataFrame:
    """Run the Elo simulation over ``matches`` (must be date-sorted-able).

    Returns a tidy history with one row per team per match containing the
    rating *before* and *after* the game.
    """
    matches = matches.sort_values("match_date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for m in matches.itertuples(index=False):
        home, away = m.home_team_name, m.away_team_name
        ra = ratings.get(home, start)
        rb = ratings.get(away, start)

        if m.home_team_score > m.away_team_score:
            sa = 1.0
        elif m.home_team_score < m.away_team_score:
            sa = 0.0
        else:
            sa = 0.5

        exp_a = _expected(ra, rb)
        mult = _margin_multiplier(int(abs(m.home_team_score - m.away_team_score)))
        delta = k * mult * (sa - exp_a)

        ratings[home] = ra + delta
        ratings[away] = rb - delta

        for team, before, after, opp, score in (
            (home, ra, ratings[home], away, sa),
            (away, rb, ratings[away], home, 1.0 - sa),
        ):
            rows.append(
                {
                    "match_id": m.match_id,
                    "date": m.match_date,
                    "year": m.year,
                    "team": team,
                    "opponent": opp,
                    "rating_before": before,
                    "rating_after": after,
                    "score": score,
                }
            )

    return pd.DataFrame(rows)


def final_ratings(history: pd.DataFrame) -> pd.Series:
    """Latest rating per team (descending)."""
    last = history.sort_values("date").groupby("team")["rating_after"].last()
    return last.sort_values(ascending=False)


def peak_ratings(history: pd.DataFrame) -> pd.Series:
    """Highest rating ever reached per team (descending)."""
    return history.groupby("team")["rating_after"].max().sort_values(ascending=False)


def ratings_before_match(history: pd.DataFrame) -> pd.DataFrame:
    """Wide table ``match_id -> (home/away pre-match rating)`` for modelling.

    Built from the tidy history; within each match the first appended row is
    the home team and the second is the away team.
    """
    h = history.assign(_n=history.groupby("match_id").cumcount())
    home = (h[h["_n"] == 0][["match_id", "team", "rating_before"]]
            .rename(columns={"team": "home_team_name", "rating_before": "home_elo"}))
    away = (h[h["_n"] == 1][["match_id", "team", "rating_before"]]
            .rename(columns={"team": "away_team_name", "rating_before": "away_elo"}))
    return home.merge(away, on="match_id")
