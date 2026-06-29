"""Feature engineering: build analysis-ready match and team-match tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def normalize_team(s: pd.Series) -> pd.Series:
    """Collapse historical/split nations into a single modern entity."""
    return s.replace(config.TEAM_ALIASES)


def _stage_rank(stage_name: str) -> int:
    return config.STAGE_RANK.get(str(stage_name).lower(), -1)


def build_matches(competition: str = "men") -> pd.DataFrame:
    """Return the consolidated, analysis-ready match table.

    Parameters
    ----------
    competition:
        ``"men"``, ``"women"`` or ``"all"``.

    Notes
    -----
    Adds engineered columns: ``total_goals``, ``goal_difference``,
    ``stage_rank``, ``is_knockout``, ``went_to_penalties``, ``decade`` and
    normalized team names, and merges edition metadata (year, host, winner).
    """
    matches = data.load_matches()
    tournaments = data.load_tournaments()

    df = matches.merge(
        tournaments[["tournament_id", "year", "host_country", "winner", "count_teams"]],
        on="tournament_id",
        how="left",
    )

    df["competition"] = np.where(
        df["tournament_name"].str.contains("Women", case=False, na=False), "women", "men"
    )
    if competition != "all":
        df = df[df["competition"] == competition].copy()

    for col in ("home_team_name", "away_team_name", "winner", "host_country"):
        df[col] = normalize_team(df[col])

    df["total_goals"] = df["home_team_score"] + df["away_team_score"]
    df["goal_difference"] = (df["home_team_score"] - df["away_team_score"]).abs()
    df["stage_rank"] = df["stage_name"].map(_stage_rank)
    df["is_knockout"] = df["knockout_stage"].astype(bool)
    df["went_to_penalties"] = df["penalty_shootout"].astype(bool)
    df["decade"] = (df["year"] // 10) * 10

    return df.sort_values("match_date").reset_index(drop=True)


def build_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape to one row per team per match (long format).

    Columns: ``team, opponent, year, stage_name, stage_rank, venue,
    goals_for, goals_against, win, draw, loss``.
    """
    base = ["match_id", "year", "stage_name", "stage_rank"]
    home = matches[base + ["home_team_name", "away_team_name", "home_team_score", "away_team_score"]].copy()
    home.columns = base + ["team", "opponent", "goals_for", "goals_against"]
    home["venue"] = "home"

    away = matches[base + ["away_team_name", "home_team_name", "away_team_score", "home_team_score"]].copy()
    away.columns = base + ["team", "opponent", "goals_for", "goals_against"]
    away["venue"] = "away"

    long = pd.concat([home, away], ignore_index=True)
    long["win"] = long["goals_for"] > long["goals_against"]
    long["draw"] = long["goals_for"] == long["goals_against"]
    long["loss"] = long["goals_for"] < long["goals_against"]
    return long


def team_strengths(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-edition attack/defense for each team (mean goals for/against).

    Returns columns ``tournament_id, team, attack, defense, n_matches``.
    """
    long = build_team_matches(matches.assign(tournament_id=matches["tournament_id"]))
    long = long.merge(matches[["match_id", "tournament_id"]].drop_duplicates(), on="match_id")
    grp = long.groupby(["tournament_id", "team"]).agg(
        attack=("goals_for", "mean"),
        defense=("goals_against", "mean"),
        n_matches=("goals_for", "size"),
    )
    return grp.reset_index()
