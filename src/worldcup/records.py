"""Player- and team-level records built from the richer World Cup tables
(goals, squads, players, bookings, penalty shootouts).

Every function returns plain data (DataFrame / Series / dict) so it can be
reused by the CLI, dashboard, tests and notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, features


def _competition_mask(df: pd.DataFrame, competition: str) -> pd.Series:
    # Note: "Women" contains the substring "men", so we key off "Women" only.
    if competition == "all" or "tournament_name" not in df.columns:
        return pd.Series(True, index=df.index)
    is_women = df["tournament_name"].str.contains("Women", case=False, na=False)
    return is_women if competition == "women" else ~is_women


def _full_name(df: pd.DataFrame) -> pd.Series:
    given = df["given_name"].fillna("").replace("not applicable", "")
    return (given + " " + df["family_name"].fillna("")).str.strip()


# --- Goals -------------------------------------------------------------------
def top_scorers(competition: str = "men", top: int = 15) -> pd.DataFrame:
    """All-time top scorers (own goals excluded), with penalty breakdown."""
    goals = data.load_goals()
    goals = goals[_competition_mask(goals, competition) & (goals["own_goal"] == 0)].copy()
    goals["player"] = _full_name(goals)
    out = goals.groupby("player").agg(
        goals=("goal_id", "count"),
        penalties=("penalty", "sum"),
        team=("team_name", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
    )
    return out.sort_values("goals", ascending=False).head(top)


def goal_timing(competition: str = "men") -> pd.DataFrame:
    """Share of goals scored in each 15-minute block (+ stoppage time)."""
    goals = data.load_goals()
    goals = goals[_competition_mask(goals, competition)].copy()
    minute = pd.to_numeric(goals["minute_regulation"], errors="coerce")
    stoppage = pd.to_numeric(goals["minute_stoppage"], errors="coerce").fillna(0)

    labels = [f"{lo + 1}-{lo + 15}" for lo in range(0, 120, 15)]
    block_idx = np.clip(((minute - 1) // 15).fillna(0).astype(int), 0, len(labels) - 1)
    cut = pd.Series(np.array(labels)[block_idx.to_numpy()], index=goals.index)
    cut = cut.where(stoppage == 0, other="stoppage")

    counts = cut.value_counts()
    order = [lbl for lbl in labels if lbl in counts.index] + (["stoppage"] if "stoppage" in counts.index else [])
    counts = counts.reindex(order)
    return pd.DataFrame({"goals": counts, "pct": (counts / counts.sum() * 100).round(1)})


# --- Penalty shootouts -------------------------------------------------------
def shootout_conversion(competition: str = "men", min_kicks: int = 5, top: int = 15) -> pd.DataFrame:
    """Penalty-shootout conversion rate by team (teams with >= ``min_kicks``)."""
    pk = data.load_penalty_kicks()
    pk = pk[_competition_mask(pk, competition)]
    grp = pk.groupby("team_name").agg(kicks=("converted", "size"), scored=("converted", "sum"))
    grp = grp[grp["kicks"] >= min_kicks]
    grp["conversion_pct"] = (grp["scored"] / grp["kicks"] * 100).round(1)
    return grp.sort_values("conversion_pct", ascending=False).head(top)


# --- Discipline --------------------------------------------------------------
def discipline_by_decade(competition: str = "men") -> pd.DataFrame:
    """Yellow and red cards per match, aggregated by decade."""
    bookings = data.load_bookings().copy()
    bookings = bookings[_competition_mask(bookings, competition)]
    bookings["year"] = pd.to_datetime(bookings["match_date"], errors="coerce").dt.year
    bookings["decade"] = (bookings["year"] // 10 * 10).astype("Int64")

    by = bookings.groupby("decade").agg(
        yellows=("yellow_card", "sum"), reds=("sending_off", "sum")
    )
    matches = features.build_matches(competition)
    n_matches = matches.groupby("decade").size().rename("matches")
    out = by.join(n_matches, how="left")
    out["yellows_per_match"] = (out["yellows"] / out["matches"]).round(2)
    out["reds_per_match"] = (out["reds"] / out["matches"]).round(2)
    return out.reset_index()


# --- Squad demographics ------------------------------------------------------
def squad_age(competition: str = "men") -> pd.DataFrame:
    """Average squad age per team per edition (sorted youngest first)."""
    squads = data.load_squads()
    squads = squads[_competition_mask(squads, competition)]
    players = data.load_players()[["player_id", "birth_date"]]
    tournaments = data.load_tournaments()[["tournament_id", "year", "start_date"]]

    df = squads.merge(players, on="player_id", how="left").merge(tournaments, on="tournament_id", how="left")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["age"] = (df["start_date"] - df["birth_date"]).dt.days / 365.25
    df = df.dropna(subset=["age"])

    out = df.groupby(["year", "team_name"]).agg(avg_age=("age", "mean"), squad_size=("age", "size"))
    out["avg_age"] = out["avg_age"].round(2)
    return out.reset_index().sort_values("avg_age")
