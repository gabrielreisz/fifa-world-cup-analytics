"""Descriptive and inferential analyses over the World Cup data.

Each function returns plain data (DataFrame / dict) so results can be reused by
the CLI, the dashboard, tests and notebooks without any plotting side effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import features


# --- Descriptive -------------------------------------------------------------
def biggest_rivalries(matches: pd.DataFrame, min_games: int = 3, top: int = 15) -> pd.DataFrame:
    """Most frequent head-to-head matchups with the win/draw balance."""
    pair = matches.apply(
        lambda r: " x ".join(sorted([str(r["home_team_name"]), str(r["away_team_name"])])), axis=1
    )
    df = matches.assign(pair=pair)

    def _balance(g: pd.DataFrame) -> pd.Series:
        a, b = sorted([str(g["home_team_name"].iloc[0]), str(g["away_team_name"].iloc[0])])
        wins_a = ((g["home_team_name"] == a) & (g["home_team_score"] > g["away_team_score"])).sum()
        wins_a += ((g["away_team_name"] == a) & (g["away_team_score"] > g["home_team_score"])).sum()
        draws = (g["home_team_score"] == g["away_team_score"]).sum()
        return pd.Series({"team_a": a, "team_b": b, "wins_a": wins_a, "draws": draws,
                          "wins_b": len(g) - wins_a - draws})

    counts = df.groupby("pair").size().rename("games")
    bal = df.groupby("pair").apply(_balance, include_groups=False)
    out = bal.join(counts)
    return out[out["games"] >= min_games].sort_values("games", ascending=False).head(top).reset_index(drop=True)


def top_attacks(matches: pd.DataFrame, min_matches: int = 10, top: int = 10) -> pd.DataFrame:
    """Teams ranked by mean goals scored per match (min sample size)."""
    long = features.build_team_matches(matches)
    g = long.groupby("team").agg(goals_per_match=("goals_for", "mean"), matches=("goals_for", "size"))
    return g[g["matches"] >= min_matches].sort_values("goals_per_match", ascending=False).head(top)


def clean_sheets(matches: pd.DataFrame, top: int = 10) -> pd.Series:
    """Teams with the most matches without conceding."""
    long = features.build_team_matches(matches)
    return long[long["goals_against"] == 0]["team"].value_counts().head(top)


# --- Inferential -------------------------------------------------------------
def goals_trend(matches: pd.DataFrame) -> dict:
    """Has scoring declined over time? Spearman + Pearson of year vs goals."""
    rho, p_rho = stats.spearmanr(matches["year"], matches["total_goals"])
    r, p_r = stats.pearsonr(matches["year"], matches["total_goals"])
    return {"spearman": rho, "spearman_p": p_rho, "pearson": r, "pearson_p": p_r,
            "goals_first_decade": matches[matches["year"] <= 1950]["total_goals"].mean(),
            "goals_last_decade": matches[matches["year"] >= 2014]["total_goals"].mean()}


def host_advantage(matches: pd.DataFrame) -> dict:
    """Welch t-test of goals with vs without the host on the pitch, plus the
    share of hosts reaching at least the semi-final at home."""
    is_host = (matches["host_country"] == matches["home_team_name"]) | (
        matches["host_country"] == matches["away_team_name"]
    )
    with_host = matches[is_host]["total_goals"]
    without = matches[~is_host]["total_goals"]
    t, p = stats.ttest_ind(with_host, without, equal_var=False)

    depth = matches[is_host].groupby(["year", "host_country"])["stage_rank"].max()
    return {"mean_with_host": with_host.mean(), "mean_without_host": without.mean(),
            "t_stat": t, "p_value": p,
            "pct_hosts_reaching_semi": float((depth >= 3).mean() * 100)}


def poisson_zero_zero(matches: pd.DataFrame) -> dict:
    """Compare the Poisson-predicted share of 0-0 games to reality."""
    lam = matches["total_goals"].mean()
    theoretical = float(stats.poisson.pmf(0, lam))
    empirical = float((matches["total_goals"] == 0).mean())
    return {"lambda": lam, "theoretical_0_0": theoretical, "empirical_0_0": empirical}


def penalties_given_extra_time(matches: pd.DataFrame) -> dict:
    """P(shootout | knockout match reached extra time)."""
    et = matches[(matches["is_knockout"]) & (matches["extra_time"] == 1)]
    p = float(et["went_to_penalties"].mean()) if len(et) else float("nan")
    return {"n_extra_time": int(len(et)), "p_penalties": p}


# --- Elo-based ---------------------------------------------------------------
def biggest_upsets(history: pd.DataFrame, matches: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Matches where the pre-match Elo underdog won, ranked by the rating gap."""
    winners = matches.copy()
    winners["winner_team"] = np.where(
        winners["home_team_score"] > winners["away_team_score"], winners["home_team_name"],
        np.where(winners["away_team_score"] > winners["home_team_score"], winners["away_team_name"], None),
    )
    decided = winners[winners["winner_team"].notna()]

    pre = history[["match_id", "team", "rating_before"]]
    rows = []
    for m in decided.itertuples(index=False):
        r = pre[pre["match_id"] == m.match_id]
        if len(r) != 2:
            continue
        rmap = dict(zip(r["team"], r["rating_before"], strict=False))
        winner_elo = rmap.get(m.winner_team)
        loser = m.away_team_name if m.winner_team == m.home_team_name else m.home_team_name
        loser_elo = rmap.get(loser)
        if winner_elo is None or loser_elo is None:
            continue
        rows.append({"year": m.year, "match": m.match_name, "stage": m.stage_name,
                     "winner": m.winner_team, "loser": loser,
                     "elo_gap": round(loser_elo - winner_elo, 1)})
    out = pd.DataFrame(rows)
    return out[out["elo_gap"] > 0].sort_values("elo_gap", ascending=False).head(top).reset_index(drop=True)
