"""Reusable plotting helpers. Each ``plot_*`` saves a figure and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from . import analysis, config, elo

matplotlib.use("Agg")  # headless-safe
sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str) -> Path:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_goals_distribution(matches) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(matches["total_goals"], bins=range(0, 13), kde=True, ax=ax, color="#4c72b0")
    ax.axvline(matches["total_goals"].mean(), ls="--", color="crimson",
               label=f"mean = {matches['total_goals'].mean():.2f}")
    ax.set(title="Goals per match", xlabel="Total goals", ylabel="Matches")
    ax.legend()
    return _save(fig, "goals_distribution.png")


def plot_goals_trend(matches) -> Path:
    per_year = matches.groupby("year")["total_goals"].mean()
    res = analysis.goals_trend(matches)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(per_year.index, per_year.values, "o-", color="#1f4e79")
    ax.axhline(matches["total_goals"].mean(), ls="--", color="grey")
    ax.set(title=f"Average goals per match by edition (Spearman r={res['spearman']:.2f}, "
                 f"p={res['spearman_p']:.3f})", xlabel="Year", ylabel="Goals / match")
    return _save(fig, "goals_trend.png")


def plot_elo_evolution(history, teams: list[str] | None = None, top: int = 6) -> Path:
    if teams is None:
        teams = list(elo.final_ratings(history).head(top).index)
    fig, ax = plt.subplots(figsize=(11, 6))
    for team in teams:
        h = history[history["team"] == team].sort_values("date")
        ax.plot(h["date"], h["rating_after"], label=team, lw=1.8)
    ax.set(title="Elo rating evolution", xlabel="Year", ylabel="Elo rating")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    return _save(fig, "elo_evolution.png")


def plot_elo_ranking(history, top: int = 12) -> Path:
    ratings = elo.final_ratings(history).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ratings.index, ratings.values, color="#c44e52")
    ax.set(title=f"Current Elo ranking (top {top})", xlabel="Elo rating")
    ax.set_xlim(left=min(1400, ratings.min() - 20))
    return _save(fig, "elo_ranking.png")


def plot_rivalry_network(matches, min_games: int = 3) -> Path:
    riv = analysis.biggest_rivalries(matches, min_games=min_games, top=200)
    long = config  # noqa: F841 (placeholder to keep import grouping)

    G = nx.Graph()
    for r in riv.itertuples(index=False):
        G.add_edge(r.team_a, r.team_b, weight=r.games)
    games_played = (
        matches["home_team_name"].value_counts()
        .add(matches["away_team_name"].value_counts(), fill_value=0)
    )
    sizes = [80 + 14 * games_played.get(n, 0) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(G, seed=config.RANDOM_STATE, k=0.6)
    widths = [0.4 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#bbbbbb", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4c72b0", alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title("World Cup rivalry network (matchups with 3+ games)")
    ax.axis("off")
    return _save(fig, "rivalry_network.png")


def generate_all(matches, history) -> list[Path]:
    """Render the full figure set used by the README/report."""
    return [
        plot_goals_distribution(matches),
        plot_goals_trend(matches),
        plot_elo_evolution(history),
        plot_elo_ranking(history),
        plot_rivalry_network(matches),
    ]
