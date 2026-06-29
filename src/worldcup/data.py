"""Data ingestion: download, cache and load the raw World Cup tables."""
from __future__ import annotations

import pandas as pd

from . import config


def download_raw(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download the raw CSV tables and cache them under ``data/raw``.

    Parameters
    ----------
    force:
        Re-download even if a cached copy already exists.

    Returns
    -------
    dict
        Mapping ``{table_name: DataFrame}``.
    """
    tables: dict[str, pd.DataFrame] = {}
    for name, url in config.TABLES.items():
        dest = config.RAW_DIR / f"{name}.csv"
        if force or not dest.exists():
            df = pd.read_csv(url)
            df.to_csv(dest, index=False)
        else:
            df = pd.read_csv(dest)
        tables[name] = df
    return tables


def _load_table(name: str) -> pd.DataFrame:
    dest = config.RAW_DIR / f"{name}.csv"
    if not dest.exists():
        # Fall back to a network fetch (and cache) on first use.
        return download_raw()[name]
    return pd.read_csv(dest)


def load_tournaments() -> pd.DataFrame:
    """Return the tournaments table (one row per World Cup edition)."""
    return _load_table("tournaments")


def load_matches() -> pd.DataFrame:
    """Return the matches table (one row per match)."""
    df = _load_table("matches")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def load_goals() -> pd.DataFrame:
    """Return the goals table (one row per goal)."""
    return _load_table("goals")


def load_squads() -> pd.DataFrame:
    """Return the squads table (one row per player per edition)."""
    return _load_table("squads")


def load_players() -> pd.DataFrame:
    """Return the players table (one row per player, incl. birth date)."""
    return _load_table("players")


def load_bookings() -> pd.DataFrame:
    """Return the bookings table (one row per card)."""
    return _load_table("bookings")


def load_penalty_kicks() -> pd.DataFrame:
    """Return the penalty-shootout kicks table (one row per kick)."""
    return _load_table("penalty_kicks")


def load_tournament_standings() -> pd.DataFrame:
    """Return the final standings table (positions 1-4 per edition)."""
    return _load_table("tournament_standings")
