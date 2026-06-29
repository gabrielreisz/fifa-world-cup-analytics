"""Project-wide configuration: paths, data sources and domain constants."""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Data source -------------------------------------------------------------
# Fjelstul World Cup Database (CC-BY-4.0): https://github.com/jfjelstul/worldcup
DATA_BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
TABLES = {
    "tournaments": f"{DATA_BASE_URL}/tournaments.csv",
    "matches": f"{DATA_BASE_URL}/matches.csv",
}

# --- Domain constants --------------------------------------------------------
# Map historical / split nations to a single modern entity so that long-run
# team analyses (Elo, rivalries, totals) are not fragmented.
TEAM_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Yugoslavia": "Serbia",
    "Dutch East Indies": "Indonesia",
    "Zaire": "DR Congo",
}

# Ordinal depth of each tournament stage (group = 0 ... final = 4).
STAGE_RANK = {
    "group stage": 0,
    "first group stage": 0,
    "second group stage": 1,
    "round of 16": 1,
    "quarter-final": 2,
    "quarter-finals": 2,
    "semi-final": 3,
    "semi-finals": 3,
    "third-place match": 3,
    "final round": 4,
    "final": 4,
}

STAGE_LABELS = {0: "Group", 1: "Round of 16", 2: "Quarter-final", 3: "Semi-final", 4: "Final"}

ELO_START = 1500.0
ELO_K = 40.0
RANDOM_STATE = 42
