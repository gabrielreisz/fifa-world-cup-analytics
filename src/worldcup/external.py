"""External-data integrations.

* **World Bank API** (key-free): population and GDP-per-capita for each World Cup
  nation, used to ask whether economic / demographic size predicts success.
* **football-data.org** (needs a free API key in ``FOOTBALL_DATA_API_KEY``):
  thin connector for live/current data. It never hard-codes a key and raises a
  clear error if the variable is missing.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

from . import config, elo, features

WORLD_BANK_URL = "https://api.worldbank.org/v2"
_UA = {"User-Agent": "worldcup-analytics"}

# World Cup nation -> ISO-3 code (UK constituents collapse to GBR via England only;
# Taiwan/Chinese Taipei has no World Bank entry and is dropped gracefully).
TEAM_ISO3 = {
    "Algeria": "DZA", "Angola": "AGO", "Argentina": "ARG", "Australia": "AUS",
    "Austria": "AUT", "Belgium": "BEL", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA", "Bulgaria": "BGR", "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB",
    "Czech Republic": "CZE", "DR Congo": "COD", "Denmark": "DNK", "Ecuador": "ECU",
    "Egypt": "EGY", "El Salvador": "SLV", "England": "GBR", "Equatorial Guinea": "GNQ",
    "France": "FRA", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC", "Haiti": "HTI",
    "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL", "Indonesia": "IDN", "Iran": "IRN",
    "Iraq": "IRQ", "Israel": "ISR", "Italy": "ITA", "Ivory Coast": "CIV", "Jamaica": "JAM",
    "Japan": "JPN", "Kuwait": "KWT", "Mexico": "MEX", "Morocco": "MAR", "Netherlands": "NLD",
    "New Zealand": "NZL", "Nigeria": "NGA", "North Korea": "PRK", "Norway": "NOR",
    "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER", "Poland": "POL", "Portugal": "PRT",
    "Qatar": "QAT", "Republic of Ireland": "IRL", "Romania": "ROU", "Russia": "RUS",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Slovakia": "SVK",
    "Slovenia": "SVN", "South Africa": "ZAF", "South Korea": "KOR", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "Thailand": "THA", "Togo": "TGO",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United States": "USA", "Uruguay": "URY",
}

INDICATORS = {"population": "SP.POP.TOTL", "gdp_per_capita": "NY.GDP.PCAP.CD"}


def _get_json(url: str) -> list:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted host)
        return json.load(resp)


def _fetch_indicator(indicator: str, date: int) -> dict[str, float]:
    url = f"{WORLD_BANK_URL}/country/all/indicator/{indicator}?format=json&per_page=400&date={date}"
    payload = _get_json(url)
    rows = payload[1] if len(payload) > 1 and payload[1] else []
    return {r["countryiso3code"]: r["value"] for r in rows if r["value"] is not None}


def country_indicators(date: int = 2022, force: bool = False) -> pd.DataFrame:
    """World Bank population & GDP-per-capita per World Cup nation (cached)."""
    cache = config.RAW_DIR / f"worldbank_{date}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    series = {name: _fetch_indicator(code, date) for name, code in INDICATORS.items()}
    rows = []
    for team, iso3 in TEAM_ISO3.items():
        rec = {"team": team, "iso3": iso3}
        rec.update({name: series[name].get(iso3) for name in INDICATORS})
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=list(INDICATORS))
    df.to_csv(cache, index=False)
    return df


def socioeconomic_vs_success(competition: str = "men", date: int = 2022) -> dict:
    """Do population / wealth predict World Cup success?

    Correlates each nation's (log) population and GDP-per-capita with its current
    Elo rating and its number of titles. Returns Spearman coefficients and the
    merged table.
    """
    indicators = country_indicators(date=date)
    matches = features.build_matches(competition)
    ratings = elo.final_ratings(elo.compute_elo(matches)).rename("elo")

    tournaments = features.data.load_tournaments()
    comp_mask = ~tournaments["tournament_name"].str.contains("Women", case=False, na=False)
    if competition == "women":
        comp_mask = ~comp_mask
    titles = (tournaments[comp_mask]["winner"].replace(config.TEAM_ALIASES)
              .value_counts().rename("titles"))

    df = (indicators.merge(ratings, left_on="team", right_index=True, how="inner")
          .merge(titles, left_on="team", right_index=True, how="left"))
    df["titles"] = df["titles"].fillna(0).astype(int)
    df["log_population"] = np.log10(df["population"])
    df["log_gdp_per_capita"] = np.log10(df["gdp_per_capita"])

    def rho(a, b):
        r, p = stats.spearmanr(df[a], df[b])
        return {"spearman": round(float(r), 3), "p_value": round(float(p), 4)}

    return {
        "n_countries": len(df),
        "elo_vs_log_population": rho("elo", "log_population"),
        "elo_vs_log_gdp_per_capita": rho("elo", "log_gdp_per_capita"),
        "titles_vs_log_gdp_per_capita": rho("titles", "log_gdp_per_capita"),
        "table": df.sort_values("elo", ascending=False).reset_index(drop=True),
    }


# --- football-data.org (requires a free API key) -----------------------------
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"


def _football_data_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise RuntimeError(
            "Set FOOTBALL_DATA_API_KEY (free key from https://www.football-data.org) "
            "to use the live-data connector."
        )
    return key


def football_data_get(path: str) -> dict:
    """GET a football-data.org v4 endpoint, e.g. ``competitions/WC/teams``."""
    req = urllib.request.Request(
        f"{FOOTBALL_DATA_URL}/{path.lstrip('/')}", headers={"X-Auth-Token": _football_data_key()}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.load(resp)


def competition_teams(code: str = "WC") -> pd.DataFrame:
    """Current teams in a competition (requires an API key)."""
    payload = football_data_get(f"competitions/{code}/teams")
    return pd.DataFrame([
        {"team": t.get("name"), "tla": t.get("tla"), "founded": t.get("founded")}
        for t in payload.get("teams", [])
    ])
