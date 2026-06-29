"""Tests for the Dixon-Coles model and the evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup import advanced_models, evaluation


@pytest.fixture
def league() -> pd.DataFrame:
    """Round-robin synthetic league over several seasons with clear strengths."""
    rng = np.random.default_rng(11)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 1.8, "B": 1.3, "C": 1.0, "D": 0.6}
    rows, mid = [], 0
    for year in range(2006, 2023, 4):
        for _ in range(6):  # several round-robins per edition
            for i, h in enumerate(teams):
                for a in teams[i + 1:]:
                    mid += 1
                    rows.append({
                        "match_id": f"M{mid}", "year": year,
                        "home_team_name": h, "away_team_name": a,
                        "home_team_score": int(rng.poisson(strength[h])),
                        "away_team_score": int(rng.poisson(strength[a])),
                        "match_date": pd.Timestamp(f"{year}-06-15"),
                    })
    return pd.DataFrame(rows)


def test_dixon_coles_fit_and_predict(league):
    dc = advanced_models.DixonColes.fit(league, min_matches=5)
    assert set(dc.teams) == {"A", "B", "C", "D"}
    # sum-to-zero identifiability constraint on attack
    assert sum(dc.attack.values()) == pytest.approx(0.0, abs=1e-6)
    pred = dc.predict("A", "D")
    assert pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"] == pytest.approx(1.0, abs=1e-6)
    # strongest team should be favoured against the weakest
    assert pred["p_home_win"] > pred["p_away_win"]
    assert dc.ratings().loc["A", "attack"] > dc.ratings().loc["D", "attack"]


def test_brier_and_calibration():
    y = pd.Series(["home_win", "draw", "away_win", "home_win"])
    proba = pd.DataFrame({"away_win": [0.1, 0.2, 0.7, 0.2],
                          "draw": [0.2, 0.6, 0.2, 0.2],
                          "home_win": [0.7, 0.2, 0.1, 0.6]})
    assert 0 <= evaluation.brier_score(y, proba) <= 2
    cal = evaluation.calibration_curve(y, proba["home_win"].to_numpy(), "home_win", n_bins=5)
    assert {"predicted", "observed", "n"} <= set(cal.columns)


def test_rolling_backtest(league):
    bt = evaluation.rolling_backtest(league, start_year=2010, min_train=20)
    assert not bt.empty
    assert bt.iloc[-1]["year"] == "ALL"
    assert 0 <= bt.iloc[-1]["accuracy"] <= 1
