"""Tests for the Monte Carlo simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup import models, simulation


def _ratings_only_predictor(ratings: dict[str, float]) -> models.MatchPredictor:
    # simulate_knockout only needs .rating(); outcome/score are unused there.
    return models.MatchPredictor(outcome=None, score=None, ratings=pd.Series(ratings))


def test_elo_win_prob_symmetry():
    assert simulation.elo_win_prob(1500, 1500) == pytest.approx(0.5)
    assert simulation.elo_win_prob(1800, 1500) > 0.8
    assert simulation.elo_win_prob(1500, 1800) < 0.2


def test_knockout_probabilities_are_valid():
    pred = _ratings_only_predictor({"A": 1900, "B": 1700, "C": 1600, "D": 1500})
    res = simulation.simulate_knockout(pred, ["A", "B", "C", "D"], n_sims=4000, seed=1)
    assert res["p_champion"].sum() == pytest.approx(1.0, abs=1e-9)
    # the strongest team should win most often
    assert res.iloc[0]["team"] == "A"
    # reach-stage probabilities are ordered: champion <= final
    assert (res["p_champion"] <= res["p_final"] + 1e-9).all()


def test_knockout_trims_to_power_of_two():
    pred = _ratings_only_predictor({c: 1500 + i for i, c in enumerate("ABCDEF")})
    res = simulation.simulate_knockout(pred, list("ABCDEF"), n_sims=500, seed=0)
    assert len(res) == 4  # 6 teams -> top 4 by Elo


@pytest.fixture
def trained_predictor() -> models.MatchPredictor:
    rng = np.random.default_rng(3)
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strength = {t: i for i, t in enumerate(teams)}  # H strongest
    rows, mid = [], 0
    for year in (2010, 2014, 2018):
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                mid += 1
                hs = rng.poisson(1 + 0.3 * strength[h])
                as_ = rng.poisson(1 + 0.3 * strength[a])
                rows.append({"match_id": f"M{mid}", "match_name": f"{h} vs {a}", "year": year,
                             "home_team_name": h, "away_team_name": a,
                             "home_team_score": hs, "away_team_score": as_,
                             "match_date": pd.Timestamp(f"{year}-06-{1 + mid % 25:02d}")})
    return models.MatchPredictor.train(pd.DataFrame(rows))


def test_tournament_runs(trained_predictor):
    teams = list(trained_predictor.ratings.index)
    res = simulation.simulate_tournament(trained_predictor, teams, n_groups=2, n_sims=300, seed=2)
    assert {"p_advance", "p_champion"} <= set(res.columns)
    assert res["p_champion"].sum() == pytest.approx(1.0, abs=1e-9)
    assert (res["p_champion"] <= res["p_advance"] + 1e-9).all()
