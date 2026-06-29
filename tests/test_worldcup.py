"""Unit tests for the worldcup package (synthetic data -> no network needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup import analysis, elo, features, models


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """A small, self-contained consolidated match table."""
    rng = np.random.default_rng(0)
    teams = ["Brazil", "Germany", "Italy", "Spain"]
    rows = []
    mid = 0
    for year in (2010, 2014, 2018):
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                mid += 1
                hs, as_ = int(rng.integers(0, 4)), int(rng.integers(0, 4))
                rows.append({
                    "match_id": f"M-{mid}",
                    "match_name": f"{home} vs {away}",
                    "year": year,
                    "stage_name": "group stage",
                    "stage_rank": 0,
                    "is_knockout": False,
                    "extra_time": 0,
                    "penalty_shootout": 0,
                    "went_to_penalties": False,
                    "host_country": "Brazil",
                    "home_team_name": home,
                    "away_team_name": away,
                    "home_team_score": hs,
                    "away_team_score": as_,
                    "total_goals": hs + as_,
                    "goal_difference": abs(hs - as_),
                    "match_date": pd.Timestamp(f"{year}-06-{10 + mid % 15:02d}"),
                })
    return pd.DataFrame(rows)


def test_team_matches_doubles_rows(sample_matches):
    long = features.build_team_matches(sample_matches)
    assert len(long) == 2 * len(sample_matches)
    assert (long["win"] & long["draw"]).sum() == 0  # mutually exclusive


def test_elo_is_zero_sum_per_match(sample_matches):
    hist = elo.compute_elo(sample_matches)
    assert len(hist) == 2 * len(sample_matches)
    # the home gain exactly mirrors the away loss within each match
    for _, g in hist.groupby("match_id"):
        deltas = (g["rating_after"] - g["rating_before"]).to_numpy()
        assert deltas.sum() == pytest.approx(0.0, abs=1e-9)


def test_ratings_before_match_columns(sample_matches):
    hist = elo.compute_elo(sample_matches)
    pre = elo.ratings_before_match(hist)
    assert {"match_id", "home_elo", "away_elo"} <= set(pre.columns)
    assert len(pre) == len(sample_matches)


def test_outcome_labels(sample_matches):
    frame = models.build_match_features(sample_matches)
    expected = np.where(
        frame["home_goals"] > frame["away_goals"], "home_win",
        np.where(frame["home_goals"] < frame["away_goals"], "away_win", "draw"))
    assert (frame["outcome"].to_numpy() == expected).all()


def test_predictor_probabilities_sum_to_one(sample_matches):
    predictor = models.MatchPredictor.train(sample_matches)
    r = predictor.predict("Brazil", "Germany")
    assert r["p_home_win"] + r["p_draw"] + r["p_away_win"] == pytest.approx(1.0, abs=1e-6)
    assert r["exp_home_goals"] >= 0 and r["exp_away_goals"] >= 0


def test_poisson_zero_zero_keys(sample_matches):
    res = analysis.poisson_zero_zero(sample_matches)
    assert {"lambda", "theoretical_0_0", "empirical_0_0"} == set(res)
    assert 0 <= res["empirical_0_0"] <= 1


def test_rivalries_balance_adds_up(sample_matches):
    riv = analysis.biggest_rivalries(sample_matches, min_games=1)
    totals = riv["wins_a"] + riv["draws"] + riv["wins_b"]
    assert (totals == riv["games"]).all()


@pytest.mark.integration
def test_real_data_pipeline():
    """End-to-end with the real dataset; skipped if it cannot be loaded."""
    try:
        matches = features.build_matches("men")
    except Exception as exc:  # pragma: no cover - network/offline
        pytest.skip(f"dataset unavailable: {exc}")
    assert len(matches) > 800
    assert {"total_goals", "stage_rank", "competition"} <= set(matches.columns)
