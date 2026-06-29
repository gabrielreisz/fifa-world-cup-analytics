"""Tests for the analysis and visualization modules (synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup import analysis, config, elo, features, viz


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    teams = ["Brazil", "Germany", "Italy", "Spain", "France"]
    rows, mid = [], 0
    for year in (2006, 2010, 2014, 2018):
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                mid += 1
                hs, as_ = int(rng.integers(0, 4)), int(rng.integers(0, 4))
                rows.append({
                    "match_id": f"M-{mid}", "match_name": f"{home} vs {away}",
                    "year": year, "stage_name": "group stage", "stage_rank": 0,
                    "is_knockout": False, "extra_time": 0, "penalty_shootout": 0,
                    "went_to_penalties": False, "host_country": "Brazil",
                    "home_team_name": home, "away_team_name": away,
                    "home_team_score": hs, "away_team_score": as_,
                    "total_goals": hs + as_, "goal_difference": abs(hs - as_),
                    "match_date": pd.Timestamp(f"{year}-06-{10 + mid % 15:02d}"),
                })
    return pd.DataFrame(rows)


def test_goals_trend_keys(sample_matches):
    res = analysis.goals_trend(sample_matches)
    assert {"spearman", "spearman_p", "pearson", "pearson_p"} <= set(res)
    assert -1 <= res["spearman"] <= 1


def test_host_advantage(sample_matches):
    res = analysis.host_advantage(sample_matches)
    assert {"mean_with_host", "mean_without_host", "t_stat", "p_value"} <= set(res)
    assert 0 <= res["pct_hosts_reaching_semi"] <= 100


def test_top_attacks_and_clean_sheets(sample_matches):
    atk = analysis.top_attacks(sample_matches, min_matches=1, top=3)
    assert "goals_per_match" in atk.columns and len(atk) <= 3
    cs = analysis.clean_sheets(sample_matches)
    assert (cs.values >= 0).all()


def test_penalties_given_extra_time_handles_empty(sample_matches):
    res = analysis.penalties_given_extra_time(sample_matches)
    assert res["n_extra_time"] == 0  # fixture has no extra time


def test_elo_helpers(sample_matches):
    hist = elo.compute_elo(sample_matches)
    final = elo.final_ratings(hist)
    peak = elo.peak_ratings(hist)
    assert final.is_monotonic_decreasing
    assert (peak >= final.reindex(peak.index) - 1e-6).all()  # peak never below final


def test_team_strengths(sample_matches):
    s = features.team_strengths(sample_matches.assign(tournament_id="WC-X"))
    assert {"attack", "defense", "n_matches"} <= set(s.columns)


def test_biggest_upsets_runs(sample_matches):
    hist = elo.compute_elo(sample_matches)
    ups = analysis.biggest_upsets(hist, sample_matches, top=5)
    assert (ups["elo_gap"] > 0).all() if len(ups) else True


def test_viz_writes_figures(sample_matches, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(viz.config, "FIGURES_DIR", tmp_path)
    hist = elo.compute_elo(sample_matches)
    p1 = viz.plot_goals_distribution(sample_matches)
    p2 = viz.plot_goals_trend(sample_matches)
    p3 = viz.plot_elo_ranking(hist, top=4)
    p4 = viz.plot_rivalry_network(sample_matches, min_games=1)
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 0
