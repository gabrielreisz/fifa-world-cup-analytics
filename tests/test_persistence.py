"""Test that a trained predictor survives a save/load round-trip."""
from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup import models


def _matches() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    teams = ["A", "B", "C", "D"]
    rows, mid = [], 0
    for year in (2014, 2018):
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                mid += 1
                rows.append({"match_id": f"M{mid}", "match_name": f"{h} vs {a}", "year": year,
                             "home_team_name": h, "away_team_name": a,
                             "home_team_score": int(rng.integers(0, 4)),
                             "away_team_score": int(rng.integers(0, 4)),
                             "match_date": pd.Timestamp(f"{year}-06-{10 + mid:02d}")})
    return pd.DataFrame(rows)


def test_save_load_roundtrip(tmp_path):
    predictor = models.MatchPredictor.train(_matches())
    before = predictor.predict("A", "B")

    path = predictor.save(tmp_path / "model.joblib")
    assert path.exists()

    reloaded = models.MatchPredictor.load(path)
    after = reloaded.predict("A", "B")
    assert before == after
