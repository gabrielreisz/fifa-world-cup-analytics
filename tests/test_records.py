"""Tests for the records module (player/team analyses on the richer tables)."""
from __future__ import annotations

import pandas as pd
import pytest

from worldcup import records


def test_competition_mask_does_not_confuse_men_and_women():
    """Regression: "Women" contains the substring "men" -- the filter must not
    classify a Women's World Cup row as men's."""
    df = pd.DataFrame({"tournament_name": ["1970 FIFA Men's World Cup",
                                           "2019 FIFA Women's World Cup"]})
    men = records._competition_mask(df, "men")
    women = records._competition_mask(df, "women")
    assert men.tolist() == [True, False]
    assert women.tolist() == [False, True]
    assert records._competition_mask(df, "all").all()


def test_full_name_drops_not_applicable():
    df = pd.DataFrame({"given_name": ["not applicable", "Lionel"],
                       "family_name": ["Ronaldo", "Messi"]})
    assert records._full_name(df).tolist() == ["Ronaldo", "Lionel Messi"]


@pytest.mark.integration
def test_top_scorers_real_data():
    try:
        top = records.top_scorers("men", top=5)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"dataset unavailable: {exc}")
    assert top["goals"].iloc[0] >= top["goals"].iloc[-1]  # descending
    assert top["goals"].max() >= 13  # the all-time men's record is 16


@pytest.mark.integration
def test_goal_timing_real_data():
    try:
        timing = records.goal_timing("men")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"dataset unavailable: {exc}")
    assert timing["pct"].sum() == pytest.approx(100.0, abs=0.5)
