"""Tests for the external-data integrations."""
from __future__ import annotations

import pytest

from worldcup import external


def test_team_iso3_mapping_is_sane():
    assert external.TEAM_ISO3["Brazil"] == "BRA"
    assert external.TEAM_ISO3["England"] == "GBR"
    # codes are 3-letter uppercase
    assert all(len(v) == 3 and v.isupper() for v in external.TEAM_ISO3.values())


def test_football_data_requires_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FOOTBALL_DATA_API_KEY"):
        external.competition_teams()


@pytest.mark.integration
def test_world_bank_socioeconomic():
    try:
        res = external.socioeconomic_vs_success("men")
    except Exception as exc:  # pragma: no cover - network
        pytest.skip(f"World Bank API unavailable: {exc}")
    assert res["n_countries"] > 40
    assert -1 <= res["elo_vs_log_gdp_per_capita"]["spearman"] <= 1
    assert {"population", "gdp_per_capita", "elo"} <= set(res["table"].columns)
