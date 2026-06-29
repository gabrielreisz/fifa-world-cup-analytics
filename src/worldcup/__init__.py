"""worldcup -- analytics, Elo ratings and match prediction for the FIFA World Cup."""
from __future__ import annotations

from . import advanced_models, analysis, config, data, elo, evaluation, features, models, viz
from .advanced_models import DixonColes
from .features import build_matches, build_team_matches
from .models import MatchPredictor

__version__ = "0.1.0"

__all__ = [
    "advanced_models",
    "analysis",
    "config",
    "data",
    "elo",
    "evaluation",
    "features",
    "models",
    "viz",
    "build_matches",
    "build_team_matches",
    "MatchPredictor",
    "DixonColes",
]
