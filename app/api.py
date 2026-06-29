"""FastAPI service exposing the match-prediction model.

Run with::

    pip install -e ".[app]"
    uvicorn app.api:app --reload

Then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from worldcup import elo, features, models

app = FastAPI(
    title="World Cup Match Predictor",
    version="0.1.0",
    description="Elo + Poisson match predictions for FIFA World Cup teams.",
)


@lru_cache(maxsize=2)
def _predictor(competition: str) -> models.MatchPredictor:
    return models.MatchPredictor.train(features.build_matches(competition))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ratings")
def ratings(competition: str = "men", top: int = 20) -> dict:
    matches = features.build_matches(competition)
    series = elo.final_ratings(elo.compute_elo(matches)).head(top).round(1)
    return {"competition": competition, "ratings": series.to_dict()}


@app.get("/predict")
def predict(home: str, away: str, competition: str = "men") -> dict:
    predictor = _predictor(competition)
    known = set(predictor.ratings.index)
    missing = [t for t in (home, away) if t not in known]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown team(s): {missing}")
    return predictor.predict(home, away)
