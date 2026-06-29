"""Match-prediction models.

Two complementary, leak-free models built on chronological Elo ratings:

* :class:`OutcomeModel` -- multinomial logistic regression mapping the Elo gap
  to calibrated ``P(home win / draw / away win)`` probabilities.
* :class:`PoissonScoreModel` -- Poisson regression on Elo that yields expected
  goals and a full scoreline probability matrix (a Dixon-Coles-lite model).

Both consume ``home_elo``/``away_elo`` taken *before* each match, so there is no
target leakage and a strict temporal train/test split is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss

from . import config, elo

OUTCOMES = ["away_win", "draw", "home_win"]


def build_match_features(matches: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join pre-match Elo onto each match and label the outcome.

    Returns columns: ``match_id, year, home_team_name, away_team_name,
    home_elo, away_elo, elo_diff, home_goals, away_goals, outcome``.
    """
    if history is None:
        history = elo.compute_elo(matches)
    pre = elo.ratings_before_match(history)

    df = matches.merge(pre[["match_id", "home_elo", "away_elo"]], on="match_id", how="inner").copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_goals"] = df["home_team_score"]
    df["away_goals"] = df["away_team_score"]
    df["outcome"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] < df["away_goals"]],
        ["home_win", "away_win"],
        default="draw",
    )
    return df[["match_id", "year", "home_team_name", "away_team_name", "home_elo",
               "away_elo", "elo_diff", "home_goals", "away_goals", "outcome"]]


class OutcomeModel:
    """Calibrated 1X2 probabilities from the Elo gap."""

    def __init__(self) -> None:
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, frame: pd.DataFrame) -> OutcomeModel:
        self.clf.fit(frame[["elo_diff"]], frame["outcome"])
        return self

    def predict_proba(self, elo_diff: np.ndarray | float) -> pd.DataFrame:
        x = pd.DataFrame({"elo_diff": np.asarray(elo_diff, dtype=float).reshape(-1)})
        proba = self.clf.predict_proba(x)
        return pd.DataFrame(proba, columns=list(self.clf.classes_))


class PoissonScoreModel:
    """Expected goals and scoreline distribution from Elo.

    Trains a single Poisson regression on a team-perspective table where the
    target is ``goals_for`` and the features are the team's own Elo, the
    opponent's Elo and a home indicator.
    """

    def __init__(self, max_goals: int = 10) -> None:
        self.max_goals = max_goals
        self.reg = PoissonRegressor(alpha=1e-6, max_iter=300)

    def fit(self, frame: pd.DataFrame) -> PoissonScoreModel:
        home = pd.DataFrame({"team_elo": frame["home_elo"], "opp_elo": frame["away_elo"],
                             "is_home": 1.0, "goals": frame["home_goals"]})
        away = pd.DataFrame({"team_elo": frame["away_elo"], "opp_elo": frame["home_elo"],
                             "is_home": 0.0, "goals": frame["away_goals"]})
        long = pd.concat([home, away], ignore_index=True).dropna()
        self.reg.fit(long[["team_elo", "opp_elo", "is_home"]], long["goals"])
        return self

    def expected_goals(self, home_elo: float, away_elo: float) -> tuple[float, float]:
        cols = ["team_elo", "opp_elo", "is_home"]
        x = pd.DataFrame([[home_elo, away_elo, 1.0], [away_elo, home_elo, 0.0]], columns=cols)
        lam_home, lam_away = (float(v) for v in self.reg.predict(x))
        return lam_home, lam_away

    def scoreline_matrix(self, home_elo: float, away_elo: float) -> np.ndarray:
        lam_home, lam_away = self.expected_goals(home_elo, away_elo)
        g = np.arange(self.max_goals + 1)
        ph = stats.poisson.pmf(g, lam_home)
        pa = stats.poisson.pmf(g, lam_away)
        return np.outer(ph, pa)

    def predict(self, home_elo: float, away_elo: float) -> dict:
        m = self.scoreline_matrix(home_elo, away_elo)
        lam_home, lam_away = self.expected_goals(home_elo, away_elo)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        return {
            "exp_home_goals": round(lam_home, 2),
            "exp_away_goals": round(lam_away, 2),
            "p_home_win": float(np.tril(m, -1).sum()),
            "p_draw": float(np.trace(m)),
            "p_away_win": float(np.triu(m, 1).sum()),
            "most_likely_score": f"{i}-{j}",
        }


@dataclass
class EvalResult:
    accuracy: float
    log_loss: float
    baseline_accuracy: float
    baseline_log_loss: float
    n_train: int
    n_test: int


def temporal_evaluate(matches: pd.DataFrame, history: pd.DataFrame | None = None,
                      cutoff_year: int = 2014) -> EvalResult:
    """Train on editions before ``cutoff_year``, test on the rest.

    Compares the Elo outcome model against a class-prior baseline.
    """
    frame = build_match_features(matches, history)
    train, test = frame[frame["year"] < cutoff_year], frame[frame["year"] >= cutoff_year]

    model = OutcomeModel().fit(train)
    proba = model.predict_proba(test["elo_diff"].to_numpy())
    classes = list(model.clf.classes_)
    pred = proba.idxmax(axis=1)

    prior = train["outcome"].value_counts(normalize=True).reindex(classes).fillna(0.0)
    base_proba = np.tile(prior.to_numpy(), (len(test), 1))

    return EvalResult(
        accuracy=accuracy_score(test["outcome"], pred),
        log_loss=log_loss(test["outcome"], proba[classes].to_numpy(), labels=classes),
        baseline_accuracy=(test["outcome"] == prior.idxmax()).mean(),
        baseline_log_loss=log_loss(test["outcome"], base_proba, labels=classes),
        n_train=len(train),
        n_test=len(test),
    )


@dataclass
class MatchPredictor:
    """High-level predictor bundling Elo, the outcome model and the score model."""

    outcome: OutcomeModel
    score: PoissonScoreModel
    ratings: pd.Series

    @classmethod
    def train(cls, matches: pd.DataFrame, history: pd.DataFrame | None = None) -> MatchPredictor:
        if history is None:
            history = elo.compute_elo(matches)
        frame = build_match_features(matches, history)
        return cls(
            outcome=OutcomeModel().fit(frame),
            score=PoissonScoreModel().fit(frame),
            ratings=elo.final_ratings(history),
        )

    def rating(self, team: str) -> float:
        return float(self.ratings.get(team, config.ELO_START))

    def save(self, path: str | Path) -> Path:
        """Persist the trained predictor to disk (joblib)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> MatchPredictor:
        """Load a predictor previously saved with :meth:`save`."""
        return joblib.load(path)

    def predict(self, home: str, away: str) -> dict:
        rh, ra = self.rating(home), self.rating(away)
        proba = self.outcome.predict_proba(rh - ra).iloc[0]
        score = self.score.predict(rh, ra)
        return {
            "home": home, "away": away, "home_elo": round(rh, 1), "away_elo": round(ra, 1),
            # headline 1X2 probabilities come from the calibrated outcome model
            "p_home_win": float(proba.get("home_win", 0.0)),
            "p_draw": float(proba.get("draw", 0.0)),
            "p_away_win": float(proba.get("away_win", 0.0)),
            # the Poisson model contributes expected goals + the modal scoreline
            "exp_home_goals": score["exp_home_goals"],
            "exp_away_goals": score["exp_away_goals"],
            "most_likely_score": score["most_likely_score"],
        }
