"""Model evaluation utilities: probabilistic scoring, calibration and a
rolling-origin (walk-forward) back-test across World Cup editions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .models import OutcomeModel, build_match_features


def brier_score(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Multiclass Brier score (lower is better)."""
    classes = list(proba.columns)
    onehot = pd.get_dummies(pd.Categorical(y_true, categories=classes)).to_numpy()
    return float(np.mean(np.sum((proba.to_numpy() - onehot) ** 2, axis=1)))


def calibration_curve(y_true: pd.Series, p_pred: np.ndarray, positive: str,
                      n_bins: int = 10) -> pd.DataFrame:
    """Reliability table for one class: predicted vs observed frequency per bin."""
    y = (np.asarray(y_true) == positive).astype(int)
    p = np.asarray(p_pred, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}", "n": int(m.sum()),
                     "predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return pd.DataFrame(rows)


def rolling_backtest(matches: pd.DataFrame, start_year: int = 1990,
                     min_train: int = 80) -> pd.DataFrame:
    """Walk-forward back-test: for each edition >= ``start_year`` train on all
    earlier matches and evaluate on that edition. Returns per-edition metrics
    for the Elo outcome model and a class-prior baseline, plus an ``ALL`` row.
    """
    frame = build_match_features(matches)
    rows = []
    for year in sorted(frame.loc[frame["year"] >= start_year, "year"].unique()):
        train = frame[frame["year"] < year]
        test = frame[frame["year"] == year]
        if len(train) < min_train or test.empty:
            continue
        model = OutcomeModel().fit(train)
        proba = model.predict_proba(test["elo_diff"].to_numpy())
        classes = list(model.clf.classes_)
        prior = train["outcome"].value_counts(normalize=True).reindex(classes).fillna(0.0)
        rows.append({
            "year": int(year), "n": len(test),
            "accuracy": accuracy_score(test["outcome"], proba.idxmax(axis=1)),
            "log_loss": log_loss(test["outcome"], proba[classes].to_numpy(), labels=classes),
            "brier": brier_score(test["outcome"], proba[classes]),
            "baseline_acc": (test["outcome"] == prior.idxmax()).mean(),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        w = out["n"]
        agg = {"year": "ALL", "n": int(w.sum())}
        for col in ["accuracy", "log_loss", "brier", "baseline_acc"]:
            agg[col] = float(np.average(out[col], weights=w))
        out = pd.concat([out, pd.DataFrame([agg])], ignore_index=True)
    return out
