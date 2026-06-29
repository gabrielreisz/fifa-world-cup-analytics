"""Advanced statistical match models.

`DixonColes` implements the classic Dixon & Coles (1997) bivariate-Poisson model
for football scores: per-team attack/defence strengths, a home-advantage term,
the low-score dependence correction (``rho``) and optional exponential
time-decay weighting so that recent matches matter more. Parameters are fitted
by maximum likelihood with :func:`scipy.optimize.minimize`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(h: np.ndarray, a: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles dependence correction for low scores."""
    out = np.ones_like(lam, dtype=float)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    out = np.where(m00, 1.0 - lam * mu * rho, out)
    out = np.where(m01, 1.0 + lam * rho, out)
    out = np.where(m10, 1.0 + mu * rho, out)
    out = np.where(m11, 1.0 - rho, out)
    return out


@dataclass
class DixonColes:
    """Dixon-Coles bivariate-Poisson scoreline model."""

    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    max_goals: int = 10

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        xi: float = 0.0,
        min_matches: int = 8,
        max_goals: int = 10,
    ) -> DixonColes:
        """Fit by maximum likelihood.

        Parameters
        ----------
        xi:
            Time-decay rate (per year). ``0`` weights all matches equally;
            larger values down-weight older matches (Dixon-Coles use ~0.003/day).
        min_matches:
            Drop teams with fewer than this many matches (stabilises estimates).
        """
        df = matches.dropna(subset=["home_team_score", "away_team_score"]).copy()
        counts = pd.concat([df["home_team_name"], df["away_team_name"]]).value_counts()
        keep = set(counts[counts >= min_matches].index)
        df = df[df["home_team_name"].isin(keep) & df["away_team_name"].isin(keep)]

        teams = sorted(set(df["home_team_name"]) | set(df["away_team_name"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = df["home_team_name"].map(idx).to_numpy()
        ai = df["away_team_name"].map(idx).to_numpy()
        hg = df["home_team_score"].to_numpy().astype(int)
        ag = df["away_team_score"].to_numpy().astype(int)

        if xi > 0 and "match_date" in df:
            age = (df["match_date"].max() - df["match_date"]).dt.days.to_numpy() / 365.25
            weights = np.exp(-xi * age)
        else:
            weights = np.ones(len(df))

        def unpack(p):
            attack_free = p[: n - 1]
            attack = np.append(attack_free, -attack_free.sum())  # sum-to-zero constraint
            defence = p[n - 1 : 2 * n - 1]
            return attack, defence, p[-2], p[-1]

        def neg_log_lik(p):
            attack, defence, home, rho = unpack(p)
            lam = np.exp(home + attack[hi] + defence[ai])
            mu = np.exp(attack[ai] + defence[hi])
            tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)
            ll = np.log(tau) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            return -np.sum(weights * ll)

        x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25, -0.05]])
        bounds = [(-3, 3)] * (n - 1) + [(-3, 3)] * n + [(-1, 1), (-0.2, 0.2)]
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400})

        attack, defence, home, rho = unpack(res.x)
        return cls(
            teams=teams,
            attack=dict(zip(teams, attack, strict=True)),
            defence=dict(zip(teams, defence, strict=True)),
            home_adv=float(home),
            rho=float(rho),
            max_goals=max_goals,
        )

    # --- prediction ----------------------------------------------------------
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        lam = np.exp(self.home_adv + self.attack[home] + self.defence[away])
        mu = np.exp(self.attack[away] + self.defence[home])
        return float(lam), float(mu)

    def scoreline_matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        # apply the low-score correction to the 2x2 corner
        corr = np.array([[1 - lam * mu * self.rho, 1 + lam * self.rho],
                         [1 + mu * self.rho, 1 - self.rho]])
        mat[:2, :2] *= corr
        return mat / mat.sum()

    def predict(self, home: str, away: str) -> dict:
        mat = self.scoreline_matrix(home, away)
        lam, mu = self.expected_goals(home, away)
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        return {
            "home": home, "away": away,
            "exp_home_goals": round(lam, 2), "exp_away_goals": round(mu, 2),
            "p_home_win": float(np.tril(mat, -1).sum()),
            "p_draw": float(np.trace(mat)),
            "p_away_win": float(np.triu(mat, 1).sum()),
            "most_likely_score": f"{i}-{j}",
        }

    def ratings(self) -> pd.DataFrame:
        """Attack/defence table (higher attack = better, lower defence = better)."""
        return (pd.DataFrame({"attack": self.attack, "defence": self.defence})
                .sort_values("attack", ascending=False).round(3))
