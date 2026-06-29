"""Interactive World Cup dashboard.

Run with::

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `worldcup` importable on Streamlit Community Cloud even when the package
# is not pip-installed (the Cloud only runs `pip install -r requirements.txt`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from worldcup import analysis, elo, features, models, records, simulation

st.set_page_config(page_title="World Cup Analytics", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Loading World Cup data…")
def load(competition: str) -> pd.DataFrame:
    return features.build_matches(competition)


@st.cache_resource(show_spinner="Training models…")
def get_predictor(competition: str) -> models.MatchPredictor:
    return models.MatchPredictor.train(load(competition))


st.title("⚽ FIFA World Cup — Analytics & Match Predictor")
st.caption("Elo ratings + a Poisson scoreline model, built on the Fjelstul World Cup database.")

competition = st.sidebar.selectbox("Competition", ["men", "women"], format_func=str.title)
matches = load(competition)
history = elo.compute_elo(matches)
predictor = get_predictor(competition)
teams = sorted(predictor.ratings.index)

tab_predict, tab_sim, tab_records, tab_ratings, tab_trends = st.tabs(
    ["🔮 Predict", "🏆 Simulate", "🏅 Records", "📊 Elo ratings", "📈 Trends"]
)

with tab_predict:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home / Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = c2.selectbox("Away / Team B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
    if home == away:
        st.warning("Pick two different teams.")
    else:
        r = predictor.predict(home, away)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{r['p_home_win']*100:.0f}%")
        m2.metric("Draw", f"{r['p_draw']*100:.0f}%")
        m3.metric(f"{away} win", f"{r['p_away_win']*100:.0f}%")
        st.write(
            f"**Expected goals:** {home} {r['exp_home_goals']} – {r['exp_away_goals']} {away} "
            f"· most likely scoreline **{r['most_likely_score']}** "
            f"(Elo {r['home_elo']} vs {r['away_elo']})"
        )
        st.bar_chart(pd.Series(
            {home: r["p_home_win"], "Draw": r["p_draw"], away: r["p_away_win"]}, name="probability"))


@st.cache_data(show_spinner="Running Monte Carlo…")
def run_simulation(competition: str, top: int, mode: str, n_sims: int) -> pd.DataFrame:
    pred = get_predictor(competition)
    chosen = list(pred.ratings.head(top).index)
    if mode == "Full tournament":
        return simulation.simulate_tournament(pred, chosen, n_sims=n_sims)
    return simulation.simulate_knockout(pred, chosen, n_sims=n_sims)


with tab_sim:
    st.subheader("Monte Carlo title odds")
    st.caption("Simulate the strongest teams thousands of times using the Elo + Poisson models.")
    c1, c2 = st.columns(2)
    topn = c1.slider("Teams (by Elo)", 4, 32, 16, step=4)
    mode = c2.radio("Format", ["Knockout", "Full tournament"], horizontal=True)
    sim = run_simulation(competition, topn, mode, 20000 if mode == "Knockout" else 4000)
    st.bar_chart(sim.set_index("team")["p_champion"].head(12))
    pct_cols = [c for c in sim.columns if c.startswith("p_")]
    show = sim.assign(**{c: (sim[c] * 100).round(1) for c in pct_cols})
    st.dataframe(show, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner="Loading records…")
def cached(fn_name: str, competition: str):
    return getattr(records, fn_name)(competition)


with tab_records:
    st.subheader("🥇 All-time top scorers")
    st.dataframe(cached("top_scorers", competition), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⏱️ When are goals scored?")
        st.bar_chart(cached("goal_timing", competition)["pct"])
    with c2:
        st.subheader("🎯 Shootout conversion")
        st.dataframe(cached("shootout_conversion", competition)[["kicks", "conversion_pct"]],
                     use_container_width=True)

with tab_ratings:
    st.subheader("Current Elo ranking")
    st.dataframe(elo.final_ratings(history).round(0).head(20).rename("Elo"), use_container_width=True)

with tab_trends:
    st.subheader("Average goals per match by edition")
    st.line_chart(matches.groupby("year")["total_goals"].mean())
    st.subheader("Most frequent rivalries")
    st.dataframe(analysis.biggest_rivalries(matches).drop(columns="team_a", errors="ignore"),
                 use_container_width=True, hide_index=True)
