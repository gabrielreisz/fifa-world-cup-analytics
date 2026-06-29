"""Command-line interface: ``worldcup <command>``."""
from __future__ import annotations

import typer

from . import analysis, data, elo, features, models, records, simulation, viz

app = typer.Typer(add_completion=False, help="FIFA World Cup analytics toolkit.")


@app.command("build-data")
def build_data(force: bool = typer.Option(False, help="Re-download even if cached.")) -> None:
    """Download and cache the raw datasets."""
    tables = data.download_raw(force=force)
    for name, df in tables.items():
        typer.echo(f"  {name:12s} {df.shape[0]:>5} rows  -> data/raw/{name}.csv")
    typer.secho("Data ready.", fg=typer.colors.GREEN)


@app.command("report")
def report(competition: str = typer.Option("men", help="men | women | all")) -> None:
    """Print headline findings and render the figure set to reports/figures."""
    matches = features.build_matches(competition=competition)
    history = elo.compute_elo(matches)

    trend = analysis.goals_trend(matches)
    host = analysis.host_advantage(matches)
    pois = analysis.poisson_zero_zero(matches)

    typer.secho(f"\n=== {competition.upper()} World Cup -- {len(matches)} matches ===", bold=True)
    typer.echo(f"Goals/match trend over time : Spearman={trend['spearman']:.3f} (p={trend['spearman_p']:.4f})")
    typer.echo(f"Host on pitch t-test        : p={host['p_value']:.3f} "
               f"({host['mean_with_host']:.2f} vs {host['mean_without_host']:.2f} goals)")
    typer.echo(f"0-0 Poisson vs reality      : {pois['theoretical_0_0']*100:.1f}% vs "
               f"{pois['empirical_0_0']*100:.1f}%")

    typer.secho("\nTop 5 Elo (current):", bold=True)
    for team, r in elo.final_ratings(history).head(5).items():
        typer.echo(f"  {team:18s} {r:6.0f}")

    paths = viz.generate_all(matches, history)
    typer.secho(f"\nSaved {len(paths)} figures to reports/figures/", fg=typer.colors.GREEN)


@app.command("evaluate")
def evaluate(cutoff: int = typer.Option(2014, help="Test on editions from this year on.")) -> None:
    """Temporally back-test the Elo outcome model."""
    matches = features.build_matches("men")
    res = models.temporal_evaluate(matches, cutoff_year=cutoff)
    typer.secho(f"\nModel vs baseline (test = {cutoff}+):", bold=True)
    typer.echo(f"  accuracy : {res.accuracy:.3f}  (baseline {res.baseline_accuracy:.3f})")
    typer.echo(f"  log-loss : {res.log_loss:.3f}  (baseline {res.baseline_log_loss:.3f})")
    typer.echo(f"  n_train={res.n_train}  n_test={res.n_test}")


@app.command("predict")
def predict(home: str, away: str) -> None:
    """Predict a hypothetical match, e.g. ``worldcup predict Brazil Argentina``."""
    matches = features.build_matches("men")
    predictor = models.MatchPredictor.train(matches)
    r = predictor.predict(home, away)
    typer.secho(f"\n{home} (Elo {r['home_elo']}) vs {away} (Elo {r['away_elo']})", bold=True)
    typer.echo(f"  P({home} win) = {r['p_home_win']*100:5.1f}%")
    typer.echo(f"  P(draw)      = {r['p_draw']*100:5.1f}%")
    typer.echo(f"  P({away} win) = {r['p_away_win']*100:5.1f}%")
    typer.echo(f"  expected goals: {r['exp_home_goals']} - {r['exp_away_goals']} "
               f"(most likely {r['most_likely_score']})")


@app.command("simulate")
def simulate(
    top: int = typer.Option(16, help="Use the top-N teams by current Elo."),
    mode: str = typer.Option("knockout", help="knockout | tournament"),
    n_sims: int = typer.Option(20000, help="Number of Monte Carlo runs."),
) -> None:
    """Monte Carlo title odds for the strongest teams."""
    matches = features.build_matches("men")
    predictor = models.MatchPredictor.train(matches)
    teams = list(predictor.ratings.head(top).index)
    if mode == "tournament":
        res = simulation.simulate_tournament(predictor, teams, n_sims=max(n_sims // 4, 1000))
        cols = ["team", "elo", "p_advance", "p_champion"]
    else:
        res = simulation.simulate_knockout(predictor, teams, n_sims=n_sims)
        cols = ["team", "elo", "p_champion", "p_final", "p_semifinal"]

    typer.secho(f"\nTitle odds ({mode}, top {top}):", bold=True)
    for r in res[cols].head(10).itertuples(index=False):
        typer.echo(f"  {r.team:16s} Elo {int(r.elo):>4}  champion {r.p_champion * 100:5.1f}%")


@app.command("top-scorers")
def top_scorers(competition: str = typer.Option("men"), top: int = typer.Option(10)) -> None:
    """All-time top scorers (own goals excluded)."""
    df = records.top_scorers(competition, top=top)
    typer.secho(f"\nTop scorers ({competition}):", bold=True)
    for name, row in df.iterrows():
        typer.echo(f"  {name:24s} {int(row['goals']):>2} goals  ({row['team']})")


if __name__ == "__main__":
    app()
