import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import gamma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import Model
from src.tournament import WorldCup2026

FRIENDLY_LOSS_WEIGHT = 0.2


def get_results(team, second_team=None, res=None, start_date=None, end_date=None):
    if res is None:
        raise ValueError("res is required")
    df_ = res.query(f"home_team == '{team}' or away_team == '{team}'")
    if second_team is not None:
        df_ = df_.query(f"home_team == '{second_team}' or away_team == '{second_team}'")
    if start_date is not None:
        df_ = df_.loc[df_.date >= start_date]
    if end_date is not None:
        df_ = df_.loc[df_.date <= end_date]
    return df_


def calc_loss(res, friendly_loss_weight=FRIENDLY_LOSS_WEIGHT):
    w = np.where(res.tournament == "Friendly", friendly_loss_weight, 1)
    return (
        (res.loss_result * w).sum() / w.sum(),
        (res.loss_score * w).sum() / w.sum(),
    )


def main():
    results = pd.read_csv("match_results/results_clean.csv", parse_dates=["date"])
    results["year"] = results["date"].apply(lambda x: x.year)
    all_teams = pd.read_csv("reference_data/team_universe.csv")
    current_teams = all_teams.query("category != 'past_team'").team.tolist()

    important_tournaments = [
        "FIFA World Cup",
        "FIFA World Cup qualification",
        "Copa América",
        "Gold Cup",
        "UEFA Euro",
        "UEFA Euro qualification",
        "UEFA Nations League",
        "AFC Asian Cup",
        "AFC Asian Cup qualification",
        "African Cup of Nations",
        "African Cup of Nations qualification",
        "British Home Championship",
        "Nordic Championship",
        "Central European International Cup",
    ]
    results["important"] = results.tournament.isin(important_tournaments)
    results["importance_class"] = 1
    results["importance_class"] = np.where(
        results["important"], 2, results["importance_class"]
    )
    results["importance_class"] = np.where(
        results["tournament"] == "Friendly", 0, results["importance_class"]
    )

    model = Model()
    total_matches = int(len(results))
    progress_every = max(1, total_matches // 20)

    def _fit_progress(done, total, row):
        if total <= 0:
            return
        pct = (done / total) * 100
        if row is None:
            message = f"[fit] 0/{total} (0.0%) starting"
            print(message, end="\r", flush=True)
            return
        date = pd.Timestamp(row.date).date()
        message = (
            f"[fit] {done}/{total} ({pct:.1f}%) {date} {row.home_team} vs {row.away_team}"
        )
        if done >= total:
            print(message)
        else:
            print(message, end="\r", flush=True)

    model.fit(results.iloc[::], progress_cb=_fit_progress, progress_every=progress_every)

    df_state = model.export_state_df()
    df_mu = model.export_mu_df()
    df_hga = model.export_hga_df()
    df_state["quality"] = df_state["mu_attack"] + df_state["mu_defense"]
    df_state["quality_low"] = df_state["quality"] - 2 * np.sqrt(
        df_state["sigma_attack"]
        + df_state["sigma_defense"]
        + 2 * df_state["sigma_ad"]
    )
    df_state["mu_attack_low"] = df_state["mu_attack"] - 2 * np.sqrt(
        df_state["sigma_attack"]
    )
    df_state["mu_defense_low"] = df_state["mu_defense"] - 2 * np.sqrt(
        df_state["sigma_defense"]
    )
    df_state["year"] = df_state.date.apply(lambda x: x.year)

    a = 6.0
    b = 2.1
    c = 1.0
    cdf_func = lambda x: 1.0 - gamma.cdf(c - x, a=a, scale=b / a)
    rating_func = lambda x: 100 * cdf_func(x)
    df_state["rating_attack"] = rating_func(df_state["mu_attack"])
    df_state["rating_defense"] = rating_func(df_state["mu_defense"])
    df_state["rating"] = rating_func(df_state["quality"] / 2)

    df_history = df_state.set_index(["date", "team"]).unstack().ffill()

    for team in all_teams.query("category == 'past_team'").team:
        last_date = get_results(team, res=results).date.max().date()
        df_history.loc[
            df_history.index > last_date,
            [c for c in df_history.columns if c[1] == team],
        ] = np.nan

    current_rating = (
        df_history.rating[current_teams].iloc[-1].sort_values(ascending=False).round(2)
    )
    current_rating.name = "rating"
    elo_ratings = pd.read_csv("reference_data/alt_rankings/elo_ratings_20260104.csv")
    fifa_ratings = pd.read_csv("reference_data/alt_rankings/fifa_rankings_20251218.csv")

    all_ratings = df_history.iloc[-1].unstack().T
    all_rankings = current_rating.to_frame().reset_index().rename(columns={"rating": "model"})
    all_rankings = all_rankings.merge(elo_ratings, on="team")
    all_rankings = all_rankings.merge(
        fifa_ratings[["team", "fifa_points"]].rename(columns={"fifa_points": "fifa"}),
        on="team",
    )
    all_rankings.set_index("team", inplace=True)
    C = all_rankings.columns.tolist()
    for attr in C:
        all_rankings[f"rank_{attr}"] = (
            all_rankings[attr]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype("Int64")
        )
    for attr in C:
        if attr == "model":
            continue
        all_rankings[f"rank_diff_{attr}"] = all_rankings["rank_model"] - all_rankings[
            f"rank_{attr}"
        ]

    os.makedirs("model_output", exist_ok=True)
    all_ratings.to_csv("model_output/ratings_current.csv")
    df_history.to_csv("model_output/ratings_history.csv")

    ratings_yearly = df_history["rating"].copy()
    ratings_yearly.index = pd.to_datetime(ratings_yearly.index)
    ratings_yearly = ratings_yearly.groupby(ratings_yearly.index.year).last()
    ratings_yearly.index = pd.to_datetime(
        [f"{year}-01-01" for year in ratings_yearly.index]
    )
    ratings_yearly.insert(0, "date", ratings_yearly.index.strftime("%Y-%m-%d"))
    ratings_yearly.reset_index(drop=True).to_csv(
        "model_output/ratings_history_yearly.csv", index=False
    )

    with open("model_output/model.pkl", "wb") as f:
        pickle.dump(model, f)

    group_matches_df = pd.read_csv("reference_data/world_cup_2026_group_matches.csv")
    knockout_matches_df = pd.read_csv(
        "reference_data/world_cup_2026_knockout_matches.csv"
    )
    round_of_32_df = pd.read_csv(
        "reference_data/world_cup_2026_round_of_32_combinations.csv"
    )
    round_of_32_combos = {}
    for row in round_of_32_df.to_dict(orient="records"):
        combo = str(row.get("combo", "")).strip()
        round_of_32_combos[combo] = {
            "1A": str(row.get("1A", "")).strip(),
            "1B": str(row.get("1B", "")).strip(),
            "1D": str(row.get("1D", "")).strip(),
            "1E": str(row.get("1E", "")).strip(),
            "1G": str(row.get("1G", "")).strip(),
            "1I": str(row.get("1I", "")).strip(),
            "1K": str(row.get("1K", "")).strip(),
            "1L": str(row.get("1L", "")).strip(),
        }

    t0 = WorldCup2026(
        group_matches_df=group_matches_df,
        knockout_matches_df=knockout_matches_df,
        round_of_32_combos=round_of_32_combos,
    )
    t0.simulate(model, random_state=0, record_params=False, fast_mode=True)
    wc_teams = sorted(set(t0.teams))

    data = {}

    def extract(t1, t2, is_neutral):
        s = pd.Series(
            model.predict_match(t1, t2, requires_result=True, is_neutral=is_neutral)
        )[["p_home", "p_draw", "p_away", "p_home_pens", "p_away_pens", "score_matrix"]].to_dict()
        s["score_matrix"] = s["score_matrix"].tolist()
        return s

    for team1 in wc_teams:
        for team2 in wc_teams:
            if team1 == team2:
                continue
            data.setdefault(team1, {})[team2] = {
                "home": extract(team1, team2, False),
                "neutral": extract(team1, team2, True),
            }

    with open("model_output/win_probabilities.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
