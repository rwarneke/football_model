import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from scipy.stats import gamma

from src.model import Model
from src.model_elo import EloModel
from src.tournament import WorldCup2026

## Read in data ##

FRIENDLY_LOSS_WEIGHT = 0.2

results_raw = pd.read_csv("match_results/results.csv", parse_dates=["date"])
results = pd.read_csv("match_results/results_clean.csv", parse_dates=["date"])
results["year"] = results["date"].apply(lambda x: x.year)
confederations = pd.read_csv("reference_data/confederations.csv")
first_confederation = confederations.query("start_year.isna()").set_index("team").confederation
all_teams = pd.read_csv("reference_data/team_universe.csv")
current_fifa_members = all_teams.query("category == 'fifa_member'").team.tolist()
other_teams = all_teams.query("category != 'fifa_member'").team.tolist()
current_teams = all_teams.query("category != 'past_team'").team.tolist()

important_tournaments = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "Copa América", # South America
    "Gold Cup", # Noth America
    "UEFA Euro", # Europe
    "UEFA Euro qualification",
    "UEFA Nations League", # Europe 2
    "AFC Asian Cup", # Asia"
    "AFC Asian Cup qualification",
    "African Cup of Nations", # Africa
    "African Cup of Nations qualification",
    # defunct but once important
    "British Home Championship",
    "Nordic Championship",
    "Central European International Cup",
]
big_teams = ['Spain', 'Argentina', 'Brazil', 'Colombia', 'England', 'Portugal', 'France', 'Netherlands', 'Germany', 'Norway', 'Belgium', 'Switzerland', 'Croatia', 'Denmark', 'Ecuador', 'Uruguay', 'Japan', 'Italy', 'Senegal', 'Morocco', 'Austria', 'Canada', 'Greece', 'Turkey', 'Mexico', 'Chile', 'Russia', 'Paraguay', 'Serbia', 'Ukraine', 'South Korea', 'Australia', 'USA', 'Sweden', 'Iran', 'Poland', 'Algeria', 'Venezuela', 'Scotland', 'Czechia']
results["important"] = results.tournament.isin(important_tournaments)
results["importance_class"] = 1
results["importance_class"] = np.where(results["important"], 2, results["importance_class"])
results["importance_class"] = np.where(results["tournament"] == "Friendly", 0, results["importance_class"])

def get_results(team, second_team=None, res=results, start_date=None, end_date = None):
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
        (res.loss_score * w).sum() / w.sum()
    )

model = Model()
res = model.fit(results.iloc[::])

## Export metrics and define qualities ##

df_state = model.export_state_df()
df_mu = model.export_mu_df()
df_hga = model.export_hga_df()
df_state["quality"] = df_state["mu_attack"] + df_state["mu_defense"]
df_state["quality_low"] = df_state["quality"] - 2 * np.sqrt(df_state["sigma_attack"] + df_state["sigma_defense"] + 2 * df_state["sigma_ad"])
df_state["mu_attack_low"] = df_state["mu_attack"] - 2 * np.sqrt(df_state["sigma_attack"])
df_state["mu_defense_low"] = df_state["mu_defense"] - 2 * np.sqrt(df_state["sigma_defense"])
df_state["year"] = df_state.date.apply(lambda x: x.year)

## convert from params to ratings ##

a = 6.0 # larger = more teams get to be high 90s
b = 2.1 # larger = teams drawn away from zero
c = 1.0 # anything above this will be treated identically, so teams should not get near it
cdf_func = lambda x: 1.0 - gamma.cdf(c - x, a=a, scale=b/a)
rating_func = lambda x: 100 * cdf_func(x)
df_state["rating_attack"] = rating_func(df_state["mu_attack"])
df_state["rating_defense"] = rating_func(df_state["mu_defense"])
df_state["rating"] = rating_func(df_state["quality"] / 2)

## produce team histories, ratings, and rankings ##

df_history = df_state.set_index(["date", "team"]).unstack().ffill()

for team in all_teams.query("category == 'past_team'").team:
    last_date = get_results(team).date.max().date()
    df_history.loc[
        df_history.index > last_date,
        [c for c in df_history.columns if c[1] == team]
    ] = np.nan

current_rating = df_history.rating[current_teams].iloc[-1].sort_values(ascending=False).round(2)
current_rating.name = "rating"
elo_ratings = pd.read_csv("reference_data/alt_rankings/elo_ratings_20260104.csv")
fifa_ratings = pd.read_csv("reference_data/alt_rankings/fifa_rankings_20251218.csv")

# all_ratings = df_history[["rating", "rating_attack", "rating_defense", "quality", "mu_attack", "mu_defense"]].iloc[-1].unstack().T
all_ratings = df_history.iloc[-1].unstack().T
all_rankings = current_rating.to_frame().reset_index().rename(columns={"rating": "model"})
all_rankings = all_rankings.merge(elo_ratings, on="team")
all_rankings = all_rankings.merge(fifa_ratings[["team", "fifa_points"]].rename(columns={"fifa_points": "fifa"}), on="team")
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
    all_rankings[f"rank_diff_{attr}"] = all_rankings["rank_model"] - all_rankings[f"rank_{attr}"]

all_ratings.to_csv("model_output/ratings_current.csv", float_format="%.6f")
df_history.to_csv("model_output/ratings_history.csv", float_format="%.6f")
df_history_sampled = df_history.groupby(pd.to_datetime(df_history.index).strftime("%Y")).last()
df_history_sampled.to_csv("model_output/ratings_history_yearly.csv", float_format="%.6f")

## Simulations ##

N = 100_000

group_matches_df = pd.read_csv("reference_data/world_cup_2026_group_matches.csv")
knockout_matches_df = pd.read_csv("reference_data/world_cup_2026_knockout_matches.csv")
round_of_32_df = pd.read_csv("reference_data/world_cup_2026_round_of_32_combinations.csv")
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
t0.simulate(model, record_params=False)
wc_teams = sorted(set(
    t0.results_frame().home_team.tolist()
    + t0.results_frame().away_team.tolist()
))

stage_counts = None
won_group_counts = pd.Series(0, index=wc_teams, dtype=int)
wc_team_set = set(wc_teams)

for i in range(N):
    t = WorldCup2026(
        group_matches_df=group_matches_df,
        knockout_matches_df=knockout_matches_df,
        round_of_32_combos=round_of_32_combos,
    )
    t.simulate(model, random_state=i, record_params=False, fast_mode=True)
    elim = t.stage_of_elimination()
    if stage_counts is None:
        stages = sorted(set(elim.values()))
        stage_counts = pd.DataFrame(0, index=wc_teams, columns=stages, dtype=int)
    for team, stage in elim.items():
        if team in wc_team_set:
            stage_counts.at[team, stage] += 1
    won_group = t.won_group_stage()
    for team, won in won_group.items():
        if team in wc_team_set and won:
            won_group_counts.at[team] += 1
    if (i + 1) % 100 == 0 or i == N - 1:
        print(f"\r{i+1 :7} / {N}", end="")
print()

ratings_with_wins = all_ratings.loc[wc_teams].sort_values("quality", ascending=False).join(stage_counts)
wongroup = won_group_counts.infer_objects(copy=False).fillna(0).astype(int)
wongroup.name = "Won group"
ratings_with_wins = ratings_with_wins.join(wongroup.to_frame())

ratings_with_wins["odds_champion"] = (N / ratings_with_wins["8. Champion"]).round(2)
ratings_with_wins = ratings_with_wins.join(
    pd.read_csv("reference_data/betfair.csv")
    .replace({"Republic of Ireland": "Ireland", "Bosnia": "Bosnia and Herzegovina"})
    .set_index("team")
    .rename(columns={"odds": "odds_betfair"})
)
ratings_with_wins["kelly"] = (
    100
    * np.maximum(
        0,
        1 / ratings_with_wins["odds_champion"]
        - (1 - 1 / ratings_with_wins["odds_champion"])
        / (1 + 0.94 * (ratings_with_wins["odds_betfair"] - 1) - 1),
    )
).round(1)

stage_probs_output = ratings_with_wins.copy()
stage_probs_output["Qualify"] = (N - stage_probs_output["0. Qualifying"]) / N
stage_probs_output["Win Group"] = stage_probs_output["Won group"] / N
stage_probs_output["Reach R32"] = (
    N - stage_probs_output[["0. Qualifying", "1. Group"]].sum(axis=1)
) / N
stage_probs_output["Reach R16"] = (
    N
    - stage_probs_output[["0. Qualifying", "1. Group", "2. Round of 32"]].sum(axis=1)
) / N
stage_probs_output["Reach QF"] = (
    N
    - stage_probs_output[
        ["0. Qualifying", "1. Group", "2. Round of 32", "3. Round of 16"]
    ].sum(axis=1)
) / N
stage_probs_output["Reach SF"] = (
    N
    - stage_probs_output[
        [
            "0. Qualifying",
            "1. Group",
            "2. Round of 32",
            "3. Round of 16",
            "4. Quarterfinal",
        ]
    ].sum(axis=1)
) / N
stage_probs_output["Reach Final"] = (
    stage_probs_output[["7. Final", "8. Champion"]].sum(axis=1)
) / N
stage_probs_output["Champion"] = stage_probs_output["8. Champion"] / N
stage_probs_output.iloc[:, -8:].to_csv(
    "model_output/simulation_results.csv",
    float_format="%.6f",
)

## Win probabilities ##

DATA = {}

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
        DATA.setdefault(team1, {})[team2] = {
            "home": extract(team1, team2, False),
            "neutral": extract(team1, team2, True),
        }

with open("model_output/win_probabilities.json", "w") as f:
    json.dump(DATA, f)
