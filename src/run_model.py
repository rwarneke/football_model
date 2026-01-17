import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import gamma

from src.model import Model
from src.model_elo import EloModel

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
    if c == "model":
        continue
    all_rankings[f"rank_diff_{attr}"] = all_rankings["rank_model"] - all_rankings[f"rank_{attr}"]

all_ratings.to_csv("model_output/ratings_current.csv")
df_history.to_csv("model_output/ratings_history.csv")