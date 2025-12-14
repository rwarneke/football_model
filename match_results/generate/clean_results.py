### Run this from inside match_results/ ###

import pandas as pd

results_raw = pd.read_csv("results.csv", parse_dates=["date"])
shootouts_raw = pd.read_csv("shootouts.csv", parse_dates=["date"])

results_merged = pd.merge(
    results_raw,
    shootouts_raw.rename(columns={"winner": "shootout_winner", "first_shooter": "shootout_first_shooter"}),
    on=["date", "home_team", "away_team"],
    how="left"
)

renamer = {
    row.original_name: row.replacement_name
    for row in pd.read_csv("../reference_data/kaggle_team_to_canonical_name_map.csv").itertuples()
}

results_merged["home_team"] = results_merged["home_team"].apply(lambda t: renamer.get(t, t))
results_merged["away_team"] = results_merged["away_team"].apply(lambda t: renamer.get(t, t))
results_merged["shootout_winner"] = results_merged["shootout_winner"].apply(lambda t: renamer.get(t, t))
results_merged["shootout_first_shooter"] = results_merged["shootout_first_shooter"].apply(lambda t: renamer.get(t, t))

team_universe = pd.read_csv("../reference_data/team_universe.csv", header=None)[0]
results = results_merged.loc[
    (results_merged.home_team.isin(team_universe))
    & (results_merged.away_team.isin(team_universe))
]

results.to_csv("results_clean.csv", index=False)
