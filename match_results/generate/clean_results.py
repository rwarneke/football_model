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
fil = (results_merged.home_team.isin(team_universe)) & (results_merged.away_team.isin(team_universe))
results = results_merged.loc[fil]


# add confederations (date-aware membership)
conf = pd.read_csv("../reference_data/confederations.csv")

# Expect columns: team,confederation,start_year,end_year
# Make sure types are sane
conf = conf.rename(columns={
    "team": "team",
    "confederation": "confederation",
    "start_year": "start_year",
    "end_year": "end_year",
})

conf["team"] = conf["team"].astype(str).str.strip()
conf["confederation"] = conf["confederation"].astype(str).str.strip()

# Empty -> NA, then to numeric
conf["start_year"] = pd.to_numeric(conf["start_year"], errors="coerce")
conf["end_year"] = pd.to_numeric(conf["end_year"], errors="coerce")

# We'll match on year granularity (consistent with your schema)
results = results.copy()
results["match_year"] = results["date"].dt.year

def attach_confederation(results_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    side: 'home' or 'away'
    Adds columns:
      - {side}_confederation
    """
    team_col = f"{side}_team"
    out_col = f"{side}_confederation"

    # Join on team to get all candidate memberships, then filter by year window
    tmp = results_df[[team_col, "match_year"]].merge(
        conf,
        left_on=team_col,
        right_on="team",
        how="left",
    )

    in_window = (
        tmp["confederation"].notna()
        & (tmp["start_year"].isna() | (tmp["start_year"] <= tmp["match_year"]))
        & (tmp["end_year"].isna() | (tmp["match_year"] <= tmp["end_year"]))
    )
    tmp = tmp.loc[in_window, [team_col, "match_year", "confederation", "start_year", "end_year"]]

    # If multiple memberships match (should be rare), pick the most specific:
    # prefer the one with the latest start_year (closest, most recent assignment)
    tmp = tmp.sort_values(
        by=[team_col, "match_year", "start_year", "end_year"],
        ascending=[True, True, False, True],
        kind="mergesort",  # stable
    )

    tmp = tmp.drop_duplicates(subset=[team_col, "match_year"], keep="first")
    tmp = tmp.rename(columns={"confederation": out_col})

    # Merge back onto results
    results_df = results_df.merge(
        tmp[[team_col, "match_year", out_col]],
        on=[team_col, "match_year"],
        how="left",
        validate="m:1",
    )
    return results_df

results = attach_confederation(results, "home")
results = attach_confederation(results, "away")

# Optional diagnostics: confederation not found for a team-year
missing_home = results.loc[results["home_confederation"].isna(), "home_team"].unique()
missing_away = results.loc[results["away_confederation"].isna(), "away_team"].unique()
missing = sorted(set(missing_home).union(set(missing_away)))

if missing:
    print(f"[confederations] Missing confederation for {len(missing)} team(s) in at least one match-year.")
    print(missing[:50], "..." if len(missing) > 50 else "")

# Optional: remove helper column if you don't want it in output
results = results.drop(columns=["match_year"])


results.to_csv("results_clean.csv", index=False)
