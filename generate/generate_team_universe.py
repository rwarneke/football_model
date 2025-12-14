import pandas as pd

fifa_members_raw = pd.read_csv("fifa_members_with_flag_sources.csv")
mapping = {
    row.original_name: row.replacement_name
    for row in pd.read_csv("fifa_member_to_canonical_name_map.csv").itertuples()
}

EXTRA_TEAMS = [
    "Czechoslovakia", 
    "East Germany", 
    "Saarland",
    "Yugoslavia", 
]

with open("team_universe.csv", "w") as f:
    for team in fifa_members_raw.country:
        f.write(mapping.get(team, team) + "\n")
    for team in EXTRA_TEAMS:
        f.write(team + "\n")
        