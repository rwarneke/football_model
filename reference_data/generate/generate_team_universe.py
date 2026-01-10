import pandas as pd

if input("This will overwrite the list into a raw state with empty confederations - proceed? (type yes): ") != "yes":
    raise Exception("cancelled.")

fifa_members_raw = pd.read_csv("reference_data/fifa_members_with_flag_sources.csv")
mapping = {
    row.original_name: row.replacement_name
    for row in pd.read_csv("reference_data/fifa_member_to_canonical_name_map.csv").itertuples()
}

EXTRA_TEAMS_PAST = [
    "Czechoslovakia", 
    "East Germany", 
    "Saarland",
    "Yugoslavia", 
]

EXTRA_TEAMS_NON_FIFA = [
    "Martinique",
    "Guadeloupe",
    "French Guiana",
    "Bonaire",
    "Sint Maarten",
    "Northern Mariana Islands"
]

with open("reference_data/team_universe.csv", "w") as f:
    f.write("team,category\n")
    for team in fifa_members_raw.country:
        f.write(f"{mapping.get(team, team)},fifa_member\n")
    for team in EXTRA_TEAMS_NON_FIFA:
        f.write(f"{team},non_fifa_member\n")
    for team in EXTRA_TEAMS_PAST:
        f.write(f"{team},past_team\n")
