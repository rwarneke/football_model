"""Estimate Australia's R32 schedule distribution without writing model outputs."""
import pickle
from collections import Counter

import pandas as pd

from src.tournament import WorldCup2026
from src.world_cup_results import load_results_wc2026

N = 10_000

with open("model_output/model.pkl", "rb") as f:
    model = pickle.load(f)

group_matches = pd.read_csv("reference_data/world_cup_2026_group_matches.csv")
knockout_matches = pd.read_csv("reference_data/world_cup_2026_knockout_matches.csv")
combinations = pd.read_csv("reference_data/world_cup_2026_round_of_32_combinations.csv")
round_of_32_combos = {
    str(row["combo"]): {
        key: str(row[key]) for key in ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
    }
    for row in combinations.to_dict(orient="records")
}

counts = Counter()
for seed in range(N):
    tournament = WorldCup2026(
        group_matches_df=group_matches,
        knockout_matches_df=knockout_matches,
        round_of_32_combos=round_of_32_combos,
        completed_match_results=load_results_wc2026(),
    )
    tournament.simulate(model, random_state=seed, record_params=False, fast_mode=False)
    r32_match = next(
        (
            match
            for match in tournament.matches
            if match.stage == "Round of 32"
            and "Australia" in (match.home_team, match.away_team)
        ),
        None,
    )
    rank = tournament.group_rankings["D"].index("Australia") + 1
    if rank == 1:
        counts[81] += 1
    elif rank == 2:
        counts[88] += 1
    elif r32_match is None:
        counts["Miss R32"] += 1
    else:
        opponent = (
            r32_match.away_team
            if r32_match.home_team == "Australia"
            else r32_match.home_team
        )
        group_to_match = {"E": 74, "I": 77, "K": 87}
        third_place_slot = next(
            group
            for group, match_id in group_to_match.items()
            if opponent == tournament.group_rankings[group][0]
        )
        counts[group_to_match[third_place_slot]] += 1

for key, count in sorted(counts.items(), key=lambda item: str(item[0])):
    print(f"{key}: {count} ({count / N:.2%})")
