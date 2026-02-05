import os
import sys
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.tournament import WorldCup2026

STAGES = [
    "Group",
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Third place",
    "Final",
]
GROUPS = WorldCup2026.GROUPS

_SIM_MODEL = None
_SIM_GROUP_MATCHES_DF = None
_SIM_KNOCKOUT_MATCHES_DF = None
_SIM_ROUND_OF_32_COMBOS = None
_SIM_TEAM_TO_ID = None
_SIM_STAGE_INDEX = None
_SIM_GROUP_INDEX = None


def _init_worker(model, group_matches_df, knockout_matches_df, round_of_32_combos, team_to_id):
    global _SIM_MODEL
    global _SIM_GROUP_MATCHES_DF
    global _SIM_KNOCKOUT_MATCHES_DF
    global _SIM_ROUND_OF_32_COMBOS
    global _SIM_TEAM_TO_ID
    global _SIM_STAGE_INDEX
    global _SIM_GROUP_INDEX

    _SIM_MODEL = model
    _SIM_GROUP_MATCHES_DF = group_matches_df
    _SIM_KNOCKOUT_MATCHES_DF = knockout_matches_df
    _SIM_ROUND_OF_32_COMBOS = round_of_32_combos
    _SIM_TEAM_TO_ID = team_to_id
    _SIM_STAGE_INDEX = {stage: idx for idx, stage in enumerate(STAGES)}
    _SIM_GROUP_INDEX = {group: idx for idx, group in enumerate(GROUPS)}


def _score_or_default(value, default):
    return int(value) if value is not None else default


def _match_record(match):
    stage_id = _SIM_STAGE_INDEX.get(match.stage)
    if stage_id is None:
        return None
    group_id = _SIM_GROUP_INDEX.get(match.group, -1) if match.stage == "Group" else -1
    home_id = _SIM_TEAM_TO_ID.get(match.home_team)
    away_id = _SIM_TEAM_TO_ID.get(match.away_team)
    if home_id is None or away_id is None:
        return None

    hs = int(match.home_score)
    as_ = int(match.away_score)
    went_et = 1 if match.went_extra_time else 0
    went_pens = 1 if match.went_penalties else 0
    pen_winner_id = (
        _SIM_TEAM_TO_ID.get(match.penalty_winner, -1)
        if match.penalty_winner
        else -1
    )
    hs90 = _score_or_default(match.home_score_90, hs)
    as90 = _score_or_default(match.away_score_90, as_)
    hs120 = _score_or_default(match.home_score_120, -1)
    as120 = _score_or_default(match.away_score_120, -1)

    return [
        stage_id,
        group_id,
        home_id,
        away_id,
        hs,
        as_,
        went_et,
        went_pens,
        pen_winner_id,
        hs90,
        as90,
        hs120,
        as120,
    ]


def _group_payload(group_tables, group_rankings):
    group_rank_payload = []
    group_table_payload = []

    for group in GROUPS:
        ranking = group_rankings.get(group, [])
        group_rank_payload.append([_SIM_TEAM_TO_ID[t] for t in ranking])

        table_entries = []
        table = group_tables.get(group)
        if table is not None:
            for team in ranking:
                row = table.loc[team]
                table_entries.append(
                    [
                        _SIM_TEAM_TO_ID[team],
                        int(row["points"]),
                        int(row["gd"]),
                        int(row["gf"]),
                        int(row["ga"]),
                        int(row["w"]),
                        int(row["d"]),
                        int(row["l"]),
                    ]
                )
        group_table_payload.append(table_entries)

    return group_rank_payload, group_table_payload


def _simulate_chunk(seed_start, seed_count):
    lines = []
    for seed in range(seed_start, seed_start + seed_count):
        t = WorldCup2026(
            group_matches_df=_SIM_GROUP_MATCHES_DF,
            knockout_matches_df=_SIM_KNOCKOUT_MATCHES_DF,
            round_of_32_combos=_SIM_ROUND_OF_32_COMBOS,
        )
        t.simulate(_SIM_MODEL, random_state=seed, record_params=False, fast_mode=False)

        group_rank_payload, group_table_payload = _group_payload(
            t.group_tables, t.group_rankings
        )

        matches = []
        for match in t.matches:
            rec = _match_record(match)
            if rec is not None:
                matches.append(rec)

        record = {
            "s": seed,
            "seed": seed,
            "gr": group_rank_payload,
            "gt": group_table_payload,
            "m": matches,
        }
        lines.append(json.dumps(record, separators=(",", ":")))

    return lines, seed_count


def main():
    model_path = os.environ.get("SIM_MODEL_PATH", "model_output/model.pkl")
    output_path = os.environ.get("SIM_OUTPUT_PATH", "model_output/simulation_runs.jsonl")
    meta_path = os.environ.get("SIM_META_PATH", "model_output/simulation_runs_meta.json")
    n_sims = int(os.environ.get("SIM_N", "10000"))
    workers = int(os.environ.get("SIM_WORKERS", "8"))
    chunk_size = int(os.environ.get("SIM_CHUNK_SIZE", str(math.ceil(n_sims / workers))))

    with open(model_path, "rb") as f:
        model = pickle.load(f)

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

    all_teams = pd.read_csv("reference_data/team_universe.csv")
    current_teams = (
        all_teams.query("category != 'past_team'")
        .team.astype(str)
        .str.strip()
        .tolist()
    )
    team_to_id = {team: idx for idx, team in enumerate(current_teams)}

    meta = {
        "format_version": 1,
        "teams": current_teams,
        "groups": GROUPS,
        "stages": STAGES,
        "match_fields": [
            "stage_id",
            "group_id",
            "home_id",
            "away_id",
            "home_score",
            "away_score",
            "went_extra_time",
            "went_penalties",
            "penalty_winner_id",
            "home_score_90",
            "away_score_90",
            "home_score_120",
            "away_score_120",
        ],
        "group_table_fields": ["team_id", "points", "gd", "gf", "ga", "w", "d", "l"],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    tasks = []
    for seed_start in range(0, n_sims, chunk_size):
        seed_count = min(chunk_size, n_sims - seed_start)
        tasks.append((seed_start, seed_count))

    completed = 0
    with open(output_path, "w") as out_f, ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            model,
            group_matches_df,
            knockout_matches_df,
            round_of_32_combos,
            team_to_id,
        ),
    ) as executor:
        futures = [executor.submit(_simulate_chunk, start, count) for start, count in tasks]
        for future in as_completed(futures):
            lines, chunk_done = future.result()
            for line in lines:
                out_f.write(line + "\n")
            completed += chunk_done
            pct = (completed / n_sims) * 100 if n_sims else 100.0
            print(f"\r{completed:7} / {n_sims} ({pct:5.1f}%)", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
