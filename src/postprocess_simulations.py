import os
import sys
import json
import shutil
import re
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.world_cup_results import copy_results_wc2026_to_public

OPPONENT_STAGE_KEYS = [
    ("Round of 32", "R32_opponent_probability"),
    ("Round of 16", "R16_opponent_probability"),
    ("Quarterfinal", "QF_opponent_probability"),
    ("Semifinal", "SF_opponent_probability"),
    ("Third place", "Third_place_opponent_probability"),
    ("Final", "Final_opponent_probability"),
]

VALUE_STRIKES = (20, 40, 60, 80)
PROGRESSION_STAGE_VALUES = {
    "Round of 32": 5.0,
    "Round of 16": 10.0,
    "Quarterfinal": 20.0,
    "Semifinal": 40.0,
    "Final": 60.0,
    "Champion": 80.0,
}
WIN_VALUE_PER_90 = 5.0
GROUP_MATCHES_PATH = os.path.join(ROOT_DIR, "reference_data", "world_cup_2026_group_matches.csv")
KNOCKOUT_MATCHES_PATH = os.path.join(
    ROOT_DIR, "reference_data", "world_cup_2026_knockout_matches.csv"
)
SIMULATION_STATUS_PATH = os.path.join(
    ROOT_DIR, "model_output", "simulation_results_status.csv"
)
NEXT_STAGE_BY_STAGE = {
    "Group": "Round of 32",
    "Round of 32": "Round of 16",
    "Round of 16": "Quarterfinal",
    "Quarterfinal": "Semifinal",
    "Semifinal": "Final",
}


def _is_placeholder_team(team_name: str) -> bool:
    trimmed = str(team_name or "").strip()
    if not trimmed:
        return True
    return bool(
        re.match(r"^(Winner|Runner-up|3rd|Loser)\b", trimmed, flags=re.IGNORECASE)
        or trimmed.lower().endswith(" winner")
    )


def _load_conditional_advancement_specs():
    specs = []

    group_matches = pd.read_csv(GROUP_MATCHES_PATH)
    for row in group_matches.itertuples(index=False):
        specs.append(
            {
                "match_id": int(row.match_id),
                "stage": "Group",
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "next_stage": NEXT_STAGE_BY_STAGE["Group"],
                "basis": "full_time",
            }
        )

    knockout_matches = pd.read_csv(KNOCKOUT_MATCHES_PATH)
    for row in knockout_matches.itertuples(index=False):
        stage = str(row.stage)
        if stage not in NEXT_STAGE_BY_STAGE:
            continue
        home_team = str(row.home)
        away_team = str(row.away)
        if _is_placeholder_team(home_team) or _is_placeholder_team(away_team):
            continue
        specs.append(
            {
                "match_id": int(row.match_id),
                "stage": stage,
                "home_team": home_team,
                "away_team": away_team,
                "next_stage": NEXT_STAGE_BY_STAGE[stage],
                "basis": "after_90",
            }
        )

    return specs


def _match_outcome_from_record(match, flipped: bool, basis: str) -> str:
    if basis == "after_90":
        home_score = int(match[9])
        away_score = int(match[10])
    else:
        home_score = int(match[4])
        away_score = int(match[5])

    if home_score > away_score:
        outcome = "home_win"
    elif away_score > home_score:
        outcome = "away_win"
    else:
        outcome = "draw"

    if not flipped or outcome == "draw":
        return outcome
    return "away_win" if outcome == "home_win" else "home_win"


def _find_stage_match(stage_matches, home_id: int, away_id: int):
    for match in stage_matches:
        sim_home_id = int(match[2])
        sim_away_id = int(match[3])
        if sim_home_id == home_id and sim_away_id == away_id:
            return match, False
        if sim_home_id == away_id and sim_away_id == home_id:
            return match, True
    return None, False


def _winner(home_id, away_id, home_score, away_score, went_penalties, penalty_winner_id):
    if went_penalties and penalty_winner_id != -1:
        return penalty_winner_id
    if home_score > away_score:
        return home_id
    return away_id


def _suggestion_eligible_stages(team_id, sim_count, stage_counts, group_stage_team_ids):
    if team_id not in group_stage_team_ids:
        return set()

    guaranteed = {
        stage for stage, counts in stage_counts.items() if int(counts[team_id]) == sim_count
    }

    if "Win Group" not in guaranteed and "Reach R32" not in guaranteed:
        return {"Win Group", "Reach R32"}

    progression_order = [
        "Reach R32",
        "Reach R16",
        "Reach QF",
        "Reach SF",
        "Reach Final",
        "Champion",
    ]
    for stage in progression_order:
        if stage not in guaranteed:
            return {stage}

    return set()


def _validate_simulation_statuses(teams, sim_count, stage_counts, group_stage_team_ids):
    """Validate manual G/U/I display statuses against the simulation outcomes."""
    if not os.path.exists(SIMULATION_STATUS_PATH):
        print(f"Status validation skipped; file not found: {SIMULATION_STATUS_PATH}")
        return

    statuses = pd.read_csv(SIMULATION_STATUS_PATH, dtype=str).fillna("")
    if "team" not in statuses.columns:
        raise ValueError(f"Status file is missing a 'team' column: {SIMULATION_STATUS_PATH}")

    team_to_id = {team: idx for idx, team in enumerate(teams)}
    errors = []
    suggestions = []
    for row in statuses.to_dict(orient="records"):
        team = str(row["team"]).strip()
        team_id = team_to_id.get(team)
        if team_id is None:
            errors.append(f"{team}: not present in the simulation team list")
            continue

        eligible_suggestion_stages = _suggestion_eligible_stages(
            team_id, sim_count, stage_counts, group_stage_team_ids
        )

        for stage, counts in stage_counts.items():
            status = str(row.get(stage, "")).strip()
            if status not in {"G", "U", "I"}:
                errors.append(f"{team} — {stage}: invalid status {status!r} (expected G, U, or I)")
                continue

            count = int(counts[team_id])
            if status == "G" and count != sim_count:
                errors.append(
                    f"{team} — {stage}: status G, but simulations are "
                    f"{count}/{sim_count} ({count / sim_count:.2%})"
                )
            elif status == "I" and count != 0:
                errors.append(
                    f"{team} — {stage}: status I, but simulations are "
                    f"{count}/{sim_count} ({count / sim_count:.2%})"
                )
            elif (
                status == "U"
                and stage in eligible_suggestion_stages
                and count in {0, sim_count}
            ):
                suggested_status = "I" if count == 0 else "G"
                suggestions.append(
                    f"{team} — {stage}: status U, but simulations are "
                    f"{count}/{sim_count}; consider status {suggested_status}"
                )

    for suggestion in suggestions:
        print(f"STATUS SUGGESTION: {suggestion}")
    if errors:
        formatted_errors = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Simulation status validation failed:\n{formatted_errors}")


def main():
    sim_path = os.environ.get("SIM_OUTPUT_PATH", "model_output/simulation_runs.jsonl")
    meta_path = os.environ.get("SIM_META_PATH", "model_output/simulation_runs_meta.json")
    output_dir = os.environ.get("SIM_OUTPUT_DIR", "model_output")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    teams = meta["teams"]
    stages = meta["stages"]
    team_count = len(teams)
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    conditional_specs = _load_conditional_advancement_specs()
    conditional_advancement = {
        spec["match_id"]: {
            "match_id": int(spec["match_id"]),
            "stage": spec["stage"],
            "home_team": spec["home_team"],
            "away_team": spec["away_team"],
            "next_stage": spec["next_stage"],
            "basis": spec["basis"],
            "outcomes": {
                "home_win": {
                    "count": 0,
                    "home_team_reaches_next_stage": 0,
                    "away_team_reaches_next_stage": 0,
                },
                "draw": {
                    "count": 0,
                    "home_team_reaches_next_stage": 0,
                    "away_team_reaches_next_stage": 0,
                },
                "away_win": {
                    "count": 0,
                    "home_team_reaches_next_stage": 0,
                    "away_team_reaches_next_stage": 0,
                },
            },
        }
        for spec in conditional_specs
    }

    qualify = np.zeros(team_count, dtype=int)
    win_group = np.zeros(team_count, dtype=int)
    reach_r32 = np.zeros(team_count, dtype=int)
    reach_r16 = np.zeros(team_count, dtype=int)
    reach_qf = np.zeros(team_count, dtype=int)
    reach_sf = np.zeros(team_count, dtype=int)
    reach_final = np.zeros(team_count, dtype=int)
    champion = np.zeros(team_count, dtype=int)
    third_place = np.zeros(team_count, dtype=int)
    progression_value_sum = np.zeros(team_count, dtype=float)
    win_value_sum = np.zeros(team_count, dtype=float)
    call_value_sum = {
        strike: np.zeros(team_count, dtype=float) for strike in VALUE_STRIKES
    }
    put_value_sum = {
        strike: np.zeros(team_count, dtype=float) for strike in VALUE_STRIKES
    }

    group_pos_counts = np.zeros((team_count, 4), dtype=int)
    opponent_counts = {
        stage: [defaultdict(int) for _ in range(team_count)]
        for stage, _ in OPPONENT_STAGE_KEYS
    }

    sim_count = 0
    with open(sim_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sim_count += 1

            group_rankings = record.get("gr", [])
            qualified = set()
            for ranking in group_rankings:
                for idx, team_id in enumerate(ranking):
                    qualified.add(team_id)
                    if 0 <= idx < 4:
                        group_pos_counts[team_id, idx] += 1
                    if idx == 0:
                        win_group[team_id] += 1

            for team_id in qualified:
                qualify[team_id] += 1

            matches = record.get("m", [])
            matches_by_stage = defaultdict(list)
            teams_by_stage = defaultdict(set)
            for match in matches:
                stage_name = stages[match[0]]
                matches_by_stage[stage_name].append(match)
                teams_by_stage[stage_name].add(int(match[2]))
                teams_by_stage[stage_name].add(int(match[3]))

            for spec in conditional_specs:
                home_id = team_to_id.get(spec["home_team"])
                away_id = team_to_id.get(spec["away_team"])
                if home_id is None or away_id is None:
                    continue
                stage_matches = matches_by_stage.get(spec["stage"], [])
                match, flipped = _find_stage_match(stage_matches, home_id, away_id)
                if match is None:
                    continue
                outcome = _match_outcome_from_record(match, flipped, spec["basis"])
                outcome_counts = conditional_advancement[spec["match_id"]]["outcomes"][outcome]
                outcome_counts["count"] += 1
                next_stage_teams = teams_by_stage.get(spec["next_stage"], set())
                if home_id in next_stage_teams:
                    outcome_counts["home_team_reaches_next_stage"] += 1
                if away_id in next_stage_teams:
                    outcome_counts["away_team_reaches_next_stage"] += 1

            run_progression_value = np.zeros(team_count, dtype=float)
            run_win_value = np.zeros(team_count, dtype=float)
            for match in matches:
                stage_id = match[0]
                stage_name = stages[stage_id]
                home_id = match[2]
                away_id = match[3]
                home_score = match[4]
                away_score = match[5]
                went_penalties = match[7]
                penalty_winner_id = match[8]
                home_score_90 = match[9]
                away_score_90 = match[10]

                if stage_name == "Group":
                    if home_score > away_score:
                        run_win_value[home_id] += WIN_VALUE_PER_90
                    elif away_score > home_score:
                        run_win_value[away_id] += WIN_VALUE_PER_90
                    continue

                if home_score_90 > away_score_90:
                    run_win_value[home_id] += WIN_VALUE_PER_90
                elif away_score_90 > home_score_90:
                    run_win_value[away_id] += WIN_VALUE_PER_90

                if stage_name == "Round of 32":
                    reach_r32[home_id] += 1
                    reach_r32[away_id] += 1
                elif stage_name == "Round of 16":
                    reach_r16[home_id] += 1
                    reach_r16[away_id] += 1
                elif stage_name == "Quarterfinal":
                    reach_qf[home_id] += 1
                    reach_qf[away_id] += 1
                elif stage_name == "Semifinal":
                    reach_sf[home_id] += 1
                    reach_sf[away_id] += 1
                elif stage_name == "Final":
                    reach_final[home_id] += 1
                    reach_final[away_id] += 1

                if stage_name in opponent_counts:
                    opponent_counts[stage_name][home_id][away_id] += 1
                    opponent_counts[stage_name][away_id][home_id] += 1

                winner = _winner(
                    home_id,
                    away_id,
                    home_score,
                    away_score,
                    went_penalties,
                    penalty_winner_id,
                )
                loser = away_id if winner == home_id else home_id

                if stage_name in (
                    "Round of 32",
                    "Round of 16",
                    "Quarterfinal",
                    "Semifinal",
                ):
                    run_progression_value[loser] = PROGRESSION_STAGE_VALUES[stage_name]

                if stage_name == "Final":
                    run_progression_value[loser] = PROGRESSION_STAGE_VALUES["Final"]
                    run_progression_value[winner] = PROGRESSION_STAGE_VALUES["Champion"]
                    champion[winner] += 1
                elif stage_name == "Third place":
                    third_place[winner] += 1

            progression_value_sum += run_progression_value
            win_value_sum += run_win_value
            run_total_value = run_progression_value + run_win_value
            for strike in VALUE_STRIKES:
                call_value_sum[strike] += np.maximum(run_total_value - strike, 0.0)
                put_value_sum[strike] += np.maximum(strike - run_total_value, 0.0)

    if sim_count == 0:
        raise ValueError("No simulations found in output file.")

    qualify_prob = qualify / sim_count
    win_group_prob = win_group / sim_count
    reach_r32_prob = reach_r32 / sim_count
    reach_r16_prob = reach_r16 / sim_count
    reach_qf_prob = reach_qf / sim_count
    reach_sf_prob = reach_sf / sim_count
    reach_final_prob = reach_final / sim_count
    champion_prob = champion / sim_count
    third_place_prob = third_place / sim_count
    progression_fair_value = progression_value_sum / sim_count
    win_fair_value = win_value_sum / sim_count
    total_fair_value = progression_fair_value + win_fair_value
    group_stage_team_ids = {
        team_id for team_id in range(team_count) if int(group_pos_counts[team_id].sum()) > 0
    }

    _validate_simulation_statuses(
        teams,
        sim_count,
        {
            "Qualify": qualify,
            "Win Group": win_group,
            "Reach R32": reach_r32,
            "Reach R16": reach_r16,
            "Reach QF": reach_qf,
            "Reach SF": reach_sf,
            "Reach Final": reach_final,
            "Champion": champion,
        },
        group_stage_team_ids,
    )

    active_mask = qualify > 0
    active_teams = [teams[i] for i in range(team_count) if active_mask[i]]

    results_df = pd.DataFrame(
        {
            "team": active_teams,
            "Qualify": qualify_prob[active_mask],
            "Win Group": win_group_prob[active_mask],
            "Reach R32": reach_r32_prob[active_mask],
            "Reach R16": reach_r16_prob[active_mask],
            "Reach QF": reach_qf_prob[active_mask],
            "Reach SF": reach_sf_prob[active_mask],
            "Reach Final": reach_final_prob[active_mask],
            "Champion": champion_prob[active_mask],
        }
    )
    results_df.to_csv(
        os.path.join(output_dir, "simulation_results.csv"),
        index=False,
        float_format="%.6f",
    )

    team_entries = {}
    for team_id, team in enumerate(teams):
        if not active_mask[team_id]:
            continue
        group_probs = {
            str(idx + 1): group_pos_counts[team_id, idx] / sim_count
            for idx in range(4)
        }
        stage_probs = {
            "Qualify": float(qualify_prob[team_id]),
            "Win Group": float(win_group_prob[team_id]),
            "Reach R32": float(reach_r32_prob[team_id]),
            "Reach R16": float(reach_r16_prob[team_id]),
            "Reach QF": float(reach_qf_prob[team_id]),
            "Reach SF": float(reach_sf_prob[team_id]),
            "Third place": float(third_place_prob[team_id]),
            "Reach Final": float(reach_final_prob[team_id]),
            "Champion": float(champion_prob[team_id]),
        }

        opponent_probs = {}
        for stage_name, key in OPPONENT_STAGE_KEYS:
            counts = opponent_counts[stage_name][team_id]
            if not counts:
                opponent_probs[key] = {}
                continue
            opp_dict = {
                teams[opp_id]: count / sim_count for opp_id, count in counts.items()
            }
            opponent_probs[key] = opp_dict

        team_entries[team] = {
            "stage_probability": stage_probs,
            "group_stage_rank_probability": group_probs,
            **opponent_probs,
        }

    team_value_pricing = {
        "value_definition": {
            "progression_stage_values": PROGRESSION_STAGE_VALUES,
            "win_value_per_90": WIN_VALUE_PER_90,
            "call_put_strikes": list(VALUE_STRIKES),
        },
        "teams": {},
    }
    for team_id, team in enumerate(teams):
        if not active_mask[team_id]:
            continue
        team_value_pricing["teams"][team] = {
            "progression_fair_value": float(progression_fair_value[team_id]),
            "win_fair_value": float(win_fair_value[team_id]),
            "total_fair_value": float(total_fair_value[team_id]),
            "calls": {
                str(strike): float(call_value_sum[strike][team_id] / sim_count)
                for strike in VALUE_STRIKES
            },
            "puts": {
                str(strike): float(put_value_sum[strike][team_id] / sim_count)
                for strike in VALUE_STRIKES
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    team_prob_path = os.path.join(output_dir, "simulation_team_probabilities.json")
    with open(team_prob_path, "w") as f:
        json.dump(team_entries, f)
    team_value_path = os.path.join(output_dir, "simulation_team_value_pricing.json")
    with open(team_value_path, "w") as f:
        json.dump(team_value_pricing, f)
    conditional_advancement_payload = {"matches": {}}
    for match_id, entry in conditional_advancement.items():
        payload_entry = {
            "match_id": entry["match_id"],
            "stage": entry["stage"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "next_stage": entry["next_stage"],
            "basis": entry["basis"],
            "outcomes": {},
        }
        for outcome_key, counts in entry["outcomes"].items():
            count = counts["count"]
            payload_entry["outcomes"][outcome_key] = {
                "count": count,
                "home_team_probability": (
                    counts["home_team_reaches_next_stage"] / count if count else None
                ),
                "away_team_probability": (
                    counts["away_team_reaches_next_stage"] / count if count else None
                ),
            }
        conditional_advancement_payload["matches"][str(match_id)] = payload_entry
    conditional_advancement_path = os.path.join(
        output_dir, "simulation_match_conditional_advancement.json"
    )
    with open(conditional_advancement_path, "w") as f:
        json.dump(conditional_advancement_payload, f)

    public_output_dir = os.environ.get(
        "PUBLIC_MODEL_OUTPUT_DIR", os.path.join("web", "public", "model_output")
    )
    os.makedirs(public_output_dir, exist_ok=True)
    shutil.copy2(os.path.join(output_dir, "simulation_results.csv"), public_output_dir)
    shutil.copy2(team_prob_path, public_output_dir)
    shutil.copy2(team_value_path, public_output_dir)
    shutil.copy2(conditional_advancement_path, public_output_dir)
    for filename in ("ratings_current.csv", "ratings_history_yearly.csv"):
        source = os.path.join(output_dir, filename)
        if os.path.exists(source):
            shutil.copy2(source, public_output_dir)
    copy_results_wc2026_to_public()


if __name__ == "__main__":
    main()
