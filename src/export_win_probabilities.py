from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.model import MAX_GOALS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, "web", "public", "model_output")


def round_sig(value, sig=7):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return float(value)
    if not np.isfinite(value):
        return float(value)
    if value == 0:
        return 0.0
    return float(f"{float(value):.{sig}g}")


def current_teams() -> list[str]:
    all_teams = pd.read_csv(os.path.join(ROOT_DIR, "reference_data", "team_universe.csv"))
    return sorted(all_teams.query("category != 'past_team'").team.tolist())


def _load_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for filename in (
        os.path.join(ROOT_DIR, "reference_data", "fifa_member_to_canonical_name_map.csv"),
        os.path.join(ROOT_DIR, "reference_data", "kaggle_team_to_canonical_name_map.csv"),
    ):
        if not os.path.exists(filename):
            continue
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            original = str(row.get("original_name", "")).strip()
            replacement = str(row.get("replacement_name", "")).strip()
            if original and replacement:
                mapping[original] = replacement
    return mapping


def _is_placeholder(name: str) -> bool:
    return bool(re.search(r"\bwinner$", name, re.IGNORECASE))


def world_cup_subset_teams(model) -> list[str]:
    group_matches_df = pd.read_csv(os.path.join(ROOT_DIR, "reference_data", "world_cup_2026_group_matches.csv"))
    knockout_matches_df = pd.read_csv(
        os.path.join(ROOT_DIR, "reference_data", "world_cup_2026_knockout_matches.csv")
    )
    round_of_32_df = pd.read_csv(
        os.path.join(ROOT_DIR, "reference_data", "world_cup_2026_round_of_32_combinations.csv")
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

    from src.tournament import WorldCup2026

    t0 = WorldCup2026(
        group_matches_df=group_matches_df,
        knockout_matches_df=knockout_matches_df,
        round_of_32_combos=round_of_32_combos,
    )
    t0.simulate(model, random_state=0, record_params=False, fast_mode=True)
    wc_teams = sorted(set(t0.teams))

    name_map = _load_name_map()

    def _normalize_name(raw: str) -> str:
        trimmed = str(raw).strip()
        if not trimmed or trimmed.lower() == "nan":
            return ""
        if _is_placeholder(trimmed):
            return trimmed
        return name_map.get(trimmed, trimmed)

    extra_teams = set()
    groups_df = pd.read_csv(os.path.join(ROOT_DIR, "reference_data", "world_cup_2026_groups.csv"))
    for team in groups_df.get("team", []):
        name = _normalize_name(team)
        if name and not _is_placeholder(name):
            extra_teams.add(name)
    qualifiers_df = pd.read_csv(
        os.path.join(ROOT_DIR, "reference_data", "world_cup_2026_remaining_qualifiers.csv")
    )
    for col in ("home_team", "away_team"):
        for team in qualifiers_df.get(col, []):
            name = _normalize_name(team)
            if name and not _is_placeholder(name):
                extra_teams.add(name)

    return sorted(set(wc_teams).union(extra_teams))


def build_payload(model, teams: Iterable[str]) -> dict:
    teams = sorted(dict.fromkeys(teams))
    team_ids = {team: idx for idx, team in enumerate(teams)}

    def extract_entry(t1: str, t2: str, is_neutral: bool, is_friendly: bool):
        output = model.predict_match(
            t1,
            t2,
            requires_result=True,
            is_neutral=is_neutral,
            importance_class=0 if is_friendly else 1,
        )
        return [
            int(team_ids[t1]),
            int(team_ids[t2]),
            1 if is_neutral else 0,
            1 if is_friendly else 0,
            round_sig(output.get("nu", 0.0)),
            round_sig(output.get("lam_home", 0.0)),
            round_sig(output.get("lam_away", 0.0)),
            round_sig(output.get("p_home", 0.0)),
            round_sig(output.get("p_draw", 0.0)),
            round_sig(output.get("p_away", 0.0)),
            round_sig(output.get("p_home_pens", 0.0)),
            round_sig(output.get("p_away_pens", 0.0)),
        ]

    entries = []
    for team1 in teams:
        for team2 in teams:
            if team1 == team2:
                continue
            entries.append(extract_entry(team1, team2, False, False))
            entries.append(extract_entry(team1, team2, True, False))
            entries.append(extract_entry(team1, team2, False, True))
            entries.append(extract_entry(team1, team2, True, True))

    return {
        "version": 3,
        "max_goals": int(MAX_GOALS),
        "teams": teams,
        "entries": entries,
    }


def write_payload(payload: dict, output_dir: str = "model_output", filename: str = "win_probabilities.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    local_gzip_path = f"{output_path}.gz"
    with open(output_path, "rb") as f_in, gzip.open(local_gzip_path, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.makedirs(PUBLIC_MODEL_OUTPUT_DIR, exist_ok=True)
    public_path = os.path.join(PUBLIC_MODEL_OUTPUT_DIR, filename)
    shutil.copy2(output_path, public_path)
    gzip_path = f"{public_path}.gz"
    with open(output_path, "rb") as f_in, gzip.open(gzip_path, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    return output_path


def main() -> None:
    model_path = os.path.join(ROOT_DIR, "model_output", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    world_cup_path = write_payload(
        build_payload(model, world_cup_subset_teams(model)),
        filename="win_probabilities.json",
    )
    full_path = write_payload(
        build_payload(model, current_teams()),
        filename="win_probabilities_full.json",
    )
    print(f"Wrote World Cup subset win probabilities to {world_cup_path}")
    print(f"Wrote full-universe win probabilities to {full_path}")


if __name__ == "__main__":
    main()
