import os
import sys
import json
import math
import shutil
import gzip
import re
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PUBLIC_MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, "web", "public", "model_output")

from src.model import Model, MAX_GOALS
from src.model_elo import EloModel
from src.tournament import WorldCup2026

FRIENDLY_LOSS_WEIGHT = 0.2
STAGE_ORDER = [
    "0. Qualifying",
    "1. Group",
    "2. Round of 32",
    "3. Round of 16",
    "4. Quarterfinal",
    "5. Fourth place",
    "6. Third place",
    "7. Final",
    "8. Champion",
]

_SIM_MODEL = None
_SIM_GROUP_MATCHES_DF = None
_SIM_KNOCKOUT_MATCHES_DF = None
_SIM_ROUND_OF_32_COMBOS = None
_SIM_WC_TEAMS = None
_SIM_WC_TEAM_SET = None
_SIM_TEAM_INDEX = None
_SIM_STAGE_INDEX = None


def get_results(team, second_team=None, res=None, start_date=None, end_date=None):
    if res is None:
        raise ValueError("res is required")
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
        (res.loss_score * w).sum() / w.sum(),
    )


def _init_worker(model, group_matches_df, knockout_matches_df, round_of_32_combos, wc_teams):
    global _SIM_MODEL
    global _SIM_GROUP_MATCHES_DF
    global _SIM_KNOCKOUT_MATCHES_DF
    global _SIM_ROUND_OF_32_COMBOS
    global _SIM_WC_TEAMS
    global _SIM_WC_TEAM_SET
    global _SIM_TEAM_INDEX
    global _SIM_STAGE_INDEX

    _SIM_MODEL = model
    _SIM_GROUP_MATCHES_DF = group_matches_df
    _SIM_KNOCKOUT_MATCHES_DF = knockout_matches_df
    _SIM_ROUND_OF_32_COMBOS = round_of_32_combos
    _SIM_WC_TEAMS = list(wc_teams)
    _SIM_WC_TEAM_SET = set(wc_teams)
    _SIM_TEAM_INDEX = {team: idx for idx, team in enumerate(_SIM_WC_TEAMS)}
    _SIM_STAGE_INDEX = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}


def _simulate_chunk(seed_start, seed_count):
    stage_counts = np.zeros((len(_SIM_WC_TEAMS), len(STAGE_ORDER)), dtype=int)
    won_group_counts = np.zeros(len(_SIM_WC_TEAMS), dtype=int)

    for seed in range(seed_start, seed_start + seed_count):
        t = WorldCup2026(
            group_matches_df=_SIM_GROUP_MATCHES_DF,
            knockout_matches_df=_SIM_KNOCKOUT_MATCHES_DF,
            round_of_32_combos=_SIM_ROUND_OF_32_COMBOS,
        )
        t.simulate(_SIM_MODEL, random_state=seed, record_params=False, fast_mode=True)
        elim = t.stage_of_elimination()
        for team, stage in elim.items():
            if team in _SIM_WC_TEAM_SET:
                stage_idx = _SIM_STAGE_INDEX.get(stage)
                if stage_idx is not None:
                    stage_counts[_SIM_TEAM_INDEX[team], stage_idx] += 1
        won_group = t.won_group_stage()
        for team, won in won_group.items():
            if team in _SIM_WC_TEAM_SET and won:
                won_group_counts[_SIM_TEAM_INDEX[team]] += 1

    return stage_counts, won_group_counts, seed_count


def main():
    ## Read in data ##

    results_raw = pd.read_csv("match_results/results.csv", parse_dates=["date"])
    results = pd.read_csv("match_results/results_clean.csv", parse_dates=["date"])
    results["year"] = results["date"].apply(lambda x: x.year)
    confederations = pd.read_csv("reference_data/confederations.csv")
    first_confederation = confederations.query("start_year.isna()").set_index(
        "team"
    ).confederation
    all_teams = pd.read_csv("reference_data/team_universe.csv")
    current_fifa_members = all_teams.query("category == 'fifa_member'").team.tolist()
    other_teams = all_teams.query("category != 'fifa_member'").team.tolist()
    current_teams = all_teams.query("category != 'past_team'").team.tolist()

    important_tournaments = [
        "FIFA World Cup",
        "FIFA World Cup qualification",
        "Copa América",  # South America
        "Gold Cup",  # Noth America
        "UEFA Euro",  # Europe
        "UEFA Euro qualification",
        "UEFA Nations League",  # Europe 2
        "AFC Asian Cup",  # Asia"
        "AFC Asian Cup qualification",
        "African Cup of Nations",  # Africa
        "African Cup of Nations qualification",
        # defunct but once important
        "British Home Championship",
        "Nordic Championship",
        "Central European International Cup",
    ]
    big_teams = [
        "Spain",
        "Argentina",
        "Brazil",
        "Colombia",
        "England",
        "Portugal",
        "France",
        "Netherlands",
        "Germany",
        "Norway",
        "Belgium",
        "Switzerland",
        "Croatia",
        "Denmark",
        "Ecuador",
        "Uruguay",
        "Japan",
        "Italy",
        "Senegal",
        "Morocco",
        "Austria",
        "Canada",
        "Greece",
        "Turkey",
        "Mexico",
        "Chile",
        "Russia",
        "Paraguay",
        "Serbia",
        "Ukraine",
        "South Korea",
        "Australia",
        "USA",
        "Sweden",
        "Iran",
        "Poland",
        "Algeria",
        "Venezuela",
        "Scotland",
        "Czechia",
    ]
    results["important"] = results.tournament.isin(important_tournaments)
    results["importance_class"] = 1
    results["importance_class"] = np.where(
        results["important"], 2, results["importance_class"]
    )
    results["importance_class"] = np.where(
        results["tournament"] == "Friendly", 0, results["importance_class"]
    )

    model = Model()
    res = model.fit(results.iloc[::])

    ## Export metrics and define qualities ##

    df_state = model.export_state_df()
    df_mu = model.export_mu_df()
    df_hga = model.export_hga_df()
    df_state["quality"] = df_state["mu_attack"] + df_state["mu_defense"]
    df_state["quality_low"] = df_state["quality"] - 2 * np.sqrt(
        df_state["sigma_attack"]
        + df_state["sigma_defense"]
        + 2 * df_state["sigma_ad"]
    )
    df_state["mu_attack_low"] = df_state["mu_attack"] - 2 * np.sqrt(
        df_state["sigma_attack"]
    )
    df_state["mu_defense_low"] = df_state["mu_defense"] - 2 * np.sqrt(
        df_state["sigma_defense"]
    )
    df_state["year"] = df_state.date.apply(lambda x: x.year)

    ## convert from params to ratings ##

    a = 6.0  # larger = more teams get to be high 90s
    b = 2.1  # larger = teams drawn away from zero
    c = 1.0  # anything above this will be treated identically, so teams should not get near it
    cdf_func = lambda x: 1.0 - gamma.cdf(c - x, a=a, scale=b / a)
    rating_func = lambda x: 100 * cdf_func(x)
    df_state["rating_attack"] = rating_func(df_state["mu_attack"])
    df_state["rating_defense"] = rating_func(df_state["mu_defense"])
    df_state["rating"] = rating_func(df_state["quality"] / 2)

    ## produce team histories, ratings, and rankings ##

    df_history = df_state.set_index(["date", "team"]).unstack().ffill()

    for team in all_teams.query("category == 'past_team'").team:
        last_date = get_results(team, res=results).date.max().date()
        df_history.loc[
            df_history.index > last_date,
            [c for c in df_history.columns if c[1] == team],
        ] = np.nan

    current_rating = (
        df_history.rating[current_teams].iloc[-1].sort_values(ascending=False).round(2)
    )
    current_rating.name = "rating"
    elo_ratings = pd.read_csv("reference_data/alt_rankings/elo_ratings_20260104.csv")
    fifa_ratings = pd.read_csv("reference_data/alt_rankings/fifa_rankings_20251218.csv")

    # all_ratings = df_history[["rating", "rating_attack", "rating_defense", "quality", "mu_attack", "mu_defense"]].iloc[-1].unstack().T
    all_ratings = df_history.iloc[-1].unstack().T
    all_rankings = current_rating.to_frame().reset_index().rename(columns={"rating": "model"})
    all_rankings = all_rankings.merge(elo_ratings, on="team")
    all_rankings = all_rankings.merge(
        fifa_ratings[["team", "fifa_points"]].rename(columns={"fifa_points": "fifa"}),
        on="team",
    )
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
        all_rankings[f"rank_diff_{attr}"] = all_rankings["rank_model"] - all_rankings[
            f"rank_{attr}"
        ]

    all_ratings.to_csv("model_output/ratings_current.csv")
    df_history.to_csv("model_output/ratings_history.csv")

    ## Simulations ##

    N = int(os.environ.get("SIM_N", "1000"))
    workers = int(os.environ.get("SIM_WORKERS", "8"))
    chunk_size = int(os.environ.get("SIM_CHUNK_SIZE", str(math.ceil(N / workers))))

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

    t0 = WorldCup2026(
        group_matches_df=group_matches_df,
        knockout_matches_df=knockout_matches_df,
        round_of_32_combos=round_of_32_combos,
    )
    t0.simulate(model, record_params=False)
    wc_teams = sorted(
        set(
            t0.results_frame().home_team.tolist()
            + t0.results_frame().away_team.tolist()
        )
    )

    stage_counts = np.zeros((len(wc_teams), len(STAGE_ORDER)), dtype=int)
    won_group_counts = np.zeros(len(wc_teams), dtype=int)

    tasks = []
    for seed_start in range(0, N, chunk_size):
        seed_count = min(chunk_size, N - seed_start)
        tasks.append((seed_start, seed_count))

    completed = 0
    sim_start = dt.datetime.now()
    if workers <= 1:
        _init_worker(
            model,
            group_matches_df,
            knockout_matches_df,
            round_of_32_combos,
            wc_teams,
        )
        for start, count in tasks:
            chunk_stage_counts, chunk_won_group_counts, chunk_done = _simulate_chunk(
                start, count
            )
            stage_counts += chunk_stage_counts
            won_group_counts += chunk_won_group_counts
            completed += chunk_done
            print(f"\r{completed:7} / {N}", end="")
        print()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                model,
                group_matches_df,
                knockout_matches_df,
                round_of_32_combos,
                wc_teams,
            ),
        ) as executor:
            futures = [
                executor.submit(_simulate_chunk, start, count) for start, count in tasks
            ]
            for future in as_completed(futures):
                chunk_stage_counts, chunk_won_group_counts, chunk_done = future.result()
                stage_counts += chunk_stage_counts
                won_group_counts += chunk_won_group_counts
                completed += chunk_done
                print(f"\r{completed:7} / {N}", end="")
        print()
    sim_elapsed = (dt.datetime.now() - sim_start).total_seconds()
    print(f"Simulation time (workers={workers}, chunk={chunk_size}): {sim_elapsed:.2f}s")

    stage_counts = pd.DataFrame(stage_counts, index=wc_teams, columns=STAGE_ORDER, dtype=int)
    won_group_counts = pd.Series(won_group_counts, index=wc_teams, dtype=int)

    ratings_with_wins = (
        all_ratings.loc[wc_teams].sort_values("quality", ascending=False).join(stage_counts)
    )
    wongroup = won_group_counts.infer_objects().fillna(0).astype(int)
    wongroup.name = "Won group"
    ratings_with_wins = ratings_with_wins.join(wongroup.to_frame())

    ratings_with_wins["odds_champion"] = (
        N / ratings_with_wins["8. Champion"]
    ).round(2)
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
        - stage_probs_output[["0. Qualifying", "1. Group", "2. Round of 32"]].sum(
            axis=1
        )
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

    def _load_name_map():
        mapping = {}
        for filename in (
            "reference_data/fifa_member_to_canonical_name_map.csv",
            "reference_data/kaggle_team_to_canonical_name_map.csv",
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

    name_map = _load_name_map()

    def _is_placeholder(name: str) -> bool:
        return bool(re.search(r"\bwinner$", name, re.IGNORECASE))

    def _normalize_name(raw: str) -> str:
        trimmed = str(raw).strip()
        if not trimmed or trimmed.lower() == "nan":
            return ""
        if _is_placeholder(trimmed):
            return trimmed
        return name_map.get(trimmed, trimmed)

    extra_teams = set()
    groups_df = pd.read_csv("reference_data/world_cup_2026_groups.csv")
    for team in groups_df.get("team", []):
        name = _normalize_name(team)
        if name and not _is_placeholder(name):
            extra_teams.add(name)
    qualifiers_df = pd.read_csv("reference_data/world_cup_2026_remaining_qualifiers.csv")
    for col in ("home_team", "away_team"):
        for team in qualifiers_df.get(col, []):
            name = _normalize_name(team)
            if name and not _is_placeholder(name):
                extra_teams.add(name)

    teams = sorted(wc_teams)
    if extra_teams:
        teams = sorted(set(teams).union(extra_teams))
    team_ids = {team: idx for idx, team in enumerate(teams)}

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

    def extract_entry(t1, t2, is_neutral):
        output = model.predict_match(t1, t2, requires_result=True, is_neutral=is_neutral)
        return [
            int(team_ids[t1]),
            int(team_ids[t2]),
            1 if is_neutral else 0,
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
            entries.append(extract_entry(team1, team2, False))
            entries.append(extract_entry(team1, team2, True))

    payload = {
        "version": 2,
        "max_goals": int(MAX_GOALS),
        "teams": teams,
        "entries": entries,
    }

    with open("model_output/win_probabilities.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    os.makedirs(PUBLIC_MODEL_OUTPUT_DIR, exist_ok=True)
    for filename in (
        "ratings_current.csv",
        "ratings_history_yearly.csv",
        "simulation_results.csv",
        "win_probabilities.json",
    ):
        source = os.path.join("model_output", filename)
        if os.path.exists(source):
            shutil.copy2(source, PUBLIC_MODEL_OUTPUT_DIR)
            if filename == "win_probabilities.json":
                gzip_path = os.path.join(PUBLIC_MODEL_OUTPUT_DIR, f"{filename}.gz")
                with open(source, "rb") as f_in, gzip.open(gzip_path, "wb", compresslevel=9) as f_out:
                    shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    main()
