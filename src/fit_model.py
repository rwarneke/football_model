import os
import sys
import json
import pickle
import shutil
import gzip
import re

import numpy as np
import pandas as pd
from scipy.stats import gamma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PUBLIC_MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, "web", "public", "model_output")

from src.model import Model, MAX_GOALS
from src.tournament import WorldCup2026

FRIENDLY_LOSS_WEIGHT = 0.2


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


def main():
    results = pd.read_csv("match_results/results_clean.csv", parse_dates=["date"])
    results["year"] = results["date"].apply(lambda x: x.year)
    all_teams = pd.read_csv("reference_data/team_universe.csv")
    current_teams = all_teams.query("category != 'past_team'").team.tolist()

    important_tournaments = [
        "FIFA World Cup",
        "FIFA World Cup qualification",
        "Copa América",
        "Gold Cup",
        "UEFA Euro",
        "UEFA Euro qualification",
        "UEFA Nations League",
        "AFC Asian Cup",
        "AFC Asian Cup qualification",
        "African Cup of Nations",
        "African Cup of Nations qualification",
        "British Home Championship",
        "Nordic Championship",
        "Central European International Cup",
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
    total_matches = int(len(results))
    progress_every = max(1, total_matches // 20)

    def _fit_progress(done, total, row):
        if total <= 0:
            return
        pct = (done / total) * 100
        if row is None:
            message = f"[fit] 0/{total} (0.0%) starting"
            print(message, end="\r", flush=True)
            return
        date = pd.Timestamp(row.date).date()
        message = (
            f"[fit] {done}/{total} ({pct:.1f}%) {date} {row.home_team} vs {row.away_team}"
        )
        if done >= total:
            print(message)
        else:
            print(message, end="\r", flush=True)

    model.fit(results.iloc[::], progress_cb=_fit_progress, progress_every=progress_every)

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

    a = 6.0
    b = 2.1
    c = 1.0
    cdf_func = lambda x: 1.0 - gamma.cdf(c - x, a=a, scale=b / a)
    rating_func = lambda x: 100 * cdf_func(x)
    df_state["rating_attack"] = rating_func(df_state["mu_attack"])
    df_state["rating_defense"] = rating_func(df_state["mu_defense"])
    df_state["rating"] = rating_func(df_state["quality"] / 2)

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

    os.makedirs("model_output", exist_ok=True)
    all_ratings.to_csv("model_output/ratings_current.csv")
    df_history.to_csv("model_output/ratings_history.csv")

    ratings_yearly = df_history["rating"].copy()
    ratings_yearly.index = pd.to_datetime(ratings_yearly.index)
    ratings_yearly = ratings_yearly.groupby(ratings_yearly.index.year).last()
    ratings_yearly.index = pd.to_datetime(
        [f"{year}-01-01" for year in ratings_yearly.index]
    )
    ratings_yearly.insert(0, "date", ratings_yearly.index.strftime("%Y-%m-%d"))
    ratings_yearly.reset_index(drop=True).to_csv(
        "model_output/ratings_history_yearly.csv", index=False
    )

    with open("model_output/model.pkl", "wb") as f:
        pickle.dump(model, f)

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
    t0.simulate(model, random_state=0, record_params=False, fast_mode=True)
    wc_teams = sorted(set(t0.teams))

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
    with open("model_output/win_probabilities.json", "rb") as f_in, gzip.open(
        "model_output/win_probabilities.json.gz", "wb", compresslevel=9
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.makedirs(PUBLIC_MODEL_OUTPUT_DIR, exist_ok=True)
    for filename in (
        "ratings_current.csv",
        "ratings_history_yearly.csv",
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
