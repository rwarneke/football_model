from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DATA_DIR = ROOT_DIR / "reference_data"
MATCH_RESULTS_DIR = ROOT_DIR / "match_results"
PUBLIC_MODEL_OUTPUT_DIR = ROOT_DIR / "web" / "public" / "model_output"

GROUP_MATCHES_PATH = REFERENCE_DATA_DIR / "world_cup_2026_group_matches.csv"
KNOCKOUT_MATCHES_PATH = REFERENCE_DATA_DIR / "world_cup_2026_knockout_matches.csv"
RESULTS_WC2026_PATH = MATCH_RESULTS_DIR / "results_wc2026.csv"

HOST_TEAM_COUNTRIES = {
    "USA": "USA",
    "United States": "USA",
    "Canada": "Canada",
    "Mexico": "Mexico",
}

RESULTS_WC2026_REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "stage",
    "group",
    "home_team",
    "away_team",
    "stadium",
    "city",
    "country",
    "tournament",
    "neutral",
    "home_score",
    "away_score",
    "home_score_90",
    "away_score_90",
    "home_score_120",
    "away_score_120",
    "went_extra_time",
    "went_penalties",
    "penalty_winner",
    "first_shooter",
]


def build_world_cup_2026_schedule() -> pd.DataFrame:
    group_matches = pd.read_csv(GROUP_MATCHES_PATH, parse_dates=["date"]).copy()
    group_matches["stage"] = "Group"
    knockout_matches = pd.read_csv(KNOCKOUT_MATCHES_PATH, parse_dates=["date"]).copy()
    knockout_matches = knockout_matches.rename(
        columns={"home": "home_team", "away": "away_team"}
    )
    knockout_matches["group"] = pd.NA

    schedule = pd.concat(
        [
            group_matches[
                [
                    "match_id",
                    "date",
                    "stage",
                    "group",
                    "home_team",
                    "away_team",
                    "stadium",
                    "city",
                    "country",
                ]
            ],
            knockout_matches[
                [
                    "match_id",
                    "date",
                    "stage",
                    "group",
                    "home_team",
                    "away_team",
                    "stadium",
                    "city",
                    "country",
                ]
            ],
        ],
        ignore_index=True,
    )
    schedule["tournament"] = "FIFA World Cup"
    return schedule.sort_values("match_id").reset_index(drop=True)


def _normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _is_knockout_placeholder(label: str) -> bool:
    cleaned = _normalize_string(label)
    if not cleaned:
        return True
    prefixes = ("Winner ", "Runner-up ", "Loser ", "3rd ")
    return cleaned.startswith(prefixes) or cleaned.lower().endswith(" winner")


def infer_world_cup_neutral(home_team: str, away_team: str, country: str) -> bool:
    home_country = _normalize_string(HOST_TEAM_COUNTRIES.get(home_team, ""))
    away_country = _normalize_string(HOST_TEAM_COUNTRIES.get(away_team, ""))
    match_country = _normalize_string(country)
    home_advantage = bool(home_country and match_country and home_country.casefold() == match_country.casefold())
    away_advantage = bool(away_country and match_country and away_country.casefold() == match_country.casefold())
    return not (home_advantage ^ away_advantage)


def load_results_wc2026() -> pd.DataFrame:
    if not RESULTS_WC2026_PATH.exists():
        raise FileNotFoundError(f"Missing World Cup results file: {RESULTS_WC2026_PATH}")

    schedule = build_world_cup_2026_schedule()
    results = pd.read_csv(
        RESULTS_WC2026_PATH,
        parse_dates=["date"],
        keep_default_na=True,
        na_values=["NA"],
    )

    missing = [col for col in RESULTS_WC2026_REQUIRED_COLUMNS if col not in results.columns]
    if missing:
        raise ValueError(
            "results_wc2026.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    results = results[RESULTS_WC2026_REQUIRED_COLUMNS].copy()
    if results["match_id"].isna().any():
        raise ValueError("results_wc2026.csv contains rows with missing match_id values.")
    results["match_id"] = pd.to_numeric(results["match_id"], errors="raise").astype(int)

    if results["match_id"].duplicated().any():
        dupes = results.loc[results["match_id"].duplicated(), "match_id"].tolist()
        raise ValueError(
            "results_wc2026.csv contains duplicate match_id values: "
            f"{sorted(set(dupes))}"
        )

    merged = schedule.merge(
        results,
        on="match_id",
        how="left",
        suffixes=("_schedule", ""),
        validate="1:1",
    )
    if merged[RESULTS_WC2026_REQUIRED_COLUMNS].isna().all(axis=1).any():
        missing_rows = merged.loc[
            merged[RESULTS_WC2026_REQUIRED_COLUMNS].isna().all(axis=1),
            "match_id",
        ].tolist()
        raise ValueError(
            "results_wc2026.csv is missing one or more scheduled matches: "
            f"{missing_rows}"
        )

    for col in ("stage", "stadium", "city", "country", "tournament"):
        expected = merged[f"{col}_schedule"].map(_normalize_string)
        provided = merged[col].map(_normalize_string)
        mismatch = provided != expected
        if mismatch.any():
            sample = merged.loc[
                mismatch,
                ["match_id", f"{col}_schedule", col],
            ].head(10)
            raise ValueError(
                f"results_wc2026.csv has unexpected values in column {col}.\n"
                f"{sample.to_string(index=False)}"
            )

    expected_groups = merged["group_schedule"].map(_normalize_string)
    provided_groups = merged["group"].map(_normalize_string)
    group_mismatch = expected_groups != provided_groups
    if group_mismatch.any():
        sample = merged.loc[
            group_mismatch,
            ["match_id", "group_schedule", "group"],
        ].head(10)
        raise ValueError(
            "results_wc2026.csv has unexpected values in column group.\n"
            f"{sample.to_string(index=False)}"
        )

    group_mask = merged["stage_schedule"].eq("Group")
    group_team_mismatch = (
        group_mask
        & (
            merged["home_team"].map(_normalize_string)
            != merged["home_team_schedule"].map(_normalize_string)
        )
    ) | (
        group_mask
        & (
            merged["away_team"].map(_normalize_string)
            != merged["away_team_schedule"].map(_normalize_string)
        )
    )
    if group_team_mismatch.any():
        sample = merged.loc[
            group_team_mismatch,
            ["match_id", "home_team_schedule", "home_team", "away_team_schedule", "away_team"],
        ].head(10)
        raise ValueError(
            "Group-stage rows in results_wc2026.csv must match the published schedule.\n"
            f"{sample.to_string(index=False)}"
        )

    for col in (
        "home_score",
        "away_score",
        "home_score_90",
        "away_score_90",
        "home_score_120",
        "away_score_120",
    ):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    merged["neutral"] = merged["neutral"].map(_normalize_bool)
    merged["went_extra_time"] = merged["went_extra_time"].map(_normalize_bool)
    merged["went_penalties"] = merged["went_penalties"].map(_normalize_bool)

    for col in ("home_team", "away_team", "penalty_winner", "first_shooter"):
        merged[col] = merged[col].map(_normalize_string)
        merged.loc[merged[col] == "", col] = pd.NA

    score_known = merged["home_score"].notna() | merged["away_score"].notna()
    partial_scores = merged["home_score"].notna() ^ merged["away_score"].notna()
    if partial_scores.any():
        sample = merged.loc[
            partial_scores,
            ["match_id", "home_score", "away_score"],
        ].head(10)
        raise ValueError(
            "results_wc2026.csv contains partial scorelines.\n"
            f"{sample.to_string(index=False)}"
        )

    merged["completed"] = score_known

    completed = merged["completed"]
    merged.loc[completed & merged["neutral"].isna(), "neutral"] = merged.loc[
        completed & merged["neutral"].isna()
    ].apply(
        lambda row: infer_world_cup_neutral(
            _normalize_string(row["home_team"]),
            _normalize_string(row["away_team"]),
            _normalize_string(row["country"]),
        ),
        axis=1,
    )
    merged.loc[completed & merged["went_extra_time"].isna(), "went_extra_time"] = False
    merged.loc[completed & merged["went_penalties"].isna(), "went_penalties"] = False

    completed_knockout = completed & ~merged["stage_schedule"].eq("Group")
    unresolved_knockout_teams = completed_knockout & (
        merged["home_team"].map(lambda value: _is_knockout_placeholder(_normalize_string(value)))
        | merged["away_team"].map(lambda value: _is_knockout_placeholder(_normalize_string(value)))
    )
    if unresolved_knockout_teams.any():
        sample = merged.loc[
            unresolved_knockout_teams,
            ["match_id", "home_team", "away_team"],
        ].head(10)
        raise ValueError(
            "Completed knockout rows in results_wc2026.csv must contain concrete team names.\n"
            f"{sample.to_string(index=False)}"
        )

    regular_time_missing = completed & (
        merged["home_score_90"].isna() | merged["away_score_90"].isna()
    )
    if regular_time_missing.any():
        sample = merged.loc[
            regular_time_missing,
            ["match_id", "home_score_90", "away_score_90"],
        ].head(10)
        raise ValueError(
            "Completed rows in results_wc2026.csv must include home_score_90 and away_score_90.\n"
            f"{sample.to_string(index=False)}"
        )

    group_extra_time = completed & merged["stage_schedule"].eq("Group") & (
        merged["went_extra_time"].fillna(False)
        | merged["went_penalties"].fillna(False)
    )
    if group_extra_time.any():
        sample = merged.loc[
            group_extra_time,
            ["match_id", "went_extra_time", "went_penalties"],
        ].head(10)
        raise ValueError(
            "Group-stage rows in results_wc2026.csv cannot have extra time or penalties.\n"
            f"{sample.to_string(index=False)}"
        )

    group_bad_final = completed & merged["stage_schedule"].eq("Group") & (
        (merged["home_score"] != merged["home_score_90"])
        | (merged["away_score"] != merged["away_score_90"])
        | merged["home_score_120"].notna()
        | merged["away_score_120"].notna()
    )
    if group_bad_final.any():
        sample = merged.loc[
            group_bad_final,
            [
                "match_id",
                "home_score",
                "away_score",
                "home_score_90",
                "away_score_90",
                "home_score_120",
                "away_score_120",
            ],
        ].head(10)
        raise ValueError(
            "Group-stage rows in results_wc2026.csv must resolve entirely at 90 minutes.\n"
            f"{sample.to_string(index=False)}"
        )

    extra_time_without_scores = completed & merged["went_extra_time"].fillna(False) & (
        merged["home_score_120"].isna() | merged["away_score_120"].isna()
    )
    if extra_time_without_scores.any():
        sample = merged.loc[
            extra_time_without_scores,
            ["match_id", "home_score_120", "away_score_120"],
        ].head(10)
        raise ValueError(
            "Extra-time rows in results_wc2026.csv must include home_score_120 and away_score_120.\n"
            f"{sample.to_string(index=False)}"
        )

    knockout_without_extra_time_breakdown = completed_knockout & ~merged["went_extra_time"].fillna(False) & (
        (merged["home_score"] != merged["home_score_90"])
        | (merged["away_score"] != merged["away_score_90"])
        | merged["home_score_120"].notna()
        | merged["away_score_120"].notna()
    )
    if knockout_without_extra_time_breakdown.any():
        sample = merged.loc[
            knockout_without_extra_time_breakdown,
            [
                "match_id",
                "home_score",
                "away_score",
                "home_score_90",
                "away_score_90",
                "home_score_120",
                "away_score_120",
            ],
        ].head(10)
        raise ValueError(
            "Knockout rows without extra time must resolve at 90 minutes only.\n"
            f"{sample.to_string(index=False)}"
        )

    extra_time_final_mismatch = completed & merged["went_extra_time"].fillna(False) & (
        (merged["home_score"] != merged["home_score_120"])
        | (merged["away_score"] != merged["away_score_120"])
    )
    if extra_time_final_mismatch.any():
        sample = merged.loc[
            extra_time_final_mismatch,
            [
                "match_id",
                "home_score",
                "away_score",
                "home_score_120",
                "away_score_120",
            ],
        ].head(10)
        raise ValueError(
            "Extra-time rows in results_wc2026.csv must have final scores equal to 120-minute scores.\n"
            f"{sample.to_string(index=False)}"
        )

    pens_mask = completed & merged["went_penalties"].fillna(False)
    if (pens_mask & ~merged["went_extra_time"].fillna(False)).any():
        sample = merged.loc[
            pens_mask & ~merged["went_extra_time"].fillna(False),
            ["match_id", "went_extra_time", "went_penalties"],
        ].head(10)
        raise ValueError(
            "Penalty shootout rows in results_wc2026.csv must also have went_extra_time=true.\n"
            f"{sample.to_string(index=False)}"
        )

    if (pens_mask & (merged["home_score"] != merged["away_score"])).any():
        sample = merged.loc[
            pens_mask & (merged["home_score"] != merged["away_score"]),
            ["match_id", "home_score", "away_score"],
        ].head(10)
        raise ValueError(
            "Penalty shootout rows in results_wc2026.csv must be level after 120 minutes.\n"
            f"{sample.to_string(index=False)}"
        )

    bad_penalty_winner = pens_mask & ~merged.apply(
        lambda row: row["penalty_winner"] in {row["home_team"], row["away_team"]},
        axis=1,
    )
    if bad_penalty_winner.any():
        sample = merged.loc[
            bad_penalty_winner,
            ["match_id", "home_team", "away_team", "penalty_winner"],
        ].head(10)
        raise ValueError(
            "Penalty shootout rows in results_wc2026.csv must name the winning team in penalty_winner.\n"
            f"{sample.to_string(index=False)}"
        )

    unexpected_penalty_winner = completed & ~merged["went_penalties"].fillna(False) & merged["penalty_winner"].notna()
    if unexpected_penalty_winner.any():
        sample = merged.loc[
            unexpected_penalty_winner,
            ["match_id", "penalty_winner"],
        ].head(10)
        raise ValueError(
            "results_wc2026.csv cannot set penalty_winner unless went_penalties=true.\n"
            f"{sample.to_string(index=False)}"
        )

    drawn_knockouts_without_pens = completed_knockout & (merged["home_score"] == merged["away_score"]) & ~merged["went_penalties"].fillna(False)
    if drawn_knockouts_without_pens.any():
        sample = merged.loc[
            drawn_knockouts_without_pens,
            ["match_id", "home_score", "away_score", "went_penalties"],
        ].head(10)
        raise ValueError(
            "Completed knockout rows in results_wc2026.csv cannot finish level without penalties.\n"
            f"{sample.to_string(index=False)}"
        )

    winner = pd.Series(pd.NA, index=merged.index, dtype="object")
    home_win = completed & (merged["home_score"] > merged["away_score"])
    away_win = completed & (merged["away_score"] > merged["home_score"])
    winner.loc[home_win] = merged.loc[home_win, "home_team"]
    winner.loc[away_win] = merged.loc[away_win, "away_team"]
    winner.loc[pens_mask] = merged.loc[pens_mask, "penalty_winner"]
    merged["winner"] = winner

    return merged[
        [
            "match_id",
            "date",
            "stage_schedule",
            "group_schedule",
            "home_team",
            "away_team",
            "stadium_schedule",
            "city_schedule",
            "country_schedule",
            "tournament_schedule",
            "neutral",
            "home_score",
            "away_score",
            "home_score_90",
            "away_score_90",
            "home_score_120",
            "away_score_120",
            "went_extra_time",
            "went_penalties",
            "penalty_winner",
            "first_shooter",
            "winner",
            "completed",
        ]
    ].rename(
        columns={
            "stage_schedule": "stage",
            "group_schedule": "group",
            "stadium_schedule": "stadium",
            "city_schedule": "city",
            "country_schedule": "country",
            "tournament_schedule": "tournament",
        }
    )


def world_cup_results_for_clean_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = load_results_wc2026()
    completed = results.loc[results["completed"]].copy()
    match_rows = completed[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
            "home_score_90",
            "away_score_90",
            "home_score_120",
            "away_score_120",
            "went_extra_time",
            "went_penalties",
        ]
    ].copy()
    shootouts = completed.loc[
        completed["went_penalties"],
        ["date", "home_team", "away_team", "penalty_winner", "first_shooter"],
    ].rename(
        columns={
            "penalty_winner": "winner",
            "first_shooter": "first_shooter",
        }
    )
    return match_rows, shootouts, completed


def copy_results_wc2026_to_public(public_output_dir: Path | None = None) -> Path:
    output_dir = public_output_dir or PUBLIC_MODEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / RESULTS_WC2026_PATH.name
    shutil.copy2(RESULTS_WC2026_PATH, destination)
    return destination
