### Run this from the repo root or anywhere. ###

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Clean match results data.")
parser.add_argument(
    "--no-goalscorers",
    action="store_true",
    help="Skip joining goalscorers.csv for score breakdowns.",
)
ARGS, _ = parser.parse_known_args()

ROOT_DIR = Path(__file__).resolve().parents[2]
MATCH_RESULTS_DIR = ROOT_DIR / "match_results"
SCORE_RECONCILED_MIN_SHARE = 0.5

results_raw = pd.read_csv(MATCH_RESULTS_DIR / "results.csv", parse_dates=["date"])
shootouts_raw = pd.read_csv(MATCH_RESULTS_DIR / "shootouts.csv", parse_dates=["date"])

results_merged = pd.merge(
    results_raw,
    shootouts_raw.rename(columns={"winner": "shootout_winner", "first_shooter": "shootout_first_shooter"}),
    on=["date", "home_team", "away_team"],
    how="left"
)

renamer = {
    row.original_name: row.replacement_name
    for row in pd.read_csv(ROOT_DIR / "reference_data/kaggle_team_to_canonical_name_map.csv").itertuples()
}

results_merged["home_team"] = results_merged["home_team"].apply(lambda t: renamer.get(t, t))
results_merged["away_team"] = results_merged["away_team"].apply(lambda t: renamer.get(t, t))
results_merged["shootout_winner"] = results_merged["shootout_winner"].apply(lambda t: renamer.get(t, t))
results_merged["shootout_first_shooter"] = results_merged["shootout_first_shooter"].apply(lambda t: renamer.get(t, t))

team_universe = pd.read_csv(ROOT_DIR / "reference_data/team_universe.csv")["team"]
fil = (results_merged.home_team.isin(team_universe)) & (results_merged.away_team.isin(team_universe))
results = results_merged.loc[fil]

# apply any hardcoded corrections
corrections_path = MATCH_RESULTS_DIR / "results_corrections.csv"
if corrections_path.exists():
    corrections = pd.read_csv(corrections_path, keep_default_na=False)
    corrections = corrections.replace({"": np.nan})
    corrections["date"] = pd.to_datetime(corrections["date"], errors="coerce")

    def parse_nullable_bool(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in {"true", "t", "1", "yes"}:
            return True
        if s in {"false", "f", "0", "no"}:
            return False
        return np.nan

    for col in ["home_score", "away_score"]:
        corrections[col] = pd.to_numeric(corrections[col], errors="coerce")
    corrections["neutral"] = corrections["neutral"].apply(parse_nullable_bool)

    results = results.merge(
        corrections,
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_corr"),
    )
    for col in ["home_score", "away_score", "neutral", "tournament"]:
        corr_col = f"{col}_corr"
        if corr_col in results.columns:
            results[col] = results[corr_col].where(results[corr_col].notna(), results[col])
            results = results.drop(columns=[corr_col])

    results["home_score"] = results["home_score"].astype("Int64")
    results["away_score"] = results["away_score"].astype("Int64")

score_missing = results["home_score"].isna() ^ results["away_score"].isna()
if score_missing.any():
    results.loc[score_missing, ["home_score", "away_score"]] = pd.NA


# add confederations (date-aware membership)
conf = pd.read_csv(ROOT_DIR / "reference_data/confederations.csv")

# Expect columns: team,confederation,start_year,end_year
# Make sure types are sane
conf = conf.rename(columns={
    "team": "team",
    "confederation": "confederation",
    "start_year": "start_year",
    "end_year": "end_year",
})

conf["team"] = conf["team"].astype(str).str.strip()
conf["confederation"] = conf["confederation"].fillna("NO_CONFEDERATION").astype(str).str.strip()

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

# add closing odds (where available)
odds_path = ROOT_DIR / "reference_data/closing_odds.csv"
if odds_path.exists():
    ODDS_TO_CANONICAL_TEAM_MAP = {
        "Salvador": "El Salvador",
        "Czech Republic": "Czechia",
        "Macedonia": "North Macedonia",
        "Bosnia &amp; Herzego": "Bosnia and Herzegovina",
        "Trinidad &amp; Tobag": "Trinidad and Tobago",
        "Chinese Taipei": "Taiwan",
        "D.R. Congo": "DR Congo",
    }

    odds = pd.read_csv(odds_path, parse_dates=["match_date"])
    odds.home_team = odds.home_team.apply(lambda x: ODDS_TO_CANONICAL_TEAM_MAP.get(x, x))
    odds.away_team = odds.away_team.apply(lambda x: ODDS_TO_CANONICAL_TEAM_MAP.get(x, x))
    odds = odds.rename(columns={
        "match_date": "date",
        "avg_odds_home_win": "odds_home",
        "avg_odds_draw": "odds_draw",
        "avg_odds_away_win": "odds_away",
    })

    odds_key = ["date", "home_team", "away_team"]
    dup_mask = odds.duplicated(odds_key, keep=False)
    if dup_mask.any():
        dup_rows = odds.loc[
            dup_mask,
            odds_key + ["home_score", "away_score", "odds_home", "odds_draw", "odds_away"],
        ]
        dup_keys = dup_rows[odds_key].drop_duplicates()
        print(f"[odds] {len(dup_keys)} duplicate key(s) found in odds data; averaging odds.")
        odds = (
            odds.groupby(odds_key, as_index=False)
            .agg(
                home_score=("home_score", "first"),
                away_score=("away_score", "first"),
                odds_home=("odds_home", "mean"),
                odds_draw=("odds_draw", "mean"),
                odds_away=("odds_away", "mean"),
            )
        )

    odds = odds.reset_index(drop=True)
    odds["odds_id"] = odds.index

    recognized_odds = odds.loc[
        odds.home_team.isin(team_universe)
        & odds.away_team.isin(team_universe)
    ].copy()

    def make_candidates(df: pd.DataFrame, date_offset_days: int, swapped: bool) -> pd.DataFrame:
        out = df.copy()
        out["source_date"] = out["date"]
        if date_offset_days:
            out["date"] = out["date"] + pd.to_timedelta(date_offset_days, unit="D")
        out["date_diff"] = date_offset_days
        out["swapped"] = swapped
        if swapped:
            temp = out["home_team"].copy()
            out["home_team"] = out["away_team"]
            out["away_team"] = temp
            temp = out["home_score"].copy()
            out["home_score"] = out["away_score"]
            out["away_score"] = temp
            temp = out["odds_home"].copy()
            out["odds_home"] = out["odds_away"]
            out["odds_away"] = temp
        return out

    candidates = pd.concat(
        [
            make_candidates(recognized_odds, 0, False),
            make_candidates(recognized_odds, -1, False),
            make_candidates(recognized_odds, 1, False),
            make_candidates(recognized_odds, 0, True),
            make_candidates(recognized_odds, -1, True),
            make_candidates(recognized_odds, 1, True),
        ],
        ignore_index=True,
    )
    candidates["abs_date_diff"] = candidates["date_diff"].abs()

    results_scores = results[["date", "home_team", "away_team", "home_score", "away_score"]]
    candidates_vs_results = candidates.merge(
        results_scores,
        on=["date", "home_team", "away_team"],
        how="left",
        indicator=True,
        suffixes=("_odds", "_results"),
    )

    matched = candidates_vs_results.loc[candidates_vs_results["_merge"] == "both"].copy()
    best_match_by_odds = matched.sort_values(
        by=["odds_id", "abs_date_diff", "swapped", "source_date"],
        kind="mergesort",
    ).drop_duplicates(subset=["odds_id"], keep="first")

    missing_results = recognized_odds.loc[~recognized_odds["odds_id"].isin(best_match_by_odds["odds_id"])]
    if not missing_results.empty:
        print(
            "[odds] "
            f"{len(missing_results)} match(es) have recognized teams but no matching result."
        )

    both_scores_present = (
        best_match_by_odds["home_score_odds"].notna()
        & best_match_by_odds["home_score_results"].notna()
        & best_match_by_odds["away_score_odds"].notna()
        & best_match_by_odds["away_score_results"].notna()
    )
    score_mismatch = best_match_by_odds.loc[
        both_scores_present
        & (
            (best_match_by_odds["home_score_odds"] != best_match_by_odds["home_score_results"])
            | (best_match_by_odds["away_score_odds"] != best_match_by_odds["away_score_results"])
        )
    ]
    if not score_mismatch.empty:
        print(
            "[odds] "
            f"{len(score_mismatch)} match(es) have odds/results score mismatches."
        )

    dup_candidate_keys = candidates.duplicated(["date", "home_team", "away_team"], keep=False)
    if dup_candidate_keys.any():
        dup_count = candidates.loc[
            dup_candidate_keys, ["date", "home_team", "away_team"]
        ].drop_duplicates().shape[0]
        print(
            "[odds] "
            f"{dup_count} fuzzy match key(s) had multiple candidate odds rows; using best match."
        )

    best_candidates = candidates.sort_values(
        by=["date", "home_team", "away_team", "abs_date_diff", "swapped", "source_date"],
        kind="mergesort",
    ).drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

    odds_for_merge = best_candidates[
        ["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]
    ]
    results = results.merge(
        odds_for_merge,
        on=["date", "home_team", "away_team"],
        how="left",
        validate="m:1",
    )
else:
    results[["odds_home", "odds_draw", "odds_away"]] = float("nan")

# add goalscorer-based score breakdowns (optional)
goalscorers_path = MATCH_RESULTS_DIR / "goalscorers.csv"
if not ARGS.no_goalscorers and goalscorers_path.exists():
    goalscorers = pd.read_csv(goalscorers_path, parse_dates=["date"])
    for col in ["home_team", "away_team", "team"]:
        goalscorers[col] = goalscorers[col].apply(lambda t: renamer.get(t, t))

    def parse_minute_base(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if "+" in s:
            s = s.split("+", 1)[0]
        try:
            return int(float(s))
        except ValueError:
            return np.nan

    goalscorers["minute_raw"] = goalscorers["minute"].astype(str).str.strip()
    goalscorers["minute_is_na"] = goalscorers["minute"].isna() | goalscorers[
        "minute_raw"
    ].isin(["NA", ""])
    goalscorers["minute_base"] = goalscorers["minute"].apply(parse_minute_base)
    goalscorers["bucket"] = np.where(
        goalscorers["minute_base"].notna() & (goalscorers["minute_base"] <= 45),
        45,
        np.where(
            goalscorers["minute_base"].notna() & (goalscorers["minute_base"] <= 90),
            90,
            np.where(
                goalscorers["minute_base"].notna(),
                120,
                np.where(goalscorers["minute_is_na"], 90, np.nan),
            ),
        ),
    )

    home_totals = (
        goalscorers.loc[goalscorers["team"] == goalscorers["home_team"]]
        .groupby(["date", "home_team", "away_team"])
        .size()
        .rename("home_goals_total")
        .reset_index()
    )
    away_totals = (
        goalscorers.loc[goalscorers["team"] == goalscorers["away_team"]]
        .groupby(["date", "home_team", "away_team"])
        .size()
        .rename("away_goals_total")
        .reset_index()
    )

    bucketed = goalscorers.loc[goalscorers["bucket"].notna()].copy()
    home_bucket = (
        bucketed.loc[bucketed["team"] == bucketed["home_team"]]
        .groupby(["date", "home_team", "away_team", "bucket"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    away_bucket = (
        bucketed.loc[bucketed["team"] == bucketed["away_team"]]
        .groupby(["date", "home_team", "away_team", "bucket"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for b in (45, 90, 120):
        if b not in home_bucket.columns:
            home_bucket[b] = 0
        if b not in away_bucket.columns:
            away_bucket[b] = 0

    home_bucket = home_bucket.rename(
        columns={45: "home_score_45", 90: "home_score_90", 120: "home_score_120"}
    )
    away_bucket = away_bucket.rename(
        columns={45: "away_score_45", 90: "away_score_90", 120: "away_score_120"}
    )

    results = results.merge(
        home_totals, on=["date", "home_team", "away_team"], how="left"
    )
    results = results.merge(
        away_totals, on=["date", "home_team", "away_team"], how="left"
    )
    results = results.merge(
        home_bucket, on=["date", "home_team", "away_team"], how="left"
    )
    results = results.merge(
        away_bucket, on=["date", "home_team", "away_team"], how="left"
    )

    for col in [
        "home_score_45",
        "home_score_90",
        "home_score_120",
        "away_score_45",
        "away_score_90",
        "away_score_120",
    ]:
        results[col] = results[col].fillna(0)

    home_45 = results["home_score_45"]
    home_90 = results["home_score_90"]
    home_120 = results["home_score_120"]
    away_45 = results["away_score_45"]
    away_90 = results["away_score_90"]
    away_120 = results["away_score_120"]

    results["home_score_90"] = home_45 + home_90
    results["home_score_120"] = home_45 + home_90 + home_120
    results["away_score_90"] = away_45 + away_90
    results["away_score_120"] = away_45 + away_90 + away_120

    home_bucket_total = results["home_score_120"]
    away_bucket_total = results["away_score_120"]
    home_goals_total = results["home_goals_total"].fillna(0)
    away_goals_total = results["away_goals_total"].fillna(0)
    mismatch = (home_bucket_total != results["home_score"]) | (
        away_bucket_total != results["away_score"]
    )
    results["score_reconciled"] = ~mismatch
    if "tournament" in results.columns:
        scores_known = results["home_score"].notna() & results["away_score"].notna()
        non_zero = scores_known & ~(
            (results["home_score"] == 0) & (results["away_score"] == 0)
        )
        match_year = results["date"].dt.year
        recon_share = (
            results.loc[non_zero]
            .assign(match_year=match_year.loc[non_zero])
            .groupby(["match_year", "tournament"])["score_reconciled"]
            .mean()
        )
        share_index = pd.MultiIndex.from_arrays([match_year, results["tournament"]])
        share_values = recon_share.reindex(share_index).to_numpy(
            dtype=float, na_value=np.nan
        )
        zero_zero = (
            scores_known
            & (results["home_score"] == 0)
            & (results["away_score"] == 0)
        )
        zero_zero_mask = zero_zero.fillna(False).to_numpy(dtype=bool)
        zero_zero_share = share_values[zero_zero_mask]
        results.loc[zero_zero, "score_reconciled"] = (
            zero_zero_share > SCORE_RECONCILED_MIN_SHARE
        )

    results.loc[
        ~results["score_reconciled"],
        [
            "home_score_45",
            "home_score_90",
            "home_score_120",
            "away_score_45",
            "away_score_90",
            "away_score_120",
        ],
    ] = np.nan
    score_inconsistent = mismatch & (
        (home_goals_total > 0) | (away_goals_total > 0)
    )
    if score_inconsistent.any():
        sample = results.loc[
            score_inconsistent,
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "home_score_45",
                "home_score_90",
                "home_score_120",
                "away_score_45",
                "away_score_90",
                "away_score_120",
                "home_goals_total",
                "away_goals_total",
            ],
        ].head(50)
        raise ValueError(
            "[goalscorers] partial or inconsistent goalscorer data detected.\n"
            f"{sample.to_string(index=False)}"
        )

    results.loc[
        mismatch,
        [
            "home_score_45",
            "home_score_90",
            "home_score_120",
            "away_score_45",
            "away_score_90",
            "away_score_120",
        ],
    ] = np.nan
    et_matches = (
        goalscorers.loc[goalscorers["minute_base"].notna() & (goalscorers["minute_base"] > 90)]
        .groupby(["date", "home_team", "away_team"])
        .size()
        .rename("goals_after_90")
        .reset_index()
    )
    results = results.merge(
        et_matches, on=["date", "home_team", "away_team"], how="left"
    )
    results["had_extra_time"] = results["goals_after_90"].notna()
    results = results.drop(columns=["goals_after_90"])

    had_penalties_series = results["shootout_winner"].notna()
    no_extra_time_zero = (
        results["home_score"].fillna(0).eq(0)
        & results["away_score"].fillna(0).eq(0)
        & ~had_penalties_series
    )
    results.loc[no_extra_time_zero, ["home_score_120", "away_score_120"]] = np.nan
    results = results.drop(
        columns=[
            "home_goals_total",
            "away_goals_total",
        ]
    )
elif not ARGS.no_goalscorers:
    print("[goalscorers] goalscorers.csv not found; skipping join.")

no_extra_time_rule = pd.Series(False, index=results.index, dtype=bool)
extra_time_ref_path = MATCH_RESULTS_DIR / "extra_time_reference.csv"
if extra_time_ref_path.exists():
    ref = pd.read_csv(extra_time_ref_path, keep_default_na=False)
    ref = ref.replace({"": np.nan})
    ref["start_date"] = pd.to_datetime(ref["start_date"], errors="coerce")
    ref["end_date"] = pd.to_datetime(ref["end_date"], errors="coerce")

    def parse_no_extra_time(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in {"true", "t", "1", "yes"}:
            return True
        if s in {"false", "f", "0", "no"}:
            return False
        return np.nan

    ref["no_extra_time"] = ref["no_extra_time"].apply(parse_no_extra_time)

    for row in ref.itertuples(index=False):
        if pd.isna(row.no_extra_time):
            continue
        mask = pd.Series(True, index=results.index, dtype=bool)
        if pd.notna(row.tournament):
            mask &= results["tournament"] == row.tournament
        if hasattr(row, "home_team") and pd.notna(row.home_team):
            mask &= results["home_team"] == row.home_team
        if hasattr(row, "away_team") and pd.notna(row.away_team):
            mask &= results["away_team"] == row.away_team
        if pd.notna(row.start_date):
            mask &= results["date"] >= row.start_date
        if pd.notna(row.end_date):
            mask &= results["date"] <= row.end_date
        no_extra_time_rule.loc[mask] = bool(row.no_extra_time)

results["had_penalties"] = results["shootout_winner"].notna()
had_extra_time_from_goals = (
    results["had_extra_time"]
    if "had_extra_time" in results.columns
    else pd.Series(False, index=results.index, dtype=bool)
)

conflict = no_extra_time_rule & had_extra_time_from_goals
if conflict.any():
    sample = results.loc[
        conflict, ["date", "home_team", "away_team", "tournament"]
    ].head(20)
    raise ValueError(
        "[extra_time_reference] no-extra-time rule conflicts with goals after 90.\n"
        f"{sample.to_string(index=False)}"
    )

results["had_extra_time"] = had_extra_time_from_goals | (
    results["had_penalties"] & ~no_extra_time_rule
)
if "home_score_120" in results.columns and "away_score_120" in results.columns:
    results.loc[
        ~results["had_extra_time"], ["home_score_120", "away_score_120"]
    ] = np.nan


results.to_csv(MATCH_RESULTS_DIR / "results_clean.csv", index=False)
