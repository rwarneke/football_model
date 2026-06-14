from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_cup_results import (  # noqa: E402
    RESULTS_WC2026_PATH,
    RESULTS_WC2026_REQUIRED_COLUMNS,
    build_world_cup_2026_schedule,
    infer_world_cup_neutral,
    load_results_wc2026,
)

API_BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_LEAGUE_ID = 1
DEFAULT_SEASON = 2026
FINAL_STATUS_CODES = {"FT", "AET", "PEN"}
PENALTY_EVENT_DETAILS = {"Penalty", "Missed Penalty"}
KNOCKOUT_PLACEHOLDER_PREFIXES = ("Winner ", "Runner-up ", "Loser ", "3rd ")
TEAM_NAME_MAP_PATHS = [
    ROOT / "reference_data" / "kaggle_team_to_canonical_name_map.csv",
    ROOT / "reference_data" / "fifa_member_to_canonical_name_map.csv",
]


@dataclass(frozen=True)
class ApiFixture:
    fixture_id: int
    stage: str
    round_label: str
    timestamp: int | None
    status_short: str
    home_team: str
    away_team: str
    home_team_winner: bool | None
    away_team_winner: bool | None
    venue_name: str
    venue_city: str
    home_goals: int | None
    away_goals: int | None
    home_score_90: int | None
    away_score_90: int | None
    home_score_120: int | None
    away_score_120: int | None
    home_penalties: int | None
    away_penalties: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update match_results/results_wc2026.csv from API-Football data."
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("APIFOOTBALL_API_KEY")
            or os.environ.get("API_FOOTBALL_API_KEY")
            or os.environ.get("APISPORTS_KEY")
        ),
        help=(
            "API-Football key. Defaults to APIFOOTBALL_API_KEY, "
            "API_FOOTBALL_API_KEY, or APISPORTS_KEY."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("APIFOOTBALL_BASE_URL", API_BASE_URL),
        help=f"API base URL. Defaults to {API_BASE_URL}.",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=DEFAULT_LEAGUE_ID,
        help=f"Competition id. Defaults to {DEFAULT_LEAGUE_ID} for the World Cup.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help=f"Competition season. Defaults to {DEFAULT_SEASON}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_WC2026_PATH,
        help=f"Output CSV path. Defaults to {RESULTS_WC2026_PATH.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation with src.world_cup_results.load_results_wc2026().",
    )
    return parser


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def parse_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def parse_bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    return None


def build_team_name_normalizer() -> dict[str, str]:
    normalizer: dict[str, str] = {}
    for path in TEAM_NAME_MAP_PATHS:
        mapping = pd.read_csv(path)
        for row in mapping.itertuples(index=False):
            original = clean_text(row.original_name)
            replacement = clean_text(row.replacement_name)
            if original:
                normalizer[original.casefold()] = replacement
            if replacement:
                normalizer[replacement.casefold()] = replacement
    return normalizer


TEAM_NAME_NORMALIZER = build_team_name_normalizer()


def normalize_team_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return TEAM_NAME_NORMALIZER.get(text.casefold(), text)


def is_knockout_placeholder(label: str) -> bool:
    cleaned = clean_text(label)
    if not cleaned:
        return True
    return cleaned.startswith(KNOCKOUT_PLACEHOLDER_PREFIXES) or cleaned.lower().endswith(" winner")


def normalize_stage(round_label: Any) -> str:
    text = clean_text(round_label)
    lowered = text.casefold()
    if lowered.startswith("group stage"):
        return "Group"
    if lowered == "round of 32":
        return "Round of 32"
    if lowered == "round of 16":
        return "Round of 16"
    if lowered in {"quarter-finals", "quarterfinals", "quarterfinal", "quarter-final"}:
        return "Quarterfinal"
    if lowered in {"semi-finals", "semifinals", "semifinal", "semi-final"}:
        return "Semifinal"
    if lowered in {
        "third place",
        "third-place",
        "third place final",
        "3rd place",
        "3rd place final",
        "3rd-place",
        "3rd-place final",
    }:
        return "Third place"
    if lowered == "final":
        return "Final"
    raise ValueError(f"Unsupported API round label: {text!r}")


def ssl_context():
    import ssl

    ctx = ssl.create_default_context()
    try:
        import certifi
    except Exception:
        return ctx
    try:
        ctx.load_verify_locations(certifi.where())
    except Exception:
        return ctx
    return ctx


def api_get_json(
    *,
    api_key: str,
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    max_attempts: int = 4,
) -> dict[str, Any]:
    query = parse.urlencode(
        {key: value for key, value in params.items() if value is not None},
        doseq=True,
    )
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "wc2026-results-updater/1.0",
        "x-apisports-key": api_key,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=30, context=ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"API request failed ({exc.code}) for {url}\n{body}") from exc
            retry_after = exc.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else 2 ** (attempt - 1)
            time.sleep(sleep_seconds)
        except error.URLError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise RuntimeError(f"API request failed for {url}: {exc}") from exc
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"API request failed for {url}: {last_error}")


def fetch_all_fixtures(*, api_key: str, base_url: str, league_id: int, season: int) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = api_get_json(
            api_key=api_key,
            base_url=base_url,
            endpoint="/fixtures",
            params={"league": league_id, "season": season, "page": page},
        )
        page_rows = payload.get("response") or []
        fixtures.extend(page_rows)
        paging = payload.get("paging") or {}
        current = parse_int_or_none(paging.get("current"))
        total = parse_int_or_none(paging.get("total"))
        if not page_rows:
            break
        if total is None or current is None or current >= total:
            break
        page += 1
    return fixtures


def fetch_fixture_events(*, api_key: str, base_url: str, fixture_id: int) -> list[dict[str, Any]]:
    payload = api_get_json(
        api_key=api_key,
        base_url=base_url,
        endpoint="/fixtures/events",
        params={"fixture": fixture_id},
    )
    response = payload.get("response")
    return response if isinstance(response, list) else []


def build_api_fixture(raw: dict[str, Any]) -> ApiFixture:
    fixture_obj = raw.get("fixture") or {}
    league_obj = raw.get("league") or {}
    teams_obj = raw.get("teams") or {}
    home_team_obj = teams_obj.get("home") or {}
    away_team_obj = teams_obj.get("away") or {}
    goals_obj = raw.get("goals") or {}
    score_obj = raw.get("score") or {}
    fulltime_obj = score_obj.get("fulltime") or {}
    extratime_obj = score_obj.get("extratime") or {}
    penalty_obj = score_obj.get("penalty") or {}
    fixture_status_obj = fixture_obj.get("status") or {}
    venue_obj = fixture_obj.get("venue") or {}

    fixture_id = parse_int_or_none(fixture_obj.get("id"))
    if fixture_id is None:
        raise ValueError(f"Fixture payload is missing fixture.id: {raw}")

    return ApiFixture(
        fixture_id=fixture_id,
        stage=normalize_stage(league_obj.get("round")),
        round_label=clean_text(league_obj.get("round")),
        timestamp=parse_int_or_none(fixture_obj.get("timestamp")),
        status_short=clean_text(fixture_status_obj.get("short")).upper(),
        home_team=normalize_team_name(home_team_obj.get("name")),
        away_team=normalize_team_name(away_team_obj.get("name")),
        home_team_winner=parse_bool_or_none(home_team_obj.get("winner")),
        away_team_winner=parse_bool_or_none(away_team_obj.get("winner")),
        venue_name=clean_text(venue_obj.get("name")),
        venue_city=clean_text(venue_obj.get("city")),
        home_goals=parse_int_or_none(goals_obj.get("home")),
        away_goals=parse_int_or_none(goals_obj.get("away")),
        home_score_90=parse_int_or_none(fulltime_obj.get("home")),
        away_score_90=parse_int_or_none(fulltime_obj.get("away")),
        home_score_120=parse_int_or_none(extratime_obj.get("home")),
        away_score_120=parse_int_or_none(extratime_obj.get("away")),
        home_penalties=parse_int_or_none(penalty_obj.get("home")),
        away_penalties=parse_int_or_none(penalty_obj.get("away")),
    )


def sort_key_for_fixture(fixture: ApiFixture) -> tuple[int, int]:
    return (fixture.timestamp if fixture.timestamp is not None else 0, fixture.fixture_id)


def map_group_stage_fixtures(
    fixtures: list[ApiFixture], group_schedule: pd.DataFrame
) -> dict[int, ApiFixture]:
    grouped = fixtures
    schedule_by_pair: dict[tuple[str, str], int] = {}
    for row in group_schedule.itertuples(index=False):
        key = (normalize_team_name(row.home_team), normalize_team_name(row.away_team))
        if key in schedule_by_pair:
            raise ValueError(f"Duplicate group-stage schedule key: {key}")
        schedule_by_pair[key] = int(row.match_id)

    mapping: dict[int, ApiFixture] = {}
    for fixture in grouped:
        key = (fixture.home_team, fixture.away_team)
        match_id = schedule_by_pair.get(key)
        if match_id is None:
            raise ValueError(
                f"Could not map group-stage fixture {fixture.fixture_id}: "
                f"{fixture.home_team} vs {fixture.away_team}"
            )
        mapping[match_id] = fixture
    return mapping


def map_knockout_fixtures(
    fixtures: list[ApiFixture], knockout_schedule: pd.DataFrame
) -> dict[int, ApiFixture]:
    mapping: dict[int, ApiFixture] = {}
    for stage, stage_rows in knockout_schedule.groupby("stage", sort=False):
        schedule_rows = list(stage_rows.itertuples(index=False))
        api_rows = sorted([fixture for fixture in fixtures if fixture.stage == stage], key=sort_key_for_fixture)
        if len(api_rows) != len(schedule_rows):
            raise ValueError(
                f"API returned {len(api_rows)} fixtures for stage {stage!r}, "
                f"expected {len(schedule_rows)}."
            )
        for schedule_row, api_fixture in zip(schedule_rows, api_rows, strict=True):
            mapping[int(schedule_row.match_id)] = api_fixture
    return mapping


def infer_penalty_winner(fixture: ApiFixture) -> str | None:
    if fixture.home_penalties is not None and fixture.away_penalties is not None:
        if fixture.home_penalties > fixture.away_penalties:
            return fixture.home_team
        if fixture.away_penalties > fixture.home_penalties:
            return fixture.away_team
    if fixture.home_team_winner is True:
        return fixture.home_team
    if fixture.away_team_winner is True:
        return fixture.away_team
    return None


def infer_first_shooter(
    *,
    fixture: ApiFixture,
    api_key: str,
    base_url: str,
) -> str | None:
    if fixture.status_short != "PEN":
        return None

    events = fetch_fixture_events(api_key=api_key, base_url=base_url, fixture_id=fixture.fixture_id)
    penalty_events: list[tuple[int, int, str]] = []
    for event in events:
        detail = clean_text(event.get("detail"))
        if detail not in PENALTY_EVENT_DETAILS:
            continue
        team_name = normalize_team_name((event.get("team") or {}).get("name"))
        if team_name not in {fixture.home_team, fixture.away_team}:
            continue
        time_obj = event.get("time") or {}
        elapsed = parse_int_or_none(time_obj.get("elapsed"))
        extra = parse_int_or_none(time_obj.get("extra")) or 0
        if elapsed is None:
            continue
        penalty_events.append((elapsed, extra, team_name))

    likely_shootout = [event for event in penalty_events if event[0] >= 120]
    if len(likely_shootout) < 2:
        distinct_teams = {team for _, _, team in penalty_events}
        if len(penalty_events) >= 6 and len(distinct_teams) == 2:
            likely_shootout = penalty_events
        else:
            return None

    likely_shootout.sort()
    return likely_shootout[0][2]


def completed_row_payload(
    *,
    fixture: ApiFixture,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    if fixture.status_short not in FINAL_STATUS_CODES:
        return {}

    if fixture.status_short == "FT":
        home_score = fixture.home_goals
        away_score = fixture.away_goals
        home_score_90 = fixture.home_score_90 if fixture.home_score_90 is not None else home_score
        away_score_90 = fixture.away_score_90 if fixture.away_score_90 is not None else away_score
        if home_score is None or away_score is None or home_score_90 is None or away_score_90 is None:
            raise ValueError(f"Fixture {fixture.fixture_id} is FT but is missing final scores.")
        return {
            "home_score": home_score,
            "away_score": away_score,
            "home_score_90": home_score_90,
            "away_score_90": away_score_90,
            "home_score_120": pd.NA,
            "away_score_120": pd.NA,
            "went_extra_time": False,
            "went_penalties": False,
            "penalty_winner": pd.NA,
            "first_shooter": pd.NA,
        }

    if fixture.home_score_90 is None or fixture.away_score_90 is None:
        raise ValueError(
            f"Fixture {fixture.fixture_id} is {fixture.status_short} but is missing 90-minute scores."
        )
    if fixture.home_score_120 is None or fixture.away_score_120 is None:
        raise ValueError(
            f"Fixture {fixture.fixture_id} is {fixture.status_short} but is missing extra-time scores."
        )

    if fixture.status_short == "AET":
        return {
            "home_score": fixture.home_score_120,
            "away_score": fixture.away_score_120,
            "home_score_90": fixture.home_score_90,
            "away_score_90": fixture.away_score_90,
            "home_score_120": fixture.home_score_120,
            "away_score_120": fixture.away_score_120,
            "went_extra_time": True,
            "went_penalties": False,
            "penalty_winner": pd.NA,
            "first_shooter": pd.NA,
        }

    penalty_winner = infer_penalty_winner(fixture)
    if not penalty_winner:
        raise ValueError(
            f"Fixture {fixture.fixture_id} is PEN but the penalty winner could not be determined."
        )
    first_shooter = infer_first_shooter(
        fixture=fixture,
        api_key=api_key,
        base_url=base_url,
    )
    if first_shooter is None:
        print(
            f"Warning: could not derive first_shooter for penalty fixture {fixture.fixture_id}.",
            file=sys.stderr,
        )
    return {
        "home_score": fixture.home_score_120,
        "away_score": fixture.away_score_120,
        "home_score_90": fixture.home_score_90,
        "away_score_90": fixture.away_score_90,
        "home_score_120": fixture.home_score_120,
        "away_score_120": fixture.away_score_120,
        "went_extra_time": True,
        "went_penalties": True,
        "penalty_winner": penalty_winner,
        "first_shooter": first_shooter or pd.NA,
    }


def maybe_concrete_team_name(name: str) -> str | None:
    normalized = normalize_team_name(name)
    if not normalized or is_knockout_placeholder(normalized):
        return None
    return normalized


def build_results_frame(*, fixtures: list[ApiFixture], api_key: str, base_url: str) -> pd.DataFrame:
    schedule = build_world_cup_2026_schedule()
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.strftime("%Y-%m-%d")

    group_schedule = schedule.loc[schedule["stage"].eq("Group")].copy()
    knockout_schedule = schedule.loc[~schedule["stage"].eq("Group")].copy()

    group_fixtures = [fixture for fixture in fixtures if fixture.stage == "Group"]
    knockout_fixtures = [fixture for fixture in fixtures if fixture.stage != "Group"]

    fixture_by_match_id: dict[int, ApiFixture] = {}
    fixture_by_match_id.update(map_group_stage_fixtures(group_fixtures, group_schedule))
    fixture_by_match_id.update(map_knockout_fixtures(knockout_fixtures, knockout_schedule))

    rows: list[dict[str, Any]] = []
    for row in schedule.itertuples(index=False):
        fixture = fixture_by_match_id.get(int(row.match_id))

        home_team = clean_text(row.home_team)
        away_team = clean_text(row.away_team)
        neutral: Any = pd.NA

        if row.stage == "Group":
            neutral = infer_world_cup_neutral(home_team, away_team, clean_text(row.country))
        elif fixture is not None:
            mapped_home = maybe_concrete_team_name(fixture.home_team)
            mapped_away = maybe_concrete_team_name(fixture.away_team)
            if mapped_home and mapped_away:
                home_team = mapped_home
                away_team = mapped_away
                neutral = infer_world_cup_neutral(home_team, away_team, clean_text(row.country))

        output_row: dict[str, Any] = {
            "match_id": int(row.match_id),
            "date": clean_text(row.date),
            "stage": clean_text(row.stage),
            "group": clean_text(row.group) or pd.NA,
            "home_team": home_team,
            "away_team": away_team,
            "stadium": clean_text(row.stadium),
            "city": clean_text(row.city),
            "country": clean_text(row.country),
            "tournament": "FIFA World Cup",
            "neutral": neutral,
            "home_score": pd.NA,
            "away_score": pd.NA,
            "home_score_90": pd.NA,
            "away_score_90": pd.NA,
            "home_score_120": pd.NA,
            "away_score_120": pd.NA,
            "went_extra_time": pd.NA,
            "went_penalties": pd.NA,
            "penalty_winner": pd.NA,
            "first_shooter": pd.NA,
        }

        if fixture is not None and fixture.status_short in FINAL_STATUS_CODES:
            output_row.update(
                completed_row_payload(
                    fixture=fixture,
                    api_key=api_key,
                    base_url=base_url,
                )
            )
            if pd.isna(output_row["neutral"]):
                output_row["neutral"] = infer_world_cup_neutral(
                    output_row["home_team"],
                    output_row["away_team"],
                    output_row["country"],
                )

        rows.append(output_row)

    return pd.DataFrame(rows, columns=RESULTS_WC2026_REQUIRED_COLUMNS)


def validate_results_file(path: Path) -> None:
    import src.world_cup_results as world_cup_results

    original_path = world_cup_results.RESULTS_WC2026_PATH
    try:
        world_cup_results.RESULTS_WC2026_PATH = path
        load_results_wc2026()
    finally:
        world_cup_results.RESULTS_WC2026_PATH = original_path


def write_results_frame(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False, na_rep="NA")
    temp_path.replace(output_path)


def format_path_for_display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing API key. Set APIFOOTBALL_API_KEY (or pass --api-key) before running."
        )

    raw_fixtures = fetch_all_fixtures(
        api_key=args.api_key,
        base_url=args.base_url,
        league_id=args.league_id,
        season=args.season,
    )
    fixtures = [build_api_fixture(raw) for raw in raw_fixtures]
    results_df = build_results_frame(
        fixtures=fixtures,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    temp_validation_path = args.output.with_suffix(args.output.suffix + ".candidate")
    try:
        write_results_frame(results_df, temp_validation_path)
        if not args.no_validate:
            validate_results_file(temp_validation_path)
        temp_validation_path.replace(args.output)
    finally:
        if temp_validation_path.exists():
            temp_validation_path.unlink()

    completed_mask = results_df["home_score"].notna() & results_df["away_score"].notna()
    penalties_mask = results_df["went_penalties"].fillna(False).astype(bool)
    print(
        f"Wrote {format_path_for_display(args.output)} with "
        f"{int(completed_mask.sum())} completed matches "
        f"and {int(penalties_mask.sum())} penalty shootouts."
    )


if __name__ == "__main__":
    main()
