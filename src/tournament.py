from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import math
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests

from src.model import Model, safe_exp

ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DATA_DIR = ROOT_DIR / "reference_data"
QUALIFIED_TEAMS_PATH = REFERENCE_DATA_DIR / "world_cup_2026_qualified.csv"
TEAM_UNIVERSE_PATH = REFERENCE_DATA_DIR / "team_universe.csv"
CONFEDERATIONS_PATH = REFERENCE_DATA_DIR / "confederations.csv"
WORLD_CUP_2026_GROUPS_PATH = REFERENCE_DATA_DIR / "world_cup_2026_groups.csv"
WORLD_CUP_2026_GROUP_MATCHES_PATH = (
    REFERENCE_DATA_DIR / "world_cup_2026_group_matches.csv"
)
REMAINING_QUALIFIERS_PATH = REFERENCE_DATA_DIR / "world_cup_2026_remaining_qualifiers.csv"
ROUND_OF_32_COMBINATIONS_PATH = (
    REFERENCE_DATA_DIR / "world_cup_2026_round_of_32_combinations.csv"
)
KNOCKOUT_MATCHES_PATH = REFERENCE_DATA_DIR / "world_cup_2026_knockout_matches.csv"
KAGGLE_TEAM_MAP_PATH = REFERENCE_DATA_DIR / "kaggle_team_to_canonical_name_map.csv"
FIFA_TEAM_MAP_PATH = REFERENCE_DATA_DIR / "fifa_member_to_canonical_name_map.csv"

WORLD_CUP_START_DATE = pd.Timestamp("2026-06-11")

QUALIFICATION_SLOTS = {
    "AFC": {"direct": 8, "interconfed_playoff": 1},
    "CAF": {"direct": 9, "interconfed_playoff": 1},
    "CONCACAF": {"direct": 6, "interconfed_playoff": 2},
    "CONMEBOL": {"direct": 6, "interconfed_playoff": 1},
    "OFC": {"direct": 1, "interconfed_playoff": 1},
    "UEFA": {"direct": 16, "interconfed_playoff": 0},
}

QUALIFICATION_FORMAT = {
    "AFC": {
        "groups": 8,
        "group_direct_slots": 8,
        "playoff_candidates": 4,
        "confed_playoff_direct": 0,
    },
    "CAF": {
        "groups": 9,
        "group_direct_slots": 9,
        "playoff_candidates": 4,
        "confed_playoff_direct": 0,
    },
    "CONCACAF": {
        "groups": 3,
        "group_direct_slots": 3,
        "playoff_candidates": 4,
        "confed_playoff_direct": 0,
    },
    "CONMEBOL": {
        "groups": 1,
        "group_direct_slots": 6,
        "playoff_candidates": 1,
        "confed_playoff_direct": 0,
    },
    "OFC": {
        "groups": 2,
        "group_direct_slots": 1,
        "playoff_candidates": 2,
        "confed_playoff_direct": 0,
    },
    "UEFA": {
        "groups": 12,
        "group_direct_slots": 12,
        "playoff_candidates": 8,
        "confed_playoff_direct": 4,
    },
}

QUALIFICATION_LEAD_DAYS = 210
QUALIFICATION_STAGE_DAYS = 30


@dataclass
class MatchResult:
    stage: str
    day: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    is_neutral: bool = True
    date: Optional[pd.Timestamp] = None
    stadium: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    group: Optional[str] = None
    home_score_90: Optional[int] = None
    away_score_90: Optional[int] = None
    home_score_120: Optional[int] = None
    away_score_120: Optional[int] = None
    went_extra_time: bool = False
    went_penalties: bool = False
    penalty_winner: Optional[str] = None
    winner: Optional[str] = None


@dataclass
class TeamSimState:
    m: np.ndarray
    last_day: int


class Tournament:
    def __init__(self, name: str, teams: Optional[List[str]] = None):
        self.name = name
        self.teams = list(teams or [])
        self.matches: List[MatchResult] = []
        self.group_tables: Dict[str, pd.DataFrame] = {}
        self.finished = False
        self.champion: Optional[str] = None

    def simulate(self, model: Model, random_state: Optional[int] = None) -> "Tournament":
        rng = np.random.default_rng(random_state)
        self._simulate(model, rng)
        self.finished = True
        return self

    def _simulate(self, model: Model, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def results_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(m) for m in self.matches])

    def stage_of_elimination(self) -> Dict[str, str]:
        res = self.results_frame()
        if res.empty:
            return {}
        third_place = res.loc[res["stage"] == "Third place", "winner"]
        third_place_winner = third_place.iloc[0] if not third_place.empty else None
        res = res.sort_values(["day"])
        mapping = {
            "Group": "1. Group",
            "Round of 32": "2. Round of 32",
            "Round of 16": "3. Round of 16",
            "Quarterfinal": "4. Quarterfinal",
            "Fourth place": "5. Fourth place",
            "Third place": "6. Third place",
            "Final": "7. Final",
            "Champion": "8. Champion",
        }
        stages: Dict[str, str] = {}
        qualifying_teams: set[str] = set()
        if self.qualification_matches:
            qualifying_teams = {
                t
                for m in self.qualification_matches
                for t in (m.home_team, m.away_team)
                if t
            }
        else:
            qualifying_matches = res.loc[res["stage"].str.contains("Qual", na=False)]
            if not qualifying_matches.empty:
                qualifying_teams = set(qualifying_matches["home_team"]).union(
                    set(qualifying_matches["away_team"])
                )
        for team in sorted(qualifying_teams.difference(self.teams)):
            stages[team] = "0. Qualifying"
        for team in self.teams:
            team_res = res[(res["home_team"] == team) | (res["away_team"] == team)]
            if team_res.empty:
                continue
            stage = team_res.iloc[-1]["stage"]
            if stage == "Final" and getattr(self, "champion", None) == team:
                stage = "Champion"
            elif stage == "Third place":
                if third_place_winner and third_place_winner != team:
                    stage = "Fourth place"
            stages[team] = mapping.get(stage, f"0.{stage}")
        return stages


class WorldCup2026(Tournament):
    """
    2026 format (as of today):
      - 48 teams, 12 groups of 4
      - top 2 from each group + 8 best third-place teams advance (32 total)
      - knockout from round of 32

    The round-of-32 bracket is seeded by group performance (winners, then runners-up,
    then best thirds) and paired 1-v-32, 2-v-31, ... with a best-effort swap to avoid
    same-group matchups in the first knockout round.
    """

    GROUPS = [chr(ord("A") + i) for i in range(12)]
    GROUP_MATCHDAY_DAYS = [0, 4, 8]
    ROUND_DAY_OFFSETS = {
        "Round of 32": 14,
        "Round of 16": 18,
        "Quarterfinal": 22,
        "Semifinal": 25,
        "Third place": 28,
        "Final": 29,
    }

    def __init__(
        self,
        teams: Optional[List[str]] = None,
        groups: Optional[Dict[str, List[str]]] = None,
        host_teams: Optional[List[str]] = None,
        start_day: int = 0,
        start_date: Optional[str] = None,
        include_qualification: bool = True,
        groups_path: Optional[Path] = None,
        group_matches_path: Optional[Path] = None,
        host_team_countries: Optional[Dict[str, str]] = None,
        use_wikipedia_draw: bool = False,
        allow_group_simulation: bool = False,
    ):
        self._provided_groups = groups
        self._provided_teams = teams
        self.include_qualification = bool(include_qualification)
        self.groups: Dict[str, List[str]] = {}
        self.qualification_paths: Dict[str, str] = {}
        self.qualification_matches: List[MatchResult] = []
        self.qualification_teams: List[str] = []
        self.slot_winners: Dict[str, str] = {}
        self._needs_group_draw = False
        self.groups_path = Path(groups_path) if groups_path else None
        self.group_matches_path = (
            Path(group_matches_path) if group_matches_path else None
        )
        self.use_wikipedia_draw = bool(use_wikipedia_draw)
        self.allow_group_simulation = bool(allow_group_simulation)
        self._team_name_map: Optional[Dict[str, str]] = None

        if groups is not None:
            self.groups = {g: list(ts) for g, ts in groups.items()}
            if set(self.groups.keys()) != set(self.GROUPS):
                raise ValueError("WorldCup2026 requires groups A-L")
            for g, ts in self.groups.items():
                if len(ts) != 4:
                    raise ValueError(f"Group {g} must have 4 teams")
            all_teams = [t for ts in self.groups.values() for t in ts]
            if len(all_teams) != 48 or len(set(all_teams)) != 48:
                raise ValueError("WorldCup2026 requires 48 unique teams across groups")
            super().__init__(name="WorldCup2026", teams=all_teams)
        else:
            if teams is not None and len(teams) != 48:
                raise ValueError("WorldCup2026 requires 48 teams when groups not provided")
            super().__init__(name="WorldCup2026", teams=teams or [])
            self._needs_group_draw = True
        ordered_hosts = list(dict.fromkeys(host_teams or []))
        self.host_teams = set(ordered_hosts)
        self.host_teams_ordered = ordered_hosts
        default_host_countries = {
            "USA": "United States",
            "United States": "United States",
            "Canada": "Canada",
            "Mexico": "Mexico",
        }
        self.host_team_countries = {
            **default_host_countries,
            **(host_team_countries or {}),
        }
        self.start_day = int(start_day)
        self.start_date = (
            pd.Timestamp(start_date) if start_date is not None else WORLD_CUP_START_DATE
        )

    def _simulate(self, model: Model, rng: np.random.Generator) -> None:
        states: Dict[str, TeamSimState]
        if self._provided_groups:
            states = self._init_team_states(model, rng)
        elif self._provided_teams:
            states = self._init_team_states(model, rng, teams=self._provided_teams)
            if self._needs_group_draw:
                self.groups = self._resolve_groups(self._provided_teams, model)
        else:
            if not self.include_qualification:
                raise ValueError("Qualification is required when no teams/groups are provided")
            states, qualified, slot_winners = self._simulate_remaining_qualification(model, rng)
            self.slot_winners = dict(slot_winners)
            self.teams = list(qualified)
            self.qualification_teams = list(qualified)
            if self.qualification_matches:
                self.matches.extend(self.qualification_matches)
            if self._needs_group_draw:
                self.groups = self._resolve_groups(qualified, model, slot_winners)

        if not self.groups:
            raise ValueError("Groups not initialized for WorldCup2026")
        self.teams = [t for ts in self.groups.values() for t in ts]

        group_matches = self._group_stage_schedule(model)
        group_results = self._simulate_group_stage(model, rng, states, group_matches)
        self.matches.extend(group_results)

        group_tables, group_rankings = self._group_tables(group_results, rng)
        self.group_tables = group_tables

        _qualifiers, _third_place, best_third = self._select_qualifiers(
            group_rankings, rng
        )
        knockout_results = self._simulate_knockout_official(
            model, rng, states, group_rankings, best_third
        )
        self.matches.extend(knockout_results)

    def _init_team_states(
        self,
        model: Model,
        rng: np.random.Generator,
        teams: Optional[List[str]] = None,
        start_day: Optional[int] = None,
    ) -> Dict[str, TeamSimState]:
        states: Dict[str, TeamSimState] = {}
        init_day = self.start_day if start_day is None else int(start_day)
        for team in teams or self.teams:
            st = model.teams.get(team)
            if st is None:
                raise ValueError(f"Team not in model: {team}")
            m = rng.multivariate_normal(st.m, st.sigma2)
            states[team] = TeamSimState(m=m, last_day=init_day)
        return states

    def _advance_state(
        self,
        model: Model,
        rng: np.random.Generator,
        state: TeamSimState,
        day: int,
    ) -> None:
        delta_days = day - state.last_day
        if delta_days <= 0:
            return
        if model.variance_per_year > 0.0:
            delta_years = delta_days / 365.0
            step_var = model.variance_per_year * delta_years
            if step_var > 0.0:
                cross = step_var * model.cross_var_ratio
                cov = np.array([[step_var, cross], [cross, step_var]], dtype=float)
                step = rng.multivariate_normal([0.0, 0.0], cov)
                state.m = state.m + step
        state.last_day = day

    def _date_to_day(self, date: pd.Timestamp) -> int:
        return int((pd.Timestamp(date) - self.start_date).days)

    def _day_to_date(self, day: int) -> pd.Timestamp:
        return self.start_date + pd.Timedelta(days=int(day))

    def _match_params(
        self,
        model: Model,
        mu_h: np.ndarray,
        mu_a: np.ndarray,
        home_advantage: bool,
        away_advantage: bool,
        extra_time_mult: float = 1.0,
    ) -> Tuple[float, float, float, float, float, float]:
        if home_advantage and away_advantage:
            raise ValueError("Match cannot assign home advantage to both teams")

        if home_advantage or away_advantage:
            a_hga, d_hga = model._current_hga_components()
        else:
            a_hga = 0.0
            d_hga = 0.0

        if home_advantage:
            eta_h = model.mu + (mu_h[0] + a_hga) - mu_a[1]
            eta_a = model.mu + mu_a[0] - (mu_h[1] + d_hga)
        elif away_advantage:
            eta_h = model.mu + mu_h[0] - (mu_a[1] + d_hga)
            eta_a = model.mu + (mu_a[0] + a_hga) - mu_h[1]
        else:
            eta_h = model.mu + mu_h[0] - mu_a[1]
            eta_a = model.mu + mu_a[0] - mu_h[1]
        if extra_time_mult != 1.0:
            shift = math.log(extra_time_mult)
            eta_h += shift
            eta_a += shift

        m_h = safe_exp(float(eta_h))
        m_a = safe_exp(float(eta_a))

        rho = float(model.rho)
        rho = min(max(rho, 0.0), 1.0 - 1e-9)
        smin = float(Model._smin_eps(m_h, m_a, eps=model.smin_eps_epsilon))
        if smin < 0.0:
            smin = 0.0
        nu = rho * smin
        lam_h = max(m_h - nu, 0.0)
        lam_a = max(m_a - nu, 0.0)
        skilldiff = float(eta_h - eta_a)
        return m_h, m_a, lam_h, lam_a, nu, skilldiff

    def _sample_score(
        self,
        rng: np.random.Generator,
        lam_h: float,
        lam_a: float,
        nu: float,
    ) -> Tuple[int, int]:
        u = rng.poisson(nu) if nu > 0.0 else 0
        v_h = rng.poisson(lam_h) if lam_h > 0.0 else 0
        v_a = rng.poisson(lam_a) if lam_a > 0.0 else 0
        return int(u + v_h), int(u + v_a)

    def _simulate_match(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        day: int,
        match_date: Optional[pd.Timestamp],
        home_team: str,
        away_team: str,
        stage: str,
        group: Optional[str],
        allow_draw: bool,
        neutral_override: Optional[bool] = None,
        stadium: Optional[str] = None,
        city: Optional[str] = None,
        country: Optional[str] = None,
    ) -> MatchResult:
        self._advance_state(model, rng, states[home_team], day)
        self._advance_state(model, rng, states[away_team], day)

        home_advantage = False
        away_advantage = False
        if neutral_override is None:
            neutral = True
            if country:
                match_country = str(country).strip()
                home_country = self.host_team_countries.get(home_team, "")
                away_country = self.host_team_countries.get(away_team, "")
                if (
                    home_country
                    and match_country
                    and home_country.casefold() == match_country.casefold()
                ):
                    home_advantage = True
                if (
                    away_country
                    and match_country
                    and away_country.casefold() == match_country.casefold()
                ):
                    away_advantage = True
                if home_advantage and away_advantage:
                    home_advantage = False
                    away_advantage = False
                elif home_advantage ^ away_advantage:
                    neutral = False
            elif self.host_teams:
                home_is_host = home_team in self.host_teams
                away_is_host = away_team in self.host_teams
                if home_is_host ^ away_is_host:
                    neutral = False
                    home_advantage = home_is_host
                    away_advantage = away_is_host
            if away_advantage and not home_advantage:
                home_team, away_team = away_team, home_team
                home_advantage, away_advantage = away_advantage, home_advantage
        else:
            neutral = bool(neutral_override)
            if not neutral:
                home_advantage = True

        m_h, m_a, lam_h, lam_a, nu, skilldiff = self._match_params(
            model,
            states[home_team].m,
            states[away_team].m,
            home_advantage,
            away_advantage,
            extra_time_mult=1.0,
        )
        home_90, away_90 = self._sample_score(rng, lam_h, lam_a, nu)

        if allow_draw or home_90 != away_90:
            winner = None
            if home_90 > away_90:
                winner = home_team
            elif away_90 > home_90:
                winner = away_team
            return MatchResult(
                stage=stage,
                day=day,
                date=match_date,
                home_team=home_team,
                away_team=away_team,
                home_score=home_90,
                away_score=away_90,
                is_neutral=neutral,
                stadium=stadium,
                city=city,
                country=country,
                group=group,
                home_score_90=home_90,
                away_score_90=away_90,
                home_score_120=None,
                away_score_120=None,
                went_extra_time=False,
                went_penalties=False,
                penalty_winner=None,
                winner=winner,
            )

        m_h_et, m_a_et, lam_h_et, lam_a_et, nu_et, _ = self._match_params(
            model,
            states[home_team].m,
            states[away_team].m,
            home_advantage,
            away_advantage,
            extra_time_mult=model.extra_time_exp_score_mult,
        )
        home_et, away_et = self._sample_score(rng, lam_h_et, lam_a_et, nu_et)
        home_120 = home_90 + home_et
        away_120 = away_90 + away_et

        if home_120 != away_120:
            winner = home_team if home_120 > away_120 else away_team
            return MatchResult(
                stage=stage,
                day=day,
                date=match_date,
                home_team=home_team,
                away_team=away_team,
                home_score=home_120,
                away_score=away_120,
                is_neutral=neutral,
                stadium=stadium,
                city=city,
                country=country,
                group=group,
                home_score_90=home_90,
                away_score_90=away_90,
                home_score_120=home_120,
                away_score_120=away_120,
                went_extra_time=True,
                went_penalties=False,
                penalty_winner=None,
                winner=winner,
            )

        p_home_pen = 1.0 / (1.0 + safe_exp(-model.shootout_skilldiff_coef * skilldiff))
        pen_winner = home_team if rng.random() < p_home_pen else away_team
        return MatchResult(
            stage=stage,
            day=day,
            date=match_date,
            home_team=home_team,
            away_team=away_team,
            home_score=home_120,
            away_score=away_120,
            is_neutral=neutral,
            stadium=stadium,
            city=city,
            country=country,
            group=group,
            home_score_90=home_90,
            away_score_90=away_90,
            home_score_120=home_120,
            away_score_120=away_120,
            went_extra_time=True,
            went_penalties=True,
            penalty_winner=pen_winner,
            winner=pen_winner,
        )

    def _team_strength(self, state: TeamSimState) -> float:
        return float(state.m[0] + state.m[1])

    def _resolve_groups(
        self,
        teams: List[str],
        model: Model,
        slot_winners: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[str]]:
        if self._provided_groups:
            return self.groups
        groups_path = self.groups_path or WORLD_CUP_2026_GROUPS_PATH
        if groups_path.exists():
            groups = self._load_groups_from_csv(groups_path, model)
            groups = self._fill_group_slots(groups, slot_winners, model)
            self._validate_groups(groups, model)
            return groups
        if self.use_wikipedia_draw:
            groups = self._load_groups_from_wikipedia(model)
            self._write_groups_csv(groups_path, groups)
            groups = self._fill_group_slots(groups, slot_winners, model)
            self._validate_groups(groups, model)
            return groups
        if self.allow_group_simulation:
            return self._draw_groups(teams, model)
        raise ValueError(
            "WorldCup2026 group draw not available. "
            "Provide groups, set groups_path, or enable allow_group_simulation."
        )

    def _draw_groups(self, teams: List[str], model: Model) -> Dict[str, List[str]]:
        if len(teams) != 48:
            raise ValueError("WorldCup2026 draw requires 48 teams")
        missing = [t for t in teams if t not in model.teams]
        if missing:
            raise ValueError(
                "WorldCup2026 draw missing teams in model: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        strength_order = sorted(
            teams,
            key=lambda t: (
                -float(model.teams[t].m[0] + model.teams[t].m[1]),
                t,
            ),
        )
        pot_size = 12
        host_list = [t for t in self.host_teams_ordered if t in teams]
        if len(host_list) > pot_size:
            raise ValueError("Too many host teams for pot 1")

        seeded = [t for t in strength_order if t not in host_list]
        pot1 = host_list + seeded[: pot_size - len(host_list)]
        remaining = seeded[pot_size - len(host_list) :]
        pots = [
            pot1,
            remaining[:pot_size],
            remaining[pot_size : 2 * pot_size],
            remaining[2 * pot_size : 3 * pot_size],
        ]
        if any(len(pot) != pot_size for pot in pots):
            raise ValueError("Draw produced invalid pot sizes")

        groups = {g: [] for g in self.GROUPS}
        for g, host in zip(self.GROUPS, host_list):
            groups[g].append(host)

        pot1_remaining = [t for t in pot1 if t not in host_list]
        for g in self.GROUPS:
            if groups[g]:
                continue
            groups[g].append(pot1_remaining.pop(0))

        for pot in pots[1:]:
            for g in self.GROUPS:
                if len(groups[g]) >= 4:
                    continue
                if pot:
                    groups[g].append(pot.pop(0))

        for g, ts in groups.items():
            if len(ts) != 4:
                raise ValueError(f"Draw produced invalid group size for {g}")
        return groups

    def _load_groups_from_csv(self, path: Path, model: Model) -> Dict[str, List[str]]:
        df = pd.read_csv(path)
        if not {"group", "team"}.issubset(df.columns):
            raise ValueError("Groups CSV must include 'group' and 'team' columns")
        df["group"] = df["group"].astype(str).str.strip()
        df["team"] = df["team"].astype(str).str.strip()
        groups: Dict[str, List[str]] = {}
        for group, sub in df.groupby("group"):
            group = str(group).strip()
            if group not in self.GROUPS:
                raise ValueError(f"Invalid group name in {path}: {group}")
            teams = [self._canonicalize_team_name(t, model) for t in sub["team"].tolist()]
            groups[group] = teams
        return groups

    def _write_groups_csv(self, path: Path, groups: Dict[str, List[str]]) -> None:
        rows = []
        for group, teams in groups.items():
            for team in teams:
                rows.append({"group": group, "team": team})
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def _load_groups_from_wikipedia(self, model: Model) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for group in self.GROUPS:
            url = f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group}"
            html = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
            ).text
            soup = BeautifulSoup(html, "html.parser")
            teams = self._parse_group_table(soup)
            if teams:
                groups[group] = [self._canonicalize_team_name(t, model) for t in teams]
        if len(groups) != len(self.GROUPS):
            raise ValueError(
                "Wikipedia does not list a complete 2026 group draw yet."
            )
        return groups

    def _parse_group_table(self, soup: BeautifulSoup) -> List[str]:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])]
            if "draw position" in headers and "team" in headers:
                team_idx = headers.index("team")
                teams: List[str] = []
                for row in rows[1:]:
                    cells = row.find_all(["th", "td"])
                    if len(cells) <= team_idx:
                        continue
                    raw = cells[team_idx].get_text(" ", strip=True)
                    cleaned = self._clean_team_name(raw)
                    if not cleaned:
                        continue
                    lower = cleaned.casefold()
                    if lower.startswith("november") or lower.startswith("june"):
                        continue
                    teams.append(cleaned)
                if len(teams) >= 4:
                    return teams[:4]
        return []

    def _clean_team_name(self, raw: str) -> str:
        cleaned = re.sub(r"\\[.*?\\]", "", str(raw))
        cleaned = re.sub(r"\\s*\\(.*?\\)\\s*", "", cleaned)
        cleaned = cleaned.replace("\\xa0", " ").strip()
        cleaned = re.sub(r"\\s+", " ", cleaned)
        return unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")

    def _load_team_name_map(self) -> Dict[str, str]:
        if self._team_name_map is not None:
            return self._team_name_map
        mapping: Dict[str, str] = {}
        for path in (KAGGLE_TEAM_MAP_PATH, FIFA_TEAM_MAP_PATH):
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if not {"original_name", "replacement_name"}.issubset(df.columns):
                continue
            for row in df.itertuples():
                orig = self._clean_team_name(row.original_name)
                repl = self._clean_team_name(row.replacement_name)
                mapping[orig.casefold()] = repl
        self._team_name_map = mapping
        return mapping

    def _canonicalize_team_name(self, team: str, model: Model) -> str:
        cleaned = self._clean_team_name(team)
        if cleaned in model.teams:
            return cleaned
        mapping = self._load_team_name_map()
        mapped = mapping.get(cleaned.casefold(), cleaned)
        return mapped

    def _is_slot_placeholder(self, team: str) -> bool:
        return bool(
            re.match(r"^UEFA Path [A-D] winner$", team)
            or re.match(r"^IC Path [12] winner$", team)
        )

    def _fill_group_slots(
        self,
        groups: Dict[str, List[str]],
        slot_winners: Optional[Dict[str, str]],
        model: Model,
    ) -> Dict[str, List[str]]:
        slot_winners = slot_winners or {}
        filled: Dict[str, List[str]] = {}
        for group, teams in groups.items():
            resolved: List[str] = []
            for team in teams:
                if self._is_slot_placeholder(team):
                    if team not in slot_winners:
                        raise ValueError(
                            f"Group draw includes unresolved slot: {team}"
                        )
                    resolved_team = slot_winners[team]
                    if resolved_team not in model.teams:
                        raise ValueError(
                            f"Slot winner not in model: {team} -> {resolved_team}"
                        )
                    resolved.append(resolved_team)
                    self.qualification_paths[resolved_team] = team
                else:
                    resolved.append(team)
            filled[group] = resolved
        return filled

    def _validate_groups(self, groups: Dict[str, List[str]], model: Model) -> None:
        if set(groups.keys()) != set(self.GROUPS):
            raise ValueError("WorldCup2026 groups must include A-L")
        all_teams = [t for ts in groups.values() for t in ts]
        if len(all_teams) != 48 or len(set(all_teams)) != 48:
            raise ValueError("WorldCup2026 requires 48 unique teams across groups")
        missing = [t for t in all_teams if t not in model.teams]
        if missing:
            raise ValueError(
                "Groups include teams missing from model: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

    def _load_qualified_teams(self) -> pd.DataFrame:
        if not QUALIFIED_TEAMS_PATH.exists():
            raise FileNotFoundError(f"Missing qualified teams file: {QUALIFIED_TEAMS_PATH}")
        df = pd.read_csv(QUALIFIED_TEAMS_PATH)
        if "team" not in df.columns:
            raise ValueError("Qualified teams file must include a 'team' column")
        df["team"] = df["team"].astype(str).str.strip()
        return df

    def _load_remaining_qualifiers(self, model: Model) -> pd.DataFrame:
        if not REMAINING_QUALIFIERS_PATH.exists():
            raise FileNotFoundError(
                f"Missing remaining qualifiers file: {REMAINING_QUALIFIERS_PATH}"
            )
        df = pd.read_csv(REMAINING_QUALIFIERS_PATH)
        required = {
            "date",
            "stage",
            "path",
            "round",
            "home_team",
            "away_team",
            "neutral",
            "home_source",
            "away_source",
            "winner_slot",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Remaining qualifiers file missing columns: {sorted(missing)}"
            )
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df["stage"] = df["stage"].astype(str).str.strip()
        df["path"] = df["path"].astype(str).str.strip()
        df["round"] = df["round"].astype(str).str.strip()
        df["home_team"] = df["home_team"].astype(str).str.strip()
        df["away_team"] = df["away_team"].astype(str).str.strip()
        def parse_bool(val) -> bool:
            if isinstance(val, bool):
                return val
            text = str(val).strip().lower()
            return text in {"1", "true", "yes", "y"}

        df["neutral"] = df["neutral"].apply(parse_bool)
        df["home_source"] = df["home_source"].astype(str).str.strip()
        df["away_source"] = df["away_source"].astype(str).str.strip()
        df["winner_slot"] = df["winner_slot"].astype(str).str.strip()

        def canonicalize(team: str) -> str:
            if not team or team.lower() == "nan":
                return ""
            return self._canonicalize_team_name(team, model)

        df["home_team"] = df["home_team"].apply(canonicalize)
        df["away_team"] = df["away_team"].apply(canonicalize)
        return df

    def _load_group_matches(self, model: Model) -> pd.DataFrame:
        path = self.group_matches_path or WORLD_CUP_2026_GROUP_MATCHES_PATH
        if not path.exists():
            raise FileNotFoundError(f"Missing group matches file: {path}")
        df = pd.read_csv(path)
        required = {"date", "group", "home_team", "away_team", "city", "country"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Group matches file missing columns: {sorted(missing)}")
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df["group"] = df["group"].astype(str).str.strip()
        df["home_team"] = df["home_team"].astype(str).str.strip()
        df["away_team"] = df["away_team"].astype(str).str.strip()
        df["stadium"] = df["stadium"].astype(str).str.strip() if "stadium" in df.columns else ""
        df["city"] = df["city"].astype(str).str.strip()
        df["country"] = df["country"].astype(str).str.strip()

        def canonicalize(team: str) -> str:
            if not team or team.lower() == "nan":
                return ""
            return self._canonicalize_team_name(team, model)

        df["home_team"] = df["home_team"].apply(canonicalize)
        df["away_team"] = df["away_team"].apply(canonicalize)
        return df

    def _load_round_of_32_combinations(self) -> Dict[str, Dict[str, str]]:
        if not ROUND_OF_32_COMBINATIONS_PATH.exists():
            raise FileNotFoundError(
                f"Missing round-of-32 combinations file: {ROUND_OF_32_COMBINATIONS_PATH}"
            )
        df = pd.read_csv(ROUND_OF_32_COMBINATIONS_PATH)
        required = {"combo", "1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Round-of-32 combinations file missing columns: {sorted(missing)}"
            )
        combos: Dict[str, Dict[str, str]] = {}
        for row in df.to_dict(orient="records"):
            combo = str(row.get("combo", "")).strip()
            combos[combo] = {
                "1A": str(row.get("1A", "")).strip(),
                "1B": str(row.get("1B", "")).strip(),
                "1D": str(row.get("1D", "")).strip(),
                "1E": str(row.get("1E", "")).strip(),
                "1G": str(row.get("1G", "")).strip(),
                "1I": str(row.get("1I", "")).strip(),
                "1K": str(row.get("1K", "")).strip(),
                "1L": str(row.get("1L", "")).strip(),
            }
        return combos

    def _load_knockout_matches(self) -> pd.DataFrame:
        if not KNOCKOUT_MATCHES_PATH.exists():
            raise FileNotFoundError(
                f"Missing knockout matches file: {KNOCKOUT_MATCHES_PATH}"
            )
        df = pd.read_csv(KNOCKOUT_MATCHES_PATH)
        required = {"match_id", "stage", "date", "home", "away", "city", "country"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Knockout matches file missing columns: {sorted(missing)}"
            )
        df["match_id"] = pd.to_numeric(df["match_id"], errors="raise").astype(int)
        df["stage"] = df["stage"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df["home"] = df["home"].astype(str).str.strip()
        df["away"] = df["away"].astype(str).str.strip()
        df["stadium"] = df["stadium"].astype(str).str.strip() if "stadium" in df.columns else ""
        df["city"] = df["city"].astype(str).str.strip()
        df["country"] = df["country"].astype(str).str.strip()
        if df["match_id"].duplicated().any():
            dupes = df.loc[df["match_id"].duplicated(), "match_id"].unique().tolist()
            raise ValueError(
                "Knockout matches file contains duplicate match_id values: "
                f"{sorted(dupes)}"
            )
        return df.sort_values("match_id")

    def _confederation_map(self, year: int = 2026) -> Dict[str, str]:
        conf = pd.read_csv(CONFEDERATIONS_PATH)
        conf["team"] = conf["team"].astype(str).str.strip()
        conf["confederation"] = conf["confederation"].astype(str).str.strip()
        conf["start_year"] = pd.to_numeric(conf["start_year"], errors="coerce")
        conf["end_year"] = pd.to_numeric(conf["end_year"], errors="coerce")

        mapping: Dict[str, str] = {}
        for team, group in conf.groupby("team"):
            in_window = group[
                (group["start_year"].isna() | (group["start_year"] <= year))
                & (group["end_year"].isna() | (group["end_year"] >= year))
            ]
            if in_window.empty:
                in_window = group
            in_window = in_window.sort_values(
                ["start_year", "end_year"], ascending=[False, True]
            )
            mapping[team] = str(in_window.iloc[0]["confederation"])
        return mapping

    def _make_groups(
        self,
        teams: List[str],
        group_count: int,
        rng: np.random.Generator,
        prefix: str,
    ) -> Dict[str, List[str]]:
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        groups = {f"{prefix}{i+1}": [] for i in range(group_count)}
        shuffled = list(teams)
        rng.shuffle(shuffled)
        keys = list(groups.keys())
        for idx, team in enumerate(shuffled):
            groups[keys[idx % group_count]].append(team)
        return groups

    def _simulate_group_round_robin(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        groups: Dict[str, List[str]],
        day: int,
        stage: str,
        neutral_override: bool = True,
    ) -> Tuple[List[MatchResult], Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        matches: List[MatchResult] = []
        tables: Dict[str, pd.DataFrame] = {}
        rankings: Dict[str, List[str]] = {}

        for group, teams in groups.items():
            table = pd.DataFrame(
                index=teams,
                columns=["points", "gf", "ga", "gd", "w", "d", "l"],
                data=0,
            )
            group_matches: List[MatchResult] = []
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    home = teams[i]
                    away = teams[j]
                    res = self._simulate_match(
                        model,
                        rng,
                        states,
                        day=day,
                        match_date=self._day_to_date(day),
                        home_team=home,
                        away_team=away,
                        stage=stage,
                        group=group,
                        allow_draw=True,
                        neutral_override=neutral_override,
                        stadium=None,
                        city=None,
                        country=None,
                    )
                    matches.append(res)
                    group_matches.append(res)
                    hs = res.home_score
                    as_ = res.away_score
                    table.loc[home, "gf"] += hs
                    table.loc[home, "ga"] += as_
                    table.loc[away, "gf"] += as_
                    table.loc[away, "ga"] += hs
                    if hs > as_:
                        table.loc[home, "points"] += 3
                        table.loc[home, "w"] += 1
                        table.loc[away, "l"] += 1
                    elif hs < as_:
                        table.loc[away, "points"] += 3
                        table.loc[away, "w"] += 1
                        table.loc[home, "l"] += 1
                    else:
                        table.loc[home, "points"] += 1
                        table.loc[away, "points"] += 1
                        table.loc[home, "d"] += 1
                        table.loc[away, "d"] += 1
            table["gd"] = table["gf"] - table["ga"]
            tables[group] = table
            rankings[group] = self._rank_group(teams, group_matches, table, rng)

        return matches, tables, rankings

    def _group_entries(
        self, tables: Dict[str, pd.DataFrame], rankings: Dict[str, List[str]]
    ) -> List[Dict]:
        entries: List[Dict] = []
        for group, ranking in rankings.items():
            table = tables[group]
            for pos, team in enumerate(ranking, start=1):
                entries.append(
                    {
                        "team": team,
                        "group": group,
                        "position": pos,
                        "points": float(table.loc[team, "points"]),
                        "gd": float(table.loc[team, "gd"]),
                        "gf": float(table.loc[team, "gf"]),
                    }
                )
        return entries

    def _simulate_playoff(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        teams: List[str],
        winners_needed: int,
        day: int,
        stage: str,
    ) -> Tuple[List[str], List[MatchResult], int]:
        if winners_needed <= 0 or not teams:
            return [], [], day
        participants = list(teams)
        matches: List[MatchResult] = []
        current_day = day

        while len(participants) > winners_needed:
            rng.shuffle(participants)
            next_round: List[str] = []
            if len(participants) % 2 == 1:
                next_round.append(participants.pop())
            for i in range(0, len(participants), 2):
                home, away = participants[i], participants[i + 1]
                res = self._simulate_match(
                    model,
                    rng,
                    states,
                    day=current_day,
                    match_date=self._day_to_date(current_day),
                    home_team=home,
                    away_team=away,
                    stage=stage,
                    group=None,
                    allow_draw=False,
                    neutral_override=True,
                    stadium=None,
                    city=None,
                    country=None,
                )
                matches.append(res)
                next_round.append(res.winner)
            participants = next_round
            current_day += QUALIFICATION_STAGE_DAYS

        return participants[:winners_needed], matches, current_day

    def _simulate_confed_qualification(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        confed: str,
        teams: List[str],
        qualified: List[str],
        confed_map: Dict[str, str],
        day: int,
        fixed_interconfed_candidates: Optional[List[str]] = None,
        fixed_confed_playoff_candidates: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], List[MatchResult], int, Dict[str, str]]:
        slot_info = QUALIFICATION_SLOTS[confed]
        format_info = QUALIFICATION_FORMAT[confed]
        direct_total = int(slot_info["direct"])
        playoff_slots = int(slot_info["interconfed_playoff"])
        fixed_interconfed_candidates = list(dict.fromkeys(fixed_interconfed_candidates or []))
        fixed_confed_playoff_candidates = list(
            dict.fromkeys(fixed_confed_playoff_candidates or [])
        )
        playoff_slots_remaining = max(0, playoff_slots - len(fixed_interconfed_candidates))
        confed_playoff_direct = int(format_info.get("confed_playoff_direct", 0))
        group_count = int(format_info["groups"])
        if group_count > len(teams):
            group_count = len(teams)

        qualified_confed = [
            t for t in qualified if confed_map.get(t) == confed
        ]
        direct_remaining = max(0, direct_total - len(qualified_confed))
        if direct_remaining == 0 and playoff_slots_remaining == 0 and confed_playoff_direct == 0:
            return [], list(fixed_interconfed_candidates), [], day, {}
        if not teams and confed_playoff_direct > 0 and fixed_confed_playoff_candidates:
            if direct_remaining != confed_playoff_direct:
                raise ValueError(
                    f"{confed} playoff candidates provided but {direct_remaining} direct slots remain"
                )
            winners, playoff_matches, day = self._simulate_playoff(
                model,
                rng,
                states,
                fixed_confed_playoff_candidates,
                winners_needed=confed_playoff_direct,
                day=day,
                stage=f"{confed} Playoff",
            )
            slot_winners: Dict[str, str] = {}
            if confed == "UEFA":
                path_letters = ["A", "B", "C", "D"]
                if len(winners) != len(path_letters):
                    raise ValueError("UEFA playoff did not produce 4 winners")
                for letter, team in zip(path_letters, winners):
                    slot_winners[f"UEFA Path {letter} winner"] = team
            return winners, list(fixed_interconfed_candidates), playoff_matches, day, slot_winners
        if not teams:
            raise ValueError(
                f"{confed} has remaining slots but no remaining teams. "
                "Update qualified teams or playoff candidates."
            )

        groups = self._make_groups(teams, group_count, rng, prefix=f"{confed}-G")
        matches, tables, rankings = self._simulate_group_round_robin(
            model,
            rng,
            states,
            groups,
            day=day,
            stage=f"{confed} Qualifying",
            neutral_override=True,
        )
        day += QUALIFICATION_STAGE_DAYS
        entries = self._group_entries(tables, rankings)
        entries_sorted = sorted(
            entries,
            key=lambda e: (
                e["position"],
                -e["points"],
                -e["gd"],
                -e["gf"],
                rng.random(),
            ),
        )

        group_direct_slots = int(format_info.get("group_direct_slots", direct_remaining))
        if confed_playoff_direct > 0:
            group_direct_slots = max(0, direct_remaining - confed_playoff_direct)
        group_direct_slots = min(group_direct_slots, direct_remaining)
        direct_entries: List[Dict] = []
        for entry in entries_sorted:
            if len(direct_entries) >= group_direct_slots:
                break
            direct_entries.append(entry)
        direct_teams = [e["team"] for e in direct_entries]
        slot_winners: Dict[str, str] = {}

        remaining_entries = [
            e for e in entries_sorted if e["team"] not in direct_teams
        ]

        direct_needed = max(0, direct_remaining - len(direct_teams))
        confed_playoff_needed = min(confed_playoff_direct, direct_needed)
        if confed_playoff_needed > 0 and remaining_entries:
            candidate_count = max(
                format_info.get("playoff_candidates", 0), confed_playoff_needed * 2
            )
            candidates = remaining_entries[:candidate_count]
            playoff_teams = [c["team"] for c in candidates]
            winners, playoff_matches, day = self._simulate_playoff(
                model,
                rng,
                states,
                playoff_teams,
                winners_needed=confed_playoff_needed,
                day=day,
                stage=f"{confed} Playoff",
            )
            matches.extend(playoff_matches)
            direct_teams.extend(winners)
            if confed == "UEFA":
                path_letters = ["A", "B", "C", "D"]
                if len(winners) != len(path_letters):
                    raise ValueError("UEFA playoff did not produce 4 winners")
                for letter, team in zip(path_letters, winners):
                    slot_winners[f"UEFA Path {letter} winner"] = team
            remaining_entries = [
                e for e in remaining_entries if e["team"] not in winners
            ]
            direct_needed = max(0, direct_remaining - len(direct_teams))

        if direct_needed > 0 and remaining_entries:
            extra_direct = [e["team"] for e in remaining_entries[:direct_needed]]
            direct_teams.extend(extra_direct)
            remaining_entries = [
                e for e in remaining_entries if e["team"] not in extra_direct
            ]

        interconfed_candidates: List[str] = []
        if fixed_interconfed_candidates:
            interconfed_candidates.extend(fixed_interconfed_candidates)
        if playoff_slots_remaining > 0 and remaining_entries:
            candidate_count = max(
                format_info.get("playoff_candidates", 0), playoff_slots_remaining * 2
            )
            candidates = remaining_entries[:candidate_count]
            playoff_teams = [c["team"] for c in candidates]
            winners, playoff_matches, day = self._simulate_playoff(
                model,
                rng,
                states,
                playoff_teams,
                winners_needed=playoff_slots_remaining,
                day=day,
                stage=f"{confed} Playoff (Interconfed)",
            )
            matches.extend(playoff_matches)
            interconfed_candidates.extend(winners)

        return direct_teams, interconfed_candidates, matches, day, slot_winners

    def _simulate_interconfed_playoff(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        candidates: List[str],
        day: int,
    ) -> Tuple[List[str], List[MatchResult], int]:
        if len(candidates) <= 2:
            return candidates, [], day
        matches: List[MatchResult] = []
        ordered = sorted(
            candidates, key=lambda t: self._team_strength(states[t]), reverse=True
        )
        if len(ordered) == 6:
            seeds = ordered[:2]
            others = ordered[2:]
            winners_round1, playoff_matches, day = self._simulate_playoff(
                model,
                rng,
                states,
                others,
                winners_needed=2,
                day=day,
                stage="Interconfed Playoff",
            )
            matches.extend(playoff_matches)
            finalists = seeds + winners_round1
            winners, playoff_matches, day = self._simulate_playoff(
                model,
                rng,
                states,
                finalists,
                winners_needed=2,
                day=day,
                stage="Interconfed Playoff Final",
            )
            matches.extend(playoff_matches)
            return winners, matches, day

        winners, playoff_matches, day = self._simulate_playoff(
            model,
            rng,
            states,
            ordered,
            winners_needed=2,
            day=day,
            stage="Interconfed Playoff",
        )
        matches.extend(playoff_matches)
        return winners, matches, day

    def _simulate_remaining_qualification(
        self,
        model: Model,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, TeamSimState], List[str], Dict[str, str]]:
        qualified_df = self._load_qualified_teams()
        qualified = qualified_df["team"].tolist()
        self.qualification_paths = {
            row.team: f"Qualified: {row.method}" for row in qualified_df.itertuples()
        }
        missing_qualified = [t for t in qualified if t not in model.teams]
        if missing_qualified:
            raise ValueError(
                "Qualified teams missing from model: "
                f"{missing_qualified[:10]}{'...' if len(missing_qualified) > 10 else ''}"
            )

        remaining = self._load_remaining_qualifiers(model)
        if remaining.empty:
            raise ValueError("Remaining qualifiers file is empty")

        remaining["day"] = remaining["date"].apply(self._date_to_day)
        base_day = min(self.start_day, int(remaining["day"].min()))

        team_pool = set(qualified)
        for team in remaining["home_team"].tolist() + remaining["away_team"].tolist():
            if team:
                team_pool.add(team)

        states = self._init_team_states(
            model, rng, teams=sorted(team_pool), start_day=base_day
        )
        matches: List[MatchResult] = []
        slot_winners: Dict[str, str] = {}

        round_order = {"semi1": 1, "semi2": 2, "semi": 1, "final": 3}

        for path, path_df in remaining.groupby("path"):
            path_df = path_df.copy()
            path_df["_order"] = path_df["round"].map(round_order).fillna(99)
            path_df = path_df.sort_values(["day", "_order"])
            winners: Dict[str, str] = {}

            for row in path_df.itertuples(index=False):
                home_team = row.home_team or ""
                away_team = row.away_team or ""
                if not home_team and row.home_source:
                    if row.home_source not in winners:
                        raise ValueError(
                            f"Missing winner for {path} {row.home_source}"
                        )
                    home_team = winners[row.home_source]
                if not away_team and row.away_source:
                    if row.away_source not in winners:
                        raise ValueError(
                            f"Missing winner for {path} {row.away_source}"
                        )
                    away_team = winners[row.away_source]
                if not home_team or not away_team:
                    raise ValueError(
                        f"Unresolved matchup for {path} {row.round}: "
                        f"{home_team} vs {away_team}"
                    )

                day = int(row.day)
                res = self._simulate_match(
                    model,
                    rng,
                    states,
                    day=day,
                    match_date=pd.Timestamp(row.date),
                    home_team=home_team,
                    away_team=away_team,
                    stage=f"{row.stage} {row.round}".strip(),
                    group=None,
                    allow_draw=False,
                    neutral_override=bool(row.neutral),
                    stadium=None,
                    city=None,
                    country=None,
                )
                matches.append(res)
                winners[row.round] = res.winner

                if row.round == "final":
                    slot = row.winner_slot or f"{path} winner"
                    slot_winners[slot] = res.winner
                    qualified.append(res.winner)
                    self.qualification_paths[res.winner] = slot

        qualified_unique = list(dict.fromkeys(qualified))
        if len(qualified_unique) != 48:
            raise ValueError(
                f"Qualification produced {len(qualified_unique)} teams (expected 48)"
            )

        self.qualification_matches = matches
        return states, qualified_unique, slot_winners

    def _simulate_knockout_official(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        group_rankings: Dict[str, List[str]],
        best_third: List[Dict],
    ) -> List[MatchResult]:
        matches_df = self._load_knockout_matches()
        combos = self._load_round_of_32_combinations()

        third_place_by_group = {
            group: ranking[2] for group, ranking in group_rankings.items()
        }
        third_place_groups = sorted([entry["group"] for entry in best_third])
        combo_key = "".join(third_place_groups)
        if combo_key not in combos:
            raise ValueError(
                f"Missing round-of-32 combo mapping for groups: {combo_key}"
            )
        third_place_assignments = combos[combo_key]

        def resolve_group_placeholder(label: str) -> str:
            if label.startswith("Winner Group "):
                group = label.replace("Winner Group ", "").strip()
                return group_rankings[group][0]
            if label.startswith("Runner-up Group "):
                group = label.replace("Runner-up Group ", "").strip()
                return group_rankings[group][1]
            if label.startswith("3rd Group "):
                group = label.replace("3rd Group ", "").strip()
                if len(group) == 1:
                    return third_place_by_group[group]
            raise ValueError(f"Unrecognized group placeholder: {label}")

        results: List[MatchResult] = []
        results_by_match: Dict[int, MatchResult] = {}

        for row in matches_df.itertuples(index=False):
            home_label = str(row.home).strip()
            away_label = str(row.away).strip()

            def resolve_label(label: str, opponent_label: str) -> str:
                if label.startswith("Winner Match "):
                    match_id = int(label.replace("Winner Match ", "").strip())
                    if match_id not in results_by_match:
                        raise ValueError(f"Missing result for Match {match_id}")
                    return results_by_match[match_id].winner
                if label.startswith("Loser Match "):
                    match_id = int(label.replace("Loser Match ", "").strip())
                    if match_id not in results_by_match:
                        raise ValueError(f"Missing result for Match {match_id}")
                    res = results_by_match[match_id]
                    return res.away_team if res.winner == res.home_team else res.home_team
                if label.startswith("3rd Group ") and opponent_label.startswith("Winner Group "):
                    winner_group = opponent_label.replace("Winner Group ", "").strip()
                    key = f"1{winner_group}"
                    if key not in third_place_assignments:
                        raise ValueError(f"Missing third-place assignment for {key}")
                    third_group = third_place_assignments[key]
                    return third_place_by_group[third_group]
                return resolve_group_placeholder(label)

            home_team = resolve_label(home_label, away_label)
            away_team = resolve_label(away_label, home_label)

            day = self._date_to_day(row.date)
            res = self._simulate_match(
                model,
                rng,
                states,
                day=day,
                match_date=pd.Timestamp(row.date),
                home_team=home_team,
                away_team=away_team,
                stage=str(row.stage).strip(),
                group=None,
                allow_draw=False,
                neutral_override=None,
                stadium=getattr(row, "stadium", "") or "",
                city=getattr(row, "city", "") or "",
                country=getattr(row, "country", "") or "",
            )
            results.append(res)
            results_by_match[int(row.match_id)] = res

        final_match = results_by_match.get(104)
        if final_match:
            self.champion = final_match.winner

        return results

    def _simulate_qualification(
        self,
        model: Model,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, TeamSimState], List[str], Dict[str, str]]:
        return self._simulate_remaining_qualification(model, rng)

    def _group_stage_schedule(
        self, model: Model
    ) -> List[Tuple[int, pd.Timestamp, str, str, str, str, str, str]]:
        df = self._load_group_matches(model)
        slot_winners = self.slot_winners or {}

        def resolve_team(team: str) -> str:
            if self._is_slot_placeholder(team):
                if team not in slot_winners:
                    raise ValueError(f"Group schedule includes unresolved slot: {team}")
                return slot_winners[team]
            return team

        df["home_team"] = df["home_team"].apply(resolve_team)
        df["away_team"] = df["away_team"].apply(resolve_team)

        for group, sub in df.groupby("group"):
            group = str(group).strip()
            if group not in self.GROUPS:
                raise ValueError(f"Invalid group in schedule: {group}")
            expected = set(self.groups.get(group, []))
            if expected:
                teams_in_schedule = set(sub["home_team"]).union(sub["away_team"])
                missing = sorted(teams_in_schedule.difference(expected))
                if missing:
                    raise ValueError(
                        f"Group schedule includes team not in group {group}: {missing}"
                    )

        df["day"] = df["date"].apply(self._date_to_day)
        sort_cols = ["date"]
        if "match_id" in df.columns:
            sort_cols.append("match_id")
        df = df.sort_values(sort_cols)

        matches: List[Tuple[int, pd.Timestamp, str, str, str, str, str, str]] = []
        for row in df.itertuples(index=False):
            matches.append(
                (
                    int(row.day),
                    row.date,
                    str(row.group),
                    row.home_team,
                    row.away_team,
                    getattr(row, "stadium", "") or "",
                    getattr(row, "city", "") or "",
                    getattr(row, "country", "") or "",
                )
            )
        return matches

    def _simulate_group_stage(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        matches: List[Tuple[int, pd.Timestamp, str, str, str, str, str, str]],
    ) -> List[MatchResult]:
        results: List[MatchResult] = []
        for day, match_date, group, home, away, stadium, city, country in matches:
            res = self._simulate_match(
                model,
                rng,
                states,
                day=day,
                match_date=match_date,
                home_team=home,
                away_team=away,
                stage="Group",
                group=group,
                allow_draw=True,
                stadium=stadium,
                city=city,
                country=country,
            )
            results.append(res)
        return results

    def _group_tables(
        self,
        group_results: List[MatchResult],
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        tables: Dict[str, pd.DataFrame] = {}
        rankings: Dict[str, List[str]] = {}
        results_by_group: Dict[str, List[MatchResult]] = {}
        for res in group_results:
            results_by_group.setdefault(res.group or "", []).append(res)

        for group, matches in results_by_group.items():
            teams = self.groups[group]
            table = pd.DataFrame(
                index=teams,
                columns=["points", "gf", "ga", "gd", "w", "d", "l"],
                data=0,
            )
            for m in matches:
                home = m.home_team
                away = m.away_team
                hs = m.home_score
                as_ = m.away_score
                table.loc[home, "gf"] += hs
                table.loc[home, "ga"] += as_
                table.loc[away, "gf"] += as_
                table.loc[away, "ga"] += hs
                if hs > as_:
                    table.loc[home, "points"] += 3
                    table.loc[home, "w"] += 1
                    table.loc[away, "l"] += 1
                elif hs < as_:
                    table.loc[away, "points"] += 3
                    table.loc[away, "w"] += 1
                    table.loc[home, "l"] += 1
                else:
                    table.loc[home, "points"] += 1
                    table.loc[away, "points"] += 1
                    table.loc[home, "d"] += 1
                    table.loc[away, "d"] += 1
            table["gd"] = table["gf"] - table["ga"]
            tables[group] = table
            rankings[group] = self._rank_group(teams, matches, table, rng)
        return tables, rankings

    def _rank_group(
        self,
        teams: List[str],
        matches: List[MatchResult],
        table: pd.DataFrame,
        rng: np.random.Generator,
    ) -> List[str]:
        base = table.loc[teams, ["points", "gd", "gf"]].copy()
        base["team"] = base.index
        base = base.sort_values(by=["points"], ascending=[False])

        ranked: List[str] = []
        i = 0
        teams_list = list(base["team"].tolist())
        while i < len(teams_list):
            current = teams_list[i]
            tied = [current]
            i += 1
            while i < len(teams_list):
                nxt = teams_list[i]
                if base.loc[current, "points"] == base.loc[nxt, "points"]:
                    tied.append(nxt)
                    i += 1
                else:
                    break
            if len(tied) == 1:
                ranked.append(tied[0])
                continue
            ranked.extend(self._head_to_head_rank(tied, matches, table, rng))
        return ranked

    def _head_to_head_rank(
        self,
        tied: List[str],
        matches: List[MatchResult],
        overall_table: pd.DataFrame,
        rng: np.random.Generator,
    ) -> List[str]:
        h2h_table = pd.DataFrame(
            index=tied, columns=["points", "gf", "ga", "gd"], data=0
        )
        for m in matches:
            if m.home_team not in tied or m.away_team not in tied:
                continue
            hs = m.home_score
            as_ = m.away_score
            h2h_table.loc[m.home_team, "gf"] += hs
            h2h_table.loc[m.home_team, "ga"] += as_
            h2h_table.loc[m.away_team, "gf"] += as_
            h2h_table.loc[m.away_team, "ga"] += hs
            if hs > as_:
                h2h_table.loc[m.home_team, "points"] += 3
            elif hs < as_:
                h2h_table.loc[m.away_team, "points"] += 3
            else:
                h2h_table.loc[m.home_team, "points"] += 1
                h2h_table.loc[m.away_team, "points"] += 1
        h2h_table["gd"] = h2h_table["gf"] - h2h_table["ga"]
        combined = pd.DataFrame(
            {
                "h2h_points": h2h_table["points"],
                "h2h_gd": h2h_table["gd"],
                "h2h_gf": h2h_table["gf"],
                "overall_gd": h2h_table.index.map(
                    lambda t: float(overall_table.loc[t, "gd"])
                ),
                "overall_gf": h2h_table.index.map(
                    lambda t: float(overall_table.loc[t, "gf"])
                ),
            }
        )
        sort_cols = ["h2h_points", "h2h_gd", "h2h_gf", "overall_gd", "overall_gf"]
        ranked = combined.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        ranked_list = ranked.index.tolist()

        # Drawing of lots for any remaining ties (fair play not modeled).
        final_order: List[str] = []
        i = 0
        while i < len(ranked_list):
            cur = ranked_list[i]
            tied_block = [cur]
            i += 1
            while i < len(ranked_list):
                nxt = ranked_list[i]
                if all(ranked.loc[cur, col] == ranked.loc[nxt, col] for col in sort_cols):
                    tied_block.append(nxt)
                    i += 1
                else:
                    break
            if len(tied_block) > 1:
                rng.shuffle(tied_block)
            final_order.extend(tied_block)
        return final_order

    def _select_qualifiers(
        self, group_rankings: Dict[str, List[str]], rng: np.random.Generator
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        third_place = []
        qualifiers = []
        for group, ranking in group_rankings.items():
            table = self.group_tables[group]
            for idx, team in enumerate(ranking, start=1):
                entry = {
                    "team": team,
                    "group": group,
                    "position": idx,
                    "points": float(table.loc[team, "points"]),
                    "gd": float(table.loc[team, "gd"]),
                    "gf": float(table.loc[team, "gf"]),
                }
                if idx <= 2:
                    qualifiers.append(entry)
                elif idx == 3:
                    third_place.append(entry)

        third_place.sort(key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True)

        # Drawing of lots for any remaining ties (fair play not modeled).
        sorted_third: List[Dict] = []
        i = 0
        while i < len(third_place):
            cur = third_place[i]
            tied_block = [cur]
            i += 1
            while i < len(third_place):
                nxt = third_place[i]
                if (
                    cur["points"] == nxt["points"]
                    and cur["gd"] == nxt["gd"]
                    and cur["gf"] == nxt["gf"]
                ):
                    tied_block.append(nxt)
                    i += 1
                else:
                    break
            if len(tied_block) > 1:
                rng.shuffle(tied_block)
            sorted_third.extend(tied_block)

        third_place = sorted_third
        best_third = third_place[:8]
        qualifiers.extend(best_third)
        return qualifiers, third_place, best_third

    def _build_round_of_32(
        self, qualifiers: List[Dict], rng: np.random.Generator
    ) -> List[Tuple[str, str, int]]:
        def seed_key(entry: Dict) -> Tuple[int, float, float, float, float]:
            pos_rank = 0 if entry["position"] == 1 else 1 if entry["position"] == 2 else 2
            return (pos_rank, entry["points"], entry["gd"], entry["gf"], rng.random())

        seeds = sorted(qualifiers, key=seed_key, reverse=True)
        pairs = []
        for i in range(16):
            pairs.append([seeds[i], seeds[-1 - i]])

        for i, (a, b) in enumerate(pairs):
            if a["group"] != b["group"]:
                continue
            swap_done = False
            for j in range(i + 1, len(pairs)):
                c, d = pairs[j]
                if a["group"] != d["group"] and c["group"] != b["group"]:
                    pairs[i][1], pairs[j][1] = d, b
                    swap_done = True
                    break
            if not swap_done:
                for j in range(i + 1, len(pairs)):
                    c, d = pairs[j]
                    if a["group"] != c["group"] and d["group"] != b["group"]:
                        pairs[i][1], pairs[j][0] = c, b
                        break

        day = self.start_day + self.ROUND_DAY_OFFSETS["Round of 32"]
        return [(p[0]["team"], p[1]["team"], day) for p in pairs]

    def _simulate_knockout(
        self,
        model: Model,
        rng: np.random.Generator,
        states: Dict[str, TeamSimState],
        round32: List[Tuple[str, str, int]],
    ) -> List[MatchResult]:
        results: List[MatchResult] = []

        def play_round(round_name: str, pairs: List[Tuple[str, str, int]]) -> List[str]:
            winners: List[str] = []
            for home, away, day in pairs:
                res = self._simulate_match(
                    model,
                    rng,
                    states,
                    day=day,
                    match_date=self._day_to_date(day),
                    home_team=home,
                    away_team=away,
                    stage=round_name,
                    group=None,
                    allow_draw=False,
                    stadium=None,
                    city=None,
                    country=None,
                )
                results.append(res)
                winners.append(res.winner)
            return winners

        winners32 = play_round("Round of 32", round32)

        def make_pairs(teams: List[str], round_name: str) -> List[Tuple[str, str, int]]:
            day = self.start_day + self.ROUND_DAY_OFFSETS[round_name]
            return [
                (teams[i], teams[i + 1], day) for i in range(0, len(teams), 2)
            ]

        winners16 = play_round("Round of 16", make_pairs(winners32, "Round of 16"))
        winners8 = play_round("Quarterfinal", make_pairs(winners16, "Quarterfinal"))
        winners4 = play_round("Semifinal", make_pairs(winners8, "Semifinal"))

        # Third place
        semi_results = [r for r in results if r.stage == "Semifinal"]
        if len(semi_results) == 2:
            losers = [
                r.away_team if r.winner == r.home_team else r.home_team
                for r in semi_results
            ]
            third_place_pair = make_pairs(losers, "Third place")
            play_round("Third place", third_place_pair)

        # Final
        final_pair = make_pairs(winners4, "Final")
        final_winners = play_round("Final", final_pair)
        if final_winners:
            self.champion = final_winners[0]

        return results


def get_results(team, second_team=None, res=None, start_date=None, end_date = None):
    assert res is not None
    df_ = res.query(f"home_team == '{team}' or away_team == '{team}'")
    if second_team is not None:
        df_ = df_.query(f"home_team == '{second_team}' or away_team == '{second_team}'")
    if start_date is not None:
        df_ = df_.loc[df_.date >= start_date]
    if end_date is not None:
        df_ = df_.loc[df_.date <= end_date]
    return df_

def stage_of_elimination(team, T):
    e = []
    for t in T:
        stage = get_results(team, res=t.results_frame()).iloc[-1].stage
        if stage == "Final" and t.champion == team:
            stage = "Champion"
        elif stage == "Third place":
            w = t.results_frame().query("stage == 'Third place'").winner.values[0]
            if w == team:
                stage = "Third place"
            else:
                stage = "Fourth place"
        e.append({
            "Group": "1. Group",
            "Round of 32": "2. Round of 32",
            "Round of 16": "3. Round of 16",
            "Quarterfinal": "4. Quarterfinal",
            "Fourth place": "5. Fourth place",
            "Third place": "6. Third place",
            "Final": "7. Final",
            "Champion": "8. Champion",
        }.get(stage, "0." + stage))
    return pd.Series(e).value_counts().sort_index()
