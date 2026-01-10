import math
from typing import Callable, Optional

import numpy as np
import pandas as pd


EPSILON = 1e-10


def _elo_win_prob(diff: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def _default_draw_prob(diff: float) -> float:
    return 0.25 * math.exp(-abs(diff) / 400.0)


class TeamState:
    def __init__(self, elo: float, last_update, maintain_param_history: bool = False):
        self.elo = float(elo)
        self.last_update = last_update
        self._do_maintain_param_history = maintain_param_history
        if maintain_param_history:
            self.history = {}

    def update(self, elo: float, timestamp):
        self.elo = float(elo)
        self.last_update = timestamp
        if self._do_maintain_param_history:
            self.history[timestamp] = {"elo": float(elo)}


class EloModel:
    def __init__(
        self,
        k_factor: float = 20.0,
        base_elo: float = 1500.0,
        elo_scale: float = 400.0,
        quality_prior_decay: float = 0.0,
        hga: float = 0.0,
        hga_rw_var_per_year: float = 0.0,
        hga_prior_var: float = 1.0,
        hga_override: Optional[float] = None,
        hga_step_scale: float = 0.1,
        hga_max_step: Optional[float] = 100.0,
        hga_prob_floor: Optional[float] = 1e-6,
        draw_prob_fn: Optional[Callable[[float], float]] = None,
        draw_deriv_eps: float = 1e-4,
        friendly_weight: float = 1.0,
        maintain_param_history: bool = False,
    ):
        """
        k_factor: Elo K factor (step size).
        base_elo: Baseline Elo for new teams.
        elo_scale: Scale for Elo win probability (usually 400).
        quality_prior_decay: Decrease in initial Elo per year after first match year.
        hga: Home ground advantage (Elo points).
        hga_rw_var_per_year: Random-walk variance per year for HGA drift.
        hga_prior_var: Prior variance for HGA at the first year.
        hga_override: Fixed HGA override (disables learning).
        hga_step_scale: Damping factor for annual HGA updates.
        hga_max_step: Max absolute HGA change per year (None disables clamp).
        hga_prob_floor: Probability floor for HGA updates (None disables floor).
        draw_prob_fn: Function mapping Elo diff (incl. HGA) to draw probability.
        draw_deriv_eps: Finite-difference step for draw_prob_fn derivative.
        friendly_weight: Match weight multiplier when tournament == "Friendly".
        maintain_param_history: Store per-team Elo history.
        """
        self.k_factor = float(k_factor)
        self.base_elo = float(base_elo)
        self.elo_scale = float(elo_scale)
        self.quality_prior_decay = float(quality_prior_decay)

        self._hga_fixed = hga_override is not None
        self.hga = float(hga_override) if self._hga_fixed else float(hga)
        self.hga_rw_var_per_year = float(hga_rw_var_per_year)
        self.hga_prior_var = float(hga_prior_var)
        self._hga_var = float(hga_prior_var)
        self._hga_year = None
        self._hga_grad_sum = 0.0
        self._hga_curv_sum = 0.0
        self._hga_step_last = None
        self._hga_prior_var_last = None
        self._hga_post_var_last = None
        self._hga_grad_last = None
        self._hga_curv_last = None
        self._hga_history = {}
        self.hga_step_scale = float(hga_step_scale)
        self.hga_max_step = None if hga_max_step is None else float(hga_max_step)
        if self.hga_step_scale < 0.0:
            raise ValueError("hga_step_scale must be non-negative")
        if self.hga_max_step is not None and self.hga_max_step <= 0.0:
            raise ValueError("hga_max_step must be positive or None")

        if hga_prob_floor is None:
            self.hga_prob_floor = 0.0
        else:
            self.hga_prob_floor = float(hga_prob_floor)
        if self.hga_prob_floor < 0.0:
            raise ValueError("hga_prob_floor must be non-negative or None")

        self.draw_prob_fn = draw_prob_fn or _default_draw_prob
        self.draw_deriv_eps = float(draw_deriv_eps)
        self.friendly_weight = float(friendly_weight)
        if self.friendly_weight < 0.0:
            raise ValueError("friendly_weight must be non-negative")

        self.maintain_param_history = maintain_param_history
        self.teams = {}
        self.is_fit = False
        self._first_match_year = None

    def _get_or_init_team(self, team: str, t: pd.Timestamp) -> TeamState:
        st = self.teams.get(team)
        if st is not None:
            return st
        year_delta = 0
        if self._first_match_year is not None:
            year_delta = int(t.year) - int(self._first_match_year)
        init_elo = self.base_elo - float(year_delta) * self.quality_prior_decay
        st = TeamState(
            elo=init_elo,
            last_update=t,
            maintain_param_history=self.maintain_param_history,
        )
        self.teams[team] = st
        return st

    def _draw_prob_and_deriv(self, diff: float) -> tuple[float, float]:
        p_raw = float(self.draw_prob_fn(diff))
        eps = self.draw_deriv_eps
        if eps <= 0.0:
            dp = 0.0
        else:
            p_plus = float(self.draw_prob_fn(diff + eps))
            p_minus = float(self.draw_prob_fn(diff - eps))
            dp = (p_plus - p_minus) / (2.0 * eps)
        p = min(max(p_raw, 0.0), 1.0)
        if p != p_raw:
            dp = 0.0
        return p, float(dp)

    def _match_probs(self, diff: float):
        raw_p = _elo_win_prob(diff, self.elo_scale)
        raw_p = min(max(raw_p, EPSILON), 1.0 - EPSILON)

        p_draw, dp_draw = self._draw_prob_and_deriv(diff)
        max_draw = max(0.0, 2.0 * min(raw_p, 1.0 - raw_p))
        if p_draw > max_draw:
            p_draw = max_draw
            dp_draw = 0.0
        if p_draw < 0.0:
            p_draw = 0.0
            dp_draw = 0.0

        p_home = raw_p - 0.5 * p_draw
        p_away = 1.0 - p_draw - p_home

        p_home = min(max(p_home, EPSILON), 1.0 - EPSILON)
        p_draw = min(max(p_draw, EPSILON), 1.0 - EPSILON)
        p_away = min(max(p_away, EPSILON), 1.0 - EPSILON)

        raw_p_prime = (math.log(10.0) / self.elo_scale) * raw_p * (1.0 - raw_p)
        dp_home = raw_p_prime - 0.5 * dp_draw
        dp_away = -raw_p_prime - 0.5 * dp_draw

        return raw_p, p_home, p_draw, p_away, dp_home, dp_draw, dp_away

    def _accumulate_hga_stats(self, diff: float, outcome: str, match_weight: float):
        if self.hga_rw_var_per_year <= 0.0 or self._hga_fixed:
            return
        if match_weight <= 0.0:
            return
        _, p_home, p_draw, p_away, dp_home, dp_draw, dp_away = self._match_probs(diff)

        prob_floor = self.hga_prob_floor
        if prob_floor > 0.0:
            p_home_safe = max(p_home, prob_floor)
            p_draw_safe = max(p_draw, prob_floor)
            p_away_safe = max(p_away, prob_floor)
        else:
            p_home_safe = p_home
            p_draw_safe = p_draw
            p_away_safe = p_away

        if outcome == "home":
            dlogp_ddiff = dp_home / p_home_safe
        elif outcome == "away":
            dlogp_ddiff = dp_away / p_away_safe
        else:
            dlogp_ddiff = dp_draw / p_draw_safe

        grad = -dlogp_ddiff * match_weight
        curv = 0.0
        curv += (dp_home * dp_home) / p_home_safe
        curv += (dp_draw * dp_draw) / p_draw_safe
        curv += (dp_away * dp_away) / p_away_safe
        curv *= match_weight

        self._hga_grad_sum += float(grad)
        self._hga_curv_sum += float(curv)

    def _finalize_hga_year(self, next_year=None):
        if self._hga_year is None or self._hga_fixed:
            return
        if self._hga_var <= 0.0:
            self._hga_var = self.hga_prior_var if self.hga_prior_var > 0.0 else 1.0
        prior_prec = 1.0 / self._hga_var
        post_prec = prior_prec + self._hga_curv_sum
        if post_prec > 0.0:
            step = (self._hga_grad_sum / post_prec) * self.hga_step_scale
            if self.hga_max_step is not None:
                step = max(min(step, self.hga_max_step), -self.hga_max_step)
            self.hga = self.hga - step
            self._hga_var = 1.0 / post_prec
            self._hga_step_last = float(step)
        else:
            self._hga_step_last = 0.0

        self._hga_prior_var_last = float(1.0 / prior_prec) if post_prec > 0.0 else float(self._hga_var)
        self._hga_post_var_last = float(self._hga_var)
        self._hga_grad_last = float(self._hga_grad_sum)
        self._hga_curv_last = float(self._hga_curv_sum)

        self._hga_history[int(self._hga_year)] = (
            float(self.hga),
            float(self._hga_var),
            float(self._hga_grad_last),
            float(self._hga_curv_last),
            float(self._hga_step_last),
            float(self._hga_prior_var_last),
            float(self._hga_post_var_last),
        )
        self._hga_grad_sum = 0.0
        self._hga_curv_sum = 0.0

        if next_year is not None:
            delta_years = max(0, int(next_year) - int(self._hga_year))
            if self.hga_rw_var_per_year > 0.0:
                self._hga_var += self.hga_rw_var_per_year * delta_years
            self._hga_year = int(next_year)

    def export_hga_df(self) -> pd.DataFrame:
        if not self._hga_history:
            return pd.DataFrame(
                columns=[
                    "year",
                    "hga",
                    "hga_var",
                    "hga_grad",
                    "hga_curv",
                    "hga_step",
                    "hga_prior_var",
                    "hga_post_var",
                ]
            )
        rows = []
        for year, pair in sorted(self._hga_history.items()):
            if len(pair) == 7:
                hga, var, grad, curv, step, prior_var, post_var = pair
            else:
                hga, var = pair
                grad = float("nan")
                curv = float("nan")
                step = float("nan")
                prior_var = float("nan")
                post_var = float("nan")
            rows.append(
                {
                    "year": int(year),
                    "hga": float(hga),
                    "hga_var": float(var),
                    "hga_grad": float(grad),
                    "hga_curv": float(curv),
                    "hga_step": float(step),
                    "hga_prior_var": float(prior_var),
                    "hga_post_var": float(post_var),
                }
            )
        return pd.DataFrame(rows)

    def export_state_df(self) -> pd.DataFrame:
        if not self.maintain_param_history:
            raise ValueError("maintain_param_history must be True to export state history")
        if not self.teams:
            return pd.DataFrame(columns=["date", "team", "elo"])
        dates = sorted({t for st in self.teams.values() for t in st.history.keys()})
        rows = []
        for t in dates:
            for team, st in self.teams.items():
                hist = st.history.get(t)
                if hist is None:
                    continue
                rows.append({"date": t, "team": team, "elo": float(hist["elo"])})
        return pd.DataFrame(rows)

    def fit(self, results: pd.DataFrame) -> pd.DataFrame:
        if self.is_fit:
            raise ValueError("Model is already fit")
        if not isinstance(results, pd.DataFrame):
            raise TypeError("results must be a pandas DataFrame")

        df = results.sort_values("date").reset_index(drop=True)
        needed = {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"results is missing required columns: {sorted(missing)}")
        has_tournament = "tournament" in df.columns

        if not df.empty:
            self._first_match_year = int(pd.Timestamp(df.loc[0, "date"]).year)

        extra_rows = []
        games_played = {}
        hga_year = None

        for row in df.itertuples():
            t = pd.Timestamp(row.date).date()
            if self.hga_rw_var_per_year > 0.0 and not self._hga_fixed:
                year = int(t.year)
                if hga_year is None:
                    hga_year = year
                    self._hga_year = year
                    self._hga_var = float(self.hga_prior_var)
                elif year != hga_year:
                    self._finalize_hga_year(next_year=year)
                    hga_year = year

            home = row.home_team
            away = row.away_team
            is_neutral = row.neutral
            tournament = row.tournament if has_tournament else None
            score_h = int(row.home_score)
            score_a = int(row.away_score)

            st_h = self._get_or_init_team(home, t)
            st_a = self._get_or_init_team(away, t)
            games_played[home] = games_played.get(home, 0) + 1
            games_played[away] = games_played.get(away, 0) + 1

            hga = 0.0 if is_neutral else self.hga
            diff = st_h.elo + hga - st_a.elo

            is_friendly = False
            if tournament is not None and not pd.isna(tournament):
                if isinstance(tournament, str):
                    is_friendly = tournament.strip().lower() == "friendly"
            match_weight = self.friendly_weight if is_friendly else 1.0

            raw_p, p_home, p_draw, p_away, _dp_home, _dp_draw, _dp_away = self._match_probs(diff)
            if score_h > score_a:
                outcome = "home"
                s = 1.0
                p_result = p_home
            elif score_h < score_a:
                outcome = "away"
                s = 0.0
                p_result = p_away
            else:
                outcome = "draw"
                s = 0.5
                p_result = p_draw

            p_result_safe = max(p_result, EPSILON)
            loss_result = -math.log(p_result_safe) / (-math.log(1.0 / 3.0))

            if match_weight > 0.0:
                delta = self.k_factor * match_weight * (s - raw_p)
            else:
                delta = 0.0

            elo_home_post = st_h.elo + delta
            elo_away_post = st_a.elo - delta

            self._accumulate_hga_stats(diff, outcome, match_weight)

            extra_rows.append(
                {
                    "home_elo_pre": float(st_h.elo),
                    "away_elo_pre": float(st_a.elo),
                    "home_elo_post": float(elo_home_post),
                    "away_elo_post": float(elo_away_post),
                    "elo_diff_pre": float(diff),
                    "hga_pre": float(hga),
                    "p_home": float(p_home),
                    "p_draw": float(p_draw),
                    "p_away": float(p_away),
                    "p_result": float(p_result),
                    "loss_result": float(loss_result),
                    "match_weight": float(match_weight),
                    "home_games_played": int(games_played[home]),
                    "away_games_played": int(games_played[away]),
                }
            )

            st_h.update(elo_home_post, t)
            st_a.update(elo_away_post, t)

        if self.hga_rw_var_per_year > 0.0 and not self._hga_fixed and hga_year is not None:
            self._finalize_hga_year(next_year=None)

        self.is_fit = True
        extra_df = pd.DataFrame(extra_rows)
        base_df = df.reset_index(drop=True)
        return pd.concat([base_df, extra_df], axis=1)


"""
Notebook demo cell

def draw_prob_fn(diff, p0=0.30, elo_scale=400.0):
    raw_p = 1.0 / (1.0 + 10.0 ** (-diff / elo_scale))
    return p0 * 2.0 * min(raw_p, 1.0 - raw_p)


model_elo = EloModel(
    hga = 100,
    hga_prior_var = 500,
    hga_rw_var_per_year = 20,
    maintain_param_history=True,
    hga_prob_floor=0.01,
    draw_prob_fn=draw_prob_fn
)
res_elo = model_elo.fit(results)
df_hga_elo = model_elo.export_hga_df().set_index("year")
df_state_elo = model_elo.export_state_df()
res_elo["min_games_played"] = np.minimum(res_elo["home_games_played"], res_elo["away_games_played"])
R_elo = res_elo.query("date >= '1980' and min_games_played >= 10")
"""