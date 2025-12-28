import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.special import logsumexp
from dataclasses import dataclass

from src.optimiser import Optimiser


EPSILON = 1e-10
MAX_GOALS = 10


def log_factorial(n: int) -> float:
    return math.lgamma(n + 1.0)


class TeamState:

    def __init__(self, m, sigma2, last_update, maintain_param_history=False):
        self.m = m
        self.sigma2 = sigma2
        self.last_update = last_update
        self._do_maintain_param_history = maintain_param_history
        if maintain_param_history:
            self.history = {}

    def update_params(self, m, sigma2, timestamp):
        self.m = m
        self.sigma2 = sigma2 
        self.last_update = timestamp
        if self._do_maintain_param_history:
            self.history[timestamp] = {
                "m": m,
                "sigma2": sigma2
            }


class Model:

    def __init__(
        self,
        mu=0.0,
        a_hga=0.0,
        d_hga=0.0,
        a_hga_override=None,
        d_hga_override=None,
        rho=0.1,
        quality_prior_decay=0.007,
        init_var=np.array([[5, 0], [0, 5]]),
        smin_eps_epsilon=1e-6,
        maintain_param_history=False,
        cov_mode="fast",
        cov_every_n=50,
        recenter_params=True,
        variance_per_year=0.0,
        inactivity_grace_days=0,
        inactivity_decay_per_year=0.0,
        inactivity_max_years=0.0,
        hga=None,
        hga_ratio=None,
        hga_rw_var_per_year=0.0,
        hga_prior_var=1.0,
    ):
        """
        mu:    Baseline number of goals
        a_hga: Deprecated; use a_hga_override for a fixed attack HGA
        d_hga: Deprecated; use d_hga_override for a fixed defense HGA
        a_hga_override: Fixed attack HGA override (bypasses dynamic HGA)
        d_hga_override: Fixed defense HGA override (bypasses dynamic HGA)
        hga:   Optional starting HGA value for both attack/defense (dynamic baseline)
        hga_ratio: Optional ratio used to derive a_hga from hga (a_hga = hga_ratio * hga)
        hga_rw_var_per_year: Random-walk variance per year for HGA drift (applied independently)
        hga_prior_var: Prior variance for HGA at the first year (applied independently)
        rho:   Controls correlation 
        cov_mode: "fast" (curvature), "numeric", or "numeric_every_n"
        cov_every_n: interval for numerical Hessian when cov_mode="numeric_every_n"
        variance_per_year: additive variance increase per year without a match
        inactivity_grace_days: no penalty window for inactivity
        inactivity_decay_per_year: mu decrease per year of inactivity beyond grace
        inactivity_max_years: cap on inactivity penalty in years
        """
        self.mu = float(mu)
        if a_hga_override is None:
            a_hga_override = a_hga
        if d_hga_override is None:
            d_hga_override = d_hga

        if hga is not None:
            base_hga = float(hga)
            ratio = 1.0 if hga_ratio is None else float(hga_ratio)
            self.a_hga = ratio * base_hga
            self.d_hga = base_hga
        else:
            self.a_hga = float(a_hga_override)
            self.d_hga = float(d_hga_override)
        self.hga_rw_var_per_year = float(hga_rw_var_per_year)
        self.hga_prior_var = float(hga_prior_var)
        self._hga_var_a = float(hga_prior_var)
        self._hga_var_d = float(hga_prior_var)
        self._hga_year = None
        self._hga_grad_sum_a = 0.0
        self._hga_grad_sum_d = 0.0
        self._hga_curv_sum_a = 0.0
        self._hga_curv_sum_d = 0.0
        self._hga_history = {}
        self.rho = float(rho)
        self.smin_eps_epsilon = float(smin_eps_epsilon)

        # self.init_var = np.eye(2)
        self.init_var = init_var

        self.is_fit = False
        self.teams = {}
        self.maintain_param_history = maintain_param_history
        self.cov_mode = str(cov_mode)
        self.cov_every_n = int(cov_every_n)
        self.recenter_params = bool(recenter_params)
        self.quality_prior_decay = float(quality_prior_decay)
        self.variance_per_year = float(variance_per_year)
        self.inactivity_grace_days = int(inactivity_grace_days)
        self.inactivity_decay_per_year = float(inactivity_decay_per_year)
        self.inactivity_max_years = float(inactivity_max_years)
        self._match_index = 0
        self._last_numeric_cov = None
        self._mu_history = {}
        self._first_match_year = None

    def fit(self, results):
        assert not self.is_fit

        if not isinstance(results, pd.DataFrame):
            raise TypeError("results must be a pandas DataFrame")

        df = results.sort_values("date").reset_index(drop=True)

        needed = {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"results is missing required columns: {sorted(missing)}")

        log_fact = np.array([log_factorial(i) for i in range(MAX_GOALS + 1)], dtype=float)
        extra_rows = []

        if not df.empty:
            self._first_match_year = int(pd.Timestamp(df.loc[0, "date"]).year)

        games_played = {}
        hga_year = None

        for row in df.itertuples():
            t = pd.Timestamp(row.date).date()
            if self.hga_rw_var_per_year > 0.0:
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
            score_h = row.home_score
            score_a = row.away_score
            score_h_int = int(score_h)
            score_a_int = int(score_a)
            score_h_cap = min(score_h_int, MAX_GOALS)
            score_a_cap = min(score_a_int, MAX_GOALS)
            
            # init
            st_h = self._get_or_init_team(home, t)
            st_a = self._get_or_init_team(away, t)
            games_played[home] = games_played.get(home, 0) + 1
            games_played[away] = games_played.get(away, 0) + 1

            # propagate
            if self.variance_per_year > 0.0:
                self._propagate_to_time(st_h, t)
                self._propagate_to_time(st_a, t)

            # compute posterior
            mu_h_pre = st_h.m.copy()
            mu_a_pre = st_a.m.copy()
            sigma_h_pre = np.array(st_h.sigma2, dtype=float)
            sigma_a_pre = np.array(st_a.sigma2, dtype=float)
            mu_global_pre = float(self.mu)

            (
                mu_h_post, sigma_h_post, mu_a_post, sigma_a_post, _
            ) = self._compute_updated_ratings(
                score_h,
                score_a,
                is_neutral,
                st_h.m,
                st_h.sigma2,
                st_a.m,
                st_a.sigma2,
            )
            if self.hga_rw_var_per_year > 0.0 and not is_neutral:
                self._accumulate_hga_stats(_)

            # write new team params
            p_home, p_draw, p_away, p_score, debug = self._result_probabilities(
                mu_h_pre,
                mu_a_pre,
                is_neutral,
                score_h_cap,
                score_a_cap,
                log_fact,
            )
            m_home_pre = float(debug["mH"])
            m_away_pre = float(debug["mA"])
            if score_h_int > score_a_int:
                p_result = p_home
            elif score_h_int < score_a_int:
                p_result = p_away
            else:
                p_result = p_draw

            p_result_safe = max(p_result, EPSILON)
            p_score_safe = max(p_score, EPSILON)
            loss_result = -math.log(p_result_safe) / (-math.log(1.0 / 3.0))
            loss_score = -math.log(p_score_safe)

            extra_rows.append(
                {
                    "home_mu_attack_pre": float(mu_h_pre[0]),
                    "home_mu_defense_pre": float(mu_h_pre[1]),
                    "home_sigma_attack_pre": float(sigma_h_pre[0, 0]),
                    "home_sigma_defense_pre": float(sigma_h_pre[1, 1]),
                    "home_sigma_ad_pre": float(sigma_h_pre[0, 1]),
                    "away_mu_attack_pre": float(mu_a_pre[0]),
                    "away_mu_defense_pre": float(mu_a_pre[1]),
                    "away_sigma_attack_pre": float(sigma_a_pre[0, 0]),
                    "away_sigma_defense_pre": float(sigma_a_pre[1, 1]),
                    "away_sigma_ad_pre": float(sigma_a_pre[0, 1]),
                    "home_mu_attack_post": float(mu_h_post[0]),
                    "home_mu_defense_post": float(mu_h_post[1]),
                    "home_sigma_attack_post": float(sigma_h_post[0, 0]),
                    "home_sigma_defense_post": float(sigma_h_post[1, 1]),
                    "home_sigma_ad_post": float(sigma_h_post[0, 1]),
                    "away_mu_attack_post": float(mu_a_post[0]),
                    "away_mu_defense_post": float(mu_a_post[1]),
                    "away_sigma_attack_post": float(sigma_a_post[0, 0]),
                    "away_sigma_defense_post": float(sigma_a_post[1, 1]),
                    "away_sigma_ad_post": float(sigma_a_post[0, 1]),
                    "p_home": float(p_home),
                    "p_draw": float(p_draw),
                    "p_away": float(p_away),
                    "p_result": float(p_result),
                    "p_score": float(p_score),
                    "loss_result": float(loss_result),
                    "loss_score": float(loss_score),
                    "mu_global_pre": float(mu_global_pre),
                    "m_home_pre": m_home_pre,
                    "m_away_pre": m_away_pre,
                    "exp_home_goals": m_home_pre,
                    "exp_away_goals": m_away_pre,
                    "lam_home_pre": float(debug["lamH"]),
                    "lam_away_pre": float(debug["lamA"]),
                    "nu_pre": float(debug["nu"]),
                    "home_games_played": int(games_played[home]),
                    "away_games_played": int(games_played[away]),
                }
            )

            st_h.update_params(mu_h_post, sigma_h_post, t)
            st_a.update_params(mu_a_post, sigma_a_post, t)
            if self.recenter_params:
                self._recenter_params(t)
            mu_global_post = float(self.mu)
            self._mu_history[t] = float(self.mu)
            self._match_index += 1
            reason = _.get("reason")
            if reason == "max_steps":
                print(
                    "warning: max_steps reached "
                    f"date={t} {home} vs {away} score={score_h}-{score_a} "
                    f"neutral={bool(is_neutral)} "
                    f"iters={_.get('iters')} grad_norm={_.get('grad_norm')} "
                    f"rel_improve_last={_.get('rel_improve_last')} f={_.get('f')}"
                )
            if reason == "lm_failed_to_find_descent":
                print(
                    "warning: lm_failed_to_find_descent "
                    f"date={t} {home} vs {away} score={score_h}-{score_a} "
                    f"neutral={bool(is_neutral)} "
                    f"lm_retries={_.get('lm_retries')} "
                    f"lm_last={_.get('lm_last')} "
                    f"best_step_norm={_.get('best_step_norm')} "
                    f"best_f_trial={_.get('best_f_trial')} "
                    f"saw_finite_trial={_.get('saw_finite_trial')} "
                    f"saw_improve={_.get('saw_improve')}"
                )

            extra_rows[-1]["mu_global_post"] = mu_global_post

        if self.hga_rw_var_per_year > 0.0 and hga_year is not None:
            self._finalize_hga_year(next_year=None)

        self.is_fit = True
        extra_df = pd.DataFrame(extra_rows)
        base_df = df.reset_index(drop=True)
        param_cols = [
            "home_mu_attack_pre",
            "home_mu_defense_pre",
            "home_sigma_attack_pre",
            "home_sigma_defense_pre",
            "home_sigma_ad_pre",
            "away_mu_attack_pre",
            "away_mu_defense_pre",
            "away_sigma_attack_pre",
            "away_sigma_defense_pre",
            "away_sigma_ad_pre",
            "home_mu_attack_post",
            "home_mu_defense_post",
            "home_sigma_attack_post",
            "home_sigma_defense_post",
            "home_sigma_ad_post",
            "away_mu_attack_post",
            "away_mu_defense_post",
            "away_sigma_attack_post",
            "away_sigma_defense_post",
            "away_sigma_ad_post",
        ]
        metric_cols = [
            "mu_global_pre",
            "mu_global_post",
            "m_home_pre",
            "m_away_pre",
            "exp_home_goals",
            "exp_away_goals",
            "lam_home_pre",
            "lam_away_pre",
            "nu_pre",
            "home_games_played",
            "away_games_played",
            "p_home",
            "p_draw",
            "p_away",
            "p_result",
            "p_score",
            "loss_result",
            "loss_score",
        ]
        ordered_cols = list(base_df.columns) + param_cols + metric_cols
        combined = pd.concat([base_df, extra_df], axis=1)
        return combined.loc[:, ordered_cols]

    def _recenter_params(self, t: pd.Timestamp):
        if not self.teams:
            return
        ms = np.array([st.m for st in self.teams.values()], dtype=float)
        mean_a, mean_d = ms.mean(axis=0)
        if mean_a == 0.0 and mean_d == 0.0:
            return
        shift = np.array([mean_a, mean_d], dtype=float)
        for st in self.teams.values():
            st.m = st.m - shift
            if st._do_maintain_param_history:
                st.history[t] = {
                    "m": st.m.copy(),
                    "sigma2": np.array(st.sigma2, dtype=float),
                }
        self.mu += mean_a - mean_d
        
    def _current_hga_components(self):
        return self.a_hga, self.d_hga

    def _propagate_to_time(self, st: TeamState, t: pd.Timestamp):
        if self.variance_per_year <= 0.0:
            pass
        if t <= st.last_update:
            return
        delta_days = (t - st.last_update).days
        if delta_days <= 0:
            return
        delta_years = delta_days / 365.0
        if self.variance_per_year > 0.0:
            inc = self.variance_per_year * delta_years
            st.sigma2 = np.array(st.sigma2, dtype=float) + inc * np.eye(2)
        if self.inactivity_decay_per_year > 0.0:
            grace_days = max(0, self.inactivity_grace_days)
            if delta_days > grace_days:
                inactive_years = (delta_days - grace_days) / 365.0
                if self.inactivity_max_years > 0.0:
                    inactive_years = min(inactive_years, self.inactivity_max_years)
                penalty = self.inactivity_decay_per_year * inactive_years
                st.m = np.array(st.m, dtype=float) - np.array([penalty, penalty], dtype=float)
        st.last_update = t

    def _accumulate_hga_stats(self, info):
        if self.hga_rw_var_per_year <= 0.0:
            return
        df_detaH = info.get("hga_df_detaH")
        df_detaA = info.get("hga_df_detaA")
        mH = info.get("hga_mH")
        mA = info.get("hga_mA")
        if df_detaH is None or df_detaA is None or mH is None or mA is None:
            return
        grad_a = float(df_detaH)
        grad_d = -float(df_detaA)
        curv_a = float(mH)
        curv_d = float(mA)
        self._hga_grad_sum_a += grad_a
        self._hga_grad_sum_d += grad_d
        self._hga_curv_sum_a += curv_a
        self._hga_curv_sum_d += curv_d

    def _finalize_hga_year(self, next_year=None):
        if self._hga_year is None:
            return
        if self._hga_var_a <= 0.0:
            self._hga_var_a = self.hga_prior_var if self.hga_prior_var > 0.0 else 1.0
        if self._hga_var_d <= 0.0:
            self._hga_var_d = self.hga_prior_var if self.hga_prior_var > 0.0 else 1.0

        prior_prec_a = 1.0 / self._hga_var_a
        prior_prec_d = 1.0 / self._hga_var_d
        post_prec_a = prior_prec_a + self._hga_curv_sum_a
        post_prec_d = prior_prec_d + self._hga_curv_sum_d
        if post_prec_a > 0.0:
            self.a_hga = self.a_hga - (self._hga_grad_sum_a / post_prec_a)
            self._hga_var_a = 1.0 / post_prec_a
        if post_prec_d > 0.0:
            self.d_hga = self.d_hga - (self._hga_grad_sum_d / post_prec_d)
            self._hga_var_d = 1.0 / post_prec_d

        self._hga_history[int(self._hga_year)] = (float(self.a_hga), float(self.d_hga))
        self._hga_grad_sum_a = 0.0
        self._hga_grad_sum_d = 0.0
        self._hga_curv_sum_a = 0.0
        self._hga_curv_sum_d = 0.0
        if next_year is not None:
            delta_years = max(0, int(next_year) - int(self._hga_year))
            if self.hga_rw_var_per_year > 0.0:
                self._hga_var_a += self.hga_rw_var_per_year * delta_years
                self._hga_var_d += self.hga_rw_var_per_year * delta_years
            self._hga_year = int(next_year)
        self._current_hga_components()

    def _get_or_init_team(self, team: str, t: pd.Timestamp) -> TeamState:
        st = self.teams.get(team)
        if st is not None:
            return st
        year_delta = 0
        if self._first_match_year is not None:
            year_delta = int(t.year) - int(self._first_match_year)
        decay = self.quality_prior_decay
        init_mu = -float(year_delta) * decay
        st = TeamState(
            m=np.array([init_mu, init_mu], dtype=float),
            sigma2=self.init_var.copy(),
            last_update=t,
            maintain_param_history=self.maintain_param_history
        )
        self.teams[team] = st
        return st

    def export_state_df(self) -> pd.DataFrame:
        if not self.maintain_param_history:
            raise ValueError("maintain_param_history must be True to export full state history")
        if not self.teams:
            return pd.DataFrame(
                columns=[
                    "date",
                    "team",
                    "mu_attack",
                    "mu_defense",
                    "sigma_attack",
                    "sigma_defense",
                    "sigma_ad",
                    "mu_global",
                ]
            )
        dates = sorted({t for st in self.teams.values() for t in st.history.keys()})
        rows = []
        for t in dates:
            for team, st in self.teams.items():
                hist = st.history.get(t)
                if hist is None:
                    continue
                m = np.array(hist["m"], dtype=float)
                s = np.array(hist["sigma2"], dtype=float)
                rows.append(
                    {
                        "date": t,
                        "team": team,
                        "mu_attack": float(m[0]),
                        "mu_defense": float(m[1]),
                        "sigma_attack": float(s[0, 0]),
                        "sigma_defense": float(s[1, 1]),
                        "sigma_ad": float(s[0, 1]),
                        "mu_global": float(self._mu_history.get(t, self.mu)),
                    }
                )
        return pd.DataFrame(rows)

    def export_hga_df(self) -> pd.DataFrame:
        if not self._hga_history:
            return pd.DataFrame(columns=["year", "a_hga", "d_hga"])
        rows = []
        for year, hga_pair in sorted(self._hga_history.items()):
            a_hga, d_hga = hga_pair
            rows.append(
                {
                    "year": int(year),
                    "a_hga": float(a_hga),
                    "d_hga": float(d_hga),
                }
            )
        return pd.DataFrame(rows)

    def export_mu_df(self) -> pd.DataFrame:
        if not self._mu_history:
            return pd.DataFrame(columns=["date", "mu_global"])
        rows = []
        for t, mu in sorted(self._mu_history.items()):
            rows.append({"date": t, "mu_global": float(mu)})
        return pd.DataFrame(rows)

    @staticmethod
    def _smin_eps(m, a, eps=1e-6):
        return 0.5 * (m + a - np.sqrt((m - a) ** 2 + eps ** 2))

    def _compute_updated_ratings(
        self,
        score_h: int,
        score_a: int,
        is_neutral: bool,
        mu_h_prior: np.ndarray,
        sigma_h_prior: np.ndarray,
        mu_a_prior: np.ndarray,
        sigma_a_prior: np.ndarray,
        max_steps=500,
        tol_grad=1e-6,
        tol_step=1e-10,
        tol_improve=1e-6,
        alpha_max=1.0,
        beta=0.5,
        c1=1e-4,
        max_backtracks=50,
        lm_damping=1e-2,     # default damping ON (helps low-score cases)
        cov_jitter=1e-9,
    ):
        x = int(score_h)
        y = int(score_a)

        if is_neutral:
            a_hga = 0.0
            d_hga = 0.0
        else:
            a_hga, d_hga = self._current_hga_components()

        mu0 = np.array([mu_h_prior[0], mu_h_prior[1], mu_a_prior[0], mu_a_prior[1]], dtype=float)

        Sigma0 = np.zeros((4, 4), dtype=float)
        Sigma0[:2, :2] = np.array(sigma_h_prior, dtype=float)
        Sigma0[2:, 2:] = np.array(sigma_a_prior, dtype=float)

        Sigma0_inv = np.linalg.solve(Sigma0 + 1e-12*np.eye(4), np.eye(4))

        J = np.array(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 1.0, 0.0],
            ],
            dtype=float,
        )

        eps_smin = self.smin_eps_epsilon

        def _forward(theta):
            aH, dH, aA, dA = theta

            etaH = self.mu + (aH + a_hga) - dA
            etaA = self.mu + aA - (dH + d_hga)

            mH = float(np.exp(etaH))
            mA = float(np.exp(etaA))

            rho = float(self.rho)
            rho = min(max(rho, 0.0), 1.0 - 1e-9)

            smin = float(Model._smin_eps(mH, mA, eps=eps_smin))
            if smin < 0.0:
                smin = 0.0

            nu = rho * smin
            lamH = mH - nu
            lamA = mA - nu

            if lamH <= 0.0 or lamA <= 0.0 or nu < 0.0:
                return etaH, etaA, mH, mA, nu, lamH, lamA, False

            return etaH, etaA, mH, mA, nu, lamH, lamA, True

        def _loglik_and_ubar(mH, mA, nu, lamH, lamA):
            Umax = min(x, y)
            log_t = np.full(Umax + 1, -np.inf, dtype=float)

            if nu < 1e-14:
                log_t[0] = x * np.log(lamH) - log_factorial(x) + y * np.log(lamA) - log_factorial(y)
                logS = float(log_t[0])
                ubar = 0.0
                logp = -(mH + mA - nu) + logS
                return float(logp), float(ubar)

            log_nu = np.log(nu)
            log_lamH = np.log(lamH)
            log_lamA = np.log(lamA)

            for u in range(Umax + 1):
                log_t[u] = (
                    u * log_nu
                    - log_factorial(u)
                    + (x - u) * log_lamH
                    - log_factorial(x - u)
                    + (y - u) * log_lamA
                    - log_factorial(y - u)
                )

            if Umax <= 10:
                m = float(np.max(log_t))
                logS = m + float(np.log(np.sum(np.exp(log_t - m))))
            else:
                logS = float(logsumexp(log_t))
            w = np.exp(log_t - logS)
            ubar = float(np.dot(w, np.arange(Umax + 1, dtype=float)))

            logp = -(mH + mA - nu) + logS
            return float(logp), float(ubar)

        def _nu_partials(mH, mA):
            rho = float(self.rho)
            rho = min(max(rho, 0.0), 1.0 - 1e-9)

            d = mH - mA
            s = float(np.sqrt(d * d + eps_smin * eps_smin))

            dsmin_dmH = 0.5 * (1.0 - d / s)
            dsmin_dmA = 0.5 * (1.0 + d / s)

            smin = float(Model._smin_eps(mH, mA, eps=eps_smin))
            if smin <= 0.0:
                return 0.0, 0.0

            return rho * dsmin_dmH, rho * dsmin_dmA

        def f(theta):
            theta = np.asarray(theta, dtype=float)
            etaH, etaA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return float("inf")

            logp, _ = _loglik_and_ubar(mH, mA, nu, lamH, lamA)
            nll = -logp

            diff = theta - mu0
            prior = 0.5 * float(diff.T @ Sigma0_inv @ diff)
            return float(nll + prior)

        def grad(theta):
            theta = np.asarray(theta, dtype=float)
            etaH, etaA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            diff = theta - mu0
            g_prior = Sigma0_inv @ diff

            if not ok:
                return g_prior

            _, ubar = _loglik_and_ubar(mH, mA, nu, lamH, lamA)
            nu_mH, nu_mA = _nu_partials(mH, mA)

            inv_nu = 0.0 if nu < 1e-14 else 1.0 / nu

            dlogp_dmH = (
                -(1.0 - nu_mH)
                + nu_mH * (ubar * inv_nu)
                + (1.0 - nu_mH) * ((x - ubar) / lamH)
                - nu_mH * ((y - ubar) / lamA)
            )

            dlogp_dmA = (
                -(1.0 - nu_mA)
                + nu_mA * (ubar * inv_nu)
                + (1.0 - nu_mA) * ((y - ubar) / lamA)
                - nu_mA * ((x - ubar) / lamH)
            )

            dlogp_detaH = mH * dlogp_dmH
            dlogp_detaA = mA * dlogp_dmA

            df_detaH = -dlogp_detaH
            df_detaA = -dlogp_detaA

            g_like = np.array(
                [
                    df_detaH,
                    -df_detaA,
                    df_detaA,
                    -df_detaH,
                ],
                dtype=float,
            )

            return g_like + g_prior

        def _likelihood_partials(theta):
            theta = np.asarray(theta, dtype=float)
            etaH, etaA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return None

            _, ubar = _loglik_and_ubar(mH, mA, nu, lamH, lamA)
            nu_mH, nu_mA = _nu_partials(mH, mA)

            inv_nu = 0.0 if nu < 1e-14 else 1.0 / nu

            dlogp_dmH = (
                -(1.0 - nu_mH)
                + nu_mH * (ubar * inv_nu)
                + (1.0 - nu_mH) * ((x - ubar) / lamH)
                - nu_mH * ((y - ubar) / lamA)
            )

            dlogp_dmA = (
                -(1.0 - nu_mA)
                + nu_mA * (ubar * inv_nu)
                + (1.0 - nu_mA) * ((y - ubar) / lamA)
                - nu_mA * ((x - ubar) / lamH)
            )

            dlogp_detaH = mH * dlogp_dmH
            dlogp_detaA = mA * dlogp_dmA

            df_detaH = -dlogp_detaH
            df_detaA = -dlogp_detaA
            return df_detaH, df_detaA, mH, mA

        def curv(theta):
            theta = np.asarray(theta, dtype=float)
            etaH, etaA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return Sigma0_inv.copy()

            H_eta = np.diag([mH, mA])
            H_like = J.T @ H_eta @ J
            H = H_like + Sigma0_inv
            return 0.5 * (H + H.T)

        theta_map, info, cov = Optimiser.fisher_scoring_armijo_fast(
            f=f,
            grad=grad,
            curv=curv,
            init_val=mu0,
            tol_grad=tol_grad,
            tol_step=tol_step,
            max_steps=max_steps,
            tol_improve=tol_improve,
            alpha_max=alpha_max,
            beta=beta,
            c1=c1,
            max_backtracks=max_backtracks,
            lm_damping=lm_damping,
            cov_jitter=cov_jitter,
            return_cov=True,
        )
        if info.get("reason") == "lm_failed_to_find_descent":
            theta_map, info, cov = Optimiser.fisher_scoring_armijo(
                f=f,
                grad=grad,
                curv=curv,
                init_val=mu0,
                tol_grad=tol_grad,
                tol_step=tol_step,
                tol_improve=tol_improve,
                max_steps=max_steps,
                alpha_max=alpha_max,
                beta=beta,
                c1=c1,
                max_backtracks=max_backtracks,
                lm_damping=lm_damping,
                cov_jitter=cov_jitter,
                return_cov=True,
            )
            info = dict(info)
            info["fallback_armijo"] = True

        cov_mode = self.cov_mode
        if cov_mode == "numeric":
            H = numerical_hessian_from_grad(grad, theta_map, eps=1e-4)
            cov, _H_pd = pd_inverse(H, min_eig=1e-8)
            self._last_numeric_cov = cov
        elif cov_mode == "numeric_every_n":
            if self.cov_every_n <= 0:
                raise ValueError("cov_every_n must be positive for numeric_every_n mode")
            if (self._match_index % self.cov_every_n) == 0:
                H = numerical_hessian_from_grad(grad, theta_map, eps=1e-4)
                cov, _H_pd = pd_inverse(H, min_eig=1e-8)
                self._last_numeric_cov = cov
            elif self._last_numeric_cov is not None:
                cov = self._last_numeric_cov
        elif cov_mode != "fast":
            raise ValueError(f"Unsupported cov_mode: {cov_mode}")

        Sigma_post = cov
        mu_h_post = np.array([theta_map[0], theta_map[1]], dtype=float)
        mu_a_post = np.array([theta_map[2], theta_map[3]], dtype=float)
        Sigma_h_post = Sigma_post[:2, :2].copy()
        Sigma_a_post = Sigma_post[2:, 2:].copy()

        # Extra sanity diagnostics (you can comment these out later)
        g_at_map = grad(theta_map)
        info = dict(info)
        hga_partials = _likelihood_partials(theta_map)
        if hga_partials is not None:
            df_detaH, df_detaA, mH_like, mA_like = hga_partials
            info["hga_df_detaH"] = float(df_detaH)
            info["hga_df_detaA"] = float(df_detaA)
            info["hga_mH"] = float(mH_like)
            info["hga_mA"] = float(mA_like)
        info["grad_norm_recomputed"] = float(np.linalg.norm(g_at_map))
        info["theta_map"] = theta_map.copy()

        return mu_h_post, Sigma_h_post, mu_a_post, Sigma_a_post, info

    def _poisson_pmf(self, lam: float, log_fact: np.ndarray) -> np.ndarray:
        if lam <= 0.0:
            pmf = np.zeros(MAX_GOALS + 1, dtype=float)
            pmf[0] = 1.0
            return pmf
        k = np.arange(MAX_GOALS + 1, dtype=float)
        log_p = -lam + k * np.log(lam) - log_fact
        return np.exp(log_p)

    def _result_probabilities(
        self,
        mu_h: np.ndarray,
        mu_a: np.ndarray,
        is_neutral: bool,
        score_h: int,
        score_a: int,
        log_fact: np.ndarray,
    ):
        if is_neutral:
            a_hga = 0.0
            d_hga = 0.0
        else:
            a_hga, d_hga = self._current_hga_components()

        aH, dH = float(mu_h[0]), float(mu_h[1])
        aA, dA = float(mu_a[0]), float(mu_a[1])

        etaH = self.mu + (aH + a_hga) - dA
        etaA = self.mu + aA - (dH + d_hga)

        mH = float(np.exp(etaH))
        mA = float(np.exp(etaA))

        rho = float(self.rho)
        rho = min(max(rho, 0.0), 1.0 - 1e-9)

        smin = float(Model._smin_eps(mH, mA, eps=self.smin_eps_epsilon))
        if smin < 0.0:
            smin = 0.0

        nu = rho * smin
        lamH = mH - nu
        lamA = mA - nu

        if lamH <= 0.0 or lamA <= 0.0 or nu < 0.0:
            return 0.0, 0.0, 0.0, 0.0, {"mH": mH, "mA": mA, "lamH": lamH, "lamA": lamA, "nu": nu}

        p_x = self._poisson_pmf(lamH, log_fact)
        p_y = self._poisson_pmf(lamA, log_fact)

        if nu < 1e-14:
            joint = np.outer(p_x, p_y)
        else:
            p_u = self._poisson_pmf(nu, log_fact)
            joint = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1), dtype=float)
            for u in range(MAX_GOALS + 1):
                if p_u[u] == 0.0:
                    continue
                px = p_x[: MAX_GOALS + 1 - u]
                py = p_y[: MAX_GOALS + 1 - u]
                joint[u:, u:] += p_u[u] * np.outer(px, py)

        total = float(joint.sum())
        if total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, {"mH": mH, "mA": mA, "lamH": lamH, "lamA": lamA, "nu": nu}

        p_home = float(np.tril(joint, k=-1).sum())
        p_draw = float(np.trace(joint))
        p_away = float(np.triu(joint, k=1).sum())
        p_score = float(joint[score_h, score_a]) if score_h <= MAX_GOALS and score_a <= MAX_GOALS else 0.0

        if total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        inv_total = 1.0 / total
        return (
            p_home * inv_total,
            p_draw * inv_total,
            p_away * inv_total,
            p_score * inv_total,
            {"mH": mH, "mA": mA, "lamH": lamH, "lamA": lamA, "nu": nu},
        )


def numerical_hessian_from_grad(g, x, eps=1e-4):
    """
    Central-difference Hessian approximation using gradient function g.
    g: R^n -> R^n
    x: (n,)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    for j in range(n):
        step = np.zeros(n, dtype=float)
        step[j] = eps
        gp = g(x + step)
        gm = g(x - step)
        H[:, j] = (gp - gm) / (2.0 * eps)

    # Symmetrise to clean numerical noise
    H = 0.5 * (H + H.T)
    return H

def pd_inverse(H, min_eig=1e-8):
    """
    Return inverse of a numerically PD-projected matrix.
    """
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, min_eig)
    H_pd = (V * w) @ V.T
    cov = (V * (1.0 / w)) @ V.T
    return cov, H_pd
