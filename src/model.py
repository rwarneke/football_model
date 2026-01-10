import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.special import logsumexp

from src.optimiser import Optimiser


EPSILON = 1e-10
MAX_GOALS = 10
MAX_LOG_EXP = 700.0


def safe_exp(x: float) -> float:
    return float(np.exp(np.clip(x, -MAX_LOG_EXP, MAX_LOG_EXP)))


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
        mu=1.0,
        rho=0.15,
        # Priors
        mu_prior_decay=150e-4,
        mu_prior_shift={
            "AFC":       0.2,
            "UEFA":      0.0,
            "CAF":       0.4,
            "OFC":      -0.6,
            "CONCACAF": -0.5,
            "CONMEBOL":  0.8,
        },
        # Learning rates
        init_var_diag=0.2,
        cross_var_ratio=0.0,
        variance_per_year=40e-4,
        variance_max=None,
        variance_min=None,
        # Extra time handling
        extra_time_exp_score_mult=0.23,
        use_extra_time_updates=True,
        # Inactivity
        inactivity_grace_days=0,
        inactivity_decay_per_year=0.0,
        inactivity_max_years=0.0,
        # Home ground advantage
        hga=0.2,
        hga_prior_var=5e-4,
        hga_rw_var_per_year=0.1e-4,
        a_hga_override=None,
        d_hga_override=None,
        # Learning weights
        friendly_weight=1.0,
        # Shootout model
        shootout_skilldiff_coef=0.353,
        # Prediction correction
        lognormal_score_correction=False,
        # Recentering (aesthetic, but priors need to shift)
        recenter_params=True,
        recenter_top_n=20,
        recenter_team_divisor=5,
        # Logging
        maintain_param_history=True,
        # Numerical optimisation
        smin_eps_epsilon=1e-6,
        cov_mode="fast",
        cov_every_n=50,
    ):
        """
        mu:    Baseline score rate
        hga:   Starting HGA value for both attack/defense (dynamic baseline)
        hga_rw_var_per_year: Random-walk variance per year for HGA drift (applied independently)
        hga_prior_var: Prior variance for HGA at the first year (applied independently)
        a_hga_override: Fixed attack HGA override (bypasses dynamic HGA)
        d_hga_override: Fixed defense HGA override (bypasses dynamic HGA)
        rho:   Controls correlation 
        cov_mode: "fast" (curvature), "numeric", or "numeric_every_n"
        cov_every_n: interval for numerical Hessian when cov_mode="numeric_every_n"
        variance_per_year: additive variance increase per year without a match
        variance_max: cap on variance values (None disables cap)
        variance_min: floor on variance values (None disables floor)
        inactivity_grace_days: no penalty window for inactivity
        inactivity_decay_per_year: mu decrease per year of inactivity beyond grace
        inactivity_max_years: cap on inactivity penalty in years
        init_var_diag: prior variance for attack/defense
        cross_var_ratio: off-diagonal covariance ratio vs diagonal variance (0 to 1)
        friendly_weight: match weight multiplier when tournament == "Friendly"
        shootout_skilldiff_coef: logit coefficient for shootout home_win vs skilldiff
        lognormal_score_correction: apply log-normal correction to expected scores in prediction
        mu_prior_decay: mu decrease per year for first-time teams after the first match year
        mu_prior_shift: dict mapping confederation to extra mu shift for new teams
        extra_time_exp_score_mult: scale factor for expected scores in extra time
        use_extra_time_updates: if True, split extra-time matches into 90-min and ET updates
        recenter_top_n: if set, cap top-N recentering using clip(ceil(k/5), 2, recenter_top_n)
        recenter_team_divisor: divisor for k in clip(ceil(k/divisor), 2, recenter_top_n)
        """
        self.mu = float(mu)
        self._hga_fixed_a = a_hga_override is not None
        self._hga_fixed_d = d_hga_override is not None
        base_hga = float(hga)
        self.a_hga = float(a_hga_override) if self._hga_fixed_a else base_hga
        self.d_hga = float(d_hga_override) if self._hga_fixed_d else base_hga
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

        self.cross_var_ratio = float(cross_var_ratio)
        if not (0.0 <= self.cross_var_ratio <= 1.0):
            raise ValueError("cross_var_ratio must be between 0 and 1")
        init_var_diag_f = float(init_var_diag)
        init_var_cross = init_var_diag_f * self.cross_var_ratio
        self.init_var = np.array(
            [[init_var_diag_f, init_var_cross],
             [init_var_cross, init_var_diag_f]],
            dtype=float,
        )

        self.is_fit = False
        self.teams = {}
        self.maintain_param_history = maintain_param_history
        self.cov_mode = str(cov_mode)
        self.cov_every_n = int(cov_every_n)
        self.recenter_params = bool(recenter_params)
        if recenter_top_n is None:
            self.recenter_top_n = None
        else:
            self.recenter_top_n = int(recenter_top_n)
            if self.recenter_top_n < 2:
                raise ValueError("recenter_top_n must be >= 2 or None")
        self.recenter_team_divisor = float(recenter_team_divisor)
        if self.recenter_team_divisor <= 0.0:
            raise ValueError("recenter_team_divisor must be positive")
        self.mu_prior_decay = float(mu_prior_decay)
        self.mu_prior_shift = {} if mu_prior_shift is None else dict(mu_prior_shift)
        self.extra_time_exp_score_mult = float(extra_time_exp_score_mult)
        if self.extra_time_exp_score_mult <= 0.0:
            raise ValueError("extra_time_exp_score_mult must be positive")
        self.use_extra_time_updates = bool(use_extra_time_updates)
        self.variance_per_year = float(variance_per_year)
        self.variance_max = None if variance_max is None else float(variance_max)
        self.variance_min = None if variance_min is None else float(variance_min)
        if self.variance_max is not None and self.variance_max < 0.0:
            raise ValueError("variance_max must be non-negative or None")
        if self.variance_min is not None and self.variance_min < 0.0:
            raise ValueError("variance_min must be non-negative or None")
        if (
            self.variance_max is not None
            and self.variance_min is not None
            and self.variance_min > self.variance_max
        ):
            raise ValueError("variance_min must be <= variance_max")
        self.inactivity_grace_days = int(inactivity_grace_days)
        self.inactivity_decay_per_year = float(inactivity_decay_per_year)
        self.inactivity_max_years = float(inactivity_max_years)
        self.friendly_weight = float(friendly_weight)
        if self.friendly_weight < 0.0:
            raise ValueError("friendly_weight must be non-negative")
        self.shootout_skilldiff_coef = float(shootout_skilldiff_coef)
        self.lognormal_score_correction = bool(lognormal_score_correction)
        self._match_index = 0
        self._last_numeric_cov = None
        self._mu_history = {}
        self._first_match_year = None

    def _apply_variance_bounds(self, sigma2: np.ndarray) -> np.ndarray:
        if self.variance_max is None and self.variance_min is None:
            return sigma2
        s = np.array(sigma2, dtype=float)
        if self.variance_max is not None:
            s[0, 0] = min(s[0, 0], self.variance_max)
            s[1, 1] = min(s[1, 1], self.variance_max)
        if self.variance_min is not None:
            s[0, 0] = max(s[0, 0], self.variance_min)
            s[1, 1] = max(s[1, 1], self.variance_min)
        return s

    def fit(self, results):
        assert not self.is_fit

        if not isinstance(results, pd.DataFrame):
            raise TypeError("results must be a pandas DataFrame")

        df = results.sort_values("date").reset_index(drop=True)

        needed = {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"results is missing required columns: {sorted(missing)}")
        has_tournament = "tournament" in df.columns
        has_confed = (
            "home_confederation" in df.columns and "away_confederation" in df.columns
        )
        if self.mu_prior_shift and not has_confed:
            raise ValueError(
                "mu_prior_shift requires home_confederation/away_confederation columns"
            )
        if self.use_extra_time_updates:
            needed_et = {
                "had_extra_time",
                "score_reconciled",
                "home_score_90",
                "away_score_90",
                "home_score_120",
                "away_score_120",
            }
            missing_et = needed_et - set(df.columns)
            if missing_et:
                raise ValueError(
                    "use_extra_time_updates requires columns: "
                    f"{sorted(missing_et)}"
                )
            if df["had_extra_time"].isna().any():
                sample = df.loc[
                    df["had_extra_time"].isna(), ["date", "home_team", "away_team"]
                ].head(20)
                raise ValueError(
                    "had_extra_time contains NaN values.\n"
                    f"{sample.to_string(index=False)}"
                )
            if df["score_reconciled"].isna().any():
                sample = df.loc[
                    df["score_reconciled"].isna(), ["date", "home_team", "away_team"]
                ].head(20)
                raise ValueError(
                    "score_reconciled contains NaN values.\n"
                    f"{sample.to_string(index=False)}"
                )
            et_mask = df["had_extra_time"].astype(bool) & df["score_reconciled"].astype(bool)
            if et_mask.any():
                score_cols = [
                    "home_score_90",
                    "away_score_90",
                    "home_score_120",
                    "away_score_120",
                ]
                missing_scores = df.loc[et_mask, score_cols].isna().any(axis=1)
                if missing_scores.any():
                    sample = df.loc[
                        et_mask & missing_scores,
                        ["date", "home_team", "away_team"] + score_cols,
                    ].head(20)
                    raise ValueError(
                        "extra-time matches missing 90/120 scores.\n"
                        f"{sample.to_string(index=False)}"
                    )
                invalid_scores = (
                    (df.loc[et_mask, "home_score_120"] < df.loc[et_mask, "home_score_90"])
                    | (df.loc[et_mask, "away_score_120"] < df.loc[et_mask, "away_score_90"])
                )
                if invalid_scores.any():
                    sample = df.loc[
                        et_mask & invalid_scores,
                        ["date", "home_team", "away_team"] + score_cols,
                    ].head(20)
                    raise ValueError(
                        "extra-time matches have 120 scores below 90 scores.\n"
                        f"{sample.to_string(index=False)}"
                    )

        log_fact = np.array([log_factorial(i) for i in range(MAX_GOALS + 1)], dtype=float)
        extra_rows = []

        if not df.empty:
            self._first_match_year = int(pd.Timestamp(df.loc[0, "date"]).year)

        games_played = {}
        hga_year = None

        for row in df.itertuples():
            t = pd.Timestamp(row.date).date()
            extra_time_info = None
            if self.hga_rw_var_per_year > 0.0:
                year = int(t.year)
                if hga_year is None:
                    hga_year = year
                    self._hga_year = year
                    self._hga_var_a = float(self.hga_prior_var)
                    self._hga_var_d = float(self.hga_prior_var)
                elif year != hga_year:
                    self._finalize_hga_year(next_year=year)
                    hga_year = year
            home = row.home_team
            away = row.away_team
            is_neutral = row.neutral
            tournament = row.tournament if has_tournament else None
            score_h = row.home_score
            score_a = row.away_score
            score_h_int = int(score_h)
            score_a_int = int(score_a)
            score_h_cap = min(score_h_int, MAX_GOALS)
            score_a_cap = min(score_a_int, MAX_GOALS)
            had_extra_time = (
                bool(row.had_extra_time) and bool(row.score_reconciled)
                if self.use_extra_time_updates
                else False
            )
            
            # init
            home_confed = row.home_confederation if has_confed else None
            away_confed = row.away_confederation if has_confed else None
            st_h = self._get_or_init_team(home, t, home_confed)
            st_a = self._get_or_init_team(away, t, away_confed)
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
            if is_neutral:
                a_hga = 0.0
                d_hga = 0.0
            else:
                a_hga, d_hga = self._current_hga_components()
            eta_home_pre = float(self.mu + (mu_h_pre[0] + a_hga) - mu_a_pre[1])
            eta_away_pre = float(self.mu + mu_a_pre[0] - (mu_h_pre[1] + d_hga))
            f_home_pre = eta_home_pre
            fprime_home_pre = 1.0
            g_home_pre = 0.0
            f_away_pre = eta_away_pre
            fprime_away_pre = 1.0
            g_away_pre = 0.0
            is_friendly = False
            if tournament is not None and not pd.isna(tournament):
                if isinstance(tournament, str):
                    is_friendly = tournament.strip().lower() == "friendly"
            match_weight = self.friendly_weight if is_friendly else 1.0

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

            if self.use_extra_time_updates and had_extra_time:
                score_h_90 = row.home_score_90
                score_a_90 = row.away_score_90
                score_h_120 = row.home_score_120
                score_a_120 = row.away_score_120
                score_h_et = score_h_120 - score_h_90
                score_a_et = score_a_120 - score_a_90
                if score_h_et < 0 or score_a_et < 0:
                    raise ValueError(
                        "extra-time goals are negative "
                        f"date={t} {home} vs {away} "
                        f"score_90={score_h_90}-{score_a_90} "
                        f"score_120={score_h_120}-{score_a_120}"
                    )
                (
                    mu_h_mid,
                    sigma_h_mid,
                    mu_a_mid,
                    sigma_a_mid,
                    info_90,
                ) = self._compute_updated_ratings(
                    score_h_90,
                    score_a_90,
                    is_neutral,
                    st_h.m,
                    st_h.sigma2,
                    st_a.m,
                    st_a.sigma2,
                    match_weight=match_weight,
                )
                sigma_h_mid = self._apply_variance_bounds(sigma_h_mid)
                sigma_a_mid = self._apply_variance_bounds(sigma_a_mid)
                if self.hga_rw_var_per_year > 0.0 and not is_neutral:
                    self._accumulate_hga_stats(info_90)
                mu_before_et = float(self.mu)
                mu_shift = math.log(self.extra_time_exp_score_mult)
                self.mu = mu_before_et + mu_shift
                try:
                    (
                        mu_h_post,
                        sigma_h_post,
                        mu_a_post,
                        sigma_a_post,
                        info_et,
                    ) = self._compute_updated_ratings(
                        score_h_et,
                        score_a_et,
                        is_neutral,
                        mu_h_mid,
                        sigma_h_mid,
                        mu_a_mid,
                        sigma_a_mid,
                        match_weight=match_weight,
                    )
                finally:
                    self.mu = mu_before_et
                sigma_h_post = self._apply_variance_bounds(sigma_h_post)
                sigma_a_post = self._apply_variance_bounds(sigma_a_post)
                if self.hga_rw_var_per_year > 0.0 and not is_neutral:
                    self._accumulate_hga_stats(info_et)
                update_info = info_90
                extra_time_info = info_et
            else:
                (
                    mu_h_post,
                    sigma_h_post,
                    mu_a_post,
                    sigma_a_post,
                    update_info,
                ) = self._compute_updated_ratings(
                    score_h,
                    score_a,
                    is_neutral,
                    st_h.m,
                    st_h.sigma2,
                    st_a.m,
                    st_a.sigma2,
                    match_weight=match_weight,
                )
                sigma_h_post = self._apply_variance_bounds(sigma_h_post)
                sigma_a_post = self._apply_variance_bounds(sigma_a_post)
                if self.hga_rw_var_per_year > 0.0 and not is_neutral:
                    self._accumulate_hga_stats(update_info)

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
                    "exp_home_score": m_home_pre,
                    "exp_away_score": m_away_pre,
                    "lam_home_pre": float(debug["lamH"]),
                    "lam_away_pre": float(debug["lamA"]),
                    "nu_pre": float(debug["nu"]),
                    "eta_home_pre": eta_home_pre,
                    "eta_away_pre": eta_away_pre,
                    "f_home_pre": float(f_home_pre),
                    "f_away_pre": float(f_away_pre),
                    "g_home_pre": float(g_home_pre),
                    "g_away_pre": float(g_away_pre),
                    "fprime_home_pre": float(fprime_home_pre),
                    "fprime_away_pre": float(fprime_away_pre),
                    "dlogp_df_home_pre": float(update_info.get("dlogp_df_home_pre", np.nan)),
                    "dlogp_df_away_pre": float(update_info.get("dlogp_df_away_pre", np.nan)),
                    "match_weight": float(match_weight),
                    "home_games_played": int(games_played[home]),
                    "away_games_played": int(games_played[away]),
                    "min_games_played": int(min(games_played[home], games_played[away])),
                }
            )

            st_h.update_params(mu_h_post, sigma_h_post, t)
            st_a.update_params(mu_a_post, sigma_a_post, t)
            if self.recenter_params:
                self._recenter_params(t)
            mu_global_post = float(self.mu)
            self._mu_history[t] = float(self.mu)
            self._match_index += 1
            reason = update_info.get("reason")
            if reason == "max_steps":
                print(
                    "warning: max_steps reached "
                    f"date={t} {home} vs {away} score={score_h}-{score_a} "
                    f"neutral={bool(is_neutral)} "
                    f"iters={update_info.get('iters')} grad_norm={update_info.get('grad_norm')} "
                    f"rel_improve_last={update_info.get('rel_improve_last')} f={update_info.get('f')}"
                )
            if reason == "lm_failed_to_find_descent":
                print(
                    "warning: lm_failed_to_find_descent "
                    f"date={t} {home} vs {away} score={score_h}-{score_a} "
                    f"neutral={bool(is_neutral)} "
                    f"lm_retries={update_info.get('lm_retries')} "
                    f"lm_last={update_info.get('lm_last')} "
                    f"best_step_norm={update_info.get('best_step_norm')} "
                    f"best_f_trial={update_info.get('best_f_trial')} "
                    f"saw_finite_trial={update_info.get('saw_finite_trial')} "
                    f"saw_improve={update_info.get('saw_improve')}"
                )
            if extra_time_info is not None:
                reason_et = extra_time_info.get("reason")
                if reason_et == "max_steps":
                    print(
                        "warning: max_steps (extra_time) "
                        f"date={t} {home} vs {away} score={score_h}-{score_a} "
                        f"neutral={bool(is_neutral)} "
                        f"iters={extra_time_info.get('iters')} "
                        f"grad_norm={extra_time_info.get('grad_norm')} "
                        f"rel_improve_last={extra_time_info.get('rel_improve_last')} "
                        f"f={extra_time_info.get('f')}"
                    )
                if reason_et == "lm_failed_to_find_descent":
                    print(
                        "warning: lm_failed_to_find_descent (extra_time) "
                        f"date={t} {home} vs {away} score={score_h}-{score_a} "
                        f"neutral={bool(is_neutral)} "
                        f"lm_retries={extra_time_info.get('lm_retries')} "
                        f"lm_last={extra_time_info.get('lm_last')} "
                        f"best_step_norm={extra_time_info.get('best_step_norm')} "
                        f"best_f_trial={extra_time_info.get('best_f_trial')} "
                        f"saw_finite_trial={extra_time_info.get('saw_finite_trial')} "
                        f"saw_improve={extra_time_info.get('saw_improve')}"
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
            "exp_home_score",
            "exp_away_score",
            "lam_home_pre",
            "lam_away_pre",
            "nu_pre",
            "eta_home_pre",
            "eta_away_pre",
            "f_home_pre",
            "f_away_pre",
            "g_home_pre",
            "g_away_pre",
            "fprime_home_pre",
            "fprime_away_pre",
            "dlogp_df_home_pre",
            "dlogp_df_away_pre",
            "match_weight",
            "home_games_played",
            "away_games_played",
            "min_games_played",
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
        k = ms.shape[0]
        if k == 0:
            return
        if self.recenter_top_n is None:
            subset = ms
        else:
            upper = int(self.recenter_top_n)
            top_n = int(math.ceil(k / self.recenter_team_divisor))
            top_n = max(2, min(top_n, upper))
            top_n = min(top_n, k)
            scores = ms[:, 0] + ms[:, 1]
            idx = np.argsort(scores)[-top_n:]
            subset = ms[idx, :]
        if subset.size == 0:
            return
        mean_a, mean_d = subset.mean(axis=0)
        shift = np.array([mean_a, mean_d], dtype=float)
        if mean_a == 0.0 and mean_d == 0.0 and not self.maintain_param_history:
            return
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
            cross_inc = inc * self.cross_var_ratio
            st.sigma2 = np.array(st.sigma2, dtype=float) + np.array(
                [[inc, cross_inc], [cross_inc, inc]],
                dtype=float,
            )
            st.sigma2 = self._apply_variance_bounds(st.sigma2)
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
        fprimeH = info.get("hga_fprimeH")
        fprimeA = info.get("hga_fprimeA")
        match_weight = float(info.get("match_weight", 1.0))
        if match_weight <= 0.0:
            return
        if df_detaH is None or df_detaA is None or mH is None or mA is None:
            return
        grad_a = float(df_detaH) * match_weight
        grad_d = -float(df_detaA) * match_weight
        if fprimeH is None or fprimeA is None:
            curv_a = float(mH) * match_weight
            curv_d = float(mA) * match_weight
        else:
            curv_a = float(mH) * float(fprimeH) ** 2 * match_weight
            curv_d = float(mA) * float(fprimeA) ** 2 * match_weight
        if not self._hga_fixed_a:
            self._hga_grad_sum_a += grad_a
            self._hga_curv_sum_a += curv_a
        if not self._hga_fixed_d:
            self._hga_grad_sum_d += grad_d
            self._hga_curv_sum_d += curv_d

    def _finalize_hga_year(self, next_year=None):
        if self._hga_year is None:
            return
        if not self._hga_fixed_a:
            if self._hga_var_a <= 0.0:
                self._hga_var_a = self.hga_prior_var if self.hga_prior_var > 0.0 else 1.0
            prior_prec_a = 1.0 / self._hga_var_a
            post_prec_a = prior_prec_a + self._hga_curv_sum_a
            if post_prec_a > 0.0:
                self.a_hga = self.a_hga - (self._hga_grad_sum_a / post_prec_a)
                self._hga_var_a = 1.0 / post_prec_a
        if not self._hga_fixed_d:
            if self._hga_var_d <= 0.0:
                self._hga_var_d = self.hga_prior_var if self.hga_prior_var > 0.0 else 1.0
            prior_prec_d = 1.0 / self._hga_var_d
            post_prec_d = prior_prec_d + self._hga_curv_sum_d
            if post_prec_d > 0.0:
                self.d_hga = self.d_hga - (self._hga_grad_sum_d / post_prec_d)
                self._hga_var_d = 1.0 / post_prec_d

        self._hga_history[int(self._hga_year)] = (
            float(self.a_hga),
            float(self.d_hga),
            float(self._hga_var_a),
            float(self._hga_var_d),
        )
        self._hga_grad_sum_a = 0.0
        self._hga_grad_sum_d = 0.0
        self._hga_curv_sum_a = 0.0
        self._hga_curv_sum_d = 0.0
        if next_year is not None:
            delta_years = max(0, int(next_year) - int(self._hga_year))
            if self.hga_rw_var_per_year > 0.0:
                if not self._hga_fixed_a:
                    self._hga_var_a += self.hga_rw_var_per_year * delta_years
                if not self._hga_fixed_d:
                    self._hga_var_d += self.hga_rw_var_per_year * delta_years
            self._hga_year = int(next_year)
        self._current_hga_components()

    def _get_or_init_team(self, team: str, t: pd.Timestamp, confederation=None) -> TeamState:
        st = self.teams.get(team)
        if st is not None:
            return st
        year_delta = 0
        if self._first_match_year is not None:
            year_delta = int(t.year) - int(self._first_match_year)
        decay = self.mu_prior_decay
        init_mu = -float(year_delta) * decay
        confed_shift = 0.0
        if confederation is not None and not pd.isna(confederation):
            confed_shift = float(self.mu_prior_shift.get(confederation, 0.0))
        init_mu = init_mu + confed_shift
        st = TeamState(
            m=np.array([init_mu, init_mu], dtype=float),
            sigma2=self._apply_variance_bounds(self.init_var.copy()),
            last_update=t,
            maintain_param_history=self.maintain_param_history
        )
        if self.maintain_param_history:
            t_prev = t - dt.timedelta(days=1)
            st.history[t_prev] = {
                "m": np.array(st.m, dtype=float),
                "sigma2": np.array(st.sigma2, dtype=float),
            }
            if t_prev not in self._mu_history:
                self._mu_history[t_prev] = float(self.mu)
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
            return pd.DataFrame(columns=["year", "a_hga", "d_hga", "a_hga_var", "d_hga_var"])
        rows = []
        for year, hga_pair in sorted(self._hga_history.items()):
            if len(hga_pair) == 4:
                a_hga, d_hga, a_var, d_var = hga_pair
            else:
                a_hga, d_hga = hga_pair
                a_var = float("nan")
                d_var = float("nan")
            rows.append(
                {
                    "year": int(year),
                    "a_hga": float(a_hga),
                    "d_hga": float(d_hga),
                    "a_hga_var": float(a_var),
                    "d_hga_var": float(d_var),
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
        return 0.5 * (m + a - np.hypot(m - a, eps))

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
        match_weight=1.0,
    ):
        x = int(score_h)
        y = int(score_a)
        max_goals = MAX_GOALS
        x_censored = x > max_goals
        y_censored = y > max_goals
        if is_neutral:
            a_hga = 0.0
            d_hga = 0.0
        else:
            a_hga, d_hga = self._current_hga_components()

        mu0_team = np.array([mu_h_prior[0], mu_h_prior[1], mu_a_prior[0], mu_a_prior[1]], dtype=float)
        mu0 = mu0_team
        Sigma0 = np.zeros((4, 4), dtype=float)
        Sigma0[:2, :2] = np.array(sigma_h_prior, dtype=float)
        Sigma0[2:, 2:] = np.array(sigma_a_prior, dtype=float)

        Sigma0_inv = np.linalg.solve(
            Sigma0 + 1e-12 * np.eye(Sigma0.shape[0]), np.eye(Sigma0.shape[0])
        )
        match_weight = float(match_weight)
        if match_weight < 0.0:
            raise ValueError("match_weight must be non-negative")

        J = np.array(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 1.0, 0.0],
            ],
            dtype=float,
        )

        eps_smin = self.smin_eps_epsilon

        def _forward(theta):
            aH, dH, aA, dA = theta[:4]

            etaH = self.mu + (aH + a_hga) - dA
            etaA = self.mu + aA - (dH + d_hga)

            fH = float(etaH)
            fprimeH = 1.0
            fA = float(etaA)
            fprimeA = 1.0

            mH = safe_exp(fH)
            mA = safe_exp(fA)

            rho = float(self.rho)
            rho = min(max(rho, 0.0), 1.0 - 1e-9)

            smin = float(Model._smin_eps(mH, mA, eps=eps_smin))
            if smin < 0.0:
                smin = 0.0

            nu = rho * smin
            lamH = mH - nu
            lamA = mA - nu

            if lamH <= 0.0 or lamA <= 0.0 or nu < 0.0:
                return fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, False

            return fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, True

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

        log_fact_small = None
        if x_censored or y_censored:
            log_fact_small = np.array([log_factorial(i) for i in range(max_goals + 1)], dtype=float)

        def _poisson_pmf_k(lam: float):
            if lam <= 0.0:
                pmf = np.zeros(max_goals + 1, dtype=float)
                pmf[0] = 1.0
                return pmf
            k = np.arange(max_goals + 1, dtype=float)
            log_p = -lam + k * np.log(lam) - log_fact_small
            return np.exp(log_p)

        def _poisson_pmf_and_grad(lam: float):
            pmf = _poisson_pmf_k(lam)
            if lam <= 0.0:
                dpmf = np.zeros_like(pmf)
                dpmf[0] = -1.0
            else:
                k = np.arange(max_goals + 1, dtype=float)
                dpmf = pmf * (k / lam - 1.0)
            return pmf, dpmf

        def _poisson_survival_and_grad(pmf: np.ndarray, dpmf: np.ndarray, k: int):
            if k <= 0:
                return 1.0, 0.0
            if k > pmf.size:
                return 0.0, 0.0
            s = 1.0 - float(np.sum(pmf[:k]))
            ds = -float(np.sum(dpmf[:k]))
            return s, ds

        def _loglik_and_dlogp(mH, mA, nu, lamH, lamA):
            if not x_censored and not y_censored:
                logp, ubar = _loglik_and_ubar(mH, mA, nu, lamH, lamA)
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
                return float(logp), float(dlogp_dmH), float(dlogp_dmA)

            if lamH <= 0.0 or lamA <= 0.0 or nu < 0.0:
                return float("-inf"), 0.0, 0.0

            pmf_nu, dpmf_nu = _poisson_pmf_and_grad(nu)
            pmf_x, dpmf_x = _poisson_pmf_and_grad(lamH)
            pmf_y, dpmf_y = _poisson_pmf_and_grad(lamA)

            p = 0.0
            dp_dlamH = 0.0
            dp_dlamA = 0.0
            dp_dnu = 0.0

            if x_censored and not y_censored:
                for u in range(0, y + 1):
                    p_u = pmf_nu[u]
                    dp_u = dpmf_nu[u]
                    k_y = y - u
                    if k_y < 0:
                        continue
                    p_y = pmf_y[k_y]
                    dp_y = dpmf_y[k_y]
                    k = max_goals - u
                    s_x, ds_x = _poisson_survival_and_grad(pmf_x, dpmf_x, k)
                    term = p_u * p_y * s_x
                    p += term
                    dp_dlamH += p_u * p_y * ds_x
                    dp_dlamA += p_u * dp_y * s_x
                    dp_dnu += dp_u * p_y * s_x
            elif y_censored and not x_censored:
                for u in range(0, x + 1):
                    p_u = pmf_nu[u]
                    dp_u = dpmf_nu[u]
                    k_x = x - u
                    if k_x < 0:
                        continue
                    p_x = pmf_x[k_x]
                    dp_x = dpmf_x[k_x]
                    k = max_goals - u
                    s_y, ds_y = _poisson_survival_and_grad(pmf_y, dpmf_y, k)
                    term = p_u * p_x * s_y
                    p += term
                    dp_dlamH += p_u * dp_x * s_y
                    dp_dlamA += p_u * p_x * ds_y
                    dp_dnu += dp_u * p_x * s_y
            else:
                sum_pu = float(np.sum(pmf_nu[:max_goals]))
                sum_dpu = float(np.sum(dpmf_nu[:max_goals]))
                tail = 1.0 - sum_pu
                p += tail
                dp_dnu += -sum_dpu
                for u in range(0, max_goals):
                    p_u = pmf_nu[u]
                    dp_u = dpmf_nu[u]
                    k = max_goals - u
                    s_x, ds_x = _poisson_survival_and_grad(pmf_x, dpmf_x, k)
                    s_y, ds_y = _poisson_survival_and_grad(pmf_y, dpmf_y, k)
                    term = p_u * s_x * s_y
                    p += term
                    dp_dlamH += p_u * ds_x * s_y
                    dp_dlamA += p_u * s_x * ds_y
                    dp_dnu += dp_u * s_x * s_y

            if p <= 0.0 or not np.isfinite(p):
                return float("-inf"), 0.0, 0.0

            dlogp_dlamH = dp_dlamH / p
            dlogp_dlamA = dp_dlamA / p
            dlogp_dnu = dp_dnu / p

            nu_mH, nu_mA = _nu_partials(mH, mA)
            dlogp_dmH = (
                dlogp_dlamH * (1.0 - nu_mH)
                + dlogp_dlamA * (-nu_mH)
                + dlogp_dnu * nu_mH
            )
            dlogp_dmA = (
                dlogp_dlamA * (1.0 - nu_mA)
                + dlogp_dlamH * (-nu_mA)
                + dlogp_dnu * nu_mA
            )
            return float(np.log(p)), float(dlogp_dmH), float(dlogp_dmA)

        def _dlogp_df(theta):
            theta = np.asarray(theta, dtype=float)
            fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return None, None, None, None, None, None

            logp, dlogp_dmH, dlogp_dmA = _loglik_and_dlogp(mH, mA, nu, lamH, lamA)
            if not np.isfinite(logp):
                return None, None, None, None, None, None

            dlogp_dfH = mH * dlogp_dmH
            dlogp_dfA = mA * dlogp_dmA
            return float(dlogp_dfH), float(dlogp_dfA), float(mH), float(mA)

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
            diff = theta - mu0
            prior = 0.5 * float(diff.T @ Sigma0_inv @ diff)
            if match_weight == 0.0:
                return float(prior)

            fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return float("inf")

            logp, _dlogp_dmH, _dlogp_dmA = _loglik_and_dlogp(mH, mA, nu, lamH, lamA)
            nll = -logp
            return float(match_weight * nll + prior)

        def grad(theta):
            theta = np.asarray(theta, dtype=float)
            diff = theta - mu0
            g_prior = Sigma0_inv @ diff
            if match_weight == 0.0:
                return g_prior

            fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, ok = _forward(theta)

            if not ok:
                return g_prior

            logp, dlogp_dmH, dlogp_dmA = _loglik_and_dlogp(mH, mA, nu, lamH, lamA)
            if not np.isfinite(logp):
                return g_prior

            dlogp_dfH = mH * dlogp_dmH
            dlogp_dfA = mA * dlogp_dmA
            like_vec = np.array([dlogp_dfH, dlogp_dfA], dtype=float)
            J_full = np.zeros((2, 4), dtype=float)
            J_full[0, 0] = fprimeH
            J_full[0, 3] = -fprimeH
            J_full[1, 1] = -fprimeA
            J_full[1, 2] = fprimeA
            g_like = -J_full.T @ like_vec

            return match_weight * g_like + g_prior

        def _likelihood_partials(theta):
            theta = np.asarray(theta, dtype=float)
            fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return None

            logp, dlogp_dmH, dlogp_dmA = _loglik_and_dlogp(mH, mA, nu, lamH, lamA)
            if not np.isfinite(logp):
                return None

            dlogp_detaH = mH * dlogp_dmH * fprimeH
            dlogp_detaA = mA * dlogp_dmA * fprimeA

            df_detaH = -dlogp_detaH
            df_detaA = -dlogp_detaA
            return df_detaH, df_detaA, mH, mA, fprimeH, fprimeA

        def curv(theta):
            theta = np.asarray(theta, dtype=float)
            if match_weight == 0.0:
                return Sigma0_inv.copy()
            fH, fA, fprimeH, fprimeA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return Sigma0_inv.copy()

            H_eta = np.diag([mH, mA])
            J_full = np.zeros((2, 4), dtype=float)
            J_full[0, 0] = fprimeH
            J_full[0, 3] = -fprimeH
            J_full[1, 1] = -fprimeA
            J_full[1, 2] = fprimeA
            H_like = J_full.T @ H_eta @ J_full
            H = match_weight * H_like + Sigma0_inv
            return 0.5 * (H + H.T)

        pre_diag = {}
        dlogp_dfH_pre, dlogp_dfA_pre, mH_pre, mA_pre = _dlogp_df(mu0)
        if dlogp_dfH_pre is not None:
            pre_diag["dlogp_df_home_pre"] = float(dlogp_dfH_pre) * match_weight
            pre_diag["dlogp_df_away_pre"] = float(dlogp_dfA_pre) * match_weight

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

        if pre_diag:
            info = dict(info)
            info.update(pre_diag)

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
        Sigma_a_post = Sigma_post[2:4, 2:4].copy()

        # Extra sanity diagnostics (you can comment these out later)
        g_at_map = grad(theta_map)
        info = dict(info)
        hga_partials = _likelihood_partials(theta_map)
        if hga_partials is not None:
            df_detaH, df_detaA, mH_like, mA_like, fprimeH, fprimeA = hga_partials
            info["hga_df_detaH"] = float(df_detaH)
            info["hga_df_detaA"] = float(df_detaA)
            info["hga_mH"] = float(mH_like)
            info["hga_mA"] = float(mA_like)
            info["hga_fprimeH"] = float(fprimeH)
            info["hga_fprimeA"] = float(fprimeA)
        info["match_weight"] = float(match_weight)
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

    def _score_matrix(
        self,
        mu_h: np.ndarray,
        mu_a: np.ndarray,
        is_neutral: bool,
        log_fact: np.ndarray,
        sigma_h=None,
        sigma_a=None,
        lognormal_correction: bool = False,
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

        mH_raw = safe_exp(float(etaH))
        mA_raw = safe_exp(float(etaA))
        if lognormal_correction and sigma_h is not None and sigma_a is not None:
            var_etaH = float(sigma_h[0, 0]) + float(sigma_a[1, 1])
            var_etaA = float(sigma_a[0, 0]) + float(sigma_h[1, 1])
            var_etaH = max(var_etaH, 0.0)
            var_etaA = max(var_etaA, 0.0)
            mH = safe_exp(float(etaH + 0.5 * var_etaH))
            mA = safe_exp(float(etaA + 0.5 * var_etaA))
        else:
            mH = mH_raw
            mA = mA_raw

        rho = float(self.rho)
        rho = min(max(rho, 0.0), 1.0 - 1e-9)

        smin = float(Model._smin_eps(mH, mA, eps=self.smin_eps_epsilon))
        if smin < 0.0:
            smin = 0.0

        nu = rho * smin
        lamH = mH - nu
        lamA = mA - nu

        K = MAX_GOALS
        score_matrix = np.zeros((K + 1, K + 1), dtype=float)
        if lamH <= 0.0 or lamA <= 0.0 or nu < 0.0:
            return score_matrix, {
                "mH": mH,
                "mA": mA,
                "lamH": lamH,
                "lamA": lamA,
                "nu": nu,
                "mH_raw": mH_raw,
                "mA_raw": mA_raw,
            }

        p_x = self._poisson_pmf(lamH, log_fact)
        p_y = self._poisson_pmf(lamA, log_fact)

        if nu < 1e-14:
            joint = np.outer(p_x[:K], p_y[:K])
            tail_x = 1.0 - float(p_x[:K].sum())
            tail_y = 1.0 - float(p_y[:K].sum())
            score_matrix[:K, :K] = joint
            score_matrix[K, :K] = tail_x * p_y[:K]
            score_matrix[:K, K] = p_x[:K] * tail_y
            score_matrix[K, K] = tail_x * tail_y
            return score_matrix, {
                "mH": mH,
                "mA": mA,
                "lamH": lamH,
                "lamA": lamA,
                "nu": nu,
                "mH_raw": mH_raw,
                "mA_raw": mA_raw,
            }

        p_u = self._poisson_pmf(nu, log_fact)
        joint = np.zeros((K, K), dtype=float)
        for u in range(K):
            if p_u[u] == 0.0:
                continue
            px = p_x[: K - u]
            py = p_y[: K - u]
            joint[u:, u:] += p_u[u] * np.outer(px, py)
        score_matrix[:K, :K] = joint

        cdf_x = np.cumsum(p_x)
        cdf_y = np.cumsum(p_y)
        tail_x = np.ones(K + 1, dtype=float)
        tail_y = np.ones(K + 1, dtype=float)
        tail_x[1:] = 1.0 - cdf_x[:-1]
        tail_y[1:] = 1.0 - cdf_y[:-1]

        for j in range(K):
            prob = 0.0
            for u in range(j + 1):
                if p_u[u] == 0.0:
                    continue
                prob += p_u[u] * p_y[j - u] * tail_x[K - u]
            score_matrix[K, j] = prob

        for i in range(K):
            prob = 0.0
            for u in range(i + 1):
                if p_u[u] == 0.0:
                    continue
                prob += p_u[u] * p_x[i - u] * tail_y[K - u]
            score_matrix[i, K] = prob

        mass = float(score_matrix[:K, :K].sum() + score_matrix[K, :K].sum() + score_matrix[:K, K].sum())
        score_matrix[K, K] = max(0.0, 1.0 - mass)

        return score_matrix, {
            "mH": mH,
            "mA": mA,
            "lamH": lamH,
            "lamA": lamA,
            "nu": nu,
            "mH_raw": mH_raw,
            "mA_raw": mA_raw,
        }

    def _result_probabilities(
        self,
        mu_h: np.ndarray,
        mu_a: np.ndarray,
        is_neutral: bool,
        score_h: int,
        score_a: int,
        log_fact: np.ndarray,
    ):
        score_matrix, debug = self._score_matrix(mu_h, mu_a, is_neutral, log_fact)
        total = float(score_matrix.sum())
        if total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, debug

        p_home = float(np.tril(score_matrix, k=-1).sum())
        p_draw = float(np.trace(score_matrix))
        p_away = float(np.triu(score_matrix, k=1).sum())
        p_score = (
            float(score_matrix[score_h, score_a])
            if score_h <= MAX_GOALS and score_a <= MAX_GOALS
            else 0.0
        )

        inv_total = 1.0 / total
        return (
            p_home * inv_total,
            p_draw * inv_total,
            p_away * inv_total,
            p_score * inv_total,
            debug,
        )

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
        mode: str = "point",
        n_samples: int = 2000,
        random_state=None,
        requires_result: bool = False,
        lognormal_score_correction=None,
    ):
        if not self.is_fit:
            raise ValueError("model must be fit before predicting")
        st_h = self.teams.get(home_team)
        st_a = self.teams.get(away_team)
        if st_h is None or st_a is None:
            missing = [t for t, st in ((home_team, st_h), (away_team, st_a)) if st is None]
            raise ValueError(f"unknown team(s): {missing}")

        mode = str(mode).lower()
        if mode not in {"point", "mc", "monte_carlo"}:
            raise ValueError("mode must be 'point' or 'mc'/'monte_carlo'")
        use_lognormal_correction = (
            self.lognormal_score_correction
            if lognormal_score_correction is None
            else bool(lognormal_score_correction)
        )
        if use_lognormal_correction and mode != "point":
            raise ValueError(
                "lognormal_score_correction is only supported in point mode; "
                "use mode='mc' to integrate uncertainty."
            )

        log_fact = np.array([log_factorial(i) for i in range(MAX_GOALS + 1)], dtype=float)
        if mode == "point":
            score_matrix, debug = self._score_matrix(
                st_h.m,
                st_a.m,
                is_neutral,
                log_fact,
                sigma_h=st_h.sigma2,
                sigma_a=st_a.sigma2,
                lognormal_correction=use_lognormal_correction,
            )
            total = float(score_matrix.sum())
            if total <= 0.0:
                return {
                    "p_home": 0.0,
                    "p_draw": 0.0,
                    "p_away": 0.0,
                    "score_matrix": score_matrix,
                    "exp_home_score": float(debug.get("mH", 0.0)),
                    "exp_away_score": float(debug.get("mA", 0.0)),
                    "lam_home": float(debug.get("lamH", 0.0)),
                    "lam_away": float(debug.get("lamA", 0.0)),
                    "nu": float(debug.get("nu", 0.0)),
                }

            inv_total = 1.0 / total
            score_matrix = score_matrix * inv_total
            p_home = float(np.tril(score_matrix, k=-1).sum())
            p_draw = float(np.trace(score_matrix))
            p_away = float(np.triu(score_matrix, k=1).sum())

            output = {
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "score_matrix": score_matrix,
                "exp_home_score": float(debug.get("mH", 0.0)),
                "exp_away_score": float(debug.get("mA", 0.0)),
                "lam_home": float(debug.get("lamH", 0.0)),
                "lam_away": float(debug.get("lamA", 0.0)),
                "nu": float(debug.get("nu", 0.0)),
            }
            if requires_result:
                if is_neutral:
                    a_hga = 0.0
                    d_hga = 0.0
                else:
                    a_hga, d_hga = self._current_hga_components()
                skilldiff = (
                    self.mu + (st_h.m[0] + a_hga) - st_a.m[1]
                    - (self.mu + st_a.m[0] - (st_h.m[1] + d_hga))
                )
                mu_before = float(self.mu)
                self.mu = mu_before + math.log(self.extra_time_exp_score_mult)
                try:
                    score_matrix_et, _debug_et = self._score_matrix(
                        st_h.m,
                        st_a.m,
                        is_neutral,
                        log_fact,
                        sigma_h=st_h.sigma2,
                        sigma_a=st_a.sigma2,
                        lognormal_correction=use_lognormal_correction,
                    )
                finally:
                    self.mu = mu_before
                total_et = float(score_matrix_et.sum())
                if total_et > 0.0:
                    score_matrix_et = score_matrix_et / total_et
                else:
                    score_matrix_et = np.zeros_like(score_matrix_et)
                    score_matrix_et[0, 0] = 1.0
                p_home_et = float(np.tril(score_matrix_et, k=-1).sum())
                p_draw_et = float(np.trace(score_matrix_et))
                p_away_et = float(np.triu(score_matrix_et, k=1).sum())
                score_matrix_120 = np.zeros_like(score_matrix)
                for i in range(MAX_GOALS + 1):
                    for j in range(MAX_GOALS + 1):
                        p90 = score_matrix[i, j]
                        if p90 == 0.0:
                            continue
                        if i != j:
                            score_matrix_120[i, j] += p90
                            continue
                        for k in range(MAX_GOALS + 1):
                            for l in range(MAX_GOALS + 1):
                                pet = score_matrix_et[k, l]
                                if pet == 0.0:
                                    continue
                                h = min(i + k, MAX_GOALS)
                                a = min(j + l, MAX_GOALS)
                                score_matrix_120[h, a] += p90 * pet
                p_home_120 = p_home + p_draw * p_home_et
                p_draw_120 = p_draw * p_draw_et
                p_away_120 = p_away + p_draw * p_away_et
                p_home_pen = 1.0 / (1.0 + safe_exp(-self.shootout_skilldiff_coef * skilldiff))
                output.update(
                    {
                        "p_home_90": p_home,
                        "p_draw_90": p_draw,
                        "p_away_90": p_away,
                        "score_matrix_120": score_matrix_120,
                        "p_home_120": p_home_120,
                        "p_draw_120": p_draw_120,
                        "p_away_120": p_away_120,
                        "p_home_pens": p_home_120 + p_draw_120 * p_home_pen,
                        "p_away_pens": p_away_120 + p_draw_120 * (1.0 - p_home_pen),
                        "p_home_penalty": p_home_pen,
                    }
                )
            return output

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        def _prepare_cov(cov):
            cov = np.array(cov, dtype=float)
            cov = 0.5 * (cov + cov.T)
            w, v = np.linalg.eigh(cov)
            w = np.clip(w, 1e-10, None)
            return (v * w) @ v.T

        rng = np.random.default_rng(random_state)
        cov_h = _prepare_cov(st_h.sigma2)
        cov_a = _prepare_cov(st_a.sigma2)
        draws_h = rng.multivariate_normal(st_h.m, cov_h, size=n_samples)
        draws_a = rng.multivariate_normal(st_a.m, cov_a, size=n_samples)

        score_matrix_sum = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1), dtype=float)
        score_matrix_120_sum = np.zeros_like(score_matrix_sum)
        p_home_sum = 0.0
        p_draw_sum = 0.0
        p_away_sum = 0.0
        p_home_120_sum = 0.0
        p_draw_120_sum = 0.0
        p_away_120_sum = 0.0
        p_home_pens_sum = 0.0
        p_away_pens_sum = 0.0
        p_home_penalty_sum = 0.0
        exp_home_sum = 0.0
        exp_away_sum = 0.0
        lam_home_sum = 0.0
        lam_away_sum = 0.0
        nu_sum = 0.0
        valid = 0

        if is_neutral:
            a_hga = 0.0
            d_hga = 0.0
        else:
            a_hga, d_hga = self._current_hga_components()
        for mu_h, mu_a in zip(draws_h, draws_a):
            score_matrix, debug = self._score_matrix(mu_h, mu_a, is_neutral, log_fact)
            total = float(score_matrix.sum())
            if total <= 0.0:
                continue
            score_matrix = score_matrix / total
            score_matrix_sum += score_matrix
            p_home = float(np.tril(score_matrix, k=-1).sum())
            p_draw = float(np.trace(score_matrix))
            p_away = float(np.triu(score_matrix, k=1).sum())
            p_home_sum += p_home
            p_draw_sum += p_draw
            p_away_sum += p_away
            exp_home_sum += float(debug.get("mH", 0.0))
            exp_away_sum += float(debug.get("mA", 0.0))
            lam_home_sum += float(debug.get("lamH", 0.0))
            lam_away_sum += float(debug.get("lamA", 0.0))
            nu_sum += float(debug.get("nu", 0.0))
            if requires_result:
                skilldiff = (
                    self.mu + (mu_h[0] + a_hga) - mu_a[1]
                    - (self.mu + mu_a[0] - (mu_h[1] + d_hga))
                )
                mu_before = float(self.mu)
                self.mu = mu_before + math.log(self.extra_time_exp_score_mult)
                try:
                    score_matrix_et, _debug_et = self._score_matrix(
                        mu_h, mu_a, is_neutral, log_fact
                    )
                finally:
                    self.mu = mu_before
                total_et = float(score_matrix_et.sum())
                if total_et > 0.0:
                    score_matrix_et = score_matrix_et / total_et
                else:
                    score_matrix_et = np.zeros_like(score_matrix_et)
                    score_matrix_et[0, 0] = 1.0
                p_home_et = float(np.tril(score_matrix_et, k=-1).sum())
                p_draw_et = float(np.trace(score_matrix_et))
                p_away_et = float(np.triu(score_matrix_et, k=1).sum())
                score_matrix_120 = np.zeros_like(score_matrix)
                for i in range(MAX_GOALS + 1):
                    for j in range(MAX_GOALS + 1):
                        p90 = score_matrix[i, j]
                        if p90 == 0.0:
                            continue
                        if i != j:
                            score_matrix_120[i, j] += p90
                            continue
                        for k in range(MAX_GOALS + 1):
                            for l in range(MAX_GOALS + 1):
                                pet = score_matrix_et[k, l]
                                if pet == 0.0:
                                    continue
                                h = min(i + k, MAX_GOALS)
                                a = min(j + l, MAX_GOALS)
                                score_matrix_120[h, a] += p90 * pet
                score_matrix_120_sum += score_matrix_120
                p_home_120 = p_home + p_draw * p_home_et
                p_draw_120 = p_draw * p_draw_et
                p_away_120 = p_away + p_draw * p_away_et
                p_home_pen = 1.0 / (1.0 + safe_exp(-self.shootout_skilldiff_coef * skilldiff))
                p_home_120_sum += p_home_120
                p_draw_120_sum += p_draw_120
                p_away_120_sum += p_away_120
                p_home_pens_sum += p_home_120 + p_draw_120 * p_home_pen
                p_away_pens_sum += p_away_120 + p_draw_120 * (1.0 - p_home_pen)
                p_home_penalty_sum += p_home_pen
            valid += 1

        if valid == 0:
            return {
                "p_home": 0.0,
                "p_draw": 0.0,
                "p_away": 0.0,
                "score_matrix": score_matrix_sum,
                "exp_home_score": 0.0,
                "exp_away_score": 0.0,
                "lam_home": 0.0,
                "lam_away": 0.0,
                "nu": 0.0,
                "n_samples": 0,
            }

        inv_valid = 1.0 / valid
        output = {
            "p_home": p_home_sum * inv_valid,
            "p_draw": p_draw_sum * inv_valid,
            "p_away": p_away_sum * inv_valid,
            "score_matrix": score_matrix_sum * inv_valid,
            "exp_home_score": exp_home_sum * inv_valid,
            "exp_away_score": exp_away_sum * inv_valid,
            "lam_home": lam_home_sum * inv_valid,
            "lam_away": lam_away_sum * inv_valid,
            "nu": nu_sum * inv_valid,
            "n_samples": valid,
        }
        if requires_result:
            output.update(
                {
                    "p_home_90": output["p_home"],
                    "p_draw_90": output["p_draw"],
                    "p_away_90": output["p_away"],
                    "score_matrix_120": score_matrix_120_sum * inv_valid,
                    "p_home_120": p_home_120_sum * inv_valid,
                    "p_draw_120": p_draw_120_sum * inv_valid,
                    "p_away_120": p_away_120_sum * inv_valid,
                    "p_home_pens": p_home_pens_sum * inv_valid,
                    "p_away_pens": p_away_pens_sum * inv_valid,
                    "p_home_penalty": p_home_penalty_sum * inv_valid,
                }
            )
        return output

    def predict_match_extra_time(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
        mode: str = "point",
        n_samples: int = 2000,
        random_state=None,
        lognormal_score_correction=None,
    ):
        output = self.predict_match(
            home_team,
            away_team,
            is_neutral=is_neutral,
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            requires_result=True,
            lognormal_score_correction=lognormal_score_correction,
        )
        keys = [
            "p_home_90",
            "p_draw_90",
            "p_away_90",
            "p_home_120",
            "p_draw_120",
            "p_away_120",
            "score_matrix_120",
        ]
        return {k: output[k] for k in keys if k in output}

    def predict_match_penalties(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
        mode: str = "point",
        n_samples: int = 2000,
        random_state=None,
        lognormal_score_correction=None,
    ):
        output = self.predict_match(
            home_team,
            away_team,
            is_neutral=is_neutral,
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            requires_result=True,
            lognormal_score_correction=lognormal_score_correction,
        )
        keys = [
            "p_home_120",
            "p_draw_120",
            "p_away_120",
            "p_home_pens",
            "p_away_pens",
            "p_home_penalty",
        ]
        return {k: output[k] for k in keys if k in output}


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

def calibrate_et(res):
    r = res.query("had_extra_time and score_reconciled and year >= 1980").copy()
    r["home_score_et"] = r["home_score_120"] - r["home_score_90"]
    r["away_score_et"] = r["away_score_120"] - r["away_score_90"]
    r["exp_home_score_et"] = 0.23 * r["exp_home_score"]
    r["exp_away_score_et"] = 0.23 * r["exp_away_score"]
    # s = r[["exp_home_score", "exp_away_score", "home_score_et", "away_score_et"]].sum()
    # s.loc["home_score_et"] / s.loc["exp_home_score"], s.loc["away_score_et"] / s.loc["exp_away_score"]
    # r.set_index("date")[["home_score_et", "exp_home_score_et", "away_score_et", "exp_away_score_et"]].cumsum().plot()
    r["better_score_et"] = np.where(r.exp_home_score_et > r.exp_away_score_et, r.home_score_et, r.away_score_et)
    r["worse_score_et"] = np.where(r.exp_home_score_et <= r.exp_away_score_et, r.home_score_et, r.away_score_et)
    r["better_exp_score_et"] = np.where(r.exp_home_score_et > r.exp_away_score_et, r.exp_home_score_et, r.exp_away_score_et)
    r["worse_exp_score_et"] = np.where(r.exp_home_score_et <= r.exp_away_score_et, r.exp_home_score_et, r.exp_away_score_et)
    r.iloc[:,-4:].cumsum().plot()

def calibrate_pens(res):
    pens = res.query("shootout_winner.notna()").copy()
    pens = pens.eval("home_win = shootout_winner == home_team")
    pens["skilldiff"] = pens["eta_home_pre"] - pens["eta_away_pre"]
    pens["ev_home"] = pens["p_home"] + 0.5 * pens["p_draw"]

    import statsmodels.api as sm

    df = pens[["home_win", "skilldiff"]].dropna().copy()
    X = df[["skilldiff"]]  # no constant → intercept fixed at 0
    y = df["home_win"].astype(int)

    logit = sm.Logit(y, X).fit(disp=False)
    logit.summary2().tables[1][["Coef.", "P>|z|"]]
