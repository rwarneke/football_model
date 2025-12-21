import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.special import logsumexp


EPSILON = 1e-10


class Optimiser:
    @staticmethod
    def fisher_scoring_armijo(
        f,
        grad,
        curv,
        init_val,
        tol_grad=1e-6,
        tol_step=1e-10,
        max_steps=500,
        alpha_max=1.0,
        beta=0.5,
        c1=1e-4,
        max_backtracks=50,
        lm_damping=0.0,
        lm_increase=10.0,
        lm_decrease=10.0,
        lm_max=1e12,
        cov_jitter=1e-9,
        return_cov=True,
        use_pinv_for_cov=True,
    ):
        """
        Minimise f(x) using damped Fisher scoring / Gauss–Newton with Armijo backtracking.

        f:      R^n -> R
        grad:   R^n -> R^n
        curv:   R^n -> R^{n x n}  (PSD curvature approximation, ideally likelihood-curv + prior-precision)
        """

        def _sym(A: np.ndarray) -> np.ndarray:
            return 0.5 * (A + A.T)

        def _safe_cov_from_curv(H: np.ndarray, I: np.ndarray) -> np.ndarray:
            Hs = _sym(np.array(H, dtype=float))
            Hj = Hs + cov_jitter * I
            if use_pinv_for_cov:
                return np.linalg.pinv(Hj)
            try:
                return np.linalg.inv(Hj)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(Hj)

        x = np.atleast_1d(np.array(init_val, dtype=float))
        fx = float(f(x))

        n = x.shape[0]
        I = np.eye(n)
        lam = float(lm_damping)

        for it in range(max_steps):
            g = np.atleast_1d(np.array(grad(x), dtype=float))
            gnorm = float(np.linalg.norm(g))

            if (not np.isfinite(fx)) or (not np.all(np.isfinite(g))):
                H_here = np.array(curv(x), dtype=float)
                cov = _safe_cov_from_curv(H_here, I) if return_cov else None
                info = {"converged": False, "reason": "non_finite_state", "iters": it, "f": fx, "grad_norm": gnorm}
                return (x, info, cov) if return_cov else (x, info)

            if gnorm < tol_grad:
                if return_cov:
                    H_here = np.array(curv(x), dtype=float)
                    cov = _safe_cov_from_curv(H_here, I)
                    return x, {"converged": True, "iters": it, "f": fx, "grad_norm": gnorm}, cov
                return x, {"converged": True, "iters": it, "f": fx, "grad_norm": gnorm}

            H = _sym(np.array(curv(x), dtype=float))

            p = None
            while True:
                Hp = H + lam * I
                try:
                    p_try = np.linalg.solve(Hp, -g)
                except np.linalg.LinAlgError:
                    lam = max(1e-12, (lam if lam > 0 else 1e-12) * lm_increase)
                    if lam > lm_max:
                        if return_cov:
                            cov = _safe_cov_from_curv(H, I)
                            return x, {"converged": False, "reason": "curvature_solve_failed", "iters": it, "f": fx, "grad_norm": gnorm}, cov
                        return x, {"converged": False, "reason": "curvature_solve_failed", "iters": it, "f": fx, "grad_norm": gnorm}

                if float(np.dot(g, p_try)) < 0.0:
                    p = p_try
                    break

                lam = max(1e-12, (lam if lam > 0 else 1e-12) * lm_increase)
                if lam > lm_max:
                    if return_cov:
                        cov = _safe_cov_from_curv(H, I)
                        return x, {"converged": False, "reason": "non_descent_direction", "iters": it, "f": fx, "grad_norm": gnorm}, cov
                    return x, {"converged": False, "reason": "non_descent_direction", "iters": it, "f": fx, "grad_norm": gnorm}

            alpha = float(alpha_max)
            gp = float(np.dot(g, p))  # negative for descent

            accepted = False
            for _ in range(max_backtracks):
                x_trial = x + alpha * p

                if np.all(x_trial == x):
                    accepted = False
                    break

                if not np.all(np.isfinite(x_trial)):
                    alpha *= beta
                    continue

                f_trial = float(f(x_trial))
                if not np.isfinite(f_trial):
                    alpha *= beta
                    continue

                if f_trial <= fx + c1 * alpha * gp:
                    accepted = True
                    x_next, fx_next = x_trial, f_trial
                    break

                alpha *= beta

            if not accepted:
                if return_cov:
                    cov = _safe_cov_from_curv(np.array(curv(x), dtype=float), I)
                    return x, {"converged": False, "reason": "line_search_failed", "iters": it, "f": fx, "grad_norm": gnorm}, cov
                return x, {"converged": False, "reason": "line_search_failed", "iters": it, "f": fx, "grad_norm": gnorm}

            if lam > 0.0:
                lam = max(0.0, lam / lm_decrease)

            step_norm = float(np.linalg.norm(x_next - x))
            if step_norm < tol_step * (1.0 + float(np.linalg.norm(x))):
                x, fx = x_next, fx_next
                if return_cov:
                    H_final = np.array(curv(x), dtype=float)
                    cov = _safe_cov_from_curv(H_final, I)
                    g_final = np.atleast_1d(np.array(grad(x), dtype=float))
                    return x, {"converged": True, "iters": it + 1, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}, cov
                g_final = np.atleast_1d(np.array(grad(x), dtype=float))
                return x, {"converged": True, "iters": it + 1, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}

            x, fx = x_next, fx_next

        if return_cov:
            H_final = np.array(curv(x), dtype=float)
            cov = _safe_cov_from_curv(H_final, I)
            g_final = np.atleast_1d(np.array(grad(x), dtype=float))
            return x, {"converged": False, "reason": "max_steps", "iters": max_steps, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}, cov

        g_final = np.atleast_1d(np.array(grad(x), dtype=float))
        return x, {"converged": False, "reason": "max_steps", "iters": max_steps, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}


def log_factorial(n: int) -> float:
    return math.lgamma(n + 1.0)


class Model:
    def __init__(self, mu=0.0, a_hga=0.0, d_hga=0.0, rho=0.1, smin_eps_epsilon=1e-6):
        """
        mu:    Baseline number of goals
        a_hga: Increase in attacking quality due to home ground advantage
        d_hga: Increase in defensive quality due to home ground advnatage
        rho:   Controls correlation 
        """
        self.mu = float(mu)
        self.a_hga = float(a_hga)
        self.d_hga = float(d_hga)
        self.rho = float(rho)
        self.smin_eps_epsilon = float(smin_eps_epsilon)

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
        *,
        max_steps=500,
        tol_grad=1e-6,
        tol_step=1e-10,
        alpha_max=1.0,
        beta=0.5,
        c1=1e-4,
        max_backtracks=50,
        lm_damping=1e-2,     # default damping ON (helps low-score cases)
        cov_jitter=1e-9,
    ):
        x = int(score_h)
        y = int(score_a)

        a_hga = 0.0 if is_neutral else self.a_hga
        d_hga = 0.0 if is_neutral else self.d_hga

        mu0 = np.array([mu_h_prior[0], mu_h_prior[1], mu_a_prior[0], mu_a_prior[1]], dtype=float)

        Sigma0 = np.zeros((4, 4), dtype=float)
        Sigma0[:2, :2] = np.array(sigma_h_prior, dtype=float)
        Sigma0[2:, 2:] = np.array(sigma_a_prior, dtype=float)

        Sigma0_inv = np.linalg.inv(Sigma0)

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

        def curv(theta):
            theta = np.asarray(theta, dtype=float)
            etaH, etaA, mH, mA, nu, lamH, lamA, ok = _forward(theta)
            if not ok:
                return Sigma0_inv.copy()

            H_eta = np.diag([mH, mA])
            H_like = J.T @ H_eta @ J
            H = H_like + Sigma0_inv
            return 0.5 * (H + H.T)

        theta_map, info, cov = Optimiser.fisher_scoring_armijo(
            f=f,
            grad=grad,
            curv=curv,
            init_val=mu0,
            tol_grad=tol_grad,
            tol_step=tol_step,
            max_steps=max_steps,
            alpha_max=alpha_max,
            beta=beta,
            c1=c1,
            max_backtracks=max_backtracks,
            lm_damping=lm_damping,
            cov_jitter=cov_jitter,
            return_cov=True,
        )

        Sigma_post = cov
        mu_h_post = np.array([theta_map[0], theta_map[1]], dtype=float)
        mu_a_post = np.array([theta_map[2], theta_map[3]], dtype=float)
        Sigma_h_post = Sigma_post[:2, :2].copy()
        Sigma_a_post = Sigma_post[2:, 2:].copy()

        # Extra sanity diagnostics (you can comment these out later)
        g_at_map = grad(theta_map)
        info = dict(info)
        info["grad_norm_recomputed"] = float(np.linalg.norm(g_at_map))
        info["theta_map"] = theta_map.copy()

        return mu_h_post, Sigma_h_post, mu_a_post, Sigma_a_post, info
