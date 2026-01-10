import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.special import logsumexp



class Optimiser:
    @staticmethod
    def fisher_scoring_armijo(
        f,
        grad,
        curv,
        init_val,
        tol_grad=1e-6,
        tol_step=1e-10,
        tol_improve=1e-6,
        max_steps=50,
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

        rel_improve_last = None
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
                    continue

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
                    info = {"converged": False, "reason": "line_search_failed", "iters": it, "f": fx, "grad_norm": gnorm}
                    if rel_improve_last is not None:
                        info["rel_improve_last"] = float(rel_improve_last)
                    return x, info, cov
                info = {"converged": False, "reason": "line_search_failed", "iters": it, "f": fx, "grad_norm": gnorm}
                if rel_improve_last is not None:
                    info["rel_improve_last"] = float(rel_improve_last)
                return x, info

            if lam > 0.0:
                lam = max(0.0, lam / lm_decrease)

            step_norm = float(np.linalg.norm(x_next - x))
            rel_improve_last = abs(fx - fx_next) / (1.0 + abs(fx))
            if rel_improve_last < tol_improve:
                x, fx = x_next, fx_next
                if return_cov:
                    H_final = np.array(curv(x), dtype=float)
                    cov = _safe_cov_from_curv(H_final, I)
                    g_final = np.atleast_1d(np.array(grad(x), dtype=float))
                    info = {"converged": True, "iters": it + 1, "f": fx, "grad_norm": float(np.linalg.norm(g_final)), "reason": "small_improvement"}
                    info["rel_improve_last"] = float(rel_improve_last)
                    return x, info, cov
                g_final = np.atleast_1d(np.array(grad(x), dtype=float))
                info = {"converged": True, "iters": it + 1, "f": fx, "grad_norm": float(np.linalg.norm(g_final)), "reason": "small_improvement"}
                info["rel_improve_last"] = float(rel_improve_last)
                return x, info
            if step_norm < tol_step * (1.0 + float(np.linalg.norm(x))):
                x, fx = x_next, fx_next
                if return_cov:
                    H_final = np.array(curv(x), dtype=float)
                    cov = _safe_cov_from_curv(H_final, I)
                    g_final = np.atleast_1d(np.array(grad(x), dtype=float))
                    info = {"converged": True, "iters": it + 1, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}
                    if rel_improve_last is not None:
                        info["rel_improve_last"] = float(rel_improve_last)
                    return x, info, cov
                g_final = np.atleast_1d(np.array(grad(x), dtype=float))
                info = {"converged": True, "iters": it + 1, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}
                if rel_improve_last is not None:
                    info["rel_improve_last"] = float(rel_improve_last)
                return x, info

            x, fx = x_next, fx_next

        if return_cov:
            H_final = np.array(curv(x), dtype=float)
            cov = _safe_cov_from_curv(H_final, I)
            g_final = np.atleast_1d(np.array(grad(x), dtype=float))
            info = {"converged": False, "reason": "max_steps", "iters": max_steps, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}
            if rel_improve_last is not None:
                info["rel_improve_last"] = float(rel_improve_last)
            return x, info, cov

        g_final = np.atleast_1d(np.array(grad(x), dtype=float))
        info = {"converged": False, "reason": "max_steps", "iters": max_steps, "f": fx, "grad_norm": float(np.linalg.norm(g_final))}
        if rel_improve_last is not None:
            info["rel_improve_last"] = float(rel_improve_last)
        return x, info


    @staticmethod
    def fisher_scoring_armijo_fast(
        *,
        f,
        grad,
        curv,
        init_val,
        tol_grad=1e-6,
        tol_step=1e-10,
        max_steps=500,
        tol_improve=1e-6,
        alpha_max=1.0,          # ignored
        beta=0.5,               # ignored
        c1=1e-4,                # ignored
        max_backtracks=50,      # ignored
        lm_damping=1e-2,
        cov_jitter=1e-9,
        return_cov=True,        # IMPORTANT: match your call site expectations
    ):
        """
        Fast damped Fisher scoring / Gauss–Newton.
        - Fixed small number of outer steps (like notebook)
        - No Armijo backtracking
        - Adaptive LM damping with simple accept/reject to prevent blow-ups
        - Returns (x, info, cov) exactly like the Armijo version
        """

        def _sym(A: np.ndarray) -> np.ndarray:
            return 0.5 * (A + A.T)

        x = np.atleast_1d(np.array(init_val, dtype=float))
        n = x.shape[0]
        I = np.eye(n)

        # Limit how many times we’ll increase lambda per step
        MAX_LM_RETRIES = 8

        lam = float(lm_damping)
        fx = float(f(x))
        fx_prev = fx

        info = {
            "converged": True,
            "iters": 0,
            "f": fx,
            "grad_norm": None,
            "lm_final": lam,
            "reason": "ok",
        }

        rel_improve_last = None
        for it in range(max_steps):
            g = np.atleast_1d(np.array(grad(x), dtype=float))
            gnorm = float(np.linalg.norm(g))
            info["grad_norm"] = gnorm

            if (not np.isfinite(fx)) or (not np.all(np.isfinite(g))):
                info.update({"converged": False, "reason": "non_finite_state", "iters": it, "f": fx})
                break

            if gnorm < tol_grad:
                info.update({"iters": it, "f": fx})
                break

            H = _sym(np.array(curv(x), dtype=float))

            accepted = False
            lm_retries = 0
            best_f_trial = float("inf")
            best_step_norm = float("inf")
            saw_finite_trial = False
            saw_improve = False
            for _ in range(MAX_LM_RETRIES):
                lm_retries += 1
                H_eff = H + (lam + cov_jitter) * I
                try:
                    # CRITICAL: minus sign for minimisation
                    step = np.linalg.solve(H_eff, -g)
                except np.linalg.LinAlgError:
                    lam = max(1e-12, 10.0 * (lam if lam > 0 else 1e-12))
                    continue

                step_norm = float(np.linalg.norm(step))
                if step_norm < best_step_norm:
                    best_step_norm = step_norm
                if step_norm < tol_step * (1.0 + float(np.linalg.norm(x))):
                    accepted = True
                    x = x + step
                    fx = float(f(x))
                    break

                x_trial = x + step
                if not np.all(np.isfinite(x_trial)):
                    lam = max(1e-12, 10.0 * (lam if lam > 0 else 1e-12))
                    continue

                f_trial = float(f(x_trial))
                # Simple acceptance: must be finite and improve objective
                if np.isfinite(f_trial) and (f_trial <= fx):
                    x, fx = x_trial, f_trial
                    # if it worked, relax damping a touch
                    lam = max(0.0, lam / 10.0)
                    accepted = True
                    break
                if np.isfinite(f_trial):
                    saw_finite_trial = True
                    if f_trial <= fx:
                        saw_improve = True
                    if f_trial < best_f_trial:
                        best_f_trial = f_trial

                # otherwise increase damping and try again
                lam = max(1e-12, 10.0 * (lam if lam > 0 else 1e-12))

            info["iters"] = it + 1
            info["f"] = fx
            info["lm_final"] = lam

            if not accepted:
                info.update(
                    {
                        "converged": False,
                        "reason": "lm_failed_to_find_descent",
                        "lm_retries": lm_retries,
                        "lm_last": lam,
                        "best_step_norm": best_step_norm,
                        "best_f_trial": best_f_trial,
                        "saw_finite_trial": saw_finite_trial,
                        "saw_improve": saw_improve,
                    }
                )
                break

            denom = 1.0 + abs(fx_prev)
            rel_improve_last = abs(fx_prev - fx) / denom
            if rel_improve_last < tol_improve:
                info.update({"iters": it + 1, "f": fx, "reason": "small_improvement"})
                break
            fx_prev = fx
        else:
            info.update({"converged": False, "reason": "max_steps", "iters": max_steps, "f": fx})

        if rel_improve_last is not None:
            info["rel_improve_last"] = float(rel_improve_last)

        cov = None
        if return_cov:
            try:
                H_final = _sym(np.array(curv(x), dtype=float)) + cov_jitter * I
                cov = np.linalg.inv(H_final)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(H_final)

        return x, info, cov
