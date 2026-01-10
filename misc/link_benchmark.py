import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import Model


def bucket_summary(df, exp_col, actual_col, q=20):
    bins = pd.qcut(df[exp_col].round(3), q=q, duplicates="drop")
    summary = df.groupby(bins, observed=False)[[actual_col, exp_col]].mean()
    summary["resid"] = summary[actual_col] - summary[exp_col]
    counts = df.groupby(bins, observed=False)[exp_col].size().rename("count")
    return summary.join(counts)


def print_head_tail(summary, name, n=3):
    print(f"\n{name} (lowest {n} bins):")
    print(summary.head(n).round(3))
    print(f"\n{name} (highest {n} bins):")
    print(summary.tail(n).round(3))
    print(
        f"\n{name} mean abs resid: {summary['resid'].abs().mean():.3f} "
        f"(mean resid: {summary['resid'].mean():.3f})"
    )


def main():
    results = pd.read_csv("match_results/results_clean.csv", parse_dates=["date"])

    model = Model(
        maintain_param_history=False,
        recenter_params=True,
        variance_per_year=0.2,
        rho=0.1,
        hga=0.2,
        hga_prior_var=5e-3,
        hga_rw_var_per_year=1e-3,
    )

    res = model.fit(results)

    R = res.query("date > '2000' and home_games_played >= 10 and away_games_played >= 10").copy()
    R["home_score_clipped"] = R.home_score.clip(0, 8)
    R["away_score_clipped"] = R.away_score.clip(0, 8)
    R["total_score_clipped"] = R["home_score_clipped"] + R["away_score_clipped"]
    R["exp_total_score"] = R["exp_home_score"] + R["exp_away_score"]

    rng = np.random.default_rng(0)
    sim_u = rng.poisson(R["nu_pre"].to_numpy())
    sim_h = rng.poisson(R["lam_home_pre"].to_numpy())
    sim_a = rng.poisson(R["lam_away_pre"].to_numpy())
    R["sim_home_score"] = sim_h + sim_u
    R["sim_away_score"] = sim_a + sim_u
    R["sim_home_score_clipped"] = R["sim_home_score"].clip(0, 8)
    R["sim_away_score_clipped"] = R["sim_away_score"].clip(0, 8)

    q = 20
    non_neutral = R.query("~neutral")
    neutral = R.query("neutral")

    home_summary = bucket_summary(non_neutral, "exp_home_score", "home_score_clipped", q=q)
    away_summary = bucket_summary(non_neutral, "exp_away_score", "away_score_clipped", q=q)
    neutral_home_summary = bucket_summary(neutral, "exp_home_score", "home_score_clipped", q=q)
    neutral_away_summary = bucket_summary(neutral, "exp_away_score", "away_score_clipped", q=q)

    sim_home_summary = bucket_summary(non_neutral, "exp_home_score", "sim_home_score_clipped", q=q)
    sim_away_summary = bucket_summary(non_neutral, "exp_away_score", "sim_away_score_clipped", q=q)
    sim_neutral_home_summary = bucket_summary(neutral, "exp_home_score", "sim_home_score_clipped", q=q)
    sim_neutral_away_summary = bucket_summary(neutral, "exp_away_score", "sim_away_score_clipped", q=q)

    print_head_tail(home_summary, "Home (non-neutral)")
    print_head_tail(away_summary, "Away (non-neutral)")
    print_head_tail(neutral_home_summary, "Home (neutral)")
    print_head_tail(neutral_away_summary, "Away (neutral)")

    print_head_tail(sim_home_summary, "Sim Home (non-neutral)")
    print_head_tail(sim_away_summary, "Sim Away (non-neutral)")
    print_head_tail(sim_neutral_home_summary, "Sim Home (neutral)")
    print_head_tail(sim_neutral_away_summary, "Sim Away (neutral)")


if __name__ == "__main__":
    main()
