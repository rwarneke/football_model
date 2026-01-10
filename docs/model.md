# Model Math

This document describes the mathematical model and parameterization used for match scores.

## Latent team strengths

For each team \(i\), we maintain a latent attack and defense vector:

```
theta_i = [a_i, d_i]^T
```

where higher `a_i` increases scoring and higher `d_i` decreases opponent scoring.

## Expected score rates

For a match between home team `H` and away team `A`:

```
m_H = exp( mu + (a_H + a_HGA) - d_A )
m_A = exp( mu + a_A - (d_H + d_HGA) )
```

If the match is neutral, `a_HGA = d_HGA = 0`.

## Score distribution (bivariate Poisson)

Scores are modeled with a bivariate Poisson:

```
X_H = U + V_H
X_A = U + V_A
U ~ Pois(nu)
V_H ~ Pois(lambda_H)
V_A ~ Pois(lambda_A)
```

with

```
nu = rho * min(m_H, m_A)
lambda_H = m_H - nu
lambda_A = m_A - nu
```

This yields:

```
E[X_H] = m_H
E[X_A] = m_A
Corr(X_H, X_A) = rho * sqrt( min(m_H, m_A) / max(m_H, m_A) )
```

## Score censoring (10+ bucket)

We cap scores at `MAX_GOALS = 10` when computing the score matrix and likelihood.
Any score greater than 10 is treated as a `10+` bucket. In the likelihood, this
is handled as right-censoring in the corresponding margin(s).

## Priors for new teams

New teams are initialized as:

```
theta_i ~ N( [mu_0, mu_0]^T, Sigma_0 )
```

where:

```
mu_0 = - year_delta * mu_prior_decay + confed_shift
```

`year_delta` is the number of years after the first match year. `confed_shift` is an optional confederation-specific offset.

`Sigma_0` is the prior covariance for attack and defense.

## Time evolution and inactivity

Between matches, uncertainty increases as:

```
Sigma_i(t+) = Sigma_i(t) + variance_per_year * delta_years * I
```

Optionally, inactivity penalizes means:

```
theta_i(t+) = theta_i(t) - inactivity_decay_per_year * inactive_years * [1, 1]^T
```

## Home ground advantage (HGA)

`a_HGA` and `d_HGA` are treated as latent parameters with Gaussian priors and an optional random walk:

```
Var(a_HGA) += hga_rw_var_per_year * delta_years
Var(d_HGA) += hga_rw_var_per_year * delta_years
```

## Extra time handling

When a match is marked `had_extra_time=True` and the score breakdown is
reconciled, the update is split into two stages:

1) A 90-minute update with the observed `(x_H_90, x_A_90)` and expected scores
   `(m_H, m_A)` from the regular-time model above.
2) An extra-time update with the observed ET goals
   `(x_H_ET, x_A_ET) = (x_H_120 - x_H_90, x_A_120 - x_A_90)` and expected
   scores scaled by:

```
m_H_ET = extra_time_exp_score_mult * m_H
m_A_ET = extra_time_exp_score_mult * m_A
```

In log-rate terms this is equivalent to adding `log(extra_time_exp_score_mult)`
to the global `mu` during the ET update.

## Penalties (shootout) model

To convert a 120-minute draw into a final winner, we use a logistic model
based on the pre-match skill difference:

```
skilldiff = eta_home - eta_away
p_home_pen = sigmoid( shootout_skilldiff_coef * skilldiff )
```

Then the final win probabilities are:

```
p_home_pens = p_home_120 + p_draw_120 * p_home_pen
p_away_pens = p_away_120 + p_draw_120 * (1 - p_home_pen)
```

This is used for prediction, not for training the main score model.

## Predictive uncertainty

Default predictions use point estimates (`theta_i` means). A Monte Carlo mode
optionally samples team parameters from their posterior covariances and averages
the resulting score matrices.

An optional log-normal correction can be applied to point predictions by
inflating expected scores with `exp(0.5 * Var(eta))`, where `eta` is the
log-rate for the relevant team. This approximates `E[exp(eta)]` without sampling.

## Observations

Observed scores `(x_H, x_A)` are assumed to be generated from the bivariate Poisson above. Model fitting updates the latent parameters to maximize the posterior under these assumptions. This document focuses only on the parameterization and likelihood, not the optimization details.
