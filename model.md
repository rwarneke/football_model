# Soccer model

## Final model

### Joint score model

Team $i$ has attacking strength $a_i$ and defensive strength $d_i$. We model the expected goals of each team as
$$
\begin{aligned}
m_H &= \exp\left( \mu + (a_H + a_\text{HGA}) - d_A \right), \\
m_A &= \exp\left( \mu + a_A - (d_H + d_\text{HGA}) \right),
\end{aligned}
$$
where $a_\text{HGA}$ is the home ground advantage (HGA) boost to attacking, $d_\text{HGA}$ is the HGA boost to defending, and $\mu$ captures the baseline scoring result inherent to the sport (ChatGPT insists that this intercept-style parameter will not just be learnt naturally). Given these parameters, we model the score of the match as a bivariate Poisson $$(X_H, X_A) \sim \text{BivariatePoisson}(m_H, m_A, \rho),$$
where $\rho$ controls the correlation between $X_H$, $X_A$. We model
$$
\begin{aligned}
X_H &= U + V_H, \\
X_A &= U + V_A,
\end{aligned}
$$
where $U \sim \text{Pois}(\nu)$ and $V_i \sim \text{Pois}(\lambda_i)$. The shared mean is defined as $\nu = \rho \min(m_H, m_A)$. The additional component of the mean is defined to match the required means: $\lambda_i = m_i - \nu$. The correlation is maximised when $m_H = m_A$ at $\rho$, and in general is
$$
\text{Corr}(X_H, X_A)
= \rho \cdot \sqrt{\frac{\min(m_H, m_A)}{\max(m_H, m_A)}}.
$$

### Bayesian updates

We maintain Bayesian belief distribution $\theta_i = (a_i, d_i)^\intercal \sim \mathcal{N}\left( \mu_i, \Sigma_i \right)$ for each team, independently.

Before each match, we form the joint prior distribution
$$
\theta
= \begin{bmatrix}
a_H \\ d_H \\ a_A \\ d_A
\end{bmatrix}
\sim
\mathcal{N}
\left(
\mu^{(0)} = \begin{bmatrix}
\mu_H^{(0)} \\
\mu_A^{(0)}
\end{bmatrix},
\begin{bmatrix}
\Sigma_H^{(0)} & 0 \\
0 & \Sigma_A^{(0)}
\end{bmatrix}
\right).
$$
The exact posterior is non-Gaussian:
$$
p(\theta \mid x_H, x_A) \propto p(x_H, x_A \mid \theta)\,p(\theta),
$$
so we approximate it by a Gaussian for computational tractability:
$$
p(\theta \mid x_H, x_A) \approx \mathcal{N}\!\left(\mu^{(1)}, \Sigma^{(1)}\right).
$$

Define the negative log posterior (up to an additive constant)
$$
f(\theta)
=
- \log p(\theta \mid x_H, x_A)
=
- \log p(x_H, x_A \mid \theta) - \log p(\theta).
$$

We compute the MAP
$$
\theta^* = \arg\min_\theta f(\theta),
\qquad
\mu^{(1)} = \theta^{*}.
$$

For the covariance we use a curvature-based Laplace approximation:
$$
\Sigma_{\text{post}} \approx \tilde{H}(\theta^*)^{-1},
$$
where $\tilde{H}(\theta)$ is a positive semidefinite curvature approximation to $\nabla^2 f(\theta)$.
In practice we take $\tilde{H}$ to be Fisher scoring / Gauss--Newton curvature of the likelihood plus the prior precision:
$$
\tilde{H}(\theta) = \tilde{H}_{\text{like}}(\theta) + \Sigma_0^{-1}.
$$

To ensure numerical stability, we compute
$$
\Sigma_{\text{post}} \approx \left(\tilde{H}(\theta^*) + \varepsilon I\right)^{-1}
$$
for a small $\varepsilon > 0$.

Since a match couples home and away parameters, the joint Gaussian approximation generally has cross-team covariance:
$$
\Sigma_{\text{post}}
=
\begin{bmatrix}
\Sigma_{H}^{(1)} & \Sigma_{HA} \\
\Sigma_{AH} & \Sigma_{A}^{(1)}
\end{bmatrix}.
$$
To keep state per team and prevent the global covariance from densifying across teams, we project to independent team marginals by discarding cross-team blocks:
$$
\Sigma^{(1)}
=
\begin{bmatrix}
\Sigma_{H}^{(1)} & 0 \\
0 & \Sigma_{A}^{(1)}
\end{bmatrix}.
$$

We compute $\theta^*$ and $\tilde{H}(\theta^*)$ using damped Fisher scoring / Gauss-Newton with Armijo backtracking line search.
This requires evaluating $f(\theta)$, $\nabla f(\theta)$, and $\tilde{H}(\theta)$.


## Background

I want to build a model for international soccer. My strong preference is to use a Bayesian-style version of the model introduced by Maher (1982) and refined by Dixon and Coles (1997). This is more or less what I put together back in COVID, and is more or less what ChatGPT suggested. The difference this time round, is that I want to be efficient and clean-cut with the modelling and implementation, whereas last time I think I quite aggressively overfit and got a bit lost.

## Data

I'm using the Kaggle dataset for international football, which includes the first ever international match (Scotland 0-0 England on 1872/11/30) up to the most recent, and appears complete. I've filtered this dataset for matches in which both teams are from thoute "team universe", defined as the 211 current FIFA members plus the four teams who have competed in World Cup qualification or finals:
* Czechoslovakia
* East Germany
* Saarland
* Yugoslavia

This filter retains 94.9% of the results.

| Subset | Num. entries | Proportion |
| - | -: | -: |
| Both teams in universe (**retained**) | 46355 | 94.9% |
| One team in universe | 1394 | 2.9% |
| Neither team in universe | 1101 | 2.3% |
| Total | 0 | 100.0% |

The five teams who have played the most matches but are excluded from the universe are shown below.

| Team | Num. games | Year of most recent game |
| - | - | - |
| Martinique | 329 | 2025 |
| Guadeloupe | 271 | 2025 |
| Guernsey | 240 | 2023 |
| Jersey | 235 | 2025 |
| Zanzibar | 209 | 2025 |

If we were to expand the team universe, the most natural way to do so would be to include the CONCACAF teams who are not currently in the universe but have participated in the major championship (Gold Cup):

* Martinique
* Guadeloupe
* French Guiana
* Saint Martin
* Sint Maarten

## Evaluation

I will use 2016-2025 as a strict holdout period. Nothing I do in the modelling stage will include this data whatsoever. At the end of the modelling stage, I will have a list of models that will include, at minimum:
* Standard ELO model with some simple modelling of initial ratings
* Bayes-style ELO model
* Each of the above with simple draw-probability modelling
* Bayes-style attack/defense model

We'll measure model performance using two few metrics:
* Normalised log loss, draws are half a win for each team
  * $-\log(p) / \log(1/2)$ if there is a result, where $p$ is the predicted probability of the winner winning
  * $- (\log(p) + \log(1 - p)) / \log(1/2)$ for draws
* Normalised log loss, draws are a result
  * $- \log(p) / \log(1/3)$, where $p$ is the predicted probability of the result
* Exact scoreline
  * $-\log(p)$, where $p$ is the predicted probability of the exact scoreline

We will (somewhat arbitrarily):
* Assign 0.5x importance to friendlies
* Not score matches in which either team has played fewer than 5 games
