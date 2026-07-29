import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from MixedEffectsModeling.core.marginal_rqr import nb_marginal_mean_var

_TAU2_EPS = 1e-12


def simulate_marginal_nb(mu, alpha, tau2, n_reps, seed):
    # Posterior-predictive replicates matching marginal_rqr's mixture assumption:
    # b ~ N(0,tau2) per (rep, sample), mu_b = mu*exp(b), y_rep ~ NB(mu_b, alpha).
    # Draws the batch intercept fresh each rep so rep_var/rep_zero carry the same
    # across-sample mu spread the observed data has (a fixed-mu draw would not).
    mu = np.asarray(mu, dtype=np.float64)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), mu.shape)
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=np.float64), mu.shape)
    rng = np.random.default_rng(seed)
    n_obs = len(mu)
    b = np.where(tau2 < _TAU2_EPS, 0.0, rng.normal(0.0, np.sqrt(np.maximum(tau2, 0.0)), size=(n_reps, n_obs)))
    mu_b = np.clip(mu[None, :] * np.exp(b), 1e-8, 1e10)
    n_ = np.broadcast_to(1.0 / np.maximum(alpha, 1e-8), (n_reps, n_obs))
    p_ = np.clip(n_ / (n_ + mu_b), 1e-10, 1 - 1e-10)
    return rng.negative_binomial(n_, p_)


def ppc_moment_pvalues(y, mu, alpha, tau2, n_reps=500, seed=0):
    # Bayesian posterior-predictive p = P(rep_stat >= obs_stat) for mean/var/zero/max.
    # p ~ U(0,1) under a correct model; p near 0/1 flags a moment the model cannot reproduce.
    y = np.asarray(y, dtype=np.float64)
    y_rep = simulate_marginal_nb(mu, alpha, tau2, n_reps, seed)
    return {
        "p_mean": float(np.mean(y_rep.mean(1) >= y.mean())),
        "p_var": float(np.mean(y_rep.var(1) >= y.var())),
        "p_zero": float(np.mean((y_rep == 0).mean(1) >= (y == 0).mean())),
        "p_max": float(np.mean(y_rep.max(1) >= y.max())),
    }


def predictive_moments(mu, alpha, tau2):
    # Proper held-out total predictive mean/variance for a gene:
    # total var = mean(Var[Y|x]) + Var(E[Y|x]), the second term being the covariate-driven
    # across-sample mean spread the per-sample marginal variance omits.
    m_i, v_i = nb_marginal_mean_var(np.asarray(mu, dtype=np.float64),
                                    np.asarray(alpha, dtype=np.float64),
                                    np.asarray(tau2, dtype=np.float64))
    return float(m_i.mean()), float(v_i.mean() + m_i.var())
