import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import nbinom, norm, poisson

RQR_EPS = 1e-8


# Poisson-family RQR fallback for route "pool" when fit_pooled_glmm's own
# deviance/df check selects family=="poisson" over negbin. Copied from
# Modeling/model_engine.py's _poisson_rqr, per the isolation requirement (see
# _nb_rqr's comment below).
def _poisson_rqr(y, mu, seed=None):
    y = np.asarray(y)
    lo = np.where(y > 0, poisson.cdf(y - 1, mu), 0.0)
    hi = poisson.cdf(y, mu)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS); hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)


def _nb_cdf(y, mu, alpha):
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    return nbinom.cdf(y, n, p)


# Copied from Modeling/model_engine.py's _nb_rqr (point-mass NB RQR fallback
# for tau2~=0) -- MixedEffectsModeling stays fully independent of Modeling/,
# so this is a duplicate, not an import, per the isolation requirement.
def _nb_rqr(y, mu, alpha, seed=None):
    y = np.asarray(y)
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    lo = np.where(y > 0, nbinom.cdf(y - 1, n, p), 0.0)
    hi = nbinom.cdf(y, n, p)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS); hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)


def _nb_logpmf(y, mu, alpha):
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    return nbinom.logpmf(y, n, p)


# Held-out marginal log-likelihood (mixture over the batch random intercept,
# integrating the PMF via Gauss-Hermite -- NOT the CDF-mixture used for RQR).
# W1/RQR calibration is randomization-smoothed and loses almost all power to
# detect misspecification when y is mostly 0/1 (verified empirically: shuffling
# y relative to mu barely moved w1 on real pool-route genes). Log-likelihood
# doesn't have that blind spot -- a wrong mu strictly costs probability mass.
def marginal_nb_loglik(y, mu, alpha, tau2, n_nodes=7):
    # tau2 may be scalar or per-sample array (e.g. concatenated across CV folds
    # with different fold-level tau2 estimates) -- sd broadcasts elementwise.
    y = np.asarray(y)
    tau2 = np.asarray(tau2, dtype=np.float64)
    if np.all(tau2 < 1e-6):
        return _nb_logpmf(y, mu, alpha)
    nodes, weights = hermegauss(n_nodes)
    weights = weights / weights.sum()
    sd = np.sqrt(np.maximum(tau2, 0.0))
    logpmf_k = np.stack([
        _nb_logpmf(y, mu * np.exp(sd * node), alpha) + np.log(w)
        for node, w in zip(nodes, weights)
    ])
    m = logpmf_k.max(axis=0)
    return m + np.log(np.exp(logpmf_k - m).sum(axis=0))


def marginal_nb_rqr(y, mu, alpha, tau2, seed, n_nodes=7):
    y = np.asarray(y)
    if tau2 < 1e-6:
        return _nb_rqr(y, mu, alpha, seed)

    nodes, weights = hermegauss(n_nodes)  # integrate against exp(-x^2/2), matches N(0,1)
    weights = weights / weights.sum()
    sd = np.sqrt(tau2)
    lo = np.zeros_like(y, dtype=np.float64)
    hi = np.zeros_like(y, dtype=np.float64)
    for node, w in zip(nodes, weights):
        mu_b = mu * np.exp(sd * node)
        lo += w * np.where(y > 0, _nb_cdf(y - 1, mu_b, alpha), 0.0)
        hi += w * _nb_cdf(y, mu_b, alpha)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS)
    hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)
