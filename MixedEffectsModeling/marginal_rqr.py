import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import nbinom, norm

RQR_EPS = 1e-8


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
