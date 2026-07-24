import numpy as np

# Mixed-model-aware single-replicate simulator for later PPC use, matching
# marginal_rqr.py's mixture assumption (b ~ N(0,tau2) independent per sample,
# mu_b = mu*exp(b), y ~ NB(mu_b, alpha)). Unlike Modeling's _simulate_once,
# this must draw the random intercept -- ignoring tau2 would understate the
# simulated variance for any gene demoted less than "nb_fixed"/"intercept".
def simulate_mixed(mu, alpha, tau2, family="negbin", seed=42):
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=np.float64)
    tau2 = np.asarray(tau2, dtype=np.float64)
    b = rng.normal(0.0, np.sqrt(np.maximum(tau2, 0.0)))
    mu_b = np.clip(mu * np.exp(b), 1e-8, 1e10)
    if family == "poisson":
        return rng.poisson(mu_b)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = 1.0 / alpha
    p = np.clip(n / (n + mu_b), 1e-10, 1 - 1e-10)
    return rng.negative_binomial(n, p)


# Stacks simulate_mixed over n_reps distinct seeds -> shape (n_samples, n_reps),
# for PPC replicate distributions (obs vs pred mean/std/nonzero-frac).
def simulate_many(mu, alpha, tau2, family="negbin", n_reps=300, seed=42):
    return np.stack([simulate_mixed(mu, alpha, tau2, family, seed=seed + r) for r in range(n_reps)], axis=1)
