import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

IDENTITY = (0.0, 1.0, 0.0, 1.0)  # xi, eta, eps, delta -- no-op transform


def shash_logpdf(x, xi, eta, eps, delta):
    z = (x - xi) / eta
    S = np.sinh(delta * np.arcsinh(z) - eps)
    C = np.sqrt(1 + S**2)
    return (np.log(delta) + np.log(C) - np.log(eta) - 0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(1 + z**2) - 0.5 * S**2)


def shash_quantile(p, xi, eta, eps, delta):
    zq = norm.ppf(p)
    return xi + eta * np.sinh((np.arcsinh(zq) + eps) / delta)


def shash_transform_to_z(x, xi, eta, eps, delta):
    z_raw = (x - xi) / eta
    return np.sinh(delta * np.arcsinh(z_raw) - eps)


def fit_shash(x):
    x = np.asarray(x, dtype=np.float64)

    def negll(params):
        xi, log_eta, eps, log_delta = params
        eta, delta = np.exp(log_eta), np.exp(log_delta)
        ll = shash_logpdf(x, xi, eta, eps, delta)
        return 1e10 if not np.all(np.isfinite(ll)) else -ll.sum()

    x0 = [np.median(x), np.log(max(x.std(), 1e-6)), 0.0, 0.0]
    res = minimize(negll, x0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-7})
    xi, log_eta, eps, log_delta = res.x
    eta, delta = np.exp(log_eta), np.exp(log_delta)
    ok = bool(res.success) and np.isfinite([xi, eta, eps, delta]).all()
    return (xi, eta, eps, delta, ok) if ok else (*IDENTITY, False)
