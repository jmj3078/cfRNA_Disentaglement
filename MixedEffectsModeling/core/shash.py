import numpy as np
import pandas as pd
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


def load_shash_params(cv_stats_path):
    """Per-gene SHASH params fit on in-fold CV Z (core/calibration.py
    gene_shash_calibration) -- the correction for residual per-gene
    skew/kurtosis a raw RQR Z carries before it can be treated as N(0,1)."""
    return pd.read_csv(cv_stats_path).set_index("gene")[
        ["cv_shash_ok", "cv_shash_xi", "cv_shash_eta", "cv_shash_eps", "cv_shash_delta"]]


def shash_correct_col(z, row):
    if not bool(row["cv_shash_ok"]):
        return z
    return shash_transform_to_z(z, row["cv_shash_xi"], row["cv_shash_eta"], row["cv_shash_eps"], row["cv_shash_delta"])


def shash_correct_matrix(Z, gene_names, params):
    """Z: (n_samples, len(gene_names)) raw RQR. Genes missing from params or
    with cv_shash_ok=False are left uncorrected."""
    Zc = Z.copy()
    for j, g in enumerate(gene_names):
        if g in params.index:
            Zc[:, j] = shash_correct_col(Z[:, j], params.loc[g])
    return Zc
