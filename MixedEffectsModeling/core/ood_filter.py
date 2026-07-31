import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import mahalanobis


class MahalanobisFilter:
    """Flags samples whose covariate (X) profile falls outside the HC
    distribution the model was trained on -- a sample far from HC in Mahalanobis
    distance can show a large |Z| purely from covariate extrapolation, not
    disease biology. Threshold is an HC-empirical percentile (ported from
    _legacy/GLM_Modeling_Cascade_v1/Modeling/sample_filter.py, plotting dropped)."""

    def __init__(self, percentile=95, reg=1e-8):
        self.percentile = percentile
        self.reg = reg
        self._fitted = False

    def fit(self, X_hc):
        X_hc = np.asarray(X_hc, dtype=float)
        p = X_hc.shape[1]
        self.mu_ = X_hc.mean(axis=0)
        self.cov_inv_ = inv(np.cov(X_hc.T) + np.eye(p) * self.reg)
        hc_dist = self._distances(X_hc)
        self.threshold_ = float(np.percentile(hc_dist, self.percentile))
        self._fitted = True
        return self

    def _distances(self, X):
        return np.array([mahalanobis(x, self.mu_, self.cov_inv_) for x in X])

    def distances(self, X):
        if not self._fitted:
            raise RuntimeError("Call fit(X_hc) first.")
        return self._distances(np.asarray(X, dtype=float))

    def mask(self, X):
        """Boolean keep-mask: True = inlier (within threshold)."""
        return self.distances(X) <= self.threshold_
