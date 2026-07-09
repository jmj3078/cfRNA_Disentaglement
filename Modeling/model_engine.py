"""NZ-gated normative model engine (single standard pipeline)."""

import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as rpyn
import scanpy as sc
import statsmodels.api as sm
from rpy2.robjects.conversion import localconverter
from scipy.sparse import issparse
from scipy.stats import nbinom, norm, poisson
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config
from dispersion_trend import build_trend, load_trend, save_trend

MP = config.MODELING_PARAMS

# Shared RQR probability-clip epsilon. Applied identically across all stage RQRs so the
# z-score ceiling is consistent: norm.ppf(1 - RQR_EPS) ~= 5.61 for every stage (previously
RQR_EPS = 1e-8


# ---- Pure-Python RQR helpers (mirrors model_engine.py) ------------------

def _poisson_rqr(y, mu, seed=None):
    y = np.asarray(y)
    lo = np.where(y > 0, poisson.cdf(y - 1, mu), 0.0)
    hi = poisson.cdf(y, mu)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS); hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)


def _nb_rqr(y, mu, alpha, seed=None):
    y = np.asarray(y)
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    lo = np.where(y > 0, nbinom.cdf(y - 1, n, p), 0.0)
    hi = nbinom.cdf(y, n, p)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS); hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)


def _w1_normal(z):
    """Mean abs deviation between sorted z and theoretical N(0,1) quantiles.
    Guideline (cv_gamlss_nb.py): > 0.25 indicates poor calibration."""
    v = z[np.isfinite(z)]
    n = len(v)
    if n < 8:
        return np.nan
    ref = norm.ppf(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
    return float(np.mean(np.abs(np.sort(v) - ref)))


def _nbi_rqr_from_coeffs(mu_coef, sigma_coef, X_test, y_test, seed=None):
    n = len(y_test)
    Xa = np.column_stack([np.ones(n), X_test])
    mu = np.exp(Xa @ mu_coef).clip(1e-4, 1e6)
    sigma = np.exp(Xa @ sigma_coef).clip(1e-8, 1e3)
    theta = (1.0 / sigma).clip(1e-4, 1e4)
    p_nb = np.clip(theta / (theta + mu), RQR_EPS, 1 - RQR_EPS)
    yi = np.asarray(y_test, dtype=int)
    a = np.where(yi > 0, nbinom.cdf(yi - 1, n=theta, p=p_nb), 0.0)
    b = nbinom.cdf(yi, n=theta, p=p_nb)
    lo = np.clip(np.minimum(a, b), RQR_EPS, 1 - RQR_EPS)
    hi = np.clip(np.maximum(a, b), RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(lo, hi)).astype(np.float32)


# ---- rpy2 helpers ---------------------------------------------------------

def _to_r_matrix(arr, col_names):
    with localconverter(ro.default_converter + rpyn.converter):
        r_mat = ro.conversion.py2rpy(np.ascontiguousarray(arr, dtype=np.float64))
    return ro.r["matrix"](r_mat, nrow=arr.shape[0], ncol=arr.shape[1],
                          dimnames=ro.r["list"](ro.NULL, ro.StrVector(col_names)))


def _to_r_vec(arr):
    with localconverter(ro.default_converter + rpyn.converter):
        return ro.conversion.py2rpy(np.ascontiguousarray(arr, dtype=np.float64))


# ---- stage "nb_fixed": pure-Python unpenalized mean-only NB (fixed dispersion) ----

def _nb_irls(y, X, alpha, max_iter=100, tol=1e-8):
    """Unpenalized NB2(alpha fixed) mean-model IRLS. Returns (beta, converged)."""
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-3))
    for _ in range(max_iter):
        eta = X @ beta
        if np.max(np.abs(eta)) > 30:
            # exp(30) ~ 1e13 is already unphysical for a mean-count model; the clip
            # below would silently mask this as a converged fit instead of a
            # diverging one, so treat it as IRLS divergence explicitly.
            return beta, False
        mu = np.clip(np.exp(eta), 1e-6, 1e8)
        w = mu / (1.0 + alpha * mu)
        z = eta + (y - mu) / np.clip(mu, 1e-6, None)
        WX = X * w[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            return beta, False
        if not np.all(np.isfinite(beta_new)):
            return beta, False
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            return beta, True
        beta = beta_new
    # Loop exhausted without meeting tol: report honest non-convergence rather
    # than claiming success, mirroring the stage-nbi convergence gate. The
    # caller (fit_route_b_gene) demotes this to the closed-form intercept stage.
    return beta, False


def _nb_deviance(y, mu, alpha):
    mu = np.clip(mu, 1e-8, None)
    y_safe = np.where(y > 0, y, 1e-8)
    term = y * np.log(y_safe / mu) - (y + 1.0 / alpha) * np.log((1 + alpha * y) / (1 + alpha * mu))
    return float(2 * term.sum())


def fit_intercept_only_gene(y_train, alpha_fn):
    """Closed-form intercept-only NB: mu = mean(y_train), dispersion fixed from
    the covariate-free trend. No optimization, no covariates -- succeeds for any
    y with a finite, positive mean. Used both (a) as the intercept side of Route
    B's full-vs-intercept GAIC comparison, and (b) as the final fallback when
    Route B's full IRLS itself fails to converge -- one implementation, no
    duplicated closed-form math. Returns dict(success, beta, alpha, fail_reason)."""
    mean_y = float(y_train.mean()) if np.all(np.isfinite(y_train)) else np.nan
    if not np.isfinite(mean_y) or mean_y <= 0:
        return dict(success=False, fail_reason="intercept_only_undefined_mean", n_removed=0)
    alpha_g = alpha_fn(mean_y)
    if not np.isfinite(alpha_g) or alpha_g <= 0:
        return dict(success=False, fail_reason="intercept_only_invalid_alpha", n_removed=0)
    beta = np.array([np.log(mean_y)])
    return dict(success=True, beta=beta, alpha=alpha_g, n_removed=0, fail_reason="")


def _select_outliers(z, outlier_z, max_remove_frac, n_total, n_removed_so_far):
    """Indices of the worst |z| points to drop this iteration, capped at the
    remaining removal budget over ALL iterations combined (mirrors
    gamlss.r's .select_outliers). Taking the worst points up to the cap --
    instead of refusing to remove anything once the raw outlier count exceeds
    max_remove_frac -- means the loop always makes progress toward the budget."""
    outlier = np.isfinite(z) & (np.abs(z) > outlier_z)
    budget = int(max_remove_frac * n_total) - n_removed_so_far
    if not outlier.any() or budget <= 0:
        return np.array([], dtype=int)
    idx = np.where(outlier)[0]
    if len(idx) > budget:
        idx = idx[np.argsort(-np.abs(z[idx]))][:budget]
    return idx


def fit_route_b_gene(y_train, X_train, alpha_fn, outlier_z, max_iter, max_remove_frac,
                     beta_explode_thr=None, gaic_k=None):
    """Unpenalized mean-only NB (fixed dispersion from the trend), GAIC full-vs-intercept."""
    beta_explode_thr = MP["beta_explode_thr"] if beta_explode_thr is None else beta_explode_thr
    gaic_k = MP["gaic_k"] if gaic_k is None else gaic_k
    n = len(y_train)
    Xa = np.column_stack([np.ones(n), X_train])
    keep = np.ones(n, dtype=bool)
    n_removed = 0
    beta = None
    alpha_g = alpha_fn(float(y_train.mean()))

    for _ in range(max_iter):
        y_k, X_k = y_train[keep], Xa[keep]
        alpha_g = alpha_fn(float(y_k.mean()))
        beta, ok = _nb_irls(y_k, X_k, alpha_g)
        if not ok or not np.all(np.isfinite(beta)):
            return dict(success=False, fail_reason="irls_not_converged", n_removed=n_removed)
        mu_k = np.clip(np.exp(X_k @ beta), 1e-6, 1e8)
        z_k = _nb_rqr(y_k, mu_k, alpha_g, seed=0)
        drop_idx = _select_outliers(z_k, outlier_z, max_remove_frac, n, n_removed)
        if len(drop_idx) == 0:
            break
        idx_keep = np.where(keep)[0]
        keep[idx_keep[drop_idx]] = False
        n_removed += len(drop_idx)

    if beta is None or not np.all(np.isfinite(beta)):
        return dict(success=False, fail_reason="fit_failed", n_removed=n_removed)

    y_k, X_k = y_train[keep], Xa[keep]
    mu_full = np.clip(np.exp(X_k @ beta), 1e-6, 1e8)
    dev_full = _nb_deviance(y_k, mu_full, alpha_g)
    edf_full = X_k.shape[1]
    gaic_full = dev_full + gaic_k * edf_full

    null_res = fit_intercept_only_gene(y_k, alpha_fn)
    if not null_res["success"]:
        # The full IRLS already succeeded, so a usable fit exists regardless;
        # the closed-form intercept model failing here would only happen for a
        # pathological y_k (should not occur once the full fit converged), so
        # just skip the comparison and keep the full fit.
        return dict(success=True, beta=beta, beta_null=beta, alpha=alpha_g,
                   gaic_full=gaic_full, gaic_null=np.inf, chosen="full",
                   beta_max=float(np.abs(beta[1:]).max()), n_removed=n_removed, fail_reason="")

    beta_null = np.concatenate([null_res["beta"], np.zeros(Xa.shape[1] - 1)])
    mu_null = np.full(len(y_k), np.exp(null_res["beta"][0])).clip(1e-6, 1e8)
    dev_null = _nb_deviance(y_k, mu_null, null_res["alpha"])
    gaic_null = dev_null + gaic_k * 1

    beta_max = float(np.abs(beta[1:]).max())
    if not np.isfinite(gaic_full) or beta_max > beta_explode_thr:
        chosen = "intercept"
    else:
        chosen = "full" if gaic_full < gaic_null else "intercept"

    chosen_alpha = null_res["alpha"] if chosen == "intercept" else alpha_g
    return dict(success=True, beta=beta, beta_null=beta_null, alpha=chosen_alpha,
               gaic_full=gaic_full, gaic_null=gaic_null, chosen=chosen,
               beta_max=beta_max, n_removed=n_removed, fail_reason="")


# ---- Gene record ----------------------------------------------------------

@dataclass
class GeneRecord:
    name: str
    initial_route: str      # "pool" | "model"  (Phase 1 gating)
    route: str = ""          # final route actually used: "pool" | "model" | "excluded"
    stage: str = ""          # which model stage produced the fit: "nbi" | "nb_fixed" | "intercept"
    nz: int = 0
    attempted: bool = False  # True once train() has processed this gene (vs. skipped by --limit)

    # stage == "nbi" (full NBI GAMLSS, mu and sigma on covariates)
    mu_coef: np.ndarray = None
    sigma_coef: np.ndarray = None
    nbi_explode: str = ""    # "" | "mu" | "sigma" | "mu+sigma"  (which submodel triggered demotion)

    # stage in {"nb_fixed", "intercept"} (mean-only NB, dispersion fixed from the trend)
    beta: np.ndarray = None
    alpha: float = None
    mean_model_chosen: str = ""  # "full" | "intercept"  (GAIC choice within stage == "nb_fixed")

    # route == "pool" (rare pooling)
    mean_hc: float = None

    fit_ok: bool = False
    n_removed: int = 0
    fail_reason: str = ""
    w1_train: float = None  # in-sample W1 calibration of the accepted fit (all stages except pool)

    @property
    def branch(self):
        """Downstream-taxonomy view: pool route -> 'rare', everything else -> 'count'.
        Kept so pipeline/scoring._scores_long can label genes without knowing stages."""
        return "rare" if self.route == "pool" else "count"


GeneRecordV2 = GeneRecord  # back-compat alias for engines pickled under the old name


# ---- Engine ---------------------------------------------------------------

class NormativeModelEngine:
    def __init__(self, nz_a_max=None, trend_min_nz=None,
                ridge_lambda_sigma=None, outlier_z=None, max_outlier_iter=None,
                max_remove_frac=None, beta_explode_thr=None, gaic_k=None,
                rare_overdisp_thr=None):
        self.nz_a_max = nz_a_max or MP["nz_a_max"]
        self.trend_min_nz = trend_min_nz or MP["trend_min_nz"]
        self.ridge_lambda_sigma = ridge_lambda_sigma or MP["ridge_lambda_sigma"]
        self.gaic_k = gaic_k or MP["gaic_k"]
        self.outlier_z = outlier_z or MP["outlier_z"]
        self.max_outlier_iter = max_outlier_iter or MP["max_outlier_iter"]
        self.max_remove_frac = max_remove_frac or MP["max_remove_frac"]
        self.beta_explode_thr = beta_explode_thr or MP["beta_explode_thr"]
        self.rare_overdisp_thr = rare_overdisp_thr or MP["rare_overdisp_thr"]

        self.X_hc_scaled = None
        self.Y_hc = None
        self.scaler = None
        self.is_hc = None
        self.pc_gene_names = []
        self.pc_indices = None
        self._gene_col = {}

        self.genes = {}
        self.alpha_fn = None
        self.rare_glm = None

        self._r_nbi_fn = None

    # ---- Data loading -----------------------------------------------------

    def load_hc_data(self, h5ad_path=config.H5AD_PATH):
        print("Loading HC data...")
        adata = sc.read_h5ad(h5ad_path)
        adata = adata[adata.obs["QC_Passed"] == True]
        adata = adata[adata.obs["Phenotype_Processed"].notna()]
        adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
        adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]
        self.is_hc = (adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values

        X_raw = adata.obs[config.BIAS_COLUMNS].values.astype(np.float64)
        self.scaler = StandardScaler()
        self.X_hc_scaled = self.scaler.fit_transform(X_raw[self.is_hc])

        Y_raw = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        self.Y_hc = np.round(Y_raw[self.is_hc]).astype(np.float64)

        is_pc = (adata.var["GeneType"] == "protein_coding").values
        self.pc_gene_names = adata.var_names[is_pc].tolist()
        self.pc_indices = np.where(is_pc)[0]
        self._gene_col = {g: self.pc_indices[i] for i, g in enumerate(self.pc_gene_names)}
        print(f"  HC={self.is_hc.sum()}  protein-coding={len(self.pc_gene_names)}")

    # ---- Phase 0: dispersion trend -----------------------------------------

    def build_dispersion_trend(self):
        Y_pc = self.Y_hc[:, self.pc_indices]
        trend = build_trend(Y_pc, min_nz=self.trend_min_nz)
        save_trend(trend)
        self.alpha_fn = load_trend()
        print(f"Dispersion trend built: n_reliable={trend['n_reliable']} "
              f"n_bins={trend['n_bins_used']}")

    # ---- Phase 1: gating ----------------------------------------------------

    def assign_routes(self):
        """Only NZ-based decision in the whole pipeline: nz < nz_a_max goes to
        rare pooling directly; everything else attempts the nbi stage first and
        lets the nbi -> nb_fixed -> intercept demotion chain decide the rest from
        actual fit outcomes, not a fixed NZ cutoff."""
        assert self.Y_hc is not None, "Call load_hc_data() first."
        Y_pc = self.Y_hc[:, self.pc_indices]
        nz = (Y_pc > 0).sum(axis=0)
        self.genes = {}
        for i, g in enumerate(self.pc_gene_names):
            n = int(nz[i])
            route = "pool" if n < self.nz_a_max else "model"
            self.genes[g] = GeneRecord(name=g, initial_route=route, nz=n)
        counts = pd.Series([r.initial_route for r in self.genes.values()]).value_counts()
        print(f"Phase 1 gating: pool={counts.get('pool',0)}  "
              f"model-candidates(nbi-first)={counts.get('model',0)}  (nz_a_max={self.nz_a_max})")
        return counts

    # ---- R init -------------------------------------------------------------

    def _init_r(self):
        if self._r_nbi_fn is None:
            ro.r(f'source("{config.R_HELPER}")')
            self._r_nbi_fn = ro.globalenv["train_nbi_coeffs"]

    def _gene_y(self, g):
        return self.Y_hc[:, self._gene_col[g]]

    def _record_calibration(self, rec, z_train):
        """In-sample W1, recorded for diagnostics/flagging only -- does not gate
        route/stage acceptance. A hard demotion/exclusion threshold on z-score
        normality turned out to be theoretically awkward here (RQR's own
        optimism bias at low NZ, n_valid varying gene-to-gene under CV, and
        statistical significance vs. practical effect size all pulling in
        different directions), so this is left as a training_summary.csv column
        for downstream review rather than an accept/reject gate."""
        rec.w1_train = _w1_normal(z_train)

    # ---- stage "nbi": full NBI GAMLSS (mu AND sigma on covariates) -----------

    def _fit_nbi(self, rec):
        """Try full NBI only -- no intercept-only competitor is fit here. Any
        failure demotes straight to stage "nb_fixed", which does its own
        full-vs-intercept GAIC comparison using the shared
        fit_intercept_only_gene closed form. Failure covers: rpy2-level
        exception, gamlss hard error (success=FALSE), gamlss non-convergence
        (converged=FALSE -- gamlss reports this as a warning, not an error, so
        it must be checked explicitly), non-finite mu/sigma coefficients, and
        coefficient explosion in mu or sigma. In-sample W1 is recorded
        (w1_train) but does not itself gate acceptance -- see
        _record_calibration."""
        self._init_r()
        y = self._gene_y(rec.name)
        try:
            res_full = self._r_nbi_fn(
                _to_r_vec(y), _to_r_matrix(self.X_hc_scaled, config.BIAS_COLUMNS),
                ro.IntVector([50]), ro.FloatVector([self.outlier_z]),
                ro.IntVector([self.max_outlier_iter]), ro.FloatVector([self.max_remove_frac]),
                ro.FloatVector([self.ridge_lambda_sigma]),
            )
        except Exception as exc:
            rec.fail_reason = f"nbi_full_error:{exc}"
            return False

        if not bool(res_full.rx2("success")[0]):
            rec.fail_reason = str(res_full.rx2("msg")[0]) or "nbi_full_not_converged"
            return False

        if not bool(res_full.rx2("converged")[0]):
            rec.fail_reason = "nbi_not_converged"
            return False
        if not (bool(res_full.rx2("mu_finite")[0]) and bool(res_full.rx2("sigma_finite")[0])):
            rec.fail_reason = "nbi_nonfinite_coef"
            return False

        beta_full = np.array(res_full.rx2("mu_coef"))
        sigma_full = np.array(res_full.rx2("sigma_coef"))
        mu_explode = float(np.abs(beta_full[1:]).max()) > self.beta_explode_thr
        sigma_explode = float(np.abs(sigma_full[1:]).max()) > self.beta_explode_thr
        if mu_explode or sigma_explode:
            rec.nbi_explode = "+".join(t for t, e in [("mu", mu_explode), ("sigma", sigma_explode)] if e)
            rec.fail_reason = f"beta_explode:{rec.nbi_explode}"
            return False

        z_train = _nbi_rqr_from_coeffs(beta_full, sigma_full, self.X_hc_scaled, y, seed=0)
        self._record_calibration(rec, z_train)

        rec.mu_coef = beta_full
        rec.sigma_coef = sigma_full
        rec.n_removed = int(res_full.rx2("n_removed")[0])
        rec.route = "model"
        rec.stage = "nbi"
        rec.fit_ok = True
        return True

    # ---- stage "nb_fixed": unpenalized mean-only NB ---------------------------

    def _fit_nb_fixed(self, rec):
        """False (IRLS divergence) demotes to the final intercept-only stage."""
        y = self._gene_y(rec.name)
        res = fit_route_b_gene(y, self.X_hc_scaled, self.alpha_fn,
                               self.outlier_z, self.max_outlier_iter, self.max_remove_frac,
                               beta_explode_thr=self.beta_explode_thr, gaic_k=self.gaic_k)
        if not res["success"]:
            rec.fail_reason = res["fail_reason"]
            return False

        Xa = np.column_stack([np.ones(len(y)), self.X_hc_scaled])
        if res["chosen"] == "full":
            mu = np.clip(np.exp(Xa @ res["beta"]), 1e-6, 1e8)
        else:
            mu = np.full(len(y), np.exp(res["beta_null"][0])).clip(1e-6, 1e8)
        z_train = _nb_rqr(y, mu, res["alpha"], seed=0)
        self._record_calibration(rec, z_train)

        rec.beta = res["beta"] if res["chosen"] == "full" else res["beta_null"]
        rec.alpha = res["alpha"]
        rec.mean_model_chosen = res["chosen"]
        rec.n_removed = res["n_removed"]
        rec.route = "model"
        rec.stage = "nb_fixed"
        rec.fit_ok = True
        return True

    # ---- stage "intercept": closed-form intercept-only NB --------------------

    def _fit_intercept(self, rec):
        """Last step of the demotion chain, reached only when stage "nb_fixed"'s
        full IRLS itself diverges. fit_intercept_only_gene is a closed-form
        computation (mu=mean(y), dispersion from the trend) that succeeds for
        any y with a finite positive mean -- i.e. essentially always. The rare
        pathological failure (non-finite y or invalid trend lookup) is excluded
        entirely rather than silently defaulted elsewhere, per policy."""
        y = self._gene_y(rec.name)
        res = fit_intercept_only_gene(y, self.alpha_fn)
        if not res["success"]:
            rec.fail_reason = res["fail_reason"]
            return False

        mu = np.full(len(y), np.exp(res["beta"][0])).clip(1e-6, 1e8)
        z_train = _nb_rqr(y, mu, res["alpha"], seed=0)
        self._record_calibration(rec, z_train)

        rec.beta = res["beta"]
        rec.alpha = res["alpha"]
        rec.mean_model_chosen = "intercept"
        rec.n_removed = res["n_removed"]
        rec.route = "model"
        rec.stage = "intercept"
        rec.fit_ok = True
        return True

    # ---- route "pool" (rare pooling, pooled GLM, always succeeds) ------------

    def train_rare(self, gene_list):
        if not gene_list:
            return
        n_hc = self.X_hc_scaled.shape[0]
        eps = 1.0 / (2 * n_hc)
        cols = [self._gene_col[g] for g in gene_list]
        Y_rare = self.Y_hc[:, cols]
        mean_hc = Y_rare.mean(axis=0)
        for g, m in zip(gene_list, mean_hc):
            self.genes[g].mean_hc = float(m)
            self.genes[g].route = "pool"
            self.genes[g].fit_ok = True
            self.genes[g].attempted = True

        n_rare = len(gene_list)
        sample_idx = np.repeat(np.arange(n_hc), n_rare)
        gene_idx = np.tile(np.arange(n_rare), n_hc)
        Xc = np.column_stack([np.ones(n_hc * n_rare), self.X_hc_scaled[sample_idx]])
        y = Y_rare[sample_idx, gene_idx]
        offset = np.log(mean_hc[gene_idx] + eps)
        pois = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=offset).fit()
        ratio = float(pois.deviance / pois.df_resid)
        if ratio <= self.rare_overdisp_thr:
            family, beta, alpha = "poisson", np.asarray(pois.params), None
        else:
            nb = NegativeBinomial(y, Xc, offset=offset).fit(disp=False)
            family, beta, alpha = "negbin", np.asarray(nb.params[:-1]), float(nb.params[-1])
        # Covariate-multiplier clip bounds, from the range actually seen in HC training
        # (per-sample exp(covariate @ slopes), intercept excluded). Scoring clips new
        # samples' multiplier to this range so a few-covariate-extreme (OOD-adjacent)
        # sample cannot extrapolate mu to an implausible value and manufacture a
        # false extreme z. [0.1, 99.9] pct, not min/max, to ignore lone HC outliers.
        hc_mult = np.exp(self.X_hc_scaled @ beta[1:])
        mult_lo = float(np.percentile(hc_mult, 0.1))
        mult_hi = float(np.percentile(hc_mult, 99.9))
        self.rare_glm = {"family": family, "beta": beta, "alpha": alpha,
                         "eps": eps, "overdisp_ratio": ratio,
                         "mult_lo": mult_lo, "mult_hi": mult_hi}
        print(f"Route pool (rare pooling): {n_rare} genes pooled, family={family}, "
              f"deviance/df={ratio:.3f}, covariate-mult clip=[{mult_lo:.3f}, {mult_hi:.3f}]")

    # ---- Bulk training with demotion chain -----------------------------------

    def train(self, verbose=True, limit=None):
        """nbi -> nb_fixed -> intercept, one gene at a time, each step attempted
        only after the previous one actually failed. Pool candidates (NZ <
        nz_a_max) never enter this chain -- they go straight to pooled rare
        fitting."""
        assert self.genes, "Call assign_routes() first."
        if self.alpha_fn is None:
            self.build_dispersion_trend()

        all_genes = list(self.genes.keys())[:limit]
        model_candidates = [g for g in all_genes if self.genes[g].initial_route == "model"]
        pool_candidates = [g for g in all_genes if self.genes[g].initial_route == "pool"]
        print(f"Training: model-candidates(nbi-first)={len(model_candidates)}  "
              f"pool-candidates(rare)={len(pool_candidates)}")

        demoted_to_nb_fixed = []
        for i, g in enumerate(model_candidates):
            rec = self.genes[g]
            rec.attempted = True
            try:
                ok = self._fit_nbi(rec)
            except Exception as exc:
                ok = False
                rec.fail_reason = str(exc)
            if not ok:
                demoted_to_nb_fixed.append(g)
            if verbose and (i + 1) % 500 == 0:
                print(f"  [stage nbi {i+1:5d}/{len(model_candidates)}] "
                      f"demoted_so_far={len(demoted_to_nb_fixed)}")
        print(f"Step 1 (stage nbi): {len(model_candidates)-len(demoted_to_nb_fixed)} fitted, "
              f"{len(demoted_to_nb_fixed)} demoted to stage nb_fixed")

        demoted_to_intercept = []
        for i, g in enumerate(demoted_to_nb_fixed):
            rec = self.genes[g]
            try:
                ok = self._fit_nb_fixed(rec)
            except Exception as exc:
                ok = False
                rec.fail_reason = str(exc)
            if not ok:
                demoted_to_intercept.append(g)
            if verbose and (i + 1) % 500 == 0:
                print(f"  [stage nb_fixed {i+1:5d}/{len(demoted_to_nb_fixed)}] "
                      f"demoted_so_far={len(demoted_to_intercept)}")
        print(f"Step 2 (stage nb_fixed): {len(demoted_to_nb_fixed)-len(demoted_to_intercept)} fitted, "
              f"{len(demoted_to_intercept)} demoted to stage intercept")

        excluded = []
        for i, g in enumerate(demoted_to_intercept):
            rec = self.genes[g]
            try:
                ok = self._fit_intercept(rec)
            except Exception as exc:
                ok = False
                rec.fail_reason = str(exc)
            if not ok:
                excluded.append(g)
                rec.route = "excluded"
        print(f"Step 3 (stage intercept): "
              f"{len(demoted_to_intercept)-len(excluded)} fitted, {len(excluded)} EXCLUDED")

        self.train_rare(pool_candidates)

        n_fitted = sum(1 for r in self.genes.values() if r.fit_ok)
        n_excluded = sum(1 for r in self.genes.values() if r.route == "excluded")
        print(f"Training complete: {n_fitted} fitted, {n_excluded} excluded, "
              f"total={len(self.genes)}")

    # ---- Scoring --------------------------------------------------------------

    def _rare_z(self, rec, X_test, y_col, seed):
        g = self.rare_glm
        X_test = np.asarray(X_test)
        # split intercept from covariate slopes so the covariate multiplier can be clipped
        # to the HC-observed range (guards against OOD-adjacent samples extrapolating mu).
        mult = np.exp(X_test @ g["beta"][1:])
        if "mult_lo" in g:
            mult = np.clip(mult, g["mult_lo"], g["mult_hi"])
        mu = (rec.mean_hc + g["eps"]) * np.exp(g["beta"][0]) * mult
        mu = np.clip(mu, 1e-12, 1e8)
        if g["family"] == "poisson":
            z = _poisson_rqr(y_col, mu, seed)
        else:
            z = _nb_rqr(y_col, mu, g["alpha"], seed)
        return z.astype(np.float32)

    def score(self, X_test_raw, Y_test, gene_names=None, seed=42, as_dict=False):
        """Score new samples. Returns the raw Z matrix by default (used by CV).

        With as_dict=True, returns the downstream-compatible dict consumed by
        pipeline/scoring.py:
          "combined"          : Z with pool-route (rare) columns zeroed -- the canonical
                                engine-only placeholder contract for Z_disease.npy
          "combined_all"      : full Z including rare columns (source for the flagged parquet)
          "gene_names"        : column order (== gene_names)
          "rare"              : (n_test, n_rare) submatrix of pool-route genes' Z
          "rare_gene_names"   : list of pool-route gene ids
        """
        gene_names = gene_names or [g for g in self.genes if self.genes[g].fit_ok]
        X_test = self.scaler.transform(X_test_raw.astype(np.float64))
        n_test, n_gene = len(X_test), len(gene_names)
        if Y_test.shape[1] != n_gene:
            raise ValueError(f"Y_test has {Y_test.shape[1]} columns, expected {n_gene}")

        Z = np.full((n_test, n_gene), np.nan, dtype=np.float32)
        Xa = np.column_stack([np.ones(n_test), X_test])
        for j, g in enumerate(gene_names):
            rec = self.genes.get(g)
            if rec is None or not rec.fit_ok:
                continue
            y_col = Y_test[:, j].astype(np.float64)
            try:
                if rec.route == "pool":
                    Z[:, j] = self._rare_z(rec, X_test, y_col, seed + j)
                elif rec.stage == "nbi":
                    Z[:, j] = _nbi_rqr_from_coeffs(rec.mu_coef, rec.sigma_coef, X_test, y_col, seed + j)
                elif rec.stage in ("nb_fixed", "intercept"):
                    if rec.mean_model_chosen == "full":
                        mu = np.clip(np.exp(Xa @ rec.beta), 1e-6, 1e8)
                    else:
                        mu = np.full(n_test, np.exp(rec.beta[0])).clip(1e-6, 1e8)
                    Z[:, j] = _nb_rqr(y_col, mu, rec.alpha, seed + j)
            except Exception:
                pass
        if not as_dict:
            return Z

        rare_idx = [j for j, g in enumerate(gene_names)
                    if self.genes.get(g) and self.genes[g].route == "pool"]
        combined = Z.copy()
        if rare_idx:
            combined[:, rare_idx] = 0.0
        return {
            "combined": combined,
            "combined_all": Z,
            "gene_names": list(gene_names),
            "rare": Z[:, rare_idx] if rare_idx else np.zeros((n_test, 0), np.float32),
            "rare_gene_names": [gene_names[j] for j in rare_idx],
        }

    # ---- Diagnostics & persistence --------------------------------------------

    def training_summary(self, attempted_only=True):
        """attempted_only=True (default) reports only genes train() actually processed
        -- relevant when train(limit=N) was used for a smoke test, since the remaining
        genes were gated (initial_route set) but never attempted."""
        recs = [r for r in self.genes.values() if r.attempted] if attempted_only else self.genes.values()
        rows = [{"gene": r.name, "initial_route": r.initial_route, "route": r.route,
                 "stage": r.stage, "nz": r.nz, "fit_ok": r.fit_ok, "attempted": r.attempted,
                 "nbi_explode": r.nbi_explode, "mean_model_chosen": r.mean_model_chosen,
                 "n_removed": r.n_removed, "w1_train": r.w1_train, "fail_reason": r.fail_reason}
                for r in recs]
        return pd.DataFrame(rows).set_index("gene")

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "genes.pkl", "wb") as f: pickle.dump(self.genes, f)
        with open(directory / "scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        if self.rare_glm is not None:
            with open(directory / "rare_glm.pkl", "wb") as f: pickle.dump(self.rare_glm, f)
        _SKIP = {"genes", "scaler", "X_hc_scaled", "Y_hc", "is_hc", "rare_glm",
                 "pc_gene_names", "pc_indices", "alpha_fn"}
        cfg = {k: v for k, v in vars(self).items()
               if not k.startswith("_") and k not in _SKIP}
        with open(directory / "config.pkl", "wb") as f: pickle.dump(cfg, f)
        df = self.training_summary(attempted_only=False)
        df.to_csv(directory / "training_summary.csv")
        print(f"Engine saved to {directory}/")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "config.pkl", "rb") as f: cfg = pickle.load(f)
        engine = cls(**{k: v for k, v in cfg.items()
                        if k in cls.__init__.__code__.co_varnames})
        with open(directory / "genes.pkl", "rb") as f: engine.genes = pickle.load(f)
        with open(directory / "scaler.pkl", "rb") as f: engine.scaler = pickle.load(f)
        rare_glm_path = directory / "rare_glm.pkl"
        if rare_glm_path.exists():
            with open(rare_glm_path, "rb") as f: engine.rare_glm = pickle.load(f)
        engine.alpha_fn = load_trend()
        n_ok = sum(1 for r in engine.genes.values() if r.fit_ok)
        print(f"Engine loaded from {directory}/  ({n_ok} fitted genes)")
        return engine


# Resolve genes.pkl pickled under the old module name model_engine_v2.
sys.modules.setdefault("model_engine_v2", sys.modules[__name__])
