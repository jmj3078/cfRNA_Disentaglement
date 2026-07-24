import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import build_trend, load_trend, save_trend
from MixedEffectsModeling.core.marginal_rqr import _poisson_rqr, marginal_nb_rqr

MP = config.SPIKE_PARAMS


@dataclass
class GeneRecordMixed:
    name: str
    route: str = ""
    stage: str = ""
    nz: int = 0
    ok: bool = False
    singular: bool = False
    tau2: float = 0.0
    mu_coef: np.ndarray = None
    disp_coef: np.ndarray = None
    fail_reason: str = ""
    mean_hc: float = None
    fixed_alpha: float = None


class NormativeModelEngineMixed:
    def __init__(self):
        self.X_hc_scaled = None
        self.Y_hc = None
        self.scaler = None
        self.batch = None
        self.pc_gene_names = []
        self._gene_col = {}
        self.genes = {}
        self.alpha_fn = None
        self.nz_a_max = None
        self.rare_glm = None

    def load_hc_data(self, h5ad_path=config.H5AD_PATH):
        adata = sc.read_h5ad(h5ad_path)
        adata = adata[adata.obs["QC_Passed"] == True]
        adata = adata[adata.obs["Phenotype_Processed"].notna()]
        adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
        adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]
        is_hc = (adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
        X_raw = adata.obs[config.BIAS_COLUMNS].values.astype(np.float64)
        self.scaler = StandardScaler()
        self.X_hc_scaled = self.scaler.fit_transform(X_raw[is_hc])
        self.batch = adata.obs["Batch_ID"].astype(str).values[is_hc]
        Y_raw = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        self.Y_hc = np.round(Y_raw[is_hc]).astype(np.float64)
        is_pc = (adata.var["GeneType"] == "protein_coding").values
        self.pc_gene_names = adata.var_names[is_pc].tolist()
        pc_indices = np.where(is_pc)[0]
        self._gene_col = {g: pc_indices[i] for i, g in enumerate(self.pc_gene_names)}

    def build_dispersion_trend(self):
        Y_pc = self.Y_hc[:, list(self._gene_col.values())]
        trend = build_trend(Y_pc, min_nz=MP["trend_min_nz"])
        save_trend(trend)
        self.alpha_fn = load_trend()

    def assign_routes(self):
        # nz_a_max is deferred (Task 6a/6b) -- default to 0 (no gene routed to
        # "pool", every gene attempts the model cascade) until a real threshold
        # is chosen and Threshold_Sweep/nz_a_max.txt exists.
        nz_a_max_path = config.THRESHOLD_SWEEP_DIR / "nz_a_max.txt"
        self.nz_a_max = int(nz_a_max_path.read_text().strip()) if nz_a_max_path.exists() else 0
        nz = (self.Y_hc[:, list(self._gene_col.values())] > 0).sum(axis=0)
        for i, g in enumerate(self.pc_gene_names):
            n = int(nz[i])
            route = "pool" if n < self.nz_a_max else "model"
            self.genes[g] = GeneRecordMixed(name=g, route=route, nz=n)

    def train(self, limit=None, tmp_dir="/tmp/glmm_train"):
        Path(tmp_dir).mkdir(exist_ok=True)
        model_genes = [g for g, r in self.genes.items() if r.route == "model"][:limit]
        pd.DataFrame(self.X_hc_scaled, columns=config.BIAS_COLUMNS).to_csv(f"{tmp_dir}/X.csv.gz")
        Y_model = self.Y_hc[:, [self._gene_col[g] for g in model_genes]]
        pd.DataFrame(Y_model, columns=model_genes).to_csv(f"{tmp_dir}/Y.csv.gz")
        pd.DataFrame({"Batch_ID": self.batch}).to_csv(f"{tmp_dir}/batch.csv.gz")
        pd.DataFrame({"gene": model_genes}).to_csv(f"{tmp_dir}/genes.csv", index=False)

        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp_dir}/X.csv.gz", "--y", f"{tmp_dir}/Y.csv.gz",
            "--batch", f"{tmp_dir}/batch.csv.gz", "--genes", f"{tmp_dir}/genes.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "cascade", "--out", f"{tmp_dir}/results.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        results = pd.read_csv(f"{tmp_dir}/results.csv").set_index("gene")
        for g, row in results.iterrows():
            rec = self.genes[g]
            rec.stage, rec.ok, rec.singular, rec.tau2 = row["stage"], bool(row["ok"]), bool(row["singular"]), float(row["tau2"])
            rec.fixed_alpha = float(row["fixed_alpha"]) if "fixed_alpha" in row and not pd.isna(row["fixed_alpha"]) else None
            rec.mu_coef = row[[c for c in results.columns if c.startswith("mu_coef_")]].values.astype(float)
            rec.disp_coef = row[[c for c in results.columns if c.startswith("disp_coef_")]].values.astype(float)
            rec.fail_reason = row["fail_reason"]
            if not rec.ok:
                rec.route = "excluded"

        self.train_pool(tmp_dir=tmp_dir)

    def train_pool(self, tmp_dir="/tmp/glmm_train"):
        """Route "pool": one shared-beta pooled GLM (+ batch random intercept)
        fit jointly across all pool-route genes via glmm_helpers.R's
        fit_pooled_glmm, called through a small dedicated Rscript wrapper
        (glmm_fit_pool.R) rather than glmm_fit.R's per-gene cascade."""
        pool_genes = [g for g, r in self.genes.items() if r.route == "pool"]
        if not pool_genes:
            return
        Path(tmp_dir).mkdir(exist_ok=True)
        Y_pool = self.Y_hc[:, [self._gene_col[g] for g in pool_genes]]
        pd.DataFrame(Y_pool, columns=pool_genes).to_csv(f"{tmp_dir}/Y_pool.csv.gz")
        pd.DataFrame({"gene": pool_genes}).to_csv(f"{tmp_dir}/genes_pool.csv", index=False)
        # X.csv.gz/batch.csv.gz are the same HC design/batch already written above
        # for the model-route cascade -- reused as-is, not recomputed.

        subprocess.run([
            "Rscript", str(config.GLMM_FIT_POOL_R), "--x", f"{tmp_dir}/X.csv.gz", "--y", f"{tmp_dir}/Y_pool.csv.gz",
            "--batch", f"{tmp_dir}/batch.csv.gz", "--genes", f"{tmp_dir}/genes_pool.csv",
            "--rare-overdisp-thr", str(MP["rare_overdisp_thr"]), "--out", f"{tmp_dir}/results_pool.json",
        ], check=True, cwd=str(config.GLMM_FIT_POOL_R.parent))

        with open(f"{tmp_dir}/results_pool.json") as f:
            fit = json.load(f)

        if not fit["ok"]:
            for g in pool_genes:
                self.genes[g].route = "excluded"
                self.genes[g].fail_reason = "fit_pooled_glmm failed"
            return

        n_hc = self.X_hc_scaled.shape[0]
        self.rare_glm = {"family": fit["family"], "beta": np.asarray(fit["beta"]),
                         "alpha": fit["alpha"], "eps": 1.0 / (2 * n_hc),
                         "tau2": float(fit["tau2"]) if fit.get("tau2") is not None else 0.0,
                         "mult_lo": fit["mult_lo"], "mult_hi": fit["mult_hi"]}
        for g, m in zip(fit["gene"], fit["mean_hc"]):
            rec = self.genes[g]
            rec.mean_hc, rec.ok, rec.stage = float(m), True, "pool"

    def training_summary(self):
        rows = [dict(gene=r.name, route=r.route, stage=r.stage, nz=r.nz, ok=r.ok,
                    singular=r.singular, tau2=r.tau2, fail_reason=r.fail_reason)
               for r in self.genes.values()]
        return pd.DataFrame(rows).set_index("gene")

    def score(self, X_test_raw, Y_test, gene_names=None, seed=42, as_dict=False):
        gene_names = gene_names or [g for g in self.genes if self.genes[g].ok]
        X_test = self.scaler.transform(X_test_raw.astype(np.float64))
        Xa = np.column_stack([np.ones(len(X_test)), X_test])
        Z = np.full((len(X_test), len(gene_names)), np.nan, dtype=np.float32)
        for j, g in enumerate(gene_names):
            rec = self.genes.get(g)
            if rec is None or not rec.ok:
                continue
            if rec.route == "pool":
                # Shared-beta pooled GLM: log(mu) = log(mean_hc+eps) + Xa @ beta
                # (fit_pooled_glmm's offset + fixed-effect formula). tau2 from the
                # model's own (1|batch__) term -- marginalized via marginal_nb_rqr,
                # not assumed 0.
                # Covariate multiplier (slopes only, intercept excluded) clipped to the
                # HC-observed [0.1, 99.9] pct range -- mirrors Modeling/model_engine.py's
                # _rare_z() clip so an OOD-adjacent sample can't extrapolate mu wildly.
                mult = np.exp(X_test @ self.rare_glm["beta"][1:])
                if "mult_lo" in self.rare_glm and self.rare_glm["mult_lo"] is not None:
                    mult = np.clip(mult, self.rare_glm["mult_lo"], self.rare_glm["mult_hi"])
                mu = np.clip((rec.mean_hc + self.rare_glm["eps"]) * np.exp(self.rare_glm["beta"][0]) * mult, 1e-6, 1e8)
                if self.rare_glm["family"] == "poisson":
                    Z[:, j] = _poisson_rqr(Y_test[:, j].astype(np.float64), mu, seed + j)
                else:
                    Z[:, j] = marginal_nb_rqr(Y_test[:, j].astype(np.float64), mu, self.rare_glm["alpha"],
                                              self.rare_glm.get("tau2", 0.0), seed + j)
                continue
            mu = np.clip(np.exp(Xa @ np.nan_to_num(rec.mu_coef, nan=0.0)), 1e-6, 1e8)
            # Full per-sample linear predictor (intercept + covariate slopes), not just
            # the intercept -- sigma_glmmtmb(x) = exp(-X @ disp_coef), verified sign
            # convention from the Step 0 spike (task-4-report.md). Using disp_coef[0]
            # alone silently flattened dispersion for every stage-nbi gene.
            if not np.all(np.isnan(rec.disp_coef)):
                alpha = np.exp(-Xa @ np.nan_to_num(rec.disp_coef, nan=0.0))
            elif rec.fixed_alpha is not None:
                # nb_fixed/intercept: dispersion fixed at TRAINING time from the trend
                # evaluated at the training mean -- reuse that value verbatim rather than
                # recomputing from the scored batch's own mean (which would make
                # dispersion depend on the cohort being scored).
                alpha = np.full(len(X_test), rec.fixed_alpha)
            else:
                alpha = np.full(len(X_test), self.alpha_fn(float(mu.mean())))
            Z[:, j] = marginal_nb_rqr(Y_test[:, j].astype(np.float64), mu, alpha, rec.tau2, seed + j)
        return Z if not as_dict else {"combined": Z, "gene_names": list(gene_names)}

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "genes.pkl", "wb") as f: pickle.dump(self.genes, f)
        with open(directory / "scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        if self.rare_glm is not None:
            with open(directory / "rare_glm.pkl", "wb") as f: pickle.dump(self.rare_glm, f)
        self.training_summary().to_csv(directory / "training_summary.csv")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        engine = cls()
        with open(directory / "genes.pkl", "rb") as f: engine.genes = pickle.load(f)
        with open(directory / "scaler.pkl", "rb") as f: engine.scaler = pickle.load(f)
        rare_glm_path = directory / "rare_glm.pkl"
        if rare_glm_path.exists():
            with open(rare_glm_path, "rb") as f: engine.rare_glm = pickle.load(f)
        engine.alpha_fn = load_trend()
        return engine
