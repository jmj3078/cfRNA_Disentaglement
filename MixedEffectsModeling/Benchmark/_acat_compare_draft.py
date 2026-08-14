import sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, '/project/cfRNA_NormativeModeling')

import MixedEffectsModeling.config as config
from MixedEffectsModeling.Benchmark import db_hit_compare as dc
from MixedEffectsModeling.core.calibration import bh_fdr_reject
from scipy.stats import hypergeom, norm

ZDIR = dc.ZDIR
Z = np.load(ZDIR / "Z_disease_shash.npy")
sm = pd.read_csv(ZDIR / "sample_meta.csv")
gene_names = pickle.load(open(ZDIR / "gene_names.pkl", "rb"))
sym_of = dc.ensg_to_symbol()
sym_arr = sym_of.reindex(gene_names).values
ref = dc.load_reference()
N_genes = len(gene_names)

sm = sm.copy()
sm["study"] = sm["batch"].str.replace(r"_Batch_\d+$", "", regex=True)


def acat_group_p(Zc, eps=1e-15):
    """ACAT (Cauchy combination, Liu & Xie 2020) of per-patient p-values -> one p per gene.
    p_i = 2*norm.sf(|z_i|); T = mean_i tan((0.5-p_i)*pi); p_ACAT = 0.5 - arctan(T)/pi."""
    finite = np.isfinite(Zc)
    p = np.full(Zc.shape, np.nan)
    p[finite] = 2 * norm.sf(np.abs(Zc[finite]))
    p = np.clip(p, eps, 1 - eps)
    tanvals = np.tan((0.5 - p) * np.pi)
    tanvals = np.where(finite, tanvals, 0.0)
    n = finite.sum(axis=0)
    T = np.divide(tanvals.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
    return 0.5 - np.arctan(T) / np.pi


min_n = 3
acat_out = {}
for (pheno, study), sub in sm[sm.ood_keep].groupby(["phenotype", "study"]):
    if len(sub) < min_n:
        continue
    acat_out[(pheno, study)] = acat_group_p(Z[sub.index.values])

# union across studies per phenotype (same convention as group_level_z_test)
q = 0.05
acat_sets, n_acat = {}, {}
for (pheno, study), pv in acat_out.items():
    finite = np.isfinite(pv)
    reject = np.zeros(len(pv), dtype=bool)
    reject[finite] = bh_fdr_reject(pv[finite], q=q)
    sig = {s for s in sym_arr[reject] if pd.notna(s)}
    acat_sets.setdefault(pheno, set()).update(sig)

rows = []
for pheno, sig in acat_sets.items():
    dref = ref.get(pheno, set())
    if not dref:
        continue
    K, n, x = len(dref), len(sig), len(sig & dref)
    pval = hypergeom.sf(x - 1, N_genes, K, n) if n > 0 else np.nan
    rows.append(dict(method="acat_group", phenotype=pheno, n_sig=n, overlap=x, ref_size=K, pval=pval))
acat_df = pd.DataFrame(rows)

from statsmodels.stats.multitest import multipletests
_, padj, _, _ = multipletests(acat_df.pval.fillna(1), method="fdr_bh")
acat_df["padj"] = padj
acat_df["sig_fdr05"] = padj < 0.05

# compare against existing stouffer (normative_group_z) results already cached
gene_z = pd.read_csv(config.ROOT / 'MixedEffectsModeling' / 'Benchmark' / 'group_level_z_test_multidesign.csv')
stouffer_q05 = gene_z[(gene_z.q == 0.05) & (gene_z.method == 'normative_group_z')]

cmp = stouffer_q05.set_index('phenotype')[['n_sig', 'overlap', 'ref_size', 'pval', 'padj', 'sig_fdr05']]
cmp.columns = ['stouffer_' + c for c in cmp.columns]
acat_i = acat_df.set_index('phenotype')[['n_sig', 'overlap', 'ref_size', 'pval', 'padj', 'sig_fdr05']]
acat_i.columns = ['acat_' + c for c in acat_i.columns]
merged = cmp.join(acat_i, how='outer').sort_values('stouffer_n_sig', ascending=False)
pd.set_option('display.width', 200)
print(merged.to_string())
merged.to_csv('/tmp/acat_vs_stouffer.csv')
