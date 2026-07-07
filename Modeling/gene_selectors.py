"""
Per-phenotype mean-Z ranking utility for GSEA prerank input.

The classifier-based gene selectors (proportion / effect_size / svd /
effect_size_specific / l1_logistic) and their discrimination-control evaluation
were removed pending a redesigned selection/validation scheme. Only the GSEA
ranking helper is kept here.

Usage
-----
    from gene_selectors import GeneSelector

    gs = GeneSelector(Z_dis, dis_pheno, gene_names)
    gs.set_id2sym(adata.var['GeneName'])
    ranking = gs.mean_z_ranking(phenotype)
"""

import numpy as np


class GeneSelector:
    def __init__(self, Z_dis, dis_pheno, gene_names):
        self.Z_dis = Z_dis
        self.pheno = np.array(dis_pheno)
        self.gn = np.array(gene_names)
        self.phenos = np.unique(self.pheno)
        self._id2sym = None

    def set_id2sym(self, mapping):
        if hasattr(mapping, 'to_dict'):
            self._id2sym = mapping.to_dict()
        else:
            self._id2sym = dict(mapping)
        return self

    def _to_sym(self, ids):
        if self._id2sym is None:
            return ids
        return np.array([self._id2sym.get(g, g) for g in ids])

    def compute_ubiquity(self, abs_z_thr=0.5):
        """Per-gene fraction of phenotypes where |mean_Z| > abs_z_thr.

        Returns array of shape (n_genes,) with values in [0, 1].
        Genes close to 1.0 are anomalous across nearly all disease phenotypes
        regardless of direction — non-specific signal candidates.
        """
        counts = np.zeros(self.Z_dis.shape[1], dtype=np.float32)
        for ph in self.phenos:
            m = self.pheno == ph
            counts += np.abs(self.Z_dis[m].mean(axis=0)) > abs_z_thr
        return counts / len(self.phenos)

    def mean_z_ranking(self, phenotype, jitter=1e-7, seed=42,
                       ubiquity_thr=None, ubiquity_abs_z=0.5):
        """Returns {symbol: mean_z} for a given phenotype — GSEA prerank input.

        jitter        : tiny Gaussian noise to break ties among 0-filled genes.
        ubiquity_thr  : float in (0, 1] or None. When set, genes whose
                        cross-disease ubiquity score >= ubiquity_thr are zeroed
                        out before ranking. This suppresses non-specific signal
                        (genes anomalous in most diseases regardless of direction)
                        from dominating the GSEA leading edge.
                        Recommended starting value: 0.5 (anomalous in >50% of
                        phenotypes). Set to None (default) to disable.
        ubiquity_abs_z: |mean_Z| threshold per phenotype used inside
                        compute_ubiquity(). Default 0.5.
        """
        m = self.pheno == phenotype
        mean_z = self.Z_dis[m].mean(axis=0).copy()
        if ubiquity_thr is not None:
            ubiq = self.compute_ubiquity(abs_z_thr=ubiquity_abs_z)
            mean_z[ubiq >= ubiquity_thr] = 0.0
        if jitter > 0:
            rng = np.random.default_rng(seed)
            mean_z = mean_z + rng.normal(0, jitter, len(mean_z))
        syms = self._to_sym(self.gn).tolist()
        return dict(zip(syms, mean_z.tolist()))
