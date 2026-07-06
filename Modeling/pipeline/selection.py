import gc

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import config
from gene_selectors import GeneSelector

MP = config.MODELING_PARAMS
CV = MP['n_splits']
K = 15
BASE_FPR = np.linspace(0, 1, 101)


def gene_index(gene_names):
    g2i = {g: i for i, g in enumerate(gene_names)}
    return g2i, (lambda genes: [g2i[g] for g in genes if g in g2i])


def eval_unsupervised(idx, Z_dis, dis_pheno):
    X = Z_dis[:, idx]
    sil = silhouette_samples(X, dis_pheno, metric='euclidean')
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric='euclidean', n_jobs=-1).fit(X)
    nn = nbrs.kneighbors(X)[1][:, 1:]
    pur = np.array([np.mean(dis_pheno[nn[i]] == dis_pheno[i]) for i in range(len(dis_pheno))])
    rows = [{'phenotype': ph,
             'silhouette': float(sil[dis_pheno == ph].mean()),
             'knn_purity': float(pur[dis_pheno == ph].mean())}
            for ph in np.unique(dis_pheno)]
    return pd.DataFrame(rows).set_index('phenotype'), float(sil.mean())


def _select_idx(Z, pheno, gene_names, method, n_per_pheno):
    """Run a GeneSelector method on an arbitrary row subset, return column indices.

    Single source of truth for selection logic so the nested-CV path cannot drift from
    the full-data selectors used elsewhere.
    """
    gs = GeneSelector(Z, pheno, gene_names)
    genes = {'proportion': gs.proportion,
             'effect_size': gs.effect_size,
             'svd': gs.svd_signature,
             'effect_size_specific': gs.effect_size_specific,
             'l1': gs.l1_logistic}[method](n_per_pheno=n_per_pheno)
    g2i = {g: i for i, g in enumerate(gene_names)}
    return [g2i[g] for g in genes if g in g2i]


CONTRAST_METHODS = ('effect_size_specific', 'l1')


def _fold_select(method, X, tr, yy, sub_ph, gene_names, n_per_pheno):
    if method in CONTRAST_METHODS:
        lab = np.where(yy[tr] == 1, '_d', '_hc')
        return _select_idx(X[tr], lab, gene_names, method, n_per_pheno)
    pos = tr[yy[tr] == 1]
    if len(pos) < 2:
        return None
    return _select_idx(X[pos], sub_ph[pos], gene_names, method, n_per_pheno)


def _binary_auc_nested(X, y, gene_names, method, n_per_pheno, seed):
    skf = StratifiedKFold(CV, shuffle=True, random_state=seed)
    a = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        if method == 'full_z':
            w = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)   # class-mean direction
            p = X[te] @ w                                               # held-out projection
        else:
            idx = _fold_select(method, X, tr, y, np.where(y == 1, '_d', '_hc'),
                               gene_names, n_per_pheno)
            if not idx:
                continue
            lr = LogisticRegression(max_iter=200, C=0.1, solver='liblinear').fit(X[tr][:, idx], y[tr])
            p = lr.predict_proba(X[te][:, idx])[:, 1]
        a.append(roc_auc_score(y[te], p))
    return float(np.mean(a)) if a else np.nan


def _random_labels(n, seed):
    """Label ~half of n samples at random as pseudo-disease (breaks batch structure)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[rng.choice(n, n // 2, replace=False)] = 1
    return y


def _batch_group_labels(batch, seed):
    """Assign whole batches to the pseudo-disease side until ~half the samples are labelled
    (preserves batch structure). Returns None if fewer than 2 batches are available."""
    cnt = pd.Series(batch).value_counts()
    if len(cnt) < 2:
        return None
    ub = cnt.index.values.copy()
    np.random.default_rng(seed).shuffle(ub)
    picked, cum = set(), 0
    for b in ub[:-1]:                       # keep at least one batch on the null side
        picked.add(b)
        cum += cnt[b]
        if cum >= len(batch) // 2:
            break
    return np.isin(batch, list(picked)).astype(int)


def discrimination_control(Z_dis, dis_pheno, dis_batch, Z_hc, hc_batch, gene_names,
                           methods=('full_z', 'proportion', 'effect_size', 'svd'),
                           n_per_pheno=30, n_seeds=4, seed=42):
    dis_pheno = np.array(dis_pheno)
    dis_batch = np.array([str(b) for b in dis_batch])
    hc_batch = np.array([str(b) for b in hc_batch])
    seeds = list(range(seed, seed + n_seeds))
    phenos = [ph for ph in np.unique(dis_pheno) if (dis_pheno == ph).sum() >= CV]
    rows = []
    for method in methods:
        for s in seeds:
            auc = _binary_auc_nested(Z_hc, _random_labels(len(Z_hc), s),
                                     gene_names, method, n_per_pheno, s)
            rows.append({'method': method, 'phenotype': '', 'kind': 'random',
                         'seed': s, 'n_pos': len(Z_hc) // 2, 'auc': auc})
        bn_cache = {}                       # dropped-batch set -> {seed: batch_null AUC}
        for ph in phenos:
            m = dis_pheno == ph
            drop = frozenset(dis_batch[m])
            hc_keep = ~np.isin(hc_batch, list(drop))       # HC outside the disease's batch
            if hc_keep.sum() < CV or len(set(hc_batch[hc_keep])) < 2:
                continue                                    # no valid disease-matched null
            Zhc_k, batch_k = Z_hc[hc_keep], hc_batch[hc_keep]
            if drop not in bn_cache:        # diseases sharing a batch share the same null
                bn_cache[drop] = {}
                for s in seeds:
                    yb = _batch_group_labels(batch_k, s)
                    bn_cache[drop][s] = (np.nan if yb is None else
                                         _binary_auc_nested(Zhc_k, yb, gene_names, method, n_per_pheno, s))
            for s in seeds:
                rows.append({'method': method, 'phenotype': ph, 'kind': 'batch_null',
                             'seed': s, 'n_pos': hc_keep.sum() // 2, 'auc': bn_cache[drop][s]})
                X = np.vstack([Z_dis[m], Zhc_k])
                y = np.r_[np.ones(m.sum()), np.zeros(len(Zhc_k))].astype(int)
                dz = _binary_auc_nested(X, y, gene_names, method, n_per_pheno, s)
                rows.append({'method': method, 'phenotype': ph, 'kind': 'disease',
                             'seed': s, 'n_pos': int(m.sum()), 'auc': dz})
                
                del X, y
                gc.collect()
        med = lambda k: np.nanmedian([r['auc'] for r in rows
                                      if r['method'] == method and r['kind'] == k] or [np.nan])
        print(f'{method}: random median={med("random"):.3f}  '
              f'batch_null median={med("batch_null"):.3f}  disease median={med("disease"):.3f}')
        del bn_cache
        gc.collect()
        
    return pd.DataFrame(rows)


def run_selection(Z_dis, Z_hc, dis_pheno, gene_names, n_per_pheno=30):
    """Return (all_results dict, selectors) with the selected gene set and leakage-free
    unsupervised structure (silhouette / kNN purity) per selector.

    Supervised discrimination is NOT computed here -- see discrimination_control() for the
    leakage-controlled random / batch / disease held-out AUC diagnostic. Z_hc is kept in the
    signature for backward compatibility.
    """
    import time
    gs = GeneSelector(Z_dis, dis_pheno, gene_names)
    selectors = gs.get_selectors(n_per_pheno=n_per_pheno)
    _, idx_of = gene_index(gene_names)
    all_results = {}
    for name, selector in selectors.items():
        t0 = time.perf_counter()
        genes = selector()
        idx = idx_of(genes)
        per_pheno, macro_sil = eval_unsupervised(idx, Z_dis, dis_pheno)
        per_pheno.insert(0, 'n', pd.Series(dis_pheno).value_counts())
        all_results[name] = dict(
            genes=genes, per_pheno=per_pheno, macro_sil=macro_sil,
            n_genes=len(genes), t=time.perf_counter() - t0)
        print(f'{name}: sil={macro_sil:.3f}  n_genes={len(genes)}  ({all_results[name]["t"]:.0f}s)')
    return all_results, selectors
