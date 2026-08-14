import pickle
import re

import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm

import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.calibration import bh_fdr_reject

PP = config.PATHWAY_CONV_PARAMS
PCDIR = config.PATHWAY_CONV_DIR


def slugify(phenotype):
    return phenotype.strip().replace(" ", "_").replace("/", "-")


def pairwise_overlap(masks):
    """Jaccard + overlap-coefficient for every patient pair, given a (n_patients, n_items) bool
    matrix. Mirrors Wolfers 2018 JAMA Psych / Segal 2023 Nat Neurosci deviation-overlap design.
    NOTE: raw Jaccard is NOT comparable across item universes of different size (N) -- with small
    per-patient hit counts relative to N, near-zero Jaccard is a combinatorial near-certainty
    regardless of whether hits are real signal or noise. Use `overlap_enrichment` for a
    size-matched-null-normalized statistic before claiming "convergence" or "heterogeneity"."""
    inter = masks.astype(int) @ masks.astype(int).T
    sizes = masks.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    jacc = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
    minsz = np.minimum(sizes[:, None], sizes[None, :])
    ovc = np.divide(inter, minsz, out=np.zeros_like(inter, dtype=float), where=minsz > 0)
    iu = np.triu_indices(masks.shape[0], k=1)
    return jacc[iu], ovc[iu]


def overlap_enrichment(masks, n_perm=200, seed=42):
    """Size-matched-null-normalized pairwise overlap: is observed Jaccard higher than what
    patients with the SAME per-patient hit counts would show if hits were random draws from the
    N-item universe? Each null rep reassigns each patient's own hit count to random items
    (independently per patient) and recomputes the pairwise Jaccard; the null distribution is
    over per-rep MEDIAN Jaccard (matching the summary statistic reported for the observed data),
    not over individual pairs, since ~n_pat*(n_pat-1)/2 pairs from ONE null draw are not
    independent replicates of the null. Returns (obs_median, null_median, null_sd, enrichment_ratio,
    perm_pvalue) -- one-sided: fraction of null-rep medians >= observed median."""
    rng = np.random.default_rng(seed)
    n_pat, N = masks.shape
    sizes = masks.sum(axis=1)
    obs_jacc, _ = pairwise_overlap(masks)
    obs_median = np.median(obs_jacc)

    null_medians = np.empty(n_perm)
    for r in range(n_perm):
        null_masks = np.zeros((n_pat, N), dtype=bool)
        for i, k in enumerate(sizes):
            if k > 0:
                idx = rng.choice(N, size=min(int(k), N), replace=False)
                null_masks[i, idx] = True
        nj, _ = pairwise_overlap(null_masks)
        null_medians[r] = np.median(nj)

    null_median = null_medians.mean()
    null_sd = null_medians.std(ddof=1)
    if null_median > 0:
        ratio = obs_median / null_median
    elif obs_median > 0:
        ratio = np.inf  # unbeatable convergence: null never produced ANY overlap, but observed did
    else:
        ratio = np.nan  # both zero -- truly untestable, not "no signal" in either direction
    pval = (1 + np.sum(null_medians >= obs_median)) / (n_perm + 1)
    return dict(obs_median=obs_median, null_median=null_median, null_sd=null_sd,
                enrichment_ratio=ratio, perm_pvalue=pval)


# ENSG -> gene-symbol vocabulary is phenotype-independent (fixed by the H5AD var table), cached once
# and reused across phenotypes. Zu/Fm (the collapsed Z matrix) is NOT -- it depends on which patients
# are in the cohort, so it's computed fresh per phenotype in collapse_to_symbols below.
def load_symbol_vocab(gene_names):
    path = PCDIR / "symbol_vocab.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    sym_of = sc.read_h5ad(config.H5AD_PATH, backed="r").var["GeneName"].reindex(gene_names)
    syms_all = sym_of.values
    universe_syms = pd.unique(syms_all[sym_of.notna().values])
    sym2idx = {s: i for i, s in enumerate(universe_syms)}
    col2sym = np.array([sym2idx.get(s, -1) if pd.notna(s) else -1 for s in syms_all])
    pickle.dump((universe_syms, sym2idx, col2sym), open(path, "wb"))
    return universe_syms, sym2idx, col2sym


def collapse_to_symbols(Zc, col2sym, N):
    n_pat = Zc.shape[0]
    keep_cols = col2sym >= 0
    Zc_v, sym_idx_v = Zc[:, keep_cols], col2sym[keep_cols]
    sum_mat, finite_cnt = np.zeros((n_pat, N)), np.zeros((n_pat, N))
    for i in range(n_pat):
        np.add.at(sum_mat[i], sym_idx_v, np.nan_to_num(Zc_v[i], nan=0.0))
        np.add.at(finite_cnt[i], sym_idx_v, np.isfinite(Zc_v[i]).astype(float))
    Zu_raw = np.divide(sum_mat, finite_cnt, out=np.full_like(sum_mat, np.nan), where=finite_cnt > 0)
    Zu = np.nan_to_num(Zu_raw, nan=0.0)
    Fm = np.isfinite(Zu_raw).astype(float)
    return Zu, Fm


# pathway gene-set membership (KEGG+Reactome, housekeeping-excluded) is also phenotype-independent
def load_pathway_library():
    lib_path = PCDIR / "gene_set_matrix.pkl"
    if lib_path.exists():
        terms_raw, M_raw = pickle.load(open(lib_path, "rb"))
    else:
        libs = {}
        for lib in PP["gene_sets"]:
            libs.update(gp.get_library(lib, organism="human"))
        universe_syms, sym2idx, _ = load_symbol_vocab(None)
        N = len(universe_syms)
        terms_raw = list(libs.keys())
        M_raw = np.zeros((len(terms_raw), N), dtype=bool)
        for ti, t in enumerate(terms_raw):
            for g in libs[t]:
                j = sym2idx.get(g)
                if j is not None:
                    M_raw[ti, j] = True
        keep = M_raw.sum(axis=1) >= PP["min_pathway_size"]
        terms_raw, M_raw = [t for t, k in zip(terms_raw, keep) if k], M_raw[keep]
        pickle.dump((terms_raw, M_raw), open(lib_path, "wb"))

    ribo_idx = np.where(M_raw[terms_raw.index(PP["ribo_reference_term"])])[0]
    frac_ribo = M_raw[:, ribo_idx].sum(axis=1) / M_raw.sum(axis=1)
    kw_rx = re.compile("|".join(PP["exclude_keywords"]), re.IGNORECASE)
    keep_hk = (frac_ribo <= PP["ribo_frac_max"]) & np.array([not kw_rx.search(t) for t in terms_raw])
    terms, M = [t for t, k in zip(terms_raw, keep_hk) if k], M_raw[keep_hk]
    return terms, M


# per-patient/per-pathway mean-Z + per-patient permutation null (see 4_gene_enrichment.ipynb for the
# statistical rationale: burden-confound-free convergence test). Extraction only -- literature-curated
# pathway shortlisting and visualization stay manual, phenotype-specific follow-up work.
def run_phenotype(phenotype, Z, gene_names, meta, universe_syms, sym2idx, col2sym, terms, M,
                  include_batches=None, label=None):
    # label lets a phenotype be split into per-study sub-cohorts (own slug/output dir, own
    # gene_sig/path_sig/Jaccard/Sankey) instead of pooling different studies' technical variance together
    label = label or phenotype
    slug = slugify(label)
    # exact match, not stripped -- meta has a separate 'Pancreatic Cancer ' (trailing space, 2 rows)
    # variant; merging it in would silently change the cohort against the existing 72-patient cache
    cohort_mask = (meta["phenotype"] == phenotype) & meta["ood_keep"]
    if include_batches:
        cohort_mask &= meta["batch"].isin(include_batches)
    idx = np.where(cohort_mask.values)[0]
    n_pat = len(idx)
    if n_pat == 0:
        return None
    names_c = meta["sample"].values[idx]
    Zc = Z[idx]

    pdir = PCDIR / slug
    pdir.mkdir(parents=True, exist_ok=True)

    p_all = 2 * norm.sf(np.abs(Zc))
    gene_sig = np.zeros_like(p_all, dtype=bool)
    for i in range(n_pat):
        row = p_all[i]
        finite = np.isfinite(row)
        reject = np.zeros(len(row), dtype=bool)
        if finite.any():
            reject[finite] = bh_fdr_reject(row[finite], q=PP["fdr_q"])
        gene_sig[i] = reject
    n_gene_sig = gene_sig.sum(axis=1)

    N = len(universe_syms)
    zu_path = pdir / "universe.pkl"
    if zu_path.exists():
        Zu, Fm = pickle.load(open(zu_path, "rb"))
    else:
        Zu, Fm = collapse_to_symbols(Zc, col2sym, N)
        pickle.dump((Zu, Fm), open(zu_path, "wb"))

    Mf = M.astype(float)
    null_path = pdir / "null.npz"
    if null_path.exists():
        d = np.load(null_path)
        T, null_mean, null_sd = d["T"], d["null_mean"], d["null_sd"]
    else:
        num, den = (Zu * Fm) @ Mf.T, Fm @ Mf.T
        T = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        rng = np.random.default_rng(PP["seed"])
        null_sum, null_sumsq = np.zeros_like(T), np.zeros_like(T)
        for _ in range(PP["n_null_perm"]):
            perm = rng.permutation(N)
            Zp, Fp = Zu[:, perm], Fm[:, perm]
            nu, de = (Zp * Fp) @ Mf.T, Fp @ Mf.T
            Tn = np.divide(nu, de, out=np.zeros_like(nu), where=de > 0)
            null_sum += Tn
            null_sumsq += Tn ** 2
        null_mean = null_sum / PP["n_null_perm"]
        null_sd = np.sqrt(np.clip(null_sumsq / PP["n_null_perm"] - null_mean ** 2, 1e-12, None))
        np.savez(null_path, T=T, null_mean=null_mean, null_sd=null_sd)

    Tz = (T - null_mean) / null_sd
    p_path = 2 * norm.sf(np.abs(Tz))
    path_sig = np.zeros_like(p_path, dtype=bool)
    for i in range(n_pat):
        path_sig[i] = bh_fdr_reject(p_path[i], q=PP["fdr_q"])
    n_path_sig = path_sig.sum(axis=1)

    pickle.dump(dict(names_c=names_c, terms=terms, gene_sig=gene_sig, path_sig=path_sig, Tz=Tz),
                open(pdir / "sig.pkl", "wb"))

    return dict(
        phenotype=label, n_pat=n_pat,
        gene_sig_median=float(np.median(n_gene_sig)), gene_sig_max=int(n_gene_sig.max()),
        path_sig_median=float(np.median(n_path_sig)), path_sig_max=int(n_path_sig.max()),
        n_zero_path_sig=int((n_path_sig == 0).sum()),
    )


def strip_reactome_code(name):
    return name.split(" R-HSA-")[0]


# subagent-written literature reviews all use "## Selected pathways" -> "### [N.] Name [-- tier note]"
# under that heading, but the exact heading format (numbered or not, trailing annotations or not)
# varied by agent -- keep the raw heading text and let match_pathway_index try several derived forms.
def parse_selected_pathways(md_path):
    lines = open(md_path).read().splitlines()
    names, in_section = [], False
    for line in lines:
        if line.strip().startswith("## "):
            in_section = line.strip().lstrip("#").strip().lower() == "selected pathways"
            continue
        if in_section:
            m = re.match(r'^###\s*(?:\d+\.\s*)?(.+?)\s*$', line.strip())
            if m:
                names.append(m.group(1))
    return names


def match_pathway_index(name, terms):
    stripped = [strip_reactome_code(t) for t in terms]

    # derive candidate forms: raw heading, text before an em-dash/hyphen tier-note suffix, and any
    # parenthetical content (agents sometimes wrote "Short Name (Actual Pathway Term)")
    candidates = [name]
    candidates.append(re.split(r'\s+[-—]{1,2}\s+', name)[0])
    paren = re.search(r'\(([^)]+)\)', name)
    if paren:
        candidates.append(paren.group(1))
        candidates.append(re.sub(r'\s*\([^)]+\)', "", name).strip())

    import difflib
    for cand in candidates:
        low = cand.lower().strip()
        if not low:
            continue
        for j, t in enumerate(stripped):
            if t.lower() == low:
                return j
        for j, t in enumerate(stripped):
            if t.lower().startswith(low) or low.startswith(t.lower()):
                return j
        close = difflib.get_close_matches(cand, stripped, n=1, cutoff=0.8)
        if close:
            return stripped.index(close[0])
    return None
