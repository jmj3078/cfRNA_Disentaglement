import pickle
import re

import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom

import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.calibration import bh_fdr_reject

PP = config.PATHWAY_CONV_PARAMS
PCDIR = config.PATHWAY_CONV_DIR


def slugify(phenotype):
    return phenotype.strip().replace(" ", "_").replace("/", "-")


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


# per-sample pathway over-representation: |Z|>z_thresh flags "significant" genes (symbol space,
# universe = genes with a finite collapsed Z in that sample), then a one-sided hypergeometric test
# per pathway (scipy's exact 2x2 Fisher equivalent) asks whether significant genes are enriched in
# that pathway beyond chance, BH-FDR'd across pathways. Adopted over 3 alternatives (HC-population-
# null mean-Z, CAMERA-style PAGE, singscore) after a negative-control benchmark (held-out HC scored
# as fake patients, true null) in _scratch_pathway_methods/2026-08 -- see PP["z_thresh"] comment in
# config.py for the full rationale. z_thresh=1.96 chosen by a 4-point sweep (1.64/1.96/2.33/2.58) as
# the best sensitivity/specificity tradeoff, not just "standard alpha=0.05" convention.
def _cohort(phenotype, Z, meta, include_batches):
    # exact match, not stripped -- meta has a separate 'Pancreatic Cancer ' (trailing space, 2 rows)
    # variant; merging it in would silently change the cohort against the existing cache
    cohort_mask = (meta["phenotype"] == phenotype) & meta["ood_keep"]
    if include_batches:
        cohort_mask &= meta["batch"].isin(include_batches)
    idx = np.where(cohort_mask.values)[0]
    if len(idx) == 0:
        return None, None, None
    return idx, meta["sample"].values[idx], Z[idx]


def _load_universe(pdir, Zc, col2sym, N):
    zu_path = pdir / "universe.pkl"
    if zu_path.exists():
        return pickle.load(open(zu_path, "rb"))
    Zu, Fm = collapse_to_symbols(Zc, col2sym, N)
    pickle.dump((Zu, Fm), open(zu_path, "wb"))
    return Zu, Fm


def run_phenotype(phenotype, Z, gene_names, meta, universe_syms, sym2idx, col2sym, terms, M,
                  include_batches=None, label=None):
    label = label or phenotype
    slug = slugify(label)
    idx, names_c, Zc = _cohort(phenotype, Z, meta, include_batches)
    if idx is None:
        return None
    n_pat = len(idx)

    pdir = PCDIR / slug
    pdir.mkdir(parents=True, exist_ok=True)

    N = len(universe_syms)
    Zu, Fm = _load_universe(pdir, Zc, col2sym, N)

    Mi = M.astype(np.int32)
    n_path = len(terms)
    p_path = np.ones((n_pat, n_path))
    path_sig = np.zeros((n_pat, n_path), dtype=bool)
    n_sig_gene = np.zeros(n_pat, dtype=int)

    for i in range(n_pat):
        present = Fm[i] > 0
        sig = present & (np.abs(Zu[i]) > PP["z_thresh"])
        n_universe, n_sig = int(present.sum()), int(sig.sum())
        n_sig_gene[i] = n_sig
        if n_sig == 0:
            continue
        k_path = Mi @ present.astype(np.int32)
        overlap = Mi @ sig.astype(np.int32)
        valid = k_path > 0
        p = np.ones(n_path)
        p[valid] = hypergeom.sf(overlap[valid] - 1, n_universe, k_path[valid], n_sig)
        p_path[i] = p
        path_sig[i] = bh_fdr_reject(p, q=PP["fdr_q"])

    n_path_sig = path_sig.sum(axis=1)

    pickle.dump(dict(names_c=names_c, terms=terms, n_sig_gene=n_sig_gene, p_path=p_path,
                      path_sig=path_sig, z_thresh=PP["z_thresh"], fdr_q=PP["fdr_q"]),
                open(pdir / "sig.pkl", "wb"))

    return dict(
        phenotype=label, n_pat=n_pat,
        gene_sig_median=float(np.median(n_sig_gene)), gene_sig_max=int(n_sig_gene.max()),
        path_sig_median=float(np.median(n_path_sig)), path_sig_max=int(n_path_sig.max()),
        n_zero_path_sig=int((n_path_sig == 0).sum()),
    )


# directional variant of run_phenotype: up- and down-regulated significant genes tested against
# each pathway as two separate one-sided hypergeometric ORAs (background = all present genes in
# both, "success" = only that direction -- standard Enrichr/DAVID-style up/down split), BH-FDR'd
# JOINTLY across the concatenated [up, down] p-values per sample so the per-sample FDR budget
# covers both directions together. Written to a separate sig_directional.pkl -- does not touch or
# require the undirected run_phenotype() output, reuses the same universe.pkl cache.
def run_phenotype_directional(phenotype, Z, gene_names, meta, universe_syms, sym2idx, col2sym, terms, M,
                              include_batches=None, label=None):
    label = label or phenotype
    slug = slugify(label)
    idx, names_c, Zc = _cohort(phenotype, Z, meta, include_batches)
    if idx is None:
        return None
    n_pat = len(idx)

    pdir = PCDIR / slug
    pdir.mkdir(parents=True, exist_ok=True)

    N = len(universe_syms)
    Zu, Fm = _load_universe(pdir, Zc, col2sym, N)

    Mi = M.astype(np.int32)
    n_path = len(terms)
    p_up = np.ones((n_pat, n_path))
    p_down = np.ones((n_pat, n_path))
    path_sig_up = np.zeros((n_pat, n_path), dtype=bool)
    path_sig_down = np.zeros((n_pat, n_path), dtype=bool)
    n_sig_up = np.zeros(n_pat, dtype=int)
    n_sig_down = np.zeros(n_pat, dtype=int)

    for i in range(n_pat):
        present = Fm[i] > 0
        n_universe = int(present.sum())
        sig_up = present & (Zu[i] > PP["z_thresh"])
        sig_down = present & (Zu[i] < -PP["z_thresh"])
        n_sig_up[i], n_sig_down[i] = int(sig_up.sum()), int(sig_down.sum())
        if n_sig_up[i] == 0 and n_sig_down[i] == 0:
            continue
        k_path = Mi @ present.astype(np.int32)
        valid = k_path > 0
        pu, pd = np.ones(n_path), np.ones(n_path)
        if n_sig_up[i] > 0:
            overlap_up = Mi @ sig_up.astype(np.int32)
            pu[valid] = hypergeom.sf(overlap_up[valid] - 1, n_universe, k_path[valid], n_sig_up[i])
        if n_sig_down[i] > 0:
            overlap_down = Mi @ sig_down.astype(np.int32)
            pd[valid] = hypergeom.sf(overlap_down[valid] - 1, n_universe, k_path[valid], n_sig_down[i])
        p_up[i], p_down[i] = pu, pd

        # joint BH-FDR across both directions' 2*n_path p-values for this sample
        reject = bh_fdr_reject(np.concatenate([pu, pd]), q=PP["fdr_q"])
        path_sig_up[i], path_sig_down[i] = reject[:n_path], reject[n_path:]

    pickle.dump(dict(names_c=names_c, terms=terms, n_sig_up=n_sig_up, n_sig_down=n_sig_down,
                      p_up=p_up, p_down=p_down, path_sig_up=path_sig_up, path_sig_down=path_sig_down,
                      z_thresh=PP["z_thresh"], fdr_q=PP["fdr_q"]),
                open(pdir / "sig_directional.pkl", "wb"))

    n_up, n_down = path_sig_up.sum(axis=1), path_sig_down.sum(axis=1)
    return dict(
        phenotype=label, n_pat=n_pat,
        path_sig_up_median=float(np.median(n_up)), path_sig_up_max=int(n_up.max()),
        path_sig_down_median=float(np.median(n_down)), path_sig_down_max=int(n_down.max()),
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
