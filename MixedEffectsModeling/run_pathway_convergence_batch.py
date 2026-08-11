import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.pathway_convergence import (
    load_pathway_library, load_symbol_vocab, run_phenotype, slugify,
)

ZDIR = Path(__file__).resolve().parent / "Z_scores_mixed"
PCDIR = config.PATHWAY_CONV_DIR

# n>=20 (excluding CAD_HF+/CAD_HF-), sorted descending by n -- confirmed with user. Pancreatic Cancer
# is included too so its output files pick up the same {phenotype}_{null,universe,sig}.pkl naming
# as everything else, replacing the old bare Pancreatic_Cancer_null.npz-only cache.
# ME/CFS dropped -- literature review found only one weakly-supported pathway (thin, contested
# biomarker base), not enough for a meaningful downstream story.
PHENOTYPES = [
    "Tuberculosis", "Pancreatitis", "Pancreatic Cancer", "Pre-eclampsia",
    "Colorectal Cancer", "Lung Cancer", "Esophagus Cancer", "Stomach Cancer",
]

# Liver Cancer pools 3 different studies (Roskams-Hieter B n=28, Chen n=10, Block n=2). Pooling
# studies (not just batches within one study) blends technical variance into the pathway-convergence
# stats, so run each study as its own cohort instead of merging -- Block et al. (n=2) dropped, too
# small to support even its own gene_sig/path_sig estimate.
LIVER_CANCER_STUDIES = [
    ("Liver Cancer (Roskams-Hieter B et al.)", ["Roskams-Hieter B et al._Batch_2"]),
    ("Liver Cancer (Chen et al.)", ["Chen et al._Batch_1"]),
]

if __name__ == "__main__":
    Z = np.load(ZDIR / "Z_disease_shash.npy")
    gene_names = pickle.load(open(ZDIR / "gene_names.pkl", "rb"))
    meta = pd.read_csv(ZDIR / "sample_meta.csv")

    universe_syms, sym2idx, col2sym = load_symbol_vocab(gene_names)
    terms, M = load_pathway_library()
    print(f'universe: {len(universe_syms)} symbols, {len(terms)} pathways', flush=True)

    jobs = [(ph, ph, None) for ph in PHENOTYPES]
    jobs += [(label, "Liver Cancer", batches) for label, batches in LIVER_CANCER_STUDIES]

    results = []
    for label, phenotype, include_batches in jobs:
        print(f'--- {label} ---', flush=True)
        summary = run_phenotype(phenotype, Z, gene_names, meta, universe_syms, sym2idx, col2sym, terms, M,
                               include_batches=include_batches, label=label)
        if summary is None:
            print(f'  skipped (0 OOD-kept patients)', flush=True)
            continue
        print(f'  n_pat={summary["n_pat"]}  gene_sig_median={summary["gene_sig_median"]:.0f}  '
              f'path_sig_median={summary["path_sig_median"]:.0f}  zero_path_sig={summary["n_zero_path_sig"]}',
              flush=True)
        results.append(summary)

    pd.DataFrame(results).to_csv(PCDIR / "batch_summary.csv", index=False)
    print("done, summary written to PathwayConvergence/batch_summary.csv", flush=True)
