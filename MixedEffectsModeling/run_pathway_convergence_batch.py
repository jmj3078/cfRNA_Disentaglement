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

ZDIR = Path(__file__).resolve().parent / 'Z_scores_mixed'
PCDIR = config.PATHWAY_CONV_DIR

# n>=20 (excluding CAD_HF+/CAD_HF-), sorted descending by n -- confirmed with user. Pancreatic Cancer
# is included too so its output files pick up the same {phenotype}_{null,universe,sig}.pkl naming
# as everything else, replacing the old bare Pancreatic_Cancer_null.npz-only cache.
PHENOTYPES = [
    'Tuberculosis', 'ME/CFS', 'Pancreatitis', 'Pancreatic Cancer', 'Pre-eclampsia',
    'Liver Cancer', 'Colorectal Cancer', 'Lung Cancer', 'Esophagus Cancer', 'Stomach Cancer',
]

if __name__ == '__main__':
    Z = np.load(ZDIR / 'Z_disease_shash.npy')
    gene_names = pickle.load(open(ZDIR / 'gene_names.pkl', 'rb'))
    meta = pd.read_csv(ZDIR / 'sample_meta.csv')

    universe_syms, sym2idx, col2sym = load_symbol_vocab(gene_names)
    terms, M = load_pathway_library()
    print(f'universe: {len(universe_syms)} symbols, {len(terms)} pathways', flush=True)

    results = []
    for phenotype in PHENOTYPES:
        print(f'--- {phenotype} ---', flush=True)
        summary = run_phenotype(phenotype, Z, gene_names, meta, universe_syms, sym2idx, col2sym, terms, M)
        if summary is None:
            print(f'  skipped (0 OOD-kept patients)', flush=True)
            continue
        print(f'  n_pat={summary["n_pat"]}  gene_sig_median={summary["gene_sig_median"]:.0f}  '
              f'path_sig_median={summary["path_sig_median"]:.0f}  zero_path_sig={summary["n_zero_path_sig"]}',
              flush=True)
        results.append(summary)

    pd.DataFrame(results).to_csv(PCDIR / 'batch_summary.csv', index=False)
    print('done, summary written to PathwayConvergence/batch_summary.csv', flush=True)
