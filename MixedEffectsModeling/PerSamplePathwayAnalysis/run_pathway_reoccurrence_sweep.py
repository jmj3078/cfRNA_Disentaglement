import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import MixedEffectsModeling.config as config
from MixedEffectsModeling.PerSamplePathwayAnalysis.pathway_convergence import run_reoccurrence_detail

# same jobs as run_pathway_convergence_batch.py -- run that first, this only reuses its sig.pkl/
# universe.pkl/sig_directional.pkl caches (no Z/engine access needed, just re-thresholding).
PCDIR = config.PATHWAY_CONV_DIR
PHENOTYPES = [
    "Tuberculosis", "Pancreatitis", "Pancreatic Cancer", "Pre-eclampsia",
    "Colorectal Cancer", "Lung Cancer", "Esophagus Cancer", "Stomach Cancer",
]
LIVER_CANCER_STUDIES = [
    "Liver Cancer (Roskams-Hieter B et al.)", "Liver Cancer (Chen et al.)",
]
QS = [0.05, 0.10, 0.15, 0.20, 0.25]

if __name__ == "__main__":
    results = []
    for label in PHENOTYPES + LIVER_CANCER_STUDIES:
        for q in QS:
            summary = run_reoccurrence_detail(label, q, label=label)
            if summary is None:
                print(f'{label} q={q}: skipped (sig.pkl missing, run run_pathway_convergence_batch.py first)',
                      flush=True)
                continue
            print(f'{label} q={q}: n_pat={summary["n_pat"]} '
                  f'gene_median={summary["n_sig_gene_median"]:.0f} path_median={summary["n_sig_path_median"]:.0f}'
                  + (f' up_median={summary["n_sig_path_up_median"]:.0f} down_median={summary["n_sig_path_down_median"]:.0f}'
                     if "n_sig_path_up_median" in summary else ''), flush=True)
            results.append(summary)

    pd.DataFrame(results).to_csv(PCDIR / "reoccurrence_sweep_summary.csv", index=False)
    print("done, summary written to PerSamplePathwayAnalysis/reoccurrence_sweep_summary.csv "
          "-- per-sample detail in PerSamplePathwayAnalysis/<phenotype>/reoccurrence_q*.pkl", flush=True)
