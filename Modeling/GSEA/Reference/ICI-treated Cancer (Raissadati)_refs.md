# ICI-treated Cancer (Raissadati) — GSEA with_rare reference (skill-retrieved)

ot_disease: None (heterogeneous mixed-cancer cohort). NO single Open Targets disease reference → DB cross-validation NOT possible. Interpretation literature-only / reserved. Sources: PubMed E-utilities. Retrieved 2026-07-06.

| Gene / set | Pathway context | PubMed magnitude | Representative PMIDs (title) | Notes |
|---|---|---|---|---|
| OR13F1;OR6K2;OR2K2;OR10G2;OR10H3;OR2S2 | GO Detection of chemical stimulus / smell (up, NES +1.75, rare lead) | ~7 (olfactory receptor genes AND cell-free RNA) | 40020072 (Cystic fibrosis alters olfactory epithelium and OR expression) | Olfactory receptors not expressed in blood/plasma cfRNA; no ICI/cancer-specific literature → strong low-expression artifact, not a signal |
| TAS2R42 | GO Detection of chemical stimulus / taste (up, NES +1.82, rare lead) | ~39 (taste receptor AND leukocyte expression) | 41710893, 41129033 (leukocyte taste-receptor, not ICI/cancer-specific) | Bitter taste receptor, low blood expression → artifact candidate |
| Immune-activation set (Allograft rejection, T-cell mediated cytotoxicity, antigen presentation) | up (nbi, not rare) | ~656 (immune checkpoint inhibitor AND cell-free RNA AND cancer) | 42358824 (ICI + cfRNA + cancer, monitoring context) | Directionally consistent with ICI mechanism but cannot be attributed to a specific disease (mixed cohort) → reserved |
| Translation/ribosome/rRNA (axis) | broad downregulation (nbi) | (not separately queried) | — | robust, rare-independent; cell-of-origin/disease-specificity indeterminate |

Notes:
- No OT reference exists for this heterogeneous cohort; all interpretation is literature-only and reserved.
- All 3 rare-led terms are chemosensory (OR*/TAS2R) receptor clusters → judged artifact, not biological signal.
