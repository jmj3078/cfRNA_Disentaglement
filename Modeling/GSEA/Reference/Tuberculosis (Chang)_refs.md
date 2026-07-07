# Tuberculosis (Chang) — retrieved references

Sources actually queried: PubMed E-utilities (esearch count / relevance idlist), Open Targets Platform GraphQL (associatedDiseases, disease=tuberculosis). Counts are approximate PubMed hit magnitudes; PMIDs representative.

| Gene | Disease context | PubMed hits (query) | Representative PMIDs | Open Targets (tuberculosis) | Call |
|------|-----------------|---------------------|----------------------|-----------------------------|------|
| LTF (lactoferrin) | Mtb infection / antimicrobial biomarker | ~77 ("lactoferrin AND tuberculosis"); ~25 ("LTF AND tuberculosis") | 26788020; 28642848 | assoc score 0.517 (packet 0.52) | Established |
| GBP1 | IFN-γ antimycobacterial effector / blood signature | ~25 ("GBP1 AND tuberculosis") | 36769182; 35753598 | packet 0.08 | Established |
| GBP5 | IFN-γ effector / blood signature | ~31 ("GBP5 AND tuberculosis") | 36769182; 35753598 | assoc score 0.088 (packet 0.10); meningeal TB 0.068 | Established |
| CORO1A (coronin-1/TACO) | Mtb phagosome survival | ~5 ("CORO1A AND tuberculosis") | 22256790 | packet 0.38 | Established (mechanistic) |
| S100A9 (calprotectin) | Active TB neutrophil signature | ~27 ("S100A9 AND tuberculosis") | 35935235; 34849408 | packet 0.09 | Established |
| CXCL9 (MIG) | Active TB blood/plasma biomarker | ~138 ("CXCL9 AND tuberculosis") | 38888093; 37740371 | — | Established |
| CCL5 (RANTES) | TB chemokine | ~130 ("CCL5 AND tuberculosis") | — | packet 0.09 | Established |
| CRP | TB inflammation marker | ~1108 ("CRP AND tuberculosis") | — | packet 0.12 | Established |
| KRT28 | TB (rare-led epithelium development) | 0 ("KRT28 AND tuberculosis") | — | none | No literature found — artifact candidate |
| DKK4 | TB (rare-led WNT/TCF) | 0 ("DKK4 AND tuberculosis") | — | none | No literature found — artifact candidate |

Notes:
- Established axes (LTF, GBP1/5, CORO1A, S100A9, CXCL9) = only_nbi-significant, novel=False.
- rare-led novel (KRT28 epithelium, DKK4 WNT): 0 TB literature; leading-edge real anchors are housekeeping/metabolic genes -> conservative artifact candidates. DB-supported novel = 0.
