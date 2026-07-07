# HIV + Tuberculosis (Chang) — retrieved references

Sources actually queried: PubMed E-utilities (esearch count / relevance idlist), Open Targets Platform GraphQL (associatedDiseases). Counts are approximate PubMed hit magnitudes; PMIDs representative. rare-led novel = 0 in this phenotype.

| Gene | Disease context | PubMed hits (query) | Representative PMIDs | Open Targets | Call |
|------|-----------------|---------------------|----------------------|--------------|------|
| LTF (lactoferrin) | Antimicrobial humoral response (TB) | ~77 ("lactoferrin AND tuberculosis") | 26788020; 28642848 | tuberculosis 0.517 (packet 0.52) | Established |
| CXCL9 (MIG) | Active TB blood biomarker | ~138 ("CXCL9 AND tuberculosis") | 38888093; 37740371 | packet 0.11 | Established |
| CCL5 (RANTES) | TB / HIV-suppressive β-chemokine | ~130 ("CCL5 AND tuberculosis") | — | packet 0.09 | Established |
| S100A9 | Active TB neutrophil signature | ~27 ("S100A9 AND tuberculosis") | 35935235; 34849408 | packet 0.09 | Established |
| CTSG | Neutrophil / antimicrobial | (TB myeloid signature; see S100A9/NET refs) | 35935235 | packet 0.37 | Established (myeloid axis) |
| IFNG | TB IFN-γ axis | broad TB IFN-γ literature | 36769182 | packet 0.12 | Established |
| RPS27A | Ribosome/translation down (dominant) | ribosomal housekeeping (not disease-specific) | — | packet 0.46 | Systemic-response surrogate |
| RPL10L | rare gene in rRNA-processing leading edge | 0 ("RPL10L AND tuberculosis") | — | none | No literature found — not independent signal |
| GGTLC3 | rare gene in peptide-biosynthesis leading edge | 0 ("GGTLC3 AND tuberculosis") | — | none | No literature found — not independent signal |
| — | HIV/TB co-infection cfRNA (dataset context) | ~1 ("HIV AND tuberculosis coinfection cfRNA") | — | — | Prior work essentially absent — validation target |

Notes:
- rare-led novel = 0; RPL10L/GGTLC3 appear only as incidental members of RPS27A-driven ribosome/translation terms (novel=False), not independent rare signals.
- Up axes shared TB + HIV antimicrobial/antiviral (LTF, CXCL9, CCL5, S100A9, CTSG, IFNG) + strong ribosome/translation down (RPS27A).
