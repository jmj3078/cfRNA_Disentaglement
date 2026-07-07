# HIV (Chang) — retrieved references

Sources actually queried: PubMed E-utilities (esearch, rettype=count / relevance idlist), Open Targets Platform GraphQL (associatedDiseases). Counts are approximate PubMed hit magnitudes; PMIDs are representative, not exhaustive.

| Gene | Disease context | PubMed hits (query) | Representative PMIDs | Open Targets (HIV-1 infection) | Call |
|------|-----------------|---------------------|----------------------|-------------------------------|------|
| MX2 (MxB) | HIV restriction factor | ~114 ("MX2 AND HIV") | 30258007; 25568212 | assoc score 0.105 (packet MX2 0.11) | Established |
| EIF2AK2 (PKR) | HIV antiviral translation block | ~14 ("EIF2AK2 AND HIV") | — | packet 0.52 | Established |
| PPIA (cyclophilin A) | HIV-1 capsid / replication cofactor | ~376 ("cyclophilin A AND HIV"); ~16 ("PPIA AND HIV") | 39480090; 38948800 | packet 0.62 | Established |
| CXCL10 (IP-10) | HIV progression/reservoir biomarker | ~444 ("CXCL10 AND HIV") | 29122683; 37920466 | packet 0.10 | Established |
| IL31 | HIV (rare-led IL-6 family term) | ~4 ("IL31 AND HIV"); ~362 ("IL31 AND inflammation") | — | none | Sparse / novel candidate (artifact watch) |
| MYL10 | HIV (rare-led leukocyte migration) | 0 ("MYL10 AND HIV") | — | none | No literature found — artifact candidate |
| CLDN17 | HIV (rare-led leukocyte migration) | 0 ("CLDN17 AND HIV") | — | none | No literature found — artifact candidate |
| INSL5 | HIV (rare-led relaxin signaling) | 0 ("INSL5 AND HIV") | — | none | No literature found — artifact candidate |
| NOX3 | HIV (RHO GTPase effectors) | ~1 ("NOX3 AND HIV") | — | none | No literature found — novel candidate |
| OPRK1 | HIV (defense response to virus) | ~3 ("OPRK1 AND HIV") | — | packet 0.32 | Sparse |

Notes:
- Established antiviral axis (MX2, EIF2AK2, PPIA, CXCL10) = only_nbi-significant, novel=False; not attributable to rare branch.
- rare-led novel pathways (IL31, MYL10/CLDN17, INSL5): leading edge driven by ultra-low-expression rare genes with 0 HIV literature -> conservative artifact candidates.
