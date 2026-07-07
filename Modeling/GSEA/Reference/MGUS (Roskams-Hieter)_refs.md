# MGUS (Roskams-Hieter) — GSEA with_rare reference (skill-retrieved)

ot_disease: benign monoclonal gammopathy. Sources: PubMed E-utilities, Open Targets GraphQL. Retrieved 2026-07-06.

| Gene | Pathway context | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| B2M | GO Antigen processing/presentation via MHC I (down, NES -2.45) | ~869 (beta-2 microglobulin AND monoclonal gammopathy) | 42295306 (Serum lncRNA TTTY15 in multiple myeloma) | packet db_support B2M 0.01 (low OT score) but B2M = established clinical marker / ISS staging in plasma-cell disease |
| H2BC1 (HIST1H2BA) | Reactome RUNX1 / Mitotic Prophase / Estrogen-dependent gene expression (up, NES +1.83–1.88, rare lead) | ~3 (H2BC1 OR HIST1H2BA AND cancer), non-specific | 34319233 (Epi-mutations for spermatogenic defects by DEHP — off-target) | none MGUS-specific. Low-expression repetitive histone cluster → novel candidate, artifact caution |
| ERBB4 | Estrogen-dependent gene expression (rare lead H2BC1) | not separately queried | — | packet db_support ERBB4 0.18 (weak); likely pathway mis-attribution via shared histone leading edge |

Notes:
- No DB-supported novel terms (n_db=0 for novel set).
- Interpretable axis = antigen presentation (MHC I/II, B2M) downregulation; literature-anchored via B2M, not OT score.
- All 3 rare-led terms attributable to H2BC1 histone cluster → conservative novel candidates.
