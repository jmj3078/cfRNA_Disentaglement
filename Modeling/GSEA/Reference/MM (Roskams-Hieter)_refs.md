# MM (Roskams-Hieter) — GSEA with_rare reference (skill-retrieved)

ot_disease: plasma cell myeloma. Sources: PubMed E-utilities (esearch/esummary), Open Targets GraphQL. Retrieved 2026-07-06.

| Gene | Pathway context | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| CCND1 | KEGG Cell cycle (up, NES +2.12) | ~236 (CCND1 AND multiple myeloma) | 42261309 (Prognostic genes related to centrosome amplification in multiple myeloma) | CCND1–plasma cell myeloma score 0.534 (also breast cancer 0.664). Established t(11;14) driver |
| RPL10L | Translation/ribosome (down, NES -3.39, rare lead) | 0 (RPL10L AND multiple myeloma); ~6 (RPL10L AND cancer) | 39380204 (SMARTdb reproductive multi-omics DB — off-target) | none MM-specific. Testis-restricted ribosomal retrogene → low-expression artifact candidate |
| DEFB113/127/125/106A/4A/116/126 | Reactome Defensins (down, NES -1.88, rare lead) | 2 (defensin AND multiple myeloma) | 33420397 (salivary changes post-HSCT), 16285021 (Limenin defensin-like peptide) — both non-mechanistic | not queried; low-expression epithelial cluster → artifact candidate |
| H4C7 (HIST1H4G) | KEGG SLE / NET formation (up, NES +1.75/+1.7, rare lead) | ~3 (H4C7 OR HIST1H4G AND cancer), non-specific | 34319233, 26911428 (not MM-specific) | NET pathway db_support PIK3CA/MTOR/RAF1 (~0.30-0.31), but rare lead is histone. Plasma-cell chromatin plausible, MM-specific literature absent → novel candidate |
| Ribosomal proteins (axis) | Translation/elongation down (nbi, not rare) | ~164 (ribosomal protein AND multiple myeloma) | 42255948 (RiboCancer / Rps15 CLL translation) | axis established; rare-independent, trustworthy |

Notes:
- MM strongest DB/literature-supported axis = cell cycle (CCND1 etc.), rare-independent.
- All 4 rare-led terms attributable to plasma-low-expression genes; flag artifact vs signal.
