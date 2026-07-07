# References — Liver Cancer (Roskams-Hieter) [hepatocellular carcinoma]
Skill-retrieved evidence only (paper-lookup = PubMed E-utilities; database-lookup = Open Targets GraphQL v4). Retrieved 2026-07-06.

| Gene | Claim / disease | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| RPL10L | rare lead (KEGG Coronavirus disease); testis-specific ribosomal L10 paralog | ~1 (RPL10L AND carcinoma) | (none disease-specific) | Not retrieved as HCC-associated. Low-expression paralog artifact candidate. |
| DEFB114 | rare lead (Negative/Regulation of MAP Kinase Activity); beta-defensin 114 | 0 (DEFB114 AND carcinoma) | none | No literature. Artifact/novel candidate. |
| MAPK-activity term (db_support, non-rare) | NF1/EGFR/FLT1/APOE/SH2B3 | n/a | n/a | Packet OT: NF1 0.47, FLT1 0.61, EGFR 0.44, APOE 0.39, SH2B3 0.37 (term-level; rare lead DEFB114 unsupported). |
| CTNNB1 | established HCC driver (Adherens junction, Signaling by VEGF) | large | 38123979 (Targeting MMP9 in CTNNB1 mutant HCC..., Gut 2024) | HCC 0.780 (literature 0.99, somatic_mutation 0.86, genetic_association 0.85). Established. |
| MET | established HCC target (Adherens junction) | ~2254 (MET AND hepatocellular carcinoma) | 40394703 (HHLA2 activates c-Met and identifies patients for targeted therapy in HCC, J Exp Clin Cancer Res 2025) | Packet db_support 0.78. Established. |
| VEGF axis (FLT4/KDR/PIK3CA/FLT1) | angiogenesis (Signaling by VEGF, non-novel) | large | n/a | Packet OT: FLT4 0.70, KDR 0.68, PIK3CA 0.68, FLT1 0.61. Established HCC angiogenesis. |

Notes:
- jaccard 0.585 (rare branch reshaped landscape most). But novel terms are largely ESTABLISHED HCC axes (Wnt/adherens, VEGF angiogenesis, RTK-MAPK, oncogene-induced senescence) resurfacing — positive control for method sensitivity.
- rare leads (RPL10L, DEFB114) are tissue-restricted low-expression genes → artifact caution, not novel biology.
