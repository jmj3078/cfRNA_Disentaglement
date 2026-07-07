# References — Lung Cancer (Chen)

OT disease reference: lung carcinoma. rare-led = 0, no DB-supported novel among displayed terms. Retrieved via paper-lookup (PubMed) and database-lookup (Open Targets db_support pre-computed in packet).

| Gene | Claim (pathway) | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| EGFR / SRC / PTPN11 | EGFR signaling | EGFR+lung cancer >31000 (very large, established) | 31562956 (Rare EGFR mutations in non-small cell lung cancer) | EGFR 0.89; SRC 0.61; PTPN11 0.49 |
| RET / SRC / PIK3R1 / PTPN11 | RET signaling / GAB1 signalosome | RET+lung cancer ~1700; SRC+lung cancer ~1490 | 40136350 (Lung Cancer: Targeted Therapy in 2025); 32846060 (Efficacy of Selpercatinib in RET Fusion-Positive NSCLC); 37627207 (The Role of GAB1 in Cancer) | SRC 0.61; PIK3R1 0.54; PTPN11 0.49 |
| JAK1 / PIK3R1 | Interleukin-7 signaling | (immune/JAK axis; not individually counted) | (see EGFR/RET set) | JAK1 0.55; PIK3R1 0.54 |
| NPM1 | Ribosome / ribonucleoprotein biogenesis | NPM1+lung cancer ~4 (sparse; NPM1 literature dominated by AML) | (no lung-specific PMID retrieved) | NPM1 0.70 |
| CACNA1D | Membrane depolarization / action potential | (ion-channel; not individually counted) | (candidate) | CACNA1D 0.53 |

Notes:
- EGFR / RET / GAB1 / JAK = established (known) RTK driver axes, signal originates in count route (NBI), not rare branch.
- NPM1 = high OT score but lung-specific literature sparse -> per-disease association a candidate; ribosome/translation downregulation is a non-specific pan-phenotype axis.
- CACNA1D = candidate (neuroendocrine/ion-channel), needs validation.
