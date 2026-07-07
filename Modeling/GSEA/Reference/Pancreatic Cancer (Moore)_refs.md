# References — Pancreatic Cancer (Moore)

OT disease reference: exocrine pancreatic carcinoma. Data below retrieved via paper-lookup (PubMed E-utilities) and database-lookup (Open Targets; scores as pre-computed db_support in packet, source = OT association score).

| Gene | Claim (pathway) | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| MSTN (myostatin) | SMAD phosphorylation (rare-led, novel) | myostatin+cancer ~362; MSTN+pancreatic cancer ~5 (sparse, cachexia context) | 33899538 (Role of myokines and osteokines in cancer cachexia); 30622678 (Molecular therapeutic strategies targeting pancreatic cancer induced cachexia) | MSTN not the OT-scored target; leading-edge co-genes TGFBR2 0.63, BMPR1A 0.46, PPARG 0.40 |
| CDKN2A | Stabilization of p53 | CDKN2A+pancreatic cancer >1000; CDKN2A+PDAC hits | 28810144 (Integrated Genomic Characterization of Pancreatic Ductal Adenocarcinoma) | CDKN2A 0.73 (established PDAC driver) |
| DDR2 | Collagen fibril organization / non-integrin ECM | DDR2+pancreatic cancer ~12 (small) | 36530986 (COL10A1-DDR2 axis promotes progression of pancreatic cancer via MEK/ERK) | DDR2 0.46; NF1 0.38 |
| U2AF1 / RBM10 | mRNA splicing via spliceosome | U2AF1+pancreatic ~8 (sparse); RBM10+cancer ~178 | (no single strong PDAC-specific PMID retrieved) | RBM10 0.59; U2AF1 0.40 |
| YAP1 / WWTR1 / FAT4 | Hippo signaling | YAP1+pancreatic cancer ~239 | (established axis; no single PMID pulled) | WWTR1 0.46; YAP1 0.40; FAT4 0.38 |

Notes:
- MSTN and U2AF1 = conservative novel candidates (sparse disease-specific literature); MSTN additionally flagged as low-expression rare-branch (residual-variance risk).
- CDKN2A, YAP1/Hippo, DDR2/ECM = established (known) axes, already significant without rare branch.
- Dominant OXPHOS/electron-transport downregulation is a pan-phenotype signal, not PDAC-specific — no per-gene DB support (n_db=0).
