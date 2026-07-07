# References — Other Cancer (Moore)

OT disease reference: generic cancer (EFO_0000311) — low disease specificity, interpret conservatively. rare-led = 6. Retrieved via paper-lookup (PubMed) and database-lookup (Open Targets db_support pre-computed in packet).

| Gene | Claim (pathway) | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| NOX1 | Positive regulation of VEGF production / RHO GTPase cycle (rare-led) | NOX1+cancer ~391; NOX1+VEGF ~45 | 34073365 (Expression and Prognostic Characteristics of Ferroptosis-Related Genes in Colon Cancer); 27874952 (Nox1 promotes colon cancer cell metastasis via ADAM17 pathway) | leading-edge co-genes BRCA1 0.93, PIK3R1 0.87, ARHGAP35 0.75 (not NOX1 itself) |
| CLDN25 / CLDN17 | Tight junction assembly (rare-led) | claudin+tight junction+cancer ~1400 (pathway large); CLDN25 total ~7; CLDN17+cancer ~11 (paralogs sparse) | 37068504 (Zolbetuximab plus mFOLFOX6 in CLDN18.2-positive gastric/GEJ adenocarcinoma, SPOTLIGHT phase 3 — claudin-family precedent) | STRN 0.77 (co-gene) |
| PIK3R1 / ARHGAP35 | RHOB GTPase cycle | (established motility axis) | (see NOX1 set) | PIK3R1 0.87; ARHGAP35 0.75 |
| KDR / NRG1 / TSC1 / RAC1 / FYN | Regulation of focal adhesion assembly | (pan-cancer adhesion/invasion) | (multi-gene, established) | KDR 0.85; NRG1 0.83; TSC1 0.81; RAC1 0.78; FYN 0.74 |
| DAZ2 / DAZ4 | Positive regulation of translational initiation (rare-led) | DAZ2 total ~41 (azoospermia context only); no cancer literature | 37612512 (The complete sequence of a human Y chromosome); 9557839 (DAZ genes encode proteins in human late spermatids and sperm tails) | no DB support |
| PYDC2 | Negative regulation of IL-1 beta production (rare-led) | PYDC2 total ~4 (very sparse); no cancer literature | (none) | no DB support |

Notes:
- NOX1 = most trustworthy rare signal (ROS-VEGF-angiogenesis pathway established); DB co-support (BRCA1) is leading-edge companion, not NOX1 itself.
- CLDN25/CLDN17 = tight-junction pathway is established in carcinoma, but these specific paralogs are low-expression with almost no literature -> conservative novel candidates (low-expression artifact risk).
- DAZ2/DAZ4 = Y-chromosome AZFc (testis-specific, azoospermia) genes, biologically unrelated to cancer -> ARTIFACT (sex/Y-chromosome compositional residual variance); exclude from interpretation.
- PYDC2 = no literature found; conservative novel candidate.
