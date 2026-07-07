# Pre-eclampsia (Moufarrej) — GSEA with_rare reference (skill-retrieved only)

Disease: preeclampsia
Open Targets disease ID: MONDO_0005081
Retrieval: paper-lookup (PubMed E-utilities, esearch counts) + database-lookup (Open Targets GraphQL evidences endpoint). Date 2026-07-06.

## Per-gene / per-axis evidence

### RPL10L — rare lead across many translation/ribosome terms (NES<0)
- PubMed: `RPL10L AND (preeclampsia OR pre-eclampsia)` = 0 hits. Gene total ~13 hits.
- Rep PMID 32111475 — "A homozygous RPL10L missense mutation associated with male factor infertility and severe oligozoospermia." Fertil Steril 2020 (testis-restricted ribosomal paralog).
- Open Targets RPL10L x MONDO_0005081: 0 evidences.
- Verdict: NO literature / NO DB support. Testis-restricted, low blood expression → strong low-expression artifact suspect. Do NOT attribute placental translation suppression to RPL10L (that axis holds via count branch + literature independently).

### Placental ribosome / translation suppression axis (non-novel, NES<0)
- PubMed: `ribosome AND preeclampsia AND placenta` ~81 hits; `placental translation preeclampsia mTOR` ~12 hits.
- Rep PMID 35950704 — "Gut Dysbiosis Promotes Preeclampsia by Regulating Macrophages and Trophoblasts." Circ Res 2022 (trophoblast context).
- Verdict: placental translation/ribosome downregulation is an established PE axis; supported without RPL10L.

### PTPRD — Presynapse Organization (GO:0099172, novel) / synapse adhesion (NES>0)
- PubMed: `PTPRD AND (preeclampsia OR pre-eclampsia)` = 1 hit.
- Open Targets PTPRD x MONDO_0005081: 2 evidences, datasource=gwas_credible_sets, datatype=genetic_association, scores 0.714 / 0.668.
- Verdict: genuine GWAS genetic association despite thin literature; interesting novel mechanistic candidate.

### LRFN2 — Protein-protein Interactions At Synapses (NES>0)
- Open Targets LRFN2 x MONDO_0005081: 1 evidence, datasource=gwas_credible_sets, genetic_association, score 0.864.
- Verdict: GWAS genetic association present; supports synapse-adhesion up-axis as candidate.

### SLCO3A1 — Sodium-Independent Organic Anion Transport (GO:0043252, novel, NES>0)
- PubMed: `SLCO3A1 AND (preeclampsia OR pre-eclampsia)` = 0 hits.
- Open Targets: not returned as PE evidence.
- Verdict: novel candidate (placental organic anion transporter), no literature.

### IFNL3 / IFN-lambda (hypothesized antiviral axis — NOT supported here)
- PubMed: `(IFNL3 OR interferon lambda 3) AND (preeclampsia OR pre-eclampsia)` = 0 hits (broader `IFNL3 AND preeclampsia` ~2). IFNL3 gene total ~1744 (mostly HCV/HBV; rep PMID 31201901).
- PubMed: `interferon lambda AND (pregnancy OR placenta)` ~60-62 hits (established placental type III IFN antiviral role; rep PMID 30995506 "Shared and Distinct Functions of Type I and Type III Interferons", Immunity 2019).
- Open Targets IFNL3 x MONDO_0005081: 0 evidences.
- Verdict: NO PE-specific IFNL3 support. The "Influenza/Viral mRNA Translation" terms in the top table are translation-machinery terms (RPL10L-driven, NES<0), NOT an IFN-lambda response. Hypothesized IFN-lambda axis is not evidenced in this result.
