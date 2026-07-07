# ME_CFS (Gardella) — GSEA with_rare reference (skill-retrieved only)

Disease: myalgic encephalomyelitis/chronic fatigue syndrome
Open Targets disease ID: MONDO_0005404
Retrieval: paper-lookup (PubMed E-utilities, esearch counts) + database-lookup (Open Targets GraphQL evidences endpoint). Date 2026-07-06.

## Per-gene / per-axis evidence

### INS (insulin) — rare-led, KEGG Autophagy / Positive Reg. Cell Differentiation
- PubMed: `insulin AND (ME/CFS OR chronic fatigue)` ~116 hits; direct insulin–ME/CFS papers largely tangential (no clean landmark).
- Open Targets INS x MONDO_0005404: 36 evidences, ALL datasource=europepmc, datatype=literature (co-mention), top resourceScore 33 / score 0.33. No genetic/known-drug evidence.
- Verdict: metabolic/energy axis established; INS itself literature-mined co-mention only, causal role unestablished.

### PLA2G10 — rare-led, Regulation Of Lipid Storage (GO:0010883)
- PubMed: `PLA2G10 AND (ME/CFS OR myalgic encephalomyelitis)` = 0 hits. Gene total ~97 hits.
- Rep PMID 38669316 — "Up-regulated PLA2G10 in cancer impairs T cell infiltration to dampen immunity." Sci Immunol 2024 (secreted PLA2 group X, immune/lipid function).
- Open Targets PLA2G10 x MONDO_0005404: 0 evidences.
- Verdict: NO literature / NO DB support for ME/CFS. Genuine novel candidate (lipid-inflammation axis), needs validation.

### LACRT — rare-led, Positive Regulation Of Autophagy (GO:0010508)
- PubMed: `LACRT AND (chronic fatigue OR ME/CFS)` = 0 hits. Gene total ~36 hits (mostly lacrimal/dry-eye tear protein).
- Open Targets: not separately queried; no known ME/CFS association expected.
- Verdict: low-expression secretory protein; likely low-expression residual-variance artifact. Do not over-interpret.

### Metabolic / energy dysregulation axis (context for INS/PLA2G10)
- PubMed: `(metabolomics OR lipidomic) AND (ME/CFS OR myalgic encephalomyelitis)` ~122 hits.
- Rep PMID 40715814 — "AI-driven multi-omics modeling of myalgic encephalomyelitis/chronic fatigue syndrome." Nat Med 2025.
- Autophagy: `autophagy AND ME/CFS` ~28 hits (some literature).
- Verdict: metabolic/energy/lipid dysregulation is an established ME/CFS axis; supports direction of novel NES<0 terms.

### Translation / ribosome axis (strongest non-novel, NES -2.6 to -2.2)
- Not gene-specific; established downregulation theme. PubMed `(ribosome OR translation) AND ME/CFS` ~342 (broad term match, interpret loosely).
