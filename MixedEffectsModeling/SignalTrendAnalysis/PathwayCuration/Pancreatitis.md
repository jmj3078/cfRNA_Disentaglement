# Pancreatitis Pathway Curation

Source: GSEA normative ranking, `normative__Pancreatitis__Moore_et_al..csv`, FDR q<0.05; inflammatory-disease hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

Note: the top ~65 significant terms in this ranking are almost entirely a single redundant supercluster of Reactome ubiquitin-proteasome/ERAD/cell-cycle-degradation annotations (Proteasome, Hh Mutants Are Degraded By ERAD, Stabilization Of P53, APC/C-mediated degradation terms, GLI2/GLI3 processing, ABC Transporter Disorders, etc.), all driven by the same core PSMA/PSMB/PSMC/PSMD proteasome-subunit genes (>50-75% Lead_genes overlap pairwise). This is treated as one redundant cluster per the dedup rule; only the terms below carry gene content and literature distinct from generic proteasome-subunit signal.

## Selected Pathways

### Defective CFTR Causes Cystic Fibrosis R-HSA-5678895
- NES: -2.353, FDR q: 0.000
- Literature: CFTR genotype is a well-established modifier of chronic pancreatitis risk and severity, independent of PRSS1/SPINK1/CTRC status (PMID 30420730, Clin Transl Gastroenterol 2018); CFTR haplotypes are associated with chronic pancreatitis susceptibility (PMID 21520337, Hum Mutat 2011); CFTR is named among the core genetic risk genes in the JAMA chronic pancreatitis clinical review (PMID 31860051). ERAD-mediated degradation of misfolded CFTR (reflected in this gene set) is the direct disease mechanism, not incidental annotation overlap.
- Lead genes (top ~10): PSMD4, PSMB8, PSMA6, PSMC5, UBC, VCP, ERLEC1, PSMD13, PSMA7, PSME2

### Metabolism Of Polyamines R-HSA-351202
- NES: -2.239, FDR q: 0.000046
- Literature: Polyamine homeostasis (ornithine decarboxylase/OAZ1 axis) is disrupted in experimental acute pancreatitis models (PMID 20531247, Pancreas 2010), and abnormal polyamine metabolism in pancreatic epithelial cells aggravates chronic-pancreatitis-associated preneoplastic lesions (PMID 42411476, 2026). Kept as the representative polyamine-axis term (see Dropped: Regulation Of Ornithine Decarboxylase (ODC), near-duplicate).
- Lead genes (top ~10): PSMD4, PSMB8, PSMA6, PSMC5, OAZ1, PSMD13, SMOX, PSMA7, PSME2, PSMD9

### Protein processing in endoplasmic reticulum (KEGG hsa04141)
- NES: -1.630, FDR q: 0.05
- Literature: genuine unfolded-protein-response/ER-stress gene set (HSPA5/BiP, EIF2AK3/PERK, DDIT3/CHOP, ATF4, ATF6B, ERO1B, WFS1, CASP12 all in Lead_genes -- zero PSM/proteasome-subunit overlap, distinct from the ERAD/proteasome supercluster). ER stress and the CHOP-mediated UPR pathway are directly implicated in cerulein-induced acute pancreatitis, with melatonin protecting via ER-stress suppression (PMID 32627032, Mol Med Rep 2020); the UPR is reviewed as an emerging therapeutic target in pancreatitis and PDAC acinar injury specifically (PMID 34774415, Pancreatology 2022).
- Lead genes (top ~10): LMAN2, ERP29, UFD1, DNAJB2, DNAJA1, DNAJB1, STUB1, RPN1, DAD1, P4HB

### Neutrophil Degranulation R-HSA-6798695
- NES: -2.121, FDR q: 0.00004
- Literature: leading-edge is genuine neutrophil-granule/effector content (CD68, HMGB1, ASAH1, SNAP23, RAB10, etc.), not proteasome-subunit signal (9/155 lead genes PSM). Neutrophil extracellular traps (NETs) directly drive pancreatic injury and are amplified by gut microbiota in hypertriglyceridemic pancreatitis (PMID 37794047, Nat Commun 2023); neutrophil-driven tissue injury is a core mechanism across acute inflammatory/pancreatitis pathophysiology (PMID 27848953, Mucosal Immunol 2017).
- Lead genes (top ~10): CAP1, CD68, HMGB1, YPEL5, DBNL, PFKL, B2M, ASAH1, ARHGAP45, SNAP23

### Interferon Gamma Signaling R-HSA-877300
- NES: -2.056, FDR q: 0.00017
- Literature: leading-edge is genuine IFN-gamma pathway content (IFNGR1, IRF7/8/9, STAT1, HLA class I/II; zero PSM overlap). IFN-gamma directly modulates cerulein-induced acute pancreatitis severity by repressing NF-kappaB activation (PMID 17513789, J Immunol 2007); IFN-gamma is among the cytokines whose early course tracks acute pancreatitis severity grade (PMID 33920566, Biomolecules 2021).
- Lead genes (top ~10): B2M, HLA-F, HLA-DRB1, SUMO1, PTPN6, IRF9, IFNGR1, HLA-A, IRF7, IRF8

### Interferon Alpha/Beta Signaling R-HSA-909733
- NES: -2.007, FDR q: 0.00047
- Literature: leading-edge is type-I-IFN-specific (IRF1/2/7/8/9, IFNAR2, MX1, OAS1, TYK2; only 1/26 lead genes PSM). Type I interferon signaling is a core coordinator of tissue inflammation and immunometabolic checkpoints in acute inflammatory disease generally (PMID 39126652, Cell Rep 2024); reviewed as part of the cytokine network in pancreatitis immunopathogenesis (PMID 27848953, Mucosal Immunol 2017).
- Lead genes (top ~10): HLA-F, PSMB8, ISG20, PTPN6, IRF9, HLA-A, IRF7, IRF8, STAT1, HLA-E

### Chemokine signaling pathway (KEGG hsa04062)
- NES: -1.668, FDR q: 0.038
- Literature: leading-edge is genuine chemokine-receptor-signaling content (GRK2/6, GNB1/2, PLCB2/G2, VAV1; zero PSM overlap). Chemokine (MCP-1/CXCL/CCL-family) up-regulation drives leukocyte recruitment and severity in acute pancreatitis, and its suppression is a validated glucocorticoid mechanism (PMID 19818401, Biochim Biophys Acta 2009).
- Lead genes (top ~10): PF4, RASGRP2, GRB2, RAP1B, BAD, MAP2K1, GRK2, GNB2, GRK6, HCK

## Dropped candidates (GSEA-significant, no adequate literature support or redundant)
- Interleukin-12 Signaling R-HSA-9020591 / IL-12 Family Signaling / JAK-STAT after IL-12 Stimulation: NES -2.06 to -2.11 -- despite low PSM overlap, top-10 Lead_genes is dominated by generic housekeeping (HNRNPF, HNRNPA2B1, TALDO1, CAPZA1, GSTO1, AURKAIP1, CDC42) with IL-12-pathway genes (IL12RB1, EBI3, STAT1) only appearing further down the list; leading-edge signal cannot be attributed to IL-12 biology specifically, so excluded as a housekeeping term wearing a disease-sounding label. (IFN-gamma downstream of the IL-12/Th1 axis is separately captured by the kept Interferon Gamma Signaling term, which does have IFN-gamma-specific leading-edge content.)
- NIK To Noncanonical NF-kB Signaling / Dectin-1 Mediated Noncanonical NF-kB / TNFR2 Non-Canonical NF-kB Pathway / FCERI Mediated NF-kB Activation / Interleukin-1 Signaling / Interleukin-1 Family Signaling / Signaling By Interleukins / Cytokine Signaling In Immune System: NES -1.65 to -2.27 -- re-checked under the relaxed rule but all remain 80-87% PSMA/PSMB/PSMC/PSMD-dominated in Lead_genes (same ERAD/proteasome supercluster wearing NF-kB/cytokine pathway labels); leading-edge signal is not attributable to the named immune pathway, so the artifact exclusion still applies.
- ER-Phagosome Pathway R-HSA-1236974: NES -2.125 -- 19/31 lead genes are PSM subunits (61%); still proteasome-supercluster-dominated despite the ER-stress-sounding name, unlike the genuinely UPR-driven "Protein processing in endoplasmic reticulum" term (kept).
- Interleukin-23 Signaling R-HSA-9020933 (NES -1.65, n=5) / Interleukin-2 Signaling R-HSA-9020558 (NES -1.64, n=6): gene sets too small/generic (P4HB, STAT3, TYK2 / PTK2B, STAT5A/B, SYK, LCK, JAK3) to attribute specific IL-23 or T-cell IL-2 biology; no pancreatitis-relevant literature found for these specific narrow sets.
- Proteasome: NES -2.327 -- canonical name for the generic proteasome-subunit supercluster; no pancreatitis-specific citation beyond the mechanistically-grounded CFTR entry (kept) which already carries this gene content.
- Hh Mutants Are Degraded By ERAD R-HSA-5362768 / Hedgehog Ligand Biogenesis R-HSA-5358346: NES -2.36/-2.35 -- >70% Lead_genes overlap with CFTR/Proteasome cluster (generic ERAD machinery); Hedgehog-pancreatitis literature exists (PMID 30001532, Gli2-mediated Hedgehog attenuates acute pancreatitis) but the GSEA gene set itself is proteasome-subunit-dominated, not Hedgehog-pathway-specific, so the signal cannot be attributed to Hedgehog biology.
- Regulation Of Ornithine Decarboxylase (ODC) R-HSA-350562: NES -2.228 -- >90% Lead_genes overlap with Metabolism Of Polyamines (kept); near-duplicate.
- Stabilization Of P53 R-HSA-69541, Autodegradation Of Cdh1 By Cdh1:APC/C, GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2, Vpu/Vif-mediated degradation terms, APC/C:Cdc20-mediated degradation terms, SCF-beta-TrCP Mediated Degradation Of Emi1, CDK-mediated Phosphorylation And Removal Of Cdc6, Degradation Of GLI1/GLI2/GLI3 By Proteasome, Degradation Of DVL, Degradation Of AXIN, ABC Transporter Disorders R-HSA-5619084: NES -2.07 to -2.36 -- all part of the same generic ubiquitin-proteasome/ERAD/cell-cycle-degradation supercluster (PSMA/PSMB/PSMC/PSMD-dominated); no pancreatitis-specific literature found beyond what is already captured by the CFTR entry.
- Non-alcoholic fatty liver disease / Diabetic cardiomyopathy / Complex I Biogenesis R-HSA-6799198: NES -2.15 to -2.28 -- mitochondrial complex I / oxidative phosphorylation gene sets (COX/NDUF/UQCR-dominated); excluded as generic OXPHOS housekeeping signal per the standing exclusion rule.
- mRNA Splicing / mRNA Splicing - Major Pathway: NES ~-2.15 -- generic splicing housekeeping machinery; excluded per standing exclusion rule.
- ECM-receptor interaction (KEGG): NES 2.063, rank 68 -- plausible fibrosis-adjacent biology but only one weak, non-specific PubMed hit found linking this exact term to pancreatitis; insufficient evidence to keep.
