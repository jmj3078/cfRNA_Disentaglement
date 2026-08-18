# Pancreatic Cancer Pathway Curation

Source: GSEA normative ranking, `normative__Pancreatic_Cancer__Moore_et_al..csv`, FDR q<0.05; pan-cancer hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

Note: as in the companion Pancreatitis curation (same Moore et al. cohort/covariate structure), the negative-NES side of this ranking is dominated by a large redundant Reactome ubiquitin-proteasome/ERAD/cell-cycle-degradation supercluster (Proteasome, Hh Mutants Are Degraded By ERAD, Stabilization Of P53, p53-Dependent G1 DNA Damage Response, APC/C-mediated degradation terms, GLI processing, Defective CFTR, ABC Transporter Disorders, etc., all PSMA/PSMB/PSMC/PSMD-dominated). These are treated as one redundant, non-PDAC-specific cluster and excluded below. The positive-NES side carries the disease-specific signal (ECM/stroma, coagulation, growth-factor, invasion biology).

## Selected Pathways

### ECM-receptor interaction
- NES: 2.379, FDR q: 0.000
- Literature: Pancreatic ductal adenocarcinoma is defined by an extreme desmoplastic stromal reaction with dense laminin/collagen/integrin-rich extracellular matrix deposited by cancer-associated fibroblasts and pancreatic stellate cells, a hallmark and active therapeutic target of the disease (PMID 35805064, Cancers 2022; PMID 40437741, Cancer Medicine 2025).
- Lead genes (top ~10): HSPG2, LAMB1, VWF, ITGB3, FN1, ITGB4, LAMC1, GP1BB, COL4A1, ITGA8

### Complement and coagulation cascades
- NES: 2.348, FDR q: 0.000
- Literature: Complement activation within the tumor microenvironment shapes anti-tumor immune response in PDAC and is being actively explored as an immunotherapy amplification target (PMID 41159389, Mol Cancer Ther 2026); PDAC is also classically associated with a hypercoagulable, complement/coagulation-cascade-perturbed plasma phenotype detectable systemically.
- Lead genes (top ~10): CFH, VWF, FGA, FGB, KNG1, CFB, C3, C1R, F2R, CFHR1

### Regulation Of IGF Transport And Uptake By IGFBPs R-HSA-381426
- NES: 2.247, FDR q: 0.000
- Literature: Circulating IGFBP3 levels are prospectively associated with pancreatic cancer incidence in a nested case-control cohort (PMID 40973159, Jpn J Clin Oncol 2025), and the IGF-axis/IGFBP system is a documented growth-signaling mechanism in PDAC. Kept as the representative IGF/IGFBP-axis term (see Dropped: Post-translational Protein Phosphorylation, near-duplicate with less specific biological interpretation).
- Lead genes (top ~10): IGFBP5, ALB, KTN1, APOB, LAMB1, FBN1, FGA, FN1, KNG1, SPARCL1

### RHOC GTPase Cycle R-HSA-9013106
- NES: 2.209, FDR q: 0.000187
- Literature: RhoC GTPase (with caveolin-1) directly regulates pancreatic cancer cell migration and invasion (PMID 15969750, Molecular Cancer 2005), and Rho GTPase inactivation via p190 RhoGAP reduces pancreatic cancer cell invasion and metastasis (PMID 16776779, Cancer Science 2006).
- Lead genes (top ~10): SLK, DLC1, CAVIN1, ROCK1, ARHGAP35, VAPB, IQGAP1, ARHGEF28, MACO1, TFRC

### YAP1- And WWTR1 (TAZ)-stimulated Gene Expression R-HSA-2032785
- NES: 1.915, FDR q: 0.022
- Literature: Pan-cancer hallmark (proliferation/organ-size-control evasion) included under the relaxed criterion. Hippo pathway inactivation with YAP1/c-Jun axis activation drives gemcitabine resistance and stemness in PDAC (PMID 37143164, J Exp Clin Cancer Res 2023); YAP1 overexpression enhances aerobic glycolysis in PDAC via EGLN2 suppression (PMID 39647834, J Gene Med 2024). Leading-edge genes are genuine Hippo/YAP-TAZ transcriptional machinery, not a proteasome/housekeeping artifact.
- Lead genes (top ~10): WWTR1, HIPK2, YAP1, TEAD1, TEAD4, CCN2, GATA4, KAT2B

## Dropped candidates (GSEA-significant, no adequate literature support or redundant)
- Post-translational Protein Phosphorylation R-HSA-8957275: NES 2.285 -- near-identical Lead_genes to Regulation Of IGF Transport And Uptake By IGFBPs (kept); vague catch-all Reactome term with no PDAC-specific interpretation beyond the IGFBP entry.
- Primary immunodeficiency (KEGG): NES -2.219 -- lymphocyte-receptor gene set (BTK, CD19, CD3D, ZAP70, JAK3); only weak, non-specific PDAC hits found (iPSC vaccine study, unrelated zinc/metallothionein paper), no citable specific mechanistic link.
- p53-Dependent G1 DNA Damage Response R-HSA-69563: NES -2.237 -- gene content is proteasome-subunit-dominated (PSMD4/PSMC5/PSMC3/PSMD6/PSME1/PSMA7), part of the generic UPS supercluster, not p53-pathway-specific despite the name; TP53 is a well-known PDAC driver but this specific gene set does not carry that signal.
- Proteasome, Hh Mutants Are Degraded By ERAD R-HSA-5362768, Negative Regulation Of NOTCH4 Signaling R-HSA-9604323, NIK To Noncanonical NF-kB Signaling R-HSA-5676590, Vpu/Vif-mediated degradation terms, Dectin-1 Mediated Noncanonical NF-kB Signaling, GSK3B/BTRC-mediated degradation of NFE2L2, Autodegradation Of E3 Ubiquitin Ligase COP1, Defective CFTR Causes Cystic Fibrosis R-HSA-5678895, Hedgehog Ligand Biogenesis R-HSA-5358346, APC/C:Cdc20-mediated degradation terms, Degradation Of DVL/GLI2/AXIN/Beta-Catenin, CDK-mediated Phosphorylation And Removal Of Cdc6, Metabolism Of Polyamines R-HSA-351202, Stabilization Of P53 R-HSA-69541, G1/S DNA Damage Checkpoints, SCF-beta-TrCP Mediated Degradation Of Emi1: NES -2.19 to -2.34 -- all part of the same generic ubiquitin-proteasome/ERAD/cell-cycle-degradation supercluster; not PDAC-specific.
- mRNA Splicing / mRNA Splicing - Major Pathway R-HSA-72163/72172: NES ~-2.21 -- generic splicing housekeeping machinery, excluded per standing exclusion rule.
- Thermogenesis / Complex I Biogenesis R-HSA-6799198: NES -2.20/-2.42 -- oxidative phosphorylation/mitochondrial housekeeping, excluded per standing exclusion rule.
- RHOB GTPase Cycle R-HSA-9013026 (NES 2.140) / RHOA GTPase Cycle R-HSA-8980692 (NES 1.922): >80% Lead_genes overlap with kept RHOC GTPase Cycle (SLK/DLC1/CAVIN1/ROCK1/ARHGAP35/AKAP13/CAV1/ANLN/STARD13/ABR core shared across all three); near-duplicates, RHOC kept as representative Rho-GTPase term.
- DNA Damage/Telomere Stress Induced Senescence R-HSA-2559586 (NES 2.070), Pre-NOTCH Transcription And Translation R-HSA-1912408 (NES 2.054): leading edge dominated by core histone genes (H1-2/H1-3/H1-4/H1-5, H2AC*/H2BC* cluster), same histone-gene-cluster artifact already excluded for SAHF below; not a genuine disease-specific signal despite NOTCH3 presence in the latter.
- Extracellular Matrix Organization R-HSA-1474244 (NES 1.965), Assembly Of Collagen Fibrils R-HSA-2022090 (NES 1.977), Collagen Formation R-HSA-1474290 (NES 1.971), Non-integrin membrane-ECM Interactions R-HSA-3000171 (NES 1.937), Laminin Interactions R-HSA-3000157 (NES 1.892): all high Lead_genes overlap with kept ECM-receptor interaction (same HSPG2/LAMB1/LAMC1/LAMA4/ITGB3/ITGB4/COL4A1/COL1A1/COL1A2 desmoplastic-stroma core); redundant, ECM-receptor interaction kept as representative term.
- RUNX3 Regulates YAP1-mediated Transcription R-HSA-8951671 (NES 1.875): 100% Lead_genes subset of kept YAP1/WWTR1-stimulated gene expression (WWTR1;YAP1;TEAD1;TEAD4;CCN2); near-duplicate.
