# Lung Cancer Pathway Curation

Source: GSEA normative ranking, `normative__Lung_Cancer__Chen_et_al..csv`, FDR q<0.05 (401 terms total before curation); pan-cancer hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

Note: same non-specific-dominance pattern as the other three phenotypes -- most FDR-significant terms are generic proliferation/protein-degradation/RNA-processing machinery shared near-identically across CRC/Esophagus/Stomach, still excluded. The three positive-NES terms unique to Lung (`Defective GALNT3 Causes HFTC`, `Defective GALNT12 Causes CRCS1`, `Retinoid Cycle In Cones`) were checked and dropped as tissue-mismatched: GALNT3/HFTC is a calcinosis disorder, GALNT12/CRCS1 is a colorectal-cancer-susceptibility gene set, and the retinoid cycle term is retina-specific (RPE65/RDH-family) -- none have a plausible lung-cancer mechanistic link. Below the artifact layer, apoptosis-evasion, innate-immune-sensing, Warburg metabolism, and mitotic-checkpoint hallmarks have clean leading edges and are kept under the relaxed criterion.

## Selected Pathways

### Mismatch repair
- NES: -1.915, FDR q: 0.00023
- Literature: Microsatellite instability (MSI-high) is a rare but recognized molecular subtype in lung adenocarcinoma, associated with distinct immune microenvironment and immunotherapy considerations; comprehensive genomic profiling of GI adenocarcinomas and related pan-cancer MSI surveys establish dMMR/MSI as a cross-tumor actionable biomarker category that extends to lung cancer, though at low prevalence (~1-2%) relative to CRC/gastric. Support is weak-to-moderate given the low base rate in lung cancer specifically.
- Lead genes (top ~10): SSBP1, POLD1, POLD4, RFC1, RFC4, MSH6, RFC2, POLD3, RPA1, RPA2, PCNA, RPA3, POLD2, MLH1

### Dissolution Of Fibrin Clot R-HSA-75205
- NES: -1.910, FDR q: 0.00024
- Literature: SERPINE1 (PAI-1), the dominant leading-edge gene, is a well-documented driver of EMT-mediated metastasis in non-small cell lung cancer via a PAI-1/PIAS3/Stat3/miR-34a feedback loop (PMID 28988111), and cancer-associated coagulopathy/impaired fibrinolysis is a recognized paraneoplastic feature of lung cancer.
- Lead genes: S100A10, SERPINE1, SERPINB8, SERPINB2, ANXA2

### Intrinsic Pathway For Apoptosis R-HSA-109606
- NES: -1.633, FDR q: 0.0156
- Literature: Evasion of intrinsic (mitochondrial) apoptosis via the BCL-2 family is a core cancer hallmark and a clinically targeted vulnerability in NSCLC; Fernald K, Kurokawa M, "Evading apoptosis in cancer," Trends Cell Biol 2013 (PMID 23958396). Leading edge is genuine core apoptosis machinery (BAX, BAK1, CASP9, CYCS, BCL2L1, TP53), no artifact contamination.
- Lead genes (top ~10): DIABLO, BID, MAPK1, CASP9, CASP7, CYCS, BAX, BAK1, BBC3, BCL2L1, AKT1, TP53

### STING Mediated Induction Of Host Immune Responses R-HSA-1834941
- NES: -1.699, FDR q: 0.0073
- Literature: The cGAS-STING cytosolic DNA-sensing pathway shapes anti-tumor innate immunity and immunotherapy response across solid tumors including lung cancer; Kwon J, Bakhoum SF, "The Cytosolic DNA-Sensing cGAS-STING Pathway in Cancer," Cancer Discov 2020 (PMID 31852718).
- Lead genes: MRE11, TBK1, XRCC6, TREX1, STING1, TRIM21

### Glycolysis / Gluconeogenesis
- NES: -1.686, FDR q: 0.0085
- Literature: Aerobic glycolysis (the Warburg effect) is a pan-cancer metabolic hallmark supporting proliferative biomass demand; Vander Heiden MG, Cantley LC, Thompson CB, "Understanding the Warburg Effect: The Metabolic Requirements of Cell Proliferation," Science 2009 (PMID 19460998).
- Lead genes (top ~10): GPI, TPI1, ALDOA, PGK1, DLAT, PDHA1, GAPDH, ENO1, PKM, LDHA, PFKL

### Mitotic Spindle Checkpoint R-HSA-69618
- NES: -1.678, FDR q: 0.0094
- Literature: Chromosomal instability from a weakened mitotic spindle assembly checkpoint (BUB3, MAD1L1, ZW10, BIRC5/survivin) is a well-established driver of aneuploidy and tumor evolution across solid cancers including lung; Bakhoum SF, Compton DA, "Chromosomal instability and cancer: a complex relationship with therapeutic potential," J Clin Invest 2012 (PMID 22269323).
- Lead genes (top ~10): BUB3, CLASP2, PPP2R5D, SKA2, UBE2S, ZW10, PPP2R1A, CKAP5, BIRC5, CENPP, ZWINT, MAD1L1

### TP53 Regulates Transcription Of Genes Involved In G2 Cell Cycle Arrest R-HSA-6804114
- NES: -1.583, FDR q: 0.0262
- Literature: The p53-dependent G2/M checkpoint is a core tumor-suppressor mechanism inactivated in the majority of lung cancers (TP53 is the single most frequently mutated gene in NSCLC); Kastan MB, Bartek J, "Cell-cycle checkpoints and cancer," Nature 2004 (PMID 15549093).
- Lead genes: CDC25C, PRMT1, PCNA, BAX, CCNB1, E2F4, TFDP1, AURKA, TP53

## Dropped candidates (GSEA-significant, no adequate literature support or still artifactual)
- Defective GALNT3 Causes HFTC R-HSA-5083625 / Defective GALNT12 Causes CRCS1 R-HSA-5083636 / Retinoid Cycle In Cones (Daylight Vision) R-HSA-2187335: NES 2.42 / 2.35 / 2.24 -- organ-mismatched gene sets (calcinosis, colorectal-cancer-susceptibility, retina-specific vision cycle respectively); no plausible lung-cancer-specific rationale despite being the only positive-NES significant terms.
- Broad set of top negative-NES terms (Mitochondrial Protein Import, Complex I Biogenesis, Autodegradation Of Cdh1 By Cdh1:APC/C, GSK3B/BTRC-mediated degradation of NFE2L2, AUF1 mRNA destabilization, Vif-mediated degradation of APOBEC3G, Degradation Of AXIN/DVL, Ubiquitin-dependent degradation of Cyclin D): NES up to -2.16 -- generic proliferation/proteostasis machinery, near-identical across all four phenotypes examined in this batch, not lung-specific and not a distinct hallmark.
- PINK1-PRKN Mediated Mitophagy R-HSA-5205685 / Mitophagy R-HSA-5205647: NES -1.78 to -1.84 -- core Parkinson's-disease mitophagy machinery per the neurodegeneration exclusion criterion; not treated as lung-cancer-specific.
- Folding Of Actin By CCT/TriC, Cooperation Of Prefoldin And TriC/CCT: NES -1.93/-1.82 -- generic chaperonin housekeeping, no lung-specific or clean pan-cancer-hallmark citation found beyond ubiquitous proliferative protein-folding demand.
- Cell Cycle R-HSA-1640170: NES -1.643 -- near-duplicate in composition to the Mitotic Spindle Checkpoint and TP53/G2-arrest terms kept above (>50% Lead_genes overlap: BUB3, ANAPC-family, CCND3), and its own leading edge is more proteasome-contaminated (28%) than either kept term.
- Negative Regulation Of MAPK Pathway R-HSA-5675221 / G2/M DNA Damage Checkpoint R-HSA-69473: NES -1.60/-1.53 -- overlapping biology with the MAPK and G2/M-checkpoint terms already represented via other kept pathways in this batch; kept the cleanest, highest-NES representative of each mechanism instead of all near-duplicates.
- Suppression Of Apoptosis R-HSA-9635465: NES -1.525 -- only 1 lead gene (TRIM27), too sparse to support as a distinct pathway beyond "Intrinsic Pathway For Apoptosis" kept above.
