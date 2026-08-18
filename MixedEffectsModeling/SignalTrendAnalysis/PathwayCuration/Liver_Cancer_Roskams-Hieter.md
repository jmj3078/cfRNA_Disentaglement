# Liver Cancer (Roskams-Hieter B et al.) Pathway Curation

Source: GSEA normative ranking, `normative__Liver_Cancer__Roskams-Hieter_B_et_al..csv`, FDR q<0.05 (17 terms total before curation); pan-cancer hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

## Selected Pathways

### Signaling By Hippo R-HSA-2028269
- NES: 2.068, FDR q: 0.002
- Literature: Hippo pathway inactivation and downstream YAP1/TAZ hyperactivation is a well-established HCC driver mechanism, including in HBV-associated HCC (lncRNA RP11-40C6.2 attenuates YAP1 ubiquitylation to inactivate Hippo signaling, PMID 36643034).
- Lead genes (top ~10): WWTR1, TJP1, LATS1, YAP1, MOB1B, AMOT, AMOTL1, AMOTL2

### Adherens Junction
- NES: 1.936, FDR q: 0.018
- Literature: HBV promotes beta-catenin signaling and disassembly of adherens junctions in a Src-kinase-dependent manner, linking junction breakdown to EMT/invasion in HCC (PMID 30338037).
- Lead genes (top ~10): PTPRB, TJP1, AFDN, CTNND1, RAC2, SNAI2, IQGAP1, CREBBP, WASL, SORBS1

### RAC1 GTPase Cycle R-HSA-9013149
- NES: 1.924, FDR q: 0.016
- Literature: RAC1 signaling drives HCC cell mitochondrial transfer and hypoxic adaptation via HMGB1-RHOT1-RAC1 axis (PMID 38378644); RAC1/Rho-family GTPases are broadly implicated in HCC invasion and metastasis. Kept as the representative Rho-GTPase term (see Dropped: RHOC/CDC42 near-duplicates).
- Lead genes (top ~10): ARHGAP29, ARHGAP31, PLEKHG1, FERMT2, SWAP70, PREX2, DOCK6, ARHGAP23, DLC1, SOS1

### RHOB GTPase Cycle R-HSA-9013026
- NES: 1.921, FDR q: 0.012
- Literature: Kept distinct from RAC1 (Lead_genes overlap ~35%, below the 50% dedup threshold, and RHOB has a distinct literature role as a stress/hypoxia-responsive Rho GTPase implicated in tumor suppression vs progression duality in hepatocarcinogenesis). RAC1 entry above documents the general Rho-GTPase-HCC link (PMID 38378644).
- Lead genes (top ~10): CAVIN1, DLC1, AKAP13, ECT2, SLK, ARHGAP35, STARD13, ABR, CAV1, ROCK1

### Signaling By VEGF R-HSA-194138
- NES: 1.889, FDR q: 0.013
- Literature: VEGF-driven angiogenesis is a core HCC therapeutic target; molecular correlates of response/resistance to the VEGF-targeting combination atezolizumab+bevacizumab in advanced HCC directly implicate this pathway (PMID 35739268, Nature Medicine 2022).
- Lead genes (top ~10): FLT1, HSP90AA1, KDR, ITGB3, NRP1, VEGFB, ARHGEF7, CAV1, WASF2, WASF1

### Oncogene Induced Senescence R-HSA-2559585
- NES: 1.914, FDR q: 0.011
- Literature: p53/RB1 pathway status determines HCC therapeutic response (CDK4/6 inhibitor combination trial stratified by RB1 status, PMID 33931882), consistent with oncogene-induced senescence circuitry (TP53, RB1, CDKN2A/B, E2F1-3) being disease-relevant in HCC.
- Lead genes (top ~10): ETS2, TFDP1, CDK6, ETS1, AGO4, ETF1, TFDP2, CDKN2A, ID1, TP53

### Transcriptional Activity Of SMAD2/SMAD3:SMAD4 Heterotrimer R-HSA-2173793
- NES: 1.829, FDR q: 0.033
- Literature: SMAD2/3/4-mediated TGF-beta signaling is a canonical driver of HCC progression and metastasis (Smad3 roles in carcinogenesis, PMID 17725494; beta-catenin/TCF-4-LINC01278-miR-1258-Smad2/3 axis promotes HCC metastasis, PMID 32372060).
- Lead genes (top ~10): WWTR1, TFDP1, EP300, SKIL, SMAD3, TFDP2, CCNK, UBE2D1, SMURF2, PARP1

### Binding And Uptake Of Ligands By Scavenger Receptors R-HSA-2173782
- NES: 1.919, FDR q: 0.010
- Literature: Hepatocyte scavenger-receptor-mediated lipoprotein uptake (SR-BI, LDLR family) is a documented liver-specific process relevant to lipid handling in hepatoma cells (PMID 17905649) and CD36-mediated lipid/iron handling is implicated in early-stage HCC immune dysfunction (PMID 40037690). Reflects liver-specific secretome/uptake biology plausibly detectable in cfRNA.
- Lead genes (top ~10): APOB, ALB, DLC1, HSP90AA1, HBA1, APOE, COL4A1, CALR, MASP1, HSPH1

### Complement And Coagulation Cascades
- NES: 1.799, FDR q: 0.042
- Literature: Complement/coagulation factor dysregulation is a documented plasma-proteome signature of cirrhosis-to-HCC progression, identified from plasma-derived extracellular vesicles in a proteomic biomarker study (PMID 39011654), supporting direct detectability in blood-derived cfRNA.
- Lead genes (top ~10): FGB, KNG1, F2R, FGA, A2M, FGG, PLG, CFB, PROCR, VTN

### RHOU GTPase Cycle R-HSA-9013420
- NES: 1.812, FDR q: 0.042
- Literature: Re-screened under relaxed pan-cancer-hallmark criterion (previously dropped for lacking disease-specific citation). RhoU/AKT1-driven EMT is directly implicated in HCC cell migration and tumor growth, suppressed by AnnexinA6 SUMOylation (PMID 38566133, Cell Commun Signal 2024) -- an HCC-specific mechanism, not merely generic Rho-GTPase biology. Leading-edge genes are genuine Rho-regulatory/cytoskeletal machinery (ARHGAP31, PEAK1, IQGAP1, ARHGEF6/7, PIK3R1), not a housekeeping artifact.
- Lead genes (top ~10): ARHGAP31, PEAK1, DST, MYO6, SPTBN1, SPTAN1, ITSN2, IQGAP1, SRGAP2, ARHGEF7

### Prostate cancer (KEGG)
- NES: 1.808, FDR q: 0.039
- Literature: Included under the relaxed pan-cancer-hallmark criterion as the canonical proliferation/cell-cycle-checkpoint oncogenic signaling core (PI3K-RAS-RAF-MEK-ERK, RB1/E2F, TP53, PTEN, CTNNB1), despite the misleading KEGG label. RAS/MEK/ERK signalling is a documented driver of primary liver cancer differentiation and progression (PMID 40355258, Gut 2025); PI3K/AKT and Ras/Raf/MEK/ERK pathway inhibition induces cell-cycle arrest in HCC cells (PMID 27654866, BMC Cancer 2016). Leading-edge genes are genuine core oncogenic-signaling components (TP53, RB1, PTEN, CTNNB1, KRAS, NRAS, HRAS, BRAF, PIK3CA, MTOR, CCND1, CCNE1, E2F1-3), not a proteasome/housekeeping artifact.
- Lead genes (top ~10): ZEB1, CCND1, ERG, HSP90AB1, SOS1, HSP90AA1, IGF1, CREBBP, TCF7L2, EP300

## Dropped candidates (GSEA-significant, no adequate literature support or redundant)
- Insulin Receptor Recycling R-HSA-77387: NES -2.209 -- gene set is dominated by generic vacuolar-ATPase subunits (ATP6V*), not HCC-specific; no citable disease-specific literature link found.
- RHOC GTPase Cycle R-HSA-9013106: NES 1.909 -- >80% Lead_genes overlap with RHOB GTPase Cycle (kept); near-duplicate.
- CDC42 GTPase Cycle R-HSA-9013148: NES 1.892 -- >60% Lead_genes overlap with RAC1 GTPase Cycle (kept); near-duplicate.
- Formation Of Senescence-Associated Heterochromatin Foci (SAHF) R-HSA-2559584: NES 1.879 -- largely a histone-gene-cluster artifact (H1-2/H1-3/H1-4/H1-5/H1-0) rather than a specific senescence signal; not a genuine pathway signal even under the relaxed rule (fails the leading-edge-composition check), and no HCC-specific citation found beyond generic senescence biology already covered by Oncogene Induced Senescence.
- Common Pathway Of Fibrin Clot Formation R-HSA-140875: NES 1.838 -- coagulation-in-TME is a includable pan-cancer hallmark theme in principle, but Lead_genes (FGB/F2R/FGA/FGG core) are highly redundant with kept Complement and Coagulation Cascades; kept term retained as representative, this one dropped as duplicate rather than for lack of relevance.
- Nephrin Family Interactions R-HSA-373753: NES 1.809 -- kidney podocyte-specific pathway (nephrin/CD2AP), no biological rationale or literature link to liver cancer; not a pan-cancer hallmark theme.
