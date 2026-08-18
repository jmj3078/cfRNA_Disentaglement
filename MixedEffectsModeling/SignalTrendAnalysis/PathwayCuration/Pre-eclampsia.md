# Pre-eclampsia Pathway Curation

Source: GSEA normative ranking, `normative__Pre-eclampsia__Moufarrej_et_al..csv`, FDR q<0.05; inflammatory-disease hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

## Selected Pathways

### Response To Elevated Platelet Cytosolic Ca2+ R-HSA-76005
- NES: -2.353, FDR q: 0.000
- Literature: Platelet activation and procoagulant membrane dynamics (elevated cytosolic Ca2+-driven degranulation) are a documented, disease-severity-correlated feature of preeclampsia, proposed as circulating biomarkers of the disease (PMID 36923708, Res Pract Thromb Haemost 2023). Kept as the representative platelet term (see Dropped: Platelet Degranulation, Platelet Activation/Signaling/Aggregation, near-duplicates).
- Lead genes (top ~10): PRKCB, F13A1, SPARC, TGFB1, MMRN1, CLU, EGF, SYTL4, PF4, THBS1

### VEGFA-VEGFR2 Pathway R-HSA-4420097
- NES: -1.858, FDR q: 0.029
- Literature: Circulating anti-angiogenic sFlt1 antagonizing VEGF/PlGF signaling is the central, mechanistically causal pathway in preeclampsia, established by Levine et al. (PMID 14764923, NEJM 2004) and foundational to the disease's diagnostic and therapeutic framework. Kept as the representative VEGF-axis term (see Dropped: Signaling By VEGF, near-duplicate).
- Lead genes (top ~10): PRKCB, ABI2, ITGB3, ITPR1, MAPKAPK2, CTNNA1, NCK2, NCKAP1, ITPR2, AKT3

### DNA Methylation R-HSA-5334118
- NES: 2.184, FDR q: 0.0028
- Literature: DNMT3A downregulation and consequent DNA-methylation-independent induction of TGF-beta receptor I is mechanistically implicated in early-onset severe preeclampsia (PMID 32794622, FASEB J 2020); placental/trophoblast DNA methylation dysregulation is an actively studied preeclampsia mechanism, and cell-free methylation signals are of direct relevance to the cfRNA/liquid-biopsy framing of this study. Kept as the representative DNA-methylation-machinery term (see Dropped: PRC2 Methylates Histones And DNA, SUMOylation Of DNA Methylation Proteins, near-duplicates).
- Lead genes (top ~10): H3-3B, H2AJ, H2BC9, DNMT1, H2BC15, H2BC14, DNMT3A, H2AC19, H2BC11, H2BC21

### Syndecan Interactions R-HSA-3000170
- NES: -2.102, FDR q: 0.0015
- Literature: Endothelial glycocalyx shedding is a documented preeclampsia mechanism; serum soluble Flt1 correlates inversely with endothelial glycocalyx components including syndecan-1 in early- and late-onset preeclampsia (PMID 34238103, J Matern Fetal Neonatal Med 2022), and endothelial glycocalyx injury markers track preeclampsia severity (PMID 39840434, Hypertension 2025).
- Lead genes (top ~10): TGFB1, ITGB1, ITGB5, THBS1, ITGB3, ACTN1, PRKCA

### Signaling By Hippo R-HSA-2028269
- NES: -2.011, FDR q: 0.0099
- Literature: Hippo pathway effector YAP is decreased in preeclamptic placenta and regulates trophoblast invasion/apoptosis (PMID 29303055, Reprod Sci 2018); miR-21 modulates Hippo signaling via PP2A-Bbeta interference to inhibit trophoblast invasion and cause preeclampsia in a rat model (PMID 36250210, Mol Ther Nucleic Acids 2022).
- Lead genes (top ~10): MOB1A, STK24, SAV1, TJP2, CASP3, WWTR1, MOB1B, STK4, TJP1

### Metal Sequestration By Antimicrobial Proteins R-HSA-6799990
- NES: 1.964, FDR q: 0.0361
- Literature: Leading-edge genes (S100A8/S100A9 = calprotectin, LCN2 = lipocalin-2/NGAL, LTF = lactoferrin) are the canonical neutrophil-granule antimicrobial/acute-phase proteins, genuinely matching the pathway name. S100A8/A9 are specifically implicated in preeclampsia pathophysiology, with elevated calprotectin tracking disease severity (PMID 41155408, Int J Mol Sci 2025, "Exploring the Relevance of S100A8 and S100A9 Proteins in Preeclampsia: A Narrative Review"; PMID 41543095, Clin Lab 2026, calprotectin as a clinical biomarker). Kept under the relaxed neutrophil-activation/acute-phase-response criterion.
- Lead genes: LCN2, S100A8, S100A9, LTF

## Dropped candidates (GSEA-significant, no adequate literature support or redundant)
- Platelet Degranulation R-HSA-114608: NES -2.347 -- near-identical Lead_genes to Response To Elevated Platelet Cytosolic Ca2+ (kept); near-duplicate.
- Platelet Activation, Signaling And Aggregation R-HSA-76002: NES -1.992 -- superset/near-duplicate of the kept platelet term, same top Lead_genes.
- Signaling By VEGF R-HSA-194138: NES -1.853 -- near-identical Lead_genes to VEGFA-VEGFR2 Pathway (kept); near-duplicate.
- PRC2 Methylates Histones And DNA R-HSA-212300: NES 2.247 -- ~70% Lead_genes overlap with DNA Methylation (kept); near-duplicate.
- SUMOylation Of DNA Methylation Proteins R-HSA-4655427: NES 2.204 -- shares DNMT1/DNMT3A/DNMT3B core with DNA Methylation (kept); near-duplicate, and SUMOylation-specific preeclampsia literature not found.
- Defective Pyroptosis R-HSA-9710421: NES 2.141 -- despite the name, gene content (EZH2, H3-3B, H2AJ, SUZ12, DNMT1, DNMT3A) is dominated by the same PRC2/DNA-methylation machinery as DNA Methylation (kept), not pyroptosis/inflammasome genes; misleading annotation, no pyroptosis-specific signal.
- Transcriptional Regulation Of Granulopoiesis R-HSA-9616222: NES 2.142 -- no preeclampsia-specific literature found for this exact granulopoiesis/RUNX1-CSF3R-SPI1 gene set despite documented maternal neutrophil activation in preeclampsia generally.
- Smooth Muscle Contraction R-HSA-445355: NES -1.900 -- plausible relevance to gestational hypertension pathophysiology, but only sparse, non-specific PubMed hits found for this exact pathway in preeclampsia; insufficient evidence to keep.
- Complex I Biogenesis R-HSA-6799198: NES 2.306 -- oxidative phosphorylation/mitochondrial housekeeping, excluded per standing exclusion rule.
- Signaling By RAF1 Mutants R-HSA-9656223 / Paradoxical Activation Of RAF Signaling By Kinase Inactive BRAF R-HSA-6802955 / Signaling By High-Kinase Activity BRAF Mutants R-HSA-6802948: NES -1.81 to -2.08 -- generic oncogenic RAS-RAF-MAPK driver gene sets, not preeclampsia-specific.
- Pancreatic cancer / Renal cell carcinoma / Hepatitis B / Glioma (KEGG cross-disease terms): NES -1.85 to -1.95 -- cross-disease KEGG signaling modules picking up shared generic pathway genes, not preeclampsia-specific evidence.
- Interferon Gamma Signaling R-HSA-877300 / Interferon Alpha/Beta Signaling R-HSA-909733 / Chemokine signaling pathway (KEGG) / Neutrophil Degranulation R-HSA-6798695 / Neutrophil extracellular trap formation (KEGG): not FDR-significant in this cohort (FDR 0.20-0.46) -- checked per the relaxed-criterion sweep since these were real hits in the sibling Pancreatitis re-screen, but do not clear the FDR<0.05 bar here.
- CLEC7A (Dectin-1) Induces NFAT Activation R-HSA-5607763 (NES -1.987, FDR 0.0097) / Calcineurin Activates NFAT R-HSA-2025928 (NES -1.867, FDR 0.0259): leading-edge is generic calcineurin/NFAT calcium-signaling core (NFATC1/2/3, CALM1, PPP3CB) shared near-identically between both terms; no CLEC7A, SYK, or CARD9 (Dectin-1-specific) genes present -- mislabeled generic signaling, not genuine Dectin-1/innate pattern-recognition activation; artifact-exclusion applies.
- Interleukin-35 Signaling R-HSA-8984722: NES -1.903, FDR 0.0204 -- leading edge is generic JAK-STAT core (IL6ST, JAK1, JAK2, STAT1, STAT4) with no EBI3/IL12A/IL27RA (IL-35-specific subunits); does not genuinely represent IL-35 signaling despite the FDR-significant hit; artifact-exclusion applies.
- AGE-RAGE signaling pathway in diabetic complications (KEGG): NES -1.816, FDR 0.0405 -- leading edge is the same generic RAS-RAF-MAPK-PI3K-AKT-JAK-STAT cascade recurring across the dropped cross-disease KEGG terms (no AGER/RAGE receptor or NOX/oxidative-stress-specific genes); not a genuine AGE-RAGE/oxidative-stress signal.
