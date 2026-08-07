# Pancreatic Cancer — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Pancreatic_Cancer.txt` (n=72 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; no exceptions were granted in this review. Pathway 7 (Stabilization Of P53) was added manually afterward — it fell just outside the automated review's top-40 candidate window.

## Selected pathways

### 1. Neutrophil Degranulation
n_sig=21/72, size=463, eff=4.88, hist_frac=0%

Neutrophils and neutrophil extracellular traps (NETs) are established drivers of pancreatic ductal adenocarcinoma (PDAC) progression and metastasis, acting through granule-protein release and NET formation in the tumor microenvironment.
- Zhu et al., Cancer Research, 2021 (PMID: 33941611) — TIMP1 secreted by PDAC cells triggers neutrophil NET formation, promoting tumor progression.
- Kajioka et al. et al. (multiple groups), see raw search log — corroborating NET/PDAC hits (33819739 single-cell TME atlas showing neutrophil states in PDAC progression).

### 2. Complement and coagulation cascades
n_sig=10/72, size=85, eff=8.06, hist_frac=0%

Complement activation (lectin and alternative pathways) has a direct, mechanistically demonstrated role in pancreatic oncogenesis and immunosuppressive myeloid recruitment in PDAC, distinct from the SLE-type false-positive pattern (autoantigen exposure) seen elsewhere in this candidate list.
- Aykut et al., Nature, 2019 (PMID: 31578522) — pancreatic mycobiome activates mannose-binding lectin (MBL)/complement to drive oncogenesis in PDAC mouse models.
- Han et al., Cancer Immunology, Immunotherapy, 2025 (PMID: 41417111) — chronic stress promotes PDAC progression via complement C5a-recruited myeloid-derived suppressor cells.

### 3. Regulation Of IGF Transport And Uptake By IGFBPs
n_sig=16/72, size=123, eff=4.72, hist_frac=0%

IGFBP2, the dominant gene in this Reactome pathway, is both a circulating diagnostic/prognostic biomarker and a functional immunosuppressive driver in PDAC.
- Xu et al., Open Medicine (Warsaw), 2024 (PMID: 39221034) — circulating IGFBP2 has prognostic and diagnostic value in pancreatic cancer.
- Li et al., Journal of Personalized Medicine, 2022 (PMID: 36556226) — IGFBP2 drives regulatory T cell differentiation via STAT3/IDO signaling in PDAC.

### 4. Extracellular Matrix Organization
n_sig=15/72, size=287, eff=4.42, hist_frac=0%

Dense, collagen-rich desmoplastic stroma and ECM remodeling are a defining histopathologic and mechanistic hallmark of PDAC, governing drug delivery, invasion, and outcome. Kept in preference to the near-duplicate "ECM-receptor interaction" (KEGG, lower score, same mechanism — see rejected list).
- Provenzano et al., Cancer Cell, 2012 (PMID: 22439937) — enzymatic (hyaluronidase) ablation of stromal barriers improves PDAC treatment response, establishing ECM as a functional driver of disease.
- Sun et al., Nature, 2022 (PMID: 36198801) — collagenolysis-dependent DDR1 signalling dictates pancreatic cancer outcome.

### 5. Platelet Degranulation
n_sig=12/72, size=123, eff=4.96, hist_frac=0%

Platelet activation/degranulation is mechanistically linked to PDAC's well-documented hypercoagulable phenotype and to platelet-tumor cell crosstalk promoting invasion and immune evasion. Kept in preference to the near-duplicate "Response To Elevated Platelet Cytosolic Ca2+" (same platelet-activation mechanism, lower score — see rejected list).
- Chen et al., Frontiers in Oncology, 2022 (PMID: 35494001) — review: challenges and opportunities associated with platelets in pancreatic cancer, covering platelet-tumor crosstalk and thrombosis risk.
- Ünlü & Versteeg / cited via Cancers 2022 (PMID: 35159000) — platelet and cancer-cell interactions modulate cancer-associated thrombosis risk across cancer types, pancreatic cancer highlighted as highest-risk.

### 6. Degradation Of GLI1 By Proteasome
n_sig=13/72, size=59, eff=3.63, hist_frac=0%

This Reactome entry reflects proteasomal turnover of GLI1, the terminal effector of Hedgehog signaling — a pathway with extensive, PDAC-specific mechanistic literature (stromal Hh ligand secretion by tumor cells driving desmoplasia and Gli-dependent transcriptional programs). Kept in preference to the near-duplicate "Degradation Of GLI2 By Proteasome" and "GLI3 Is Processed To GLI3R By Proteasome" entries, which report the same Hedgehog-GLI axis at lower score (see rejected list).
- Singh & Rai, Pharmacology & Therapeutics, 2022 (PMID: 34999181) — targeting hedgehog signaling in pancreatic ductal adenocarcinoma (review of GLI-dependent stromal/tumor mechanisms).
- Skoda et al., Journal of Experimental & Clinical Cancer Research, 2019 (PMID: 31661013) — attenuation of hedgehog/GLI signaling by NT1721 extends survival in pancreatic cancer models.

### 7. Stabilization Of P53
n_sig=11/72, size=56, eff=3.82, hist_frac=0%

Manually added after initial review: scored just outside the top-40 candidate window (rank 60/2063), so it was never presented to the automated literature pass — an omission of review scope, not a rejection on literature or signal-strength grounds. TP53 is one of the most frequently mutated genes in PDAC and defines the poor-prognosis "squamous" molecular subtype, making p53-pathway signal disease-relevant beyond a generic tumor-suppressor readout.
- Bailey P, Chang DK, Nones K et al., Nature, 2016 (PMID: 26909576) — integrated genomic analysis of 456 PDACs; the squamous molecular subtype is enriched for TP53 (and KDM6A) mutations with poor prognosis.

## Considered and rejected

- **Systemic lupus erythematosus** (n_sig=18, size=127, eff=5.31, hist_frac=57.5%). High hist_frac driven purely by histones as SLE autoantigens — the known false-positive pattern explicitly excluded by project rule; no pancreatic-cancer-specific mechanism found.
- **Neutrophil extracellular trap formation** (n_sig=14, size=183, eff=4.95, hist_frac=39.9%). Same biology as selected "Neutrophil Degranulation" (redundant, NET formation is downstream of neutrophil granule/chromatin release); excluded both for high hist_frac (histone citrullination/NET chromatin) and redundancy — lower score than the kept entry.
- **Alcoholism** (n_sig=11, size=180, eff=4.44, hist_frac=40.6%). High hist_frac (chromatin/histone-modification genes in this KEGG pathway); no PDAC-specific mechanistic literature found distinct from general alcohol-pancreatitis epidemiology.
- **Condensation Of Prophase Chromosomes** (n_sig=10, size=42, eff=4.59, hist_frac=69.0%). Histone-dominated by construction (core/linker histones needed for chromosome condensation); no disease-specific rationale beyond generic mitosis.
- **ECM-receptor interaction** (n_sig=12, size=87, eff=3.89, hist_frac=0%). Redundant with selected "Extracellular Matrix Organization" — same integrin/ECM mechanism, lower score.
- **Response To Elevated Platelet Cytosolic Ca2+** (n_sig=12, size=128, eff=4.84, hist_frac=0%). Redundant with selected "Platelet Degranulation" — same platelet-activation mechanism, lower score.
- **Complement Cascade** and **Regulation Of Complement Cascade** (n_sig=8 and 9, hist_frac=0%). Redundant with selected "Complement and coagulation cascades" — same complement mechanism, lower scores.
- **Degradation Of GLI2 By Proteasome** (n_sig=12, score=0.632) and **GLI3 Is Processed To GLI3R By Proteasome** (n_sig=12, score=0.624). Redundant with selected "Degradation Of GLI1 By Proteasome" — same Hedgehog-GLI axis, lower scores.
- **Diabetic cardiomyopathy** (n_sig=14, size=189, eff=4.00, hist_frac=0%). Diabetes-PDAC epidemiological links exist, but this specific KEGG pathway is cardiac-muscle mechanism (myocardial fibrosis/hypertrophy) with no direct pancreatic-cancer literature found.
- **HIV Infection, SARS-CoV Infections, Host Interactions Of HIV Factors, Vpu Mediated Degradation Of CD4, Vif-mediated Degradation Of APOBEC3G, FCERI Mediated NF-kB Activation** — viral-infection/mast-cell Reactome pathways with no pancreatic-cancer-specific mechanistic literature; genes likely recruited via shared innate-immune/ubiquitin-proteasome machinery rather than disease-specific biology.
- **Aspirin ADME** (n_sig=10, size=43, eff=5.05, hist_frac=0%). Drug-metabolism pathway; while aspirin/NSAID chemoprevention in PDAC has epidemiological literature, this Reactome ADME entry itself reflects a xenobiotic-metabolism gene module, not a disease mechanism.
- **Signaling By B Cell Receptor (BCR), TCR Signaling, Downstream TCR Signaling, Post-translational Protein Phosphorylation, GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2** — broad immune-signaling/proteostasis modules without pancreatic-cancer-specific mechanistic papers identified in this review; plausible bystander activation from general antitumor immune response rather than PDAC-specific evidence.
- **Olfactory transduction** (n_sig=11, size=412, eff=4.16, hist_frac=0%). No biological connection to pancreatic cancer; large olfactory receptor gene family likely reflects nonspecific low-level transcriptional noise.
- All [GENERIC]-flagged rows (Metabolism Of RNA, Immune System, Cellular Responses To Stimuli, Cellular Responses To Stress, Innate Immune System, Infectious Disease, Metabolism Of Proteins, mRNA Splicing, mRNA Splicing - Major Pathway, Disease, Spliceosome, Processing Of Capped Intron-Containing Pre-mRNA, Cytokine Signaling In Immune System, Thermogenesis) and high-histone rows other than any kept exceptions were excluded per the standing project rule; no exceptions were granted.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 33941611 — TIMP1 triggers NET formation in pancreatic cancer, Cancer Research 2021 (cited)
- 33819739 — single-cell RNA-seq TME atlas across PDAC progression, EBioMedicine 2021 (considered, not cited — supports neutrophil selection contextually)
- 39025845 — necroptosis/macrophage extracellular traps in PDAC liver metastasis, Nat Commun 2024 (checked, not cited — macrophage not neutrophil ETs)
- 32860704, 39827463, 37794047 — general NET/pancreatic cancer hits (checked, not cited — redundant with 33941611)
- 31578522 — fungal mycobiome activates MBL/complement in pancreatic oncogenesis, Nature 2019 (cited)
- 41417111 — chronic stress promotes PDAC via complement C5a-MDSC recruitment, Cancer Immunol Immunother 2025 (cited)
- 37523607, 38144520, 23614574 — generic bioinformatics/gene-model papers surfaced by broad complement+coagulation query (checked, rejected — not mechanistic complement papers)
- 39221034 — circulating IGFBP2 prognostic/diagnostic value in pancreatic cancer, Open Medicine 2024 (cited)
- 36556226 — IGFBP2 drives Treg differentiation via STAT3/IDO in PDAC, J Pers Med 2022 (cited)
- 29954406 — IGF2BP1 (different gene family, RNA-binding protein not IGFBP) in cancer review (checked, rejected — wrong gene family)
- 22439937 — enzymatic stromal ablation improves PDAC treatment, Cancer Cell 2012 (cited)
- 36198801 — collagenolysis-dependent DDR1 signalling dictates PDAC outcome, Nature 2022 (cited)
- 30366930 — IL1/TGFb shape CAF heterogeneity in PDAC, Cancer Discovery 2019 (considered, not cited — CAF biology adjacent to ECM but not directly ECM-organization pathway)
- 35494001 — review: platelets in pancreatic cancer, Front Oncol 2022 (cited)
- 35159000 — platelet-cancer interactions and thrombosis risk across cancer types, Cancers 2022 (cited)
- 39938515 — extracellular vesicles from lung pro-thrombotic niche drive cancer thrombosis (lung-focused), Cell 2025 (checked, not cited — not pancreas-specific)
- 31399545 — ARF6/AMAP1 promote PDAC invasion and immune evasion, PNAS 2019 (checked, not cited — not platelet-specific)
- 34999181 — targeting hedgehog signaling in PDAC, Pharmacol Ther 2022 (cited)
- 31661013 — NT1721 attenuates hedgehog/GLI signaling, extends survival in pancreatic cancer, J Exp Clin Cancer Res 2019 (cited)
- 29274272 — hedgehog signaling pathway in cancer, general review, Bosn J Basic Med Sci 2018 (considered, not cited — not pancreas-specific)
- 36346366 — CDK4/6 inhibitor combinations in KRAS-mutant pancreatic cancer, Cancer Research 2023 (checked, not cited — unrelated to GLI/Hedgehog)
- 26909576 — Bailey et al., PDAC molecular subtypes, squamous subtype enriched for TP53 mutation, Nature 2016 (cited, pathway 7, manually added)
