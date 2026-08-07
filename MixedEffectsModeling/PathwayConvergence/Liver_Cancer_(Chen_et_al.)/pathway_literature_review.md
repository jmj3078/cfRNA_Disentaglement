# Liver Cancer (Chen et al.) — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Liver_Cancer_(Chen_et_al.).txt` (n=10 patients, small cohort — take recurrence/effect statistics with appropriate caution). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; one GENERIC exception justified explicitly below (no hist_frac exceptions were needed — all rows considered had hist_frac=0).

## Selected pathways

### 1. Complement and coagulation cascades
n_sig=6/10, size=85, eff=5.02, hist_frac=0%

The liver is the primary synthesis site for both complement and coagulation-cascade proteins, and this KEGG pathway is repeatedly and specifically dysregulated in HCC, both as a diagnostic/prognostic gene signature and mechanistically (complement components promoting an immunosuppressive tumor microenvironment). Non-generic, disease-specific, and directly tied to core liver biology.
- Su et al., Heliyon, 2024 (PMID: 39391504) — complement-and-coagulation-cascade gene signature correlates with immune environment and drug sensitivity in HCC.
- Kwilas et al. (review), Frontiers in Oncology, 2020 (PMID: 33718121) — mechanistic review of complement system's role in HCC pathogenesis and therapeutic opportunities.

### 2. Degradation Of Beta-Catenin By Destruction Complex
n_sig=5/10, size=84, eff=5.69, hist_frac=0%

Wnt/beta-catenin signaling, gated by the APC/Axin/GSK3B destruction complex, is one of the most recurrently altered pathways in HCC (CTNNB1 activating mutations in ~20-30% of cases), driving proliferation and a distinct tumor subclass. This Reactome entry captures the specific regulatory mechanism (destruction-complex-mediated degradation), not a generic "signaling" umbrella term.
- Perugorria et al., Journal of Clinical Investigation, 2022 (PMID: 35166233) — comprehensive review of beta-catenin signaling and destruction-complex dysregulation in HCC.
- Ge et al., Cancer Research, 2018 (PMID: 29483096) — hPCL3s promotes HCC metastasis by activating beta-catenin signaling (destruction-complex evasion).

### 3. Regulation Of IGF Transport And Uptake By IGFBPs
n_sig=7/10, size=123, eff=4.69, hist_frac=0%

IGF/IGFBP axis dysregulation is a long-established feature of HCC: reduced IGFBP-3 relative to IGF-I is reported in HCC patients and is linked to hepatocyte proliferation and impaired growth-factor sequestration, distinct from generic "signaling" pathways.
- Aleem et al., Clinical Endocrinology, 2003 (PMID: 14974910) — increased IGF-I:IGFBP-3 ratio in HCC patients.

### 4. GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2
n_sig=6/10, size=51, eff=4.17, hist_frac=0%

This pathway encodes the GSK3B/beta-TrCP-driven degradation arm that keeps NRF2 (NFE2L2) activity in check; loss of this control (via KEAP1/NRF2 mutation or upstream stabilizers) is a recurrent oncogenic mechanism in HCC promoting ferroptosis resistance and survival. Selected over the redundant "Nuclear Events Mediated By NFE2L2" entry (same NRF2 axis, lower score) — see rejected list.
- Sun et al., Hepatology, 2016 (PMID: 26403645) — p62-Keap1-NRF2 pathway activation protects HCC cells against ferroptosis.
- Zhao et al., Nature Communications, 2020 (PMID: 31953436) — TRIM25 promotes HCC survival/growth via the Keap1-Nrf2 pathway.

### 5. Protein processing in endoplasmic reticulum
n_sig=4/10, size=169, eff=5.97, hist_frac=0%

ER stress and the unfolded protein response are mechanistically tied to hepatocyte pathology broadly (steatohepatitis, fibrosis) and specifically exploited/induced in HCC cells, including as a target of cytotoxic agents that trigger paraptosis via ER stress.
- Lebeaupin et al., Journal of Hepatology, 2018 (PMID: 29940269) — ER stress signaling in the pathogenesis of liver disease (mechanistic basis for hepatocyte ER-stress relevance).
- Chen et al., Phytomedicine, 2025 (PMID: 40424981) — Icaritin induces paraptosis in HCC cells via ER stress and mitochondrial dysfunction, targeting BHLHE40.

### 6. Negative Regulation Of NOTCH4 Signaling
n_sig=6/10, size=53, eff=4.00, hist_frac=0%

NOTCH4 specifically (not just generic Notch family) has direct, mechanistically demonstrated roles in HCC invasion and vasculogenic mimicry; loss of its negative regulation is consistent with the aberrant Notch activation reported across HCC subtypes.
- Zhu et al., J Huazhong Univ Sci Technol Med Sci, 2017 (PMID: 29058285) — Notch4 inhibition suppresses invasion and vasculogenic mimicry formation of HCC cells.
- Yin et al., J Cell Mol Med, 2020 (PMID: 33118329) — androgen receptor suppresses vasculogenic mimicry in HCC via circRNA7/miR-7-5p/VE-cadherin/Notch4 signaling.
- Giovannini et al. (review), J Cancer, 2019 (PMID: 31031867) — carcinogenic role of Notch signaling pathway in HCC development (family-level corroboration).

### 7. Regulation Of PTEN Stability And Activity
n_sig=6/10, size=67, eff=3.77, hist_frac=0%

PTEN loss/destabilization is a well-documented tumor-suppressor mechanism in HCC that activates the PI3K/AKT/mTOR axis; this pathway captures the specific regulatory-stability mechanism rather than the broad PI3K/AKT signaling umbrella.
- Lin et al., J Transl Med, 2025 (PMID: 40394639) — AP5Z1 affects HCC growth/autophagy by regulating PTEN ubiquitination and PI3K/Akt/mTOR pathway.
- Wei et al., Cancer Medicine, 2023 (PMID: 35861040) — CFDP1 promotes HCC progression via NEDD4/PTEN/PI3K/AKT signaling.

### 8. Transcriptional Regulation By TP53 [GENERIC override]
n_sig=5/10, size=353, eff=4.53, hist_frac=0%

Flagged GENERIC by the regex (matches "Transcriptional Regulation By X"), but kept as an explicit exception: TP53 is one of the most recurrently mutated genes in HCC (~30-50% depending on etiology, especially aflatoxin/HBV-associated cases per TCGA/CPTAC HCC atlases), and this Reactome entry is a single named TF-target pathway, not a broad umbrella term like "Cell Cycle" or "Immune System" — it is mechanistically narrow (TP53 target-gene transcription) despite the regex match.
- Ally et al. (TCGA Research Network), Cell, 2017 (PMID: 28622513) — comprehensive genomic characterization of HCC, TP53 among most frequently mutated drivers.
- Jiang et al., Cell, 2019 (PMID: 31585088) — integrated proteogenomic characterization of HBV-related HCC, TP53 mutation as a major driver axis.

## Considered and rejected

- **Post-translational Protein Phosphorylation** (n_sig=7, size=106, eff=4.45, hist_frac=0%). Reactome entry is largely composed of IGF2 propeptide-processing biology; too mechanistically vague as titled and redundant with the more specific and better-corroborated IGFBP-transport pathway already selected — dropped as redundant/non-specific.
- **Nuclear Events Mediated By NFE2L2** (n_sig=5, size=78, eff=4.49, hist_frac=0%). Same NRF2/KEAP1 axis as the selected "GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2" (lower score); rejected as redundant, not because it lacks evidence.
- **Cytokine Signaling In Immune System** (n_sig=5, size=693, eff=6.37, hist_frac=0.3%). Broad immune umbrella spanning hundreds of genes across many cytokine families; not disease-specific enough despite non-GENERIC flag — no single mechanistic paper distinguishes it from generic inflammation.
- **Neutrophil Degranulation** (n_sig=5, size=463, eff=5.44, hist_frac=0%). Broad innate-immune effector pathway; while neutrophils are implicated in HCC progression generally, the pathway itself is not HCC-mechanism-specific enough to merit inclusion over more targeted candidates.
- **Signaling By Rho GTPases** (n_sig=5, size=640, eff=5.01, hist_frac=4.4%) and its "Miro GTPases And RHOBTB3" variant (n_sig=5, size=656, eff=4.97, hist_frac=4.3%). Both very large, broad cytoskeletal-signaling umbrellas; redundant with each other and not disease-specific enough without a narrower mechanistic sub-pathway.
- **Degradation Of GLI1 By Proteasome** (n_sig=6, size=59, eff=3.80, hist_frac=0%). Hedgehog/GLI1 signaling has some HCC literature, but this specific narrow degradation-mechanism sub-pathway did not turn up a direct HCC-specific mechanistic paper in the searches performed; deprioritized in favor of higher-confidence hits given the 5-8 pathway budget.
- **NIK To Noncanonical NF-kB Signaling** (n_sig=6, size=58, eff=3.87, hist_frac=0%) and **TNFR2 Non-Canonical NF-kB Pathway** (n_sig=5, size=100, eff=4.57, hist_frac=0%). Redundant with each other (same noncanonical NF-kB axis); NF-kB is broadly implicated in HCC inflammation but neither narrow sub-pathway was corroborated with a specific mechanistic paper in this review round — deprioritized.
- **FBXL7 Down-Regulates AURKA During Mitotic Entry And In Early Mitosis** (n_sig=5, size=54, eff=4.50, hist_frac=0%). Extremely narrow single-gene-pair mechanism; AURKA is implicated in HCC generally but no FBXL7-AURKA-specific HCC paper was found — too speculative to include.
- **Interleukin-1 Signaling** / **Interleukin-1 Family Signaling** / **Signaling By Interleukins** / **Downstream TCR Signaling** / **T cell receptor signaling pathway** / **TCR Signaling** / **Fc Epsilon Receptor (FCERI) Signaling** — generic immune/cytokine signaling umbrellas without HCC-specific mechanistic corroboration attempted in this round; deprioritized given the selection budget.
- **Shigellosis**, **Salmonella infection**, **HIV Infection**, **Host Interactions Of HIV Factors**, **Yersinia infection** — infection-pathway gene-set artifacts (large host-response gene overlap with generic stress/immune biology), not liver-cancer-specific.
- **S Phase, Separation Of Sister Chromatids, Autodegradation Of Cdh1 By Cdh1:APC/C, Mitotic Metaphase And Anaphase, Mitotic Anaphase, Synthesis Of DNA, G1/S Transition, Neddylation** — generic mitotic-machinery/cell-cycle-mechanics pathways; individually non-specific to HCC beyond "cancers proliferate," not pursued for dedicated literature search given cell-cycle pathways are already broadly captured by GENERIC-flagged entries.
- **SLC-mediated Transmembrane Transport**, **Endocytosis**, **Thermogenesis** — broad transport/metabolic housekeeping pathways, no disease-specific mechanism identified.
- **Transcriptional Regulation By RUNX3** (GENERIC=True). RUNX3 is a known HCC tumor suppressor, but the pathway term itself remained too broad ("Transcriptional Regulation By X" umbrella) and no additional narrow mechanistic distinction was found to justify an override beyond what TP53 already demonstrates; not overridden.
- All other [GENERIC]-flagged rows (Metabolism Of RNA, Immune System, Infectious Disease, Gene Expression (Transcription), RNA Polymerase II Transcription, Generic Transcription Pathway, Innate Immune System, mRNA Splicing, mRNA Splicing - Major Pathway, Cellular Responses To Stress, Cellular Responses To Stimuli, Adaptive Immune System, Spliceosome, Cell Cycle, Cell Cycle Checkpoints, Developmental Biology) were excluded per the standing project rule with no override justification found.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 39391504 — complement/coagulation gene signature in HCC (cited)
- 33718121 — complement system review in HCC (cited)
- 33911900 — complement/coagulation-related, checked, not cited (redundant with 39391504)
- 37116239 — complement/coagulation-related, checked, not cited (redundant)
- 39011654 — complement/coagulation-related, checked, not cited (redundant)
- 35166233 — beta-catenin signaling in HCC review (cited)
- 29483096 — hPCL3s activates beta-catenin in HCC metastasis (cited)
- 41550750 — beta-catenin/CTNNB1 HCC, checked, not cited (redundant, non-indexed preprint-like ID)
- 26474915 — beta-catenin/CTNNB1 HCC, checked, not cited (redundant)
- 25430888 — beta-catenin/CTNNB1 HCC, checked, not cited (redundant)
- 39471694 — IGFBP/IGF HCC, checked, not cited
- 39552593 — IGFBP/IGF HCC, checked, not cited
- 25941431 — portal vein thrombosis (off-topic, rejected — not IGF-pathway relevant despite keyword match)
- 14974910 — IGF-I:IGFBP-3 ratio in HCC (cited)
- 26722313 — IGFBP/IGF HCC, checked, not cited
- 26403645 — p62-Keap1-NRF2/ferroptosis in HCC (cited)
- 31953436 — TRIM25/Keap1-Nrf2 in HCC (cited)
- 40784043 — KEAP1/NRF2 HCC, checked, not cited (redundant)
- 24011591 — KEAP1/NRF2 HCC, checked, not cited (redundant)
- 41422284 — KEAP1/NRF2 HCC, checked, not cited (redundant)
- 29940269 — ER stress signaling in liver disease pathogenesis (cited)
- 40424981 — Icaritin/ER stress paraptosis in HCC (cited)
- 35203283 — ER stress/UPR HCC, checked, not cited
- 41639449 — ER stress/UPR HCC, checked, not cited
- 38981667 — ER stress/UPR HCC, checked, not cited
- 32976798 — endothelial reprogramming/macrophages in HCC, checked, not cited (not Notch-specific enough)
- 39766289 — Notch/HCC, checked, not cited
- 34970275 — macrophage polarization in liver disease, checked, not cited (off-topic for Notch)
- 40592346 — Notch/HCC, checked, not cited
- 33579428 — Notch/HCC, checked, not cited
- 29058285 — Notch4 inhibition suppresses HCC invasion/vasculogenic mimicry (cited)
- 33118329 — AR/circRNA7/Notch4 vasculogenic mimicry in HCC (cited)
- 17696940 — Notch4/HCC, checked, not cited
- 41592600 — Notch4/HCC, checked, not cited
- 23742774 — Notch4/HCC, checked, not cited
- 32044315 — NASH fibrosis mechanisms, checked, not cited (off-topic, not HCC-Notch specific)
- 31031867 — carcinogenic role of Notch signaling in HCC (review, cited)
- 40394639 — AP5Z1/PTEN ubiquitination/PI3K-Akt-mTOR in HCC (cited)
- 35861040 — CFDP1/NEDD4/PTEN/PI3K/AKT in HCC (cited)
- 31657972 — PTEN/PI3K/AKT HCC, checked, not cited
- 41456492 — PTEN/PI3K/AKT HCC, checked, not cited
- 37083064 — PTEN/PI3K/AKT HCC, checked, not cited
- 31585088 — proteogenomic characterization of HBV-related HCC, TP53 driver (cited)
- 28622513 — TCGA comprehensive genomic characterization of HCC, TP53 driver (cited)
- 38030304 — TP53/HCC, checked, not cited
- 33992698 — TP53/HCC, checked, not cited
- 26099527 — TP53/HCC, checked, not cited
