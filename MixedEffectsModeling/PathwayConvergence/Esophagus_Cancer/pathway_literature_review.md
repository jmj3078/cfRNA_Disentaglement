# Esophagus Cancer — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Esophagus_Cancer.txt` (n=25 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; no exceptions were made in this review (no GENERIC or high-histone row had disease-specific literature strong enough to override). Pathway 7 (Transcriptional Regulation By TP53) was added manually afterward — it fell just outside the automated review's top-40 candidate window.

## Selected pathways

### 1. Nuclear Events Mediated By NFE2L2
n_sig=14/25, size=78, eff=5.33, hist_frac=0%

NFE2L2 (NRF2) is one of the most recurrently mutated/activated genes in esophageal squamous cell carcinoma (ESCC), driving an oxidative-stress-response transcriptional program that promotes chemoradiation resistance and is an active drug-target candidate in ESCC. Chose the broader "Nuclear Events Mediated By NFE2L2" Reactome node over the narrower "GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2" node (higher score, 3.16 vs 2.98) because all retrieved papers describe NRF2 stabilization/target-gene activation broadly (KEAP1-dependent or miRNA-mediated), not the specific GSK3B/beta-TrCP degradation route — citing the narrower node would overclaim mechanistic precision the literature doesn't support.
- Ninomiya et al., British Journal of Cancer, 2025 (PMID: 40781161) — oncogenic NFE2L2 mutations in plasma ctDNA and tumor predict chemoradiation response in resectable ESCC.
- Kanamoto et al./review, Annals of the New York Academy of Sciences, 2018 (PMID: 29752726) — NRF2 signaling pathway proposed as a targeted-therapy target in ESCC.
- Ostrowski et al., Molecular Cancer Research, 2017 (PMID: 28760781) — miR-432 stabilizes NRF2 by directly targeting KEAP1, relevant to the KEAP1-NRF2 degradation axis in epithelial cancers.

### 2. Regulation Of RUNX3 Expression And Activity
n_sig=14/25, size=54, eff=5.18, hist_frac=0%

RUNX3 is a well-characterized tumor suppressor that is transcriptionally silenced (largely via promoter hypermethylation) in esophageal squamous cell carcinoma, with loss of RUNX3 expression linked to worse prognosis and more aggressive tumor behavior.
- Xu et al., Medical Oncology, 2014 (PMID: 25391920) — RUNX3 inactivation predicts poor prognosis in ESCC after Ivor-Lewis esophagectomy.
- Asian Pac J Cancer Prev, 2013 (PMID: 24175838) — 5-azacytidine (demethylating agent) restores RUNX3 expression and alters biologic behavior of esophageal carcinoma TE-1 cells, supporting a methylation-silencing mechanism.
- Sano et al., Cancer Science, 2025 (PMID: 39440906) — RUNX3 methylation status associated with FOXP3+/CD8+ ratio and aggressive behavior in esophagogastric junction tumors (adjacent anatomic/tumor context, corroborating).

### 3. SCF(Skp2)-mediated Degradation Of P27/P21
n_sig=14/25, size=59, eff=5.17, hist_frac=0%

The SKP2-p27(CDKN1B) axis is a canonical cell-cycle checkpoint bypass mechanism, and SKP2-driven p27 degradation has been directly demonstrated as an oncogenic driver in ESCC, downstream of upstream tumor-suppressor loss.
- Cao et al., Chinese Journal of Cancer Research, 2021 (PMID: 35125808) — ZNF292 suppresses ESCC cell proliferation through the ZNF292/SKP2/P27 signaling axis, directly implicating SCF(Skp2)-mediated p27 degradation in ESCC.

### 4. Neddylation
n_sig=15/25, size=235, eff=4.77, hist_frac=0%

NEDD8-conjugation (neddylation) of cullin-RING ligases is overexpressed and functionally required for ESCC proliferation/invasion in multiple independent studies, including direct evidence that pharmacologic neddylation inhibition suppresses ESCC-relevant inflammatory gene programs.
- Zhang et al., Cancer Biology & Medicine, 2021 (PMID: 33733647) — NEDD8 is overexpressed and a potential therapeutic target in ESCC.
- Wang et al., International Journal of Molecular Sciences, 2021 (PMID: 33572115) — neddylation inhibition suppresses inflammation-induced MMP9 expression in ESCC.
- Wang et al., Signal Transduction and Targeted Therapy, 2020 (PMID: 32651357) — NEDD8-conjugating enzyme UBC12 is a novel therapeutic target in ESCC.

### 5. Degradation Of AXIN
n_sig=13/25, size=54, eff=5.28, hist_frac=0%

AXIN degradation is the rate-limiting step controlling beta-catenin stability in canonical Wnt signaling; a specific ESCC study shows an upstream kinase driving ESCC progression via suppression of AXIN2, mechanistically tying AXIN turnover to ESCC growth (kept over the more general "Regulation Of Activated PAK-2p34..." type proteasome-degradation entries, which lack ESCC-specific mechanistic papers).
- Wu et al., Current Pharmaceutical Design, 2023 (PMID: 37957865) — JNK2 promotes ESCC progression via inhibiting Axin2, directly linking Axin turnover/regulation to ESCC.

### 6. Keratinization
n_sig=15/25, size=207, eff=5.03, hist_frac=0%

Keratinization/squamous terminal differentiation is central to ESCC biology (ESCC arises from squamous epithelium, and the degree/pattern of keratinizing differentiation is a recognized prognostic histopathological feature), distinct from the GENERIC "Cell Cycle"/"Disease" umbrella terms in this list.
- Aiba et al., Frontiers in Oncology, 2024 (PMID: 39711958) — pathological features of the differentiation (keratinization) landscape in ESCC correlate with prognosis.
- Yamada et al., Acta Histochemica et Cytochemica, 2025 (PMID: 40535470) — CALML5, a squamous-differentiation/keratinization marker, characterized in esophageal and oropharyngeal squamous cell carcinoma.

### 7. Transcriptional Regulation By TP53
n_sig=15/25, size=353, eff=4.56, hist_frac=0%

Manually added after initial review: scored just outside the top-40 candidate window (rank 63/2063, "Stabilization Of P53" also nearby at rank 78), and its statistics are essentially unchanged from the pre-bugfix run (n_sig=15/25, eff=4.65 then vs 15/25, eff=4.56 now) — omission was a review-window artifact, not a literature or signal-strength rejection. TP53 is the single most frequently mutated gene in ESCC, with mutation loads among the highest of any solid tumor in high-incidence regions — a disease-defining, not generic-cancer, signal at this magnitude.
- Abedi-Ardekani B et al., PLoS One, 2011 (PMID: 22216294) — extremely high TP53 mutation load in ESCC from a high-incidence Iranian (Golestan) cohort.

## Considered and rejected

- **GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2** (n_sig=14, size=51, eff=5.64, hist_frac=0%). Higher score than "Nuclear Events Mediated By NFE2L2" but rejected as redundant — same NRF2 axis, and retrieved literature supports the broader NRF2/KEAP1 stabilization mechanism, not this specific GSK3B/beta-TrCP degradation route; keeping both would double-count one biological signal.
- **APC/C:Cdc20 Mediated Degradation Of Securin**, **Autodegradation Of Cdh1 By Cdh1:APC/C**, **APC/C:Cdh1 Mediated Degradation Of Cdc20...**, **CDK-mediated Phosphorylation And Removal Of Cdc6**, **Cdc20:Phospho-APC/C Mediated Degradation Of Cyclin A**, **Mitotic Anaphase**, **Mitotic Metaphase And Anaphase**, **Activation Of APC/C...**, **Regulation Of APC/C Activators...**, **SCF-beta-TrCP Mediated Degradation Of Emi1**, **Ubiquitin-dependent Degradation Of Cyclin D**, **Orc1 Removal From Chromatin**, **Synthesis Of DNA**, **Switching Of Origins To A Post-Replicative State**, **FBXL7 Down-Regulates AURKA...** (n_sig 13-15, size 49-232, eff 4.9-5.8, hist_frac 0%). All are generic APC/C-mitosis / DNA-replication cell-cycle machinery — mechanistically redundant with each other and with the GENERIC "Cell Cycle" node; no ESCC-specific paper distinguishes any single one of these from ordinary proliferation, so all were excluded as non-specific "active biology" rather than disease-specific signal.
- **Complex I Biogenesis** (n_sig=13, size=51, eff=5.83, hist_frac=0%). No ESCC-specific mechanistic paper found linking mitochondrial Complex I biogenesis itself (as opposed to general oxidative-metabolism dysregulation) to esophageal cancer; general OXPHOS/mitochondrial signal is also flagged for discount per project convention on housekeeping pathways.
- **Regulation Of RUNX3 Expression And Activity** — see selected #2 (kept).
- **Olfactory transduction / Olfactory Signaling Pathway / Expression And Translocation Of Olfactory Receptors** (n_sig 16-19, size 368-412, eff 7.3-8.1, hist_frac=0%, top-ranked by score). PubMed search returned only generic/unrelated olfactory-receptor cancer hits (no ESCC-specific mechanistic paper — top results were case reports, oropharyngeal SCC, and organoid work unrelated to olfactory receptor biology per se); very large gene sets composed mostly of olfactory receptor gene family members are a known composition-artifact-prone category in bulk/cfRNA transcriptomics, so excluded absent direct evidence.
- **Vif-mediated Degradation Of APOBEC3G**, **Host Interactions Of HIV Factors**, **HIV Infection**, **Vpu Mediated Degradation Of CD4** (n_sig 13-15). HIV-host-interaction Reactome pathways sharing degradation-machinery genes with cancer pathways; not biologically relevant to esophageal cancer, excluded.
- **Defective CFTR Causes Cystic Fibrosis** (n_sig=14, size=60, eff=5.12, hist_frac=0%). Shares ER-associated degradation machinery genes but no disease-specific link to ESCC found; excluded.
- **Hh Mutants Are Degraded By ERAD / Hh Mutants Abrogate Ligand Secretion / GLI3 Is Processed To GLI3R By Proteasome** (n_sig=14). Hedgehog-pathway ERAD/processing sub-pathways; not searched further given weak a priori link and redundancy with each other — deprioritized in favor of higher-confidence hits above given the ~5-8 pathway budget.
- **NIK To Noncanonical NF-kB Signaling**, **Autodegradation Of E3 Ubiquitin Ligase COP1**, **Regulation Of Activated PAK-2p34 By Proteasome Mediated Degradation** (n_sig=14). Generic proteasome/signaling degradation nodes without ESCC-specific literature retrieved; deprioritized.
- All [GENERIC]-flagged rows (Metabolism Of RNA, Cellular Responses To Stress, Metabolism Of Proteins, Cellular Responses To Stimuli, Gene Expression (Transcription), RNA Polymerase II Transcription, Infectious Disease, mRNA Splicing, mRNA Splicing - Major Pathway, Immune System, Disease, Spliceosome, Generic Transcription Pathway, Cell Cycle, Adaptive Immune System, "Cell Cycle, Mitotic") were excluded per the standing project rule; none had a sufficiently narrow, disease-specific mechanistic paper to justify an override.
- Rows with hist_frac>15% in the reviewed range: none in the top ~40 rows exceeded 15% (max observed was M Phase at 7.7%, "Cell Cycle, Mitotic" at 5.6%), so no histone-based exclusions applied.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 42520899 — tobacco/T cell exhaustion ESCC (checked, not cited — not pathway-specific)
- 42257210 — genomic profiling recurrent/metastatic hypopharyngeal/ESCC (checked, not cited)
- 42092279 — miR130b-IL33-PDL1 axis ESCC (checked, not cited)
- 39440906 — RUNX3-methylated esophagogastric junction tumor FOXP3/CD8 (cited)
- 37192624 — Prox2/Runx3 vagal sensory neurons esophageal motility (checked, rejected — neuronal, not tumor biology)
- 40634392 — basal-like esophageal adenocarcinoma subtype (checked, not cited)
- 39285177, 37204466 — SKP2 inhibitor drug-discovery chemistry papers (checked, rejected — not ESCC-specific biology)
- 35125808 — ZNF292/SKP2/P27 axis in ESCC (cited)
- 41857491 — Axin peptidomimetic beta-catenin inhibitor drug chemistry (checked, rejected — not ESCC-specific)
- 37957865 — JNK2 promotes ESCC via inhibiting Axin2 (cited)
- 30841855 — Wnt/beta-catenin in Barrett's esophagus in vitro model (checked, not cited — precursor lesion not ESCC/EAC tumor)
- 41730487, 39158077, 39111501 — DCN1/NEDD8 E3 ligase structure/inhibitor reviews, general (checked, not cited — not ESCC-specific)
- 33733647 — NEDD8 overexpressed therapeutic target in ESCC (cited)
- 33572115 — neddylation inhibition and MMP9 in ESCC (cited)
- 32651357 — UBC12/NEDD8-conjugating enzyme therapeutic target ESCC (cited)
- 25514805 — genome-wide hypomethylation and tumor-gene hypermethylation ESCC outcome (checked, not cited — genome-wide, not RUNX3-specific)
- 25391920 — RUNX3 inactivation predicts poor prognosis ESCC (cited)
- 24175838 — 5-azacytidine restores RUNX3 in esophageal TE-1 cells (cited)
- 42122150 — pyrimethamine restores KEAP1-mediated degradation of NRF2 mutants in ESCC (checked, considered, not cited in final list — describes KEAP1-dependent route specifically, general NRF2 citations used instead)
- 40781161 — oncogenic NFE2L2 mutations in ctDNA/tumor predict chemoradiation response in ESCC (cited)
- 32619021 — conditional NRF2-activating mutation mouse, upper GI hyperplasia (checked, not cited — mouse model, not ESCC-specific)
- 29752726 — NRF2 signaling pathway as therapeutic target in ESCC (cited)
- 28760781 — miR-432 stabilizes NRF2 via KEAP1 targeting (cited)
- 41965870, 41578601, 38424293, 37005515, 14728592, 4032830 — mitochondrial Complex I / esophageal cancer search results (checked, none ESCC-Complex-I-biogenesis-specific, not cited)
- 40523880, 40331188, 40295071, 34093800, 21938487, 17273804 — olfactory receptor / esophageal cancer search results (checked, none directly mechanistic for ESCC olfactory receptor biology, not cited)
- 22216294 — Abedi-Ardekani et al., extremely high TP53 mutation load in ESCC (cited, pathway 7, manually added)
