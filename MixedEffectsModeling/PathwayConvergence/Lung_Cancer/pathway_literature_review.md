# Lung Cancer — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Lung_Cancer.txt` (n=26 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below (none used this round — no GENERIC or high-histone row survived screening on real mechanistic literature).

## Selected pathways

### 1. Nuclear Events Mediated By NFE2L2
n_sig=19/26, size=78, eff=5.00, hist_frac=0%

NFE2L2 (NRF2) is one of the most recurrently altered pathways in lung cancer: KEAP1/NFE2L2 mutations occur in ~20% of lung adenocarcinoma and even more of squamous carcinoma, driving constitutive NRF2 activation, antioxidant gene induction, and glutamine dependence. This pathway is far more specific than the umbrella "Cell Cycle"/"Metabolism" GENERIC rows and had the best score among three overlapping NRF2-related candidates (see rejected).
- Romero R et al., Nature Medicine, 2017 (PMID: 28967920) — Keap1 loss promotes Kras-driven lung cancer via NRF2 activation and glutaminolysis dependence.
- Best SA et al./related, Cell Metabolism, 2023 (PMID: 36841242) — NRF2 activation induces NADH-reductive stress, a druggable metabolic vulnerability in lung cancer.
- Cancer Cell review, 2022 (PMID: 36270277) — Squamous cell lung cancer landscape review discussing NRF2/KEAP1 axis as a defining driver alteration.

### 2. Mitochondrial Complex I Biogenesis
n_sig=21/26, size=51, eff=4.53, hist_frac=0%

Mitochondrial complex I activity and biogenesis is a recognized metabolic dependency in non-small-cell lung cancer (NSCLC), exploited both as a target for selective cytotoxic compounds and implicated in oxidative-metabolism reprogramming.
- Nagashima R et al. (chemical-genomics target discovery), ACS Chemical Biology, 2020 (PMID: 31874028) — Target discovery of selective NSCLC toxins identifies mitochondrial complex I inhibitors, establishing complex I as an NSCLC-selective metabolic vulnerability.

### 3. GLI3 Is Processed To GLI3R By Proteasome
n_sig=19/26, size=59, eff=4.60, hist_frac=0%

This is a specific step of Hedgehog pathway regulation (proteasomal processing of GLI3 into its repressor form), not a generic proteasome/ubiquitin artifact — GLI processing is the core regulatory node of Hedgehog signaling. Hedgehog-GLI signaling is documented as a growth driver specifically in squamous lung cancer.
- Chen Q et al., Clinical Cancer Research, 2014 (PMID: 24423612) — Hedgehog-GLI signaling inhibition suppresses tumor growth in squamous lung cancer.
- Skoda AM et al., Bosnian J Basic Med Sci, 2018 (PMID: 29274272) — Comprehensive review of Hedgehog signaling in cancer, including GLI processing/activation.
- Jenkins D, Frontiers in Genetics, 2019 (PMID: 31244888) — Non-canonical Hedgehog signaling and GLI transcription factor activation beyond Smoothened.

### 4. RNA transport
n_sig=17/26, size=163, eff=5.38, hist_frac=0%

This Reactome category is dominated by nucleocytoplasmic RNA/protein export machinery (exportin/XPO1-mediated transport). XPO1-dependent nuclear export is a validated, KRAS-mutant-lung-cancer-specific therapeutic vulnerability, with selinexor and related XPO1 inhibitors tested clinically in lung cancer.
- Kim J et al., Nature, 2016 (PMID: 27680702) — XPO1-dependent nuclear export is a druggable vulnerability specifically in KRAS-mutant lung cancer.
- Gupta A et al., Journal of Thoracic Oncology, 2017 (PMID: 28647672) — Therapeutic targeting of nuclear export inhibition in lung cancer.

### 5. Negative Regulation Of NOTCH4 Signaling
n_sig=17/26, size=53, eff=5.11, hist_frac=0%

NOTCH4 itself (as distinct from the broader Notch family) has a lung-adenocarcinoma-specific mechanistic role: a NOTCH4 splice variant sensitizes EGFR-mutant lung adenocarcinomas to EGFR-TKIs via HES1 transcriptional repression, directly implicating NOTCH4 regulatory control in lung adenocarcinoma biology.
- Baumgart SJ et al., Nature Communications, 2023 (PMID: 37268635) — NOTCH4(ΔL12_16) sensitizes lung adenocarcinomas to EGFR-TKIs through transcriptional down-regulation of HES1.
- Yuan X et al., Int J Mol Med, 2020 (PMID: 31894255) — Review of Notch signaling dysregulation and precision medicine relevance across cancers including lung.

### 6. tRNA Processing
n_sig=19/26, size=105, eff=4.88, hist_frac=0%

tRNA processing/modification enzymes are dysregulated in cancer generally, and the ALKBH family of RNA/DNA demethylases specifically has documented prognostic and diagnostic value in non-small-cell lung cancer, supporting a lung-cancer-relevant link beyond a generic housekeeping signal.
- Zhang Y, Pathology Research and Practice, 2022 (PMID: 35180653) — Characterization of prognostic and diagnostic values of ALKBH family members in NSCLC.
- Orellana EA et al., Front Cell Dev Biol, 2022 (PMID: 35721477) — tRNA function and dysregulation in cancer (general mechanistic background).
- Cell Death Discovery, 2024 (PMID: 39019857) — Dysregulation of tRNA methylation in cancer, mechanisms and therapeutic targeting.

## Considered and rejected

- **KEAP1-NFE2L2 Pathway** (n_sig=19, size=100, eff=4.84, hist_frac=0%) and **GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2** (n_sig=19, size=51, eff=5.00, hist_frac=0%) — rejected as redundant with the higher-scoring "Nuclear Events Mediated By NFE2L2" (row 25, score 3.656 vs 3.535 and 3.652); same NRF2 mechanism, same literature support.
- **Degradation Of GLI2 By Proteasome** (n_sig=19, size=59, eff=4.55, hist_frac=0%) — rejected as redundant with "GLI3 Is Processed To GLI3R By Proteasome" (same Hedgehog-GLI mechanism, lower score).
- **Olfactory Signaling Pathway**, **Expression And Translocation Of Olfactory Receptors**, **Olfactory transduction**, **Sensory Perception** (n_sig=16-17, hist_frac=0%) — despite high scores and GENERIC=False, no lung-cancer-specific mechanistic paper was found. General reviews of ectopic olfactory receptor function in non-olfactory tissue (PMID 28338216, 29897292) do not establish a lung-cancer-specific role, and targeted searches (OR51E1/OR2W3 + lung cancer; olfactory receptor + lung adenocarcinoma prognosis) returned no relevant hits. Rejected per the "no Open Targets/database-only, must find a real paper" rule — evidence insufficient.
- **Mitochondrial Protein Import** (n_sig=20, size=65, eff=4.98, hist_frac=0%) — rejected; no PubMed hits for mitochondrial protein import machinery (TOMM/TIMM) specifically in lung cancer.
- **HIV Infection** and **Host Interactions Of HIV Factors** (n_sig=16-17, hist_frac=0%) — rejected as irrelevant to lung cancer mechanism; these Reactome terms capture shared host translation/ubiquitin machinery co-opted by HIV, not lung-cancer-specific biology, and are a likely gene-overlap artifact.
- **DNA Replication / Synthesis Of DNA / S Phase / M Phase / Separation Of Sister Chromatids** and the large APC/C ubiquitin-degradation cluster (Autodegradation Of Cdh1, APC/C:Cdc20-mediated degradations, Regulation Of APC/C Activators, etc., ~15 rows, all n_sig 17-19) — rejected as a single redundant cluster representing generic mitotic/proliferation machinery, mechanistically indistinguishable from the GENERIC-flagged "Cell Cycle" and "Cell Cycle, Mitotic" rows; not lung-cancer-specific beyond "the tumor proliferates."
- **Protein Localization**, **Cellular Response To Chemical Stress**, **Processing Of Capped Intron-Containing Pre-mRNA**, **tRNA Processing**'s near-duplicate DNA-replication-pre-initiation entries — rejected as too broad/non-specific or redundant with selected pathways.
- All [GENERIC]-flagged rows (Metabolism Of RNA, Cellular Responses To Stress, Cellular Responses To Stimuli, Metabolism Of Proteins, Gene Expression (Transcription), RNA Polymerase II Transcription, Cell Cycle, Infectious Disease, Generic Transcription Pathway, Cell Cycle Checkpoints, mRNA Splicing, mRNA Splicing - Major Pathway, Immune System) were excluded per the standing project rule; none were revisited as exceptions since none has a genuinely narrow, disease-specific mechanism distinct from their umbrella scope.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 28967920 — Keap1 loss promotes Kras-driven lung cancer, Nature Medicine 2017 (cited)
- 36270277 — Squamous cell lung cancer landscape review, Cancer Cell 2022 (cited)
- 36841242 — NRF2 activation induces NADH-reductive stress in lung cancer, Cell Metabolism 2023 (cited)
- 40645185, 38877143 — KEAP1/NFE2L2 + lung cancer hits, checked title only, not cited (redundant with above)
- 40239706 — Mitochondrial metabolism sustains DNMT3A-mutant clonal hematopoiesis, Nature 2025 (checked, rejected — not lung cancer)
- 40788065 — Tumor EVs and mitochondrial damage in myocardial I/R injury (checked, rejected — not lung cancer)
- 31874028 — Target discovery of selective NSCLC toxins reveals complex I inhibitors, ACS Chem Biol 2020 (cited)
- 29274272 — Hedgehog signaling pathway in cancer review, Bosn J Basic Med Sci 2018 (cited)
- 31244888 — Non-canonical Hedgehog signaling / GLI activation, Front Genet 2019 (cited)
- 24423612 — Hedgehog-GLI signaling inhibition in squamous lung cancer, Clin Cancer Res 2014 (cited)
- 28948003, 31200829 — Hedgehog/lung cancer hits, checked title only, not cited
- 27680702 — XPO1-dependent nuclear export vulnerability in KRAS-mutant lung cancer, Nature 2016 (cited)
- 28647672 — Therapeutic targeting of nuclear export inhibition in lung cancer, J Thorac Oncol 2017 (cited)
- 41440011, 38902348, 39887933 — XPO1/nuclear export hits, checked title only, not cited
- 31894255 — Precision medicine for Notch signaling dysregulation review, Int J Mol Med 2020 (cited)
- 37268635 — NOTCH4 splice variant sensitizes lung adenocarcinoma to EGFR-TKIs, Nature Communications 2023 (cited)
- 30593175, 38301911, 37961223 — NOTCH4/lung cancer hits, checked title only, not cited
- 28338216 — Ectopic olfactory receptors in non-olfactory tissues review, J Cell Physiol 2018 (considered, not cited — no lung cancer specificity)
- 29897292 — Human olfactory receptors, novel functions outside the nose, Physiol Rev 2018 (considered, not cited — no lung cancer specificity)
- 31068808, 29991799, 38015361, 42001349, 23913758, 21076235 — olfactory receptor / lung cancer targeted search hits, checked, all rejected as not mechanistically relevant
- 35721477 — tRNA function and dysregulation in cancer, Front Cell Dev Biol 2022 (cited)
- 39019857 — Dysregulation of tRNA methylation in cancer, Cell Death Discovery 2024 (cited)
- 33658722 — Expanding world of tRNA modifications and disease relevance, Nat Rev Mol Cell Biol 2021 (checked, background only, not cited)
- 40481562 — ALKBH proteins in human cancer, Eur J Med Res 2025 (checked, background)
- 29144457 — Ubiquitin-dependent signalling for ALKBH-mediated DNA dealkylation repair, Nature 2017 (checked, rejected — not cancer-specific)
- 35180653 — ALKBH family prognostic/diagnostic value in NSCLC, Pathol Res Pract 2022 (cited)
- 40505231, 35980268 — tRNA/lung cancer search hits, checked title only, not cited
- Mitochondrial protein import + TOMM/TIMM + lung cancer search — 0 PubMed hits (pathway rejected for lack of evidence)
