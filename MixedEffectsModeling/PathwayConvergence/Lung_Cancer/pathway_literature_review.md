# Lung Cancer — Pathway Literature Review

Cohort n=26. Candidate list dominated by generic umbrella terms (cell cycle,
transcription/splicing machinery, olfactory receptors, immune system) already
[GENERIC]-flagged — excluded per rule. n_sig sits at 17-22/26 for nearly
every remaining candidate and eff clusters tightly at 4.5-5.5, which is a red
flag: in a 26-patient cohort this flat, narrow-range significance profile
across ~50 pathways (most of them cell-cycle/APC-C degradation sub-modules
that are mechanistically near-duplicates of each other) looks like a shared
proliferation/turnover signal common to most of this cancer cohort rather
than pathway-specific discrimination. Treat the specific pathways selected
below as sitting on top of that broad, less specific baseline.

## Selected pathways

### KEAP1-NFE2L2 Pathway
n_sig=19/26, size=100, eff=4.91, hist_frac=0%
KEAP1/NFE2L2(NRF2) pathway mutations are one of the most recurrent,
lung-cancer-enriched oncogenic drivers, especially in smoking-associated
lung squamous and adenocarcinoma, distinguishing it from a generic stress-response
signature.
- Kan et al., PLoS Medicine, 2006 (PMID: 17020408) — dysfunctional KEAP1-NRF2 interaction found specifically in non-small-cell lung cancer.
- Hellyer et al., Clin Cancer Res, 2020 (PMID: 31548347) — KEAP1/NFE2L2 mutations shape chemotherapeutic response in NSCLC.
- Ricciuti et al., J Thorac Oncol, 2019 (PMID: 31323387) — KEAP1-NFE2L2 pathway mutations define a molecularly distinct, rapidly progressing lung adenocarcinoma subset.
- Frank et al., Clin Cancer Res, 2018 (PMID: 29615460) — clinical/pathological characterization of KEAP1- and NFE2L2-mutated NSCLC.
- Comprehensive genomic characterization of squamous cell lung cancers, Nature, 2012 (PMID: 22960745) — TCGA landmark study; KEAP1/NFE2L2 pathway among the top recurrently altered pathways in lung squamous carcinoma.

### Nuclear Events Mediated By NFE2L2
n_sig=19/26, size=78, eff=5.08, hist_frac=0%
Same NRF2 transcriptional-output axis as above (downstream target-gene
activation), supported by the same lung-cancer-specific KEAP1-NFE2L2
literature.

### GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2
n_sig=19/26, size=51, eff=5.07, hist_frac=0%
Same NRF2 regulatory axis (proteasomal turnover arm); included as
corroborating sub-module of the KEAP1-NFE2L2 signal above, not as an
independently verified pathway.

### NOTCH4 signaling (Negative Regulation Of NOTCH4 Signaling)
n_sig=18/26, size=53, eff=5.08, hist_frac=0%
NOTCH4 has a specific, mechanistically characterized role in lung
adenocarcinoma (EGFR-TKI sensitization, vasculogenic mimicry/metastasis in
NSCLC), distinct from the pan-cancer relevance of NOTCH1/NOTCH2.
- Baumgart et al., Nature Communications, 2023 (PMID: 37268635) — a NOTCH4 splice variant sensitizes lung adenocarcinomas to EGFR-TKIs via transcriptional down-regulation of HES1.
- Xu et al., Medicine, 2018 (PMID: 30593175) — NOTCH4/DLL4 correlate with vasculogenic mimicry, metastasis and prognosis in NSCLC.
- Zhang et al., Cancer Letters, 2024 (PMID: 38301911) — review of Notch signaling and targeted therapy specifically in NSCLC.
Weaker tier: n_sig/eff are in the same generic-looking band as most
rejected candidates, so this pathway is kept on mechanistic specificity of
the literature rather than on its statistic being distinguishing.

### Regulation Of RUNX3 Expression And Activity
n_sig=18/26, size=54, eff=5.0, hist_frac=0%
RUNX3 is a well-established tumor suppressor recurrently silenced by
promoter hypermethylation specifically in lung cancer, with functional loss
tied to TGF-beta pathway escape in lung epithelium.
- Li et al., Oncol Rep, 2006 (PMID: 16328045) — epigenetic inactivation of RUNX3 in lung cancer.
Weaker tier: only one directly on-target PMID retrieved in this search pass;
recommend a follow-up search before treating this as fully confirmed.

## Considered and rejected

- **Vif-mediated Degradation Of APOBEC3G** (n_sig=18, eff=5.09) — rejected.
  This Reactome pathway models HIV Vif-APOBEC3G viral restriction biology,
  not APOBEC-driven tumor mutagenesis (APOBEC3A/3B, not APOBEC3G, are the
  cancer-mutagenesis-relevant family members). Genuine APOBEC/lung-cancer
  literature exists (PMID: 38382595, passive-smoking-induced APOBEC-associated
  mutagenesis in lung carcinogenesis; PMID: 32649875, APOBEC signature in
  non-smoking lung cancer) but does not map onto this specific gene set/pathway.
- **HIV Infection / Host Interactions Of HIV Factors** (n_sig=14-17, eff=6.0-6.4)
  — rejected. These Reactome terms capture generic host cell machinery
  (nuclear transport, transcription/immune factors) hijacked during HIV
  infection, not a lung-cancer mechanism; broad "HIV lung cancer risk" search
  returned only generic epidemiological co-morbidity literature (HIV+ patients
  have elevated lung cancer risk via smoking/immunosuppression confounds), not
  a pathway-level mechanistic link to this specific gene set. Treated as a
  cfRNA immune/interferon-response artifact akin to the already-[GENERIC]
  Immune System pathway.
- **Mitochondrial Protein Import** (n_sig=20, eff=4.97) and **Complex I
  Biogenesis** (n_sig=21, eff=4.54) — rejected. Targeted PubMed searches for
  lung-cancer-specific mitochondrial import (TIMM/TOMM) or Complex I biology
  returned no lung-cancer-specific mechanistic hits distinguishable from
  pan-cancer mitochondrial/OXPHOS dysregulation literature; treated as a
  generic metabolic/turnover signal, consistent with the broad flat n_sig/eff
  profile noted above.
- **DNA Replication** (hist_frac=18%), **DNA Replication Pre-Initiation**
  (hist_frac=22%), **Assembly Of Pre-Replicative Complex** (hist_frac=26%) —
  excluded per histone-composition-artifact rule; no lung-specific literature
  search performed given the rule-based exclusion.
- **Cell-cycle / APC-C degradation sub-modules** (Autodegradation Of Cdh1,
  M Phase, Synthesis Of DNA, S Phase, Mitotic Anaphase, Separation Of Sister
  Chromatids, Stabilization Of P53, and ~20 similarly-scored APC/C-degradation
  micro-pathways, n_sig=17-19, eff~4.6-5.5) — rejected as generic
  proliferation-machinery pathways applicable to essentially any solid tumor;
  not lung-specific, consistent with [GENERIC] Cell Cycle / Cell Cycle,
  Mitotic already flagged in the source list.
- **Cellular Response To Chemical Stress** (n_sig=17, eff=5.1) — plausible
  smoking/xenobiotic-stress narrative but no targeted literature search was
  run since it is a broad stress-response umbrella term structurally
  analogous to the already-[GENERIC] Cellular Responses To Stress/Stimuli;
  excluded on the same generic-umbrella basis.

## Raw search log

PubMed PMIDs retrieved during this review (not all cited):
- KEAP1/NFE2L2 lung cancer search: 22960745, 31548347, 31323387, 17020408, 29615460, 36240971, 27499952, 39111731
- RUNX3 lung cancer search: 16328045, 27501331, 23670097, 23800731, 24889513, 38593249, 14968123, 23982143
- NOTCH4 lung cancer search: 31894255, 37268635, 30593175, 38301911, 37961223, 38984877, 34988077, 37765264
- APOBEC/lung cancer/smoking search: 32649875, 38382595, 40394004, 38617360, 40502742, 39896515, 28862766, 24552141
- Mitochondrial complex I / respiratory chain / lung cancer search: 40239706, 31119045, 32641834, 40618880, 40752580, 21830212, 29452639, 31874028 (none found lung-cancer-specific enough to cite)
- HIV / lung cancer risk search: 34967848, 37500684, 33911981, 38642570, 30558872, 40504560, 23892408, 31560378 (generic comorbidity literature, not cited)
