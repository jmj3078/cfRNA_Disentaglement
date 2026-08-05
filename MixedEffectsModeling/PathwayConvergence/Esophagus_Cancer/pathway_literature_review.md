# Esophagus Cancer — Pathway Literature Review

Candidate list source: `cand_Esophagus_Cancer.txt` (n=25 patients). Cohort majority-likely ESCC
(Reactome/KEGG library, no adenocarcinoma/squamous split available at pathway-call level, so
adenocarcinoma-specific evidence is flagged where used).

## Selected pathways

### 1. Transcriptional Regulation By TP53
n_sig=15/25, size=353, eff=4.65.
TP53 is the single most frequently mutated gene in esophageal squamous cell carcinoma (ESCC),
with mutation loads among the highest of any solid tumor in high-incidence regions — a
disease-defining, not generic-cancer, signal at this magnitude.
- Abedi-Ardekani et al., PLoS One, 2011 (PMID: 22216294) — extremely high TP53 mutation load in ESCC from a high-incidence Iranian cohort.

### 2. KEAP1-NFE2L2 Pathway / Nuclear Events Mediated By NFE2L2
n_sig=13/25 size=100 eff=5.2 (KEAP1-NFE2L2) and n_sig=14/25 size=78 eff=5.34 (Nuclear Events Mediated By NFE2L2).
NRF2 (NFE2L2)/KEAP1 pathway alterations are recurrent oncogenic drivers in ESCC specifically (not a generic stress-response readout), proposed as a targetable axis and linked to chemoradiotherapy resistance.
- Ma et al., Ann N Y Acad Sci, 2018 (PMID: 29752726) — NRF2 signaling pathway as a therapeutic target specifically in ESCC.
- Zhang et al., Thorac Cancer, 2018 (PMID: 29675925) — Nrf2/Keap1 abnormalities in ESCC associated with chemoradiotherapy response.

### 3. Regulation Of RUNX3 Expression And Activity
n_sig=14/25, size=54, eff=5.15.
RUNX3 is a well-characterized tumor suppressor recurrently silenced by promoter methylation in ESCC, with silencing associated with radioresistance and poor prognosis — a specific, mechanistically studied epigenetic lesion in this cancer.
- Sakakura et al., Oncogene, 2007 (PMID: 17384682) — frequent RUNX3 silencing in ESCC associated with radioresistance and poor prognosis.

### 4. Keratinization
n_sig=15/25, size=207, eff=5.0.
Esophageal squamous carcinogenesis proceeds through an aberrant squamous differentiation program (keratin gene dysregulation); this is specific to squamous-lineage tumors of the esophagus/aerodigestive tract, not a generic epithelial-cancer signature.
- Tian et al. (KLF4), J Biol Chem, 2015 (PMID: 25851906) — KLF4 promotes ESCC differentiation via keratin 13 up-regulation, tying keratin program directly to ESCC differentiation state.

### 5. Hedgehog Ligand Biogenesis / Hh Mutants Abrogate Ligand Secretion / GLI2 & GLI3 Proteasomal Processing
n_sig=14/25, size 51-63, eff 4.83-4.96.
Hedgehog pathway reactivation is a specific, mechanistically studied driver of the Barrett's esophagus → esophageal adenocarcinoma progression sequence, with pharmacologic Hedgehog inhibition tested to block this transition. Note: this evidence base is adenocarcinoma/Barrett's-specific rather than ESCC-specific, so treat as a distinct disease-subtype signal within "esophageal cancer" rather than pan-esophageal.
- Uchida et al., Histol Histopathol, 2016 (PMID: 26334343) — molecular background of Barrett's metaplasia to esophageal adenocarcinoma progression, including Hedgehog reactivation.
- Konda et al., Ann Surg, 2021 (PMID: 31290765) — itraconazole (Hedgehog pathway inhibitor) tested to prevent Barrett's-to-invasive-adenocarcinoma progression.

### 6. FBXL7 Down-Regulates AURKA During Mitotic Entry And In Early Mitosis
n_sig=14/25, size=54, eff=5.25.
Aurora kinase A (AURKA) is recurrently amplified/overexpressed in ESCC, distinguishing this specific mitotic-regulator pathway from the generic APC/C-degradation cluster that dominates the candidate list.
- Yang et al., Oncol Rep, 2007 (PMID: 17390048) — AURKA amplification and overexpression in ESCC.
- Ariyoshi et al., Hepatogastroenterology, 2003 (PMID: 14696419) — 20q gains (including AURKA locus) in ESCC by CGH/FISH.

### 7. Degradation Of AXIN — weaker tier
n_sig=13/25, size=54, eff=5.36.
Wnt/β-catenin pathway activation (of which AXIN turnover is a direct regulatory node) is documented specifically in Barrett's esophagus and its progression toward esophageal adenocarcinoma. Kept as weaker tier because the cited evidence addresses Wnt/β-catenin pathway activity broadly rather than AXIN degradation specifically, and is adenocarcinoma/Barrett's-specific.
- Baumeister et al., BMC Gastroenterol, 2019 (PMID: 30841855) — Wnt/β-catenin pathway characterization in a Barrett's sequence model.
- Wnt/β-catenin activation via Dickkopf-1 regulation in nondysplastic Barrett's esophagus, Neoplasia, 2015 (PMID: 26297437).

### 8. Regulation Of Ornithine Decarboxylase (ODC) — weaker tier
n_sig=14/25, size=50, eff=4.82.
Polyamine metabolism (ODC being its rate-limiting enzyme) has direct, esophagus-specific chemoprevention trial evidence (DFMO in Barrett's esophagus) and a mechanistic link to p53-deficient upper-aerodigestive carcinogenesis. Kept as weaker tier because supporting evidence is chemoprevention/animal-model rather than direct human ESCC tumor genomics.
- Garewal et al., Cancer Epidemiol Biomarkers Prev, 1994 (PMID: 8061581) — DFMO (ODC inhibitor) alters tissue polyamine content in Barrett's esophagus patients.
- Feith et al., Carcinogenesis, 2013 (PMID: 23222816) — ODC antizyme prevents upper aerodigestive tract carcinogenesis in p53-deficient mice.

## Considered and rejected

- **HIV Infection** (n_sig=15, size=228, eff=6.26) — explicitly flagged suspect artifact pattern; confirmed. A genuine HIV-and-esophageal-cancer epidemiological association exists (PMID: 38530745 Sub-Saharan Africa meta-analysis; PMID: 30939533 US veterans cohort), but this reflects clinical/immunosuppression epidemiology of HIV+ patients, not a mechanistic transcriptomic signal from the KEGG "HIV Infection" host-interaction gene set. That gene set clusters tightly in rank/effect with **Vif-Mediated Degradation Of APOBEC3G**, **Vpu Mediated Degradation Of CD4**, and **Host Interactions Of HIV Factors** (all n_sig=13-14, eff~5.2-5.6) — a signature of shared generic ubiquitin/proteasome degradation machinery overlap, not HIV/ESCC biology. Rejected, along with the three sibling HIV-subcomponent pathways.
- **Negative Regulation Of NOTCH4 Signaling** (n_sig=14, size=53, eff=5.75) — searched specifically for NOTCH4-esophageal cancer literature; no genuine NOTCH4-specific hits returned. Only NOTCH1/NOTCH3 have established, well-studied roles in ESCC (e.g., PMID: 29170450 Notch1/Notch3 interplay in EMT and tumor initiation in squamous cell carcinoma; PMID: 22877736 comparative genomics of EAC/ESCC), which does not transfer to this specific NOTCH4 gene-set entry. Rejected for lack of paralog-specific evidence.
- **Generic APC/C- and SCF-mediated proteasomal degradation cluster** (Autodegradation Of Cdh1 By Cdh1:APC/C, APC/C:Cdc20 Mediated Degradation Of Securin, CDK-mediated Phosphorylation And Removal Of Cdc6, SCF(Skp2)-mediated Degradation Of P27/P21, Ubiquitin-dependent Degradation Of Cyclin D, SCF-beta-TrCP Mediated Degradation Of Emi1, Autodegradation Of E3 Ubiquitin Ligase COP1, Regulation Of Activated PAK-2p34 By Proteasome Mediated Degradation, Degradation Of DVL, Cdc20:Phospho-APC/C Mediated Degradation Of Cyclin A, Activation Of APC/C And APC/C:Cdc20 Mediated Degradation Of Mitotic Proteins, Regulation Of APC/C Activators Between G1/S And Early Anaphase, APC/C:Cdh1 Mediated Degradation Of Cdc20..., APC/C:Cdc20 Mediated Degradation Of Mitotic Proteins) — this is generic cell-cycle/mitotic proteasomal machinery expected to score in essentially any proliferative tumor; no targeted search performed given the sheer redundancy with each other and with already-GENERIC-flagged Cell Cycle/Cell Cycle Mitotic/M Phase entries. Rejected as non-specific.
- **Defective CFTR Causes Cystic Fibrosis** (n_sig=14, size=60, eff=5.1) — unrelated disease gene set overlap artifact, no plausible ESCC mechanism; rejected without further search.
- **NIK To Noncanonical NF-kB Signaling / Dectin-1 Mediated Noncanonical NF-kB Signaling / Activation Of NF-kappaB In B Cells** — generic inflammatory signaling nodes with no esophagus-specific literature sought given redundancy with already-excluded generic Immune System pathway; rejected as likely nonspecific.
- **Synthesis Of DNA, Switching Of Origins To A Post-Replicative State, Neddylation, Proteasome, Mitotic Anaphase, Mitotic Metaphase And Anaphase, Complex I Biogenesis, Thermogenesis, Cellular Response To Chemical Stress, Regulation Of Apoptosis** — generic housekeeping/replication/proteostasis machinery with no ESCC-specific mechanism argued or searched; rejected as non-specific by category.

## Raw search log

TP53 search: 34663923, 38039962, 40500695, 34663841, 22216294 (cited)
NFE2L2/KEAP1 search: 40911942, 39500864, 40781161, 29752726 (cited), 29675925 (cited)
RUNX3 search: 17384682 (cited), 16678495
NOTCH1 search: 29170450, 40628272, 22877736, 27734031, 40667321
Hedgehog search: 26334343 (cited), 31290765 (cited), 37298253, 27331918, 23730883
Wnt/AXIN search: 30841855 (cited), 26297437 (cited)
AURKA search: 17390048 (cited), 14696419 (cited)
Keratin search: 40498854, 25851906 (cited), 29788741, 39234567, 14764456
HIV search: 38530745 (cited, epidemiology only), 30939533 (cited, epidemiology only), 34636955, 38916210, 25641622
ODC search: 10353725, 1389696, 8061581 (cited), 23222816 (cited), 11303587
NOTCH4 search: 31894255, 39309008, 17143535, 34284787, 26404689 (none yielded NOTCH4-specific ESCC evidence, all screened by title only)
