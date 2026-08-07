# Stomach Cancer — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Stomach_Cancer.txt` (n=21 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below.

## Selected pathways

### 1. Negative Regulation Of NOTCH4 Signaling
n_sig=13/21, size=53, eff=4.36, hist_frac=0%

NOTCH4 has a specific, repeatedly-reported oncogenic role in gastric cancer (growth via Wnt1/beta-catenin crosstalk, and lncRNA-mediated NOTCH4 translation driving metastasis), distinct from generic pan-cancer Notch pathway mentions. Narrow gene-set size (53 genes) and non-GENERIC status support this as a specific, mechanistically defensible hit rather than a broad umbrella term.
- Qian C et al., Mol Cell Biochem, 2015 (PMID: 25511451) — Notch4 promotes gastric cancer cell growth through activation of Wnt1/beta-catenin signaling.
- Zhang YT et al., Front Pharmacol, 2024 (PMID: 39309008) — lncRNA CADM2-AS1 promotes gastric cancer metastasis by activating NOTCH4 translation via miR-5047 sponging.

### 2. Separation Of Sister Chromatids
n_sig=14/21, size=170, eff=5.29, hist_frac=0%

Errors in sister chromatid separation are the proximate mechanistic driver of chromosomal instability (CIN), which TCGA established as one of the four defining molecular subtypes of gastric adenocarcinoma (alongside EBV+, MSI, genomically stable), marked by widespread aneuploidy and focal amplifications. This is a specific mitotic-fidelity mechanism with direct, well-established gastric cancer relevance, not a generic cell-cycle catch-all.
- Cancer Genome Atlas Research Network, Nature, 2014 (PMID: 25079317) — Comprehensive molecular characterization of gastric adenocarcinoma; defines the chromosomal instability (CIN) subtype.
- Nemtsova MV et al., Int J Mol Sci, 2023 (PMID: 38069284) — Review: chromosomal instability in gastric cancer, role in tumor development, progression, and therapy.

### 3. DNA Repair
n_sig=12/21, size=309, eff=4.89, hist_frac=8.4%

Mismatch-repair (MMR) deficiency and the resulting microsatellite instability (MSI) define a distinct, clinically actionable TCGA molecular subtype of gastric cancer (~20% of cases), with proven predictive value for immune checkpoint blockade. hist_frac (8.4%) is below the 15% exclusion threshold. This is a broad Reactome term, but its clinical/mechanistic link to gastric cancer specifically (as opposed to cancer in general) is exceptionally well documented, so it is retained.
- Marabelle A et al., J Clin Oncol, 2020 (PMID: 31682550) — KEYNOTE-158: pembrolizumab efficacy in MSI-high/dMMR cancers, including gastric.
- Ooki A et al., Gastric Cancer, 2024 (PMID: 38922524) — Review: therapeutic strategy in MSI-high/dMMR gastric cancer.
- Andre T et al., J Clin Oncol, 2023 (PMID: 35969830) — NEONIPIGA: neoadjuvant immunotherapy in dMMR/MSI-H gastric/GEJ adenocarcinoma.

### 4. Ribosome biogenesis in eukaryotes
n_sig=14/21, size=76, eff=4.76, hist_frac=0%

Multiple independent mechanistic studies show ribosome biogenesis is actively hijacked in gastric cancer to drive proliferation, chemoresistance, and replication-stress tolerance (lncRNA-driven rDNA transcription, stromal FGF2 signaling, nucleolar TCOF1/treacle), rather than being a passive housekeeping readout of general translational activity.
- Zang W et al., Cell Mol Biol Lett, 2025 (PMID: 41402710) — lncRNA LINC01940 promotes gastric cancer progression and chemoresistance by enhancing ribosome biogenesis via TAF15-mediated NOL11 SUMOylation.
- Li D et al., Int Immunopharmacol, 2024 (PMID: 38479160) — Cancer-associated fibroblasts promote gastric cancer proliferation via paracrine FGF2-driven ribosome biogenesis.
- Nie X et al., J Gastroenterol Hepatol, 2023 (PMID: 36941105) — Nucleolar TCOF1 (treacle) maintains gastric cancer proliferation by regulating R-loop-associated replication stress via ribosome biogenesis machinery.

### 5. Transcriptional Regulation By TP53 [GENERIC override]
n_sig=11/21, size=353, eff=5.06, hist_frac=0%

This row is flagged GENERIC (broad-sounding term) and is excluded by default, but is kept here as an explicit exception: TP53 is the single most frequently mutated driver gene in gastric adenocarcinoma and its mutation status is a core axis of the TCGA molecular classification (near-ubiquitous in the CIN subtype), not an umbrella term that lights up for unrelated biology. The pathway gene set directly reflects TP53's disease-defining transcriptional program in this cancer type.
- Cancer Genome Atlas Research Network, Nature, 2014 (PMID: 25079317) — TP53 mutation frequency and its role in defining the CIN molecular subtype of gastric adenocarcinoma.
- Cristescu R et al., Nat Med, 2015 (PMID: 25894828) — ACRG molecular subtypes of gastric cancer associated with distinct clinical outcomes, including TP53-mutant vs TP53-active subgroups.

## Considered and rejected

- **Autodegradation Of Cdh1 By Cdh1:APC/C** (n_sig=13, size=63, eff=4.62, hist_frac=0%). Rejected: "Cdh1" here refers to FZR1 (APC/C coactivator), not CDH1/E-cadherin — a name collision risk. Targeted PubMed search for FZR1/APC-Cdh1 in gastric cancer returned zero results; no direct literature support found.
- **Mitochondrial Protein Import** (n_sig=14, size=65, eff=4.50, hist_frac=0%). Rejected: searches returned only general mitochondrial-metabolism/OXPHOS gastric cancer papers, none addressing the TOM/TIM import machinery specifically; evidence too indirect for this narrow mechanism.
- **M Phase / Mitotic Anaphase / Mitotic Metaphase And Anaphase / Mitotic Prometaphase / Regulation Of APC/C Activators Between G1/S And Early Anaphase** (various n_sig=12-14, hist_frac=0-0.08%). Rejected as redundant with **Separation Of Sister Chromatids** — all represent the same underlying chromosomal-instability/mitotic-fidelity mechanism in gastric cancer; the selected pathway is the most mechanistically specific and among the higher-scoring of this cluster.
- **Switching Of Origins To A Post-Replicative State / Synthesis Of DNA / S Phase** (n_sig=12, hist_frac=0%). Rejected: generic DNA-replication/proliferation machinery; no gastric-cancer-specific mechanistic literature found (MCM/replication-licensing search returned zero gastric-specific hits).
- **HIV Infection / SARS-CoV-2 Infection / SARS-CoV Infections / Herpes simplex virus 1 infection / HIV Life Cycle / Host Interactions Of HIV Factors / Late Phase Of HIV Life Cycle** (n_sig=12-15). Rejected: these Reactome/KEGG viral-infection gene sets overlap heavily with core transcription/translation/immune machinery and light up as an artifact of general cellular activity; no gastric-cancer-specific virology rationale (distinct from the EBV+ TCGA subtype, which these gene sets do not represent).
- **Keratinization** (n_sig=14, size=207, eff=4.88, hist_frac=0%). Rejected: no gastric-cancer-specific mechanistic literature identified; keratin gene sets more plausibly reflect esophageal/squamous contamination or a non-specific epithelial signature in this cohort.
- **GPCR Ligand Binding / Neuroactive ligand-receptor interaction / Olfactory Signaling Pathway / Olfactory transduction / Expression And Translocation Of Olfactory Receptors** (n_sig=11-15). Rejected: extremely large, non-specific receptor gene families; classic false-positive-prone gene sets in bulk/cfRNA enrichment, no gastric-cancer mechanism identified.
- **tRNA Processing (In Nucleus) / Protein Localization / Deadenylation-dependent mRNA Decay / RNA transport / mRNA Splicing (- Major Pathway) / Mitochondrial tRNA-adjacent housekeeping sets**. Rejected: core RNA-metabolism housekeeping machinery, not disease-specific despite high recurrence rates.
- All [GENERIC]-flagged rows (Metabolism Of RNA, Gene Expression (Transcription), Metabolism Of Proteins, Cellular Responses To Stress/Stimuli, RNA Polymerase II Transcription, Cell Cycle, Cell Cycle Checkpoints, Immune System, Innate Immune System, Infectious Disease, Disease, Generic Transcription Pathway, Spliceosome, Post-translational Protein Modification, mRNA Splicing) and all high-histone rows other than the TP53 exception above were excluded per the standing project rule.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 39309008 — lncRNA CADM2-AS1/NOTCH4 gastric cancer metastasis (cited)
- 25511451 — Notch4/Wnt1-beta-catenin gastric cancer growth (cited)
- 31399040 — protein expression profiles, gastric cancer survival (checked, not cited — not NOTCH4-mechanism specific enough)
- 39309429 — E2F4/DSCC1 gastric cancer proliferation (checked, considered for sister chromatid cohesion, not cited — DSCC1 replication-fork factor, weaker fit)
- 24318971 — STAG2 in oral cancer (checked, rejected — wrong cancer type)
- 36117847 — deep neural network prognosis model citing biological pathways (checked, rejected — not primary mechanistic evidence)
- 26420833 — cohesin/chromosomal instability, pan-cancer (checked, not cited — supports mechanism generally but not gastric-specific)
- 25079317 — TCGA comprehensive molecular characterization of gastric adenocarcinoma (cited, CIN subtype + TP53)
- 38069284 — chromosomal instability in gastric cancer review (cited)
- 38328310 — ultrasensitive aneuploidy detection in gastric precancerous lesions (checked, not cited — clinical detection focus, not core mechanism)
- 31682550 — KEYNOTE-158 pembrolizumab in MSI-high/dMMR cancers (cited)
- 33592120 — current treatment and progress in gastric cancer, CA Cancer J Clin (checked, not cited — general review)
- 38922524 — MSI-high/dMMR gastric cancer therapeutic strategy review (cited)
- 35969830 — NEONIPIGA phase II, dMMR/MSI-H gastric/GEJ adenocarcinoma (cited)
- 41402710 — LINC01940/NOL11 ribosome biogenesis gastric cancer (cited)
- 38479160 — CAF/FGF2-driven ribosome biogenesis gastric cancer (cited)
- 36438703 — proteomic signatures of infiltrative gastric cancer (checked, not cited — bioinformatic survey, not mechanistic)
- 36941105 — TCOF1/treacle ribosome biogenesis, R-loop replication stress, gastric cancer (cited)
- 41361128 — GI cancer molecular pathogenesis and targeted therapy review (checked, not cited — too broad/general)
- 39279940 — DEG prognostic signature, cardia/non-cardia gastric cancer (checked, not cited — not TP53-mechanism specific)
- 31171626 — multiplex profiling of gastric peritoneal metastases (checked, not cited — not TP53-mechanism specific)
- 25894828 — ACRG molecular subtypes of gastric cancer, Nat Med (cited)
- 38959111 — metabolic signature subtypes in gastric cancer, Cell Rep (checked, not cited — metabolic focus, not TP53-specific)
- 40539884 — NPR1/lipid droplet lipolysis/mitochondrial OXPHOS, gastric cancer metastasis (checked, rejected for Mitochondrial Protein Import — metabolism, not import machinery)
- 42099017 — mitochondrial transplantation reprogramming in gastric cancer cells (checked, rejected — not import machinery)
- 24821064 — stromal MCT4/mitochondrial TOMM20 as prognostic factors, gastric cancer (checked, rejected — single marker correlate, not import-pathway mechanism)
- 35230214 — L22 ribosomal protein/DRP1-mediated gastric carcinoma progression (checked, rejected for mitochondrial import — mixed ribosome/mitochondrial fission topic)
- FZR1+Cdh1+APC/C AND gastric cancer — 0 results (search performed, no PMIDs returned; supports rejection of Autodegradation Of Cdh1 By Cdh1:APC/C)
- MCM+DNA replication licensing AND gastric cancer — 0 results (search performed, no PMIDs returned; supports rejection of Switching Of Origins To A Post-Replicative State)
