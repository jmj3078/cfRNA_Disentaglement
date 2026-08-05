# Stomach Cancer — Pathway Literature Review

Cohort n=21 (smallest in this project's batch). Candidate list dominated by
[GENERIC] umbrella terms (Cell Cycle, RNA Pol II Transcription, Spliceosome,
Immune System) and by Olfactory Signaling/Sensory Perception/Olfactory
transduction, all treated as non-specific per instructions and excluded
outright. **Caution flag**: this cohort had the highest median pathway-Jaccard
of all 10 phenotypes in the project despite being the smallest n — plausibly a
small-sample statistical artifact (permutation null instability / correlated
gene-module co-significance) rather than genuinely stronger biological
convergence. The selections below favor candidates with an independent,
mechanism-specific literature anchor for gastric adenocarcinoma specifically,
not just co-occurrence with the generic proliferation/transcription block.

## Selected pathways

### 1. Transcriptional Regulation By TP53
n_sig=13/21, size=353, eff=4.66, hist_frac=0%.
TP53 is one of the most frequently mutated genes in gastric adenocarcinoma and
defines the chromosomal-instability (CIN) molecular subtype in the TCGA
4-subtype classification of gastric cancer — this is a disease-defining,
subtype-level driver relationship, not a generic "cancer has p53 mutations"
statement.
- Cancer Genome Atlas Research Network, Nature, 2014 (PMID: 25079317) — comprehensive TCGA molecular characterization of gastric adenocarcinoma; defines 4 subtypes (EBV, MSI, genomically stable, CIN) with CIN subtype marked by near-universal TP53 mutation/inactivation.

### 2. DNA Repair
n_sig=12/21, size=309, eff=4.92, hist_frac=8%.
Mismatch-repair (MMR) deficiency / microsatellite instability (MSI) is one of the four TCGA molecular subtypes of gastric cancer specifically (~20% of cases), with distinct clinicopathological features, hypermutation, and immunotherapy response — a gastric-cancer-specific, clinically actionable stratification, not generic "DNA damage in cancer."
- Cancer Genome Atlas Research Network, Nature, 2014 (PMID: 25079317) — same TCGA study; MSI defined as one of the 4 core molecular subtypes of gastric adenocarcinoma.
- Zhang Q et al., Int J Clin Exp Pathol, 2018 (PMID: 31938371) — clinicopathological features and prognostic value of MMR protein deficiency specifically in gastric cancer cohorts.

### 3. Negative Regulation Of NOTCH4 Signaling
n_sig=13/21, size=53, eff=4.4, hist_frac=0%.
NOTCH4 specifically (not generic Notch pathway) has direct mechanistic evidence in gastric cancer: it promotes tumor growth via Wnt1/β-catenin activation and its translation is activated by a lncRNA-miRNA axis that drives gastric cancer metastasis. Weaker-tier: only 2 primary mechanistic papers found (vs. multi-cohort TCGA-level evidence for #1-2), and the candidate pathway is about NOTCH4 degradation/turnover rather than the activating axis itself, so the gene-level link to these papers is indirect (same NOTCH4-centric machinery, not identical mechanism).
- Wu Y et al. (Kong et al./Mol Cell Biochem group), Mol Cell Biochem, 2015 (PMID: 25511451) — "Notch4 promotes gastric cancer growth through activation of Wnt1/β-catenin signaling."
- Zhang YT et al., Front Pharmacol, 2024 (PMID: 39309008) — lncRNA CADM2-AS1 promotes gastric cancer metastasis by activating NOTCH4 translation via miR-5047 sponging.

## Considered and rejected

- **Herpes simplex virus 1 infection** (n_sig=11, size=491, eff=5.96, hist_frac=0%) — Epstein-Barr virus (EBV)-positive gastric cancer is a genuine, well-established TCGA molecular subtype (PMID: 25079317), and EBV is a herpesvirus. But the Reactome "Herpes simplex virus 1 infection" geneset is built around HSV1-specific host-interaction biology, not EBV; no paper directly linking this specific pathway/geneset to gastric cancer was found. Treating HSV1-annotated genes as an EBV-GC proxy would be a stretch not supported by the literature actually retrieved — rejected rather than included as "weaker tier."
- **Ribosome biogenesis in eukaryotes** (n_sig=16, size=76, eff=4.68, hist_frac=0%) — searched for nucleolar-stress/ribosome-biogenesis literature specific to gastric cancer; found only 1 weak, tangential hit. Insufficient targeted evidence.
- **GPCR Ligand Binding** (n_sig=11, size=447, eff=5.06) / **Neuroactive ligand-receptor interaction** (n_sig=11, size=331, eff=5.12) — gastrin/CCK2R signaling is a real gastric oncogenic axis, but the top literature hit for CCK2R and cancer was gastrointestinal stromal tumor pathogenesis (PMID: 22786615), not gastric adenocarcinoma, and these Reactome/KEGG genesets are broad catch-alls spanning essentially all GPCR classes (opioid, adrenergic, olfactory-adjacent, etc.) expressed by normal gastric mucosa. More consistent with tissue-of-origin composition than a gastric-cancer-specific disease signal — rejected.
- **Keratinization** (n_sig=15, size=207, eff=4.75, hist_frac=0%) — searched for keratin/intestinal-metaplasia/CDX2 links to gastric cancer; the retrieved hits were about small-bowel adenocarcinoma with gastric differentiation and Crohn-associated tumors, not primary gastric adenocarcinoma keratinization biology. No direct gastric-cancer-specific support found.
- All **[GENERIC]**-tagged pathways (Metabolism Of RNA/Proteins, Gene Expression/RNA Pol II Transcription, Cell Cycle and its mitotic/APC-C sub-pathways, mRNA Splicing/Spliceosome, Cellular Responses To Stress/Stimuli, Sensory Perception, Olfactory Signaling/transduction/receptor pathways, Infectious Disease, Disease, RNA transport, Protein Localization, Processing Of Capped Intron-Containing Pre-mRNA, Immune System, Innate Immune System) — excluded per instructions; no case made for exceptions among these for this phenotype.
- Numerous mitotic/APC-C degradation sub-pathways (M Phase, Mitotic Anaphase, Separation Of Sister Chromatids, S Phase, Synthesis Of DNA, Cdc20/Cdh1-mediated degradation cascade, etc.) — these are fragments of the same generic Cell Cycle signal already excluded; not independently investigated as they add no gastric-specific mechanism beyond proliferation.

## Raw search log

- PMID 25079317 — Cancer Genome Atlas Research Network, Nature 2014, "Comprehensive molecular characterization of gastric adenocarcinoma" (DOI: 10.1038/nature13480). Cited.
- PMID 31938371 — Zhang Q et al., Int J Clin Exp Pathol 2018, MMR protein deficiency in gastric cancer. Cited.
- PMID 35925389 — Angerilli V et al., Virchows Arch 2022, molecular subtyping of gastroesophageal dysplasia per TCGA/ACRG. Reviewed, not cited (dysplasia not adenocarcinoma cohort).
- PMID 37007634 — Boutin M, Gill S, Ther Adv Med Oncol 2023, dMMR GI cancers neoadjuvant management review. Reviewed, not cited.
- PMID 35877698 — Rigter LS et al., PLoS One 2022, molecular characterization of gastric adenocarcinoma post Hodgkin/testicular cancer treatment. Reviewed, not cited.
- PMID 29116623 — Polom K et al., Pathol Oncol Res 2019, KRAS mutation and MSI status in gastric cancer. Reviewed, not cited.
- PMID 25511451 — Notch4 promotes gastric cancer growth through Wnt1/β-catenin signaling, Mol Cell Biochem 2015. Cited.
- PMID 39309008 — Zhang YT et al., Front Pharmacol 2024, lncRNA CADM2-AS1/miR-5047/NOTCH4 axis in gastric cancer metastasis. Cited.
- PMID 22786615 — CCK2R in gastrointestinal stromal tumour pathogenesis, J Pathol 2012. Reviewed, used as rejection evidence (wrong tumor type).
- PMID 24331840 — Crohn enteritis-associated small bowel adenocarcinomas with gastric differentiation, Hum Pathol 2014. Reviewed, used as rejection evidence.
- PMID 35925388 — Claudin-18 expression in small bowel adenocarcinoma, Virchows Arch 2022. Reviewed, not cited.
- PMID 36941105 — sole hit for ribosome biogenesis / nucleolar stress + gastric cancer search. Reviewed, not cited (insufficient/tangential).
- EBV-associated gastric cancer search (33 PubMed hits, term: Epstein-Barr virus + gastric cancer + TCGA subtype) — top hits not individually pulled beyond confirming EBV-GC is TCGA-subtype-level real biology (already covered by PMID 25079317); used only to evaluate (and reject) the Herpes simplex virus 1 infection candidate.
