# Pre-eclampsia — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Pre-eclampsia.txt` (n=58 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below.

## Selected pathways

### 1. Neutrophil Degranulation
n_sig=16/58, size=463, eff=5.77, hist_frac=0%

Excessive neutrophil activation/degranulation is a well-documented feature of the systemic inflammatory response in preeclampsia, linked to endothelial dysfunction and preceding clinical GDM/PE overlap syndromes.
- Faas & de Vos et al., Frontiers in Endocrinology, 2018 (PMID: 30298053) — reviews excessive neutrophil activity in gestational diabetes as a contributor to preeclampsia development.
- Rimon et al., Frontiers in Immunology, 2020 (PMID: 32117288) — placental protein 13 (Galectin-13) polarizes neutrophils toward a regulatory phenotype, implicating neutrophil activation state in normal vs. preeclamptic placentation.

### 2. Signaling By Interleukins
n_sig=10/58, size=449, eff=4.48, hist_frac=0.4%

Interleukin-6 and related interleukin signaling are central to the maternal systemic inflammatory response in preeclampsia and correlate with disease severity, supporting this broader Reactome grouping when anchored to a specific, well-studied cytokine.
- Ozler et al., J Matern Fetal Neonatal Med, 2022 (PMID: 31964198) — IL-6 (with presepsin/pentraxin-3) associates with diagnosis and severity of late-onset preeclampsia.
- Xu et al./Fan et al., Genet Mol Res, 2017 (PMID: 28252161) — IL-6 and IL-10 polymorphisms associated with preeclampsia risk.

### 3. Diseases Of Signal Transduction By Growth Factor Receptors And Second Messengers
n_sig=11/58, size=423, eff=3.77, hist_frac=0%

This Reactome "disease" pathway captures dysregulated growth-factor-receptor signaling, and in preeclampsia the anchor mechanism is excess placental soluble fms-like tyrosine kinase-1 (sFlt1) sequestering VEGF/PlGF — the single best-established molecular pathway in preeclampsia pathophysiology (anti-angiogenic imbalance).
- Maynard et al., J Clin Invest, 2003 (PMID: 12618519) — excess placental sFlt1 causes endothelial dysfunction, hypertension, and proteinuria in preeclampsia (foundational anti-angiogenic-imbalance paper).
- Review, Int J Mol Sci, 2025 (PMID: 41226469) — role of angiogenic factors (VEGF/PlGF/sFlt1) in preeclampsia.

### 4. Platelet Degranulation
n_sig=9/58, size=123, eff=4.23, hist_frac=0%

Platelet activation and thrombo-inflammation are consistently reported in preeclampsia and used as candidate severity/predictive markers, consistent with the vascular/coagulopathy component of the disease.
- Sancak et al., Cirugia y Cirujanos, 2024 (PMID: 38537226) — platelet activation markers predict preeclampsia and its severity.
- 2025, Res Pract Thromb Haemost (PMID: 40746439) — increased platelet activation and thrombo-inflammation in early- and late-onset preeclampsia.

"Response To Elevated Platelet Cytosolic Ca2+" (n_sig=8, size=128, eff=4.42, hist_frac=0%) covers the same platelet-activation mechanism and is rejected as redundant with this better-scored entry.

### 5. NOD-like Receptor Signaling Pathway
n_sig=8/58, size=178, eff=3.94, hist_frac=0%

NLRP3 inflammasome activation downstream of NOD-like receptor signaling is directly implicated in preeclampsia pathogenesis via trophoblast pyroptosis and sterile inflammation.
- Weel et al., Frontiers in Endocrinology, 2020 (PMID: 32161574) — role of the NLRP3 inflammasome in preeclampsia.
- Vishnyakova et al., Mol Hum Reprod, 2023 (PMID: 37788097) — inflammasomes in human reproductive diseases, including preeclampsia.

### 6. NF-kappa B Signaling Pathway
n_sig=8/58, size=101, eff=4.08, hist_frac=0%

NF-kB activation in placental trophoblast is a recurrent mechanistic node in preeclampsia models, linking hypoxia/oxidative stress to the inflammatory transcriptional response.
- Wang et al., Placenta, 2024 (PMID: 38008034) — USP17 regulates preeclampsia by modulating NF-kB signaling via deubiquitinating HDAC2.

### 7. Th17 Cell Differentiation
n_sig=9/58, size=105, eff=3.82, hist_frac=0%

The Th1/Th17 vs. Treg/Th2 imbalance at the maternal-fetal interface is a well-established immunological mechanism proposed for preeclampsia, with preeclamptic dendritic cells directly shown to skew Th17 differentiation in vitro.
- Wang et al., Int J Clin Exp Med, 2014 (PMID: 25664035) — dendritic cells from preeclampsia patients influence Th1/Th17 differentiation in vitro.
- Saito et al., Frontiers in Immunology, 2020 (PMID: 32973809) — review of Th cell (incl. Th17) profiles in pregnancy and pregnancy loss/preeclampsia-related disorders.

### 8. Neutrophil Extracellular Trap Formation
n_sig=8/58, size=183, eff=3.87, hist_frac=39.9% — histone-composition exception

hist_frac exceeds the 15% default cutoff, but this is kept as an explicit exception: NETosis mechanistically requires chromatin decondensation and extracellular histone release, so the histone-gene enrichment reflects the pathway's actual biology (not an unrelated composition artifact). NETs are directly and repeatedly implicated as a driver of endothelial damage in preeclampsia.
- Zhu et al., Pharmaceuticals (Basel), 2024 (PMID: 38794175) — targeting NET formation as a pharmacological strategy for preeclampsia.
- Domingues et al./review, Int J Mol Sci, 2023 (PMID: 37958788) — role of NETs in health and disease pathophysiology, including preeclampsia-relevant vascular injury.

## Considered and rejected

- **Response To Elevated Platelet Cytosolic Ca2+** (n_sig=8, size=128, eff=4.42, hist_frac=0%). Same platelet-activation mechanism as "Platelet Degranulation"; lower-scoring duplicate, rejected as redundant.
- **Interferon Gamma Signaling** (n_sig=7, size=87, eff=4.56, hist_frac=0%). No specific, mechanistically strong preeclampsia paper identified beyond generic pregnancy-immunology reviews; not selected.
- **Cytokine Signaling In Immune System** (n_sig=18, size=693, eff=4.97, hist_frac=0.3%). Overly broad umbrella covering IL-6/NF-kB/interferon signaling already represented by more specific selected entries; redundant, not independently selected.
- **PRC2 Methylates Histones And DNA** (n_sig=9, size=41, eff=3.98, hist_frac=68.3%). PubMed search for "PRC2 + placenta + imprinting + preeclampsia" returned zero hits; no disease-specific literature found, so the high histone fraction is treated as an unresolved composition artifact and the pathway is excluded per the default rule (no exception granted).
- **DNA Methylation** (n_sig=8, size=33, eff=3.96, hist_frac=84.8%). hist_frac far exceeds cutoff; no targeted literature search performed given very small pathway size and no a priori mechanistic case distinct from PRC2 above; excluded.
- **Defective Pyroptosis** (n_sig=7, size=41, eff=4.18, hist_frac=68.3%). High histone fraction, low n_sig/size, mechanism largely overlaps with the already-selected NLRP3/NOD-like receptor pathway; excluded as redundant and under-evidenced independently.
- **Olfactory Signaling Pathway / Expression And Translocation Of Olfactory Receptors / Olfactory transduction / Sensory Perception** (top-ranked by score but no plausible preeclampsia mechanism; likely reflect large gene-family / detection-power artifacts rather than disease biology). Not investigated further, no disease link expected.
- Various viral-infection pathways (SARS-CoV Infections, SARS-CoV-2 Infection, Epstein-Barr virus infection, Herpes simplex virus 1 infection, HIV Infection, Influenza A, Hepatitis B, Measles, Human T-cell leukemia virus 1 infection, Leishmaniasis) — generic infection-response gene sets with no preeclampsia-specific mechanism argued or searched; excluded as non-specific.
- Housekeeping/general machinery pathways (Processing Of Capped Intron-Containing Pre-mRNA, tRNA Processing, rRNA Modification In Nucleus And Cytosol, RNA transport, Keratinization, Osteoclast differentiation, Neuroactive ligand-receptor interaction) — no disease-specific mechanistic case; excluded.
- All [GENERIC]-flagged rows (Metabolism Of RNA, Immune System, Innate Immune System, Cellular Responses To Stress, Gene Expression (Transcription), Cellular Responses To Stimuli, RNA Polymerase II Transcription, Metabolism Of Proteins, Infectious Disease, Generic Transcription Pathway, Disease, Spliceosome, mRNA Splicing, mRNA Splicing - Major Pathway, Adaptive Immune System, Cell Cycle, Cell Cycle Mitotic, RHO GTPase Cycle) were excluded per the standing project rule; none were investigated for exception status as they are broad umbrella terms without a narrow disease-specific reading.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 30298053 — neutrophil activity in GDM/preeclampsia (cited)
- 32117288 — Galectin-13 polarizes neutrophils (cited)
- 32986992, 19464511, 11190901 — platelet activation + preeclampsia, general esearch hits (checked, not cited)
- 38537226 — platelet activation markers predict preeclampsia severity (cited)
- 40746439 — platelet activation/thrombo-inflammation in early/late-onset preeclampsia (cited)
- 12618519 — excess placental sFlt1 causes endothelial dysfunction/hypertension/proteinuria (cited)
- 41226469 — angiogenic factors in preeclampsia review (cited)
- 12965086, 32599856, 36075122 — growth factor receptor signaling + preeclampsia + VEGF/PlGF, general esearch hits (checked, not cited)
- 32973809 — Th cell profiles in pregnancy incl. Th17 (cited)
- 25664035 — dendritic cells from preeclampsia influence Th1/Th17 differentiation (cited)
- 37573650, 23837987, 25027967 — Th17 differentiation + preeclampsia, general esearch hits (checked, not cited)
- 38008034 — USP17/NF-kB/HDAC2 in preeclampsia (cited)
- 41420525, 33099123, 36374543, 37866322 — NF-kB + preeclampsia + placenta, general esearch hits (checked, not cited)
- 19164174, 41419624, 25065683, 24175856, 33335575 — interferon gamma signaling + preeclampsia, general esearch hits (considered, not cited — no sufficiently specific mechanistic paper found)
- 32161574 — NLRP3 inflammasome role in preeclampsia (cited)
- 37788097 — inflammasomes in human reproductive diseases (cited)
- 42151142, 37226086 — NLRP3/NOD-like receptor + preeclampsia, general esearch hits (checked, not cited)
- 38794175 — targeting NET formation in preeclampsia treatment (cited)
- 37958788 — NETs in health and disease pathophysiology (cited)
- 39735539, 41106578, 41811413 — NET/preeclampsia, general esearch hits (checked, not cited)
- 40076938 — gestational diabetes overview (checked, rejected — not preeclampsia-specific, wrong top hit for interleukin query)
- 40967465, 38203306, 33378046, 23782174 — interleukin signaling + preeclampsia + IL-6, general esearch hits (checked, not cited)
- 35177217 — chronic hypertension/superimposed preeclampsia (checked, rejected — not IL-6-specific)
- 31964198 — IL-6/presepsin/pentraxin-3 in late-onset preeclampsia diagnosis/severity (cited)
- 28252161 — IL-6/IL-10 polymorphisms and preeclampsia risk (cited)
- PRC2 + placenta + imprinting + preeclampsia — 0 PubMed results (checked, exclusion of PRC2 Methylates Histones And DNA confirmed)
- 40253080, 38237587, 38519450, 34512661 — cytokine signaling + preeclampsia + placenta, general esearch hits (checked, not cited — pathway rejected as redundant umbrella)
