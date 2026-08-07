# Pancreatitis (Acute) — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Pancreatitis.txt` (n=79 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below.

This is a redo after the upstream Z-score bug fix. Overall signal in this cohort is weak (20/79 patients had zero significant pathways, path_sig_median=5), consistent with prior review. Top ~45 rows by score were examined; most are broad Reactome umbrella terms (RNA metabolism, splicing, generic immune/disease categories) correctly caught by the GENERIC filter, or composition artifacts (histone-driven "Systemic lupus erythematosus", olfactory receptor pathways from cfRNA background noise). Only a small, biologically coherent neutrophil/IL-1 axis survives rigorous vetting — consistent with the well-established central role of neutrophil activation and IL-1/NLRP3 signaling in acute pancreatitis pathophysiology.

## Selected pathways

### 1. Neutrophil Degranulation
n_sig=21/79, size=463, eff=4.60, hist_frac=0%

Neutrophil activation and degranulation are a well-documented, mechanistically central driver of acute pancreatitis (AP) severity — neutrophil-released proteases and reactive oxygen species propagate local pancreatic and remote (lung) injury, and a prior peripheral-blood transcriptome study in AP patients specifically identified an altered neutrophil-related pathway signature. This is disease-specific innate-immune effector biology, not a generic "immune system was active" hit.
- Wang et al., Biomolecules, 2023 (PMID: 36830652) — bioinformatic analysis of peripheral blood transcriptome identifies altered neutrophil-related pathway in acute pancreatitis patients.
- Osman et al., Pancreas, 2008 (PMID: 18580443) — pentoxifylline attenuates pulmonary inflammation and neutrophil activation in experimental acute pancreatitis.

### 2. Neutrophil Extracellular Trap Formation
n_sig=7/79, size=183, eff=5.43, hist_frac=39.9% — **histone exception**

Kept despite hist_frac=39.9% exceeding the 15% default cutoff: histones are not a composition artifact here but the literal effector mechanism — NETosis is defined by citrullination and extracellular release of histones (via PAD4) as part of the DNA-histone trap. A substantial, recent, mechanistically direct literature base ties NETs to acute pancreatitis severity, pancreatic injury propagation, and candidate therapeutics.
- Zhou et al., Chinese Medical Journal, 2022 (PMID: 36729096) — role of NETs in inflammatory evolution of severe acute pancreatitis.
- Merza et al., Therapeutic Advances in Gastroenterology, 2020 (PMID: 33281940) — targeting NETs in severe acute pancreatitis treatment.
- Li et al., Nature Communications, 2023 (PMID: 37794047) — gut microbiota aggravates NET-induced pancreatic injury in hypertriglyceridemic pancreatitis.
- Chen et al., Frontiers in Immunology, 2022 (PMID: 36032164) — roles, detection, and visualization of NETs in acute pancreatitis (review).
- Zhang et al., Redox Biology, 2023 (PMID: 37392517) — irisin inhibits NET formation and protects against acute pancreatitis in mice.

### 3. Interleukin-1 Signaling
n_sig=9/79, size=114, eff=3.61, hist_frac=0%

IL-1 (via NLRP3 inflammasome activation in injured acinar cells and infiltrating macrophages) is one of the best-characterized cytokine drivers of AP severity, with direct functional (IL-1 antagonism) and mechanistic (NLRP3-IL-1beta axis) evidence in experimental and human AP. Preferred over the broader, less specific "Signaling By Interleukins" and "Cytokine Signaling In Immune System" candidates per the redundancy rule — this is the specific, mechanistically anchored node within that broader signaling family.
- Norman et al., Journal of Surgical Research, 1997 (PMID: 9070189) — acute pancreatitis-induced enzyme release and necrosis attenuated by IL-1 antagonism.
- Sendler & Mayerle, International Journal of Molecular Sciences, 2020 (PMID: 32751171) — NLRP3 inflammasome-mediated inflammation in acute pancreatitis (IL-1beta axis review).
- Norman, Gastroenterology, 1996 (PMID: 8566616) — cytokines and acute pancreatitis (classic review establishing IL-1/TNF/IL-6 cascade in AP severity).

## Considered and rejected

- **Signaling By Interleukins** (n_sig=9, size=449, eff=3.95, hist_frac=0.4%). Rejected as redundant — broader umbrella covering the same IL-1 mechanism already captured, more specifically and with more direct AP evidence, by "Interleukin-1 Signaling" above.
- **Cytokine Signaling In Immune System** (n_sig=12, size=693, eff=4.45, hist_frac=0.3%). Rejected as redundant/too broad — the AP-relevant mechanistic literature (IL-1/NLRP3, TNF, IL-6) is captured more specifically by "Interleukin-1 Signaling"; this 693-gene pathway spans dozens of unrelated cytokine axes with no AP-specific evidence beyond the IL-1 subset.
- **Extracellular Matrix Organization** (n_sig=8, size=287, eff=4.40, hist_frac=0%) and **ECM-receptor interaction** (n_sig=7, size=87, eff=4.70, hist_frac=0%). Rejected — literature search found ECM remodeling/TGFbeta-ECM biology tied almost exclusively to chronic pancreatitis and pancreatic fibrosis (PMID 10576340, 36508773, 36830471), not acute-phase injury in this cohort; no acute-pancreatitis-specific mechanistic paper for ECM organization was retrieved.
- **Systemic lupus erythematosus** (n_sig=12, size=127, eff=5.23, hist_frac=57.5%). Rejected — hist_frac far exceeds cutoff and this KEGG disease-gene-set is dominated by core histone genes used as SLE autoantigens; no genuine SLE/lupus mechanistic connection to AP, this is a composition artifact, not a kept exception.
- **Downstream TCR Signaling** (n_sig=11, size=92, eff=3.92, hist_frac=0%) and **TCR Signaling** (n_sig=11, size=114, eff=3.90, hist_frac=0%). Rejected — no AP-specific mechanistic literature found; T-cell receptor signaling is not an established driver of AP pathophysiology (AP is predominantly an innate/neutrophil-macrophage-driven disease), plausible peripheral-blood lymphocyte-composition artifact.
- **Diabetic cardiomyopathy** (n_sig=10, size=189, eff=4.04, hist_frac=0%). Rejected — no direct AP mechanistic link; likely reflects nonspecific metabolic/cardiac stress gene overlap, not disease-specific.
- **Signaling By Rho GTPases** (n_sig=8, size=640, eff=4.71, hist_frac=4.4%) / **Signaling By Rho GTPases, Miro GTPases And RHOBTB3** (n_sig=7, size=656, eff=4.80, hist_frac=4.3%). Rejected — generic cytoskeletal signaling active in essentially any inflammatory/leukocyte-activation state; no AP-specific mechanistic paper found, and the two entries are near-duplicates of each other.
- **Olfactory Signaling Pathway / Olfactory transduction / Expression And Translocation Of Olfactory Receptors / Sensory Perception**. Rejected — no plausible AP biology; classic cfRNA background/technical artifact category, not investigated further.
- **Vif-mediated Degradation Of APOBEC3G, Autodegradation Of E3 Ubiquitin Ligase COP1, Stabilization Of P53, Activation Of NF-kappaB In B Cells, Vpu Mediated Degradation Of CD4, Hh Mutants Are Degraded By ERAD, SARS-CoV Infections, Signaling By B Cell Receptor (BCR)**. Rejected without individual PubMed search — small, low n_sig (9 or fewer), narrow proteostasis/viral-restriction/B-cell pathways with no plausible AP mechanism and lower scores than the selected set; screened by biological plausibility only, not pursued further given the conservative scope of this review.
- All [GENERIC]-flagged rows (Metabolism Of RNA, Cellular Responses To Stress/Stimuli, Infectious Disease, mRNA Splicing [Major Pathway], Innate Immune System, Immune System, Metabolism Of Proteins, Spliceosome, Disease, Gene Expression (Transcription), RNA Polymerase II Transcription, Generic Transcription Pathway, Cell Cycle / Cell Cycle Mitotic, RHO GTPase Cycle) were excluded per the standing project rule as broad umbrella terms that light up for almost any active biology.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 36830652 — bioinformatic peripheral blood neutrophil pathway in AP (cited)
- 18580443 — pentoxifylline/neutrophil activation experimental AP (cited)
- 36729096 — NETs in inflammatory evolution of severe AP (cited)
- 33281940 — targeting NETs in severe AP treatment (cited)
- 37794047 — gut microbiota/NETs/pancreatic injury, Nature Communications (cited)
- 36032164 — NETs in AP review, Frontiers in Immunology (cited)
- 37392517 — irisin inhibits NET formation, protects AP (cited)
- 9070189 — IL-1 antagonism attenuates AP enzyme release/necrosis (cited)
- 32751171 — NLRP3 inflammasome-mediated inflammation in AP (cited)
- 8566616 — cytokines and acute pancreatitis, Gastroenterology 1996 classic review (cited)
- 40859413 — Src reduces NET generation, resolves organ damage (checked, not cited — supportive but not needed beyond the 5 already cited)
- 10576340 — TGFbeta and ECM in pancreatitis (checked, rejected — chronic pancreatitis focus)
- 36508773 — remodeling of imbalanced ECM homeostasis, pancreatic fibrosis (checked, rejected — chronic/fibrosis focus)
- 36830471 — canine pancreatic ECM in diabetes/pancreatitis (checked, rejected — not human AP-specific)
- 41208148 — epidemiology/pathogenesis review of AP, mentions ECM broadly (checked, considered, not cited as primary evidence)
- 37563309 — resident macrophage fibrosis, pancreatic injury vs PDAC, Nature Immunology (checked, considered, not cited — PDAC/fibrosis focus, not acute)
- 40577966 — MMP9 inhibition/NETs/necroptosis in severe AP (checked, not cited — supports NET selection indirectly, not primary)
- 41045968 — fibrotic collagen-targeted delivery, pancreatic fibrosis (checked, rejected — chronic fibrosis, off-target)
- 41329112 — alcohol in pancreatic diseases review (checked, rejected — not pathway-specific)
- 24657625, 31856083, 32011822 — general AP severity/genetics papers surfaced by IL-1 search, not specific enough to IL-1 mechanism (checked, not cited)
- 37098965, 40064441, 30482771, 19009638 — off-target hits from initial broad IL-1/pathogenesis query (SRSF1/pancreatic cancer, Mendelian randomization, GVHD, general AP prediction review) — checked, rejected as not on-topic
