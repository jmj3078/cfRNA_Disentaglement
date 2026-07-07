# 대조군 비교 분석 (Comparative Signatures) — skill-retrieved references

Retrieval: PubMed E-utilities (esearch) + Open Targets Platform GraphQL v4. Date 2026-07-07.
Method: with_rare GSEA significant terms (FDR<0.05, housekeeping/neurodegeneration excluded) set-differenced between phenotype pairs; only terms unique to one side reported.

## Pancreatic Cancer (Moore) vs Pancreatitis (Moore)

- DDR2 AND pancreatic cancer: PubMed 12 hits. ECM organization/desmoplasia general (extracellular matrix AND pancreatic ductal adenocarcinoma AND desmoplasia): 189 hits. Established PDAC stromal biology.
- acute pancreatitis AND vascular permeability: PubMed 219 hits. Established acute-phase edema mechanism (VEGFR2-mediated).
- pancreatic acinar cell cAMP secretion pancreatitis: PubMed 157 hits. Established exocrine secretory signaling axis.
- Immune-evasion term (Primary Immunodeficiency / ZAP-70 synapse, down in cancer only): lead genes LCK, PTPRC, BTK — canonical T-cell receptor proximal signaling; direction (down) consistent with tumor immune evasion literature (qualitative, not independently PMID-verified in this pass).

## ICI-m (Raissadati) vs ICI-treated Cancer (Raissadati)

- MYH7/MYH6 (Cardiac Muscle Contraction, NES +2.28, ICI-m only):
  - Open Targets MYH7 x hypertrophic cardiomyopathy 0.892, cardiomyopathy 0.758.
  - PMID 36385524 — "T cells specific for α-myosin drive immunotherapy-related myocarditis." (landmark mechanistic study; myosin-specific T cells directly implicated)
  - PMID 39378095 — "Injury-induced myosin-specific tissue-resident memory T cells drive immune checkpoint inhibitor myocarditis." (2024 follow-up)
- Immunoproteasome / antigen cross-presentation (Proteasome, ER-Phagosome, Cross-presentation Of Soluble Exogenous Antigens; NES +2.16 to +2.38, lead PSMB8 among others, ICI-m only):
  - PubMed PSMB8 AND myocarditis: 5 hits.
  - PubMed immunoproteasome AND myocarditis: 13 hits. Representative: PMID 38570171 — "Mapping the interplay of immunoproteasome and autophagy in different heart failure phenotypes." (2024)
  - Mechanistic coherence (not independently confirmed as a single paper): immunoproteasome/IFN-gamma-driven antigen presentation is the plausible upstream route feeding myosin-specific T-cell priming reported above.
- ICI-treated Cancer contrast: Allograft rejection / GVHD-like terms and large-scale mitotic/cell-cycle downregulation (45 terms) appear only in this heterogeneous mixed cohort with no single Open Targets disease reference (ot_disease=None per earlier packet) — interpreted as non-specific tumor-bulk signal, not a coherent mechanism.

## CAD_HF+ (Ward) vs CAD_HF- (Ward)

- COL6A3 (Collagen Chain Trimerization, NES +2.40, HF+ only): Open Targets coronary artery disorder 0.445, myocardial infarction 0.339. PubMed COL6A3 AND (cardiac/coronary/fibrosis): ~87 hits. Representative PMID 41174767 — "Spatial-reprogramming derived GPNMB+ macrophages interact with COL6A3+ fibroblasts to enhance vascular fibrosis." (Genome Med 2025)
- FGF5 (FGFR1/FGFR2 ligand binding, NES +2.15/+2.21/+2.28, HF+ only): Open Targets coronary artery disorder 0.548, hypertensive disorder 0.584, heart failure 0.473.
- AGT (G Protein-Coupled Receptor Signaling Coupled To Cyclic Nucleotide, NES +3.76, HF- only): Open Targets essential hypertension 0.627, hypertensive disorder 0.606, cardiovascular disorder 0.487 (queried live via Open Targets GraphQL, 2026-07-07).
- P2RY12 (P2Y Receptors, NES +2.12, HF- only): Open Targets coronary artery disorder 0.602, acute coronary syndrome 0.613, myocardial infarction 0.620, Stroke 0.609 (queried live via Open Targets GraphQL, 2026-07-07). Established antiplatelet drug target (clopidogrel/ticagrelor).
- PubMed AGT AND coronary artery disease: 63 hits. PubMed P2RY12 AND coronary artery disease: 151 hits.

All scores/hit counts from direct API calls in this session; no memory-based citation.
