# CAD_HF- (Ward) — Reference backbone (skill-retrieved only)

Phenotype: coronary artery disorder, no heart-failure progression. OT disease = coronary artery disorder.
Sources: PubMed (E-utilities esearch/esummary), Open Targets Platform GraphQL v4.
Note: rare-led = 0 (rare branch not in any leading edge; effect is indirect ranking shift only).

## AGT (ENSG00000135744) — GPCR / cyclic-nucleotide 2nd-messenger signaling (novel; NES 3.76, strongest term overall)
- Open Targets: coronary artery disorder **0.387**; essential hypertension 0.627; hypertensive disorder 0.606; cardiovascular disorder 0.487; congestive heart failure 0.108; heart failure 0.103. Evidence = genetic/literature (RAAS core gene).
- PubMed AGT/angiotensinogen AND (heart failure/coronary): count ~290. Representative:
  - PMID 22842872 — "Angiotensinogen gene M235T polymorphism and risk of coronary artery disease: a meta-analysis" (Mol Med Rep 2012)
  - PMID 36042680 — "The effect of polymorphisms (M235T and T174M) on the angiotensinogen gene (AGT) in coronary artery disease" (Medicine 2022)
  - PMID 23154270 — "The M235T polymorphism in the angiotensinogen gene and heart failure: a meta-analysis" (J Renin Angiotensin Aldosterone Syst 2014)
- RAAS / GPCR 2nd-messenger axis = established CAD/hypertension biology; strongest novel signal here.

## P2RY12 (ENSG00000169313) — P2Y / purinergic receptors (established; NES 2.12 / 2.14)
- Open Targets: coronary artery disorder **0.602**; myocardial infarction 0.62; acute coronary syndrome 0.613; peripheral vascular disease 0.598; myocardial ischemia 0.506. Strong genetic + drug (ChEMBL/clopidogrel) evidence.
- PubMed P2RY12 AND (coronary/antiplatelet/clopidogrel): count ~1000 (clopidogrel subset ~575). Representative:
  - PMID 32160082 — "The P2RY12 receptor promotes VSMC-derived foam cell formation by inhibiting autophagy in advanced atherosclerosis" (Autophagy 2021)
  - PMID 20008209 — "Antiplatelet agents" (Hematology Am Soc Hematol Educ Program 2009)
- Platelet/thrombosis axis = established CAD biology (drug target of clopidogrel/prasugrel).

## PLCG2 (Coronavirus disease KEGG term; NES -1.84, downregulated)
- Open Targets: coronary artery disorder association (db_support 0.41 per packet). Direction downregulated within a viral/immune term; no CAD-specific interpretation asserted.

## PTPN11 (Signaling By CSF3 R-HSA-9674555; NES -1.93, downregulated), SF3A3 (mRNA splicing; NES -1.90), FER/PLCG2 (Fc-epsilon receptor)
- db_support present (OT ~0.38-0.46 per packet) but all downregulated within translation/splicing/immune background axes shared with HF+; treated as non-specific, no dedicated PubMed claim made.

## Neurotransmitter novel terms (Nicotine addiction KEGG NES 2.41; Regulation Of Dopamine Secretion GO:0014059; Acetylcholine Binding R-HSA-181431)
- No leading-edge DB support (n_db=0). Surface plausibility via smoking as CAD risk factor, but no gene-level OT/PubMed CAD evidence retrieved. Flagged as low-expression neuronal gene-family residual-variance artifact candidates; conservative novel only.
