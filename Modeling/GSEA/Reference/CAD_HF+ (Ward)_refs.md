# CAD_HF+ (Ward) — Reference backbone (skill-retrieved only)

Phenotype: coronary artery disorder, heart-failure progressor. OT disease = coronary artery disorder.
Sources: PubMed (E-utilities esearch/esummary), Open Targets Platform GraphQL v4.
Note: rare-led = 0 (rare branch not in any leading edge; effect is indirect ranking shift only).

## FGF5 (ENSG00000138675) — FGFR2/FGFRL1 ligand binding (novel; NES 2.21 / 2.15)
- Open Targets (target->associatedDiseases): coronary artery disorder **0.548**; hypertensive disorder 0.584; essential hypertension 0.538; heart failure 0.473; myocardial infarction 0.469. Evidence dominated by genetic/literature (GWAS BP/CAD locus).
- PubMed FGF5 AND (blood pressure/hypertension/coronary): count ~66. Representative:
  - PMID 41065563 — "Blood pressure, plasma proteins, and cardiovascular diseases: a network Mendelian randomization..." (Eur Heart J 2026)
  - PMID 38487880 — "Identification of Circulating Plasma Proteins as a Mediator of Hypertension-Driven Cardiac Remodeling..." (Hypertension 2024)
- PubMed FGF5 AND heart failure: count 6 (sparse). HF-specific link = conservative candidate, not established.

## COL6A3 (ENSG00000163359) — Collagen biosynthesis / chain trimerization (novel NES 2.04; established NES 2.40)
- Open Targets: coronary artery disorder **0.445**; myocardial infarction 0.339. (Top two CV associations for this gene.)
- PubMed COL6A3 AND (cardiac/coronary/fibrosis): count ~87. Representative:
  - PMID 41174767 — "Spatial-reprogramming derived GPNMB+ macrophages interact with COL6A3+ fibroblasts to enhance vascular fibrosis" (Genome Med 2025)
  - PMID 39895541 — "Proteome-Wide Mendelian Randomization Identifies Therapeutic Targets for Abdominal Aortic Aneurysm" (J Am Heart Assoc 2025)
- ECM/collagen axis = established vascular/cardiac fibrosis biology; relatively HF+ specific (absent from HF- top set).

## ADRB2 (ENSG00000169252) — Amine ligand-binding receptors (established NES 2.33)
- Open Targets: coronary artery disorder **0.278** (moderate); stronger CV links: heart failure 0.604, congestive heart failure 0.599, myocardial infarction 0.612. (Packet db_support 0.51 not reproduced for coronary artery disorder in current OT; verified scores used instead.)
- PubMed ADRB2 AND heart failure: count 62. Representative:
  - PMID 34642472 — "Genetic polymorphisms in ADRB2 and ADRB1 are associated with differential survival in heart failure patients..." (Pharmacogenomics J 2022)
  - PMID 38951961 — "CPIC Guideline for CYP2D6, ADRB1, ADRB2, ADRA2C, GRK4... (beta-blocker pharmacogenetics)" (Clin Pharmacol Ther 2024)
- beta-2 adrenergic axis = established heart-failure biology.

## PLXND1 (ENSG00000004399) — Semaphorin-Plexin signaling (established, db_support)
- Open Targets: coronary artery disorder **0.366**; otherwise dominated by congenital heart defect terms (multiple types), sudden cardiac arrest 0.067, spontaneous coronary artery dissection 0.061. Direct CAD evidence weak.
- PubMed PLXND1 AND (vascular/angiogenesis/atherosclerosis): count ~76. Representative:
  - PMID 32025034 — "The guidance receptor plexin D1 is a mechanosensor in endothelial cells" (Nature 2020)
  - PMID 38328196 — "Plxnd1-mediated mechanosensing of blood flow controls the caliber of the Dorsal Aorta..." (bioRxiv 2024)
- Endothelial mechanosensing relevant, but CAD-specific causal evidence = conservative candidate.

## Aminoglycan Biosynthetic Process (GO:0006023) — novel NES 2.18
- No leading-edge DB support (n_db=0). No gene-level OT/PubMed CAD link retrieved. Marked conservative novel candidate (possible vascular ECM/GAG relevance, unverified).

## Neuronal/sensory novel terms (Sensory Perception Of Pain GO:0019233; L1-Ankyrins R-HSA-445095)
- No DB support retrieved (n_db=0). Flagged as low-expression / gene-family residual-variance artifact candidates; no CAD literature asserted.
