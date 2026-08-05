# Colorectal Cancer -- Pathway Literature Review

Candidate list source: `cand_Colorectal_Cancer.txt` (34 CRC patients, KEGG+Reactome library).
[GENERIC]-flagged pathways (Cell Cycle, Immune System, RNA Pol II Transcription, Olfactory/Sensory,
mRNA Splicing/Spliceosome, Metabolism Of RNA/Proteins, Cellular Responses To Stress/Stimuli, Generic
Transcription, Infectious Disease, Disease, RNA transport, Protein Localization, Processing Of Capped
Intron-Containing Pre-mRNA) and hist_frac > 15% pathways (DNA Replication, 18%) were excluded per rule 1.

## Selected pathways

### 1. Degradation Of Beta-Catenin By Destruction Complex
n_sig=21/34, size=84, eff=4.59, hist_frac=0%
APC-AXIN-GSK3B-CK1 destruction of beta-catenin is the direct mechanistic consequence of APC loss, the single
most common (~80%) initiating mutation in colorectal tumorigenesis (Fearon-Vogelstein model) -- not a
generic cancer pathway but the CRC gatekeeper mechanism itself.
- Fearon EA, Vogelstein B, Cell, 1990 (PMID: 2188735) -- classic genetic model establishing APC/Wnt
  pathway loss as the initiating step of colorectal tumorigenesis.
- Sharifi-Rad et al., Life Sciences, 2020 (PMID: 32593708) -- review of genetic alterations in Wnt/beta-catenin
  signaling components specifically across colorectal cancer.
- Deng et al., Nature Chemical Biology, 2025 (PMID: 40240631) -- cholesterol-targeting inhibitors developed
  specifically against Wnt-beta-catenin signaling in colorectal cancer.

### 2. Degradation Of AXIN
n_sig=19/34, size=54, eff=4.85, hist_frac=0%
AXIN is the scaffold protein of the same destruction complex; CDX2-mediated transactivation of AXIN2 was
shown to suppress colon cancer cell proliferation specifically through this arm of Wnt regulation.
- Zhang et al., Cell Death & Disease, 2019 (PMID: 30631044) -- CDX2 suppresses colon cancer cell
  proliferation via transactivation of GSK-3beta and Axin2, restraining Wnt/beta-catenin signaling.
- Sharifi-Rad et al., Life Sciences, 2020 (PMID: 32593708) -- as above, includes AXIN genetic alterations.

### 3. Degradation Of DVL
n_sig=19/34, size=55, eff=4.73, hist_frac=0%
Dishevelled (DVL) is the upstream Wnt-receptor-proximal switch that inactivates the destruction complex;
two independent CRC-specific mechanisms (NKD1 mutation, tumor-specific Rac1b splice variant) converge on
DVL to drive canonical Wnt signaling in colorectal tumor cells.
- Guo et al., PLoS ONE, 2009 (PMID: 19956716) -- NKD1 mutations found in colorectal cancer alter
  Wnt/Dvl/beta-catenin signaling.
- Esufali et al. (Rac1b/dishevelled), Cancer Research, 2007 (PMID: 17363564) -- tumor-specific Rac1b splice
  variant activated by dishevelled promotes canonical Wnt signaling and decreased adhesion in colorectal
  cancer cells.

### 4. Regulation Of RUNX3 Expression And Activity
n_sig=19/34, size=54, eff=4.71, hist_frac=0%
RUNX3 is epigenetically silenced (promoter hypermethylation, EZH2-mediated) in colorectal cancer specifically,
and its expression loss correlates with disease stage and outcome -- a well-replicated CRC tumor-suppressor
mechanism, not a pan-cancer generality.
- Bae SC group, Tumour Biology, 2012 (PMID: 22274925) -- oxidative-stress-induced methylation silencing of
  RUNX3 in colorectal cancer cells.
- Carvalho et al., Carcinogenesis, 2010 (PMID: 20631058) -- EZH2 and DNA methylation jointly silence RUNX3
  in colorectal cancer.
- Soong et al., British Journal of Cancer, 2009 (PMID: 19223906) -- RUNX3 expression associated with disease
  stage and patient outcome in colorectal cancer.

### 5. Negative Regulation Of NOTCH4 Signaling
n_sig=20/34, size=53, eff=4.73, hist_frac=0%
NOTCH4 (as distinct from the broadly-studied NOTCH1) has CRC-specific immunohistochemical and prognostic
validation, plus a recent mechanistic report tying Notch3/4 knockdown to suppressed colon adenocarcinoma
progression via the tumor immune microenvironment.
- Ozawa et al./authors, International Journal of Molecular Sciences, 2023 (PMID: 37108670) -- clinical
  immunohistochemical expression of Notch4 protein in colon adenocarcinoma patients.
- 2025, Scientific Reports (PMID: 41027955) -- Notch gene receptors, including NOTCH4, as prognostic
  biomarkers in colorectal cancer.
- 2026, Cancer Medicine (PMID: 42458230) -- Notch3/4 knockdown inhibits colon adenocarcinoma progression via
  VEGFA-dependent tumor immune microenvironment remodeling.

### 6. Stabilization Of P53
n_sig=20/34, size=56, eff=4.61, hist_frac=0%
Weaker-tier inclusion (TP53 loss is broadly oncogenic across tumor types), kept because in CRC specifically
p53 stabilization/mutation is the defined, staged marker of the adenoma-to-carcinoma transition in the
canonical multistep model of colorectal tumorigenesis -- a disease-specific staging role beyond generic
"p53 matters in cancer."
- Fearon EA, Vogelstein B, Cell, 1990 (PMID: 2188735) -- p53 alterations placed as the late,
  transition-defining event from adenoma to carcinoma in the colorectal multistep genetic model.
- Song et al., Cancer Research, 2025 (PMID: 40882016) -- CLK2 regulates KEAP1/NRF2 and p53 pathways to
  suppress ferroptosis specifically in colorectal cancer (p53 stabilization axis).

### 7. Separation Of Sister Chromatids
n_sig=19/34, size=170, eff=4.93, hist_frac=0%
Chromatid-cohesion defects have been directly demonstrated as an underlying mechanism of chromosomal
instability (CIN) in human colorectal cancers specifically, distinguishing this from a generic "cell
division goes wrong in cancer" claim.
- Barber TD et al., PNAS, 2008 (PMID: 18299561) -- chromatid cohesion defects underlie chromosome
  instability specifically in human colorectal cancers (functional screen of CIN candidate genes).
- Novais et al./authors, Neoplasia, 2014 (PMID: 25246271) -- MCMBP deregulation prompts oncogenesis in
  colorectal carcinomas through chromosomal instability.

### 8. Autodegradation Of E3 Ubiquitin Ligase COP1
n_sig=19/34, size=51, eff=4.82, hist_frac=0%
Two recent CRC-specific mechanistic papers place COP1 directly in the colorectal tumor pathway -- one linking
COP1 to GSK3beta degradation that activates beta-catenin signaling (converging with pathways 1-3 above), the
other to liver metastasis and oxaliplatin resistance via LUZP1 degradation.
- Wang et al., Acta Biochimica et Biophysica Sinica, 2026 (PMID: 42057424) -- COP1-mediated GSK3beta
  degradation activates beta-catenin signaling to facilitate colorectal cancer cell proliferation.
- 2026, Experimental Hematology & Oncology (PMID: 41937206) -- multi-omics analysis of patient-derived
  organoids: COP1 promotes liver metastasis and oxaliplatin resistance in colorectal cancer via LUZP1
  degradation and MYL9 phosphorylation.

## Considered and rejected

- **Nuclear Events Mediated By NFE2L2** / **KEAP1-NFE2L2 Pathway** / **GSK3B And BTRC:CUL1-mediated-degradation
  Of NFE2L2** (candidates 28, 45, 32) -- CRC-specific reviews exist (PMID 28621229, 34067204, 40882016) but
  the NRF2/KEAP1 oxidative-stress-adaptation axis is a pan-cancer hallmark most classically defined in lung
  squamous/adenocarcinoma; the mechanism itself is not colorectal-specific despite CRC papers existing.
- **HIV Infection**, **Host Interactions Of HIV Factors**, **Vif-mediated Degradation Of APOBEC3G**,
  **Vpu Mediated Degradation Of CD4**, **SARS-CoV-2 Infection**, **SARS-CoV Infections** -- viral-pathogen
  pathways with no CRC-specific relevance; not searched further (irrelevant pathogen biology, not a case of
  premature dismissal of a plausible disease link).
- **Downstream TCR Signaling** -- searched "colorectal cancer TCR signaling tumor infiltrating lymphocytes"
  (23 PubMed hits); hits describe generic T-cell infiltration/immunotherapy relevance common to most solid
  tumors, no colorectal-specific TCR mechanism found.
- **Activation Of NF-kappaB In B Cells** -- searched "colorectal cancer NF-kB B cells tumor microenvironment"
  (only 2 PubMed hits); insufficient evidence of a disease-specific mechanism, likely a generic
  immune-pathway false positive.
- **Thermogenesis**, **Pathways of neurodegeneration**, **Complex I Biogenesis**, **Mitochondrial Protein
  Import** -- no plausible CRC-specific mechanistic link; not pursued.
- **Many APC/C cell-cycle-machinery entries** (Cdc20/Cdh1-mediated degradation cascades, Mitotic
  Anaphase/Metaphase, S/M Phase, DNA Replication-adjacent) -- generic mitotic-machinery pathways shared by
  essentially all proliferating tumors; excluded as non-disease-specific (several already [GENERIC]-flagged;
  the rest share the same generic-proliferation character).
- **DNA Replication** -- excluded per rule 1 (hist_frac=18% > 15% threshold), no exceptional disease-specific
  justification found to override.

## Raw search log

PubMed IDs retrieved during this review (cited and uncited):
2188735, 28135145, 30887153, 33374459, 30631044, 32593708, 37477088, 40240631, 19956716, 17363564,
10772417, 10410140, 22586065, 22274925, 20631058, 19223906, 18299561, 38388697, 21532624, 25246271,
17643075, 40882016, 34067204, 33493657, 28621229, 41937206, 30393117, 42057424, 41331036, 31028655,
42458230, 41027955, 40552282, 39392253, 37108670, 21160823, 15217933, 40784271, 39884085, 42329854,
41925912, 40759573.

Not independently confirmed by full-text/abstract read (title/journal metadata only, from esummary):
40552282, 39392253, 33493657, 30393117, 41331036, 31028655, 21160823, 15217933, 40784271, 39884085,
42329854, 41925912, 40759573 (NF-kB and TCR search hits, cited only as hit-count evidence for rejection,
not as positive citations).
