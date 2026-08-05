# Pancreatitis Pathway Literature Review

Cohort: chronic/acute pancreatitis patients (n=79), distinct from the Pancreatic Cancer cohort
analyzed elsewhere in this project. Candidates drawn from
`cand_Pancreatitis.txt` (per-patient/per-pathway permutation-null mean-Z convergence).
[GENERIC]-flagged umbrella terms and hist_frac>15% pathways excluded per rules unless a
disease-specific override is documented below.

## Selected pathways

### 1. Neutrophil Degranulation
n_sig=21/79, size=463, eff=4.71, hist_frac=0%

Neutrophil activation/degranulation is a core, mechanistically specific driver of acute
pancreatitis severity (not a generic infection signature) — neutrophil-derived proteases and
NETs directly cause trypsinogen activation and duct/acinar injury, and a peripheral-blood
transcriptome study found this exact pathway altered specifically in acute pancreatitis patients.
- Zhou et al., Biomolecules, 2023 (PMID: 36830652) — bioinformatic peripheral-blood transcriptome analysis found an altered neutrophil-related pathway specifically discriminating acute pancreatitis patients (with/without chylomicronemia).
- Merza et al., Gastroenterology, 2015 (PMID: 26302488) — NETs induce trypsin activation, inflammation, and tissue damage in mice with severe acute pancreatitis (seminal mechanistic study).

### 2. Neutrophil Extracellular Trap Formation
n_sig=7/79, size=183, eff=5.54, hist_frac=40% — **kept despite hist_frac>15%, weaker-tier**

Override rationale: unlike the SLE/histone-autoantigen case, the histone content here is not an
autoantigen artifact — extracellular histones released during NETosis are themselves a
proximal, mechanistically established mediator of pancreatic acinar injury and severity, so the
histone-gene enrichment is disease-mechanism signal, not composition bias. Confirmed by an
independent, directly-on-topic literature base (below).
- Merza et al., Gastroenterology, 2015 (PMID: 26302488) — NETs induce trypsin activation and tissue damage in severe acute pancreatitis (mouse model).
- Wang et al., Frontiers in Immunology, 2022 (PMID: 36032164) — review of NET roles, detection, and visualization specifically in acute pancreatitis.
- Zhou et al., Nature Communications, 2023 (PMID: 37794047) — gut microbiota aggravates NET-induced pancreatic injury in hypertriglyceridemic pancreatitis.
- Liu et al., Redox Biology, 2023 (PMID: 37392517) — irisin inhibits NET formation and protects against acute pancreatitis in mice.

### 3. Extracellular Matrix Organization
n_sig=7/79, size=287, eff=4.56, hist_frac=0%

Pancreatic stellate cell activation and ECM deposition is the defining histopathological
mechanism of chronic pancreatitis fibrosis, distinguishing it from acute-phase-only inflammation.
- Xue et al., Biomedicines, 2024 (PMID: 38255213) — review of pancreatic stellate cell activation/regulation in chronic pancreatic fibrosis.
- Shimizu, Journal of Gastroenterology, 2008 (PMID: 19012035) — mechanisms of pancreatic fibrosis and treatment implications for chronic pancreatitis.
- Omary et al., Clinical Gastroenterology and Hepatology, 2009 (PMID: 19896099) — pancreatic stellate cells in pancreatic inflammation and fibrosis.

### 4. Defective CFTR Causes Cystic Fibrosis
n_sig=7/79, size=60, eff=3.57, hist_frac=0%

CFTR dysfunction is a well-replicated genetic cause of idiopathic/chronic pancreatitis
independent of overt cystic fibrosis lung disease — this is disease-specific genetic
epidemiology, not a generic pathway hit.
- Conwell et al. (ACG guideline), JAMA, 2019 (PMID: 31860051) — CFTR genotype among established genetic risk factors in chronic pancreatitis diagnosis/management review.
- Schneider et al., Human Mutation, 2011 (PMID: 21520337) — common CFTR haplotypes and susceptibility to chronic pancreatitis.
- Masamune et al., Clinical and Translational Gastroenterology, 2018 (PMID: 30420730) — SPINK1/PRSS1/CTRC/CFTR genotypes influence chronic pancreatitis onset and outcomes.
- Rosendahl et al., PLoS One, 2013 (PMID: 23951356) — CFTR among major genetic causes of idiopathic chronic pancreatitis in 253 young French patients.

### 5. Interleukin-1 Signaling
n_sig=7/79, size=114, eff=3.73, hist_frac=0%

The NLRP3/AIM2 inflammasome-to-IL-1beta axis is a specific, functionally validated driver of
acute pancreatitis severity, distinct from generic cytokine signaling.
- Hoque et al., Gastroenterology, 2011 (PMID: 21439959) — TLR9 and NLRP3 inflammasome link acinar cell death with inflammation in acute pancreatitis.
- Fu et al., Pancreatology, 2017 (PMID: 28342645) — AIM2 inflammasome expression/activation correlates with severity in acute pancreatitis patients.
- Tan et al., Translational Research, 2014 (PMID: 25152324) — NLRP3 inflammasome inhibition reduces severity of experimental acute pancreatitis in obese mice.
- Kang et al., Life Sciences, 2021 (PMID: 33600865) — HMGB1 induces acute pancreatitis via NET activation and subsequent IL-1beta production (links entries 2 and 5 mechanistically).

### 6. Signaling By B Cell Receptor (BCR)
n_sig=11/79, size=109, eff=3.79, hist_frac=0%

IgG4-related (type 1) autoimmune pancreatitis is a well-characterized pancreatitis subtype with
a documented B-cell/plasmablast-driven, BCR-clone-specific pathogenesis, distinguishing it from
generic immune activation.
- Maillette de Buy Wenniger et al., Current Opinion in Gastroenterology, 2017 (PMID: 28509786) — clinical/experimental advances in IgG4-related disease of the biliary tract and pancreas.
- Wang et al., Hepatology, 2016 (PMID: 27015613) — IgG4+ B-cell receptor clones distinguish IgG4-related disease from primary sclerosing cholangitis and biliary/pancreatic malignancy.
- Hubers et al., Gut, 2018 (PMID: 28765476) — Annexin A11 targeted by IgG4/IgG1 autoantibodies in IgG4-related disease.

### 7. RHO GTPase Cycle
n_sig=9/79, size=439, eff=4.01, hist_frac=0%

Rho-family GTPase (Rac1) signaling is functionally required for neutrophil-mediated pancreatic
and pulmonary injury in severe acute pancreatitis, shown by direct pharmacologic/genetic
inhibition studies — a specific mechanistic link, not a generic cytoskeletal-signaling hit.
- Sundqvist et al., Journal of Leukocyte Biology, 2013 (PMID: 23744643) — geranylgeranyltransferase inhibition attenuates neutrophil accumulation and tissue injury in severe acute pancreatitis.
- Sundqvist et al., Experimental Physiology, 2008 (PMID: 18567599) — Rac1 inhibition decreases severity of pancreatitis and pancreatitis-associated lung injury in mice.

## Considered and rejected

- **Systemic lupus erythematosus** (n_sig=12, eff=5.32, hist_frac=57%): "lupus pancreatitis" is a
  real, documented rare clinical entity (Ramírez-Piqueras et al., Lupus, 2021, PMID 33307986;
  Rev Esp Enferm Dig, 2009, PMID 19785498), but the pathway's histone-gene dominance reflects
  anti-histone/anti-nucleosome autoantibody biology central to SLE serology generally, not a
  pancreatitis-specific mechanism — same composition-bias reasoning that excluded this term in
  the Pancreatic Cancer analysis. Excluded.
- **Hedgehog Ligand Biogenesis / Hh Mutants Are Degraded By ERAD / Hh Mutants Abrogate Ligand
  Secretion**: literature search returned exclusively cancer-context Hedgehog signaling reviews
  (PMID 29274272, 36424360, 32957513, 31125907, 33494284) with no pancreatitis-specific hits.
  Excluded — no disease-specific case found.
- **Diabetic cardiomyopathy** (n_sig=11, eff=3.97): post-pancreatitis (type 3c) diabetes is real
  and epidemiologically elevated after acute pancreatitis (Bharmal et al., BJS Open, 2022, PMID
  36515672) and chronic pancreatitis carries excess cardiovascular risk (Rasch et al., World J
  Gastroenterol, 2019, PMID 31802835), but no literature ties the specific
  "diabetic cardiomyopathy" gene program (cardiac mitochondrial/RAAS/fibrosis genes) mechanistically
  to pancreatitis rather than to diabetes generally. Excluded as insufficiently disease-specific.
- **T Cell Receptor / Downstream TCR Signaling, Antigen processing-Cross Presentation**: targeted
  searches for autoimmune-pancreatitis-specific TCR or cross-presentation mechanisms returned only
  off-target hits (NOD mouse type 1 diabetes Tregs, PMID 16126946; PTPN2/Tfh autoimmunity, PMID
  27658548) — no pancreatitis-specific literature found. Excluded.
- **SARS-CoV-2 Infection / SARS-CoV-2 infections / Herpes simplex virus 1 infection**: an
  acute-pancreatitis-and-COVID-19 literature exists (e.g. PMID 36185634, 35475200) but the
  association is etiological/incidental (any number of viruses can trigger pancreatitis) rather
  than a specific shared mechanism, matching the task's instruction to exclude generic
  infection associations. Excluded.
- **Alcoholism** (n_sig=5, eff=4.94, hist_frac=41%): alcohol is a leading epidemiological cause
  of pancreatitis, but the KEGG "Alcoholism" pathway's gene content (dopamine/glutamate/CREB
  synaptic-plasticity and chromatin genes for addiction neurobiology) does not correspond to
  alcohol-pancreas injury mechanisms (ADH/ALDH/CYP2E1 metabolism, oxidative stress) — the name
  match is coincidental, not mechanistic, and the composition is histone-dominated. Excluded.
- **ECM-receptor interaction** (KEGG, n_sig=6, eff=4.95, hist_frac=0%): same underlying biology
  as Extracellular Matrix Organization (#3, stellate-cell fibrosis) — not added as a separate
  entry to avoid double-counting one mechanism, but supports #3's rationale.

## Raw search log

PMIDs retrieved during research (including non-cited/off-target hits, for cross-verification):

- Neutrophil degranulation / NETs: 36729096, 33281940, 18580443, 36830652, 26302488, 37392517, 37794047, 39025845, 37491321, 36032164
- ECM / stellate cells: 41208148, 38255213, 19012035, 19896099, 40959561
- CFTR / chronic pancreatitis genetics: 31860051, 21520337, 30420730, 35084992, 23951356
- Hedgehog signaling (all off-target/cancer-only): 29274272, 36424360, 32957513, 31125907, 33494284
- Type 3c / pancreatogenic diabetes: 36515672, 31802835, 29185012, 41675641
- IL-1 / inflammasome: 24657625, 33600865, 21439959, 28342645, 25152324
- IgG4-related autoimmune pancreatitis / BCR: 25034294, 28765476, 28509786, 27015613, 26817943
- Systemic lupus erythematosus + pancreatitis: 33307986, 19785498 (plus off-target general-SLE hits: 39339212, 40418946, 34179742)
- COVID-19 / SARS-CoV-2 + acute pancreatitis: 36185634, 35475200, 35381217, 35195163, 37436286
- TCR / antigen cross-presentation (off-target): 16126946, 27658548
- Rho GTPase / Rac1: 23744643, 18567599
- Alcoholism (off-target, KEGG name-match only): 16878254
