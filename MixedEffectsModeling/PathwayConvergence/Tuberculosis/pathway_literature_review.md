# Tuberculosis — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Tuberculosis.txt` (n=101 TB patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below.

## Selected pathways

### 1. Interferon Signaling
n_sig=53/101, size=197, eff=5.29, hist_frac=0%

The blood interferon-inducible transcriptional signature is the single most replicated systemic finding in human active TB, discovered in whole blood and shown to normalize with treatment and correlate with radiographic extent — not a generic "immune system active" signal but a specific, mechanistically dissected type I/II IFN axis implicated in TB immunopathology.
- Berry MP et al., Nature, 2010 (PMID: 20725040) — interferon-inducible neutrophil-driven blood transcriptional signature discriminates active TB from latent infection/other diseases.
- Ivashkiv LB, Donlin LT, Nat Rev Immunol, 2015 (PMID: 25614319) — mechanistic review of type I IFN in infectious disease, including detrimental role in TB.

### 2. Neutrophil extracellular trap formation
n_sig=61/101, size=183, eff=5.80, hist_frac=39.9%

**Histone exception**: hist_frac is high because NET formation is mechanistically histone-dependent (citrullinated histones form the NET scaffold) — this is not a composition artifact riding on an unrelated pathway, it is the core biology. NETs have been directly visualized within caseating TB granulomas and NET-associated genes are detectable in blood months before TB diagnosis, supporting a genuine mechanistic link rather than generic neutrophil activation.
- Braian C et al. (see also) — NETosis characterized in caseating granulomas, Cell Death Dis, 2024 (PMID: 39085192) — NETs characterize caseating granulomas in human/model TB lesions.
- Roe K et al./Scriba group — Neutrophil degranulation, NETosis and platelet degranulation pathway genes co-induced in whole blood up to six months before TB diagnosis, PLoS One, 2022 (PMID: 36454773).

### 3. Bacterial invasion of epithelial cells
n_sig=67/101, size=77, eff=4.49, hist_frac=0%

Alveolar epithelial cells are an early, non-canonical entry and dissemination route for Mycobacterium tuberculosis (via receptors distinct from macrophage phagocytosis), making this pathway mechanistically specific to the initial host-pathogen interface rather than a generic "infection happened" signal.
- Ryndak MB, Laal S, Front Cell Infect Microbiol, 2019 (PMID: 31497538) — critical role of alveolar epithelial cells in Mtb primary infection and dissemination.

### 4. Neutrophil Degranulation
n_sig=46/101, size=463, eff=5.74, hist_frac=0%

Neutrophil granule-protein release is a core, well-characterized component of the TB blood transcriptional signature and precedes clinical diagnosis, distinct from broad "innate immune system" annotation — the same prospective cohort study links degranulation, NETosis, and platelet-degranulation gene modules specifically to incident TB.
- Roe K et al., PLoS One, 2022 (PMID: 36454773) — neutrophil degranulation genes co-induced in whole blood up to 6 months pre-diagnosis in progressors to TB.

### 5. Platelet Activation, Signaling And Aggregation
n_sig=51/101, size=252, eff=4.49, hist_frac=0%

Platelets are increasingly recognized as active participants in TB immunopathology (not incidental bystanders), contributing to granuloma formation and correlating with disease severity; a dedicated review synthesizes platelet-activation mechanisms specific to TB.
- Kroon EE et al., Front Immunol, 2021 (PMID: 34093524) — platelet activation and the immune response to tuberculosis, mechanistic review.

### 6. RHO GTPases Activate WASPs And WAVEs [GENERIC override]
n_sig=57/101, size=36, eff=3.91, hist_frac=0%

Flagged GENERIC by the umbrella-regex (matches broad "Signaling By..." pattern family), but this specific narrow node — actin cytoskeletal remodeling downstream of RAC1/WASP-WAVE — has direct, mechanistic TB literature: RAC1-dependent cytoskeletal remodeling is specifically required for phagocytic/macrophage resistance to Mtb, distinct from generic Rho-GTPase signaling annotations.
- Wang C et al., mBio, 2024 (PMID: 39287444) — SIRT7 remodels the cytoskeleton via RAC1 to enhance host resistance to Mycobacterium tuberculosis.
- Fort L et al., Nat Microbiol, 2019 (PMID: 31285585) — CYRI/FAM49B negatively regulates RAC1-driven cytoskeletal remodelling and protects against bacterial infection (includes mycobacterial models).

## Considered and rejected

- **Fc gamma R-mediated phagocytosis** (n_sig=63, size=96, eff=4.37, hist_frac=0%) / **FCGR3A-mediated Phagocytosis** (n_sig=60, size=59, eff=4.16) / **Fcgamma Receptor (FCGR) Dependent Phagocytosis** (n_sig=57, size=86, eff=4.27) — three near-duplicate Reactome entries for the same Fcγ-receptor phagocytosis mechanism. Literature search found only classical/general macrophage phagocytosis papers (PMID 10358769, 2108212) and complement-receptor-mediated (not FcγR-specific) Mtb uptake — no TB-specific paper isolating FcγR-mediated (antibody-opsonized) phagocytosis as mechanistically distinct was found with sufficient specificity to justify inclusion over the stronger, better-evidenced candidates above. Rejected as insufficiently TB-specific at this granularity; not selected as redundant duplicates either since none individually cleared the bar.
- **Regulation Of Actin Dynamics For Phagocytic Cup Formation** (n_sig=63, size=61, eff=4.26, hist_frac=0%) — mechanistically plausible (phagocytic cup formation during Mtb uptake) but no TB-specific primary paper isolating this exact Reactome node was retrieved (search returned one unrelated hit, PMID 33500344); redundant in mechanism with the RAC1/WASP-WAVE pathway (#6) which had a direct hit, so not double-counted.
- **Focal adhesion** (n_sig=60, size=199, eff=4.14, hist_frac=0%) — TB-specific paper exists (Mtb exploits focal adhesion kinase to induce necrotic cell death, PMID 34745115) but this is a single mechanistic paper on one kinase within a large generic pathway (199 genes) shared across virtually all adherent-cell biology; considered borderline, deprioritized in favor of stronger, higher-eff candidates within the ~5-8 slot budget.
- **Shigellosis** (n_sig=64, size=243, eff=4.37) and **Yersinia infection** (n_sig=55, size=136, eff=4.09) — cross-pathogen invasion pathway annotations (KEGG pathogen-specific gene sets) that substantially overlap in gene content with generic cytoskeleton/vesicle-trafficking machinery; no TB-specific paper was sought since these are literally other-organism pathway labels and their significance in a TB cohort most likely reflects shared host-cytoskeletal-response gene content, not Shigella/Yersinia biology — excluded as cross-pathogen composition overlap.
- **Systemic lupus erythematosus** (n_sig=48, size=127, eff=5.86, hist_frac=57.5%) — excluded per standing project rule; high hist_frac here reflects histones as SLE autoantigens (an SLE-specific mechanism), which is unrelated to why this pathway would appear in a TB cohort, so it does not qualify for the histone exception.
- **Viral carcinogenesis** (n_sig=59, size=197, eff=4.75, hist_frac=16.8%) — hist_frac exceeds 15% threshold from unrelated chromatin-modifier gene content in this large oncogenesis-viral pathway; no TB-specific mechanistic link found, and marginal excess is not a genuine chromatin-mechanism case (unlike NETosis) — excluded.
- **Alcoholism** (n_sig=47, size=180, eff=5.75, hist_frac=40.6%) — a KEGG pathway dominated by histone/chromatin genes for an unrelated reason (transcriptional/epigenetic effects of alcohol); no TB-specific mechanism, excluded.
- All [GENERIC]-flagged rows other than the one explicit override above (#6) — including Cell Cycle (Mitotic), Cell Cycle, Immune System, Infectious Disease, Innate Immune System, Disease, Cellular Responses To Stimuli/Stress, Developmental Biology, Cell Cycle Checkpoints, Hemostasis, Gene Expression (Transcription), Metabolism Of Proteins, RNA Polymerase II Transcription, Metabolism Of RNA, Vesicle-mediated Transport — were excluded per the standing project rule as umbrella terms that light up for almost any active biology, not disease-specific to TB.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 20725040 — Berry et al. Nature 2010, IFN-inducible neutrophil blood signature in TB (cited, #1)
- 25614319 — Ivashkiv & Donlin, type I IFN in infectious disease review (cited, #1)
- 39085192 — NETs characterize caseating granulomas, Cell Death Dis 2024 (cited, #2)
- 36454773 — neutrophil degranulation/NETosis/platelet genes pre-diagnosis TB, PLoS One 2022 (cited, #2 and #4)
- 31497538 — alveolar epithelial cells in Mtb primary infection/dissemination (cited, #3)
- 34093524 — platelet activation and immune response to TB review (cited, #5)
- 39287444 — SIRT7/RAC1 cytoskeleton remodeling and Mtb resistance, mBio 2024 (cited, #6)
- 31285585 — CYRI/FAM49B RAC1 cytoskeletal remodelling protects against bacterial infection, Nat Microbiol 2019 (cited, #6)
- 10358769 — Aderem & Underhill, mechanisms of phagocytosis in macrophages, Annu Rev Immunol 1999 (checked, rejected — too general, not TB- or FcγR-specific)
- 2108212 — complement-receptor-mediated (not FcγR) phagocytosis of Mtb, J Immunol 1990 (checked, rejected — wrong receptor mechanism for FcγR pathway)
- 18279703 — Fc gamma receptor / macrophage / Mtb search hit (checked, not retrieved in detail — insufficient to justify FcγR pathway inclusion)
- 34745115 — Mtb exploits focal adhesion kinase to induce necrotic cell death, Front Immunol 2021 (considered, not cited — single-kinase paper on large generic pathway, deprioritized)
- 33500344 — phagocytic cup/actin/Mtb search hit, low relevance (checked, rejected — not TB-specific to the exact Reactome node)
- 37754562 — polyphosphate/Rho GTPase in Dictyostelium bacterial survival (checked, rejected — not TB/human-relevant)
- 38892443 — host cell death and immune modulation in Mtb infection review (checked, not cited — too broad, covers multiple pathways)
- 35095914 — cGAS-STING pathway in bacterial infection/immunity (checked, not cited — different pathway, not in candidate list)
- 37733444 — human mAbs targeting Mtb arabinomannan, JCI Insight 2023 (checked, not cited — antibody/vaccine focus, not phagocytosis mechanism)
- 33193394 — mucosal IgA/IFN-gamma therapy for MDR-TB (checked, not cited — therapeutic study, not pathway mechanism)
- 32809396 — Type I Hypersensitivity Reaction (checked, rejected — irrelevant, generic immunology textbook chapter)
- 29130366 — autophagy and inflammation in chronic respiratory disease (checked, not cited — off-topic for candidate pathways reviewed)
