# Tuberculosis — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Tuberculosis.txt` (n=101 TB patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below.

## Selected pathways

### 1. Interferon Alpha/Beta Signaling
n_sig=43/101, size=71, eff=5.01, hist_frac=0%

Type I IFN is a specific, mechanistically demonstrated *driver* of TB susceptibility (not just a bystander correlate) — it promotes neutrophil-mediated lung pathology and impairs protective IFN-γ responses in Mtb-susceptible models, and single-cell work traces the earliest cellular steps of type-I-IFN-driven susceptibility in human/mouse TB.
- Moreira-Teixeira et al., Nature Communications, 2020 (PMID: 33149141) — type I IFN exacerbates disease in TB-susceptible mice via neutrophil-mediated lung inflammation and NETosis.
- Kotov et al. (or equivalent), Cell, 2023 Dec 7 (PMID: 38029747) — early cellular mechanisms of type I interferon-driven susceptibility to tuberculosis.
- McNab, Mayer-Barber, Sher, Wack, O'Garra, Nature Reviews Immunology, 2015 (PMID: 25614319) — review establishing type I IFN as pathogenic (not merely antiviral) in TB, distinguishing it from its protective role in most viral infections.

### 2. Interferon Signaling
n_sig=53/101, size=197, eff=5.4, hist_frac=0%

This is the pathway-level correlate of the single most validated TB blood biomarker: an IFN-inducible whole-blood transcriptional signature that discriminates active TB from latent infection/other diseases and normalizes with treatment — replicated across cohorts and used as the basis for concise diagnostic gene signatures.
- Berry et al., Nature, 2010 Aug 19 (PMID: 20725040) — "An interferon-inducible neutrophil-driven blood transcriptional signature in human tuberculosis" (the founding paper for this biomarker class).
- Warsinske et al. (Sweeney group), Lancet Respiratory Medicine, 2020 (PMID: 31958400) — systematic review/meta-analysis confirming concise IFN-dominated whole-blood signatures for incipient TB across many independent cohorts.

Note: overlaps mechanistically with pathway #1 (Interferon Alpha/Beta Signaling is a Reactome sub-branch of this parent term); kept as a separate row because the underlying evidence bases are distinct (diagnostic-biomarker literature here vs. pathogenic-mechanism literature for the alpha/beta-specific pathway).

### 3. Neutrophil Degranulation
n_sig=50/101, size=463, eff=5.55, hist_frac=0%

The dominant cell type behind the TB blood transcriptional signature (#2) is the neutrophil, and neutrophil granule-protein pathways are independently linked to TB susceptibility via PI3K-driven neutrophil mobilization in susceptible hosts — not a generic "innate immune" association but tied specifically to the neutrophil-driven pathophysiology unique to active TB among common infections.
- Berry et al., Nature, 2010 (PMID: 20725040) — same neutrophil-driven signature as above.
- Moreno-Molina et al. / Lienhardt group or equivalent, Frontiers in Immunology, 2018 (PMID: 30065729) — "Susceptibility to Tuberculosis Is Associated With PI3K-Dependent Increased Mobilization of Neutrophils."

Supporting (Open Targets, disease=MONDO_0018076 "tuberculosis"): CTSG (cathepsin G, a canonical neutrophil-granule protein) appears among the top ~25 target-disease association scores, consistent with a genuine neutrophil-granule signal rather than an artifact of pathway size.

### 4. Neutrophil extracellular trap formation
n_sig=61/101, size=183, eff=5.79, **hist_frac=40% — kept despite the histone-fraction flag**

Rationale for the exception: NETs are, by definition, decondensed extracellular chromatin studded with (citrullinated) histones — histone genes are a mechanistic component of this specific pathway's biology, not an unrelated composition artifact riding along (unlike the Pancreatic-Cancer "Systemic lupus erythematosus" precedent, where histones appear only as SLE autoantigens unconnected to the pathway's actual function). NETosis has direct, TB-specific mechanistic literature: it is pathologically implicated in type-I-IFN-driven lung damage in susceptible TB models and structurally characterizes caseating (necrotic) granulomas, the histological hallmark of TB.
- Moreira-Teixeira et al., Nature Communications, 2020 (PMID: 33149141) — type I IFN induces neutrophil-mediated lung inflammation and NETosis in TB-susceptible mice.
- (Author list unresolved from summary), Cell Death & Disease, 2024 Jul 31 (PMID: 39085192) — "Neutrophil extracellular traps characterize caseating granulomas" — direct histological link to the TB granuloma phenotype.

This is flagged as **weaker-tier evidence** given the histone-fraction override; independent verification of the composition-vs-biology argument is recommended before over-weighting it in downstream synthesis.

### 5. Fc gamma R-mediated phagocytosis
n_sig=66/101, size=96, eff=4.33, hist_frac=0%

Fc-receptor-mediated antibody effector function is a specific, actively studied arm of TB immunity/vaccinology — landmark work showed IgG Fc-glycosylation and Fc-receptor engagement drive functional, protective antibody activity against Mtb, distinct from adaptive/humoral immunity's generic role in most infections.
- Lu, Chen, Ackerman et al. (Alter lab), Cell, 2016 Oct 6 (PMID: 27667685) — "A Functional Role for Antibodies in Tuberculosis," the founding paper on Fc-mediated antibody function in TB.
- (Author list unresolved from summary), Immunity, 2025 Jun 10 (PMID: 40449485) — "Antibody-Fab and -Fc features promote Mycobacterium tuberculosis restriction."
- (Author list unresolved from summary), ACS Infectious Diseases, 2025 Jun 13 (PMID: 40312277) — antibody-recruiting molecule enhances FcγR-mediated uptake/killing of mycobacterial pathogens by macrophages.

Note: two near-duplicate Reactome entries for the same mechanism ("FCGR3A-mediated Phagocytosis," "Fcgamma Receptor (FCGR) Dependent Phagocytosis") were in the candidate list; only the top-n_sig representative is kept here (see Rejected).

### 6. Regulation Of Actin Dynamics For Phagocytic Cup Formation
n_sig=63/101, size=61, eff=4.27, hist_frac=0%

Phagocytic-cup actin remodeling is the specific entry mechanism Mtb exploits to invade macrophages; a direct mechanistic study shows host sphingomyelin biosynthesis is required for the phagocytic actin-signaling machinery during Mtb entry, i.e. this pathway is mechanistically upstream of the establishment of infection, not a generic "phagocytosis happens in every infection" signal.
- (Author list unresolved from summary), mBio, 2021 Jan 26 (PMID: 33500344) — "Sphingomyelin Biosynthesis Is Essential for Phagocytic Signaling during Mycobacterium tuberculosis Host Cell Entry."

Supporting (Open Targets): CORO1A (coronin 1A, a well-characterized regulator of the actin cytoskeleton at the phagocytic cup and of Mtb phagosomal arrest) and RAB5A/RAB7A (phagosome maturation) both appear among the top ~25 TB target-disease association scores.

### 7. Platelet Activation, Signaling And Aggregation
n_sig=53/101, size=252, eff=4.5, hist_frac=0%

Platelet activation and thrombocytosis are a recognized, clinically used feature specifically of active TB (part of the classic paraclinical picture, distinguishing it from most other infections), with a dedicated immunopathology literature on platelets' direct role in the TB granuloma/immune response, not just a generic acute-phase reaction.
- Fox, Lam et al. or equivalent, American Journal of Clinical Pathology, 2013 May (PMID: 23596109) — thrombocytosis is associated with M. tuberculosis infection and positive acid-fast stains in granulomas.
- (Author list unresolved from summary), Frontiers in Immunology, 2021 (PMID: 34093524) — review, "Platelet Activation and the Immune Response to Tuberculosis."
- Rahman, Zumla et al. or equivalent, European Respiratory Journal, 1998 Dec (PMID: 9877494) — in vivo platelet and T-lymphocyte activities during pulmonary tuberculosis (early direct evidence).

## Considered and rejected

- **Bacterial invasion of epithelial cells** (n_sig=66, size=77, eff=4.53, hist_frac=0%). A specific paper supports Mtb alveolar-epithelial invasion (Frontiers Cell Infect Microbiol 2019, PMID: 31497538, "Mycobacterium tuberculosis Primary Infection and Dissemination: A Critical Role for Alveolar Epithelial Cells"). Rejected anyway: this exact KEGG pathway also scores similarly (n_sig 51-64) for Shigellosis, Salmonella infection, Yersinia infection, and Pathogenic E. coli infection in the *same* patient set (rows 23/37/43/58 of the candidate list), all built from the same shared actin/cytoskeletal effector-gene core — indicating cross-pathogen composition overlap rather than TB-specific signal.
- **FCGR3A-mediated Phagocytosis / Fcgamma Receptor (FCGR) Dependent Phagocytosis** — near-duplicate Reactome pathways of Selected #5 covering the same mechanism/gene core; redundant, kept only the top-n_sig entry.
- **Signaling By Interleukins** (n_sig=52, size=449, eff=4.26). Genuine TB-specific mechanism does exist within it — the IL-12/IFN-γ axis is the basis of Mendelian Susceptibility to Mycobacterial Disease (MSMD) (Bustamante, Semin Immunol 2014, PMID: 25453225; Kerner et al., Hum Genet 2020, PMID: 32055999) — but the candidate pathway as scored is a broad umbrella (449 genes spanning IL-1/4/6/13/17/etc.) that would light up for essentially any active immune response, not isolating the IL-12/IFN-γ arm specifically. Rejected as too broad to claim TB-specificity from this gene set alone.
- **Signaling By CSF3 (G-CSF)** (n_sig=56, size=30, eff=3.9). Searched directly; only 2 PubMed hits for "tuberculosis AND G-CSF AND neutrophilia," one of which is about *M. avium* in mice, not *M. tuberculosis* (PMID: 11422200); the other is tangential. No TB-specific literature found despite targeted search — rejected.
- **Neutrophil Degranulation** vs generic "Neutrophil Extracellular Trap formation" — both kept (see Selected #3/#4); not a rejection, noted here only because they were evaluated together.
- All [GENERIC]-flagged rows (Cell Cycle*, Immune System, Rho GTPase signaling, Infectious Disease, Disease, RNA Pol II Transcription, Metabolism Of RNA/Proteins, Cellular Responses To Stress/Stimuli, Generic Transcription Pathway, Spliceosome, etc.) and high-histone rows other than #4 (Systemic lupus erythematosus hist_frac=57%, Alcoholism hist_frac=41%, Assembly Of Pre-Replicative Complex hist_frac=26%, DNA Replication hist_frac=18%, Viral carcinogenesis hist_frac=17%) were excluded per the standing project rule; no targeted search found a TB-specific mechanistic case strong enough to override for any of these (in particular, "Systemic lupus erythematosus" mirrors the Pancreatic Cancer precedent exactly — histone-autoantigen composition, not disease biology).

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 20725040 — Berry et al., Nature 2010, IFN-inducible neutrophil-driven blood transcriptional signature in TB (cited)
- 31958400 — Warsinske et al., Lancet Respir Med 2020, concise whole-blood TB signatures meta-analysis (cited)
- 25703554 — Immunological Reviews 2015, human immune response to TB (considered, not cited)
- 38029747 — Cell 2023, early cellular mechanisms of type I IFN-driven susceptibility to TB (cited)
- 25614319 — McNab et al., Nat Rev Immunol 2015, type I interferons in infectious disease (cited)
- 40449485 — Immunity 2025, antibody-Fab/-Fc features promote Mtb restriction (cited)
- 33149141 — Nat Commun 2020, type I IFN exacerbates TB disease via neutrophil NETosis (cited)
- 39085192 — Cell Death & Disease 2024, NETs characterize caseating granulomas (cited)
- 40234407 — Signal Transduct Target Ther 2025, cytokine storm review (considered, not cited — too generic)
- 29130366 — Autophagy 2018, autophagy/inflammation in chronic respiratory disease (considered, not cited — not TB-specific)
- 10358769 — Annu Rev Immunol 1999, mechanisms of phagocytosis in macrophages (considered, not cited — general review, not TB-specific)
- 40312277 — ACS Infect Dis 2025, antibody-recruiting molecule enhances FcγR uptake/killing of mycobacteria (cited)
- 38442687 — Brain 2024, antibody responses to Mtb in pulmonary vs brain infection (considered, not cited)
- 27481246 — Infect Immun 2016, neonatal Fc receptor regulation of lung Ig/CD103+ DCs in TB susceptibility (considered, not cited)
- 24250791 — PLoS ONE 2013, FCGR copy number variation in HIV/TB co-infection (considered, not cited)
- 27667685 — Lu et al., Cell 2016, "A Functional Role for Antibodies in Tuberculosis" (cited)
- 30065729 — Front Immunol 2018, PI3K-dependent neutrophil mobilization and TB susceptibility (cited)
- 11422200 — Clin Exp Immunol 2001, recombinant G-CSF during M. avium infection in mice (checked, rejected — wrong pathogen)
- 33500344 — mBio 2021, sphingomyelin biosynthesis essential for phagocytic signaling during Mtb entry (cited)
- 31497538 — Front Cell Infect Microbiol 2019, alveolar epithelial cells critical for Mtb dissemination (checked, rejected pathway — see Rejected)
- 25743470 — Pathog Dis 2015, Mce4F Mtb protein peptides inhibit epithelial invasion (considered, not cited)
- 34093524 — Front Immunol 2021, platelet activation and immune response to TB (cited)
- 34400911 — Front Immunol 2021, editorial on platelet activation in HIV/TB/pneumococcal disease (considered, not cited)
- 9877494 — Eur Respir J 1998, in vivo platelet/T-lymphocyte activity in pulmonary TB (cited)
- 38309972 — Zhonghua Jiehe... 2024, platelets and TB review (Chinese) (considered, not cited)
- 34053983 — Intern Med 2021, isoniazid-induced immune thrombocytopenia (considered, not cited — drug AE, not disease mechanism)
- 23596109 — Am J Clin Pathol 2013, thrombocytosis associated with M. tuberculosis infection/granulomas (cited)
- 25453225 — Semin Immunol 2014, Mendelian susceptibility to mycobacterial disease (MSMD) (checked, rejected pathway — see Rejected)
- 32055999 — Hum Genet 2020, monogenic basis of human tuberculosis (checked, rejected pathway)
- 16272979 — Rev Mal Respir 2005, genetic susceptibility to mycobacterial disease, IL-12/IFN-γ axis (checked, rejected pathway)
- 41053048 — Nat Commun 2025, Mtb-specific T cells restrain neutrophilic lung inflammation in TB (considered, not cited)
- 39693225 — Cell Reports 2025, CD101-negative neutrophils in type I IFN-mediated TB immunopathogenesis (considered, not cited)

Open Targets (GraphQL, disease=MONDO_0018076 "tuberculosis"): top ~25 associatedTargets by score included TLR2, MRC1, CORO1A, RAB5A, RAB7A, CTSG, MAPK1/3 — used only as supporting corroboration for pathways #3 and #6, per project rule that Open Targets scores alone must not justify a pathway.
