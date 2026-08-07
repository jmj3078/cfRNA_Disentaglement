# Liver Cancer (Roskams-Hieter B et al.) — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Liver_Cancer_(Roskams-Hieter_B_et_al.).txt` (n=28 patients). [GENERIC]-flagged and hist_frac>15% pathways excluded by default per project rule; exceptions justified explicitly below.

## Selected pathways

### 1. Signaling By Rho GTPases
n_sig=14/28, size=640, eff=4.71, hist_frac=4.4%

Rho-family GTPases (RHOA/RHOC, RAC1, CDC42 and their GAPs/GEFs) are established drivers of hepatocellular carcinoma (HCC) cytoskeletal remodeling, invasion and metastasis; a dedicated pan-cancer review covers the family specifically in HCC. Chosen over the child pathway "Signaling By Rho GTPases, Miro GTPases And RHOBTB3" (near-duplicate, lower score) and over the narrower "RAC1 GTPase Cycle"/"CDC42 GTPase Cycle" child terms, which are subsumed by this parent.
- Zhu et al., Experimental Hematology & Oncology, 2022 (PMID: 36348464) — review of the RHO GTPase family specifically in hepatocellular carcinoma.
- Xu et al., Hepatology Research, 2024 (PMID: 37792600) — ARHGAP1 (Rho GTPase-activating protein 1) promotes HCC progression via circPIP5K1A/miR-101-3p.
- Yan et al., Cell Death & Disease, 2024 (PMID: 38378644) — HMGB1 promotes mitochondrial transfer between HCC cells through RHOT1 and RAC1 under hypoxia.

### 2. Neutrophil extracellular trap formation
n_sig=12/28, size=183, eff=4.85, hist_frac=39.9% — **histone exception**: citrullinated core histones (H3/H4) are a defining structural component of NETs, not an unrelated composition artifact riding along in this pathway; the high histone fraction directly reflects the pathway's actual biology.

NETs are directly implicated in HCC growth, metastasis and immune evasion in multiple independent mechanistic studies, including HBV-driven and T-cell-suppressive mechanisms specific to liver cancer.
- Zhan et al., Cancer Communications, 2023 (PMID: 36346061) — HBV-mediated S100A9-TLR4/RAGE-ROS cascade elevates NETs to facilitate HCC growth and metastasis.
- Yang et al., Journal of Hematology & Oncology, 2020 (PMID: 31907001) — increased NETs promote HCC metastatic potential via tumorous inflammatory response.
- Zhu et al., Cancer Research, 2024 (PMID: 38381538) — NET DNA binds TMCO6 to impair CD8+ T-cell immunity in HCC.

### 3. Complement and coagulation cascades
n_sig=10/28, size=85, eff=4.72, hist_frac=0%

The liver is the primary synthesis site for both complement and coagulation factors, and dysregulation of this cascade is repeatedly linked to HCC prognosis, immune microenvironment and therapeutic vulnerability. Preferred over the broader, lower-specificity "Hemostasis" pathway (same score tier, largely overlapping gene content, no HCC-specific mechanistic paper found for the broader term).
- Ye et al., Heliyon, 2024 (PMID: 39391504) — prognostic model of complement/coagulation-cascade-related genes correlates with immune environment and drug sensitivity in HCC.
- Xu et al., Frontiers in Oncology, 2020 (PMID: 33718121) — mechanistic review of the complement system in HCC ("Complimenting the Complement").

### 4. Chromatin Modifying Enzymes
n_sig=8/28, size=235, eff=4.26, hist_frac=14.0% (below the 15% exclusion threshold, no exception needed)

Epigenetic dysregulation via chromatin-modifying enzymes (HDACs, HMTs, DNMTs, etc.) is one of the best-characterized mechanistic axes in HCC pathogenesis, reviewed extensively.
- Toh et al., Seminars in Cancer Biology, 2022 (PMID: 34324953) — "Epigenetics in hepatocellular carcinoma" review.
- Bayo et al., Journal of Experimental & Clinical Cancer Research, 2022 (PMID: 35331312) — "Epigenetic remodelling in human hepatocellular carcinoma."

### 5. Lysosome
n_sig=8/28, size=127, eff=4.16, hist_frac=0%

Autophagy-lysosomal pathway activity (including mitophagy, a lysosome-dependent process) is mechanistically tied to HCC proliferation, drug resistance and sorafenib sensitivity in recent functional studies.
- Zhang et al., Autophagy, 2022 (PMID: 34890308) — CDK9 inhibition blocks PINK1-PRKN mitophagy initiation via SIRT1-FOXO3-BNIP3, enhancing therapeutic effect in HCC.
- Chen et al., Autophagy, 2024 (PMID: 37733919) — Artesunate sensitizes HCC to sorafenib via AFAP1L2-SRC-FUNDC1-dependent mitophagy.

### 6. Senescence-Associated Secretory Phenotype (SASP)
n_sig=7/28, size=80, eff=4.16, hist_frac=35.0% — **histone exception**: SASP execution is mechanistically coupled to senescence-associated heterochromatin remodeling, and the histone genes present reflect that chromatin-state biology rather than an unrelated artifact.

SASP is directly implicated in liver carcinogenesis, including a landmark paper showing obesity-induced gut microbial metabolites drive SASP-mediated liver cancer, and direct histological evidence of SASP in the tumor stroma of steatohepatitic HCC.
- Yoshimoto et al., Nature, 2013 (PMID: 23803760) — obesity-induced gut microbial metabolite promotes liver cancer through the senescence secretome (SASP).
- Lee et al., PLoS One, 2017 (PMID: 28273155) — tumor stroma with SASP in steatohepatitic hepatocellular carcinoma.

### 7. Alcoholism
n_sig=9/28, size=180, eff=4.97, hist_frac=40.6% — **histone exception**: the KEGG "Alcoholism" pathway gene set is built around ethanol-induced histone/chromatin modification machinery — this is the pathway's actual documented mechanism, not incidental histone composition, and alcohol is a major, well-established HCC risk factor.

Ethanol metabolism drives histone acetylation/methylation changes in the liver that are mechanistically linked to alcohol-related hepatocarcinogenesis.
- Shukla and Lim, Alcohol Research, 2013 (PMID: 24313164) — epigenetic (histone-modification) effects of ethanol on the liver.
- Seitz and Stickel, Advances in Experimental Medicine and Biology, 2015 (PMID: 25427901) — alcohol and cancer overview, mechanistic role of acetaldehyde/CYP2E1 in hepatocarcinogenesis.

## Considered and rejected

- **RHO GTPase Cycle** (n_sig=17, size=439, eff=4.68, hist_frac=0%). [GENERIC]-flagged umbrella term; superseded by the more specific, non-generic "Signaling By Rho GTPases" selected above.
- **Systemic lupus erythematosus** (n_sig=14, size=127, eff=5.60, hist_frac=57.5%). High histone fraction reflects nucleosome/anti-histone autoantibody biology specific to SLE, not liver cancer; no HCC-specific mechanistic link found, rejected.
- **Signaling By Rho GTPases, Miro GTPases And RHOBTB3** (n_sig=14, size=656, eff=4.66, hist_frac=4.3%). Near-duplicate of "Signaling By Rho GTPases" (parent term, higher score, same evidence base) — rejected as redundant.
- **RAC1 GTPase Cycle** (n_sig=12, size=178, eff=4.12, hist_frac=0%) and **CDC42 GTPase Cycle** (n_sig=10, size=148, eff=4.02, hist_frac=0%). Both are child terms subsumed by "Signaling By Rho GTPases" — rejected as redundant.
- **Olfactory transduction / Olfactory Signaling Pathway / Expression And Translocation Of Olfactory Receptors** (n_sig=10-11, hist_frac=0%). Non-generic and low histone fraction, but no plausible or literature-supported mechanistic link to hepatocellular carcinoma biology; likely reflects a large gene-family/technical signature rather than disease biology — rejected.
- **Hemostasis** (n_sig=8, size=572, eff=3.84, hist_frac=0.5%). Broader superset overlapping with "Complement and coagulation cascades"; no HCC-specific mechanistic paper found for the broader term specifically — rejected as redundant/less specific.
- **RNA Polymerase I Promoter Opening** (hist_frac=90.3%), **DNA Methylation** (hist_frac=84.8%), **Condensation Of Prophase Chromosomes** (hist_frac=69.0%), **Assembly Of ORC Complex At Origin Of Replication** (hist_frac=75.7%), **Packaging Of Telomere Ends** (hist_frac=81.3%). Very high histone fractions with no evidence the pathway's core biology (as opposed to gene-list composition) is specifically about histones/chromatin in a liver-cancer-relevant way — excluded per default rule, no exception justified.
- **DNA Replication** (n_sig=8, size=154, eff=4.32, hist_frac=18.2%) and **Assembly Of Pre-Replicative Complex** (hist_frac=25.7%). Above the 15% histone threshold without a histone-specific mechanistic rationale distinct from generic proliferation — excluded.
- **Activated PKN1 Stimulates Transcription Of Androgen Receptor Regulated KLK2 And KLK3** (hist_frac=80.0%). Prostate-cancer-specific androgen receptor mechanism, not relevant to liver cancer — rejected.
- **Amyloid Fiber Formation** (hist_frac=33.3%) and **Base-Excision Repair, AP Site Formation** (hist_frac=60.5%). High histone fractions with no HCC-specific mechanistic literature identified — excluded.
- All other [GENERIC]-flagged rows in the top ~40 (Metabolism Of RNA, Gene Expression (Transcription), Metabolism Of Proteins, Cellular Responses To Stress/Stimuli, RNA Polymerase II Transcription, Generic Transcription Pathway, mRNA Splicing/Spliceosome, Cell Cycle and its children, Disease, Immune System, Cytokine Signaling In Immune System, Infectious Disease, Adaptive Immune System, Transcriptional Regulation By TP53) were excluded per the standing project rule as broad umbrella terms that light up for almost any active biology, not disease-specific.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 36348464 — RHO GTPase family in hepatocellular carcinoma (cited)
- 37792600 — ARHGAP1/circPIP5K1A/miR-101-3p in HCC progression (cited)
- 38378644 — HMGB1/RHOT1/RAC1 mitochondrial transfer under hypoxia in HCC (cited)
- 36346061 — HBV-S100A9-TLR4/RAGE-ROS-NET axis in HCC growth/metastasis (cited)
- 31907001 — NETs promote HCC metastatic potential (cited)
- 38381538 — NET DNA-TMCO6 impairs CD8+ T-cell immunity in HCC (cited)
- 39529085 — NET-related HCC study (checked, not cited — redundant with above)
- 38670307 — NET-related HCC study (checked, not cited — redundant with above)
- 39391504 — complement/coagulation gene prognostic model in HCC (cited)
- 33718121 — complement system mechanistic review in HCC (cited)
- 33911900, 37116239, 39011654 — complement/coagulation-HCC studies (checked, not cited — redundant with above)
- 34890308 — CDK9/mitophagy/HCC therapeutic effect (cited)
- 37733919 — Artesunate/FUNDC1-mitophagy/sorafenib sensitization in HCC (cited)
- 39316516, 37469132, 33794741 — autophagy-lysosome-HCC studies (checked, not cited — redundant with above)
- 34324953 — Epigenetics in hepatocellular carcinoma review (cited)
- 35331312 — Epigenetic remodelling in human HCC review (cited)
- 29454793, 36685594, 31221981 — histone/chromatin-HCC studies (checked, not cited — redundant with above)
- 23803760 — obesity/gut microbial metabolite/SASP/liver cancer, Nature (cited)
- 28273155 — SASP in steatohepatitic HCC tumor stroma (cited)
- 38825017, 40640247, 36611926 — SASP/HCC-adjacent studies (checked, not cited — less directly on SASP mechanism)
- 24313164 — epigenetic (histone) effects of ethanol on liver (cited)
- 25427901 — alcohol/acetaldehyde/CYP2E1 and cancer overview (cited)
- 27805256 — homocysteine/alcoholism/epigenetic mechanism (checked, not cited — not liver-cancer-specific)
- 33728291 — Liver cancer therapeutic models overview (checked, not cited — general background only)
- RAC1-HCC search (idlist 38378644, 38679407, 35405309, 30191377, 39286779) — checked to confirm RAC1 role is already covered under "Signaling By Rho GTPases"; not separately cited since RAC1 GTPase Cycle was rejected as redundant.
- Platelet-HCC and neuroactive-ligand searches were run informally while scanning the candidate list but did not yield a pathway strong enough to include in the top 7-8; not detailed here as none were cited.
