# Tuberculosis Pathway Curation

Source: GSEA normative ranking, `normative__Tuberculosis__Chang_et_al..csv`, FDR q<0.05; inflammatory-disease hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

## Selected Pathways

### Bacterial invasion of epithelial cells
- NES: 2.379, FDR q: 0.000
- Literature: Mycobacterium tuberculosis actively drives actin-dependent entry into non-phagocytic epithelial cells via macropinocytosis and Mce3C-triggered beta2-integrin signaling, engaging the same actin/Rho-GTPase machinery captured by this KEGG term (PMID 29125705; PMID 12901843; PMID 18487035).
- Lead genes (top ~10): ACTB, ARPC1B, ACTG1, ARPC4, RHOA, ILK, SHC1, CTNNA1, PTK2, SRC

### Regulation Of Actin Dynamics For Phagocytic Cup Formation R-HSA-2029482
- NES: 2.301, FDR q: 0.000
- Literature: Macrophage phagocytic cup formation and actin remodeling is the core cellular entry route for Mtb into its principal host cell niche; Mtb effectors (e.g. Mce3C) hijack this actin/integrin-coupled machinery to promote uptake (PMID 29125705). Kept as the representative phagocytosis term (see Dropped: FCGR3A/FCGR/Fc-gamma-R phagocytosis, near-duplicates).
- Lead genes (top ~10): ARPC1B, ACTG1, GRB2, WIPF1, FCGR1A, PTK2, LIMK1, MAPK3, WAS, CYFIP1

### Neutrophil extracellular trap formation
- NES: 2.152, FDR q: 0.000
- Literature: Type I IFN-driven NET release promotes Mtb replication and is directly associated with granuloma caseation in vivo, establishing NETosis as an active, disease-relevant TB pathomechanism rather than a bystander signature (PMID 39637864, Cell Host & Microbe 2024).
- Lead genes (top ~10): ACTB, ACTG1, H3C12, CTSG, HDAC5, FCGR1A, MPO, MAPK3, NCF2, H2BC12

### Interferon Alpha/Beta Signaling R-HSA-909733
- NES: 2.140, FDR q: 0.000
- Literature: A blood-neutrophil-driven type I/II interferon-inducible transcriptional signature is the best-validated whole-blood biomarker of active tuberculosis, correlating with radiographic disease extent and resolving with treatment (Berry et al., PMID 20725040, Nature 2010). Directly supports detectability in blood/cfRNA.
- Lead genes (top ~10): IFITM3, KPNB1, GBP2, STAT1, ADAR, PTPN11, IFIT3, HLA-C, IFI27, HLA-B

### Integrin Signaling R-HSA-354192
- NES: 2.098, FDR q: 0.000
- Literature: Mtb Mce3C activates beta2-integrin-mediated signaling to promote macrophage entry (PMID 29125705), and integrin alphaVbeta3 regulates monocyte adhesion, transendothelial migration, and ECM breakdown during Mtb infection (PMID 28646039), linking integrin signaling to both bacterial entry and granuloma tissue remodeling.
- Lead genes (top ~10): GRB2, SHC1, PTK2, SRC, ITGA2B, AKT1, CSK, RAP1A, SOS1, TLN1

### Interferon Gamma Signaling R-HSA-877300
- NES: 1.945, FDR q: 0.0004
- Literature: IFN-gamma is the central, non-redundant cytokine of anti-TB host defense -- Ifng gene-disrupted mice succumb to disseminated, uncontrolled TB (Cooper et al., PMID 8245795, J Exp Med 1993). Leading edge is dominated by canonical GBP/STAT1/IRF/OAS interferon-response machinery, distinct from the already-kept type I (alpha/beta) signature.
- Lead genes (top ~10): GBP5, MT2A, GBP1, GBP2, STAT1, FCGR1A, PTPN11, HLA-C, PRKCD, HLA-B

### Chemokine signaling pathway (KEGG)
- NES: 1.916, FDR q: 0.0006
- Literature: IP-10/CXCL10 is one of the best-validated blood chemokine biomarkers for pulmonary TB diagnosis and treatment monitoring (Qiu et al., PMID 31666025, BMC Infect Dis 2019). Leading edge includes CXCL10, CXCL11, CXCL2, CXCR2 alongside canonical GPCR/JAK-STAT chemokine-signal-transduction machinery -- genuine pathway content, no housekeeping contamination.
- Lead genes (top ~10): STAT3, GRK5, RHOA, GRB2, SHC1, STAT1, GNAI2, PTK2, GNB2, MAPK3 (plus CXCL10, CXCL11, CXCR2 further in leading edge)

### Neutrophil Degranulation R-HSA-6798695
- NES: 1.78, FDR q: 0.004
- Literature: Whole-blood transcriptomics shows a distinct activated-neutrophil/granule-protein transcriptional landscape in active TB (Geng et al., PMID 36059536, Front Immunol 2022), consistent with the classic Berry et al. neutrophil-driven IFN signature. Kept as distinct from the already-kept NETosis term: degranulation (MPO, ELANE, PRTN3, DEFA1/3/4, LTF, CTSG, MMP8) is a separate effector mechanism from NET release. A small proteasome-subunit tail (PSMD2/PSMD11/PSMC2 etc., ~7% of leading edge) is not enough to call this an artifact.
- Lead genes (top ~10): TUBB, TUBB4B, RAB5B, CNN2, PKM, CTSA, KPNB1, RHOA, LAMTOR1, BIN2 (plus MPO, ELANE, DEFA4, LTF, CTSG further in leading edge)

### Interleukin-1 Family Signaling R-HSA-446652
- NES: 1.889, FDR q: 0.0009
- Literature: IL-1 signaling (via IL1R1/MyD88/IRAK) is required for TB control and cross-talks antagonistically with type I IFN through eicosanoid regulation, a mechanism now pursued as host-directed TB therapy (Mayer-Barber et al., PMID 24990750, Nature 2014). Leading edge retains genuine IL-1-axis genes (IL1R1, IL18, IL33, IL1RN, MYD88, IRAK1, CASP1, TAB2, TAB3, IKBKG, RELA) alongside a proteasome-subunit tail (~35% of leading edge, PSMA/PSMB/PSMC/PSMD) that reflects genuine ubiquitin-proteasome-mediated turnover of IL-1 pathway components rather than a mislabeled housekeeping set.
- Lead genes (top ~10): STAT3, UBC, CTSG, UBB, PTPN11, PSMD2, NKIRAS2, CUL1, UBA52, PSMF1 (plus IL1R1, IL18, IL33, MYD88, CASP1 further in leading edge)

### Interleukin-6 Signaling R-HSA-1059683
- NES: 1.606, FDR q: 0.0251
- Literature: Serum IL-6 is consistently elevated in active pulmonary TB and tracks disease activity (Dalvi et al., PMID 31439177, Indian J Tuberc 2019). Leading edge is a small, fully clean core (STAT3, STAT1, PTPN11, JAK1, IL6ST, SOCS3, JAK2) with zero housekeeping contamination.
- Lead genes (all 7): STAT3, STAT1, PTPN11, JAK1, IL6ST, SOCS3, JAK2

### NOD-like receptor signaling pathway (KEGG)
- NES: 1.691, FDR q: 0.0109
- Literature: Mtb actively suppresses NLRP3 inflammasome activation via its PknF phosphokinase, implicating inflammasome/NOD-like receptor signaling as a genuine host-pathogen battleground in TB (Rastogi et al., PMID 34324582, PLoS Pathog 2021). Leading edge is dominated by genuine inflammasome/GBP/caspase machinery (NLRC4, CASP1/4/5, AIM2, GBP1/2/4/5, DEFA1/3/4, TBK1, MAVS), no artifact contamination.
- Lead genes (top ~10): GBP5, GBP1, RHOA, GBP2, STAT1, MAPK3, PRKCD, OAS3, NEK7, DEFA3

### Initial Triggering Of Complement R-HSA-166663
- NES: 1.675, FDR q: 0.0126
- Literature: Complement receptor 3 (CR3), engaged via C3/C1q opsonization, is a principal route of Mtb entry into macrophages (Velasco-Velazquez et al., PMID 12927520, Microb Pathog 2003). Leading edge is a small, genuine complement-cascade set (C1QA/B/C, C1R, C1S, C3, CFB, FCN2/3, MBL2) with no housekeeping contamination; kept over the near-duplicate "Classical Antibody-Mediated Complement Activation" term (same C1Q/C1R/C1S core, weaker FDR, fewer distinct genes).
- Lead genes (top ~10): C1QA, C1QB, C1QC, HNRNPC, PIKFYVE, C3, FCN3, CFB, C1R, C1S

## Dropped candidates (GSEA-significant, no adequate literature support)
- FCGR3A-mediated Phagocytosis R-HSA-9664422: NES 2.264 -- >70% Lead_genes overlap with Regulation Of Actin Dynamics For Phagocytic Cup Formation (kept); near-duplicate.
- Fcgamma Receptor (FCGR) Dependent Phagocytosis R-HSA-2029480: NES 2.209 -- near-duplicate of kept phagocytic-cup term.
- Fc gamma R-mediated phagocytosis (KEGG): NES 2.179 -- near-duplicate of kept phagocytic-cup term.
- Shigellosis (KEGG): NES 2.125 -- >60% Lead_genes overlap with Bacterial invasion of epithelial cells (kept); near-duplicate, and the disease itself (Shigella) is unrelated to TB.
- Signaling By CSF3 (G-CSF) R-HSA-9674555: NES 2.186 -- gene set is generic JAK/STAT cytokine-signaling core (JAK1/2, STAT3/5, SOCS1/3) shared across many cytokine pathways; only literature found was for GM-CSF immunotherapy in chronic TB (PMID 32382128), not a specific G-CSF-TB mechanistic link.
- Signaling By RAF1 Mutants R-HSA-9656223 / Signaling By ERBB2 R-HSA-1227986 / Signaling By KIT In Disease R-HSA-9669938 / Paradoxical Activation Of RAF Signaling By Kinase Inactive BRAF R-HSA-6802955 / Signaling By High-Kinase Activity BRAF Mutants R-HSA-6802948 / Signaling To ERKs R-HSA-187687 / MAP2K And MAPK Activation R-HSA-5674135 / ErbB signaling pathway (KEGG): NES 2.10-2.18 -- generic oncogenic RAS-RAF-MAPK driver gene sets (GRB2/SHC1/SRC-centric), designed for cancer signaling context; no TB-specific literature found.
- Viral carcinogenesis (KEGG): NES 2.102 -- viral oncogenesis pathway, biologically unrelated to a bacterial infection.
- Cooperation Of Prefoldin And TriC/CCT In Actin And Tubulin Folding R-HSA-389958 / Formation Of Tubulin Folding Intermediates By CCT/TriC R-HSA-389960: NES ~2.09-2.10 -- generic protein-folding chaperone housekeeping machinery, not TB-specific.
- Mitotic G1 Phase And G1/S Transition R-HSA-453279: NES 2.075 -- generic cell-cycle housekeeping term, likely reflects lymphocyte/neutrophil proliferative turnover rather than TB-specific biology.
- Interleukin-12 Family Signaling R-HSA-447115: NES 1.933, FDR 0.0005 -- relaxation candidate, rejected. Leading edge dilutes the genuine JAK/STAT/IL27 core (STAT1/3, JAK1/2, IL6ST, IL27) with housekeeping/hnRNP genes (HNRNPF, HNRNPA2B1, RPLP0, AURKAIP1, TCP1) and lacks IL12A/B or IL12RB1/2 -- not a clean IL-12-specific signature.
- MHC Class II Antigen Presentation R-HSA-2132295: NES 1.839, FDR 0.0019 -- relaxation candidate, rejected. Leading edge is almost entirely generic vesicular-trafficking machinery (AP1/AP2 adaptors, kinesins KIF2C/3C/4A/11/15/18A/20A/23, dynactin, COPII SEC23/24) with no HLA-DR/DQ/DM, CD74, or CIITA -- trafficking machinery wearing an MHC-II label, not genuine antigen-presentation signal.
- Antigen processing-Cross Presentation R-HSA-1236975: NES 1.9, FDR 0.0008 -- relaxation candidate, rejected. ~45% of leading edge is 26S proteasome subunits (PSMA/PSMB/PSMC/PSMD); remaining HLA-A/B/C/E/F/G and FCGR1A/NCF1/2 content overlaps mechanistically with already-kept Interferon Gamma Signaling and Neutrophil Degranulation without adding a distinct TB-specific mechanism.
- Oxidative Stress Induced Senescence R-HSA-2559580: NES 1.844, FDR 0.0018 -- relaxation candidate, rejected. Leading edge is dominated by histones (H2AC/H2BC/H3C/H4C) and generic senescence/DNA-damage factors (MDM2, TP53, CDK4, E2F1); no antioxidant/ROS-response enzymes (SOD, CAT, GPX, NQO1, HMOX1) present, so it does not genuinely represent oxidative-stress biology despite the name.
- Interleukin-15 Signaling R-HSA-8983432: NES 2.011, FDR 0.0001 -- relaxation candidate, clean leading edge (STAT3/5, GRB2, SHC1, JAK1/3, SOS1) and plausible TB literature (NK/CD8 IL-15 axis), but not kept: falls outside the specified relaxed-category list (IL-1/IL-6/TNF/chemokine, not IL-15) and redundant with already-kept IL-6/chemokine JAK-STAT signal.
