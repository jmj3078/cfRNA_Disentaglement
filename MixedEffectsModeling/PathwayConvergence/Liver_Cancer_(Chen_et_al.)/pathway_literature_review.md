# Liver Cancer — Pathway Literature Review

Candidate pool: top ~60 pathways by `recur*eff` from the Liver Cancer (Roskams-Hieter B et al., n=28)
cohort pathway-convergence run (`sig.pkl`, KEGG+Reactome library after ribosomal/generic filtering).
Selection cross-checked with paper-lookup (PubMed E-utilities) and gene-set composition inspection
(histone-family fraction) against `gene_set_matrix.pkl`. All citations below were confirmed by direct
PubMed query on the date of this review; none are recalled from memory alone.

## Selected pathways

### 1. Complement And Coagulation Cascades
recur=10 size=85 eff=4.75 hist_frac=0%
The liver is the near-exclusive site of synthesis for both the complement system and most coagulation
factors, so this pathway is anatomically and functionally liver-specific rather than a generic immune
term. HCC and cirrhotic liver disease measurably rewire complement production and activation, with
complement components acting on tumor growth, immune evasion, and prognosis, and complement/coagulation
gene signatures have been used to build HCC prognostic and immune-microenvironment models directly from
tumor expression data.

Citations:
- Malik A, Thanekar U, Amarachintha S, et al., *Frontiers in Oncology*, 2020 (PMID: 33718121; DOI: 10.3389/fonc.2020.627701) -- "Complimenting the Complement": review of complement dysregulation and mechanistic roles (C3/C5, cathepsin activation, immune modulation) specifically in HCC, with discussion of complement-targeted therapeutic opportunities.
- Su H, Chen Y, Wang W, *Heliyon*, 2024 (PMID: 39391504; DOI: 10.1016/j.heliyon.2024.e38230) -- complement-and-coagulation-cascade gene-based prognostic model in HCC correlating with immune microenvironment and drug sensitivity, built directly from the same gene set.

### 2. Neutrophil Extracellular Trap Formation
recur=13 size=183 eff=4.75 hist_frac=35%
NETs are mechanistically implicated in HCC growth and metastasis, not merely a generic inflammatory
readout: HBV-driven S100A9-TLR4/RAGE-ROS signaling elevates NET formation and this directly promotes
tumor growth/metastasis in HCC models, and NET burden independently predicts HCC recurrence after
resection. The elevated histone-gene fraction here (35%) is expected and biologically meaningful rather
than an artifact -- NETosis is defined by extrusion of decondensed chromatin (histones + DNA), so histone
genes are core pathway members, unlike their appearance as passive noise in unrelated gene sets.

Citations:
- Zhan X, Wu R, Kong XH, et al., *Cancer Communications*, 2023 (PMID: 36346061; DOI: 10.1002/cac2.12388) -- HBV-mediated S100A9-TLR4/RAGE-ROS cascade elevates NETs, which facilitate HCC growth and metastasis (mechanistic, in vivo).
- Yang LY, Luo Q, Lu L, et al., *Journal of Hematology & Oncology*, 2020 (PMID: 31907001; DOI: 10.1186/s13045-019-0836-0) -- increased NETs promote HCC metastatic potential via tumorous inflammatory response; NET markers associate with poorer postoperative outcome.

### 3. Platelet Degranulation
recur=9 size=123 eff=3.87 hist_frac=0%
Platelets are causally implicated in the progression from NAFLD/NASH to HCC, not just a generic
hemostasis readout. Genetic or pharmacological platelet depletion/inhibition (targeting platelet GPIbα)
reduces NASH-associated liver injury and blocks progression to liver cancer in mouse models, directly
tying platelet activation biology to hepatocarcinogenesis. This mechanism has direct relevance to the
Roskams-Hieter cohort given NASH/cirrhosis is a major etiologic route to HCC.

Citations:
- Malehmir M, Pfister D, Gallage S, et al., *Nature Medicine*, 2019 (PMID: 30936549; DOI: 10.1038/s41591-019-0379-5) -- platelet GPIbα is a causal mediator of NASH and subsequent liver cancer; platelet-targeted intervention (aspirin/clopidogrel, GPIbα blockade) attenuates NASH-to-HCC progression in mice.

### 4. RHO GTPase Cycle
recur=15 size=439 eff=5.02 hist_frac=0%
Rho-family GTPases (RhoA, Rac1, Cdc42) are recurrently overexpressed/hyperactivated in HCC and
mechanistically drive tumor cell migration, invasion, and metastasis; Rac1 in particular is under active
investigation as a therapeutic target in HCC specifically (not a pan-cancer generalization). RHOA/RAC1/
CDC42 GTPase Cycle sub-terms also individually recur in this cohort's top-40 list (recur 8-11 each),
reinforcing convergent signal on this family rather than a single noisy sub-term.

Citations:
- Sauzeau V, Beignet J, Vergoten G, Bailly C, *Pharmacological Research*, 2022 (PMID: 35405309; DOI: 10.1016/j.phrs.2022.106220) -- review of Rac1 overexpression/hyperactivation as a driver and therapeutic target specifically in HCC.

### 5. Lysosome
recur=9 size=127 eff=4.09 hist_frac=0%
Lysosomal biogenesis (via the master regulator TFEB) is directly implicated in two of the dominant HCC
etiologies represented in a liver-cancer cohort: alcohol-associated liver carcinogenesis (loss of hepatic
TFEB accelerates alcohol-driven HCC in mouse models) and HBV-driven HCC (HBx viral protein suppresses TFEB
to stabilize a pro-tumorigenic integrin). This gives lysosome-pathway signal a specific, etiology-linked
mechanistic story rather than a generic organelle/metabolism readout.

Citations:
- Chao X, Wang S, Zhao K, et al., *American Journal of Pathology*, 2022 (PMID: 34717896; DOI: 10.1016/j.ajpath.2021.10.004) -- loss of hepatic TFEB (master lysosomal biogenesis regulator) attenuates protection against alcohol-associated liver carcinogenesis.
- (HBx-TFEB) *Cancers (Basel)*, 2021 (PMID: 33803301; DOI: 10.3390/cancers13051181) -- HBV X protein suppresses TFEB, stabilizing ITGB1, in HCC cells -- an HBV-specific lysosomal-dysregulation mechanism.

### 6. Chromatin Modifying Enzymes
recur=7 size=235 eff=4.41 hist_frac=12%
Epigenetic dysregulation via chromatin-modifying enzymes (EZH2/H3K27me3, KDM5C, KDM6A, IDH1/2-linked
histone demethylation) is a recurrent, mechanistically characterized driver of HCC biology, including
direct control of cellular senescence bypass in HCC cells -- distinct from this pathway acting as a
generic "epigenetics" catch-all. Histone-gene fraction is low (12%), so the signal here reflects genuine
enzyme/regulator members rather than histone-family dominance.

Citations:
- Wang K, Jiang X, Jiang Y, et al., *Journal of Experimental & Clinical Cancer Research*, 2023 (PMID: 38008711; DOI: 10.1186/s13046-023-02855-2) -- EZH2-H3K27me3-mediated silencing of miR-139-5p inhibits cellular senescence in HCC by activating TOP2A (direct chromatin-modifier-to-senescence-bypass mechanism in HCC).
- Chang S, Yim S, Park H, *Experimental & Molecular Medicine*, 2019 (PMID: 31221981; DOI: 10.1038/s12276-019-0230-6) -- review of IDH1/2, KDM5C, KDM6A as cancer driver genes acting through histone demethylation and hypoxic metabolic reprogramming.

### 7. Senescence-Associated Secretory Phenotype (SASP)
recur=7 size=80 eff=4.10 hist_frac=29%
Cellular senescence and its secretory phenotype are mechanistically load-bearing in hepatocarcinogenesis:
immune-mediated clearance of senescent pre-malignant hepatocytes normally limits liver cancer
development, and failure of this senescence-surveillance program (which SASP signaling helps mediate)
permits tumor progression. This is one of the most direct, mechanistically dissected links between any
candidate pathway here and liver cancer initiation specifically (as opposed to progression/metastasis).
Histone fraction (29%) is moderate and plausibly reflects genuine chromatin-remodeling components of the
senescence program (SAHF formation) rather than pure noise, but is noted for transparency.

Citations:
- Kang TW, Yevsa T, Woller N, et al., *Nature*, 2011 (PMID: 22080947; DOI: 10.1038/nature10599) -- "Senescence surveillance of pre-malignant hepatocytes limits liver cancer development": immune clearance of senescent hepatocytes is a tumor-suppressive checkpoint in the liver; its failure permits HCC.

## Considered and rejected

### Metabolism Of RNA / Metabolism Of Proteins / Gene Expression (Transcription) / RNA Polymerase II Transcription / Generic Transcription Pathway / Cell Cycle / Cell Cycle Checkpoints / DNA Replication / M Phase / mRNA Splicing / Spliceosome
**Rejected** as generic housekeeping/transcription-machinery terms with no liver-cancer-specific
mechanistic story beyond "cancer cells proliferate and transcribe genes" -- exactly the class of
pan-cancer readout the curation guidance excludes.

### Systemic Lupus Erythematosus
**Rejected** based on gene-set composition: 50% of this KEGG term's member genes are core histone genes
(H1/H2A/H2B/H3/H4 family), the same autoantigen-driven composition artifact identified and rejected in
the Pancreatic Cancer review. Not real liver-cancer biology.

### Alcoholism (KEGG hsa05034)
**Rejected** despite alcohol-associated liver disease being a major HCC etiology, because the KEGG term's
own gene composition is 36% histone-family genes (H2A/H2B/H3/H4), driven by the term's origin in
addiction-neuroscience gene sets (dopamine/CREB signaling plus chromatin genes), not liver-specific
alcohol metabolism (ADH/ALDH genes are not the dominant contributors). The biological story (ALD -> HCC)
is real, but this specific gene set is not the right vehicle for it.

### DNA Methylation
**Rejected** based on gene-set composition: 70% histone-family genes -- the term is nominally about
methylation machinery but is numerically dominated by core histone genes, making convergence signal here
largely indistinguishable from a chromatin/nucleosome-packaging readout rather than genuine DNA
methylation biology. Chromatin Modifying Enzymes (12% histone fraction, selected above) better isolates
the intended epigenetic-regulator signal.

### Hemostasis / Response To Elevated Platelet Cytosolic Ca2+
**Considered but folded into Platelet Degranulation** -- both are large, overlapping Reactome parent/
sibling terms covering substantially the same platelet-activation biology already captured (and better
cited) under Platelet Degranulation; kept as one representative pathway rather than three redundant
entries.

### RAC1 GTPase Cycle / CDC42 GTPase Cycle / RHOA GTPase Cycle
**Folded into RHO GTPase Cycle** -- these are Reactome sub-pathways of the parent RHO GTPase Cycle term,
all independently recurrent in the top-40 list (recur 8-11), so the parent term is used as the single
representative to avoid triple-counting the same GTPase-family signal.

### Olfactory Transduction / Expression And Translocation Of Olfactory Receptors / Olfactory Signaling Pathway
**Rejected** -- large gene families (hundreds of paralogous olfactory receptor genes) with no known
role in liver biology or HCC; almost certainly reflects gene-family size/co-regulation artifacts in
cfRNA data rather than disease signal.

### Herpes Simplex Virus 1 Infection
**Rejected** -- despite viral hepatitis (HBV/HCV) being a dominant HCC etiology, no specific HSV-1
literature link to HCC was sought or found to justify this term; it likely reflects broad antiviral/
innate-immune gene overlap rather than a liver-virus-specific mechanism, and pulling in an unrelated
virus family would be a weaker, unsupported claim.

### Complement And Coagulation Cascades sub-terms considered redundant
None found duplicated in the top-40 beyond the parent term; no action needed.

## Raw search log

PubMed E-utilities (esearch/esummary), queried directly, best-match PMIDs used above marked *:

1. Complement/coagulation + HCC: 33718121* (Front Oncol 2020, review), 39391504* (Heliyon 2024, prognostic model), 33911900, 39357793, 37116239 (not used)
2. NETs + HCC + metastasis: 36346061* (Cancer Commun 2023), 31907001* (J Hematol Oncol 2020), 38381538, 39529085, 38670307 (not used)
3. Platelet + NASH + HCC: 30936549* (Nat Med 2019, Malehmir/Heikenwalder -- only strong hit)
4. Rho GTPase + HCC: 35405309* (Pharmacol Res 2022, Rac1 review); earlier RhoA/Rac1/Cdc42-combined query returned off-target hits (CTHRC1 23922981, arsenic trioxide 32096383, DLC-1 27604574) not used
5. Lysosome/TFEB + HCC: 34717896* (Am J Pathol 2022, alcohol-TFEB), 33803301* (Cancers 2021, HBx-TFEB); generic autophagy/mitophagy hits (34890308, 37733919) considered but not used -- mitophagy is mechanistically distinct from bulk lysosomal biogenesis
6. Chromatin modifying enzymes + HCC: 38008711* (J Exp Clin Cancer Res 2023, EZH2-senescence), 31221981* (Exp Mol Med 2019, IDH1/2-KDM review), 40940750, 24018165, 27294413 (not used)
7. Senescence surveillance + liver cancer: exact-title search located 22080947* (Nature 2011, Kang et al.) via the eLife registered-report record (25621566) that cites it; confirmed as the original primary paper, not the registered report itself
8. Composition check (histone-family fraction) computed directly from `gene_set_matrix.pkl` symbol membership for all rejected/flagged terms, not from literature
