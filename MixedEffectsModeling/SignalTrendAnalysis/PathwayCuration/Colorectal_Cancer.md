# Colorectal Cancer Pathway Curation

Source: GSEA normative ranking, `normative__Colorectal_Cancer__Chen_et_al..csv`, FDR q<0.05 (358 terms total before curation); pan-cancer hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

Note: the top ~130 FDR-significant terms by |NES| in this ranking are dominated by generic proliferation/DNA-replication/RNA-processing machinery (APC/C-mediated mitotic degradation, nuclear RNA export, spliceosome, OXPHOS/mitochondrial import) shared near-identically across all four Chen et al. cancer phenotypes examined in this batch -- still treated as non-specific "actively dividing tumor" signal, not relaxed. A further set of nominally disease-relevant terms (`Regulation Of RUNX3/RUNX2/PTEN Stability`, `Regulation Of RAS By GAPs`, `Hh Mutants...`) remain dropped after inspecting `Lead_genes`: the leading-edge is >80% generic 26S proteasome subunits (PSMA/PSMB/PSMC/PSMD family), not RUNX3/PTEN/Hedgehog biology -- this proteasome-subunit artifact is not relaxed either. Below the artifact layer, several pan-cancer hallmark pathways (Warburg metabolism, hypoxia/HIF, WNT, mTORC1, RAS-MAPK) have clean leading edges (<15% housekeeping-prefix genes) and are now kept under the relaxed criterion.

## Selected Pathways

### Mismatch repair
- NES: -1.932, FDR q: 0.00021
- Literature: Mismatch-repair-deficient (dMMR)/microsatellite-instable (MSI) tumors are a canonical, well-established molecular subtype of colorectal cancer (Lynch syndrome and sporadic MSI-CRC), with direct clinical relevance to immunotherapy response and remodeling of the immune/stromal compartment (PMID 37172580, Cancer Cell 2023).
- Lead genes (top ~10): SSBP1, POLD1, RFC4, POLD4, RPA1, RFC2, RPA2, MSH6, RFC1, PCNA, MLH3, POLD2, MLH1

### Repression Of WNT Target Genes R-HSA-4641265
- NES: -1.564, FDR q: 0.0350
- Literature: WNT/beta-catenin signaling is the founding oncogenic pathway of colorectal cancer (APC loss -> beta-catenin stabilization drives adenoma initiation in the canonical Fearon-Vogelstein model); Zhan T, Rindtorff N, Boutros M, "Wnt signaling in cancer," Oncogene 2017 (PMID 27617575), reviews WNT pathway dysregulation with CRC as the primary example.
- Lead genes: LEF1, CTBP2, MYC, TLE5, TCF7, HDAC1, AXIN2

### Glycolysis / Gluconeogenesis
- NES: -1.604, FDR q: 0.0243
- Literature: Aerobic glycolysis (the Warburg effect) is a pan-cancer metabolic hallmark supporting proliferative biomass demand; Vander Heiden MG, Cantley LC, Thompson CB, "Understanding the Warburg Effect: The Metabolic Requirements of Cell Proliferation," Science 2009 (PMID 19460998).
- Lead genes (top ~10): TPI1, ALDH9A1, ALDOA, PGK1, AKR1A1, GPI, GAPDH, PDHA1, PDHB, PGAM1, ENO1, LDHA, PFKP, HK2

### HIF-1 signaling pathway
- NES: -1.555, FDR q: 0.0375
- Literature: Hypoxia-inducible factor signaling drives angiogenesis, metabolic reprogramming, and invasion across solid tumors including CRC; Semenza GL, "HIF-1 mediates metabolic responses to intratumoral hypoxia and oncogenic mutations," J Clin Invest 2013 (PMID 23543062).
- Lead genes (top ~10): ALDOA, RPS6KB1, NFKB1, PGK1, SERPINE1, GAPDH, VHL, EGLN2, PIK3CD, MAPK1, TFRC, MTOR, AKT1

### Amino Acids Regulate mTORC1 R-HSA-9639288
- NES: -1.549, FDR q: 0.0398
- Literature: mTORC1 hyperactivation is a convergent driver of tumor cell growth downstream of PI3K/AKT/RAS across cancer types; Populo H, Lopes JM, Soares P, "The mTOR Signalling Pathway in Human Cancer," Int J Mol Sci 2012 (PMID 22408430).
- Lead genes (top ~10): ATP6V1E1, LAMTOR4, LAMTOR2, RHEB, LAMTOR5, RPTOR, MTOR, MIOS, DEPDC5, FLCN, RRAGB

### MAP3K8 (TPL2)-dependent MAPK1/3 Activation R-HSA-5684264
- NES: -1.570, FDR q: 0.0337
- Literature: RAS-RAF-MEK-ERK (MAPK) signaling is one of the most frequently activated pathways across solid tumors, including via KRAS mutation in CRC; Dhillon AS, Hagan S, Rath O, Kolch W, "MAP kinase signalling pathways in cancer," Oncogene 2007 (PMID 17496922).
- Lead genes: NFKB1, IKBKB, SKP1, BTRC, MAP2K1, CUL1, MAP3K8, IKBKG

## Dropped candidates (GSEA-significant, no adequate literature support or still artifactual)
- Regulation Of RUNX3 Expression And Activity R-HSA-8941858: NES -1.899 -- leading-edge is >80% proteasome subunits (PSMA/PSMB/PSMC/PSMD), not RUNX3-specific genes; label is nominal, not reflective of RUNX3 biology.
- Hh Mutants Are Degraded By ERAD R-HSA-5362768 / Hh Mutants Abrogate Ligand Secretion R-HSA-5387390: NES -1.887 / -1.874 -- same proteasome-subunit artifact as above; not a genuine Hedgehog-pathway signal.
- Regulation Of PTEN Stability And Activity R-HSA-8948751: NES -1.825 -- proteasome-subunit artifact (same lead-gene list); PTEN itself appears only as one of ~30 genes dominated by PSM* subunits.
- Regulation Of RAS By GAPs R-HSA-5658442: NES -1.805 -- proteasome-subunit artifact; HRAS/RASA4 present but swamped by the same PSM* leading edge.
- Regulation Of Ornithine Decarboxylase (ODC) R-HSA-350562 / Metabolism Of Polyamines R-HSA-351202: NES -1.863 / -1.798 -- proteasome-subunit artifact, not polyamine-specific genes.
- Non-alcoholic fatty liver disease: NES -1.845 -- disease-comorbidity gene set, not a mechanistic pathway suitable for a leading-edge Sankey; too indirect a link to CRC for this use.
- Defective TPR May Confer Susceptibility Towards Thyroid Papillary Carcinoma (TPC) R-HSA-5619107: NES -2.020 -- organ-mismatched term (thyroid-specific), no CRC-specific rationale; leading edge is generic nucleoporins (NUP*).
- Broad set of top-ranked terms (APC/C:Cdc20/Cdh1-mediated mitotic degradation, nuclear RNA/ribonucleoprotein export, Complex I biogenesis/cristae formation, viral Rev/HIV host-interaction terms, translesion synthesis, TC-NER, RNA Pol III initiation): NES up to -2.15 -- generic proliferation/OXPHOS/DNA-repair/RNA-processing housekeeping machinery, near-identical across all four phenotypes examined, not CRC-specific and not a distinct cancer hallmark (this is the "actively dividing cell" baseline, not a targeted mechanism).
- Cooperation Of Prefoldin And TriC/CCT In Actin And Tubulin Folding / Folding Of Actin By CCT/TriC: NES -1.65/-1.63 -- generic chaperonin/cytoskeletal-folding housekeeping, no CRC-specific or pan-cancer-hallmark citation found beyond ubiquitous protein-folding demand of any proliferating cell.
- RUNX1 Regulates Transcription Of Genes Involved In WNT Signaling R-HSA-8939256: NES -1.521 -- only 3 lead genes (RUNX1, AXIN1, ESR1), too sparse to support as a distinct pathway beyond the WNT term already kept.
