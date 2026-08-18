# Esophagus Cancer Pathway Curation

Source: GSEA normative ranking, `normative__Esophagus_Cancer__Chen_et_al..csv`, FDR q<0.05 (377 terms total before curation); pan-cancer hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

Note: this phenotype shows the weakest disease-specific signal of the four examined in this batch. The vast majority of FDR-significant terms are generic proliferation/DNA-repair (NER, APC/C-mediated degradation)/RNA-processing/ER-stress-autophagy machinery shared near-identically with CRC/Lung/Stomach, or (as with `Signaling By FGFR2 IIIa TM`, checked here too) proteasome/RNA-Pol-II-subunit-dominated leading edges under a disease-sounding label -- these are still excluded, not relaxed. Below the artifact layer, a handful of pan-cancer growth-signaling and cell-cycle hallmarks have clean-enough leading edges to keep under the relaxed criterion; an `Autoimmune thyroid disease` term and a SARS-CoV-2 immune term were checked and dropped as organ-mismatched / NUP-artifact-contaminated respectively, per the still-active organ-mismatch and artifact exclusions.

## Selected Pathways

### Mismatch repair
- NES: -1.852, FDR q: 0.00197
- Literature: Mismatch-repair-deficient (dMMR)/MSI esophageal adenocarcinoma is a recognized, clinically relevant minority subtype with reported immunotherapy sensitivity; MSI/dMMR status is profiled as part of comprehensive molecular characterization of gastrointestinal adenocarcinomas including esophageal/EGJ tumors (PMID 29622466, Cancer Cell 2018). Support is moderate (dMMR is a minority, not defining, esophageal cancer subtype) relative to its role in CRC/gastric cancer.
- Lead genes (top ~10): RPA1, POLD4, RFC4, SSBP1, RPA2, POLD1, RFC2, POLD3, MSH6, RPA3, RPA4, MLH3, RFC5

### Telomere Extension By Telomerase R-HSA-171319
- NES: -1.635, FDR q: 0.0264
- Literature: Telomerase reactivation (limitless replicative potential) is a core Hanahan & Weinberg cancer hallmark, general across epithelial malignancies including esophageal cancer; Shay JW, "Role of Telomeres and Telomerase in Aging and Cancer," Cancer Discov 2016 (PMID 27977688).
- Lead genes: DKC1, TERF2IP, SHQ1, RUVBL2, RUVBL1, PPP6R3, NHP2, TINF2

### Cell Cycle R-HSA-1640170
- NES: -1.726, FDR q: 0.0106
- Literature: Uncontrolled cell-cycle progression is a core cancer hallmark (sustained proliferative signaling); Hanahan D, Weinberg RA, "Hallmarks of Cancer: The Next Generation," Cell 2011 (PMID 21376230). Leading edge includes genuine mitotic/replication machinery (SMC1A, MRE11, CCND3, MAD1L1) alongside partial proteasome contamination (<30%), kept because named-pathway genes dominate.
- Lead genes (top ~10): CDKN2D, HDAC8, DCTN2, CCND3, MAD1L1, RFC4, MRE11, SMC1A, RBBP4

### Regulation Of PLK1 Activity At G2/M Transition R-HSA-2565942
- NES: -1.615, FDR q: 0.0306
- Literature: PLK1 is a master mitotic kinase whose overexpression drives the G2/M transition in a wide range of solid tumors and is an actively pursued oncology drug target; Strebhardt K, "Multifaceted polo-like kinases: drug targets and antitargets for cancer therapy," Nat Rev Drug Discov 2010 (PMID 20671765). Leading edge is genuine centrosome/mitotic-spindle machinery, not a proteasome artifact.
- Lead genes: DCTN2, TUBA4A, HAUS1, SFI1, CEP70, CEP57, ODF2, CETN2, HAUS7, CEP41, DYNC1I2, PCNT, CCNB1

### MAP3K8 (TPL2)-dependent MAPK1/3 Activation R-HSA-5684264
- NES: -1.771, FDR q: 0.0061
- Literature: RAS-RAF-MEK-ERK (MAPK) signaling is one of the most frequently activated growth-signaling pathways across solid tumors; Dhillon AS, Hagan S, Rath O, Kolch W, "MAP kinase signalling pathways in cancer," Oncogene 2007 (PMID 17496922).
- Lead genes: IKBKG, SKP1, IKBKB, CHUK, CUL1, MAP3K8, MAP2K1, NFKB1

## Dropped candidates (GSEA-significant, no adequate literature support or still artifactual)
- Signaling By FGFR2 IIIa TM R-HSA-8851708: NES -1.798 -- leading edge is entirely RNA Pol II subunits (POLR2D/E/G/H/J/L) and NCBP1/2, not FGFR2 signaling genes; not a genuine FGFR2 signal, and FGFR2 amplification is a gastric- (not esophageal-) cancer-associated subtype in the literature regardless.
- Broad set of top-ranked terms (Autodegradation Of Cdh1 By Cdh1:APC/C, APC/C:Cdc20-mediated degradation family, Complex I Biogenesis, GSK3B/BTRC-mediated NFE2L2 degradation, KEAP1-NFE2L2 Pathway, TC-NER/nucleotide excision repair family, Metabolism Of RNA): NES up to -2.30 -- generic proliferation/DNA-repair/proteostasis machinery, near-identical across all four phenotypes examined in this batch, not esophagus-specific and not a distinct targetable hallmark.
- Macroautophagy R-HSA-1632852 / Late Endosomal Microautophagy R-HSA-9615710 / Protein processing in endoplasmic reticulum / IRE1alpha Activates Chaperones R-HSA-381070 / XBP1(S) Activates Chaperone Genes R-HSA-381038: NES -1.72 to -1.75 -- generic ER-stress/unfolded-protein-response and autophagy machinery, no esophageal-cancer-specific or clean pan-cancer-hallmark citation found beyond general proteostasis-in-cancer biology.
- Autoimmune thyroid disease: NES -1.674 -- organ-mismatched disease-comorbidity gene set (generic HLA class I/II + cytokine genes), not a mechanistic pathway; same exclusion basis as the "Non-alcoholic fatty liver disease" term dropped in the CRC curation.
- SARS-CoV-2 Activates/Modulates Innate And Adaptive Immune Responses R-HSA-9705671: NES -1.616 -- leading edge is roughly half generic nucleoporins (NUP35/37/42/43/85/88/93/107/160), the same NUP-transport artifact documented across all four phenotypes; the immune-signaling genes present (IRF3, STING1, TBK1) are too diluted by this artifact to trust the term's biology.
- Hh Mutants Are Degraded By ERAD / Hh Mutants Abrogate Ligand Secretion: carries the same proteasome-subunit leading-edge artifact documented for the other three phenotypes in this batch; excluded on that basis.
