# Stomach Cancer Pathway Curation

Source: GSEA normative ranking, `normative__Stomach_Cancer__Chen_et_al..csv`, FDR q<0.05 (369 terms total before curation); pan-cancer hallmark pathways included with general (non-subtype-specific) literature support where leading-edge composition is genuine.

Note: same non-specific-dominance pattern as the other three phenotypes in this batch -- the top terms by |NES| are generic proliferation/nuclear-RNA-transport/DNA-replication machinery (near-identical across CRC/Lung/Esophagus/Stomach) or proteasome-subunit-dominated leading edges under disease-sounding labels (e.g. `Signaling By FGFR2 IIIa TM` -- checked, its leading edge is entirely RNA Pol II subunits/NCBP1-2, not FGFR2 pathway genes, so dropped despite FGFR2 amplification being a real gastric cancer subtype in the literature). Below that artifact layer, several pan-cancer proliferation/growth-signaling hallmarks (telomerase reactivation, mTORC1, RAS-MAPK, cell cycle machinery) have clean (<30% housekeeping-prefix) leading edges and are kept under the relaxed criterion.

## Selected Pathways

### Mismatch repair
- NES: -2.049, FDR q: 0.00004
- Literature: MSI (microsatellite-instable) tumors are one of the four canonical TCGA molecular subtypes of gastric adenocarcinoma (Cancer Genome Atlas Research Network, "Comprehensive molecular characterization of gastric adenocarcinoma," Nature 2014, PMID 25079317), with distinct hypermutation, immune-infiltration, and treatment-response profile.
- Lead genes (top ~10): RFC4, RPA1, SSBP1, POLD1, MSH6, POLD3, RFC1, POLD4, RPA2, RFC5, MLH1, MLH3, RPA3, RFC2

### Telomere Extension By Telomerase R-HSA-171319
- NES: -1.778, FDR q: 0.0042
- Literature: Telomerase reactivation (limitless replicative potential) is one of the original Hanahan & Weinberg cancer hallmarks and is reactivated in the large majority of gastric adenocarcinomas; Shay JW, "Role of Telomeres and Telomerase in Aging and Cancer," Cancer Discov 2016 (PMID 27977688).
- Lead genes: POT1, TINF2, RUVBL1, PPP6C, CDK2, RUVBL2, TERF1, TERF2IP, NHP2, SHQ1, ACD, PPP6R3, DKC1

### mTORC1-mediated Signaling R-HSA-166208
- NES: -1.857, FDR q: 0.0013
- Literature: mTORC1 hyperactivation is a convergent growth-signaling driver across solid tumors including gastric cancer; Populo H, Lopes JM, Soares P, "The mTOR Signalling Pathway in Human Cancer," Int J Mol Sci 2012 (PMID 22408430).
- Lead genes: RPS6KB1, LAMTOR2, RHEB, LAMTOR5, EIF4E, EEF2K, LAMTOR4, MTOR, RPS6, RRAGC, YWHAB, SLC38A9, RRAGA

### Cell Cycle R-HSA-1640170
- NES: -1.713, FDR q: 0.0092
- Literature: Uncontrolled cell-cycle progression (sustained proliferative signaling / evading growth suppressors) is a core hallmark of cancer; Hanahan D, Weinberg RA, "Hallmarks of Cancer: The Next Generation," Cell 2011 (PMID 21376230). Leading edge includes real mitotic/replication machinery (SMC1A, MRE11, TOP2A-family, NCAPG2) alongside some proteasome genes, kept because the named pathway genes dominate.
- Lead genes (top ~10): RAB1A, CLASP2, SMC1A, MRE11, NCAPG2, PRIM2, RFC4, POLD1, TUBGCP2, ABRAXAS1

### MAP3K8 (TPL2)-dependent MAPK1/3 Activation R-HSA-5684264
- NES: -1.651, FDR q: 0.0185
- Literature: RAS-RAF-MEK-ERK (MAPK) signaling is one of the most frequently activated growth-signaling pathways across solid tumors; Dhillon AS, Hagan S, Rath O, Kolch W, "MAP kinase signalling pathways in cancer," Oncogene 2007 (PMID 17496922).
- Lead genes: IKBKB, NFKB1, SKP1, CUL1, MAP3K8, MAP2K1, BTRC

## Dropped candidates (GSEA-significant, no adequate literature support or still artifactual)
- Signaling By FGFR2 IIIa TM R-HSA-8851708: NES -1.701 -- leading edge is entirely RNA Pol II subunits (POLR2C/D/E/G/J/L) and NCBP1/2, not FGFR2 signaling genes; label does not reflect the actual driving signal despite FGFR2 amplification being a real, literature-supported gastric cancer subtype (PMID 38155920) in general.
- Mismatch Repair (MMR) Directed By MSH2:MSH6 (MutSalpha) R-HSA-5358565 / Mismatch Repair R-HSA-5358508: NES -1.842 / -1.734 -- near-duplicate (>70% Lead_genes overlap) of the higher-|NES| "Mismatch repair" term kept above.
- Broad set of top-ranked terms (tRNA processing, Metabolism Of RNA/Non-Coding RNA, Autodegradation Of Cdh1 By Cdh1:APC/C, Synthesis Of DNA, viral RNP nuclear export/import, RNA transport, mRNA splicing minor pathway, TC-NER, translesion synthesis): NES up to -2.27 -- generic proliferation/RNA-processing/DNA-repair/proteasome housekeeping machinery, near-identical across all four phenotypes examined in this batch, not stomach-specific and not a distinct targetable hallmark.
- Processing Of SMDT1 R-HSA-8949664: NES -1.899 -- mitochondrial calcium uniporter subunit import, generic mitochondrial housekeeping, no stomach-cancer-specific or pan-cancer-hallmark citation found.
- Telomere C-strand (Lagging Strand) Synthesis R-HSA-174417 / Telomere C-strand Synthesis Initiation R-HSA-174430 / Extension Of Telomeres R-HSA-180786: NES -2.18/-1.80/-2.13 -- near-duplicate (>70% overlap, POT1/TINF2/RFC/POLD-family) of "Telomere Extension By Telomerase" kept above.
- Amino Acids Regulate mTORC1 R-HSA-9639288 / MTOR Signaling R-HSA-165159: NES -1.72/-1.70 -- near-duplicate (>70% overlap: LAMTOR2/4/5, RHEB, RRAGA/C, MTOR) of "mTORC1-mediated Signaling" kept above.
- Cooperation Of Prefoldin And TriC/CCT In Actin And Tubulin Folding / Folding Of Actin By CCT/TriC: NES -1.67/-1.65 -- generic chaperonin housekeeping, no citable stomach or pan-cancer-hallmark link beyond ubiquitous proliferative protein-folding demand.
- FBXW7 Mutants And NOTCH1 In Cancer R-HSA-2644605: NES -1.579 -- leading edge is only RBX1/SKP1/CUL1, generic SCF-ligase components, not NOTCH1-pathway genes; same nominal-label-vs-generic-leading-edge artifact as the proteasome-subunit exclusions.
