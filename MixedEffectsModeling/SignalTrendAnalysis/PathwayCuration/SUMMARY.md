# Pathway Curation Summary (Group-level GSEA + Literature/DB Validation)

Group-level GSEA (`Benchmark/gsea_cache/normative__*.csv`, Stouffer-Z ranking, FDR q<0.05)
filtered for housekeeping/proteasome-leading-edge artifacts, then cross-validated per
pathway against literature/DB via `/paper-lookup` and `/database-lookup`. Two passes per
phenotype: (1) disease-subtype-specific literature required, (2) relaxed to allow
pan-cancer / pan-inflammatory hallmark mechanisms with only general (non-subtype-specific)
support, provided the leading-edge gene composition is genuinely that pathway's own biology
(not a proteasome/ribosome/spliceosome artifact wearing a disease-sounding label -- this
exclusion was never relaxed). Per-phenotype detail, citations, and dropped-candidate
reasoning: `Benchmark/PathwayCuration/<phenotype>.md`.

## Kept pathway counts

| Phenotype | Kept | Category |
|---|---|---|
| Tuberculosis | 12 | inflammatory/infectious |
| Liver Cancer (Roskams-Hieter) | 11 | cancer |
| Lung Cancer | 7 | cancer |
| Pancreatitis | 7 | inflammatory |
| Pre-eclampsia | 6 | inflammatory/vascular |
| Colorectal Cancer | 6 | cancer |
| Esophagus Cancer | 5 | cancer |
| Stomach Cancer | 5 | cancer |
| Pancreatic Cancer | 5 | cancer |
| Liver Cancer (Chen) | 3 | cancer (small n, caveat below) |

## Cross-phenotype overlap (exact GSEA Term match)

**Inflammatory triad (TB / Pancreatitis / Pre-eclampsia):**
- TB & Pancreatitis share 3: `Interferon Alpha/Beta Signaling`, `Interferon Gamma Signaling`,
  `Neutrophil Degranulation` -- a real shared innate-immune/IFN-neutrophil axis across two
  independent infectious/inflammatory cohorts.
- TB & Pre-eclampsia: 0 overlap. Pancreatitis & Pre-eclampsia: 0 overlap. Pre-eclampsia was
  explicitly checked for the same IFN/chemokine/neutrophil terms found in TB/Pancreatitis --
  present in its GSEA ranking but not FDR-significant there, so absence reflects this
  cohort's statistical power/effect size, not a mechanism search that wasn't attempted.

**Cancer group (6 phenotypes):** heavy overlap, but it is *pan-cancer hallmark* overlap by
construction, not disease-specific: `Mismatch repair` (CRC/Esophagus/Lung/Stomach),
`MAP3K8 (TPL2)-dependent MAPK1/3 Activation` (CRC/Esophagus/Stomach), `Cell Cycle` /
`Telomere Extension By Telomerase` (Esophagus/Stomach), `Glycolysis / Gluconeogenesis`
(CRC/Lung). The disease-specific signal per cancer type lives in each phenotype's *unique*
pathways (e.g. Liver Cancer/Roskams-Hieter: Hippo, RAC1/RHOB/RHOU GTPase cycles, VEGF,
SMAD2/3/4, scavenger-receptor lipid uptake -- 9/11 unique; Pancreatic Cancer: ECM-receptor
interaction, IGFBP axis, RHOC GTPase -- desmoplastic-stroma/PDAC-specific).

**Cross-category:** `Signaling By Hippo` is the only term shared between the inflammatory
and cancer groups (Liver Cancer/Roskams-Hieter and Pre-eclampsia) -- plausibly independent
hits on the same growth-regulation pathway in two unrelated contexts (tumor proliferation vs
trophoblast invasion) rather than a shared disease mechanism.

## Oncogenesis-specificity of the recurring cancer terms

The 4 recurring cancer terms above (`Mismatch repair`, `MAP3K8 (TPL2)-dependent MAPK1/3
Activation`, `Cell Cycle`, `Telomere Extension By Telomerase`, `Glycolysis / Gluconeogenesis`)
are pan-cancer proliferation/metabolism hallmarks, not mechanistically specific to
oncogenesis (tumor-suppressor inactivation / oncogene activation) -- their citations are
general reviews ("X is frequently altered across solid tumors"), not disease-specific driver
mechanisms.

A more directly oncogenic, reproducible signal exists but sits outside this section's
exact-Term-match grouping: the **Hippo-pathway-inactivation / YAP1-TAZ-hyperactivation axis**,
a bona fide tumor-suppressor-pathway-inactivation mechanism (not just "frequently altered").

- `YAP1- And WWTR1 (TAZ)-stimulated Gene Expression` recurs, term-for-term, across **Liver
  Cancer (Chen) + Pancreatic Cancer** -- 2 independent cancer cohorts, both NES positive
  (+1.98 / +1.92), overlapping lead genes (WWTR1, YAP1, TEAD1/TEAD4), each with
  disease-specific citations (PDAC: gemcitabine resistance/stemness via YAP1-c-Jun, PMID
  37143164; aerobic glycolysis via YAP1-EGLN2, PMID 39647834). **This 2-cohort reproduction
  holds on its own, without needing any merge** -- it is exactly the same kind of
  exact-Term-match recurrence as the inflammatory triad or the pan-cancer hallmarks above.
- A 3rd cancer cohort, Liver Cancer (Roskams-Hieter), independently kept the *upstream* half
  of the same axis, `Signaling By Hippo` (NES +2.07, HBV-HCC YAP1-ubiquitylation-loss
  mechanism, PMID 36643034) -- mechanistically the same pathway, but a different GSEA `Term`
  string, so it does not join the YAP1/WWTR1 group under exact-match and was left out.
  Folding it in would raise the reproduction to 3 cancer cohorts, but requires a judgment
  call this document does not make automatically: `Signaling By Hippo` is *also* the literal
  term shared with Pre-eclampsia in the cross-category note just above, in the **opposite**
  NES direction (-2.01, YAP decrease in trophoblast, different biology) -- so merging on
  the term string would conflate a real 3-cohort oncogenic reproduction with an unrelated
  cross-category coincidence. Kept split for now; a manual axis-level merge (Hippo +
  YAP1/WWTR1, cancer cohorts only, excluding Pre-eclampsia) is a defensible follow-up if the
  downstream Sankey wants to represent this as one 3-cohort node.

**Sample-level caveat (applies to all of the above):** re-checking `path_sig` (per-sample
BH-FDR hypergeometric ORA on leading-edge genes, `PerSamplePathwayAnalysis/sig.pkl`) for the
YAP1/WWTR1-Hippo axis gives 0% of samples significant in every one of the 3 cancer cohorts
(0/10 Chen, 0/72 Pancreatic, 0/28 Roskams-Hieter) -- same pattern as the pan-cancer hallmark
terms and the inflammatory triad. The reproduction above is a **group-level GSEA**
(Stouffer-Z, FDR q<0.05) finding; it does not mean individual patients show a BH-significant
hit on this pathway. Do not represent group-level reproducibility and sample-level
significance as the same claim in the Sankey.

## Inflammatory panel expansion (relaxed nominal-p cross-check, for the flagship Sankey)

The strict rule above (exact `Term` match, both phenotypes' own FDR<0.05 kept list) gives only
3 inflammatory pathways vs. 8 for cancer -- a count mismatch driven by group size (3
inflammatory phenotypes vs. 6 cancer subtypes), not by inflammatory disease actually
converging less. To build a size-matched 8-pathway panel for the flagship Sankey (patient ->
gene -> pathway -> subtype -> category), each of the 25 TB/Pancreatitis/Pre-eclampsia
FDR-kept terms was cross-checked against the *other two* phenotypes' full raw GSEA ranking
(`Benchmark/gsea_cache/normative__<phenotype>.csv`, all 2054 terms, not just their own
FDR<0.05 shortlist) for **nominal `NOM p-val` < 0.05** -- looser than the FDR-significance
bar used everywhere else in this document, so treat panel membership below as suggestive
convergence, not a corrected-significance claim the way the strict "kept" list is.

5 terms cleared this bar and are genuinely core innate-immune/inflammatory mechanisms (not
tangential organ-specific hits):

| term | TB (own list?) | cross-phenotype nominal p |
|---|---|---|
| Neutrophil Extracellular Trap Formation | kept | Pancreatitis p=0.004, Pre-eclampsia p=0.000 (all 3 phenotypes) |
| Bacterial Invasion Of Epithelial Cells | kept | Pancreatitis p<0.001 |
| Regulation Of Actin Dynamics For Phagocytic Cup Formation | kept | Pancreatitis p<0.001 |
| Chemokine Signaling Pathway (KEGG) | kept | Pancreatitis p<0.001 |
| Interleukin-1 Family Signaling | kept | Pancreatitis p<0.001 |

Other candidates that also cleared nominal p<0.05 were dropped as off-theme despite the
numeric hit: Pre-eclampsia's `VEGFA-VEGFR2 Pathway`, `DNA Methylation`, and `Response To
Elevated Platelet Cytosolic Ca2+` are vascular/epigenetic, not inflammatory-mechanism, and
`Defective CFTR...`/`Metabolism Of Polyamines` (Pancreatitis) are organ-specific rather than
a shared immune axis. `Signaling By Hippo` also clears nominal significance between
Pancreatitis and Pre-eclampsia here (p=0.000/0.028) -- worth noting since it's the same term
flagged above as "cross-category" against Liver Cancer/Roskams-Hieter, but excluded from this
panel for the same off-theme reasoning.

Final flagship inflammatory panel (8): Interferon Alpha/Beta Signaling, Interferon Gamma
Signaling, Neutrophil Degranulation (strict, unchanged) + the 5 above.

## Caveats

- **Liver Cancer (Chen et al., n=10)**: 82% of its FDR-significant GSEA terms (419/513, the
  entire negative-NES tail) share a ~40-gene proteasome/spliceosome/OXPHOS leading-edge
  block regardless of nominal pathway label -- a small-n convergence artifact, excluded en
  masse. The 3 pathways kept survived this filter and 2 of 3 independently match
  Roskams-Hieter's list (Complement/coagulation, scavenger-receptor lipid uptake) plus
  YAP1/TAZ (downstream half of Roskams-Hieter's kept Hippo term) -- treat as
  lower-confidence/exploratory relative to Roskams-Hieter.
- **Pathway-label vs leading-edge mismatch** was a recurring, non-trivial finding across
  almost every phenotype: several disease-sounding Reactome terms (e.g. `Regulation Of
  RUNX3 Expression`, `Hh Mutants...`, MHC Class II Antigen Presentation in TB) turned out to
  be driven >80% by generic proteasome subunits or trafficking machinery unrelated to the
  named pathway's actual biology. Always verify `Lead_genes` composition before trusting a
  GSEA term label.
- Overlap analysis above is exact-string match on GSEA `Term`; different gene-set-library
  entries describing the same underlying biology under a different name would not be
  detected as overlapping.

## Next step

Sample-level inspection: for each kept pathway, pull its leading-edge genes' per-sample
Z-scores within that phenotype's cohort and look for patient-level patterns (subgroups,
magnitude heterogeneity) underneath the group-level GSEA signal -- this is what will
actually drive the patient-tier of the Sankey.
