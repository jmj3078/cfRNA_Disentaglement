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
