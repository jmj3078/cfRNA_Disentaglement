# Gene-level FDR threshold: rationale for reporting q=0.05/0.10/0.15/0.20 side by side

## Why not just one q

Per-patient normative Z-scores and DESeq2's pooled group-comparison statistic differ in
**power**, not in what a given FDR q means. BH-FDR at q controls the same thing (expected
false-discovery proportion among rejections) regardless of whether the underlying p-value came
from an n=1-vs-reference-distribution test or a many-vs-many group test, *provided* the p-values
are correctly calibrated (verified here via held-out HC exceedance/skew-kurtosis diagnostics,
`core/calibration.py`). So relaxing q to "compensate" for the normative model's structurally
wider per-patient predictive interval is not fixing an unfairness in validity -- it's a deliberate
sensitivity/specificity tradeoff, and must be labeled as such rather than presented as a
like-for-like comparison with a DESeq2 result computed at q=0.05.

## Empirical sweep (union K=1, no outlier-patient filtering applied)

Sensitivity (recall against the Open Targets top-300 reference) rises monotonically with q; the
hypergeometric enrichment p-value does not -- q=0.05 is not the point of strongest statistical
evidence for most phenotypes tested, but q=0.50 destroys the signal entirely (near-complete
genome coverage dilutes enrichment to nothing). See `Benchmark/db_hit_compare.py:gene_level_db_hits(q=...)`
and cached `gene_db_hit_rates_q*.csv` for the full tables.

| Phenotype | q=0.05 recall / p | q=0.10 | q=0.20 | q=0.30 | q=0.50 |
|---|---|---|---|---|---|
| Liver Cancer | 14.7% / 0.060 | 24.3% / 0.031 | 42.3% / 0.034 | **64.7% / 0.0051** | 92.0% / 0.450 |
| Colorectal Cancer | 8.3% / 0.120 | 15.3% / 0.033 | **30.3% / 0.0059** | 42.0% / 0.091 | 74.0% / 0.380 |
| Tuberculosis | 2.7% / 0.189 | 5.0% / 0.064 | 9.0% / 0.151 | **17.0% / 0.049** | 42.0% / 0.801 |
| Pancreatic Cancer | 7.3% / 0.696 | 16.7% / 0.054 | 31.0% / 0.034 | **51.0% / 0.0054** | 87.3% / 0.052 |
| Pre-eclampsia | 4.0% / 0.430 | 6.7% / 0.211 | 9.3% / 0.511 | 14.7% / 0.541 | 36.0% / 0.399 |

Bold = lowest (best) hypergeometric p at that phenotype among the q's tested. Pre-eclampsia never
reaches significance at any q -- its problem is not threshold-related.

## External precedent: GSEA's own discovery-tier convention

Subramanian A, Tamayo P, Mootha VK, et al. "Gene set enrichment analysis: A knowledge-based
approach for interpreting genome-wide expression profiles." *PNAS* 2005;102(43):15545-15550.
DOI [10.1073/pnas.0506580102](https://doi.org/10.1073/pnas.0506580102).

GSEA does not threshold individual genes at all -- it runs a weighted running-sum statistic over
the full ranked gene list, sidestepping single-gene multiple-testing entirely. Where GSEA *does*
apply a cut is at the gene-set level: NES vs. a phenotype-permutation null, with the field-standard
reporting convention being **FDR q < 0.25** (not the usual 0.05), explicitly because gene-set
testing is framed as hypothesis-generating rather than confirmatory, and the tests are highly
correlated (overlapping gene sets). The paper's own guidance is to rank by NES and inspect the
leading-edge subset for follow-up validation rather than treat the q<0.25 list as a final result.

**How this applies here**: a relaxed q (0.10-0.30) tier for normative gene-level calls is
defensible on the same footing -- as an explicitly-labeled discovery/candidate-panel tier, cross-
validated against DB references and used to prioritize genes for follow-up (literature check,
recurrence-across-patients, pathway placement), never substituted into a head-to-head "which
method wins" comparison against DESeq2's own q=0.05 result. The two q's are not making the
methods "equally valid" -- validity is already equal at any shared q; a relaxed q only trades
precision for recall within the normative arm's own discovery-tier output.

## Rule for use

- **q=0.05**: confirmatory tier -- the only tier compared directly against DESeq2 (also at padj<0.05).
- **q=0.10-0.20** (0.20 chosen close to GSEA's 0.25 convention, empirically near the best-recall/
  still-significant sweet spot above): discovery/candidate tier -- for panel-building, pathway
  placement, follow-up literature review. Report recurrence-across-patients alongside any gene
  pulled from this tier; do not cite counts from this tier as evidence of normative-vs-DESeq2
  superiority.
