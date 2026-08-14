# Gene-level FDR threshold: rationale for reporting q=0.05/0.10/0.15/0.20 side by side

## Caveat: Open Targets DB overlap is not ground truth

Every enrichment number in this file and in `5_deseq2_group_comparison.ipynb` scores against an
Open Targets top-300 disease-association reference. That reference is built overwhelmingly from
somatic mutation burden, GWAS loci, and tissue/solid-tumor expression + literature text-mining --
it is a **driver-gene panel**, not a plasma cfRNA abundance biomarker panel. A gene can be a real,
mechanistically important driver of a disease (e.g. a mutated tumor suppressor) without its mRNA
abundance ever changing detectably in circulating cell-free RNA, and conversely a gene with a real
cfRNA abundance shift (tissue leakage, immune activation) may have no OT driver-gene association at
all. So a "DB-hit" is evidence of *plausibility*, not proof of a true finding, and a failure to hit
the DB reference is not proof of a false one -- both directions of error are expected. This should
be stated explicitly whenever these numbers are reported, not just implied.

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

## Matched-threshold robustness check: is the normative-vs-DESeq2 gap q=0.05-specific?

`Benchmark/db_hit_compare.py:matched_threshold_sweep()` runs BOTH methods (normative_union and
DESeq2 no_covariate) at the SAME q in {0.05, 0.10, 0.15, 0.20}, hypergeometric-tested against the
same OT reference, BH-FDR corrected across the whole (method, q, phenotype) table together (52
tests). Cached: `Benchmark/matched_threshold_sweep.csv`.

| method | q=0.05 | q=0.10 | q=0.15 | q=0.20 |
|---|---|---|---|---|
| deseq2_no_covariate | 8/13 significant | 8/13 | 7/13 | 8/13 |
| normative_union | 1/14 significant | 0/14 | 2/14 | 1/14 |

**The gap is not a q=0.05 artifact.** DESeq2 holds ~7-8 of 13 testable phenotypes significant at
every threshold tested; normative_union stays at 0-2 of 14 at every threshold. Relaxing q does not
close the gap -- confirms this is not a threshold-choice problem for the normative arm, and the
q=0.05 result reported elsewhere is representative, not an unlucky pick.

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

## Group-level apples-to-apples (`6_group_level_comparison.ipynb`) -- NOT a superiority claim

Built to answer the reviewer's implicit question "is DESeq2's edge just that normative_union uses
an invalid statistical unit (K=1 min-p across patients)?" -- `db_hit_compare.py:group_level_z_test`/
`group_level_pathway_gsea` apply DESeq2's OWN design (one p-value per gene/pathway from a single
group-level test) to normative Z-scores via Stouffer's Z (`sum(Z)/sqrt(n)`, grouped by
`(phenotype, study)` -- NOT phenotype alone, since Liver Cancer pools 3 technically distinct
studies whose technical variance would otherwise leak into the ranking statistic) and real
preranked GSEA (Subramanian 2005) on that same Stouffer ranking.

Result at q=0.05: `deseq2_no_covariate` 9/13 phenotypes significant, `normative_group_z` 7/14 --
much closer than `normative_union`'s 1/14, confirming most of that earlier gap was the K=1-union
min-p artifact, not absent signal. `deseq2_ruvg_k1` (power-preserving bias correction) stays at
8/13 and `deseq2_covariate` drops to 3/13 (power loss) at every q tested -- reconfirms the earlier
"DESeq2's edge is uncorrected bias" hypothesis is not supported.

**This is explicitly NOT evidence of normative superiority, and should never be cited as such.**
Per statistician subagent review (2026-08-13): Stouffer's Z combination assumes a shared,
co-directional group effect exists -- averaging cancels exactly the patient-specific heterogeneous
signal that is the project's own central claim (see `overlap_enrichment` result below), so this
comparison structurally disadvantages normative modeling on a question orthogonal to its actual
value proposition. Use only as an internal validity/sanity check ("the Z-scores are not pure
noise, they recover ~most of a purpose-built group test's signal even after being forced through
an averaging operation that throws away the individual-level information they were designed to
carry") and as a second, independent confirmation of the RUVg/covariate finding above. The
individual-level results below are the actual differentiation argument.

## Individual-level deviation overlap (`3_disease_scoring.ipynb` sec. 7) -- DB-independent

Two DB-independent checks (no Open Targets reference used):

1. **Gene- vs pathway-level heterogeneity** (`core/pathway_convergence.py:overlap_enrichment`,
   size-matched permutation null, 200 reps): raw pairwise Jaccard of BH-significant calls across
   patients is NOT comparable across item universes of different size (19858 genes vs 2063
   pathway terms) -- confirmed empirically: gene-level observed Jaccard equals its own null in
   every phenotype tested (median 0 vs null 0, p=1.0) -- **not evidence of heterogeneity, just
   unpowered at current per-patient significant-gene counts (median 1-12/patient)**. Pathway-level
   shows genuine excess over its OWN size-matched null in 7/10 phenotypes (3.3x-28.8x, permutation
   p=0.005, the floor at 200 reps); 3/10 (Pancreatic Cancer, Pancreatitis, Pre-eclampsia) show no
   signal at either level.
   **Known limitation (deferred, not yet fixed):** the pathway-level null draws uniform-random
   pathway terms per patient, which ignores KEGG/Reactome's redundant/overlapping gene-set
   structure -- independent patients would already tend to co-hit correlated pathway clusters from
   library structure alone, so the reported enrichment ratios and p-values likely overstate true
   convergence (direction of the finding is probably still correct; magnitude is not trustworthy
   yet). Proper fix identified but deferred: null should permute gene labels and re-derive pathway
   hits through the same scoring pipeline used for real data (mirroring `deseq2_pathway_sig`'s own
   gene-permutation null), not draw random term-indices directly. Also unaddressed: permutation
   p-values are not multiple-testing corrected across the phenotype x level table, and sit at the
   n_perm=200 floor (0.005) -- both should be resolved before this ratio is used as a headline
   number in any writeup.
2. **Cross-study replication** (`db_hit_compare.py:cross_study_replication`): for Liver Cancer
   (the only phenotype split across 2 independent source studies -- Chen et al. n=10 vs
   Roskams-Hieter B et al. n=28), each study's significant gene/pathway set was hypergeometric-
   tested against the OTHER study's set directly (no DB reference at all). Gene-level p=1.55e-3,
   pathway-level p=4.33e-71 -- independent replication, real signal, and (unlike #1) not exposed
   to the redundant-library-null concern since this test doesn't use a random-term null at all.
