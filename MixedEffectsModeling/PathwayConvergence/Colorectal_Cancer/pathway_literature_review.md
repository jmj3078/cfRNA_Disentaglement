# Colorectal Cancer — Pathway Literature Review

Candidate list: `/tmp/claude-1000/-project-cfRNA-NormativeModeling/6379a855-c803-40c7-bbcb-ea44cd335e6a/scratchpad/cand_Colorectal_Cancer.txt` (n=34 patients). [GENERIC]-flagged rows (Metabolism Of RNA, Cellular Responses To Stress/Stimuli, Metabolism Of Proteins, Gene Expression (Transcription), RNA Polymerase II Transcription, mRNA Splicing / mRNA Splicing - Major Pathway, Infectious Disease, Spliceosome, Generic Transcription Pathway, Cell Cycle, Cell Cycle Mitotic, Immune System, Disease) and hist_frac>15% rows (DNA Replication, hist_frac=18.2%) were excluded by default per the standing project rule. No exceptions were made — no GENERIC or high-histone row had disease-specific enough evidence to override the exclusion. Pathway 7 (Stabilization Of P53) was added manually afterward — it fell just outside the automated review's top-40 candidate window.

## Selected pathways

### 1. Degradation Of Beta-Catenin By Destruction Complex
n_sig=22/34, size=84, eff=4.41, hist_frac=0%

The APC/AXIN/GSK3B destruction complex that degrades CTNNB1 (beta-catenin) is the canonical WNT-pathway lesion in colorectal cancer — APC mutation and consequent destruction-complex failure is the initiating hallmark event in the classical CRC adenoma-carcinoma sequence. This is the single most CRC-specific, mechanistically direct pathway in the candidate list.
- Wu H et al., Autophagy, 2019 (PMID: 30806153) — TRAF6 regulates selective autophagic CTNNB1/beta-catenin degradation and is itself targeted for GSK3B-mediated phosphorylation/degradation in colorectal cancer metastasis.
- Malki A et al., Int J Mol Sci, 2020 (PMID: 33374459) — Review of CRC progression/metastasis mechanisms, confirming APC-beta-catenin destruction-complex dysregulation as a core driver.

### 2. Nuclear Events Mediated By NFE2L2
n_sig=23/34, size=78, eff=4.57, hist_frac=0%

Represents KEAP1-NRF2 (NFE2L2) pathway activity, which is repeatedly implicated in CRC progression and chemoresistance via redox and detoxification gene programs. Two near-duplicate Reactome entries for the same KEAP1-NFE2L2 axis (KEAP1-NFE2L2 Pathway; GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2) were folded into this single, better-scoring representative to avoid triple-counting one mechanism.
- Liu C et al., Cell Death Differ, 2023 (PMID: 37210578) — Curcumin activates a ROS/KEAP1/NRF2/miR-34a-c cascade to suppress colorectal cancer metastasis.
- Sadeghi MR et al., Tumour Biol, 2017 (PMID: 28621229) — Review: role of the Nrf2-Keap1 axis in colorectal cancer progression and chemoresistance.

### 3. SCF(Skp2)-mediated Degradation Of P27/P21
n_sig=20/34, size=59, eff=4.60, hist_frac=0%

The SKP2-mediated ubiquitination of the CDK inhibitor p27(CDKN1B) is a well-documented CRC oncogenic mechanism — SKP2 overexpression with concomitant p27 loss correlates with tumor progression and poor prognosis, and is itself regulated by the Cdh1/APC-C axis, tying this pathway mechanistically to the APC/C entries below rather than being purely redundant with them.
- Bochis OV et al., J Gastrointestin Liver Dis, 2015 (PMID: 26114183) — Review: role of Skp2 and its substrate p27(CDKN1B) in colorectal cancer.
- Fujita T et al., Am J Pathol, 2008 (PMID: 18535175) — Regulation of the Skp2-p27 axis by the Cdh1/anaphase-promoting complex pathway in colorectal tumorigenesis.

### 4. APC/C:Cdc20 Mediated Degradation Of Securin
n_sig=22/34, size=67, eff=4.83, hist_frac=0%

Chosen as the single representative of the CDC20/APC-C mitotic-checkpoint machinery, which the candidate list contains many near-duplicate entries for (see rejected list). CDC20 overexpression is repeatedly reported as a poor-prognosis biomarker in colorectal cancer, consistent with chromosomal instability driven by premature securin/cyclin degradation and checkpoint override.
- Wu WJ et al., J Transl Med, 2013 (PMID: 23758705) — CDC20 overexpression predicts poor prognosis in colorectal cancer patients.
- Li J et al., World J Surg Oncol, 2020 (PMID: 32127012) — CDK1 and CDC20 overexpression in colorectal cancer associated with poor prognosis (integrated bioinformatics analysis).

### 5. Complex I Biogenesis
n_sig=26/34, size=51, eff=4.71, hist_frac=0%

Mitochondrial complex I activity is directly linked to colorectal cancer chemosensitivity and metastatic potential via oxidative-stress/ROS regulation, giving this narrow mitochondrial-biogenesis pathway a specific mechanistic tie to CRC beyond generic metabolic housekeeping. "Mitochondrial Protein Import" (n_sig=22, eff=4.70) covers largely the same mitochondrial-biogenesis theme and was rejected as redundant in favor of this higher-scoring, more directly-evidenced entry.
- Tang Y et al., Hum Cell, 2022 (PMID: 36059022) — SRGAP2 controls colorectal cancer chemosensitivity via regulation of mitochondrial complex I activity.
- Rai NK et al., Oncol Lett, 2020 (PMID: 33093922) — Differential regulation of mitochondrial complex I and oxidative stress based on metastatic potential of colorectal cancer cells.

### 6. Negative Regulation Of NOTCH4 Signaling
n_sig=20/34, size=53, eff=4.78, hist_frac=0%

NOTCH4 specifically (not just generic NOTCH family signaling) has direct, CRC-specific mechanistic literature: it regulates proliferation and invasiveness and is linked to clinical outcome, and a NOTCH4-GATA4-IRG1 axis has been proposed as a target in early-onset colorectal cancer. This narrow paralog-specific finding justifies keeping the pathway despite the generic "Notch signaling" umbrella normally being excluded.
- Scheurlen KM et al., Cytokine Growth Factor Rev, 2022 (PMID: 35941043) — The NOTCH4-GATA4-IRG1 axis as a novel target in early-onset colorectal cancer.
- Zhang Z et al., J Cell Physiol, 2018 (PMID: 29693251) — NOTCH4 regulates colorectal cancer proliferation, invasiveness, and determines clinical outcome of patients.

### 7. Stabilization Of P53
n_sig=20/34, size=56, eff=4.57, hist_frac=0%

Manually added after initial review: this pathway scored just outside the top-40 candidate window used for the automated literature pass (rank 63/2063), and its statistics are essentially unchanged from the pre-bugfix run (n_sig=20/34, eff=4.61 then vs 20/34, eff=4.57 now) -- omission was a review-window artifact, not a literature or signal-strength rejection. TP53 alterations mark the defined, late transition event from adenoma to carcinoma in the canonical multistep model of colorectal tumorigenesis, and CLK2-mediated regulation of the p53/KEAP1-NRF2 axis specifically suppresses ferroptosis in CRC -- a disease-specific staging role, not a generic "p53 matters in cancer" signal.
- Fearon EA, Vogelstein B, Cell, 1990 (PMID: 2188735) -- p53 alterations placed as the late, transition-defining event from adenoma to carcinoma in the colorectal multistep genetic model.
- Song et al., Cancer Research, 2025 (PMID: 40882016) -- CLK2 regulates KEAP1/NRF2 and p53 pathways to suppress ferroptosis specifically in colorectal cancer.

## Considered and rejected

- **KEAP1-NFE2L2 Pathway** (n_sig=21, size=100, eff=4.49, hist_frac=0%). Rejected as redundant with "Nuclear Events Mediated By NFE2L2" — same KEAP1-NRF2 mechanism, lower score.
- **GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2** (n_sig=21, size=51, eff=4.86, hist_frac=0%). Rejected as redundant with "Nuclear Events Mediated By NFE2L2" — same NRF2-degradation mechanism, narrower slice of the same pathway.
- **Degradation Of AXIN** (n_sig=19, size=54, eff=4.85, hist_frac=0%). Rejected as redundant with "Degradation Of Beta-Catenin By Destruction Complex" — AXIN is a core component of the same WNT destruction complex; the beta-catenin entry has a higher score and is the more direct/well-cited CRC mechanism.
- **Autodegradation Of Cdh1 By Cdh1:APC/C, Activation Of APC/C And APC/C:Cdc20 Mediated Degradation Of Mitotic Proteins, APC/C:Cdc20 Mediated Degradation Of Mitotic Proteins, Cdc20:Phospho-APC/C Mediated Degradation Of Cyclin A, APC/C-mediated Degradation Of Cell Cycle Proteins, APC:Cdc20 Mediated Degradation Of Cell Cycle Proteins Before Cycle Checkpoint Satisfied, Mitotic Anaphase, Mitotic Metaphase And Anaphase, Separation Of Sister Chromatids, M Phase, S Phase, FBXL7 Down-Regulates AURKA During Mitotic Entry And In Early Mitosis, Switching Of Origins To A Post-Replicative State, CDK-mediated Phosphorylation And Removal Of Cdc6** — all rejected as redundant with "APC/C:Cdc20 Mediated Degradation Of Securin", the single kept representative of CDC20/APC-C mitotic-checkpoint dysregulation in CRC.
- **Mitochondrial Protein Import** (n_sig=22, size=65, eff=4.70, hist_frac=0%). Rejected as redundant with "Complex I Biogenesis" — same mitochondrial-biogenesis theme, lower score.
- **Synthesis Of DNA** (n_sig=23, size=119, eff=4.78, hist_frac=0%). Generic S-phase machinery, no CRC-specific mechanistic paper found beyond general proliferation; folded conceptually into the cell-cycle/CDC20 cluster above.
- **Thermogenesis** (n_sig=24, size=219, eff=4.50, hist_frac=0%). No CRC-specific mechanistic literature found; likely reflects UCP/metabolic batch or technical variance rather than tumor biology.
- **Cellular Response To Chemical Stress** (n_sig=22, size=182, eff=4.87, hist_frac=0%). Too broad a stress-response umbrella; not searched down to a specific CRC mechanism worth citing.
- **Pathways of neurodegeneration** (n_sig=23, size=462, eff=4.23, hist_frac=0%). Off-topic — neurodegeneration-associated gene programs, not a colorectal-cancer-specific mechanism.
- **Downstream TCR Signaling, Activation Of NF-kappaB In B Cells, NIK To Noncanonical NF-kB Signaling** — lymphocyte/B-cell-intrinsic signaling; in cfRNA these more plausibly reflect immune-cell infiltrate/turnover confounds than tumor-intrinsic CRC biology, and no CRC-epithelial-specific mechanistic paper was sought for these narrow immune sub-pathways.
- **Vif-mediated Degradation Of APOBEC3G, SARS-CoV Infections, HIV Infection, Host Interactions Of HIV Factors, Olfactory Signaling Pathway, Expression And Translocation Of Olfactory Receptors, Olfactory transduction, Sensory Perception** — off-topic viral/sensory pathways with no plausible CRC mechanism; almost certainly composition artifacts of large, gene-dense Reactome branches picking up incidental patient-level signal.
- **tRNA Processing, AUF1 (hnRNP D0) Binds And Destabilizes mRNA** — generic RNA-processing/stability machinery; no CRC-specific mechanistic paper identified in this review to distinguish them from housekeeping variance.
- All [GENERIC]-flagged rows and the hist_frac>15% row (DNA Replication) were excluded per the standing project rule; no exceptions were justified for this cohort.

## Raw search log

PubMed PMIDs retrieved during this review (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- 30806153 — TRAF6/CTNNB1 degradation, CRC metastasis (cited, pathway 1)
- 33374459 — CRC progression/metastasis review (cited, pathway 1)
- 32920015, 31094179, 34352208 — beta-catenin destruction complex + CRC esearch hits (checked, not cited — redundant with above)
- 37210578 — curcumin/ROS/KEAP1/NRF2 CRC metastasis (cited, pathway 2)
- 28621229 — Nrf2-Keap1 axis CRC review (cited, pathway 2)
- 40882016, 41385828, 42173678 — KEAP1/NRF2 + CRC esearch hits (checked, not cited)
- 26114183 — Skp2/p27 CRC review (cited, pathway 3)
- 18535175 — Skp2-p27/Cdh1-APC pathway CRC (cited, pathway 3)
- 29343851, 31894983, 29135092 — SKP2/p27 + CRC esearch hits (checked, not cited)
- 36059022 — SRGAP2/mitochondrial complex I CRC chemosensitivity (cited, pathway 5)
- 33093922 — mitochondrial complex I/oxidative stress, CRC metastatic potential (cited, pathway 5)
- 37533102, 41221702, 39404412 — mitochondrial complex I + CRC esearch hits (checked, not cited)
- 23758705 — CDC20 overexpression, poor CRC prognosis (cited, pathway 4)
- 32127012 — CDK1/CDC20 overexpression, poor CRC prognosis (cited, pathway 4)
- 17679094 — APC/C(Cdc20) controls p21 degradation, general mechanism not CRC-specific (checked, not cited — non-CRC mechanistic)
- 25789873 — MEF2C degradation/G2M, not CRC-specific (checked, not cited)
- 36900162, 32453965, 41614789 — CDC20 + CRC prognosis esearch hits (checked, not cited)
- 35941043 — NOTCH4-GATA4-IRG1 axis, early-onset CRC (cited, pathway 6)
- 29693251 — NOTCH4 regulates CRC proliferation/invasiveness/outcome (cited, pathway 6)
- 30026833, 40103807, 35587061 — NOTCH4 + CRC esearch hits (checked, not cited)
- 39022865, 41461634, 38419282, 42204589, 29786110 — AXIN1 degradation + CRC esearch hits (checked, not cited — used only to confirm AXIN-degradation redundancy with beta-catenin destruction complex)
- 2188735 — Fearon & Vogelstein, colorectal multistep genetic model (cited, pathway 7, manually added)
- 40882016 — CLK2/KEAP1/NRF2/p53 ferroptosis suppression in CRC (cited, pathway 7, manually added)
