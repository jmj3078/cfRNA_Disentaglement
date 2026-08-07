# cfRNA Normative Modeling — 통합 참조 문서 (Manuscript Dossier)

작성 2026-08-07. 프로젝트 전체에 흩어져 있는 설계 근거·검증 결과·문헌 근거·폐기 결정을
논문 집필 시 한 번에 참조할 수 있도록 모은 단일 문서.

**원칙**: 이 문서는 *포인터 + 확정 사실*만 담는다. 수치의 원본은 항상 링크된 산출물이며,
"확인 필요"로 표시된 항목은 논문에 인용하기 전 원문/재계산 검증이 필요하다.

---

## 0. 원본 문서 지도

| 문서 | 내용 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | 프로젝트 공통 요구사항, 코딩/시각화 규약 |
| [MixedEffectsModeling/CLAUDE.md](MixedEffectsModeling/CLAUDE.md) | **엔진 v3 방법론 전문** (cascade, EB shrinkage, dispersion trend, PCIS, pooled GLMM, SHASH) |
| [EDA/normative_modeling_literature.md](EDA/normative_modeling_literature.md) | NM 이론배경 문헌 8개 섹션 표 |
| [MixedEffectsModeling/PCIS_Calibration/README.md](MixedEffectsModeling/PCIS_Calibration/README.md) | PCIS 임계값 경험적 null 캘리브레이션 |
| `MixedEffectsModeling/PathwayConvergence/*/pathway_literature_review.md` | 질병 10종 pathway 문헌 큐레이션 (PMID 단위) |
| `docs/superpowers/specs/2026-07-27-eb-dispersion-cook-outlier-design.md` | EB/PCIS 설계 기록 |
| `docs/superpowers/specs/2026-07-28-v3-mathematical-reference.md` | 수식 유도 전문 |
| `_legacy/Modeling_v1/Benchmark/DESeq2_vs_Normative_Report.md` | DESeq2 vs Normative 3자 비교 리포트 (v1 엔진 기준) |
| `_legacy/.../GSEA/Analysis_Provenance.md` | Open Targets/PubMed 질의 provenance (2026-07-01) |
| 세션 메모리 `~/.claude/projects/-project-cfRNA-NormativeModeling/memory/` | 아래 §6 결정 로그의 출처 |

노트북 (thin runner): `EDA/1~9_*.ipynb` (분산 분해·bias), `MixedEffectsModeling/1~5_*.ipynb`
(CV / LOBO / disease scoring / gene enrichment / DESeq2 비교).

---

## 1. 논지 1 — cfRNA의 지배 분산은 기술적이다

**산출물**: `EDA/Analysis_Results/rda_normalization_layer_summary.tsv`,
`rda_normalization_r2_results.tsv`, `rda_cascade_results.png`, `rda_results_hvg/`,
`vif_covariate_candidates.csv`, `bias_power_auc_grid.png`.
**노트북**: `EDA/3_pca_analysis.ipynb`, `4_rda_variance_decomposition.ipynb`,
`5_bias_batch_relationship.ipynb`, `6_bias_phenotype_relationship.ipynb`, `8_covariate_vif.ipynb`.

확정 수치 (layer summary, partial RDA):

| Layer | phenotype R² | bias R² 합 | bias/phenotype |
|---|---|---|---|
| EDAseq (GC+Length) | 0.0037 | 0.0610 | **16.6×** |
| + RUVg k=1 | 0.0039 | 0.0588 | 15.0× |
| + RUVg k=2 | 0.0041 | 0.0518 | 12.8× |
| + RUVg k=3 | 0.0040 | 0.0454 | 11.4× |

읽는 법: **RUVg는 bias 설명분산을 줄이지만(0.061→0.045) phenotype R²를 훼손하지도 않는다.**
즉 "기존 방법이 bias를 못 지운다"는 서사는 성립하지 않는다 — RUVg는 자기 일을 한다.
그럼에도 보정 후에도 bias가 phenotype의 11배 이상이다. 이것이 논문의 실제 주장이어야 한다.

> **서사 주의**: 이 결과 때문에 프로젝트 서사는 2026-08-04에 전환되었다.
> "기존 방법이 배치를 못 지운다" → **"기존 방법의 estimand(그룹 평균 차이)가 애초에
> 우리가 원하는 것이 아니다"**. RUVg 비교는 supplementary에서 정직하게 제시하고, 본문은
> estimand 논증으로 간다.

기술 공변량 10종은 VIF로 선별 (`vif_current_10.png`). dispersion 관점 유의성은 매우 불균등:
Gene Length Bias ~87%, GC Bias ~63%, log(Total Reads) ~62% 유전자에서 유의한 반면
NP80/NG80은 ~5%로 null rate와 동일 = dispersion 축에서는 순수 noise (§6-D).

---

## 2. 논지 2 — 그룹 비교는 대조군 구성만 바꿔도 재현되지 않는다

**산출물**: `EDA/Analysis_Results/Control_Composition/`
(`jaccard_results.csv` n=198, `deseq2_jaccard_results.csv` n=44, `null_distribution.csv` n=3600,
`gsea_split_jaccard/`, `subsets.csv`, `ruvg_W/`).
**코드**: `EDA/control_composition/run_control_composition.py`, `..._deseq2.py`, `ruvg_batch.R`.
**노트북**: `EDA/9_control_split_variation.ipynb`.

설계: case를 고정하고, **같은 study·같은 batch의 control**만 기술 공변량 축(log Total Reads,
Spliced Reads %, rRNA Fraction 등)의 tertile로 재분할 → DE를 다시 돌린다. 비교 대상은
control을 무작위로 재분할한 null 분포(3,600 draw).

확정 수치 (top-k DE 유전자 Jaccard 평균):

| | k=25 | k=50 | k=100 | k=200 | k=500 | Spearman |
|---|---|---|---|---|---|---|
| tertile split (t-stat 계열, 198) | 0.004 | 0.007 | 0.011 | 0.018 | 0.033 | 0.242 |
| tertile split (DESeq2, 44) | 0.030 | 0.050 | 0.072 | 0.071 | 0.076 | 0.255 |
| random null (3,600) | 0.007 | 0.011 | 0.018 | 0.027 | 0.048 | 0.325 |

읽는 법 — 이것이 이 논문에서 가장 강한 단일 관측이다:

1. **DE 유전자 집합의 재현성이 사실상 0이다.** top-100에서 Jaccard 0.01~0.07.
2. **구조화된 split이 무작위 split보다 나을 것이 없다.** t-stat 계열은 오히려 null보다
   *낮다*(0.011 vs 0.018) — 즉 불안정성은 "잘못된 대조군을 골라서"가 아니라 **대조군을
   어떻게 고르든** 발생한다. 이 대칭성이 "control 선택을 잘 하면 된다"는 반론을 봉쇄한다.
3. **RUVg 보정도 구제하지 못한다.** GSEA 로그 예: Pancreatic Cancer / log(Total Reads),
   `deseq2__no_covariate` n_sig=71,115,137 jacc=0.200 vs `deseq2__ruvg_k2` n_sig=31,5,19
   jacc=0.048 — 보정 후 유의 term 수 자체가 붕괴하고 재현성은 더 나빠진다.
4. DESeq2가 t-stat보다 Jaccard가 다소 높은 것은 shrinkage로 랭킹이 발현량 쪽으로
   안정화된 결과로 보이며, 절대 수준(0.03~0.08)은 여전히 재현이라 부를 수 없다.

문헌 선례: Nygaard 2016 Biostatistics (10.1093/biostatistics/kxv027) — 그룹이 배치에
불균등 분포하면 batch 제거가 오히려 downstream 과신/위양성을 만든다. Zindler 2020
BMC Bioinformatics (10.1186/s12859-020-03559-6) — ComBat 시뮬레이션 위양성.

---

## 3. 논지 3 — Normative Model: 무엇을 어떻게 만들었나

전문은 [MixedEffectsModeling/CLAUDE.md](MixedEffectsModeling/CLAUDE.md). 논문 Methods에
들어갈 핵심만:

### 3.1 모델
유전자별 NB2 GLMM, 평균·분산 모두 공변량 함수 (GAMLSS NBI 계열, Bethlehem 2022 brain
charts와 동일 기계):

- `nbi_full_eb`: log μ_i = β0 + Σβ_k X_ik + b_j , log θ_i = γ0 + Σγ_k X_ik
- 실패 시 `nbi_intercept_eb` (log θ = γ0), 둘 다 실패 시 `route="excluded"`
- batch는 **mean submodel의 random intercept `(1|batch)`에만** 진입. dispersion에는 넣지
  않는다 — HC의 singleton batch 5개가 각각 자유 파라미터를 얻어 trend가 측정하려는 바로 그
  dispersion을 깎아내리기 때문.
- scoring 시 batch BLUP 불필요: u~N(0,τ²)를 Gauss-Hermite로 주변화하므로 **훈련에서 본 적
  없는 새 batch도 채점 가능** — 이것이 Batch_ID를 고정효과로 넣지 않은 이유이자 §5의 한계의 원천.

### 3.2 정규화 장치 4종 (각각 왜 필요했는지)

| 장치 | 문제 | 해법 | 근거 |
|---|---|---|---|
| EB dispersion shrinkage (`core/eb_shrinkage.py`) | 저발현 유전자의 dispersion slope가 폭주 | limma/edgeR 모멘트 분해, `tau²=max(0,(1.4826·MAD(φ̂))²−median(SE²))`. slope τ_k 측정값 0.10–0.36 (v2의 일괄 0.05는 과도) | limma/edgeR |
| Dispersion trend (`core/dispersion_trend.py`) | intercept squeeze와 PCIS가 모두 이것에 의존 | **공변량 보정된** dispersion에 lowess(frac=0.3, bisquare it=3). raw-count MoM 방식은 공변량 유래 평균분산과 진짜 dispersion을 혼동해 고발현에서 과대추정 | 내부 유도 (math ref) |
| PCIS outlier removal (`glmm_helpers.R`) | 오염된 관측이 자기 유전자의 α를 부풀려 스스로를 은폐 | Cook형 통계량이되 분산을 **trend α**(다른 ~19,000 유전자에서 유래)로 고정. leverage는 prior-penalized mixed design에서 | Belsley/Kuh/Welsch 1980; Cook&Weisberg 1982; **Hadi&Simonoff 1993** (다중 masking → 단일삭제 진단은 원리적으로 실패); Rousseeuw&Leroy 1987 (high-breakdown) |
| Per-gene SHASH calibration (`core/calibration.py`) | held-out HC Z가 N(0,1)이 아니면 downstream FDR이 깨짐 | 유전자별 sinh-arcsinh 적합 후 보정 | Jones&Pewsey 2009 |

측정 근거 (PCIS): 20배 outlier 3개 주입 시 α가 36배 부풀며 어떤 크기에서도 탐지 0건.
Pregibon 1981(1-step deletion)은 *다른* 문제(재적합 비용)의 해법이므로 혼동 금지.

### 3.3 PCIS 임계값 = 2.28 (경험적 null)
PCIS에는 F 참조분포가 없으므로 관행적 `qf(0.99, p_eff, n−p_eff)`에 근거가 없다.
각 유전자의 자체 적합 파라미터로 깨끗한 count를 재생성 → 동일 prior로 재적합 → PCIS 재계산
(19,158 유전자 × 693 관측 = **13,276,494 null draw**). null 제거율이 실제 제거율 아래로
떨어지는 지점이 rate 1.5e-4~1e-4 사이. 배포값 **2.28** (rate 1e-4, null share 0.72).
관행값(~1.98)은 우연히 근접했을 뿐 이론적 근거는 없었음.
미해결: 실제 per-observation PCIS를 cascade 실행 시 저장하지 않아 real-vs-null 교차점은
단일 기준점에서 추정된 것 (README "Still open").

### 3.4 pooled GLMM 경계 `nz_a_max = 25`
개별 적합 수렴 실패율의 knee가 nz~20–25 (nz 1–5에서 92% → 21–25에서 10% → 31–40에서 4.5%).
cutoff를 올리면 **개별 적합이 잘 되는 유전자를 pooling으로 강등**시켜 유전자별 β를 잃는다
(rescue/downgrade 비: nz 10에서 4.70 → 25에서 1.21 → 40에서 0.74).
**미검증 가정**: pool route는 실제 HC에 한 번도 실행된 적 없다. CV로 pool route의 held-out Z
calibration이 확인되기 전까지 이 임계값 논증은 조건부다. 논문에 반드시 명시.

### 3.5 `tau2_max = 3.0`
큰 τ²는 calibration을 깨지 않는다 (τ² 구간별 rqr_msq 0.98–1.03, cov_95 0.957–0.963,
τ²>5 구간에서도 동일). 깨는 것은 **검출력**: 예측분포가 넓어져 |z|=3에 필요한 count가
관측 가능 범위를 벗어난다 (undetectable rate: τ²<0.5에서 0% → τ²>3에서 15–34%).
컷은 305 유전자(1.6%) 제거, 그중 ~21%만 입증 가능하게 undetectable — **임계값이 날카롭지
않다는 점을 인정하고**, 부수 피해가 저발현(median μ~0.05)이라는 이유로 수용.
직접적 `y_crit` 필터는 코호트 종속(모델이 넓은 것과 그런 샘플이 이 코호트에 없는 것을
구분 못함)이라 기각.

### 3.6 EXCLUDED_GENES
배제 대상은 "공변량과 무관한 유전자"가 아니라 **HC count 분포가 이중봉이라 단봉 NB 적합이
원천 불가능한 유전자**. 현재 수동 큐레이션 7개. 자동 detector 요건 = (intercept 강등)
∧ (dip test/2-comp GMM 유의) ∧ (분산분석상 공변량 관계 존재). w1(calibration)은 detector로
부적합 — 두 봉이 HC 안에서 일관되면 in-sample RQR이 멀쩡해 보인다. 자동화 deferred.

---

## 4. 검증 결과 — 무엇이 실제로 입증되었는가

### 4.1 CV / in-sample calibration
`MixedEffectsModeling/1_cv_analysis.ipynb`, `CV_Results_mixed/`.
판단 기준은 (a) per-gene RQR marginal (var-z≈1, |z|>3 비율) + (b) pseudo-disease random split
AUC≈0.5. **classifier AUC 기반 `calibration_control_hc`(l1)은 폐기된 지표** (§6-A).

### 4.2 LOBO (leave-one-batch-out) — 채택 지표는 MMD 단독
`2_lobo_validation.ipynb`, `LOBO_Results_mixed/`, `pipeline/lobo_validation.py:mmd_summary()`.

HC가 존재하는 31개 batch 전부에 대해 개별 재학습(batch당 ~15.5분, 총 ~8시간).
`MIN_N_HC=25` 기준 6개 batch 보고:

| batch | n_hc/n_dis | MMD² | p | 방향 |
|---|---|---|---|---|
| Ward Z et al._Batch_1 | 107/201 | 0.058 | 0.001 | disease 더 멂 |
| Roskams-Hieter B_Batch_2 | 26/56 | 0.014 | 0.005 | 더 멂 |
| Chen et al._Batch_2 | 28/30 | 0.030 | 0.003 | 더 멂 |
| Moufarrej_Batch_4 | 42/30 | 0.010 | 0.039 | 더 멂 |
| Moufarrej_Batch_1 | 47/19 | 0.039 | 0.004 | 더 멂 |
| Moore et al._Batch_1 | 67/161 | −0.0006 | 0.576 | **미유의** |

**6개 중 5개(83%)에서 유의+방향 일치.** Moore 예외는 노이즈바닥 불안정이 아니라 실제 신호가
약한 것 (Pancreatitis/Pancreatic Cancer/Other Cancer로 쪼개도 셋 다 구분 안 됨).

**폐기된 편차 지표** (재도입 시도 시 이 사유 먼저 확인): mean|Z|, Σz²(χ²), BH-FDR 극단비율은
Tier A 22개 중 2~4개만 유의 — 신호가 소수 유전자에 집중되고 다수 정상 유전자가 평균을 희석.
PCA(30성분)+Mahalanobis는 n=693/p=20,097에서 무작위 HC half-split 배경거리(~3.5)가 이미
disease/HC 노이즈바닥과 같은 스케일이라 폐기. MMD(RBF, 전유전자, permutation)만 n≪p를 회피.

### 4.3 개별 샘플 편차 / pathway 수렴
`3_disease_scoring.ipynb`, `4_gene_enrichment.ipynb`, `Z_scores_mixed/`,
`PathwayConvergence/*/` (질병 10종: Colorectal, Esophagus, Liver×2, Lung, Pancreatic,
Pancreatitis, Pre-eclampsia, Stomach, Tuberculosis) + `run_pathway_convergence_batch.py`.

각 질병 디렉토리의 `pathway_literature_review.md`는 선택된 pathway마다 n_sig/size/eff/hist_frac
및 **PMID 단위 기계론적 근거**, 그리고 **기각된 후보와 기각 사유**까지 기록되어 있다
(예: Lung Cancer — NFE2L2 nuclear events 19/26 채택, Olfactory 계열은 폐암 특이 논문 부재로
기각, APC/C 클러스터는 generic proliferation으로 기각). 논문 Supplementary Table로 거의
그대로 전용 가능.

큐레이션 규칙 (반드시 유지):
- 리보솜/번역, OXPHOS는 전사체 전역 조성 변화를 반영하는 비특이 신호 → discount.
- 혈액 GSEA의 신경퇴행 KEGG(Alzheimer/Parkinson/Prion/Huntington/ALS)는 사실상 OXPHOS
  proxy → 함께 discount.
- 무관한 여러 암종 KEGG term이 한 코호트에 동시 출현하면 literal 질병이 아니라 **공유
  oncogene 모듈(TP53/RAS/RB)** 하나로 묶는다.
- `[GENERIC]` 플래그 및 hist_frac>15% 행은 기본 제외, 예외는 명시적으로 정당화.

### 4.4 Normative vs DESeq2 (v1 엔진 기준, 재실행 필요)
`5_deseq2_group_comparison.ipynb`, `_legacy/Modeling_v1/Benchmark/DESeq2_vs_Normative_Report.md`,
`Benchmark/gsea_compare/`. 참조 DB = Open Targets 질병별 상위 300 유전자
(`Benchmark/disease_reference/{pheno}.json`, MONDO/EFO resolve, provenance는
`GSEA/Analysis_Provenance.md`, 수집 2026-07-01).

- **대칭 채점(동일 규칙: 자기 유의 term의 lead ∩ OT 참조)**: 정밀도는 세 방법 동등
  (pooled DESeq2 0.486 / no_filter 0.438 / with_rare 0.445). DB-지지 term **개수**는
  Normative가 ~1.7–2배 (2,236 vs 3,843 / 4,375). → **"정밀도 손실 없이 커버리지 확대"**가
  정확한 주장이다. "더 많이 찾았다"만으로는 FDR 완화와 구분되지 않는다.
- **비대칭 포착**: DESeq2는 normative DB-지지 경로의 중앙값 ~16%만 포착, 검출 term 절반
  (256 vs 518). 일치도는 표본크기와 연관(TB/HIV+TB/위암/MM 높음, PE/ME-CFS/췌장계열 낮음).
- rare 분기 포함은 **방향 보존적·가산적**: 20질병 전부 공유 term NES 부호 일치율 1.0,
  신규 유의 term +32~+225. 신규 term의 30~72%가 OT 지지.
- 미보고 후보(rare가 처음 표면화): ME/CFS PLA2G10(문헌 0건), 자간전증 IFNL3/IFN-λ(1),
  ICI 심근염 MT1B/metallothionein(2), 간암(R-H) DEFB114/MAPK(3).
- **주의**: DB 지지는 대리지표이지 정답이 아니다. 두 GSEA는 랭킹/정규화/유전자우주가 달라
  concordance 측정이지 우열 판정이 아니다. Other Cancer/Pancreatitis/췌장암에서 공유 term
  부호가 반대인데 그 대상이 번역/리보솜/유비퀴틴 housekeeping 축(조성 정규화 민감)이므로
  방법론적 불일치로 보수 해석.
- 이 수치는 **v1(GAMLSS demotion chain) 엔진 산출물**이다. v3 mixed-effects 엔진으로
  재실행하지 않았다면 논문에는 재실행본을 써야 한다.

---

## 5. 한계 — 반드시 논문에 명시할 것

1. **Batch 고유 잔차는 제거되지 않았다 (가장 큰 한계).** 모델은 Batch_ID를 직접 쓰지 않고
   10개 연속 기술지표로만 조건화한다(새 batch 채점 가능성 유지가 목적). 이 지표로 환원되지
   않는 batch 고유 기술변이(시약 로트, protocol version 등)는 그대로 남는다.
   HC-only 검증(17 batch, 637명)에서: Z-score PCA PERMANOVA pseudo-F=11.25 p=0.001,
   PC2 batch η²=0.457, batch 분류 AUC 0.954–0.963 (보정 전 CPM_log1p 0.975–0.990 대비
   **소폭 감소에 그침**). gene-wise partial RDA에서 Batch_ID unique R² mean 0.039→0.053,
   R²>10% 유전자 1,361→3,487(2.6배)로 **오히려 증가** — 연속 공변량과 겹쳐 있던 batch 성분이
   제거되면서 batch 고유 잔차가 상대적으로 더 선명해진 것으로 해석.
2. **따라서 비교의 유효 범위가 제한된다.** within-batch 비교(같은 batch의 disease vs 그 batch
   HC, 특히 LOBO 방식)만 confirmatory. HC가 없는 disease-only batch(예: Chen)의 "vs 0" 주장은
   배치효과와 분리 불가 → **exploratory/hypothesis-generating으로만 사용.**
   실증: 간암(Roskams-Hieter) 학습 분류기를 Chen 배치에 적용하면 다른 암종(Lung/Stomach/
   Esophagus/Colorectal) AUC 0.845–0.896이 간암(0.765)보다 *높다* → 질병 특이가 아니라
   Chen-배치 신호. 대조군을 pooled-HC / 모델의 0점 / permuted synthetic HC 셋으로 바꿔도
   동일 패턴 = 대조군 선택의 문제가 아니라 Z-score 자체가 배치 수준으로 치우친 것.
3. **질병 ≈ 1 batch 완전교란.** 대부분 질병이 단일 batch에만 존재. 이건 DESeq2/group-wise도
   동일하게 겪는 코호트 설계의 한계이지 본 모델 고유의 결함이 아니다 — 그렇게 프레이밍할 것.
4. **참조 표본의 선택편향이 곧 "정상"의 정의**다 (검증 불가). N≈600–700, 메타데이터 부재로
   연령/성별이 아닌 기술 공변량 중심 모델 → 이 "규범"은 technical-conditional이다.
5. **꼬리 추정 취약**: 편차 판정이 표본이 희박한 1st/99th centile 영역에 의존.
6. **편차 ≠ 병리.**
7. 횡단 참조곡선은 종단 변화를 과소추정 (Di Biase 2023 PNAS 10.1073/pnas.2216798120).
8. NM 평가 표준 미성숙 (Dinga 2021) → calibration PIT/QQ + PPC 제시가 필수.
9. **pool route 미검증** (§3.4), **PCIS real-vs-null 교차점 단일 기준점 추정** (§3.3),
   **EXCLUDED_GENES 자동화 미구현** (§3.6).

---

## 6. 결정 로그 — 시도했고 폐기한 것들

논문 심사에서 "왜 X를 안 했나"가 나올 항목들. 전부 실제로 해봤고 근거를 남겨 폐기했다.

**A. `calibration_control_hc` (l1 분류기 지표) — 폐기 (2026-07-06)**
비교 대상이 iid N(0,1) synthetic null이라 지도학습 분류기가 생물학적으로 무의미한 미세
상관을 증폭해 miscalibration으로 오인. 개입 3종(std-fit, σ=f(X), per-gene offset centering)
모두 실패 — 고칠 결함이 없었기 때문. 엔진 HC calibration은 legacy 대비 원소상관 99.76%로
사실상 동일하며 결함 없음이 확인됨.

**B. discrimination_control (분류기 AUC로 batch-null vs disease 판별) — 전면 폐기 (2026-07-07)**
사유: (1) 서로 다른 분류과제의 AUC 크기 비교는 비가산·비교 불가, (2) disease≈batch aliasing
상태에서는 0.94–0.96도 여전히 미통제 배치효과일 수 있음, (3) group-wise 분류기 검증 자체가
개별 샘플 규범모델링의 취지와 상충. 삭제물: `pipeline/selection.py`, gene_selectors의
분류기 selector 전부, discrimination_*.csv, selector figure 다수. 대체 → partial RDA + LOBO/MMD.
(pseudo-disease 진단 결과는 기록으로 남김: RANDOM split AUC≈0.5 = 엔진이 위양성을 만들지
않음. BATCH-group split AUC 0.69–0.79 = batch 구조를 질병으로 오독. 실제 질병 median
0.94–0.96이 batch-null을 크게 상회.)

**C. HC-latent-PCA 공변량 — 실패·폐기 (2026-07-22)**
외부 QC 지표 10개 대신 HC 자체 PCA latent factor(k=10)를 공변량으로 쓰는 대안.
거의 모든 study에서 confounder-unique R²가 기존보다 *나빠짐*(일부는 raw CPM보다도).
근본 원인: HC-PCA loading은 raw expression에서 추출되므로 disease에 투사하면 기술 잡음과
진짜 질병 생물학이 섞인다(out-of-distribution extrapolation). 외부 QC 지표는 transcriptome
content와 독립 측정이라 disease에서도 순수 기술 신호를 유지. 부차: 10개 PC가 HC 분산의
~44%만 설명. 브랜치·폴더 전부 삭제.

**D. nbi_full_eb vs nbi_intercept_eb 명시적 이산 모델선택 — 불필요 결론 (2026-07-28)**
"수렴하면 무조건 full 채택"이 통계적 모델선택이 아니라는 지적을 검증. 저장된 disp_coef/disp_se
만으로 Wald z 계산(n=19,042): 0개 유의 slope는 7.1%뿐, 중앙값 4/10 유의.
**nz에 따른 "사실상 intercept-only화" 비율이 매끄럽게 단조감소**(nz 1–10 ~100% → 21–30 ~70%
→ 51–75 ~15% → nz>100 ~0%) = EB prior가 일괄 축소가 아니라 정확히 데이터량에 비례해
작동하는 전형적 empirical Bayes 거동. 명시적 AIC/LRT를 추가해도 aggregate 결과가 달라질
근거 없음 → cascade 현행 유지. (잔여 리스크: nz 20–75 전환구간이 유일하게 갈릴 수 있는 영역.)

**E. Batch_ID를 직접 공변량으로 추가 — 기각 (2026-07-22)**
(a) 목적 상충 — 새 미지 batch 채점이 NM의 핵심 목적인데 fixed effect로 넣으면 원천 불가;
(b) 질병≈batch 완전교란 → 계수가 비식별(DESeq2가 이미 겪는 문제를 그대로 재현);
(c) pool route 유전자는 표본 극소수라 batch 더미 추가 시 추정 불안정;
(d) 엔진 전면 재설계+재학습+재검증 비용 대비 편익 낮음.
→ **채택된 대안이 현재 v3의 random intercept `(1|batch)`다.** 이것이 (a)를 해결한다
(새 batch는 population mean으로 축소, Gauss-Hermite 주변화로 채점 가능).

**F. RDA로 "RUVg가 실패한다"를 보이려던 접근 — 실패 (2026-08-04)**
RUVg는 bias를 잘 지우고 phenotype R²도 훼손하지 않는다(§1). 서사를 estimand 논증으로 전환.

**G. mean|Z| / χ² / BH비율 / PCA-Mahalanobis 편차 지표 — 전부 폐기** (§4.2)

---

## 7. 문헌 앵커

전체 표는 [EDA/normative_modeling_literature.md](EDA/normative_modeling_literature.md)
(역할별 8개 섹션). 논증별 핵심만:

### 정당화 4축
1. **구조적**: 개체 수준 추론은 집단 수준 추론에서 유도되지 않는다 (생태학적 오류).
   반박 불가한 기둥.
2. **실증**: 환자 간 편차 중첩이 실제로 거의 없다 (Wolfers 2018 <2%, Segal 2023 <7%)
   **그러나** 상위 구조(회로/pathway)에서는 수렴(최대 56%) → 이질성은 "신호 없음"이 아니라
   **잘못된 해상도에서 본 신호**. 본 프로젝트의 gene-vs-pathway 실험이 이 논증의 cfRNA 판본.
3. **성능**: NM 파생 피처 > raw 피처 (Rutherford 2023 eLife 10.7554/elife.85082;
   Parkes 2021 Transl Psychiatry 10.1038/s41398-021-01342-6).
4. **인프라**: 참조 모델 transfer로 소규모 연구도 대규모 HC 참조 사용 가능.

### 계보
- **Marquand 2016** Biol Psychiatry 10.1016/j.biopsych.2015.12.023 (725cit) — NM 시조.
  case-control의 artificial symmetry, 진단 라벨 자체를 검증 불가, "average patient" 오류.
- **Marquand 2019** Mol Psychiatry 10.1038/s41380-019-0441-1 — 소아 성장곡선 유비.
- RDoC (Cuthbert&Insel 2013 10.1186/1741-7015-11-126, 3096cit) — 범주적 진단 비판의
  정책적 배경. NM은 RDoC가 답하지 못한 "차원의 눈금자"를 제공하는 장치.
- 통계 전통: Cole LMS → GAMLSS 분포회귀(μ/σ/ν/τ 전부 공변량 함수).
- **Bethlehem 2022** Nature 10.1038/s41586-022-04554-y (1830cit) — brain charts, GAMLSS.
  **본 엔진과 동일 기계**. Rutherford 2022 eLife 10.7554/elife.72904 (82 사이트 n=58,836),
  Rutherford 2022 Nat Protoc 10.1038/s41596-022-00696-5.

### 방법론 진화 = 반론 응답의 연쇄
GPR(Marquand 2016) → warped BLR(Fraza 2021 NeuroImage 10.1016/j.neuroimage.2021.118715;
O(n³)·비가우시안) → GAMLSS NM(Dinga 2021 10.1101/2021.06.14.448106; 이분산/왜도 + 평가표준
부재 지적) → federated HBR(Kia 2022 PLoS ONE 10.1371/journal.pone.0278776; **site를 random
effect로 모델 내부 흡수**) → non-Gaussian HBR/SHASH(de Boer 2024 Imaging Neurosci
10.1162/imag_a_00132). 방향 하나: **정상 분포의 형태 전체를 공변량+배치 구조 포함해 단일
모델에서 추정.** 본 프로젝트의 `(1|batch)` + per-gene SHASH가 정확히 이 계보 위에 있다.

### 2단계 보정(harmonize-then-analyze) 비판 — 논지 2의 직접 선례
- **Bayer 2022** NeuroImage 10.1016/j.neuroimage.2022.119699 — site effect가 관심변수와
  복잡히 교락. **모든 2단계 harmonization에서 원 분산 90% 이상 손실.** ComBat에서 age/sex
  미보존 시 최악. 단일단계 HBR random effect가 대안.
- **Gardner 2025** Hum Brain Mapp (ComBatLS) 10.1002/hbm.70197 — harmonization이 공변량의
  분산(scale) 효과를 보존하지 못해 normative score 오차 유발, 특히 공변량이 사이트 간
  불균등 분포 시.
- **Nygaard 2016** Biostatistics 10.1093/biostatistics/kxv027 (421cit) — 유전체판 대응.
- **Zindler 2020** BMC Bioinformatics 10.1186/s12859-020-03559-6 — ComBat 위양성 시뮬레이션.
- 논증 구조: 2단계로 나누면 1단계가 질병 신호를 흡수(과보정)하거나 배치를 잔존(과소보정)시키고,
  2단계는 1단계의 불확실성을 전파받지 못해 과신한다. 정합적 해법 = 공변량/배치/편차를 동시에
  다루는 단일 생성모델.

### 편차 중첩 실증 (논지 2·§4.3의 직접 선례)
- **Wolfers 2018** JAMA Psychiatry 10.1001/jamapsychiatry.2018.2467 (540cit) — 집단 수준
  유의한 회백질 감소에도 동일 loci 극단편차를 공유하는 환자 비율은 대부분 영역 <2%.
- **Segal 2023** Nat Neurosci 10.1038/s41593-023-01404-6 (n=1,294, 6질환) — 동일 영역
  편차 공유 <7%, 기능 회로 수준 수렴은 최대 56%.
- Wolfers 2019 Psychol Med 10.1017/s0033291719000084 (ADHD), Zabihi 2020 Transl Psychiatry
  10.1038/s41398-020-01057-0 (autism, 편차 공간 기반 하위유형).

### 통계 방법론
Belsley/Kuh/Welsch 1980 (10.2307/2581267) · Cook&Weisberg 1982 · **Hadi&Simonoff 1993**
JASA 10.1080/01621459.1993.10476407 · Rousseeuw&Leroy 1987 10.1002/0471725382 ·
Pregibon 1981 10.1214/aos/1176345513 (혼동 금지: 재적합 비용 문제) · Jones&Pewsey 2009 (SHASH).

### 인용 전 원문 재확인 필요 (⚠)
- Segal 2023의 <7% / 56% 수치
- Bayer 2022의 "90% 분산 손실" 수치
둘 다 이전 세션 조사 기록 기반이며 원문 문장 단위 재확인이 되지 않았다.

---

## 8. 뇌영상 NM ↔ cfRNA 매핑 (Introduction 작성용)

| 뇌영상 NM | 본 프로젝트 |
|---|---|
| 나이 / 성별 / 사이트 | 배치 / 라이브러리 조성 등 기술 공변량 10종 (연령·성별 메타데이터 부재) |
| cortical thickness (연속, 가우시안) | gene count (이산, NB 과분산) → RQR 필수 |
| 82 사이트 harmonization | 31 batch, LOBO 검증 |
| GAMLSS brain charts (Bethlehem 2022) | GAMLSS/NBI 기반 mixed-effects 엔진 |
| Wolfers/Segal 영역별 편차 중첩 | 유전자 수준 중첩 낮음 vs pathway 수준 수렴 |
| ComBat 2단계 보정 비판 | DESeq2 그룹비교 + batch 보정 관행 비판 |

---

## 9. 논문 집필 전 남은 일

1. **⚠ pathway 수렴 결과의 반론 방어**: "pathway 수준에서 수렴한다면 pathway 수준에서
   group 비교를 하면 되지 않나"가 열린다. **어느 환자가 어느 경로로 수렴하는지가 환자마다
   다르다**(환자별 경로 프로파일의 다양성)를 반드시 함께 제시해야 한다.
2. **민감도 주장의 matched null**: "더 많이 검출"은 FDR 완화로도 나온다. HC를 가짜 case로
   돌린 matched null 대비 우위를 보여야 sensitivity 주장이 선다.
3. §4.4 DESeq2 비교 수치를 v3 엔진으로 재실행 (현재 값은 v1 기준).
4. Segal 2023 / Bayer 2022 수치 원문 재확인.
5. pool route CV calibration 확인 (§3.4의 조건부 논증 해소).
6. 후속 실험 우선순위 (2026-08-04 확정): E5(2단계 보정의 분산 손실 재현) > E1(HC 참조군
   공변량 구성 리샘플에 따른 보정값·DE set 불안정성) > E3(보정 후 잔차 SD의 공변량 의존)
   > E2(batch 조건부 잔여 연속 변동).
