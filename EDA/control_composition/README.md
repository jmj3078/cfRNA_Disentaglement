# Control-composition sensitivity experiment (Moore et al. Batch_1)

논지 2("그룹 비교는 대조군 구성만 바꿔도 재현되지 않는다")의 실증. 코드
[run_control_composition.py](run_control_composition.py) (Welch-t arm),
[run_control_composition_deseq2.py](run_control_composition_deseq2.py) (DESeq2 arm),
[ruvg_batch.R](ruvg_batch.R) (RUVg 적합). 산출물 `EDA/Analysis_Results/Control_Composition/`.

## 1. 원리

### 답하려는 질문
"DE로 뽑은 바이오마커 목록은, 케이스를 그대로 두고 **대조군 구성만 바꿨을 때** 유지되는가?"

만약 마커가 질병 생물학을 잡고 있다면 대조군을 어떻게 고르든 같은 유전자가 나와야 한다.
바뀐다면 그 마커는 질병이 아니라 **대조군의 우연한 기술적 구성**을 잡고 있는 것이다.

### 설계의 핵심 — 두 개의 통제
1. **케이스 고정**: 모든 비교에서 case cohort는 완전히 동일하다. 변하는 것은 control뿐이므로,
   목록이 변한다면 원인은 control 구성 하나로 특정된다.
2. **동일 study·동일 batch 내부에서만 분할**: control을 다른 코호트에서 가져오지 않는다.
   따라서 "다른 병원/다른 프로토콜이라 그렇다"는 설명이 원천 차단된다. 남는 것은
   *같은 배치 안에서의* 기술적 이질성뿐이다.

### 왜 null 분포가 필수인가 (이 실험의 급소)
tertile로 쪼개면 각 stratum의 n이 1/3로 줄어든다. 목록이 안 겹치는 것이 **기술 축 때문인지
단지 n이 작아서인지** 구분해야 한다. 그래서 같은 HC 풀을 **크기만 맞춘 무작위 3분할**로
나눈 null을 3,600 draw 만든다.

- tertile Jaccard **>** null → 기술 축이 구조를 만든다 (그래도 재현성은 낮음)
- tertile Jaccard **≈ 또는 <** null → 불안정성은 축과 무관, **대조군을 어떻게 고르든** 발생

실제 결과는 후자다. 이 대칭성이 **"control matching을 잘하면 되지 않느냐"는 반론을 봉쇄한다.**

## 2. 데이터

### 코호트
| 항목 | 값 |
|---|---|
| Batch | `Moore et al._Batch_1` (단일 batch) |
| 총 샘플 | **220** |
| Pancreatitis (case) | **79** |
| Pancreatic Cancer (case) | **72** |
| Healthy Control | **69** |
| 유전자 | **18,892** (protein_coding, raw count 합 >= 10) |
| RUVg control 유전자 | **126** (PalangoDB Platelet 마커) |

### 샘플 선별 경로 (`build_cache`)
1. `QC_Passed == True`
2. `Phenotype_Processed` 결측/`Unknown` 제거
3. `broad_protocol_category != "Exome-based (EB)"` 제외
4. **OOD 필터**: 10개 기술 공변량 공간에서 HC 훈련셋 기준
   `MahalanobisFilter(percentile=ood_percentile)` + `RangeFilter(n_out_thr=2)` 통과분만
   - 훈련 HC는 `MIN_HC_BATCH_SIZE` 미만 batch를 제외한 HC 전체
5. `Batch_ID == Moore et al._Batch_1` 이고 phenotype이 HC / Pancreatic Cancer / Pancreatitis

### 기술 축 11종
`config.BIAS_COLUMNS` 10개 + 표준화 후 PC1:

log(Total Reads), Spliced Reads (%), gDNA Contamination (Intron/Exon), rRNA Fraction,
RNA Degradation (3' Bias), Platelet Score, GC Bias, Gene Length Bias, NG80, (NP80/NG80),
**PC1_bias** (10개 표준화 행렬의 제1주성분 = 복합 기술 축)

### 정규화 층 9종
| 종류 | 층 |
|---|---|
| static (per-sample scaling, 재적합 불필요) | CPM_log1p, TPM_log2, TMM_log2 |
| dynamic (비교마다 RUVg 재적합) | RUVg_Platelet_k1/k2/k3, Proposed_Full_k1/k2/k3 |

DESeq2 arm은 별도로 2개 design: `~condition`, `~W_1+W_2+condition`.

## 3. 절차

### 3.1 분할
- **tertile split**: HC 69명을 각 축의 값으로 순위 매겨 3등분 → **23 / 23 / 23**
  (`rankdata(method="ordinal")` 기반이라 동점에도 크기가 정확히 유지됨)
- **random split**: 같은 HC 풀을 무작위 순열 후 동일한 크기(23/23/23)로 분할, seed 42

각 stratum은 고정된 case와 짝지어 하나의 "비교(subset)"가 된다.

### 3.2 비교 수
| | 계산 | 개수 |
|---|---|---|
| tertile subset | 2 질환 x 11 축 x 3 stratum | 66 |
| null subset | 2 질환 x 200 draw x 3 stratum | 1,200 |
| **총 RUVg 적합** | | **1,266** |

결과 행 수: tertile 198 (= 2 x 11 x 9층), null 3,600 (= 2 x 200 x 9층),
DESeq2 tertile 44 (= 2 x 11 x 2design), DESeq2 null 120 (= 2 x 30 x 2design).

### 3.3 RUVg 재적합 — 방법론적으로 가장 중요한 지점
저장된 `RUVg_Platelet_*` / `Proposed_Full_*` 층은 **전체 코호트에서 적합된 것**이라,
개별 비교만 수행하는 실제 파이프라인이 가질 수 없는 정보를 담고 있다. 그대로 쓰면
정보 누출이다. 따라서 **1,266개 비교 각각에서 (case + 해당 stratum)만으로 RUVg를 새로 적합**한다.

- `RUVSeq::RUVg(tmm_log2[, samples], control_genes, k=3, isLog=TRUE, center=TRUE)`
- `RUVg_Platelet_k*` = TMM_log2에서 W 컬럼 OLS 제거 (intercept 없음, RUVSeq `normalizedCounts`와 동치)
- `Proposed_Full_k*` = EDA_Full_All에서 W 제거 (intercept 포함, `limma::removeBatchEffect`와 동치)
- CPM/TPM/TMM는 per-sample scaling이라 재적합 대상이 아니며 저장 층을 그대로 사용

**자기검증**: `verify_residualize()`가 R이 덤프한 `normalizedCounts`와 python 잔차화 결과를
비교해 `max|diff| < 1e-6`을 assert한다. 어긋나면 조용히 발산하지 않고 즉시 실패한다.

### 3.4 통계량과 지표
- Welch t (`welch_t`), 각 층·각 비교마다 전 유전자
- `|t|` 내림차순 정렬 후 top-k 집합 (k = 25, 50, 100, 200, 500)
- **Jaccard** = |A∩B| / |A∪B|, 한 group 내부 3개 stratum의 3쌍(0-1, 0-2, 1-2) 평균
- **signflip**: 교집합 유전자 중 t 부호가 뒤집힌 비율
- **Spearman**: 전 유전자 t 랭킹 상관
- **delta_d**: 3 stratum 각각의 (case vs stratum) Cohen's d 중 max - min = 그 축에서 분할이
  만들어낸 실제 분리 폭

DESeq2 arm은 동일 subset·동일 tag·동일 seed를 쓰고, W도 Welch-t arm이 적합한 것을
재사용한다. 다만 count model에서 raw count 잔차화는 무의미하므로 **W를 GLM design에
공변량으로 투입**(`~W_1+W_2+condition`)한다. null draw는 30으로 축소 (DESeq2 적합이 ~10s/건).
고정 seed의 앞 N draw는 더 큰 실행의 prefix이므로 캐시 재사용이 성립한다.

## 4. 결과

### 4.1 층별 Jaccard (tertile vs null)

| Layer | k=25 | k=50 | k=100 | k=200 | k=500 | Spearman |
|---|---|---|---|---|---|---|
| CPM_log1p | 0.0079 | 0.0097 | 0.0148 | 0.0219 | 0.0386 | 0.229 |
| TPM_log2 | 0.0059 | 0.0082 | 0.0111 | 0.0198 | 0.0371 | 0.222 |
| TMM_log2 | 0.0067 | 0.0113 | 0.0152 | 0.0218 | 0.0369 | 0.231 |
| RUVg_Platelet_k1 | 0.0044 | 0.0049 | 0.0107 | 0.0175 | 0.0313 | 0.266 |
| RUVg_Platelet_k2 | 0.0025 | 0.0046 | 0.0104 | 0.0156 | 0.0305 | 0.275 |
| RUVg_Platelet_k3 | 0.0022 | 0.0042 | 0.0089 | 0.0150 | 0.0295 | 0.267 |
| Proposed_Full_k1 | 0.0037 | 0.0056 | 0.0093 | 0.0168 | 0.0332 | 0.179 |
| Proposed_Full_k2 | 0.0025 | 0.0056 | 0.0096 | 0.0156 | 0.0287 | 0.257 |
| Proposed_Full_k3 | 0.0019 | 0.0065 | 0.0093 | 0.0146 | 0.0284 | 0.248 |
| **tertile 전체 평균** | **0.0042** | **0.0067** | **0.0110** | **0.0176** | **0.0327** | **0.242** |
| **random null 전체 평균** | **0.0072** | **0.0115** | **0.0176** | **0.0270** | **0.0481** | **0.325** |

### 4.2 DESeq2 arm

| Design | k=25 | k=50 | k=100 | k=200 | k=500 | Spearman |
|---|---|---|---|---|---|---|
| `~condition` (tertile) | 0.0177 | 0.0284 | 0.0416 | 0.0524 | 0.0730 | 0.256 |
| `~condition` (null) | 0.0294 | 0.0471 | 0.0588 | 0.0686 | 0.0855 | 0.325 |
| `~W1+W2+condition` (tertile) | 0.0431 | 0.0716 | 0.1024 | 0.0901 | 0.0782 | 0.254 |
| `~W1+W2+condition` (null) | 0.0650 | 0.0827 | 0.1155 | 0.0974 | 0.0874 | 0.313 |

### 4.3 축별 분할 강도 (delta_d, 3 stratum Cohen's d의 max-min)

Platelet Score 2.27 > log(Total Reads) 2.21 > NG80 2.18 > gDNA 2.06 > Gene Length Bias 2.05
> Spliced Reads% 2.02 ≈ PC1_bias 2.02 > rRNA 1.98 > (NP80/NG80) 1.81 > GC Bias 1.77
> RNA Degradation 1.67

분할 자체는 확실히 작동했다 (모든 축에서 d 차이 1.7 이상). 즉 낮은 Jaccard는
"분할이 약해서"가 아니다.

## 5. 읽는 법

1. **DE 유전자 집합의 재현성이 사실상 0이다.** top-100 Jaccard 0.011~0.10.
   100개 중 겹치는 게 한 자릿수라는 뜻.
2. **구조화된 split이 무작위 split보다 나을 것이 없다.** 모든 층, 모든 k에서
   tertile <= null. 불안정성은 "잘못된 대조군을 골라서"가 아니라 **대조군을 어떻게 고르든**
   발생한다. `control matching을 개선하면 된다`는 반론이 여기서 막힌다.
3. **보정이 구제하지 못한다.** RUVg 층은 오히려 Jaccard가 더 낮다 (k=100에서 CPM 0.0148 vs
   RUVg_k3 0.0089). Spearman은 소폭 오르므로 전체 랭킹은 안정화되지만
   **상위 후보 집합은 더 불안정해진다** — 바이오마커 선택은 상위 집합에서 일어난다.
4. **DESeq2가 Welch-t보다 Jaccard가 높다** (shrinkage로 랭킹이 발현량 쪽으로 안정화된 결과).
   `~W1+W2+condition`이 가장 높지만 그래도 top-100에서 0.10이고, 자신의 null(0.116)에
   못 미친다. 절대 수준이 재현이라 부를 수 없다는 결론은 그대로다.
5. **모든 층에서 tertile Spearman < null Spearman.** 기술 축 분할이 전 유전자 랭킹까지
   무작위 분할보다 더 흔든다.

## 6. 한계 (논문에 명시할 것)

- 단일 batch(Moore Batch_1) 단일 결과다. 일반화 주장은 이 배치의 n=220에 한정된다.
- HC 69명을 3분할하므로 stratum당 n=23이다. null이 이 n을 통제하지만, n=23 자체가
  작다는 사실은 남는다. 더 큰 HC 배치에서의 재현이 필요하다.
- DESeq2 null은 30 draw로 Welch-t arm(200)보다 얕다.
- Jaccard는 집합 지표라 "거의 뽑힐 뻔한" 유전자를 반영하지 않는다. Spearman을 병기한 이유.
- Ibarra et al.은 phenotype과 batch가 설계상 완전 교락이라 애초에 이 계열 분석에서 제외됨
  (이 실험은 Moore 단일 배치이므로 직접 관련은 없음).
