# Normative Modeling 이론적 배경 - 문헌 목록

조사일 2026-08-04. NM 정당화 서사의 근거 문헌. 각 항목은 논증 역할별로 분류.

## 1. 기원 · 개념 정립

| 문헌 | DOI | cit | 역할 |
|---|---|---|---|
| Marquand et al. 2016, Biol Psychiatry, "Understanding Heterogeneity in Clinical Cohorts Using Normative Models: Beyond Case-Control Studies" | 10.1016/j.biopsych.2015.12.023 | 725 | NM 시조. case-control의 artificial symmetry, 진단 라벨 자체를 검증 불가. GPR 기반 |
| Marquand et al. 2019, Mol Psychiatry, "Conceptualizing mental disorders as deviations from normative functioning" | 10.1038/s41380-019-0441-1 | 459 | 소아 성장곡선 유비 명시. NM 프레임 정식화 |
| Cuthbert & Insel 2013, BMC Med, "Seven pillars of RDoC" | 10.1186/1741-7015-11-126 | 3096 | 범주적 진단 비판의 정책적 배경 |
| Cuthbert 2014, World Psychiatry, RDoC framework | 10.1002/wps.20087 | 1403 | 차원적 접근 전환. NM은 "차원의 눈금자" 역할 |

## 2. 핵심 실증 - 개체간 편차 중첩이 낮다 (논증의 결정타)

| 문헌 | DOI | cit | 수치 |
|---|---|---|---|
| Wolfers et al. 2018, JAMA Psychiatry (SZ/BD) | 10.1001/jamapsychiatry.2018.2467 | 540 | 집단 수준 유의한 회백질 감소에도, 동일 loci 극단편차 공유 환자 비율 대부분 영역 <2% |
| Segal et al. 2023, Nat Neurosci (n=1294, 6질환) | 10.1038/s41593-023-01404-6 | - | 동일 영역 편차 공유 <7%, 기능 회로 수준 수렴은 최대 56% |
| Wolfers et al. 2019, Psychol Med (ADHD) | 10.1017/s0033291719000084 | 227 | "average patient" 비판의 ADHD 판본 |
| Zabihi et al. 2020, Transl Psychiatry (autism) | 10.1038/s41398-020-01057-0 | 113 | 편차 공간 기반 하위유형 분해 |

## 3. 성능 우위 - NM 피처 > raw 피처

| 문헌 | DOI | cit | 내용 |
|---|---|---|---|
| Rutherford et al. 2023, eLife, "Evidence for embracing normative modeling" | 10.7554/elife.85082 | 180 | NM 출력 vs raw feature head-to-head, NM 우세 |
| Parkes et al. 2021, Transl Psychiatry | 10.1038/s41398-021-01342-6 | 107 | 편차 점수가 raw cortical volume보다 p-factor 예측 우수 |

## 4. 참조모델 인프라 (재사용 정당화)

| 문헌 | DOI | cit | 내용 |
|---|---|---|---|
| Bethlehem et al. 2022, Nature, "Brain charts for the human lifespan" | 10.1038/s41586-022-04554-y | 1830 | GAMLSS 기반 (본 프로젝트 엔진과 동일 기계). brainchart.io |
| Rutherford et al. 2022, eLife, "Charting brain growth and aging at high spatial precision" | 10.7554/elife.72904 | 216 | 82 사이트, n=58,836 참조 코호트 |
| Rutherford et al. 2022, Nat Protoc, NM framework | 10.1038/s41596-022-00696-5 | 284 | 표준 프로토콜 |

## 5. 방법론 진화 - 각 단계가 응답한 반론

| 문헌 | DOI | 해결한 문제 |
|---|---|---|
| Marquand 2016 (위) | - | GPR: 비선형 + 점별 예측 분산 |
| Fraza et al. 2021, NeuroImage, warped BLR | 10.1016/j.neuroimage.2021.118715 | GPR의 O(n^3) 한계, 비가우시안 반응 |
| Dinga et al. 2021, bioRxiv, GAMLSS NM | 10.1101/2021.06.14.448106 | 이분산/왜도를 분포 파라미터로. NM 평가 표준 부재 지적 |
| Kia et al. 2022, PLoS ONE, federated HBR | 10.1371/journal.pone.0278776 | 사이트 효과를 random effect로 모델 내부 흡수 + 프라이버시 |
| Kia et al. 2021, bioRxiv preprint | 10.1101/2021.05.28.446120 | 위 논문 preprint |
| de Boer et al. 2024, Imaging Neurosci, non-Gaussian HBR (SHASH) | 10.1162/imag_a_00132 | HBR 가우시안 가정 완화 |

## 6. 2단계 보정(harmonize-then-analyze) 비판 - 본 프로젝트 논지 2의 직접 선례

| 문헌 | DOI | 내용 |
|---|---|---|
| Bayer et al. 2022, NeuroImage | 10.1016/j.neuroimage.2022.119699 | site effect가 관심변수와 복잡히 교락. **모든 2단계 harmonization에서 원 분산 90% 이상 손실**. ComBat에서 age/sex 미보존시 최악. 단일단계 HBR random effect가 대안 |
| Gardner et al. 2025, Hum Brain Mapp (ComBatLS) | 10.1002/hbm.70197 | 기존 harmonization은 공변량의 분산(scale) 효과 미보존 -> normative score 오차, 특히 공변량이 사이트간 불균등 분포시 |
| Bayer et al. 2022, Front Neurol, site effects how-to | 10.3389/fneur.2022.923988 | 사이트 효과 처리법 총론 |
| Nygaard et al. 2016, Biostatistics | 10.1093/biostatistics/kxv027 | 유전체판 대응. 그룹이 배치에 불균등 분포시 batch 제거가 과신/false positive 유발 |
| Zindler et al. 2020, BMC Bioinformatics | 10.1186/s12859-020-03559-6 | ComBat 시뮬레이션 false positive |

## 7. 한계 · 반론

| 문헌 | DOI | 내용 |
|---|---|---|
| Di Biase et al. 2023, PNAS | 10.1073/pnas.2216798120 | 횡단 brain chart는 실제 종단 변화를 상당히 과소추정. 궤적을 개체 변화로 해석 금지 |
| Dinga et al. 2021 (위) | - | NM 사용/평가 표준 부재 -> calibration(PIT/QQ), PPC 필수 |

기타 검증 불가/미인용 한계 (문헌 인용 없이 서술): 참조 표본 선택편향이 곧 정상성 정의, 꼬리(1st/99th centile) 추정 취약성, 편차 != 병리.

## 8. cfRNA 프로젝트 매핑

| 뇌영상 NM | 본 프로젝트 |
|---|---|
| 나이/성별/사이트 | 나이/성별/배치/라이브러리 조성 |
| cortical thickness (연속, 가우시안) | gene count (이산, NB 과분산) -> RQR 필수 |
| 82 사이트 harmonization | 31 batch, LOBO 검증 |
| GAMLSS brain charts | GAMLSS/NBI 기반 엔진 |
| Wolfers/Segal 영역별 편차 중첩 | 유전자 수준 중첩 낮음 vs pathway 수렴 실험 |
| ComBat 2단계 비판 | DESeq2 그룹비교 + batch 보정 관행 비판 |

## 확인 필요

Segal 2023 (<7%, 56%)과 Bayer 2022 (90% 분산 손실) 수치는 이전 세션 조사 기록 기반. 논문 인용 전 원문 문장 단위 재확인 필요.
