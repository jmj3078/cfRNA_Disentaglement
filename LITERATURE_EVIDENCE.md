# cfRNA Normative Modeling — 문헌 근거 통합 색인

작성 2026-08-07. 프로젝트 전체에서 인용 근거로 쓰인 문헌을 **"어떤 주장을 지지하는가"** 기준으로
한 곳에 모은 문서. 방법론·산출물 정리는 [MANUSCRIPT_DOSSIER.md](MANUSCRIPT_DOSSIER.md) 참조.

원본: [EDA/normative_modeling_literature.md](EDA/normative_modeling_literature.md) ·
[MixedEffectsModeling/CLAUDE.md](MixedEffectsModeling/CLAUDE.md) §1b ·
`MixedEffectsModeling/PathwayConvergence/*/pathway_literature_review.md` ·
`_legacy/.../GSEA/Analysis_Provenance.md`

**⚠ 인용 전 원문 재확인 필요**: Segal 2023의 "<7% / 56%", Bayer 2022의 "분산 90% 이상 손실".
둘 다 조사 기록 기반이며 원문 문장 단위 확인이 되지 않았다. 아래 표에도 ⚠로 표시.

---

# Part A. 서사·방법론 문헌

## A1. 왜 Normative Modeling인가 — 정당화 4축

| 축 | 주장 | 근거 문헌 |
|---|---|---|
| 1. 구조적 | 개체 수준 추론은 집단 수준 추론에서 유도되지 않는다 (생태학적 오류). 반박 불가한 기둥 | Marquand 2016 (아래) |
| 2. 실증 | 환자 간 편차 중첩이 실제로 거의 없다 — **그러나 상위 구조에서는 수렴** → 이질성은 "신호 없음"이 아니라 잘못된 해상도에서 본 신호 | Wolfers 2018, Segal 2023 |
| 3. 성능 | NM 파생 피처가 raw 피처보다 예측력이 높다 | Rutherford 2023, Parkes 2021 |
| 4. 인프라 | 참조 모델 transfer로 소규모 연구도 대규모 HC 참조를 사용 가능 | Bethlehem 2022, Rutherford 2022 ×2 |

**축 2가 본 프로젝트의 gene-vs-pathway 실험이 재현하려는 바로 그 구조다.**

## A2. 계보 — NM의 개념적 기원

| 문헌 | DOI | cit | 이 프로젝트에서의 역할 |
|---|---|---|---|
| **Marquand et al. 2016**, Biol Psychiatry, "Understanding Heterogeneity in Clinical Cohorts Using Normative Models: Beyond Case-Control Studies" | 10.1016/j.biopsych.2015.12.023 | 725 | **NM 시조.** case-control의 artificial symmetry, 진단 라벨 자체는 검증 불가, "average patient" 오류. Introduction의 첫 인용 |
| **Marquand et al. 2019**, Mol Psychiatry, "Conceptualizing mental disorders as deviations from normative functioning" | 10.1038/s41380-019-0441-1 | 459 | 소아 성장곡선 유비 명시. NM 프레임 정식화 |
| Cuthbert & Insel 2013, BMC Med, "Seven pillars of RDoC" | 10.1186/1741-7015-11-126 | 3096 | 범주적 진단 비판의 정책적 배경. NM은 RDoC가 답하지 못한 "차원의 눈금자" |
| Cuthbert 2014, World Psychiatry, RDoC framework | 10.1002/wps.20087 | 1403 | 차원적 접근 전환 |

통계 전통 (인용 없이 서술 가능): Cole LMS → GAMLSS 분포회귀(μ/σ/ν/τ 전부 공변량 함수)
→ 임상 reference interval의 공변량조건부 고해상도 확장.

## A3. 편차 중첩 실증 — 논지의 결정타

| 문헌 | DOI | cit | 수치 |
|---|---|---|---|
| **Wolfers et al. 2018**, JAMA Psychiatry (SZ/BD) | 10.1001/jamapsychiatry.2018.2467 | 540 | 집단 수준에서 유의한 회백질 감소가 있음에도, **동일 loci의 극단 편차를 공유하는 환자 비율은 대부분 영역에서 <2%** |
| **Segal et al. 2023**, Nat Neurosci (n=1,294, 6질환) | 10.1038/s41593-023-01404-6 | — | ⚠ 동일 영역 편차 공유 **<7%**, 그러나 공통 기능 회로 수준에서는 **최대 56% 수렴** |
| Wolfers et al. 2019, Psychol Med (ADHD) | 10.1017/s0033291719000084 | 227 | "average patient" 비판의 ADHD 판본 |
| Zabihi et al. 2020, Transl Psychiatry (autism) | 10.1038/s41398-020-01057-0 | 113 | 편차 공간 기반 하위유형 분해 |

**본 프로젝트의 cfRNA 판본**: 유전자 수준 편차 중첩률은 낮고 pathway 수준에서 수렴.
Segal의 "영역 <7% → 회로 56%"가 "유전자 → pathway"에 그대로 대응한다.

> **반론 방어 필수**: "pathway 수준에서 수렴한다면 pathway 수준에서 group 비교를 하면 되지
> 않나". 대응 = **어느 환자가 어느 경로로 수렴하는지가 환자마다 다르다**(환자별 경로
> 프로파일의 다양성)를 함께 제시. Segal 논문도 같은 구조로 방어한다.

## A4. NM 성능 우위

| 문헌 | DOI | cit | 내용 |
|---|---|---|---|
| Rutherford et al. 2023, eLife, "Evidence for embracing normative modeling" | 10.7554/elife.85082 | 180 | NM 출력 vs raw feature head-to-head, NM 우세 |
| Parkes et al. 2021, Transl Psychiatry | 10.1038/s41398-021-01342-6 | 107 | 편차 점수가 raw cortical volume보다 p-factor 예측 우수 |

## A5. 참조모델 인프라 — 엔진 선택의 정당화

| 문헌 | DOI | cit | 내용 |
|---|---|---|---|
| **Bethlehem et al. 2022**, Nature, "Brain charts for the human lifespan" | 10.1038/s41586-022-04554-y | 1830 | GAMLSS 기반 — **본 프로젝트 엔진과 동일 기계**. brainchart.io |
| Rutherford et al. 2022, eLife, "Charting brain growth and aging at high spatial precision" | 10.7554/elife.72904 | 216 | 82 사이트, n=58,836 참조 코호트 |
| Rutherford et al. 2022, Nat Protoc, NM framework | 10.1038/s41596-022-00696-5 | 284 | 표준 프로토콜 |

## A6. 방법론 진화 = 반론 응답의 연쇄

각 단계가 이전 단계의 어떤 반론에 답했는지가 곧 Methods의 설계 정당화다.

| 문헌 | DOI | 해결한 문제 | 본 엔진의 대응 |
|---|---|---|---|
| Marquand 2016 | (위) | GPR: 비선형 + 점별 예측 분산 | — |
| Fraza et al. 2021, NeuroImage, warped BLR | 10.1016/j.neuroimage.2021.118715 | GPR의 O(n³) 한계, 비가우시안 반응 | NB2 GLMM |
| Dinga et al. 2021, bioRxiv, GAMLSS NM | 10.1101/2021.06.14.448106 | 이분산/왜도를 분포 파라미터로. **NM 평가 표준 부재 지적** | dispersion submodel `log θ = γ0 + Σγ_k X_k` |
| **Kia et al. 2022**, PLoS ONE, federated HBR | 10.1371/journal.pone.0278776 | **사이트 효과를 random effect로 모델 내부 흡수** + 프라이버시 | `(1\|batch)` random intercept |
| Kia et al. 2021 preprint | 10.1101/2021.05.28.446120 | 위 논문 preprint | — |
| de Boer et al. 2024, Imaging Neurosci, non-Gaussian HBR (SHASH) | 10.1162/imag_a_00132 | HBR 가우시안 가정 완화 | per-gene SHASH calibration |

방향성 하나: **정상 분포의 형태 전체를 공변량 + 배치 구조를 포함해 단일 모델에서 추정.**
본 엔진(`(1|batch)` + dispersion submodel + SHASH)은 정확히 이 계보의 마지막 지점에 있다.

## A7. 2단계 보정(harmonize-then-analyze) 비판 — 논지 2의 직접 선례

**논증 구조**: 2단계로 나누면 1단계가 질병 신호를 흡수(과보정)하거나 배치를 잔존(과소보정)
시키고, 2단계는 1단계의 불확실성을 전파받지 못해 과신한다. 정합적 해법은 공변량·배치·편차를
동시에 다루는 단일 생성모델.

| 문헌 | DOI | 내용 |
|---|---|---|
| **Bayer et al. 2022**, NeuroImage | 10.1016/j.neuroimage.2022.119699 | site effect가 관심변수와 복잡히 교락. ⚠ **모든 2단계 harmonization에서 원 분산 90% 이상 손실.** ComBat에서 age/sex 미보존 시 최악. 단일단계 HBR random effect가 대안 |
| Gardner et al. 2025, Hum Brain Mapp (ComBatLS) | 10.1002/hbm.70197 | 기존 harmonization은 공변량의 **분산(scale)** 효과를 보존하지 못함 → normative score 오차, 특히 공변량이 사이트 간 불균등 분포 시 |
| Bayer et al. 2022, Front Neurol | 10.3389/fneur.2022.923988 | 사이트 효과 처리법 총론 |
| **Nygaard et al. 2016**, Biostatistics | 10.1093/biostatistics/kxv027 | **유전체판 대응.** 그룹이 배치에 불균등 분포할 때 batch 제거가 downstream 과신/위양성 유발 |
| Zindler et al. 2020, BMC Bioinformatics | 10.1186/s12859-020-03559-6 | ComBat 시뮬레이션 위양성 |

Nygaard 2016이 뇌영상 문헌을 유전체로 잇는 다리다 — cfRNA 독자에게는 이것을 먼저 인용.

## A8. 한계 — 보수적으로 명시해야 할 것

| 한계 | 근거 |
|---|---|
| 횡단 참조곡선은 실제 종단 변화를 상당히 과소추정. 궤적을 개체 변화로 해석 금지 | Di Biase et al. 2023, PNAS, 10.1073/pnas.2216798120 |
| NM 사용/평가 표준 부재 → calibration(PIT/QQ), PPC 필수 | Dinga et al. 2021 (A6) |
| 참조 표본의 선택편향이 곧 "정상"의 정의 (검증 불가) | 문헌 인용 없이 서술 |
| 꼬리(1st/99th centile) 추정 취약 — 편차 판정이 표본 희박 영역에 의존 | 문헌 인용 없이 서술 |
| 편차 ≠ 병리 | 문헌 인용 없이 서술 |

## A9. 통계 진단 문헌 — PCIS 설계 근거

PCIS(Prior-Conditioned Impact Score)가 잔차 분산을 **해당 유전자 자신의 적합 α가 아니라
dispersion trend에 조건화**하는 이유. 2026-07-28 paper-lookup(OpenAlex/Crossref) 검증.

| 문헌 | DOI | cit | 지지하는 설계 결정 |
|---|---|---|---|
| Belsley, Kuh & Welsch 1980, *Regression Diagnostics*, Wiley | 10.2307/2581267 | 6522 | **self-masking** — 외부 studentized(deleted) 잔차의 교과서적 동기. outlier가 자기 유전자의 α를 부풀려 스스로를 은폐 (측정: 20배 outlier 3개 주입 시 α 36배 팽창, 어떤 크기에서도 탐지 0건) |
| Cook & Weisberg 1982, *Residuals and Influence in Regression* | (깨끗한 DOI 없음) | — | 위의 표준 동반 참고문헌 |
| **Hadi & Simonoff 1993**, JASA 88(424) | 10.1080/01621459.1993.10476407 | 350 | **multiple masking** — 단일 관측 삭제 진단(Cook's D, ESR, DFFITS)은 2개 이상 outlier가 동시에 있으면 원리적으로 실패. 따라서 per-observation leave-one-out α는 계산이 가능하더라도 PCIS를 완전히 고치지 못한다 |
| Rousseeuw & Leroy 1987, *Robust Regression and Outlier Detection*, Wiley | 10.1002/0471725382 | 5559 | **high-breakdown 추정이 표준 대응.** PCIS의 trend-fixed α(유사 발현 수준의 ~19,000 유전자에서 lowess+bisquare로 유래)는 같은 아이디어의 강한 판본 — 한 관측이 아니라 **유전자 전체에 외부적** |
| Pregibon 1981, Ann Stat 9(4) | 10.1214/aos/1176345513 | — | **혼동 금지.** 1-step Newton 근사는 *다른* 문제(재적합 비용)의 해법이지 masking의 해법이 아니다 |
| Jones & Pewsey 2009 (SHASH, sinh-arcsinh) | — | — | per-gene Z calibration 분포족 |

**정직한 서술**: PCIS의 이 구체적 설계는 위 논문들에서 검증된 바 없다. 유비이지, NB2 GLMM에서의
선행 사용 인용이 아니다. 논문에도 이렇게 쓸 것.

주변 방법론(인용 필요 시): limma/edgeR 모멘트 분해(EB dispersion shrinkage),
DESeq2의 `qf(0.99, ...)` 관행 — 후자는 **PCIS에 이론적 근거가 없음**을 지적하기 위해 인용.

---

# Part B. 질병별 pathway 문헌 근거

`MixedEffectsModeling/PathwayConvergence/<질병>/pathway_literature_review.md`에 원본.
각 파일은 채택 pathway의 n_sig/size/eff/hist_frac + PMID 단위 기계론적 근거 + **기각된 후보와
기각 사유**까지 담고 있어 Supplementary Table로 거의 그대로 전용 가능.

## B0. 큐레이션 규칙 (모든 질병에 공통 적용됨)

1. `[GENERIC]` 플래그 행(Metabolism Of RNA, Cell Cycle, Immune System, Gene Expression 등)과
   `hist_frac > 15%` 행은 **기본 제외**. 예외는 파일 안에 명시적으로 정당화(`[GENERIC] override`).
2. 리보솜/번역, OXPHOS 계열은 전사체 전역 조성 변화를 반영하는 비특이 신호 → discount.
3. 혈액 GSEA의 신경퇴행 KEGG(Alzheimer/Parkinson/Prion/Huntington/ALS)는 사실상 OXPHOS
   proxy → 함께 discount.
4. 무관한 여러 암종 KEGG term이 한 코호트에 동시 출현하면 literal 질병이 아니라
   **공유 oncogene 모듈(TP53/RAS/RB)** 하나로 묶는다.
5. **Open Targets 등 DB 근거만으로는 채택 불가 — 실제 논문이 있어야 한다.**
   (예: Lung Cancer의 Olfactory 계열은 점수가 높았으나 폐암 특이 논문 부재로 기각)

## B1. Lung Cancer (n=26)

| Pathway | n_sig | 근거 |
|---|---|---|
| Nuclear Events Mediated By NFE2L2 | 19/26 | Romero 2017 Nat Med **28967920** (Keap1 loss가 Kras-driven 폐암 촉진, glutaminolysis 의존) · Best 2023 Cell Metab **36841242** (NRF2 활성화 → NADH-reductive stress, 표적 가능) · Cancer Cell review 2022 **36270277** |
| Mitochondrial Complex I Biogenesis | 21/26 | Nagashima 2020 ACS Chem Biol **31874028** (complex I 억제제 = NSCLC 선택적 대사 취약점) |
| GLI3 Is Processed To GLI3R By Proteasome | 19/26 | Chen 2014 Clin Cancer Res **24423612** · Skoda 2018 **29274272** · Jenkins 2019 Front Genet **31244888** |
| RNA transport | 17/26 | Kim 2016 Nature **27680702** (XPO1 핵외수송 = KRAS-mutant 폐암 특이 표적) · Gupta 2017 J Thorac Oncol **28647672** |
| Negative Regulation Of NOTCH4 Signaling | 17/26 | Baumgart 2023 Nat Commun **37268635** (NOTCH4 splice variant가 EGFR-TKI 감수성 부여) · Yuan 2020 **31894255** |
| tRNA Processing | 19/26 | Zhang 2022 **35180653** (ALKBH family NSCLC 예후) · Orellana 2022 **35721477** · 2024 Cell Death Discov **39019857** |

기각 예: Olfactory 계열 4종(특이 논문 부재), Mitochondrial Protein Import(PubMed hit 0),
HIV Infection 계열(유전자 중첩 artifact), APC/C·DNA replication 클러스터(generic proliferation).

## B2. Colorectal Cancer (n=34)

| Pathway | 근거 |
|---|---|
| Degradation Of Beta-Catenin By Destruction Complex | Wu 2019 Autophagy **30806153** · Malki 2020 Int J Mol Sci **33374459** |
| Nuclear Events Mediated By NFE2L2 | Liu 2023 Cell Death Differ **37210578** · Sadeghi 2017 Tumour Biol **28621229** |
| SCF(Skp2)-mediated Degradation Of P27/P21 | Bochis 2015 **26114183** · Fujita 2008 Am J Pathol **18535175** |
| APC/C:Cdc20 Mediated Degradation Of Securin | Wu 2013 J Transl Med **23758705** · Li 2020 **32127012** |
| Complex I Biogenesis | Tang 2022 Hum Cell **36059022** · Rai 2020 Oncol Lett **33093922** |
| Negative Regulation Of NOTCH4 Signaling | Scheurlen 2022 **35941043** (NOTCH4-GATA4-IRG1 축, early-onset CRC) · Zhang 2018 **29693251** |
| Stabilization Of P53 *(수동 추가)* | Fearon & Vogelstein 1990 Cell **2188735** (다단계 모델의 전이 정의 사건) · Song 2025 Cancer Res **40882016** |

## B3. Esophagus Cancer (n=25)

| Pathway | 근거 |
|---|---|
| Nuclear Events Mediated By NFE2L2 | Ninomiya 2025 Br J Cancer **40781161** (ctDNA NFE2L2 변이가 ESCC 화학방사선 반응 예측) · 2018 Ann NY Acad Sci **29752726** · Ostrowski 2017 **28760781** |
| Regulation Of RUNX3 Expression And Activity | Xu 2014 Med Oncol **25391920** · 2013 APJCP **24175838** · Sano 2025 Cancer Sci **39440906** |
| SCF(Skp2)-mediated Degradation Of P27/P21 | Cao 2021 Chin J Cancer Res **35125808** (ZNF292/SKP2/P27 축) |
| Neddylation | Zhang 2021 **33733647** · Wang 2021 IJMS **33572115** · Wang 2020 Signal Transduct Target Ther **32651357** |
| Degradation Of AXIN | Wu 2023 Curr Pharm Des **37957865** |
| Keratinization | Aiba 2024 Front Oncol **39711958** · Yamada 2025 **40535470** |
| Transcriptional Regulation By TP53 *(수동 추가)* | Abedi-Ardekani 2011 PLoS One **22216294** |

## B4. Stomach Cancer (n=21)

| Pathway | 근거 |
|---|---|
| Negative Regulation Of NOTCH4 Signaling | Qian 2015 **25511451** · Zhang 2024 Front Pharmacol **39309008** |
| Separation Of Sister Chromatids | TCGA 2014 Nature **25079317** (CIN subtype 정의) · Nemtsova 2023 **38069284** |
| DNA Repair | Marabelle 2020 JCO **31682550** (KEYNOTE-158, MSI-H/dMMR) · Ooki 2024 **38922524** · Andre 2023 JCO **35969830** |
| Ribosome biogenesis in eukaryotes | Zang 2025 **41402710** · Li 2024 **38479160** · Nie 2023 **36941105** |
| Transcriptional Regulation By TP53 `[GENERIC override]` | TCGA 2014 **25079317** · Cristescu 2015 Nat Med **25894828** |

**음성 검색 기록도 보존됨** (기각 근거로 인용 가능): `FZR1+Cdh1+APC/C AND gastric cancer` → 0건,
`MCM+DNA replication licensing AND gastric cancer` → 0건.

## B5. Liver Cancer — Chen et al. (n=10, 소코호트: 재현/효과 통계 주의)

| Pathway | 근거 |
|---|---|
| Complement and coagulation cascades | Su 2024 Heliyon **39391504** · Front Oncol review 2020 **33718121** |
| Degradation Of Beta-Catenin By Destruction Complex | Perugorria 2022 JCI **35166233** · Ge 2018 Cancer Res **29483096** |
| Regulation Of IGF Transport And Uptake By IGFBPs | Aleem 2003 Clin Endocrinol **14974910** |
| GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2 | Sun 2016 Hepatology **26403645** (p62-Keap1-NRF2, ferroptosis 저항) · Zhao 2020 Nat Commun **31953436** |
| Protein processing in endoplasmic reticulum | Lebeaupin 2018 J Hepatol **29940269** · Chen 2025 Phytomedicine **40424981** |
| Negative Regulation Of NOTCH4 Signaling | Zhu 2017 **29058285** · Yin 2020 **33118329** · Giovannini 2019 **31031867** |
| Regulation Of PTEN Stability And Activity | Lin 2025 J Transl Med **40394639** · Wei 2023 Cancer Med **35861040** |
| Transcriptional Regulation By TP53 `[GENERIC override]` | TCGA 2017 Cell **28622513** · Jiang 2019 Cell **31585088** |

## B6. Liver Cancer — Roskams-Hieter et al. (n=28)

| Pathway | 근거 |
|---|---|
| Signaling By Rho GTPases | Zhu 2022 Exp Hematol Oncol **36348464** · Xu 2024 Hepatol Res **37792600** · Yan 2024 Cell Death Dis **38378644** |
| Neutrophil extracellular trap formation | Zhan 2023 Cancer Commun **36346061** · Yang 2020 J Hematol Oncol **31907001** · Zhu 2024 Cancer Res **38381538** |
| Complement and coagulation cascades | Ye 2024 Heliyon **39391504** · Xu 2020 Front Oncol **33718121** |
| Chromatin Modifying Enzymes | Toh 2022 Semin Cancer Biol **34324953** · Bayo 2022 JECCR **35331312** |
| Lysosome | Zhang 2022 Autophagy **34890308** · Chen 2024 Autophagy **37733919** |
| Senescence-Associated Secretory Phenotype (SASP) | Yoshimoto 2013 Nature **23803760** (장내 대사물이 senescence secretome 통해 간암 촉진) · Lee 2017 PLoS One **28273155** |
| Alcoholism | Shukla & Lim 2013 Alcohol Res **24313164** · Seitz & Stickel 2015 **25427901** |

> 같은 질병(간암)이 코호트 2개에서 **서로 다른 pathway 집합**으로 나타난다는 점 자체가
> 논지 2(개인/코호트 수준 이질성)의 증거로 쓰일 수 있다. 단, 배치 교란과 분리 불가하므로
> 보수적으로 서술할 것 (MANUSCRIPT_DOSSIER §5).

## B7. Pancreatic Cancer (n=72)

| Pathway | 근거 |
|---|---|
| Neutrophil Degranulation | Zhu 2021 Cancer Res **33941611** (PDAC 유래 TIMP1이 NET 형성 유발) |
| Complement and coagulation cascades | Aykut 2019 Nature **31578522** (mycobiome → MBL/complement) · Han 2025 **41417111** |
| Regulation Of IGF Transport And Uptake By IGFBPs | Xu 2024 Open Med **39221034** · Li 2022 J Pers Med **36556226** |
| Extracellular Matrix Organization | Provenzano 2012 Cancer Cell **22439937** (hyaluronidase로 stromal barrier 제거 시 치료 반응 개선) · Sun 2022 Nature **36198801** |
| Platelet Degranulation | Chen 2022 Front Oncol **35494001** |
| Degradation Of GLI1 By Proteasome | Singh & Rai 2022 Pharmacol Ther **34999181** · Skoda 2019 JECCR **31661013** |
| Stabilization Of P53 *(수동 추가)* | Bailey 2016 Nature **26909576** (456 PDAC 통합 유전체, squamous subtype의 TP53 농축) |

## B8. Pancreatitis (n=79)

| Pathway | 근거 |
|---|---|
| Neutrophil Degranulation | Wang 2023 Biomolecules **36830652** (급성췌장염 말초혈 전사체) · Osman 2008 Pancreas **18580443** |
| Neutrophil Extracellular Trap Formation | Zhou 2022 Chin Med J **36729096** · Merza 2020 **33281940** · Li 2023 Nat Commun **37794047** · Chen 2022 Front Immunol **36032164** · Zhang 2023 Redox Biol **37392517** |
| Interleukin-1 Signaling | Norman 1997 J Surg Res **9070189** · Sendler & Mayerle 2020 IJMS **32751171** (NLRP3/IL-1β) · Norman 1996 Gastroenterology **8566616** |

## B9. Pre-eclampsia (n=58)

| Pathway | 근거 |
|---|---|
| Neutrophil Degranulation | Faas & de Vos 2018 Front Endocrinol **30298053** · Rimon 2020 Front Immunol **32117288** |
| Signaling By Interleukins | Ozler 2022 **31964198** (IL-6와 late-onset PE 중증도) · Xu/Fan 2017 **28252161** |
| Diseases Of Signal Transduction By Growth Factor Receptors | **Maynard 2003 JCI 12618519** (과잉 태반 sFlt1이 내피 기능이상/고혈압/단백뇨 유발 — anti-angiogenic imbalance의 기초 논문) · 2025 IJMS review **41226469** |
| Platelet Degranulation | Sancak 2024 **38537226** |
| NOD-like Receptor Signaling Pathway | Weel 2020 Front Endocrinol **32161574** (NLRP3) · Vishnyakova 2023 **37788097** |
| NF-kappa B Signaling Pathway | Wang 2024 Placenta **38008034** |
| Th17 Cell Differentiation | Wang 2014 **25664035** · Saito 2020 Front Immunol **32973809** |
| Neutrophil Extracellular Trap Formation | Zhu 2024 Pharmaceuticals **38794175** · Domingues 2023 IJMS **37958788** |

Maynard 2003(sFlt1)은 **양성 대조 수준의 근거** — 자간전증에서 이 축이 나오지 않으면 오히려
모델을 의심해야 하는 표준 소견이다. Figure에서 positive control로 표시할 것.

## B10. Tuberculosis (n=101)

| Pathway | 근거 |
|---|---|
| Interferon Signaling | **Berry 2010 Nature 20725040** (IFN-inducible, neutrophil-driven 혈액 전사체 signature가 활동성 TB를 잠복/타질환과 구별) · Ivashkiv & Donlin 2015 Nat Rev Immunol **25614319** |
| Neutrophil extracellular trap formation | Cell Death Dis 2024 **39085192** (건락성 육아종의 NETosis) · Roe 2022 PLoS One **36454773** |
| Bacterial invasion of epithelial cells | Ryndak & Laal 2019 Front Cell Infect Microbiol **31497538** |
| Neutrophil Degranulation | Roe 2022 PLoS One **36454773** (**진단 6개월 전** 전혈에서 co-induction) |
| Platelet Activation, Signaling And Aggregation | Kroon 2021 Front Immunol **34093524** |
| RHO GTPases Activate WASPs And WAVEs `[GENERIC override]` | Wang 2024 mBio **39287444** · Fort 2019 Nat Microbiol **31285585** |

Berry 2010은 혈액 전사체 TB signature의 **정전(canonical) 논문**이자 양성 대조.
Roe 2022의 "진단 6개월 전" 결과는 본 프로젝트의 개별 샘플 편차 탐지 주장과 직접 맞물린다.

## B11. 질병 간 반복 출현 pathway (수렴의 축)

| Pathway | 출현 질병 |
|---|---|
| **Nuclear Events Mediated By NFE2L2 / KEAP1-NRF2** | Lung, Colorectal, Esophagus, Liver(Chen, GSK3B-BTRC 경유) — **암 4종** |
| **Negative Regulation Of NOTCH4 Signaling** | Lung, Colorectal, Stomach, Liver(Chen) — **암 4종** |
| **Neutrophil Degranulation / NET formation** | Pancreatitis, Pre-eclampsia, Tuberculosis, Pancreatic Cancer, Liver(R-H) — **염증성 5종** |
| Complement and coagulation cascades | Liver ×2, Pancreatic Cancer |
| Stabilization / Transcriptional Regulation By TP53 | Colorectal, Esophagus, Pancreatic, Stomach, Liver(Chen) |
| SCF(Skp2) degradation of P27/P21 | Colorectal, Esophagus |
| GLI proteasomal processing (Hedgehog) | Lung(GLI3), Pancreatic(GLI1) |
| Regulation Of IGF Transport By IGFBPs | Liver(Chen), Pancreatic |

**해석 주의 (반드시 병기)**: 이 표는 두 갈래로 읽힌다 — (a) 진짜 pan-cancer 수렴,
(b) 큐레이션 규칙 4가 예고한 **공유 oncogene 모듈의 재출현**. NRF2/NOTCH4/TP53 축은 (b)일
가능성이 높으므로 질병 특이성을 주장하지 말고 "cfRNA에서 pan-cancer 프로그램이 개별 샘플
수준에서 검출된다"로 서술하는 것이 안전하다. 반대로 Neutrophil/NET 축이 염증성 질환에만,
Maynard sFlt1 축이 자간전증에만 나타나는 **선택적 출현**이 특이성의 실제 증거다.

---

# Part C. 외부 데이터베이스 근거

| 자원 | 용도 | provenance |
|---|---|---|
| **Open Targets** | 질병별 association 상위 300 유전자 = `Benchmark/disease_reference/{pheno}.json`. GSEA lead gene의 DB 지지 채점에 사용 | `_legacy/.../GSEA/Analysis_Provenance.md` — 전 엔드포인트/GraphQL 쿼리/resolve된 MONDO·EFO ID, 수집 2026-07-01. 재생성: `Modeling/build_disease_reference.py` |
| **PubMed** | pathway 문헌 검증 (PMID 단위), 23개 쿼리 + 히트수 기록 | 같은 파일 |
| Reactome / KEGG | pathway 정의 | GSEA gene set |

**DB 지지는 대리지표이지 정답이 아니다.** 논문에서 이 수치를 쓸 때 반드시 병기할 것.
또한 두 GSEA 분기(no_filter / with_rare)는 랭킹·정규화·유전자 우주가 달라 concordance를
측정하는 것이지 우열을 판정하는 것이 아니다.

주요 정량 결과 (v1 엔진 기준 — v3로 재실행 필요):
- 동일 규칙 대칭 채점 시 **정밀도는 세 방법 동등** (DESeq2 0.486 / no_filter 0.438 / with_rare 0.445),
  DB-지지 term **개수**는 Normative가 1.7–2배 (2,236 vs 3,843 / 4,375)
  → 주장은 "더 많이 찾았다"가 아니라 **"정밀도 손실 없이 커버리지 확대"**여야 한다.
- DESeq2는 normative DB-지지 경로의 중앙값 ~16%만 포착.
- rare 분기 포함은 방향 보존적·가산적: 20질병 전부 공유 term NES 부호 일치율 1.0.
- rare가 처음 표면화한 미보고 후보: ME/CFS **PLA2G10**(관련 문헌 0건), 자간전증 **IFNL3/IFN-λ**(1건),
  ICI 심근염 **MT1B/metallothionein**(2건), 간암(R-H) **DEFB114/MAPK**(3건).

---

# Part D. 문헌이 **없는** 주장 (정직하게 표시할 것)

논문 심사에서 "근거는?"이 나올 텐데, 아래는 내부 측정만 있고 문헌 선례가 없다.
숨기지 말고 "measured, not cited"로 명시하는 편이 안전하다.

| 주장 | 상태 |
|---|---|
| PCIS의 trend-fixed α 설계 자체 | 유비는 A9로 지지되나 **NB2 GLMM에서의 선행 사용 인용 없음** |
| `pcis_cut = 2.28` | 자체 경험적 null(13,276,494 draw)로만 정당화. real-vs-null 교차점은 단일 기준점 추정 |
| `nz_a_max = 25` (pooling 경계) | 자체 수렴율 knee 측정. **pool route는 실제 HC에 실행된 적 없음 → 조건부 논증** |
| `tau2_max = 3.0` | 자체 검출력 측정. 임계값이 날카롭지 않음을 스스로 인정(제거 305 유전자 중 ~21%만 입증 가능하게 undetectable) |
| EXCLUDED_GENES 이중봉 기준 | 수동 큐레이션 7개, 자동 detector 미구현 |
| 기술 공변량만으로 만든 "규범"의 타당성 | 문헌 선례 없음. NM 문헌은 전부 생물학적 공변량(연령/성별) 기반. **이 논문의 가장 큰 개념적 확장이자 가장 큰 취약점** |

마지막 항목은 약점을 강점으로 전환 가능: 생물학적 메타데이터가 없어도 작동한다 =
메타데이터가 빈약한 **기존 공개 cfRNA 자산 전체에 소급 적용 가능**하다. Discussion의 확장성 논거.
