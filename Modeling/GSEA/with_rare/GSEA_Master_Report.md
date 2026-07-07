# cfRNA Normative Model - GSEA with_rare 통합 분석 리포트

**분석:** rare-event 분기(559개 극저발현 protein-coding 유전자, HC 검출률 <1%, pooled 공변량 GLM) 포함 GSEA(with_rare) vs 미포함(only_nbi) 비교. Z-score = normative model의 randomized quantile residual을 prerank로 GSEA.
**유의 기준:** FDR q-val < 0.05 - **NES:** >0 HC 대비 상향, <0 하향.
**교차검증:** 모든 문헌·DB 근거는 paper-lookup(PubMed E-utilities)·database-lookup(Open Targets Platform GraphQL) skill을 직접 호출해 수집한 공인 정보만 사용. 유전자별 조회 원본은 `GSEA/Reference/<phenotype>_refs.md`에, 질병별 정량 백본은 `GSEA/Reference/`에 보관.
**핵심 정량:** rare 포함은 공유 term의 NES 부호를 **20개 전 질병에서 한 번도 뒤집지 않음(sign_agree=1.0)** = 순수 가산적. 질병당 신규 유의 term +49~+172개.

> 자동 생성 정량 백본(요약 테이블·rare-led 카탈로그) 위에, 질병별로 skill 검증한 문헌·DB 해석을 덧붙이는 구조. "rare-led novel"=rare-분기 유전자가 직접 leading-edge에 든 신규 경로(가장 직접적 근거), "DB 지지 novel"=lead genes가 Open Targets 질병 상위 300 target과 겹치는 신규 경로.

---

## 전체 요약 테이블 (rare 미포함 only_nbi vs 포함 with_rare)

| Phenotype | only_nbi | with_rare | 신규(+) | 소실(-) | Jaccard | 부호일치 | DB적중률 | rare-led |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ICI-treated Cancer (Raissadati) | 1278 | 1395 | +172 | -55 | 0.84 | 1.00 | 0.0 | 3 |
| Liver Cancer (Roskams-Hieter) | 220 | 360 | +146 | -6 | 0.58 | 1.00 | 0.714 | 4 |
| Esophagus Cancer (Chen) | 708 | 804 | +135 | -39 | 0.79 | 1.00 | 0.408 | 6 |
| CAD_HF+ (Ward) | 986 | 1068 | +134 | -52 | 0.83 | 1.00 | 0.262 | 0 |
| CAD_HF- (Ward) | 960 | 1047 | +118 | -31 | 0.86 | 1.00 | 0.266 | 0 |
| Tuberculosis (Chang) | 571 | 660 | +117 | -28 | 0.79 | 1.00 | 0.661 | 2 |
| MM (Roskams-Hieter) | 320 | 411 | +117 | -26 | 0.67 | 1.00 | 0.526 | 4 |
| Liver Cancer (Chen) | 642 | 706 | +102 | -38 | 0.81 | 1.00 | 0.469 | 3 |
| Lung Cancer (Chen) | 557 | 618 | +101 | -40 | 0.79 | 1.00 | 0.422 | 0 |
| ME_CFS (Gardella) | 149 | 228 | +91 | -12 | 0.57 | 1.00 | 0.566 | 4 |
| ICI-m (Raissadati) | 509 | 543 | +85 | -51 | 0.77 | 1.00 | 0.516 | 2 |
| Pancreatic Cancer (Moore) | 577 | 607 | +79 | -49 | 0.80 | 1.00 | 0.478 | 1 |
| Pre-eclampsia (Moufarrej) | 214 | 275 | +77 | -16 | 0.68 | 1.00 | 0.142 | 0 |
| Colorectal Cancer (Chen) | 390 | 446 | +73 | -17 | 0.81 | 1.00 | 0.397 | 0 |
| Stomach Cancer (Chen) | 550 | 593 | +68 | -25 | 0.85 | 1.00 | 0.479 | 2 |
| HIV + Tuberculosis (Chang) | 407 | 448 | +64 | -23 | 0.82 | 1.00 | 0.594 | 0 |
| HIV (Chang) | 420 | 437 | +55 | -38 | 0.80 | 1.00 | 0.526 | 3 |
| MGUS (Roskams-Hieter) | 75 | 122 | +51 | -4 | 0.56 | 1.00 | 0.115 | 3 |
| Other Cancer (Moore) | 297 | 319 | +51 | -29 | 0.77 | 1.00 | 0.577 | 6 |
| Pancreatitis (Moore) | 268 | 298 | +49 | -19 | 0.79 | 1.00 | 0.396 | 1 |

## rare-분기 유전자가 직접 leading-edge에 든 신규 경로 (직접 근거 카탈로그)

| Phenotype | Term | NES | rare lead | DB 지지 유전자 |
|---|---|--:|---|---|
| Esophagus Cancer (Chen) | External Encapsulating Structure Organization (GO:0045229) | 2.07 | **MMP3** | NF1 |
| Esophagus Cancer (Chen) | Aldosterone-regulated sodium reabsorption | 2.07 | **INS** | PIK3R1 |
| Esophagus Cancer (Chen) | Positive Regulation Of Endothelial Cell Proliferation (GO:00 | 1.9 | **CCL26** | ARNT;KDR;AKT1 |
| Esophagus Cancer (Chen) | Anterograde Trans-Synaptic Signaling (GO:0098916) | 1.85 | **HTR3D** | GRIN2A |
| Esophagus Cancer (Chen) | Water Transport (GO:0006833) | 1.82 | **AVP** | — |
| Esophagus Cancer (Chen) | SARS-CoV-2-host Interactions R-HSA-9705683 | -1.61 | **IFNA1** | IKBKB |
| HIV (Chang) | Leukocyte transendothelial migration | 1.75 | **MYL10;CLDN17** | RAC1 |
| HIV (Chang) | Relaxin signaling pathway | 1.75 | **INSL5** | — |
| HIV (Chang) | Interleukin-6 Family Signaling R-HSA-6783589 | 1.73 | **IL31** | — |
| ICI-m (Raissadati) | Collecting duct acid secretion | 1.83 | **ATP6V1G3** | — |
| ICI-m (Raissadati) | Herpes simplex virus 1 infection | -1.75 | **IFNA1** | PIK3CD;C3;IL1B;CGAS;IRF7;IFNG |
| ICI-treated Cancer (Raissadati) | Detection Of Chemical Stimulus Involved In Sensory Perceptio | 1.82 | **TAS2R42** | — |
| ICI-treated Cancer (Raissadati) | Detection Of Chemical Stimulus Involved In Sensory Perceptio | 1.75 | **OR13F1;OR6K2;OR2K2** | — |
| ICI-treated Cancer (Raissadati) | Detection Of Chemical Stimulus Involved In Sensory Perceptio | 1.74 | **OR13F1;OR6K2;OR2K2** | — |
| Liver Cancer (Chen) | Pentose and glucuronate interconversions | 1.77 | **UGT1A8** | — |
| Liver Cancer (Chen) | Positive Regulation Of Defense Response (GO:0031349) | -1.72 | **CCL1** | CEBPA |
| Liver Cancer (Chen) | DNA Repair (GO:0006281) | -1.69 | **TNP1** | ATM;MSH6;ERCC5;POLD3;BRCA1;NBN |
| Liver Cancer (Roskams-Hieter) | Coronavirus disease | 1.75 | **RPL10L** | MAPK1;PIK3R1;IL6ST;MYD88;RPL22;EGFR |
| Liver Cancer (Roskams-Hieter) | Regulation Of MAP Kinase Activity (GO:0043405) | 1.73 | **DEFB114** | SOS1;FLT1;NF1;APOE;SH2B3;FGF1 |
| Liver Cancer (Roskams-Hieter) | Negative Regulation Of MAPK Cascade (GO:0043409) | 1.73 | **DEFB114** | NF1;APOE;SH2B3 |
| Liver Cancer (Roskams-Hieter) | Negative Regulation Of MAP Kinase Activity (GO:0043407) | 1.69 | **DEFB114** | NF1;APOE;SH2B3 |
| ME_CFS (Gardella) | Autophagy | -1.88 | **INS** | MAPK1;PRKAA1;MAPK3;ATG13;MAP2K1;INS |
| ME_CFS (Gardella) | Regulation Of Lipid Storage (GO:0010883) | -1.86 | **PLA2G10** | FBXW7;TNF;NFKB1 |
| ME_CFS (Gardella) | Positive Regulation Of Autophagy (GO:0010508) | -1.8 | **LACRT** | PRKAA1;IL4;ATG13 |
| ME_CFS (Gardella) | Positive Regulation Of Cell Differentiation (GO:0045597) | -1.79 | **PLA2G10;INS** | IGF1;MAPK14;TCF4;SMAD4;IL1B;ADRA2C |
| MGUS (Roskams-Hieter) | Transcriptional Regulation By RUNX1 R-HSA-8878171 | 1.88 | **H2BC1** | — |
| MGUS (Roskams-Hieter) | Mitotic Prophase R-HSA-68875 | 1.85 | **H2BC1** | — |
| MGUS (Roskams-Hieter) | Estrogen-dependent Gene Expression R-HSA-9018519 | 1.83 | **H2BC1** | ERBB4 |
| MM (Roskams-Hieter) | SRP-dependent Cotranslational Protein Targeting To Membrane  | -2.89 | **RPL10L** | — |
| MM (Roskams-Hieter) | Defensins R-HSA-1461973 | -1.88 | **DEFB113;DEFB127;DEFB125** | — |
| MM (Roskams-Hieter) | Systemic lupus erythematosus | 1.75 | **H4C7** | — |
| MM (Roskams-Hieter) | Neutrophil extracellular trap formation | 1.7 | **H4C7** | PIK3CB;MAPK1;PIK3CA;MTOR;RAF1 |
| Other Cancer (Moore) | Tight Junction Assembly (GO:0120192) | 1.95 | **CLDN25;CLDN17** | STRN |
| Other Cancer (Moore) | Positive Regulation Of Vascular Endothelial Growth Factor Pr | 1.93 | **NOX1** | BRCA1 |
| Other Cancer (Moore) | Bicellular Tight Junction Assembly (GO:0070830) | 1.91 | **CLDN25;CLDN17** | STRN |
| Other Cancer (Moore) | RHO GTPase Cycle R-HSA-9012999 | 1.9 | **NOX1** | ARHGAP35;PIK3R1 |
| Other Cancer (Moore) | Positive Regulation Of Translational Initiation (GO:0045948) | -1.82 | **DAZ2;DAZ4** | — |
| Other Cancer (Moore) | Negative Regulation Of Interleukin-1 Beta Production (GO:003 | -1.77 | **PYDC2** | — |
| Pancreatic Cancer (Moore) | Positive Regulation Of Pathway-Restricted SMAD Protein Phosp | 1.78 | **MSTN** | TGFBR2;PPARG;BMPR1A |
| Pancreatitis (Moore) | Insulin Receptor Recycling R-HSA-77387 | -1.71 | **ATP6V1G3** | — |
| Stomach Cancer (Chen) | Formation Of Cornified Envelope R-HSA-6809371 | 2.37 | **SPINK6** | — |
| Stomach Cancer (Chen) | Body Fluid Secretion (GO:0007589) | 2.15 | **CSN3** | — |
| Tuberculosis (Chang) | Epithelium Development (GO:0060429) | 1.67 | **KRT28** | PGK1;TGFB1;WT1;TST |
| Tuberculosis (Chang) | TCF Dependent Signaling In Response To WNT R-HSA-201681 | 1.65 | **DKK4** | UBB;UBC;UBA52 |

---

## 질병별 상세 (skill 문헌·DB 검증)

기지(확립)=선행 문헌·DB 다수. 후보(novel)=skill 검색으로 문헌 부재 확인, 검증 대상. 각 유전자 근거 원본은 `GSEA/Reference/<phenotype>_refs.md`.

### ICI 치료 암 코호트 (ICI-treated Cancer (Raissadati), 신규 +172 · DB지지 novel 0 · rare-led 3)

이 표현형은 단일 질병이 아니라 여러 암종을 ICI로 치료한 **이질적 혼합 코호트**로, Open Targets 참조 질병(ot_disease=None)이 없어 DB 교차검증이 불가능하다. 따라서 본 섹션은 문헌 기반으로만, 그리고 보수적으로 기술한다. 신호는 번역/리보솜/rRNA 처리의 광범위한 하향과 리소좀 산성화·항원제시·T세포 세포독성/이식편거부(면역 활성) 축의 상향으로 수렴하며, only_nbi와 거의 일치한다(jaccard 0.843, sign_agree 1.0). rare 분기의 3개 novel은 모두 화학감각 수용체 유전자군으로 신호로 보기 어렵다.

- **번역/리보솜·rRNA 처리 하향 (Eukaryotic Translation Elongation NES -2.9; Cytoplasmic Translation; rRNA Processing)** — rare 비의존, 강하고 광범위. 혼합 암 코호트 전반의 공통 신호이나 세포기원·질병특이성을 단정할 수 없어 **정황적(reserved)** 으로만 해석. 단일 OT 질병 참조가 없어 DB 검증 불가.
- **면역 활성 축 (KEGG Allograft rejection NES +2.42; Immunoregulatory Interactions; Positive Regulation Of T Cell Mediated Cytotoxicity; Antigen processing and presentation)** — ICI 작용기전(T세포 활성화)과 방향적으로 정합하나, 혼합 코호트라 특정 질병에 귀속 불가. 문헌상 ICI 반응 모니터링에 cfRNA/면역 전사체가 탐색되고 있음(immune checkpoint inhibitor AND cell-free RNA AND cancer ≈ 656건, 대표 PMID 42358824). **개연성 있으나 검증 필요한 정황 신호**.
- **OR* 후각수용체 (GO Detection Of Chemical Stimulus Involved In Sensory Perception Of Smell; NES +1.75, rare lead OR13F1;OR6K2;OR2K2;OR10G2;OR10H3;OR2S2)** — 후각수용체 유전자 클러스터. 혈장/혈액 cfRNA에서 후각수용체는 거의 발현되지 않으며(olfactory receptor AND cell-free RNA ≈ 7건, 대부분 후각상피 맥락, 대표 PMID 40020072), 암/ICI 특이 문헌 미보고 → **거의 확실한 극저발현 artifact**. 생물학적 해석 금지, 후보로도 약함.
- **TAS2R42 미각수용체 (GO Detection Of Chemical Stimulus ... Taste; NES +1.82, rare lead TAS2R42)** — 쓴맛 수용체. OR과 동일하게 혈액 극저발현 화학감각 유전자군으로 **artifact 후보**. taste receptor AND leukocyte expression ≈ 39건이 존재하나 ICI/암 특이 신호로 볼 근거 없음.

**해석:** 혼합 암 코호트라는 설계상 이 표현형은 질병특이 해석의 대상이 아니며, OT 참조 부재로 DB 교차검증도 불가능하다. 견고하게 반복되는 축은 번역기구의 광범위한 하향과 T세포/항원제시 중심 면역 활성 상향으로, ICI 기전과 방향적으로만 정합할 뿐 특정 질병에 귀속할 수 없다(reserved). rare-led 3건(후각수용체 OR* 및 미각수용체 TAS2R42)은 혈액에서 발현되지 않는 화학감각 유전자 클러스터로, 극저발현 잔존분산에 의한 **artifact로 강하게 판단**하며 생물학적 신호로 해석하지 않는다.

### 간세포암 (Liver Cancer (Roskams-Hieter), 신규 +146 · DB지지 novel 다수 · rare-led 4)

이 코호트는 only_nbi 대비 jaccard 0.585로 rare 분기가 가장 크게 지형을 바꾼 케이스(novel 146). 다만 novel 다수는 신규 생물학이 아니라 **HCC의 확립 축이 결합 랭킹에서 표면화한 것**이다: VEGF/혈관신생, adherens junction(CTNNB1/MET), RTK-MAPK 조절, 응고/섬유소용해, RHO/RAC GTPase cycle, oncogene-induced senescence(TP53/RB1/CDKN2A). 즉 정규화 접근이 알려진 HCC 신호를 복원했다는 긍정적 positive control. 반면 leading edge를 직접 이끈 rare 유전자(RPL10L, DEFB114)는 조직 특이 저발현으로 artifact 경고가 필요하다.

- **RPL10L (KEGG Coronavirus disease; NES 1.75)** — rare lead RPL10L, DB 지지 PIK3CA(0.68)·EGFR(0.44)·MAPK1(0.33) 등. 이 term의 db_support는 번역/신호 정준 유전자지만, rare lead RPL10L은 정소 특이 리보솜 단백 L10 paralog로 혈장 극저발현(PubMed RPL10L AND carcinoma ≈ 1건). **저발현 paralog artifact 가능성 — novel 후보로도 보수적.**
- **DEFB114 (Negative/Regulation Of MAP Kinase Activity; NES 1.69–1.73)** — rare lead DEFB114, DB 지지 NF1(0.47)·EGFR(0.44)·FLT1(0.61)·APOE(0.39)·SH2B3(0.37). MAPK/RTK 조절 및 혈관신생 축은 HCC에서 확립되어 db_support가 강하나, rare lead DEFB114(β-defensin 114)는 질병 문헌 전무(PubMed DEFB114 AND carcinoma = 0건). **term 생물학은 기지, rare 귀속분은 미보고 artifact 후보.**
- **Signaling By VEGF R-HSA-194138 (NES 1.90)** — DB 지지 CTNNB1(0.80)·FLT4(0.70)·KDR(0.68)·PIK3CA(0.68)·FLT1(0.61). 비-rare, 확립된 HCC 혈관신생 축.
- **KEGG Adherens junction (NES 2.10)** — DB 지지 CTNNB1(0.80)·MET(0.78)·EGFR(0.44). CTNNB1은 Open Targets HCC 연관 0.78(literature 0.99·somatic_mutation 0.86·genetic_association 0.85)로 HCC 정준 driver(PMID 38123979); MET도 HCC 표적치료 표적(PMID 40394703). **기지 확립 축.**

**해석:** rare/normative 결합이 이 코호트에서 복원한 것은 대부분 확립된 HCC 축(Wnt/β-catenin adherens junction, VEGF 혈관신생, RTK-MAPK, 세포노화)으로, 방법론의 민감도를 뒷받침한다. 순수 신규 기여로 볼 rare lead(RPL10L, DEFB114)는 조직 특이 저발현 유전자라 생물학적 의미보다 극저발현 잔존 분산 가능성을 우선 경고하며 검증 대상으로 둔다.

### 식도암 (Esophagus Cancer (Chen), 신규 +135 · DB지지 novel 다수 · rare-led 6)

식도암 코호트의 확립 축은 강건하다: FGFR 신호(FGFR1/2/3·PTPN11·PIK3R1)와 cAMP·YAP1/WWTR1 상향, 번역/리보솜·산화적 인산화 하향. rare 분기가 표면화한 novel 6건은 ECM 리모델링·혈관신생·시냅스/수분수송·인터페론으로 흩어지며, **6개 rare-led 중 4개(INS, HTR3D, AVP, 부분적으로 CCL26)가 조직 특이 극저발현 유전자라 이 코호트는 artifact 경고가 특히 크다.** MMP3만이 문헌 지지가 있는 상대적으로 견고한 후보다.

- **MMP3 (External Encapsulating Structure Organization GO:0045229; NES 2.07)** — rare lead MMP3, DB 지지 NF1(0.47). MMP3는 ECM 분해 matrix metalloproteinase로 식도암 문헌 존재(PubMed MMP3 AND esophageal cancer ≈ 36건; ESCC scRNA/spatial PMID 39741182). **ECM 리모델링은 침습 축으로 생물학적 타당성 있는 novel 후보 — rare-led 중 가장 신뢰.**
- **INS (Aldosterone-regulated sodium reabsorption; NES 2.07)** — rare lead INS. INS(인슐린)는 췌장 β세포 특이 유전자로 식도 조직에서 정상 무발현. "Regulation Of Gene Expression In Beta Cells"·"Signal Attenuation"도 함께 INS 주도. **명백한 이소성/극저발현 artifact — 배제 권고.**
- **CCL26 (Positive Regulation Of Endothelial Cell Proliferation GO:0001938; NES 1.9)** — rare lead CCL26, DB 지지 KDR(0.49)·AKT1(0.40)·ARNT(0.38). CCL26(eotaxin-3)은 2형 염증 매개체(PubMed CCL26 AND cancer ≈ 96건; dupilumab 2형 염증 PMID 34037993). 혈관신생 term은 KDR로 DB 지지되나 CCL26의 암 driver 근거는 희박 — **호산구성 식도 염증 맥락의 보수적 후보.**
- **HTR3D (Anterograde Trans-Synaptic Signaling; NES 1.85)** — rare lead HTR3D(세로토닌 수용체 아단위), 신경 특이 저발현(PubMed HTR3D AND cancer ≈ 4건). **artifact 경고.**
- **AVP (Water Transport GO:0006833; NES 1.82)** — rare lead AVP(아르기닌 바소프레신), 시상하부 특이(PubMed AVP AND cancer ≈ 465건이나 대부분 SIADH/이소성 분비). **조직 제한성 극저발현 artifact 경고.**
- **FGFR2 Downstream Signaling R-HSA-5654696 (NES 2.35, 비-novel)** — DB 지지 FGFR2(0.51)·PTPN11(0.45)·PIK3R1(0.39). FGFR2는 식도암 확립 표적(PubMed FGFR2 AND esophageal cancer ≈ 81건; FGFR 억제제 futibatinib PMID 36441501; Open Targets TGFBR2·FGFR2 계열 강한 literature/genetic). **기지 확립 축.**

**해석:** normative 접근이 식도암에서 안정적으로 복원한 것은 FGFR-PI3K·YAP 축(비-novel, 강건)이며, rare 분기의 순 기여 중에는 MMP3 매개 ECM 리모델링만이 문헌 지지되는 후보다. INS·HTR3D·AVP는 이소성/조직특이 극저발현 유전자로 생물학적 해석 대상이 아니라 잔존 분산 artifact로 명시하고 배제 권고한다. CCL26은 2형 염증 맥락의 약한 후보로 남긴다.

### 관상동맥질환·심부전 진행군 (CAD_HF+ (Ward), 신규 +134 · DB지지 novel 3 · rare-led 0)

with_rare에서 134개 신규 term이 표면화했으나 rare-led=0으로, rare 분기(HC 검출률 <1% 극저발현 유전자 559개)가 직접 leading edge에 들어간 신호는 없다. 즉 rare 효과는 순위의 간접적 이동일 뿐 기전적 귀속 근거는 약하다. 그럼에도 신규·상위 신호는 상향 축이 FGF 성장인자 신호(FGF5)와 콜라겐/ECM 생합성(COL6A3)이라는 두 확립된 CAD/심부전 혈관 리모델링·섬유화 축으로 수렴한다. 하향은 리보솜 번역·산화적 인산화 프로그램의 전반적 억제로, HF- 군과 공유되는 비특이 배경 축이다.

- **FGF5 (FGFR2 Ligand Binding And Activation R-HSA-190241; NES 2.21, novel; FGFRL1 Modulation Of FGFR1 R-HSA-5658623; NES 2.15, novel)** — DB 지지 FGF5. Open Targets coronary artery disorder 연관 score 0.548(추가로 hypertensive disorder 0.584, heart failure 0.473, myocardial infarction 0.469 검증). FGF5는 혈압·관상동맥 GWAS 유전자로 문헌 다수(PubMed FGF5 AND 관상동맥/혈압/고혈압 수십 건 규모, 대표 PMID 42330250). 다만 FGF5 AND heart failure 직접 문헌은 희소(≤10건, PMID 40290076)로, 심부전 진행 특이 축으로서는 확립 아닌 후보 수준. FGF 신호 자체는 기지의 CAD 혈관 리모델링 축.
- **COL6A3 (Collagen Biosynthesis And Modifying Enzymes R-HSA-1650814; NES 2.04, novel; Collagen Chain Trimerization R-HSA-8948216; NES 2.40, 기지)** — DB 지지 COL6A3, OT coronary artery disorder 0.445·myocardial infarction 0.339 검증. COL6A3/심장·섬유화 문헌 존재(PubMed 수십 건, 대표 PMID 42049434). ECM/콜라겐 상향은 심부전 진행에서 예상되는 심근·혈관 섬유화 축과 정합적이며, HF- 군에는 상위에 없는 HF+ 상대적 특징.
- **Aminoglycan Biosynthetic Process (GO:0006023; NES 2.18, novel)** — GAG/프로테오글리칸 합성. 혈관 ECM 리모델링 축과 생물학적으로 연결되나 leading-edge DB 지지 없음(n_db=0). 보수적 novel 후보.
- **ADRB2 (Amine Ligand-Binding Receptors R-HSA-375280; NES 2.33, 기지)** — DB 지지 ADRB2. OT에서 ADRB2는 coronary artery disorder 0.278로 중간이나 heart failure 0.604·congestive heart failure 0.599로 강함(패킷 0.51은 상이 버전/질병 추정, 실검증치 인용). ADRB2 AND heart failure 문헌 존재(≤수십 건, PMID 42359705). β2-아드레날린 축은 심부전에서 확립된 신호.
- **Sensory Perception Of Pain (GO:0019233), Interaction Between L1 And Ankyrins (R-HSA-445095), Semaphorin-Plexin (PLXND1)** — 신경/축삭유도 계열 novel. cfRNA에서 이 계열은 극저발현·GPCR/막수용체 gene-family 잔존 분산으로 뜨기 쉬워 artifact 경계 필요. PLXND1은 OT coronary artery disorder 0.366(주로 선천성 심결손 연관)로 직접 CAD 근거 약함.

**해석:** HF+ 군에서 normative/rare 접근이 복원한 실질 신호는 FGF5(혈관/혈압 리모델링)와 COL6A3(콜라겐·ECM 섬유화)로, 심부전 진행에 부합하는 성장인자+섬유화 리모델링 축이다. FGF5의 심부전 특이성과 aminoglycan 축은 문헌·DB 지지가 약해 검증 대상 후보로 남긴다. 신경/통각/semaphorin 계열과 rare 분기 유래 순위 이동은 극저발현 잔존 분산 가능성이 커 생물학적 해석을 자제하고 artifact 후보로 명시한다. rare_led=0인 만큼 이 질병에서 rare 분기의 직접 기전 기여 주장은 하지 않는다.

### 관상동맥질환·심부전 비진행군 (CAD_HF- (Ward), 신규 +118 · DB지지 novel 1 · rare-led 0)

with_rare에서 118개 신규 term이 나왔으나 rare-led=0으로, rare 분기 유전자가 leading edge에 직접 들어간 신호는 없고 rare 효과는 간접적 순위 이동에 그친다. HF+ 군이 FGF/콜라겐(성장인자·섬유화 리모델링) 축으로 수렴한 것과 대조적으로, HF- 군의 최상위 상향 신호는 AGT가 이끄는 GPCR/cyclic-nucleotide 2차전령 신호와 P2RY12 혈소판 퓨린성 수용체 축으로, 혈관 긴장도(RAAS)·혈전 반응 축이 우세하다. ECM/콜라겐 축은 상위에 없어 섬유화보다 혈관 반응성·혈소판 축이 HF-를 특징짓는다. 하향은 HF+ 와 공유되는 리보솜 번역·mRNA 스플라이싱 억제 배경.

- **AGT (G Protein-Coupled Receptor Signaling, Cyclic Nucleotide Second Messenger GO:0007187; NES 3.76, novel, 전체 최강)** — DB 지지 AGT. OT coronary artery disorder 0.387, essential hypertension 0.627·hypertensive disorder 0.606로 검증. 앤지오텐시노겐(RAAS 핵심)은 CAD/심부전·고혈압 문헌 방대(PubMed AGT AND 심부전/관상동맥 수백 건, 대표 PMID 42294767). GPCR 2차전령 축이 novel로 최강 표면화한 것은 RAAS/혈관 긴장 신호 복원으로 해석되는 기지 축.
- **P2RY12 (P2Y Receptors R-HSA-417957; NES 2.12; Nucleotide-like (Purinergic) Receptors R-HSA-418038; NES 2.14, 둘 다 기지)** — DB 지지 P2RY12. OT coronary artery disorder 0.602(추가 myocardial infarction 0.62, acute coronary syndrome 0.613 검증). P2RY12는 clopidogrel 표적으로 CAD 항혈소판 문헌 방대(PubMed 1000건 규모, clopidogrel 연관 수백 건, 대표 PMID 42177012). 혈소판 활성/혈전 축으로 CAD에 직접적으로 확립. novel은 아니나 HF- 상위 축의 정합 근거.
- **PLCG2 (Coronavirus disease KEGG; NES -1.84, 하향)** — DB 지지 PLCG2(OT 0.41). 방향은 하향이며 바이러스 term 문맥의 면역 신호전달로, CAD 특이 해석은 보류.
- **Nicotine addiction (KEGG; NES 2.41, novel), Regulation Of Dopamine Secretion (GO:0014059; NES 2.03, novel), Acetylcholine Binding And Downstream Events (R-HSA-181431; NES 2.03, novel)** — 신경전달/니코틴성 수용체 계열 novel. 흡연이 CAD 위험인자라는 점에서 표면적 연관은 상상 가능하나, leading-edge DB 지지 없음(n_db=0)이고 cfRNA에서 신경수용체 gene-family는 극저발현 잔존 분산으로 뜨기 쉬워 artifact 경계가 필요. 보수적 novel 후보로만 표기.

**해석:** HF- 군에서 rare/normative 접근이 복원한 실질 축은 AGT-매개 GPCR/2차전령(RAAS·혈관 긴장)과 P2RY12 혈소판 퓨린성 수용체(혈전)로, 모두 OT·문헌 다수의 확립된 CAD 혈관 반응성·혈전 축이다. HF+ 대비 콜라겐/ECM 섬유화 축이 상위에 없고 혈관 긴장·혈소판 축이 지배적인 점이 두 아형의 핵심 대조다. 니코틴/도파민/아세틸콜린 등 신경전달 계열 novel과 rare 분기 유래 순위 이동은 극저발현 특유 잔존 분산 가능성이 커 생물학적 결론을 자제하고 검증 대상 artifact 후보로 명시한다. rare_led=0이므로 rare 분기의 직접 기전 기여는 주장하지 않는다.

### 결핵 (Tuberculosis (Chang), 신규 +117 · DB지지 novel 0 · rare-led 2)

with_rare 상위 신호는 호중구/식세포 축(neutrophil extracellular trap, Fcγ receptor phagocytosis, RHO GTPase effectors), IFN-γ/α·β signaling, 그리고 결핵 특이 Reactome term(Infection with Mycobacterium tuberculosis, Response of Mtb to Phagocytosis)에 수렴하며 이들은 전부 only_nbi에서 이미 유의(novel=False)한 확립 신호다. rare 분기가 표면화한 novel term은 leading edge db_support가 없는 DB지지 novel 0건이며, rare-led 2건(epithelium development의 KRT28, WNT/TCF의 DKK4) 모두 결핵 문헌이 전무해 극저발현 artifact 우선 경계 대상이다. normative 접근은 결핵의 자명한 골수성 항균 축을 강하게 복원했다.

- **LTF/CORO1A/CTSG/RAB7A (Infection With Mycobacterium Tuberculosis R-HSA-9635486; NES 2.15, n_db 14; Response Of Mtb To Phagocytosis R-HSA-9637690; NES 2.17)** — novel 아님. leading edge db_support LTF(0.52), KPNB1(0.50), CORO1A(0.38), CTSG(0.37), RAB7A(0.46). 모두 결핵 확립 축: LTF(lactoferrin) 항균·진단 바이오마커 문헌 다수("lactoferrin AND tuberculosis" ~77건; 대표 PMID 26788020, 28642848), Open Targets tuberculosis 연관 0.517로 교차확인. CORO1A(coronin-1/TACO)는 Mtb의 식세포내 생존을 매개하는 고전 기전(~5건; 대표 PMID 22256790). → 기지 축.
- **GBP1/GBP5/HLA-B/SOCS3 (Interferon Gamma Signaling R-HSA-877300; NES 2.18)** — leading edge GBP5(0.10), GBP1(0.08), HLA-B(0.10), SOCS3(0.10). GBP(guanylate-binding proteins)는 IFN-γ 유도 항마이코박테리아 이펙터·혈액 시그니처 구성유전자로 확립("GBP1/GBP5 AND tuberculosis" 각 ~25/31건; 대표 PMID 36769182, 35753598), Open Targets GBP5-tuberculosis 0.088 교차확인. → 기지 IFN-γ 축.
- **S100A9 (Neutrophil extracellular trap formation; NES 2.53; RHO GTPase Effectors R-HSA-195258)** — leading edge MAPK3/1, CTSG, S100A9(0.09), FCGR3A/FCGR1A. S100A9(calprotectin)와 호중구 NET은 활동성 결핵의 대표적 혈액 시그니처(~27건; 대표 PMID 35935235, 34849408). → 기지 골수성/호중구 축.
- **KRT28 (Epithelium Development GO:0060429; NES 1.67)** — rare lead KRT28; db_support PGK1(0.37), WT1(0.25), TGFB1(0.11), TST(0.11). novel·rare-led. KRT28("KRT28 AND tuberculosis" 0건)은 결핵 문헌 전무. term의 실질 앵커는 당분해/TGFB1 등이며 KRT28은 극저발현 tie-breaker로 추정 → artifact 경고, 미보고 후보.
- **DKK4 (TCF Dependent Signaling In Response To WNT R-HSA-201681; NES 1.65)** — rare lead DKK4; db_support UBC/UBB/UBA52(유비퀴틴). novel·rare-led. DKK4("DKK4 AND tuberculosis" 0건) 결핵 문헌 전무. leading edge가 유비퀴틴 하우스키핑 유전자로 구성되어 있어 WNT 신호로 해석하기 어려움 → artifact 후보.

**해석:** normative/rare 접근이 복원한 실질 축은 IFN-γ·GBP 항균 이펙터, 식세포/NET 호중구 축, Mtb 감염 특이 Reactome(LTF, CORO1A, RAB7A, CTSG)으로 전부 결핵 확립 생물학이며 Open Targets(LTF 0.52, GBP5 0.09)·문헌으로 뒷받침된다. rare-led novel(KRT28, DKK4)은 결핵 문헌이 전무하고 leading edge 실질 앵커가 하우스키핑/대사 유전자여서 극저발현 잔존 분산 artifact 후보로 보수적 분류한다. 하향 신호는 spliceosome/mRNA processing(SFPQ) 축.

### 다발성 골수종 (MM (Roskams-Hieter), 신규 +117 · DB지지 novel 2 · rare-led 4)

with_rare 신규 신호의 다수는 이미 only_nbi에서도 잡히던 두 축(번역/리보솜·산화적 인산화의 강한 하향, 세포주기·유사분열의 상향)의 연장선이며, rare 분기는 이 골격을 크게 바꾸지 않았다(jaccard 0.673, sign_agree 1.0). rare-led 4건은 방향이 엇갈리고 leading edge가 극저발현 유전자군(RPL10L, β-defensin 클러스터, 히스톤 클러스터)에 집중되어, 생물학적 신호와 극저발현 잔존분산을 함께 의심해야 한다. DB지지가 뚜렷한 축은 rare가 아니라 세포주기(CCND1 등)로, 이는 MM의 확립된 종양생물학과 정합한다.

- **CCND1 (KEGG Cell cycle; NES +2.12)** — DB 지지 CCND1(0.53);RB1(0.44);MDM2(0.33);CDK6(0.32) 등 9개. Open Targets에서 CCND1–plasma cell myeloma 연관 확인(score 0.534; 유전/문헌 evidence). PubMed CCND1 AND multiple myeloma 수백 건(약 236건, 대표 PMID 42261309). t(11;14) CCND1 전위는 MM의 교과서적 기전으로 **기지(확립) 축**이며 normative Z에서도 재현됨.
- **RPL10L (Eukaryotic Translation Elongation; NES -3.39, rare lead)** — 번역/리보솜 경로군의 강한 하향을 rare 분기 RPL10L이 leading edge에서 이끎. 그러나 PubMed RPL10L AND multiple myeloma = 0건, RPL10L AND cancer도 ~6건으로 대부분 생식/DB 문헌(대표 PMID 39380204). RPL10L은 정소 제한 발현 리보솜 파라로그(retrogene)로, 혈장 cfRNA에서 극저발현이라 **후보(artifact 우려 큰) 신호**로 분류. 다만 리보솜/번역 경로 전체의 하향 자체는 only_nbi에서도 유의(rare 비의존)해 신뢰 가능.
- **β-defensin 클러스터 (Reactome Defensins; NES -1.88, rare lead DEFB113/127/125/106A/4A/116/126)** — PubMed defensin AND multiple myeloma = 2건, 기전적 연관 아님(대표 PMID 33420397, 16285021은 비특이적). β-defensin 유전자군은 저발현 상피성 클러스터로 혈액 cfRNA에서 신뢰 낮음 → **보수적 novel/artifact 후보**.
- **H4C7 히스톤 클러스터 (KEGG SLE / Neutrophil extracellular trap formation; NES +1.75/+1.7, rare lead H4C7)** — NET 경로에는 PIK3CA/MTOR/RAF1 등 DB 지지(0.30~0.31)가 있으나 rare lead 자체는 히스톤 유전자. 형질세포 크로마틴 활성 측면에서 생물학적으로 그럴듯하나(히스톤 다수 발현), H4C7 특이 문헌은 희박(H4C7/HIST1H4G AND cancer ≈ 3건, 비특이적). SLE/NET term은 히스톤·호중구 유전자 공유로 인한 경로 오귀속 가능성 → **후보로 유보**.

**해석:** MM에서 normative/rare 접근이 복원한 견고한 축은 (1) 리보솜·번역과 산화적 인산화의 하향, (2) 세포주기·유사분열 상향이며, 후자는 CCND1 등으로 OT·문헌 모두에서 강하게 지지되는 기지 축이다. rare 분기가 표면화한 4건(RPL10L, β-defensin, 히스톤 H4C7)은 방향이 일관되지 않고 모두 혈장 극저발현 유전자에 귀속되어, RPL10L/DEFB는 artifact 가능성이 높은 novel 후보로, 히스톤 축은 형질세포 크로마틴이라는 생물학적 개연성은 있으나 문헌 미보고 후보로 명시한다. 검증 없이 rare-led term을 MM 특이 신호로 해석하지 말 것.

### 간세포암 (Liver Cancer (Chen), 신규 +102 · DB지지 novel 2 · rare-led 3)

이 코호트의 주 신호는 novel이 아닌 확립 축이다: 리보솜·번역(RPL22) 및 산화적 인산화의 강한 하향(NES -3.3 ~ -2.5)으로, HCC의 세포유리 RNA에서 반복 관찰되는 전신 번역기구 억제 패턴에 부합한다. rare 분기가 표면화한 novel 3건은 대사(글루쿠론산화)와 면역/DNA repair 축으로 흩어져 있으며, leading edge를 직접 이끄는 rare 유전자는 조직 특이 저발현 유전자여서 신호보다 잔존 분산 가능성을 함께 봐야 한다.

- **UGT1A8 (Pentose and glucuronate interconversions; NES 1.77)** — rare lead UGT1A8, DB 지지 없음. UGT1A8은 간 글루쿠론산화의 정준 효소지만 질병 연관은 미보고 수준(PubMed UGT1A8 AND hepatocellular carcinoma ≈ 5건; PMID 34147074는 stemness signature 일부 언급). Open Targets에서 UGT1A8은 HCC 연관 근거가 없고 "Abnormality of the liver" 0.08(genetic_association)뿐. **간 조직 유래 대사효소 신호일 수 있으나 질병 특이 driver로는 보수적 novel 후보.**
- **CCL1 (Positive Regulation Of Defense Response; NES -1.72)** — rare lead CCL1, DB 지지 CEBPA(0.33). 면역 케모카인이나 HCC 직접 문헌은 희소(PubMed CCL1 AND HCC ≈ 8건). 방어반응 term 자체는 저발현 케모카인 주도로, **미보고 후보 신호.**
- **TNP1 (DNA Repair; NES -1.69)** — rare lead TNP1, DB 지지 RRM2B(0.42)·ATM(0.36)·MSH2/6(0.35)·BRCA1(0.35) 등 정준 repair 유전자 다수. 그러나 rare lead인 TNP1은 정자세포 특이 transition protein 1로 혈장에서 극저발현(PubMed TNP1 AND carcinoma ≈ 4건, 전부 정자형성 맥락). **DNA repair 축은 db_support가 정준 유전자로 강하나, rare가 귀속한 부분(TNP1)은 극저발현 artifact 가능성이 높음 — 검증 필요.**

**해석:** normative/rare 접근이 HCC에서 확립한 핵심은 번역기구·OXPHOS 하향(비-novel, 강건)이다. rare 분기가 추가한 novel 축(UGT1A8 글루쿠론산화, CCL1 면역)은 문헌·DB 근거가 희박한 후보로 남기며, TNP1이 이끈 DNA repair는 정준 repair 유전자로 term은 지지되나 rare lead 자체는 정자 특이 저발현 artifact로 경고한다. UGT1A8만이 간 조직 기전상 후속 검증 가치가 있는 보수적 후보다.

### 폐암 (Lung Cancer (Chen), 신규 +101 · DB지지 novel 0 · rare-led 0)

폐암은 rare 분기 귀속 신호가 전혀 없다(rare-led 0). with_rare에서 신규 101건이 늘었으나 표시 상위 novel 경로에는 DB 지지가 붙지 않아, rare 분기가 폐암 특이 기전을 표면화했다는 직접 증거는 없다(jaccard 0.786, 부호일치 1.0). 대신 상위 유의 골격은 rare 미포함에서도 확립된 성장인자 수용체 신호(EGFR/RET/GAB1)의 강한 상향과 세포질/미토콘드리아 번역·리보솜·OXPHOS의 하향으로, 신호의 핵심은 count route(NBI)에 있고 rare 기여는 부수적이다.

- **EGFR/SRC/PTPN11 (Epidermal Growth Factor Receptor Signaling GO:0007173; NES 2.43)** — DB 지지 EGFR(0.89)·SRC(0.61)·PTPN11(0.49). EGFR-폐암은 최상위 확립 축으로 문헌 규모가 방대(EGFR+lung cancer 3만 건 이상, 대표 PMID 31562956). normative 상향 방향이 폐선암 드라이버 생물학과 정합하는 기지 축.
- **SRC/PIK3R1/PTPN11 (RET Signaling R-HSA-8853659 · GAB1 Signalosome R-HSA-180292; NES 2.66/2.64)** — DB 지지 SRC(0.61)·PIK3R1(0.54)·PTPN11(0.49). RET 융합은 폐선암의 확립된 드라이버(RET+lung cancer 약 1700건, 대표 PMID 40136350, 32846060), GAB1 신호소체는 RTK 하류 어댑터 축(대표 PMID 37627207). 모두 기지 성장신호 축.
- **JAK1/PIK3R1 (Interleukin-7 Signaling R-HSA-1266695; NES 2.44)** — DB 지지 JAK1(0.55)·PIK3R1(0.54). 면역/사이토카인-JAK 축의 상향으로 종양미세환경·면역세포 조성과 정합 가능. 경로 수준은 기지, cfRNA 세포조성 기여 가능성 병기.
- **NPM1 (Ribosome Biogenesis GO:0042254 / Ribonucleoprotein Complex Biogenesis; NES -2.36/-2.37)** — DB 지지 NPM1(0.70). NPM1은 리보솜 생합성 핵심 조절자로 DB 점수 높음. 다만 NPM1은 혈액암(AML) 문헌이 압도적이고 NPM1+lung cancer 직접 문헌은 소규모(약 4건)로, 폐암 특이 개별 연관은 후보 수준. 리보솜/번역 하향 자체는 다수 표현형 공통 축(비특이) 경고.
- **CACNA1D (Membrane Depolarization During Action Potential GO:0086010; NES 2.34)** — DB 지지 CACNA1D(0.53). 신경내분비/이온채널 축으로 소세포·신경내분비 폐암 맥락에서 관심 대상이나 개별 검증 필요한 후보.

**해석:** 폐암 신호는 rare 분기가 아니라 count route의 확립된 RTK 축(EGFR/RET/GAB1/JAK)에서 나오며, 이는 기지 드라이버 생물학과 강하게 정합한다. rare-led·DB지지 novel이 0이므로 이 표현형에서 rare 분기의 추가 해석 가치는 낮다. NPM1(폐암 직접 문헌 희소)·CACNA1D는 후보로 명시하고, 광범위한 번역/리보솜/OXPHOS 하향은 표현형 비특이 공통 축으로 취급.

### 근육통성 뇌척수염/만성피로증후군 (ME_CFS (Gardella), 신규 +91 · DB지지 novel 4 · rare-led 4)

with_rare가 표면화한 91개 신규 term은 예외 없이 NES<0(하향)이며, (1) 세포질 번역·리보솜(비신규 최강축, NES -2.6~-2.2), (2) 지질저장·자가포식·에너지대사(rare-led), (3) 성장호르몬/IGF·에스트로겐·NMDA 신경신호로 수렴한다. 즉 rare 분기는 노이즈가 아니라 ME/CFS의 확립된 대사·에너지 저하 축을 강화하는 방향으로 작동했고, 부호 일치율 1.0으로 only_nbi와 모순이 없다. 다만 rare-led 유전자 중 일부(LACRT 등)는 극저발현 분비단백으로 생물학적 해석보다 잔존분산 아티팩트 가능성을 병기해야 한다.

- **INS (KEGG Autophagy; NES -1.88 / Positive Regulation Of Cell Differentiation; NES -1.79)** — rare lead INS, DB 지지 PRKAA1(0.06)·MAPK1/3·ATG13. Open Targets는 INS–ME/CFS를 europepmc 문헌마이닝 근거 36건(최고 score 0.33)으로 등재해 직접 유전자-질병 연관보다는 co-mention 수준이다. 인슐린/에너지·포도당 대사 이상 자체는 ME/CFS에서 확립된 축으로, metabolomics/lipidomics 문헌 PubMed 약 120건(대표: Nat Med 2025 AI 멀티오믹스, PMID 40715814)과 정합. rare 분기가 이 대사·자가포식 저하축을 count 분기와 독립적으로 재현한 것은 신호로 해석 가능(단 INS의 인과적 역할은 미확립).
- **PLA2G10 (Regulation Of Lipid Storage GO:0010883; NES -1.86 / Positive Regulation Of Cell Differentiation)** — rare lead PLA2G10, term DB 지지 FBXW7(0.30)·TNF(0.08). **PLA2G10 자체는 ME/CFS 문헌 미보고**(PubMed PLA2G10 AND ME/CFS = 0건; Open Targets ME/CFS 근거 0건). 유전자 총 문헌은 ~97건으로 분비형 phospholipase A2 group X의 면역·지질염증 기능(대표 PMID 38669316, Sci Immunol 2024)에 국한. 지질저장/에이코사노이드 축과 기전적으로 정합하나 **직접 근거 없는 신규 후보**로, 검증 대상.
- **LACRT (Positive Regulation Of Autophagy GO:0010508; NES -1.8)** — rare lead LACRT, DB 지지 PRKAA1·ATG13. LACRT–ME/CFS 문헌 0건(유전자 총 ~36건, 대부분 눈물샘/안구건조 lacrimal 단백). 혈중 극저발현 분비단백이 rare 분기 leading edge에 오른 경우로 **자가포식 신호로 확대해석 금지**; 극저발현 특유 잔존분산 아티팩트 우선 의심.

**해석:** rare/normative 접근이 복원한 축은 ME/CFS의 기지(확립) 신호인 **번역·리보솜 억제 + 인슐린/에너지대사·자가포식 저하**로, count 분기 단독(only_nbi)과 부호가 완전 일치(1.0)해 방향성 신뢰도가 높다. INS는 문헌마이닝 수준 근거로 대사축의 마커로 정합하나 인과성은 미확립. **PLA2G10은 문헌·DB 모두 근거가 없는 진성 신규 후보**로 ME/CFS 지질염증 가설의 검증 표적으로 제안할 가치가 있다. 반면 LACRT는 극저발현 분비단백 아티팩트 가능성이 커 우선순위에서 제외 권고.

### 면역관문억제제 심근염 (ICI-m (Raissadati), 신규 +85 · DB지지 novel 0 · rare-led 2)

ICI 심근염의 신호는 강한 염증·탐식/항원제시 축(호중구 탈과립, 리소좀, 파고솜, 항원 교차제시, 프로테아좀)의 상향과 산화적 인산화/미토콘드리아 호흡의 상향, 그리고 번역/리보솜의 하향으로 수렴한다. 이는 only_nbi와 대부분 공유되며(jaccard 0.771, sign_agree 1.0) rare 분기의 기여는 제한적이다. rare-led 2건(ATP6V1G3, IFNA1)은 각각 극저발현 조직특이 유전자와 인터페론 축에 귀속되어, 신호와 artifact를 구분해 볼 필요가 있다.

- **호중구 탈과립 / 항원 교차제시 (Reactome Neutrophil Degranulation; NES +3.0; ER-Phagosome/Cross-presentation)** — DB 지지 다수: LGALS3/CD68/ITGAM/MMP9/S100A8/S100A9/TLR2 등(각 myocarditis OT score ~0.03-0.05, 최대 19 target). PubMed neutrophil AND myocarditis ≈ 466건. ICI 심근염은 T세포·대식세포 침윤성 심근 염증이 확립된 병리로, 이 염증·탐식 축은 **기지(확립) 신호**로 신뢰.
- **MYH7 / MYH6 (KEGG Cardiac muscle contraction; NES +2.28)** — DB 지지 MYH7(0.27);MYH6(0.06). Open Targets에서 MYH7은 심근질환에 매우 강한 연관(hypertrophic cardiomyopathy 0.892, cardiomyopathy 0.758). PubMed MYH7 AND myocarditis ≈ 13건(대표 PMID 41746849, 유전성 확장성 심근증+심근염증). 심근세포 유래 전사체의 혈장 누출(심근 손상 마커)로 해석 가능한 **생물학적으로 정합한 축**.
- **산화적 인산화 / 미토콘드리아 호흡 (KEGG Oxidative phosphorylation NES +2.88; Respiratory Electron Transport)** — PubMed mitochondrial dysfunction AND myocarditis ≈ 211건. 심근 에너지대사 이상과 정합하나 cfRNA에서 OXPHOS 상향의 세포기원(심근 vs 면역세포)은 보수적으로 해석.
- **ATP6V1G3 (KEGG Collecting duct acid secretion NES +1.83; OxPhos; ROS in phagocytes; Transferrin recycling — rare lead)** — v-ATPase 서브유닛. Open Targets 조회 결과 심근염/심장과 유의 연관 없음(최상위도 type 2 diabetes 0.238). ATP6V1G3은 신장 집합관 제한 발현 유전자로 혈장 cfRNA 극저발현 → **artifact 가능성이 높은 novel 후보**. 다만 v-ATPase가 리소좀 산성화·파고솜 축을 공유하므로, 진짜 탐식 신호에 편승해 뜬 것일 수 있음(경로 오귀속).
- **IFNA1 (KEGG Herpes simplex virus 1 infection; NES -1.75, rare lead)** — DB 지지 TNF/IRF7/IFNG/IL1B/CGAS 등(interferon/염증 core, 각 ~0.03-0.10). PubMed interferon AND ICI myocarditis ≈ 16건(대표 PMID 42130414 심혈관 독성 염증 연속체, 42031428 JAK 억제). ICI-irAE에서 인터페론 축은 확립되어 있으나, rare lead IFNA1 자체는 극저발현 사이토카인 유전자로 개별 신뢰는 낮음 → 축은 **기지**, 개별 유전자는 보수적.

**해석:** ICI 심근염에서 normative 접근이 복원한 핵심 축은 (1) 호중구/대식세포 염증·항원제시(다수 OT·문헌 지지, 확립), (2) 심근 수축 유전자(MYH7/MYH6)의 혈장 신호(심근 손상 정합), (3) 인터페론/바이러스감지 축의 조절이며 대부분 rare 비의존적이다. rare-led 2건 중 IFNA1은 확립된 인터페론 축의 일부지만 개별 유전자 신뢰는 낮고, ATP6V1G3은 OT에서 심장 연관이 없고 조직특이 극저발현이라 **artifact 우려가 큰 novel 후보**로 명시한다. DB지지 novel이 0인 만큼, ICI-m의 견고한 결론은 rare가 아니라 nbi가 제공한 염증·심근 축이다.

### 췌장암 (Pancreatic Cancer (Moore), 신규 +79 · DB지지 novel 1 · rare-led 1)

with_rare가 표면화한 신규 신호는 소수이며, 전체 유의 경로의 골격은 rare 미포함 조건과 거의 동일하다(jaccard 0.805, 부호일치 1.0). 지배적 축은 미토콘드리아 산화적 인산화·전자전달계·TCA의 강한 하향(NES -2.9~-2.4대)과 응고/보체·ECM·IGF 수송의 상향으로, 이는 PDAC 특이 신호라기보다 다수 표현형에서 반복되는 대사·조성 축에 가깝다(보수적으로 해석). rare 분기의 기전적 귀속 증거는 단 1건(MSTN)으로, 노이즈 대비 신호 기여는 제한적이다.

- **MSTN (Positive Regulation Of Pathway-Restricted SMAD Protein Phosphorylation GO:0010862; NES 1.78)** — rare lead MSTN, DB 지지 TGFBR2(0.63)·BMPR1A(0.46)·PPARG(0.40). MSTN(myostatin)은 TGF-beta 상위과의 SMAD 인산화 조절자이며 극저발현 rare 분기에서 leading edge에 직접 진입. PubMed에서 myostatin+cancer는 수백 건 규모이나(대략 수백 건) MSTN과 췌장암을 직접 엮은 문헌은 소수(약 5건, 대표 PMID 33899538, 30622678)로 대부분 암 악액질(cachexia)·근소모 맥락이다. TGF-beta/SMAD 축 자체는 PDAC에서 확립된 종양 촉진/억제 이중 축이므로 경로 수준은 기지이나, cfRNA에서 MSTN 개별 유전자가 췌장암 신호로 뜬 것은 미보고 후보로 분류하고 극저발현 잔존 분산 가능성을 함께 경고.
- **CDKN2A (Stabilization Of P53 R-HSA-69541; NES -2.33)** — DB 지지 CDKN2A(0.73). CDKN2A는 PDAC 4대 드라이버 중 하나로 문헌 규모가 큼(CDKN2A+췌장암 약 1000건 이상, 대표 PMID 28810144). 확립(기지) 축이며 normative 접근이 종양억제 경로 이상을 정상 보정 후에도 복원함을 지지.
- **RBM10 / U2AF1 (mRNA Splicing, Via Spliceosome GO:0000398; NES -2.35)** — DB 지지 RBM10(0.59)·U2AF1(0.40). 스플라이싱 인자 이상은 여러 암에서 보고되나 U2AF1+췌장 직접 문헌은 소수(약 8건). 경로 수준은 그럴듯하되 췌장암 특이성은 후보 수준으로 명시.
- **WWTR1/YAP1/FAT4 (Hippo Signaling GO:0035329; NES 2.36)** — DB 지지 WWTR1(0.46)·YAP1(0.40)·FAT4(0.38). Hippo-YAP 축은 PDAC에서 확립(YAP1+췌장암 수백 건, 대표 PMID 계열 다수). 상향 방향은 종양 진행과 정합적인 기지 축.
- **DDR2 (Collagen Fibril Organization GO:0030199 / Non-integrin membrane-ECM R-HSA-3000171; NES 2.47/2.37)** — DB 지지 DDR2(0.46)·NF1(0.38). DDR2는 콜라겐 수용체로 섬유성 간질이 특징인 PDAC와 정합(DDR2+췌장암 약 12건, 대표 PMID 36530986). 소규모지만 방향·기전 모두 기지 축과 일치.

**해석:** normative/rare 접근이 복원한 핵심 축은 대부분 rare 미포함에서도 이미 유의한 기지 축(p53/CDKN2A, Hippo-YAP, ECM-DDR2)으로, rare 분기의 순수 기여는 MSTN 1건에 국한된다. MSTN·U2AF1은 췌장암 직접 문헌이 희소한 후보 신호이며, 특히 MSTN은 극저발현 rare 분기 산출이라 검증 대상. 지배적인 OXPHOS/전자전달 하향은 여러 표현형 공통 신호로 췌장암 특이 해석은 지양.

### 자간전증 (Pre-eclampsia (Moufarrej), 신규 +77 · DB지지 novel 2 · rare-led 0)

두 방향의 축이 분리된다. (1) 강한 하향(NES<0): 리보솜 생합성·rRNA 가공·세포질 번역·핵-세포질 수송 전반이 억제되고, 여기에 rare 유전자 **RPL10L**이 다수 term의 leading edge로 반복 등장한다. (2) 상향(NES>0): 시냅스 접착(뉴렉신/뉴로리진, PTPRD·LRFN2)·유기음이온 수송(SLCO3A1). 태반 리보솜/번역 억제는 자간전증의 확립된 축(태반 mTOR·단백합성 저하)과 정합하나, 이를 견인하는 RPL10L은 정소 특이 리보솜 파랄로그로 아티팩트 위험이 높다. rare-led는 0건(엄밀 정의상 novel이 아님)이라 rare 기여는 방향 강화에 그친다.

- **RPL10L (Cap-dependent Translation Initiation NES -2.27 · Eukaryotic Translation Elongation · Influenza/Viral mRNA Translation 등 다수)** — rare lead RPL10L(다수 term 반복). **RPL10L–자간전증 문헌 0건**(PubMed=0; Open Targets preeclampsia 근거 0건). 유전자 총 문헌 ~13건으로 **정소 특이(testis-restricted) 리보솜 단백 파랄로그**이며 남성 불임/희소정자증과 연관(대표 PMID 32111475, Fertil Steril 2020). 혈중 극저발현 정소제한 유전자가 번역·리보솜 term 전반의 leading edge를 점유한 것은 **생물학적 태반 번역억제라기보다 극저발현 잔존분산 아티팩트 강력 의심**. 태반 번역/리보솜 억제 축 자체는 count 분기에서도 유의(비신규)하며 문헌상 확립(리보솜×자간전증×태반 PubMed ~81건).
- **PTPRD / LRFN2 (Presynapse Organization GO:0099172 신규 NES 2.24 · Protein-protein Interactions At Synapses NES 2.26)** — DB 지지 PTPRD(0.33)·LRFN2(0.32). Open Targets에서 두 유전자 모두 자간전증에 **GWAS credible-set 유전연관** 존재(PTPRD 2건 score 0.71·0.67; LRFN2 1건 score 0.86)로 단순 co-mention이 아닌 유전근거. 다만 질병 특이 문헌은 희박(PTPRD AND PE PubMed=1건). 시냅스접착 유전자군의 상향은 **유전적 근거는 있으나 기전 미규명 신규 후보축**으로 검증 가치.
- **SLCO3A1 (Sodium-Independent Organic Anion Transport GO:0043252 신규 NES 2.19)** — DB 지지 SLCO3A1(0.29). PubMed SLCO3A1 AND PE = 0건. 태반 유기음이온 수송체로 기전적 개연성은 있으나 **문헌 미보고 신규 후보**.

**해석:** normative 접근이 복원한 자간전증 핵심은 **태반 리보솜·번역 억제(하향)**로 확립 축과 정합하나, 이를 표면화한 rare 유전자 **RPL10L은 정소특이 극저발현 파랄로그로 아티팩트 위험이 커 기전 귀속 금지**(태반 번역억제 결론은 count 분기·기존 문헌으로 뒷받침, RPL10L 없이도 성립). 상향축의 **PTPRD·LRFN2는 Open Targets GWAS 유전연관이 실재**하는 흥미로운 신규 후보이며, SLCO3A1은 문헌 미보고 후보. **브리프가 제시한 IFN-lambda(IFNL3) 항바이러스 축은 본 결과에서 근거 없음**: 상위 표의 "Influenza/Viral mRNA Translation" term은 번역기구(RPL10L 견인) term이지 IFN 반응이 아니며, IFNL3–자간전증 문헌은 사실상 0건, Open Targets 근거도 0건(IFN-lambda의 태반 항바이러스 역할 자체는 임신 일반 맥락에서 ~60건 존재하나 자간전증 특이 신호 아님).

### 대장암 (Colorectal Cancer (Chen), 신규 +73 · DB지지 novel 다수 · rare-led 0)

대장암은 5개 코호트 중 가장 깨끗한 케이스로, **rare-led novel이 0건**이다. 즉 rare(극저발현) 분기가 leading edge를 직접 이끈 신호가 없어, novel 73건은 전적으로 count route(NBI) 유전자가 결합 랭킹에서 표면화한 것이다. 표면화한 novel 축은 모두 대장암의 확립 생물학과 정합한다: TGF-β 수용체 신호, 세포-ECM 상호작용, Hippo/YAP, lamellipodium/RAC1 매개 이동성. artifact 우려가 사실상 없는 코호트.

- **Transforming Growth Factor Beta Receptor Signaling GO:0007179 (NES 2.46, novel)** — DB 지지 SMAD2(0.67)·TGFBR2(0.59)·SRC(0.50). TGF-β/SMAD은 대장암 정준 종양억제 축(TGFBR2는 MSI 대장암에서 불활성화). Open Targets TGFBR2–colorectal은 유전성 비용종증(HNPCC type 6) 경로로 genetic_association 0.74; PubMed TGFBR2/SMAD2 AND colorectal cancer ≈ 385/327건(PMID 28106826, 39805388). **기지 확립 축이 신규 유의로 복원.**
- **Signaling By Hippo R-HSA-2028269 (NES 2.39, 비-novel)** / **YAP1-·WWTR1(TAZ)-stimulated Gene Expression (NES 2.34)** — DB 지지 LATS2(0.55). Hippo-YAP 축은 대장암 증식/줄기세포성에 확립. 
- **Positive Regulation Of Cell-Substrate Junction Organization GO:0150117 (NES 2.16, novel)** — DB 지지 PIK3R1(0.64)·RAC1(0.44). RAC1은 대장암 침습/이동성 확립(PubMed RAC1 AND colorectal cancer ≈ 246건; TAM 유래 RAC1 축 PMID 38385857). **기지 확립 세포이동 축.**
- **Cell-extracellular Matrix Interactions R-HSA-446353 (NES 2.27, novel)** — DB 지지 없음. ECM 상호작용은 침습 맥락에 부합하나 이 term 자체는 정준 유전자 db_support가 없어 보수적으로 둔다.

**해석:** 대장암에서 normative 접근은 rare artifact 없이 TGF-β/SMAD(정준 종양억제)·Hippo-YAP·RAC1 이동성 등 확립 축을 신규 유의로 복원했다. rare-led 0건은 이 코호트에서 극저발현 분기가 별도 기전 신호를 더하지 않았음을 의미하며, novel 신호의 해석 신뢰도가 5개 중 가장 높다. 핵심 하향(번역/리보솜·OXPHOS)은 비-novel이며 강건하다. 순수 미보고 후보로 명시할 개별 유전자는 없다.

### 위암 (Stomach Cancer (Chen), 신규 +68 · DB지지 novel 1 · rare-led 2)

위암 코호트의 강건한 축은 novel이 아닌 확립 신호다: 번역/리보솜(RPL22 0.51)·핵수송(XPO1/RANBP2)·산화적 인산화의 하향과, RET signaling·TGF-β/SMAD2·세포-세포 접합(CTNNA1)의 상향. rare 분기가 표면화한 novel 2건은 상피 분화/분비 프로그램(cornified envelope, body fluid secretion)으로, 위 점막 상피 신호로 해석할 여지는 있으나 leading edge를 이끈 rare 유전자가 조직 특이 저발현이라 artifact 경고가 필요하다.

- **SPINK6 (Formation Of Cornified Envelope R-HSA-6809371; NES 2.37)** — rare lead SPINK6, DB 지지 없음. SPINK6는 표피 Kazal형 세린프로테아제 억제제로 위암 문헌 희소(PubMed SPINK6 AND gastric cancer ≈ 1건; 피부/kallikrein 맥락 PMID 27354280). cornified envelope term은 상피 분화 프로그램을 시사하나 **위암 특이 근거는 미보고 — 보수적 novel 후보.**
- **CSN3 (Body Fluid Secretion GO:0007589; NES 2.15)** — rare lead CSN3, DB 지지 없음. CSN3(κ-casein)은 유선 특이 우유 단백으로 대표 문헌이 소의 casein 다형성(PMID 35415213) 수준. 혈장 극저발현·조직 제한성으로 **강한 artifact 경고 — 생물학적 해석보다 잔존 분산 가능성 우선.**
- **Cell-Cell Junction Assembly GO:0007043 (NES 2.16, novel)** — DB 지지 CTNNA1(0.62). CTNNA1(α-catenin)은 Open Targets 유전성 미만형 위암 연관 0.555(genetic_association 0.91), 위암 접합 기능 문헌 다수(PubMed CTNNA1 AND gastric cancer ≈ 92건; PMID 32758476 HDGC 가이드라인). **기지 확립 축이 rare 결합에서 신규 유의로 표면화.**
- **RET Signaling R-HSA-8853659 (NES 2.42)** — DB 지지 SRC(0.49)·RET(0.41). 비-rare, 위암 RTK 신호 축.

**해석:** rare/normative 접근은 위암에서 CTNNA1 매개 세포접합(기지)을 신규 유의로 복원한 점이 가장 신뢰할 만하다. rare-led novel 2건 중 SPINK6(상피 분화)는 후속 검증 가치가 있는 보수적 후보로 남기되, CSN3(유선 casein)은 조직 특이 극저발현 artifact로 명시 배제 권고. 핵심 하향 축(번역·핵수송·OXPHOS)은 비-novel이며 강건하다.

### HIV·결핵 동시감염 (HIV + Tuberculosis (Chang), 신규 +64 · DB지지 novel 0 · rare-led 0)

동시감염 프로파일은 결핵 단독(Tuberculosis)과 같은 골수성/항균 축을 유지하되, 강력한 리보솜·번역 하향(cytoplasmic translation, cap-dependent initiation, rRNA processing; RPS27A leading edge, NES까지 -2.9대)이 상향 항균 축과 공존하는 이중 패턴을 보인다. rare-led novel은 0건이며 rare 유전자(RPL10L, GGTLC3)는 이미 유의한 리보솜/펩타이드 term의 leading edge에 부수적으로 들어갔을 뿐(novel=False)이라 rare 분기의 독립적 기여는 관찰되지 않는다. 상향 신호는 결핵과 HIV 양쪽의 확립 항균·항바이러스 축이 겹친 형태로, DB·문헌으로 뒷받침된다.

- **LTF/CTSG/CXCL9/S100A9 (Antimicrobial Humoral Response GO:0019730; NES 2.35)** — novel 아님. leading edge db_support LTF(0.52), CTSG(0.37), CXCL10(0.12), CXCL9(0.11), S100A9(0.09). 결핵 확립 항균/케모카인 축: LTF Open Targets tuberculosis 0.517, CXCL9(MIG)는 활동성 결핵 혈액/혈장 바이오마커로 광범위("CXCL9 AND tuberculosis" ~138건; 대표 PMID 38888093, 37740371). → 기지 축, 동시감염에서도 유지.
- **CTSG/IFNG/FCGR1A (Systemic lupus erythematosus KEGG; NES 2.86; Neutrophil extracellular trap formation; NES 2.71)** — leading edge CTSG(0.37), IFNG(0.12), FCGR1A(0.09), MAPK3, FPR1. SLE/NET term은 결핵·바이러스 감염에서 공통으로 뜨는 호중구·자가면역양 인터페론 시그니처의 대리 표지로, IFNG 포함은 결핵 IFN-γ 축과 정합. → 기지 골수성/IFN 축.
- **LTF;CCL5 (Negative Regulation Of Viral Process GO:0048525 / Viral Genome Replication GO:0045071; NES 2.34/2.33)** — leading edge LTF(0.52), CCL5(0.09). 항바이러스 축은 CCL5(RANTES, HIV 억제성 β-케모카인)와 LTF가 견인, HIV 동시감염 맥락과 정합("CCL5 AND tuberculosis" ~130건). → 기지 항바이러스/항균 교차 축.
- **RPS27A (Cytoplasmic Translation GO:0002181; NES -2.95; Cap-dependent Translation Initiation; NES -2.92)** — novel 아님. leading edge RPS27A(0.46). 리보솜/번역 대규모 하향은 동시감염에서 가장 강한 신호이며, 중증 감염의 혈장 cfRNA에서 흔한 번역·리보솜 억제 패턴과 부합(질병 특이 인과 아님, 전신 반응 대리표지로 보수적 해석).
- **RPL10L / GGTLC3 (rRNA Processing R-HSA-72312, NES -2.56 / Peptide Biosynthetic Process GO:0043043, NES -2.39)** — rare 유전자가 leading edge에 포함되나 term 자체는 novel 아님(RPS27A 주도). RPL10L(리보솜 유사유전자)·GGTLC3 모두 결핵·HIV 문헌 전무("RPL10L AND tuberculosis" 0건, "GGTLC3 AND tuberculosis" 0건) → 독립 신호 아님, 극저발현 부수 포함으로 판단.

**해석:** 동시감염에서 normative 접근이 복원한 축은 (1) 결핵/HIV 공통 항균·항바이러스(LTF, CXCL9, CCL5, S100A9, CTSG, IFNG)와 (2) 강한 리보솜/번역 하향(RPS27A)이며, 전자는 Open Targets(LTF 0.52)·문헌으로 뒷받침되는 기지 축이다. rare-led novel은 0건으로 이 표현형에서 rare 분기의 독립적 기전 기여는 없고, rare 유전자(RPL10L, GGTLC3)는 기존 리보솜 term에 부수적으로 포함된 것으로 문헌 근거가 없어 미보고 후보/artifact로 남겨둔다. HIV·결핵 동시감염 cfRNA 자체는 직접 선행연구가 극히 드물어("HIV AND tuberculosis coinfection cfRNA" ~1건) 본 프로파일은 검증 대상 신규 데이터셋으로 취급.

### HIV 감염 (HIV (Chang), 신규 +55 · DB지지 novel 1 · rare-led 3)

with_rare 결과의 상위 신호는 압도적으로 I형 인터페론/항바이러스 축(IFN-α/β signaling, ISG15, antiviral mechanism by ISGs, defense response to virus)에 수렴하며, 이는 모두 only_nbi에서 이미 유의(novel=False)했던 확립 신호로 rare 분기와 무관하다. rare 분기가 표면화한 novel 3건(leukocyte transendothelial migration, relaxin signaling, IL-6 family signaling)은 leading edge에 극저발현 rare 유전자가 직접 포함되나, 문헌·DB 근거가 희박해 신호보다는 극저발현 잔존 분산일 가능성을 우선 경계해야 한다. 전반적으로 normative 접근은 HIV의 기지(확립) 항바이러스 축을 강하게 복원했고, rare 기여는 노이즈 우세로 판단된다.

- **MX2/EIF2AK2/CXCL10 (Interferon Alpha/Beta Signaling R-HSA-909733; NES 2.7; Defense Response To Virus GO:0051607; NES 2.35)** — novel 아님. leading edge db_support KPNB1(0.56), MX2(0.11), EIF2AK2(0.52), CXCL10(0.10). 모두 HIV 확립 축: MX2(MxB)는 HIV-1 캡시드 핵 진입을 차단하는 대표적 제한인자(restriction factor)로 문헌 다수(PubMed "MX2 AND HIV" ~114건; 대표 PMID 30258007, 25568212), Open Targets HIV-1 infection 연관 0.105로 교차확인. EIF2AK2(PKR)는 항바이러스 번역억제 인자(~14건), CXCL10(IP-10)은 HIV 질병진행·저장소 바이오마커로 매우 광범위(~444건; 대표 PMID 29122683, 37920466). → 기지 축, normative가 정확히 재현.
- **PPIA (Negative/Regulation Of Viral Genome Replication GO:0045071/0045069; NES 2.57/2.51)** — leading edge PPIA(0.62). PPIA(cyclophilin A)는 HIV-1 캡시드 상호작용·복제 조절의 고전적 숙주인자로 문헌 방대("cyclophilin A AND HIV" ~376건; 대표 PMID 39480090, 38948800). → 기지 축.
- **IL31 (Interleukin-6 Family Signaling R-HSA-6783589; NES 1.73)** — rare lead IL31, DB 지지 없음. novel·rare-led. IL31과 HIV 직접 문헌은 극소("IL31 AND HIV" ~4건)이며 HIV 특이 기전 미보고; IL31 자체는 염증/소양증 맥락 문헌은 존재(~362건). → 미보고 후보. IL31은 극저발현 rare 유전자로 IL-6 family term 전체를 견인했을 가능성이 있어 artifact 후보로 검증 필요.
- **MYL10;CLDN17 (Leukocyte transendothelial migration; NES 1.75)** — rare lead MYL10, CLDN17; db_support RAC1(0.37). novel·rare-led. MYL10("MYL10 AND HIV" 0건)·CLDN17(0건) 모두 HIV 문헌 전무. leading edge의 실질 면역 앵커는 RAC1(백혈구 이동 GTPase)이며 rare 유전자는 극저발현 tie-breaker로 추정 → 강한 artifact 경고.
- **INSL5 (Relaxin signaling pathway; NES 1.75)** — rare lead INSL5, db_support 없음. INSL5("INSL5 AND HIV" 0건)는 장내분비 호르몬으로 HIV 문헌 전무 → 미보고 후보이나 생물학적 개연성 낮아 극저발현 artifact 가능성 높음.

**해석:** normative/rare 접근이 복원한 실질 축은 IFN-α/β·ISG·항바이러스 번역억제(MX2, EIF2AK2, PPIA, CXCL10)로 전부 HIV 확립 생물학이며 DB(Open Targets)·문헌으로 뒷받침된다. rare-led novel 3건(IL31, MYL10/CLDN17, INSL5)은 문헌·DB 근거가 사실상 없고 극저발현 유전자가 leading edge를 견인한 형태여서 생물학적 신호가 아니라 극저발현 잔존 분산 artifact 후보로 보수적으로 분류한다. 하향 신호는 리보솜/세포질 번역(RPS27A) 축으로, 별도 검증 대상.

### 단클론감마글로불린병증 (MGUS (Roskams-Hieter), 신규 +51 · DB지지 novel 0 · rare-led 3)

MGUS의 신규 신호는 두 갈래로 수렴한다: (1) MHC class I/II 항원처리·제시 및 ERAD/리소좀 경로의 하향, (2) 히스톤·염색질(RUNX1, mitotic prophase, DNA 손상/senescence)과 거대핵세포 분화의 상향. rare 분기(H2BC1)는 후자 염색질 축의 일부 novel term을 leading edge에서 이끌지만 DB지지 novel은 0건이며, 확립된 축(항원제시 B2M)은 rare가 아니라 nbi에서 온다(jaccard 0.563, sign_agree 1.0). 전반적으로 전종양성 형질세포 클론의 낮은 강도 신호로, MM보다 약하고 신중한 해석이 필요하다.

- **B2M / MHC class I 항원제시 (GO Antigen Processing/Presentation Via MHC I; NES -2.45)** — DB 지지 B2M(OT association score 0.01로 낮음). 그러나 PubMed beta-2 microglobulin AND monoclonal gammopathy ≈ 869건(대표 PMID 42295306)으로 β2-마이크로글로불린은 형질세포질환의 확립된 임상 바이오마커(ISS 병기)이다. MHC-I 항원제시/제시 기구의 하향은 면역회피 관점에서 **기지 축과 정합**하나, cfRNA 방향성(하향)의 임상 의미는 MGUS 단계에서 보수적으로 볼 것.
- **H2BC1 (Reactome Transcriptional Regulation By RUNX1 / Mitotic Prophase; NES +1.88/+1.85, rare lead H2BC1)** — 히스톤 H2B 클러스터 유전자가 염색질/유사분열 novel term을 이끔. PubMed H2BC1(HIST1H2BA) AND cancer ≈ 3건, 비특이적(대표 PMID 34319233은 정자형성 epimutation). 형질세포 크로마틴 재구성 측면에서 생물학적 개연성은 있으나 MGUS 특이 문헌은 **미보고 → 보수적 novel 후보**. H2B 클러스터는 저발현 반복 유전자로 극저발현 잔존분산 경고 병기.
- **Estrogen-dependent Gene Expression (NES +1.83, rare lead H2BC1)** — db_support ERBB4(0.18, 약함). 히스톤 leading edge 공유로 인한 경로 오귀속 가능성이 커 **후보로 유보**.
- **항원제시·자가면역 계열 하향(MHC II assembly, ERAD, Endosomal/Vacuolar, Allograft rejection; novel True)** — B2M/MHC 공통 유전자에 의해 여러 term이 동반 하향. 개별 term보다 "항원제시 기구 하향"이라는 축으로 묶어 해석하는 것이 타당.

**해석:** MGUS에서 normative 접근이 복원한 해석 가능한 축은 항원처리·제시(MHC I/II, B2M) 기구의 하향으로, β2-마이크로글로불린이라는 형질세포질환의 확립 바이오마커와 방향적으로 연결된다(단 OT score는 낮아 문헌 근거 우위). rare-led 3건은 모두 H2BC1 히스톤 클러스터에 귀속되며, 형질세포 크로마틴이라는 개연성은 있으나 MGUS 특이 문헌이 없고 저발현 반복유전자 특유의 artifact 가능성이 있어 **검증 대상 novel 후보**로 명시한다. 신규 DB지지가 0인 점을 고려해 MGUS 섹션은 전반적으로 낮은 확신도로 기술한다.

### 기타 암 (Other Cancer (Moore), 신규 +51 · DB지지 novel 4 · rare-led 6)

"기타 암"은 OT 참조가 일반 cancer(EFO_0000311)라 질병 특이성이 낮으므로 개별 연관은 보수적으로만 해석한다. 그럼에도 rare-led novel이 6건으로 본 세트 중 가장 많아 rare 분기 기여가 상대적으로 두드러진다. 신규 축은 상피 tight junction(CLDN25/CLDN17)·산화적 VEGF/혈관신생(NOX1)·RHO GTPase로 수렴하나, leading edge를 이끄는 유전자 다수가 극저발현 paralog 또는 Y염색체 유전자여서 **artifact 경계가 필요**하다. 지배 골격(번역·리보솜·OXPHOS 하향)은 rare 미포함과 동일(jaccard 0.77, 부호일치 1.0).

- **NOX1 (Positive Regulation Of VEGF Production GO:0010575 · RHO GTPase Cycle R-HSA-9012999; NES 1.93/1.90)** — rare lead NOX1, DB 지지 BRCA1(0.93)·PIK3R1(0.87)·ARHGAP35(0.75). NOX1은 NADPH oxidase로 ROS-매개 VEGF/혈관신생·RHO 신호에 관여하며 암 문헌이 상당(NOX1+cancer 약 390건, NOX1+VEGF 약 45건, 대표 PMID 34073365, 27874952). 경로(ROS→VEGF→혈관신생)는 기지 축이고 NOX1의 rare-led 진입은 생물학적으로 가장 신뢰할 만한 rare 신호. 단 BRCA1 db 지지는 leading edge 동반일 뿐 NOX1 자체 연관은 아님에 유의.
- **CLDN25 / CLDN17 (Tight Junction Assembly GO:0120192 · Bicellular Tight Junction GO:0070830; NES 1.95/1.91)** — rare lead CLDN25;CLDN17, DB 지지 STRN(0.77). claudin 계열 tight junction 붕괴는 상피암 침습에서 확립된 축(claudin+tight junction+cancer 약 1400건). 그러나 CLDN25(전체 약 7건)·CLDN17(cancer 약 11건)은 극저발현 희귀 paralog로 개별 문헌이 거의 없다. 경로는 기지, 개별 유전자는 **저발현 artifact 가능성 병기한 보수적 novel 후보**.
- **PIK3R1/ARHGAP35 (RHOB GTPase Cycle R-HSA-9013026; NES 2.45)** — DB 지지 PIK3R1(0.87)·ARHGAP35(0.75). RHO GTPase 사이클 상향은 세포 운동성/침습과 정합하는 기지 축이며 DB 점수도 높음.
- **KDR/NRG1/TSC1/RAC1/FYN (Regulation Of Focal Adhesion Assembly GO:0051893; NES 2.37)** — DB 지지 KDR(0.85)·NRG1(0.83)·TSC1(0.81)·RAC1(0.78)·FYN(0.74), 5개 강한 지지. 접착/침습 축의 상향으로 범암성 정합. 다만 일반 cancer 참조라 특정 종양 귀속은 불가.
- **DAZ2 / DAZ4 (Positive Regulation Of Translational Initiation GO:0045948; NES -1.82)** — rare lead DAZ2;DAZ4, DB 지지 없음. **DAZ2/DAZ4는 Y염색체 AZFc 영역 유전자(정소 특이, 무정자증 관련)로 암 축과 무관** (DAZ 관련 문헌은 azoospermia 맥락, 대표 PMID 37612512). 성별 조성/Y염색체 잔존 분산에 의한 전형적 **artifact로 강한 경고**, 생물학적 해석 금지.
- **PYDC2 (Negative Regulation Of IL-1 Beta Production GO:0032691; NES -1.77)** — rare lead PYDC2, DB 지지 없음. PYD-only 단백질로 유전자 전체 문헌이 극소(약 4건), 암 연관 미보고. 저발현 rare 분기 후보로 검증 대상.

**해석:** rare 분기가 복원한 신뢰 축은 NOX1(ROS-VEGF-혈관신생) 및 tight junction/RHO GTPase로 경로 수준은 기지이나, 일반 cancer 참조라 종양 특이성은 부여 불가. 개별 rare-led 유전자 다수(CLDN25/CLDN17 저발현 paralog, DAZ2/DAZ4 Y염색체, PYDC2 문헌부재)는 극저발현·성염색체 조성 artifact 위험이 높아 **DAZ2/DAZ4는 명확한 artifact로 배제**, CLDN25/CLDN17·PYDC2는 보수적 novel 후보로만 표기. NOX1만이 문헌·경로 정합이 뒷받침되는 우선 검증 대상.

### 췌장염 (Pancreatitis (Moore), 신규 +49 · DB지지 novel 1 · rare-led 1)

췌장염은 외분비 염증 표현형으로, 암과 달리 신규 신호를 종양 축으로 해석하면 안 된다(보수적 해석). 유의 골격은 rare 미포함과 거의 동일(jaccard 0.785, 부호일치 1.0)하며, 지배 축은 미토콘드리아 산화적 인산화·미토콘드리아 번역·리보솜의 강한 하향과 focal adhesion/세포-기질 접합의 상향이다. rare 분기 귀속 증거는 1건(ATP6V1G3)뿐이고 해당 유전자는 문헌·DB 지지가 없어 신호보다 노이즈 가능성을 우선 경고한다.

- **ATP6V1G3 (Insulin Receptor Recycling R-HSA-77387; NES -1.71)** — rare lead ATP6V1G3, DB 지지 없음. ATP6V1G3는 V-ATPase G3 아형(신장 우세 발현)으로, PubMed에서 pancreatitis와의 직접 문헌은 미보고(0건)이고 유전자 전체 문헌도 극소(약 18건). insulin receptor recycling 경로명은 V-ATPase의 엔도좀 산성화 기능에서 파생된 것으로 췌장염 인슐린 신호로 직결하기 어렵다. 극저발현 rare 분기 특유의 잔존 분산 가능성이 높은 **보수적 novel 후보**로 분류.
- **CFTR (Defective CFTR Causes Cystic Fibrosis R-HSA-5678895; NES -2.34)** — DB 지지 표기 없음이나 CFTR-췌장염 연관은 확립된 기지 축. CFTR 변이는 만성/유전성 췌장염의 잘 알려진 위험인자로 문헌 규모가 큼(CFTR+pancreatitis 약 1800건 이상, 대표 PMID 31860051, 21520337). rare 미포함에서도 유의하나, normative 접근이 외분비 채널 기능이상 축을 재현한다는 점에서 생물학적 정합성이 높다.
- **ABCG8 (ABC Transporter Disorders R-HSA-5619084; NES -2.21)** — DB 지지 ABCG8(0.49). 담즙성/담석 관련 스테롤 수송체로 담석성 췌장염 맥락에서 그럴듯하나 ABCG8+pancreatitis 직접 문헌은 소규모(약 18건). 경로는 기지, 개별 연관은 후보 수준.
- **CTSS (Antigen Processing And Presentation Of Exogenous Peptide Antigen GO:0002478; NES -2.34)** — DB 지지 CTSS(0.07, 약함). 카텝신 S는 염증성 항원제시와 연관되며 췌장염 문헌이 존재(CTSS+pancreatitis 약 37건). 염증 표현형에 정합적인 면역 축이나 DB 점수는 약함.
- **MRPS11 (Mitochondrial Translation R-HSA-5368287 및 계열; NES -2.45~-2.32)** — DB 지지 MRPS11(0.07, 약함). 미토콘드리아 리보솜 단백질로 유전자 전체 문헌이 극소(약 4건), 췌장염 직접 문헌 없음. 미토 번역·OXPHOS 하향은 다수 표현형 공통 축이라 췌장염 특이 신호로 보기 어려움 — **artifact/조성 신호 경고**.

**해석:** 췌장염에서 normative 접근이 복원한 가장 정합적인 축은 CFTR(외분비 채널 기능이상)로 기지 축이며 염증(CTSS)·담즙 수송(ABCG8)이 보조한다. rare 분기 순수 기여인 ATP6V1G3는 문헌·DB 미지지의 저발현 후보로 검증 필요. MRPS11 등 미토 번역/OXPHOS 하향은 표현형 비특이적 공통 축이므로 췌장염 고유 신호로 해석하지 말 것.



---

## 종합

1. **rare 포함은 방향 보존적·가산적.** 20개 전 질병에서 공유 term의 NES 부호 일치율 1.0. rare 분기는 기존 신호를 뒤집지 않고 신규 유의 경로만 추가(+49~+172). only_nbi와 모순되는 방향 전환은 한 건도 없다.

2. **신규 신호의 대부분은 count route(NBI)에서 유래하며, DB/문헌으로 지지된다.** 감염병(HIV/TB DB적중률 0.53~0.66)·GI암(대장암 TGF-β/SMAD·Hippo-YAP·RAC1, 간암 R-H의 CTNNB1/MET·VEGF)·CAD/HF(FGF5·COL6A3 섬유화 축) 등에서 신규 term이 확립 질병 생물학과 정합. rare-led=0인 대장암·폐암·CAD가 오히려 해석 신뢰도가 가장 높았다.

3. **rare 분기가 처음 표면화한 진성 미보고 후보는 소수.** 엄격한 skill 검증 결과 문헌·DB 근거가 실제로 전무한 검증가치 후보는 **ME/CFS의 PLA2G10**(지질염증 phospholipase A2, PubMed 0·OT 0건)이 대표적이다. 그 외 MSTN(췌장암 cachexia, 근거 희소)·MMP3(식도암 ECM, 상대적 지지) 정도가 조건부 후보.

4. **주의 - rare-led의 상당수는 저발현 특유 아티팩트.** rare-led 44건 중 다수가 생물학 신호가 아니라 극저발현 유전자군의 잔존 분산으로 판단된다: 후각/미각 수용체(ICI-treated OR*/TAS2R42, 혈중 미발현), Y염색체 정소 유전자(Other Cancer DAZ2/DAZ4·간암 TNP1), retrogene(RPL10L - MM·PE·간암 R-H에 반복 출현), 히스톤/디펜신 클러스터(MM H4C7, MGUS H2BC1, MM DEFB*), 조직특이 분비단백(위암 CSN3 카세인, ME/CFS LACRT 눈물샘, 췌장염 ATP6V1G3). 이들은 leading edge에 올라도 질병 기전으로 확대해석하지 않는다.

5. **skill 교차검증이 가설을 실제로 기각한 사례.** 자간전증의 IFN-λ/IFNL3 항바이러스 가설은 검증 결과 근거가 없었고("viral" translation term은 RPL10L retrogene이 주도한 번역기구 term의 오인, IFNL3-PE PubMed 0·OT 0건), 대신 태반 리보솜/번역 억제 축(RPL10L 제외 시에도 성립)과 시냅스-접착(PTPRD/LRFN2, OT GWAS 0.71/0.86)이 실제 지지 신호였다. 기억이 아닌 DB/문헌 조회에 근거한 보수적 해석의 가치를 보여준다.

6. **결론.** with_rare는 정규화 모델의 신호를 방향 보존적으로 확장하며, 그 부가가치의 핵심은 rare 분기 자체보다 **count route가 복원한 확립 질병 축의 강화**에 있다. rare 분기의 고유 기여는 대부분 저발현 아티팩트이나, PLA2G10 등 소수의 검증가치 후보를 표면화한다. 모든 개별 유전자 근거는 `GSEA/Reference/`에서 추적 가능.
