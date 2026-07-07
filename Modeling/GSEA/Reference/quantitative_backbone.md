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
