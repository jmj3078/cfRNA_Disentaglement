# PACKET: Liver Cancer (Roskams-Hieter)

## meta
```json
{
 "phenotype": "Liver Cancer (Roskams-Hieter)",
 "ot_disease": "hepatocellular carcinoma",
 "n_sig_with_rare": 360,
 "n_novel": 146,
 "n_rare_led": 4,
 "only_nbi_sig": 220,
 "right_only": 146,
 "left_only": 6,
 "jaccard": 0.585,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
KEGG_2021_Human__Coronavirus disease	1.75	True	RPL10L	PIK3CA(0.68);EGFR(0.44);RPL22(0.36);MYD88(0.35);PIK3R1(0.35);IL6ST(0.33);MAPK1(0.33)
GO_Biological_Process_2023__Negative Regulation Of MAPK Cascade (GO:0043409)	1.73	True	DEFB114	NF1(0.47);APOE(0.39);SH2B3(0.37)
GO_Biological_Process_2023__Regulation Of MAP Kinase Activity (GO:0043405)	1.73	True	DEFB114	FLT1(0.61);NF1(0.47);EGFR(0.44);APOE(0.39);FGF2(0.38);SH2B3(0.37);FGF1(0.37);NTRK3(0.35);SOS1(0.33)
GO_Biological_Process_2023__Negative Regulation Of MAP Kinase Activity (GO:0043407)	1.69	True	DEFB114	NF1(0.47);APOE(0.39);SH2B3(0.37)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
KEGG_2021_Human__Oxidative phosphorylation	-2.59	0.0	False			0
GO_Biological_Process_2023__Lysosomal Lumen Acidification (GO:0007042)	-2.38	0.0037246407438715	True			0
GO_Biological_Process_2023__Regulation Of Endosome Size (GO:0051036)	-2.29	0.0099323753169907	True			0
GO_Biological_Process_2023__Negative Regulation Of Blood Coagulation (GO:0030195)	2.2	0.0	False		APOE(0.39)	1
GO_Biological_Process_2023__RNA Transport (GO:0050658)	2.18	0.0	False			0
GO_Biological_Process_2023__Protein Insertion Into ER Membrane (GO:0045048)	-2.15	0.0491652578191039	False			0
GO_Biological_Process_2023__Membrane Protein Intracellular Domain Proteolysis (GO:0031293)	-2.12	0.0496618765849535	True			0
GO_Biological_Process_2023__Mitochondrial Electron Transport, Cytochrome C To Oxygen (GO:0006123)	-2.12	0.0446956889264581	True			0
GO_Biological_Process_2023__Vacuolar Acidification (GO:0007035)	-2.11	0.0419022083685545	True			0
GO_Biological_Process_2023__Positive Regulation Of G1/S Transition Of Mitotic Cell Cycle (GO:1900087)	2.1	0.0026051192280225	False		EGFR(0.44);RRM2(0.43);RRM1(0.42)	3
KEGG_2021_Human__Adherens junction	2.1	0.00347349230403	False		CTNNB1(0.80);MET(0.78);EGFR(0.44);EP300(0.39);CREBBP(0.39);TCF7L2(0.35);PTPRB(0.34);AFDN(0.33);MAPK1(0.33)	9
GO_Biological_Process_2023__Mitochondrial Respiratory Chain Complex Assembly (GO:0033108)	-2.1	0.0372464074387151	True			0
GO_Biological_Process_2023__Quinone Biosynthetic Process (GO:1901663)	-2.1	0.0350116229923922	False			0
KEGG_2021_Human__Other glycan degradation	-2.06	0.039278029662645	True			0
KEGG_2021_Human__Complement and coagulation cascades	2.06	0.002084095382418	False			0
GO_Biological_Process_2023__Negative Regulation Of Viral Genome Replication (GO:0045071)	2.05	0.001736746152015	False			0
GO_Biological_Process_2023__Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)	-2.05	0.0422125950972104	True			0
Reactome_2022__RHOU GTPase Cycle R-HSA-9013420	2.04	0.00148863955887	False		PIK3R1(0.35);CLTC(0.32)	2
Reactome_2022__Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By Uncoupling Proteins R-HSA-163200	-2.04	0.0406845373561349	True			0
GO_Biological_Process_2023__Negative Regulation Of Viral Process (GO:0048525)	2.0	0.005210238456045	False			0
GO_Biological_Process_2023__Positive Regulation Of Secretion By Cell (GO:1903532)	1.99	0.00578915384005	False			0
GO_Biological_Process_2023__Regulation Of Endothelial Cell Migration (GO:0010594)	1.97	0.009378429220881	False		FLT4(0.70);KDR(0.68);NF1(0.47);APOE(0.39);FGF2(0.38);TEK(0.38);FGF1(0.37)	7
Reactome_2022__CDC42 GTPase Cycle R-HSA-9013148	1.97	0.0085258447462554	False		PREX2(0.39);PIK3R1(0.35);ARHGEF12(0.34)	3
GO_Biological_Process_2023__Actin Filament Bundle Assembly (GO:0051017)	1.96	0.0080157514708384	False			0
Reactome_2022__Formation Of Fibrin Clot (Clotting Cascade) R-HSA-140877	1.96	0.0078153576840675	False			0
GO_Biological_Process_2023__Regulation Of Epithelial Cell Apoptotic Process (GO:1904035)	1.95	0.009031079990478	False		TEK(0.38)	1
GO_Biological_Process_2023__Fibrinolysis (GO:0042730)	1.95	0.008187517573785	False			0
GO_Biological_Process_2023__Negative Regulation Of Endothelial Cell Proliferation (GO:0001937)	1.94	0.009807507681967	False		FLT1(0.61);NF1(0.47);APOE(0.39)	3
GO_Biological_Process_2023__Actin Filament Bundle Organization (GO:0061572)	1.94	0.01042047691209	False			0
Reactome_2022__BMAL1:CLOCK,NPAS2 Activates Circadian Gene Expression R-HSA-1368108	1.93	0.01042047691209	False		CREBBP(0.39);TBL1XR1(0.35);NCOA2(0.34)	3
Reactome_2022__RHOV GTPase Cycle R-HSA-9013424	1.93	0.0099242637258	False		PIK3R1(0.35);CLTC(0.32)	2
GO_Biological_Process_2023__Positive Regulation Of Wound Healing (GO:0090303)	1.93	0.0115783076801	False		HPSE(0.38);MTOR(0.32)	2
GO_Biological_Process_2023__Positive Regulation Of Heterotypic Cell-Cell Adhesion (GO:0034116)	1.93	0.0109689230653579	False			0
GO_Biological_Process_2023__Regulation Of Viral Genome Replication (GO:0045069)	1.92	0.01042047691209	False		PABPC1(0.43);DDB1(0.39);DDX3X(0.37)	3
GO_Biological_Process_2023__Negative Regulation Of Cholesterol Transport (GO:0032375)	1.92	0.0108735411256591	False			0
Reactome_2022__Common Pathway Of Fibrin Clot Formation R-HSA-140875	1.92	0.0108546634500937	False			0
GO_Biological_Process_2023__Regulation Of Cellular Component Biogenesis (GO:0044087)	1.92	0.0099468188706313	False		APOE(0.39);LATS1(0.35);PTPN11(0.35);NF2(0.33)	4
GO_Biological_Process_2023__Positive Regulation Of Substrate Adhesion-Dependent Cell Spreading (GO:1900026)	1.91	0.009676157132655	False			0
GO_Biological_Process_2023__Myoblast Differentiation (GO:0045445)	1.91	0.01042047691209	False		RB1(0.63);TCF7L2(0.35)	2
Reactome_2022__RAC1 GTPase Cycle R-HSA-9013149	1.91	0.0100611501220179	False		PIK3CA(0.68);PREX2(0.39);PIK3R1(0.35);SOS1(0.33)	4
GO_Biological_Process_2023__Double-Strand Break Repair Via Break-Induced Replication (GO:0000727)	1.91	0.01042047691209	False			0
GO_Biological_Process_2023__Negative Regulation Of Coagulation (GO:0050819)	1.91	0.0100345333227533	False		APOE(0.39)	1
GO_Biological_Process_2023__Vasculature Development (GO:0001944)	1.9	0.0116835650226463	False		FLT4(0.70);PIK3CA(0.68);PDGFRB(0.67);STK11(0.32)	4
GO_Biological_Process_2023__Endothelial Cell Development (GO:0001885)	1.9	0.0114289101616471	False		MET(0.78)	1
Reactome_2022__Signaling By VEGF R-HSA-194138	1.9	0.012048676429604	False		CTNNB1(0.80);FLT4(0.70);KDR(0.68);PIK3CA(0.68);FLT1(0.61);NRAS(0.44);PIK3R1(0.35);MTOR(0.32)	8
Reactome_2022__Plasma Lipoprotein Assembly R-HSA-8963898	1.9	0.0113399307572744	False		APOE(0.39)	1
Reactome_2022__YAP1- And WWTR1 (TAZ)-stimulated Gene Expression R-HSA-2032785	1.89	0.0118677653721025	False			0
Reactome_2022__RHOB GTPase Cycle R-HSA-9013026	1.89	0.0112431461419918	False		PIK3R1(0.35);ARHGEF12(0.34)	2
Reactome_2022__RHOC GTPase Cycle R-HSA-9013106	1.89	0.0115470149566402	False		PIK3R1(0.35);ARHGEF12(0.34)	2
GO_Biological_Process_2023__Nuclear Chromosome Segregation (GO:0098813)	1.89	0.012206844382734	False		TOP2A(0.45);TOP2B(0.39);LATS1(0.35)	3
Reactome_2022__Signaling By High-Kinase Activity BRAF Mutants R-HSA-6802948	1.88	0.01141290328467	False		BRAF(0.68);NRAS(0.44);MAPK1(0.33)	3
KEGG_2021_Human__Prostate cancer	1.88	0.011691266779418	False		TP53(0.80);CTNNB1(0.80);BRAF(0.68);PIK3CA(0.68);RAF1(0.67);PDGFRB(0.67);RB1(0.63);KRAS(0.53);NRAS(0.44);EGFR(0.44)	23
KEGG_2021_Human__African trypanosomiasis	1.88	0.0106421891868153	False		MYD88(0.35)	1
Reactome_2022__Oncogene Induced Senescence R-HSA-2559585	1.88	0.0106470090188745	False		TP53(0.80);RB1(0.63);CDKN2A(0.61);MDM2(0.35);MAPK1(0.33);CDK4(0.32)	6
GO_Biological_Process_2023__Positive Regulation Of Cell Cycle G1/S Phase Transition (GO:1902808)	1.88	0.0108941349535486	False		EGFR(0.44);RRM2(0.43);RRM1(0.42)	3
Reactome_2022__Non-integrin membrane-ECM Interactions R-HSA-3000171	1.88	0.0122908189219523	True		DDR2(0.45)	1
GO_Biological_Process_2023__Negative Regulation Of Protein Serine/Threonine Kinase Activity (GO:0071901)	1.88	0.0119835484489035	False	DEFB114	APC(0.74);RB1(0.63);CDKN2A(0.61);NF1(0.47);APOE(0.39);SH2B3(0.37);LATS1(0.35)	7
Reactome_2022__Oncogenic MAPK Signaling R-HSA-6802957	1.88	0.0111474869292125	False		BRAF(0.68);NF1(0.47);NRAS(0.44);MAPK1(0.33)	4
KEGG_2021_Human__AGE-RAGE signaling pathway in diabetic complications	1.88	0.010883609219294	False		PIK3CA(0.68);KRAS(0.53);NRAS(0.44);CDKN1B(0.38);CCND1(0.37);TGFBR2(0.36);PIK3R1(0.35);MAPK1(0.33);CDK4(0.32)	9
GO_Biological_Process_2023__Actomyosin Structure Organization (GO:0031032)	1.87	0.0113853358854316	False			0
