# PACKET: Pancreatic Cancer (Moore)

## meta
```json
{
 "phenotype": "Pancreatic Cancer (Moore)",
 "ot_disease": "exocrine pancreatic carcinoma",
 "n_sig_with_rare": 607,
 "n_novel": 79,
 "n_rare_led": 1,
 "only_nbi_sig": 577,
 "right_only": 79,
 "left_only": 49,
 "jaccard": 0.805,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
GO_Biological_Process_2023__Positive Regulation Of Pathway-Restricted SMAD Protein Phosphorylation (GO:0010862)	1.78	True	MSTN	TGFBR2(0.63);BMPR1A(0.46);PPARG(0.40)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
GO_Biological_Process_2023__Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)	-2.87	0.0	False			0
GO_Biological_Process_2023__Aerobic Electron Transport Chain (GO:0019646)	-2.86	0.0	False			0
GO_Biological_Process_2023__Proton Motive Force-Driven ATP Synthesis (GO:0015986)	-2.86	0.0	False			0
Reactome_2022__Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By Uncoupling Proteins R-HSA-163200	-2.82	0.0	False			0
GO_Biological_Process_2023__Cellular Respiration (GO:0045333)	-2.78	0.0	False			0
KEGG_2021_Human__Complement and coagulation cascades	2.73	0.0	False			0
GO_Biological_Process_2023__Proton Motive Force-Driven Mitochondrial ATP Synthesis (GO:0042776)	-2.72	0.0	False			0
KEGG_2021_Human__Oxidative phosphorylation	-2.71	0.0	False			0
GO_Biological_Process_2023__Oxidative Phosphorylation (GO:0006119)	-2.67	0.0	False			0
Reactome_2022__Respiratory Electron Transport R-HSA-611105	-2.64	0.0	False			0
Reactome_2022__NIK To Noncanonical NF-kB Signaling R-HSA-5676590	-2.56	0.0	False			0
Reactome_2022__Autodegradation Of Cdh1 By Cdh1:APC/C R-HSA-174084	-2.53	0.0	False			0
Reactome_2022__Activation Of NF-kappaB In B Cells R-HSA-1169091	-2.49	0.0	False			0
Reactome_2022__Dectin-1 Mediated Noncanonical NF-kB Signaling R-HSA-5607761	-2.49	0.0	False			0
Reactome_2022__Post-translational Protein Phosphorylation R-HSA-8957275	2.48	0.0	False			0
Reactome_2022__Switching Of Origins To A Post-Replicative State R-HSA-69052	-2.47	0.0	False			0
GO_Biological_Process_2023__Mitochondrial Electron Transport, NADH To Ubiquinone (GO:0006120)	-2.47	0.0	False			0
KEGG_2021_Human__Primary immunodeficiency	-2.47	0.0	False		PTPRC(0.47);BTK(0.42);LCK(0.38)	3
GO_Biological_Process_2023__Collagen Fibril Organization (GO:0030199)	2.47	0.0	False		DDR2(0.46);NF1(0.38)	2
GO_Biological_Process_2023__Aerobic Respiration (GO:0009060)	-2.47	0.0	False			0
Reactome_2022__APC/C:Cdc20 Mediated Degradation Of Securin R-HSA-174154	-2.46	0.0	False			0
Reactome_2022__Translocation Of ZAP-70 To Immunological Synapse R-HSA-202430	-2.46	0.0	False		LCK(0.38)	1
KEGG_2021_Human__ECM-receptor interaction	2.45	0.0	False			0
Reactome_2022__Citric Acid (TCA) Cycle And Respiratory Electron Transport R-HSA-1428517	-2.44	0.0	False			0
Reactome_2022__Downstream TCR Signaling R-HSA-202424	-2.43	0.0	False		LCK(0.38)	1
GO_Biological_Process_2023__Mitochondrial Respiratory Chain Complex Assembly (GO:0033108)	-2.43	0.0	False			0
GO_Biological_Process_2023__Negative Regulation Of Blood Coagulation (GO:0030195)	2.42	0.0	False			0
GO_Biological_Process_2023__Camera-Type Eye Development (GO:0043010)	2.42	0.0	False		WT1(0.42);NF1(0.38)	2
Reactome_2022__Vpu Mediated Degradation Of CD4 R-HSA-180534	-2.42	0.0	False			0
Reactome_2022__Regulation Of IGF Transport And Uptake By IGFBPs R-HSA-381426	2.42	0.0	False			0
Reactome_2022__CDK-mediated Phosphorylation And Removal Of Cdc6 R-HSA-69017	-2.41	0.0	False			0
Reactome_2022__Autodegradation Of E3 Ubiquitin Ligase COP1 R-HSA-349425	-2.4	0.0	False			0
Reactome_2022__Formation Of HIV Elongation Complex In Absence Of HIV Tat R-HSA-167152	-2.4	0.0	False		ERCC2(0.38)	1
Reactome_2022__Degradation Of DVL R-HSA-4641258	-2.4	0.0	False			0
Reactome_2022__Hh Mutants Are Degraded By ERAD R-HSA-5362768	-2.39	0.0	False			0
Reactome_2022__GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2 R-HSA-9762114	-2.39	0.0	False			0
Reactome_2022__RHOC GTPase Cycle R-HSA-9013106	2.37	0.0	False			0
Reactome_2022__Complex I Biogenesis R-HSA-6799198	-2.37	0.0	False			0
Reactome_2022__APC/C:Cdc20 Mediated Degradation Of Mitotic Proteins R-HSA-176409	-2.37	0.0	False			0
Reactome_2022__Non-integrin membrane-ECM Interactions R-HSA-3000171	2.37	0.0	False		DDR2(0.46)	1
Reactome_2022__FBXL7 Down-Regulates AURKA During Mitotic Entry And In Early Mitosis R-HSA-8854050	-2.37	0.0	False			0
Reactome_2022__Negative Regulation Of NOTCH4 Signaling R-HSA-9604323	-2.36	0.0	False			0
Reactome_2022__Hh Mutants Abrogate Ligand Secretion R-HSA-5387390	-2.36	0.0	False			0
GO_Biological_Process_2023__Hippo Signaling (GO:0035329)	2.36	0.0	False		WWTR1(0.46);YAP1(0.40);FAT4(0.38)	3
Reactome_2022__Formation Of HIV-1 Elongation Complex Containing HIV-1 Tat R-HSA-167200	-2.35	0.0	False		ERCC2(0.38)	1
GO_Biological_Process_2023__mRNA Splicing, Via Spliceosome (GO:0000398)	-2.35	0.0	False		RBM10(0.59);U2AF1(0.40)	2
Reactome_2022__TCR Signaling R-HSA-202403	-2.34	0.0	False		PTPRC(0.47);LCK(0.38)	2
Reactome_2022__Activation Of APC/C And APC/C:Cdc20 Mediated Degradation Of Mitotic Proteins R-HSA-176814	-2.34	0.0	False			0
Reactome_2022__mRNA Splicing R-HSA-72172	-2.33	0.0	False		U2AF1(0.40)	1
Reactome_2022__FCERI Mediated NF-kB Activation R-HSA-2871837	-2.33	0.0	False			0
Reactome_2022__APC:Cdc20 Mediated Degradation Of Cell Cycle Proteins Before Cycle Checkpoint Satisfied R-HSA-179419	-2.33	0.0	False			0
GO_Biological_Process_2023__RNA Splicing, Via Transesterification Reactions With Bulged Adenosine As Nucleophile (GO:0000377)	-2.33	0.0	False		RBM10(0.59);U2AF1(0.40)	2
Reactome_2022__Cdc20:Phospho-APC/C Mediated Degradation Of Cyclin A R-HSA-174184	-2.33	0.0	False			0
Reactome_2022__Synthesis Of Bile Acids And Bile Salts Via 27-Hydroxycholesterol R-HSA-193807	2.33	0.0	False		NCOA2(0.44)	1
Reactome_2022__Stabilization Of P53 R-HSA-69541	-2.33	0.0	False		CDKN2A(0.73)	1
Reactome_2022__Orc1 Removal From Chromatin R-HSA-68949	-2.32	0.0	False			0
Reactome_2022__APC/C:Cdh1 Mediated Degradation Of Cdc20 And APC/C:Cdh1 Targets In Late Mitosis/Early G1 R-HSA-174178	-2.32	0.0	False			0
Reactome_2022__Regulation Of APC/C Activators Between G1/S And Early Anaphase R-HSA-176408	-2.32	0.0	False			0
GO_Biological_Process_2023__NADH Dehydrogenase Complex Assembly (GO:0010257)	-2.32	0.0	False			0
GO_Biological_Process_2023__Mitochondrial Respiratory Chain Complex I Assembly (GO:0032981)	-2.32	0.0	False			0
