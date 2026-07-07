# PACKET: ICI-m (Raissadati)

## meta
```json
{
 "phenotype": "ICI-m (Raissadati)",
 "ot_disease": "myocarditis",
 "n_sig_with_rare": 543,
 "n_novel": 85,
 "n_rare_led": 2,
 "only_nbi_sig": 509,
 "right_only": 85,
 "left_only": 51,
 "jaccard": 0.771,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
KEGG_2021_Human__Collecting duct acid secretion	1.83	True	ATP6V1G3	
KEGG_2021_Human__Herpes simplex virus 1 infection	-1.75	True	IFNA1	TNF(0.10);IRF7(0.06);IFNG(0.06);AKT1(0.06);IL1B(0.04);C3(0.04);CCL2(0.04);CCL5(0.03);CGAS(0.03);FASLG(0.03)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
KEGG_2021_Human__Lysosome	3.02	0.0	False		CD68(0.05)	1
Reactome_2022__Neutrophil Degranulation R-HSA-6798695	3.0	0.0	False		LGALS3(0.05);CD68(0.05);ITGAM(0.05);MMP9(0.04);STING1(0.03);ANPEP(0.03);GSN(0.03);S100A8(0.03);TLR2(0.03);S100A9(0.03)	19
KEGG_2021_Human__Oxidative phosphorylation	2.88	0.0	False	ATP6V1G3		0
Reactome_2022__Viral mRNA Translation R-HSA-192823	-2.82	0.0	False			0
GO_Biological_Process_2023__Proton Motive Force-Driven Mitochondrial ATP Synthesis (GO:0042776)	2.81	0.0	False			0
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-2.81	0.0	False			0
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-2.8	0.0	False			0
Reactome_2022__Selenocysteine Synthesis R-HSA-2408557	-2.8	0.0	False			0
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-2.79	0.0	False			0
GO_Biological_Process_2023__Proton Motive Force-Driven ATP Synthesis (GO:0015986)	2.77	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-2.76	0.0	False			0
Reactome_2022__Selenoamino Acid Metabolism R-HSA-2408522	-2.74	0.0	False			0
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-2.74	0.0	False			0
Reactome_2022__Response Of EIF2AK4 (GCN2) To Amino Acid Deficiency R-HSA-9633012	-2.72	0.0	False			0
GO_Biological_Process_2023__Oxidative Phosphorylation (GO:0006119)	2.72	0.0	False			0
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-2.69	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Enhanced By Exon Junction Complex (EJC) R-HSA-975957	-2.64	0.0	False			0
GO_Biological_Process_2023__Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)	2.64	0.0	False			0
Reactome_2022__Influenza Viral RNA Transcription And Replication R-HSA-168273	-2.62	0.0	False			0
GO_Biological_Process_2023__Aerobic Electron Transport Chain (GO:0019646)	2.58	0.0	False			0
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-2.58	0.0	False			0
GO_Biological_Process_2023__Cellular Respiration (GO:0045333)	2.58	0.0	False		PPARGC1A(0.03)	1
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-2.57	0.0	False			0
Reactome_2022__Antigen processing-Cross Presentation R-HSA-1236975	2.56	0.0	False		PSMB8(0.08);TLR4(0.04);CALR(0.03);S100A8(0.03);TLR2(0.03);S100A9(0.03);HLA-C(0.02);HMGB1(0.02)	8
GO_Biological_Process_2023__Protein Exit From Endoplasmic Reticulum (GO:0032527)	2.56	0.0	False			0
GO_Biological_Process_2023__Aerobic Respiration (GO:0009060)	2.55	0.0	False			0
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-2.55	0.0	False			0
GO_Biological_Process_2023__Vacuolar Acidification (GO:0007035)	2.55	0.0	False			0
GO_Biological_Process_2023__Endoplasmic Reticulum To Cytosol Transport (GO:1903513)	2.52	0.0	False			0
GO_Biological_Process_2023__Intracellular pH Reduction (GO:0051452)	2.51	0.0	False			0
Reactome_2022__Hedgehog Ligand Biogenesis R-HSA-5358346	2.48	0.0	False		PSMB8(0.08)	1
Reactome_2022__Influenza Infection R-HSA-168255	-2.47	0.0	False		TGFB1(0.05)	1
KEGG_2021_Human__Ribosome	-2.45	0.0	False			0
Reactome_2022__Defective CFTR Causes Cystic Fibrosis R-HSA-5678895	2.43	0.0	False		PSMB8(0.08)	1
Reactome_2022__Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By Uncoupling Proteins R-HSA-163200	2.42	0.0	False			0
GO_Biological_Process_2023__Lysosomal Lumen Acidification (GO:0007042)	2.41	0.0	False			0
GO_Biological_Process_2023__Retrograde Protein Transport, ER To Cytosol (GO:0030970)	2.39	0.0	False			0
Reactome_2022__Hh Mutants Abrogate Ligand Secretion R-HSA-5387390	2.38	0.0	False		PSMB8(0.08)	1
Reactome_2022__ER-Phagosome Pathway R-HSA-1236974	2.38	0.0	False		PSMB8(0.08);TLR4(0.04);CALR(0.03);S100A8(0.03);TLR2(0.03);S100A9(0.03);HLA-C(0.02);HMGB1(0.02)	8
GO_Biological_Process_2023__Mitochondrial Electron Transport, NADH To Ubiquinone (GO:0006120)	2.37	0.0	False			0
Reactome_2022__Respiratory Electron Transport R-HSA-611105	2.37	0.0	False			0
Reactome_2022__Transferrin Endocytosis And Recycling R-HSA-917977	2.36	0.0	False	ATP6V1G3		0
KEGG_2021_Human__Phagosome	2.36	0.0	False		TUBB8(0.26);ITGAM(0.05);TLR4(0.04);CALR(0.03);TLR2(0.03);ITGB2(0.02);HLA-C(0.02)	7
Reactome_2022__Hh Mutants Are Degraded By ERAD R-HSA-5362768	2.35	0.0	False		PSMB8(0.08)	1
GO_Biological_Process_2023__Ubiquitin-Dependent ERAD Pathway (GO:0030433)	2.35	0.0	False		STUB1(0.04);CALR(0.03)	2
Reactome_2022__Cross-presentation Of Soluble Exogenous Antigens (Endosomes) R-HSA-1236978	2.34	0.0	False		PSMB8(0.08)	1
Reactome_2022__SARS-CoV-2 Modulates Host Translation Machinery R-HSA-9754678	-2.33	0.0	False			0
Reactome_2022__Citric Acid (TCA) Cycle And Respiratory Electron Transport R-HSA-1428517	2.31	0.0003847716703705	False		LDLR(0.02)	1
Reactome_2022__ROS And RNS Production In Phagocytes R-HSA-1222556	2.3	0.0003607234409724	False	ATP6V1G3		0
Reactome_2022__Insulin Receptor Recycling R-HSA-77387	2.3	0.0003723596810037	False			0
KEGG_2021_Human__Proteasome	2.3	0.0003395044150328	False		PSMB8(0.08)	1
GO_Biological_Process_2023__Protein Processing (GO:0016485)	2.3	0.0003497924276096	False		CASP1(0.06)	1
KEGG_2021_Human__Other glycan degradation	2.3	0.000329804288889	False			0
Reactome_2022__Formation Of Ternary Complex, And Subsequently, 43S Complex R-HSA-72695	-2.3	0.0	False			0
GO_Biological_Process_2023__Peptide Biosynthetic Process (GO:0043043)	-2.29	0.0	False			0
Reactome_2022__Complex I Biogenesis R-HSA-6799198	2.29	0.0003206430586421	False			0
Reactome_2022__FCERI Mediated Ca+2 Mobilization R-HSA-2871809	-2.29	0.0	False			0
KEGG_2021_Human__Cardiac muscle contraction	2.28	0.0006239540600603	False		MYH7(0.27);MYH6(0.06)	2
Reactome_2022__Major Pathway Of rRNA Processing In Nucleolus And Cytosol R-HSA-6791226	-2.28	0.0	False			0
Reactome_2022__Cellular Response To Starvation R-HSA-9711097	-2.26	0.0	False			0
