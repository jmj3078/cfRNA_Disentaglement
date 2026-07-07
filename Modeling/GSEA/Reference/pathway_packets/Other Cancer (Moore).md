# PACKET: Other Cancer (Moore)

## meta
```json
{
 "phenotype": "Other Cancer (Moore)",
 "ot_disease": "cancer",
 "n_sig_with_rare": 319,
 "n_novel": 51,
 "n_rare_led": 6,
 "only_nbi_sig": 297,
 "right_only": 51,
 "left_only": 29,
 "jaccard": 0.77,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
GO_Biological_Process_2023__Tight Junction Assembly (GO:0120192)	1.95	True	CLDN25;CLDN17	STRN(0.77)
GO_Biological_Process_2023__Positive Regulation Of Vascular Endothelial Growth Factor Production (GO:0010575)	1.93	True	NOX1	BRCA1(0.93)
GO_Biological_Process_2023__Bicellular Tight Junction Assembly (GO:0070830)	1.91	True	CLDN25;CLDN17	STRN(0.77)
Reactome_2022__RHO GTPase Cycle R-HSA-9012999	1.9	True	NOX1	PIK3R1(0.87);ARHGAP35(0.75)
GO_Biological_Process_2023__Positive Regulation Of Translational Initiation (GO:0045948)	-1.82	True	DAZ2;DAZ4	
GO_Biological_Process_2023__Negative Regulation Of Interleukin-1 Beta Production (GO:0032691)	-1.77	True	PYDC2	

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-3.19	0.0	False			0
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-3.15	0.0	False			0
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-3.15	0.0	False			0
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-3.11	0.0	False			0
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-3.1	0.0	False			0
GO_Biological_Process_2023__Peptide Biosynthetic Process (GO:0043043)	-3.09	0.0	False	GGTLC3		0
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-3.08	0.0	False			0
KEGG_2021_Human__Ribosome	-3.06	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-3.06	0.0	False			0
Reactome_2022__Viral mRNA Translation R-HSA-192823	-3.06	0.0	False			0
GO_Biological_Process_2023__Macromolecule Biosynthetic Process (GO:0009059)	-3.04	0.0	False			0
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-3.03	0.0	False			0
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-3.02	0.0	False			0
Reactome_2022__Selenocysteine Synthesis R-HSA-2408557	-3.0	0.0	False			0
Reactome_2022__Response Of EIF2AK4 (GCN2) To Amino Acid Deficiency R-HSA-9633012	-3.0	0.0	False			0
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-3.0	0.0	False			0
Reactome_2022__Selenoamino Acid Metabolism R-HSA-2408522	-2.93	0.0	False			0
Reactome_2022__Regulation Of Expression Of SLITs And ROBOs R-HSA-9010553	-2.9	0.0	False		SEM1(0.75);RBX1(0.74)	2
Reactome_2022__Nonsense Mediated Decay (NMD) Enhanced By Exon Junction Complex (EJC) R-HSA-975957	-2.88	0.0	False			0
GO_Biological_Process_2023__Translation (GO:0006412)	-2.8	0.0	False			0
Reactome_2022__Cellular Response To Starvation R-HSA-9711097	-2.71	0.0	False		FLCN(0.84)	1
Reactome_2022__Influenza Viral RNA Transcription And Replication R-HSA-168273	-2.71	0.0	False		PARP1(0.73)	1
Reactome_2022__Translation R-HSA-72766	-2.7	0.0	False			0
Reactome_2022__Signaling By ROBO Receptors R-HSA-376176	-2.7	0.0	False	BUB1B-PAK6	SEM1(0.75);CXCR4(0.75);RBX1(0.74)	3
Reactome_2022__Major Pathway Of rRNA Processing In Nucleolus And Cytosol R-HSA-6791226	-2.7	0.0	False			0
Reactome_2022__rRNA Processing In Nucleus And Cytosol R-HSA-8868773	-2.67	0.0	False			0
Reactome_2022__Translation Initiation Complex Formation R-HSA-72649	-2.67	0.0	False			0
Reactome_2022__Formation Of Ternary Complex, And Subsequently, 43S Complex R-HSA-72695	-2.64	0.0	False			0
Reactome_2022__SARS-CoV-2 Modulates Host Translation Machinery R-HSA-9754678	-2.63	0.0	False			0
Reactome_2022__mRNA Activation Upon Binding Of Cap-Binding Complex And eIFs, Subsequent Binding To 43S R-HSA-72662	-2.63	0.0	False			0
Reactome_2022__rRNA Processing R-HSA-72312	-2.63	0.0	False			0
Reactome_2022__Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By Uncoupling Proteins R-HSA-163200	-2.62	0.0	False			0
Reactome_2022__Ribosomal Scanning And Start Codon Recognition R-HSA-72702	-2.59	0.0	False			0
Reactome_2022__Influenza Infection R-HSA-168255	-2.55	0.0	False		PARP1(0.73)	1
GO_Biological_Process_2023__Proton Motive Force-Driven ATP Synthesis (GO:0015986)	-2.53	0.0	False		SDHA(0.84)	1
KEGG_2021_Human__Oxidative phosphorylation	-2.49	0.0	False	COX8C		0
GO_Biological_Process_2023__Gene Expression (GO:0010467)	-2.48	0.0	False		CASP8(0.82)	1
Reactome_2022__Respiratory Electron Transport R-HSA-611105	-2.46	0.0	False			0
GO_Biological_Process_2023__Oxidative Phosphorylation (GO:0006119)	-2.46	0.0	False		SDHA(0.84)	1
Reactome_2022__RHOB GTPase Cycle R-HSA-9013026	2.45	0.0	False		PIK3R1(0.87);ARHGAP35(0.75)	2
Reactome_2022__Degradation Of DVL R-HSA-4641258	-2.42	0.0	False		SEM1(0.75);RBX1(0.74)	2
GO_Biological_Process_2023__Proton Motive Force-Driven Mitochondrial ATP Synthesis (GO:0042776)	-2.41	0.0	False		SDHA(0.84)	1
Reactome_2022__Vpu Mediated Degradation Of CD4 R-HSA-180534	-2.39	0.0	False		SEM1(0.75)	1
Reactome_2022__NIK To Noncanonical NF-kB Signaling R-HSA-5676590	-2.38	0.0	False		SEM1(0.75)	1
Reactome_2022__Autodegradation Of E3 Ubiquitin Ligase COP1 R-HSA-349425	-2.37	0.0	False		SEM1(0.75)	1
KEGG_2021_Human__Proteasome	-2.37	0.0	False		SEM1(0.75)	1
GO_Biological_Process_2023__Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)	-2.37	0.0	False		SDHA(0.84);SDHAF2(0.79)	2
Reactome_2022__Degradation Of AXIN R-HSA-4641257	-2.37	0.0	False		AXIN1(0.86);SEM1(0.75)	2
GO_Biological_Process_2023__Regulation Of Focal Adhesion Assembly (GO:0051893)	2.37	0.0	False		KDR(0.85);NRG1(0.83);TSC1(0.81);RAC1(0.78);FYN(0.74)	5
Reactome_2022__Negative Regulation Of NOTCH4 Signaling R-HSA-9604323	-2.36	0.0	False		SEM1(0.75);RBX1(0.74)	2
GO_Biological_Process_2023__Aerobic Electron Transport Chain (GO:0019646)	-2.35	0.0	False		SDHA(0.84);SDHAF2(0.79)	2
Reactome_2022__Regulation Of Apoptosis R-HSA-169911	-2.34	0.0	False		SEM1(0.75)	1
Reactome_2022__Regulation Of Activated PAK-2p34 By Proteasome Mediated Degradation R-HSA-211733	-2.33	0.0	False		SEM1(0.75)	1
Reactome_2022__Citric Acid (TCA) Cycle And Respiratory Electron Transport R-HSA-1428517	-2.32	0.0	False		SDHA(0.84)	1
Reactome_2022__Dectin-1 Mediated Noncanonical NF-kB Signaling R-HSA-5607761	-2.32	0.0	False		SEM1(0.75)	1
GO_Biological_Process_2023__Cellular Respiration (GO:0045333)	-2.32	0.0	False			0
Reactome_2022__Ubiquitin Mediated Degradation Of Phosphorylated Cdc25A R-HSA-69601	-2.32	0.0	False		SEM1(0.75)	1
Reactome_2022__SCF-beta-TrCP Mediated Degradation Of Emi1 R-HSA-174113	-2.31	0.0	False		SEM1(0.75)	1
Reactome_2022__GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2 R-HSA-9762114	-2.3	0.0	False		SEM1(0.75);RBX1(0.74)	2
GO_Biological_Process_2023__Ribosomal Small Subunit Biogenesis (GO:0042274)	-2.3	0.0	False			0
