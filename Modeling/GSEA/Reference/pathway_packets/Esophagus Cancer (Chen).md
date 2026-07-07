# PACKET: Esophagus Cancer (Chen)

## meta
```json
{
 "phenotype": "Esophagus Cancer (Chen)",
 "ot_disease": "carcinoma of esophagus",
 "n_sig_with_rare": 804,
 "n_novel": 135,
 "n_rare_led": 6,
 "only_nbi_sig": 708,
 "right_only": 135,
 "left_only": 39,
 "jaccard": 0.794,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
GO_Biological_Process_2023__External Encapsulating Structure Organization (GO:0045229)	2.07	True	MMP3	NF1(0.47)
KEGG_2021_Human__Aldosterone-regulated sodium reabsorption	2.07	True	INS	PIK3R1(0.39)
GO_Biological_Process_2023__Positive Regulation Of Endothelial Cell Proliferation (GO:0001938)	1.9	True	CCL26	KDR(0.49);AKT1(0.40);ARNT(0.38)
GO_Biological_Process_2023__Anterograde Trans-Synaptic Signaling (GO:0098916)	1.85	True	HTR3D	GRIN2A(0.47)
GO_Biological_Process_2023__Water Transport (GO:0006833)	1.82	True	AVP	
Reactome_2022__SARS-CoV-2-host Interactions R-HSA-9705683	-1.61	True	IFNA1	IKBKB(0.39)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
KEGG_2021_Human__cAMP signaling pathway	2.7	0.0	False	TSHB;PDE4C	CREBBP(0.48);CACNA1D(0.48);GRIN2A(0.47);BRAF(0.41);AKT1(0.40);PIK3R1(0.39);PTCH1(0.38);CREB3L1(0.38)	8
GO_Biological_Process_2023__SMAD Protein Signal Transduction (GO:0060395)	2.63	0.0	False			0
Reactome_2022__Regulation Of Gene Expression In Beta Cells R-HSA-210745	2.6	0.0	False	INS	AKT1(0.40);HNF1A(0.39)	2
GO_Biological_Process_2023__Mitochondrial Respiratory Chain Complex Assembly (GO:0033108)	-2.52	0.0	False			0
GO_Biological_Process_2023__Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)	-2.48	0.0	False		SDHC(0.38)	1
GO_Biological_Process_2023__Sodium Ion Homeostasis (GO:0055078)	2.45	0.0053226954762738	False			0
GO_Biological_Process_2023__Mammary Gland Development (GO:0030879)	2.44	0.0042581563810191	False		ERBB4(0.51);JAK2(0.40)	2
GO_Biological_Process_2023__Aerobic Electron Transport Chain (GO:0019646)	-2.42	0.0	False		SDHC(0.38)	1
GO_Biological_Process_2023__Cellular Respiration (GO:0045333)	-2.4	0.0	False			0
Reactome_2022__Translation R-HSA-72766	-2.4	0.0	False		CARS1(0.39)	1
GO_Biological_Process_2023__Mitochondrial Translation (GO:0032543)	-2.4	0.0	False			0
Reactome_2022__Phospholipase C-mediated Cascade; FGFR2 R-HSA-5654221	2.39	0.0070969273016985	False		FGFR2(0.51)	1
Reactome_2022__Respiratory Electron Transport R-HSA-611105	-2.39	0.0	False			0
Reactome_2022__rRNA Processing In Nucleus And Cytosol R-HSA-8868773	-2.38	0.0	False			0
Reactome_2022__FRS-mediated FGFR2 Signaling R-HSA-5654700	2.38	0.006083080544313	False		FGFR2(0.51);PTPN11(0.45)	2
GO_Biological_Process_2023__Positive Regulation Of Insulin Receptor Signaling Pathway (GO:0046628)	2.38	0.0053226954762738	False		PTPN11(0.45)	1
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-2.38	0.0	False			0
GO_Biological_Process_2023__Positive Regulation Of Cellular Response To Insulin Stimulus (GO:1900078)	2.37	0.004731284867799	False		PTPN11(0.45)	1
Reactome_2022__Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By Uncoupling Proteins R-HSA-163200	-2.35	0.0	False			0
Reactome_2022__PI-3K cascade:FGFR2 R-HSA-5654695	2.35	0.0042581563810191	False		FGFR2(0.51);PTPN11(0.45);PIK3R1(0.39)	3
Reactome_2022__Major Pathway Of rRNA Processing In Nucleolus And Cytosol R-HSA-6791226	-2.35	0.0	False			0
GO_Biological_Process_2023__Translation (GO:0006412)	-2.35	0.0	False			0
Reactome_2022__Downstream Signaling Of Activated FGFR2 R-HSA-5654696	2.35	0.0035484636508492	False		FGFR2(0.51);PTPN11(0.45);PIK3R1(0.39)	3
Reactome_2022__rRNA Processing R-HSA-72312	-2.35	0.0	False			0
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-2.35	0.0	False			0
Reactome_2022__Activated Point Mutants Of FGFR2 R-HSA-2033519	2.35	0.0038710512554719	False		FGFR2(0.51)	1
Reactome_2022__Signal Attenuation R-HSA-74749	2.34	0.0032755049084762	False	INS		0
GO_Biological_Process_2023__Mitochondrial Gene Expression (GO:0140053)	-2.34	0.0	False			0
GO_Biological_Process_2023__Ribosome Biogenesis (GO:0042254)	-2.34	0.0	False		DDX10(0.46);NPM1(0.40);XPO1(0.39)	3
GO_Biological_Process_2023__Autonomic Nervous System Development (GO:0048483)	2.33	0.0030415402721565	False		NF1(0.47)	1
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-2.33	0.0	False			0
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-2.33	0.0	False			0
GO_Biological_Process_2023__Regulation Of Postsynaptic Membrane Potential (GO:0060078)	2.33	0.0026613477381369	False		GRIN2A(0.47)	1
Reactome_2022__Downstream Signaling Of Activated FGFR3 R-HSA-5654708	2.33	0.0028387709206794	False		PTPN11(0.45);PIK3R1(0.39);FGFR3(0.39)	3
Reactome_2022__PI-3K cascade:FGFR3 R-HSA-5654710	2.32	0.0025047978711877	False		PTPN11(0.45);PIK3R1(0.39);FGFR3(0.39)	3
GO_Biological_Process_2023__Ribosomal Small Subunit Biogenesis (GO:0042274)	-2.32	0.0	False		NPM1(0.40)	1
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-2.32	0.0	False			0
KEGG_2021_Human__Regulation of lipolysis in adipocytes	2.31	0.0023656424338995	False	TSHB	AKT1(0.40);PTGS1(0.39);PIK3R1(0.39)	3
KEGG_2021_Human__Oxidative phosphorylation	-2.31	0.0	False			0
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-2.31	0.0	False			0
Reactome_2022__SHC-mediated cascade:FGFR2 R-HSA-5654699	2.31	0.0021290781905095	False		FGFR2(0.51)	1
Reactome_2022__PI-3K cascade:FGFR1 R-HSA-5654689	2.31	0.0022411349373784	False		FGFR1(0.51);PTPN11(0.45);PIK3R1(0.39)	3
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-2.31	0.0	False			0
Reactome_2022__Elastic Fibre Formation R-HSA-1566948	2.3	0.002027693514771	False			0
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-2.3	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-2.3	0.0	False			0
GO_Biological_Process_2023__Proton Motive Force-Driven ATP Synthesis (GO:0015986)	-2.29	0.0	False		SDHC(0.38)	1
GO_Biological_Process_2023__Mitochondrial Respiratory Chain Complex I Assembly (GO:0032981)	-2.29	0.0	False			0
Reactome_2022__Mitochondrial Translation Initiation R-HSA-5368286	-2.29	0.0	False			0
GO_Biological_Process_2023__NADH Dehydrogenase Complex Assembly (GO:0010257)	-2.29	0.0	False			0
GO_Biological_Process_2023__Positive Regulation Of Glucose Import (GO:0046326)	2.29	0.0018513723395735	False		PTPN11(0.45);AKT1(0.40);PIK3R1(0.39)	3
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-2.29	0.0	False			0
Reactome_2022__FGFR2 Ligand Binding And Activation R-HSA-190241	2.29	0.0019355256277359	False		FGFR2(0.51)	1
KEGG_2021_Human__Ribosome	-2.29	0.0	False			0
Reactome_2022__Mitochondrial Translation R-HSA-5368287	-2.29	0.0	False			0
Reactome_2022__Viral mRNA Translation R-HSA-192823	-2.29	0.0	False			0
GO_Biological_Process_2023__Oxidative Phosphorylation (GO:0006119)	-2.28	0.0	False		SDHC(0.38)	1
Reactome_2022__YAP1- And WWTR1 (TAZ)-stimulated Gene Expression R-HSA-2032785	2.28	0.0017742318254246	False		YAP1(0.47);WWTR1(0.46)	2
GO_Biological_Process_2023__Ribonucleoprotein Complex Biogenesis (GO:0022613)	-2.28	0.0	False		NPM1(0.40);XPO1(0.39)	2
Reactome_2022__Role Of Second Messengers In Netrin-1 Signaling R-HSA-418890	2.27	0.0025548938286114	False			0
