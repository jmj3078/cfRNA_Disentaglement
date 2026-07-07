# PACKET: MM (Roskams-Hieter)

## meta
```json
{
 "phenotype": "MM (Roskams-Hieter)",
 "ot_disease": "plasma cell myeloma",
 "n_sig_with_rare": 411,
 "n_novel": 117,
 "n_rare_led": 4,
 "only_nbi_sig": 320,
 "right_only": 117,
 "left_only": 26,
 "jaccard": 0.673,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-2.89	True	RPL10L	
Reactome_2022__Defensins R-HSA-1461973	-1.88	True	DEFB113;DEFB127;DEFB125;DEFB106A;DEFB4A;DEFB116;DEFB126	
KEGG_2021_Human__Systemic lupus erythematosus	1.75	True	H4C7	
KEGG_2021_Human__Neutrophil extracellular trap formation	1.7	True	H4C7	PIK3CA(0.31);MTOR(0.31);RAF1(0.31);PIK3CB(0.31);MAPK1(0.30)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-3.39	0.0	False	RPL10L	EEF1A2(0.34)	1
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-3.27	0.0	False	RPL10L		0
Reactome_2022__Selenoamino Acid Metabolism R-HSA-2408522	-2.93	0.0	False	RPL10L		0
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-2.91	0.0	False	RPL10L		0
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-2.91	0.0	False	RPL10L		0
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-2.89	0.0	True	RPL10L		0
GO_Biological_Process_2023__Immunoglobulin Mediated Immune Response (GO:0016064)	-2.89	0.0	False		CSF2RB(0.35)	1
Reactome_2022__Response Of EIF2AK4 (GCN2) To Amino Acid Deficiency R-HSA-9633012	-2.78	0.0	False	RPL10L		0
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-2.78	0.0	False		RPL10(0.44)	1
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-2.77	0.0	False	RPL10L		0
Reactome_2022__Selenocysteine Synthesis R-HSA-2408557	-2.76	0.0	False	RPL10L		0
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-2.73	0.0	False	RPL10L	PABPC1(0.42)	1
KEGG_2021_Human__Ribosome	-2.62	0.0	False	RPL10L	RPL10(0.44)	1
Reactome_2022__Viral mRNA Translation R-HSA-192823	-2.53	0.0	False	RPL10L		0
GO_Biological_Process_2023__Cellular Respiration (GO:0045333)	-2.51	0.0	False			0
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-2.49	0.0	False	RPL10L	PABPC1(0.42)	1
GO_Biological_Process_2023__Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)	-2.48	0.0	False			0
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-2.42	0.0	False	RPL10L		0
GO_Biological_Process_2023__Aerobic Electron Transport Chain (GO:0019646)	-2.41	0.0	False			0
GO_Biological_Process_2023__Mitochondrial Respiratory Chain Complex I Assembly (GO:0032981)	-2.36	0.0003482240869701	False			0
GO_Biological_Process_2023__NADH Dehydrogenase Complex Assembly (GO:0010257)	-2.36	0.0003482240869701	False			0
KEGG_2021_Human__Asthma	-2.35	0.0003323957193806	False			0
KEGG_2021_Human__Oxidative phosphorylation	-2.29	0.0003179437315814	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Enhanced By Exon Junction Complex (EJC) R-HSA-975957	-2.29	0.0003046960760988	False	RPL10L		0
GO_Biological_Process_2023__Gas Transport (GO:0015669)	2.27	0.0	False			0
GO_Biological_Process_2023__Hydrogen Peroxide Catabolic Process (GO:0042744)	2.24	0.0	False			0
Reactome_2022__Translocation Of ZAP-70 To Immunological Synapse R-HSA-202430	-2.24	0.0005850164661098	False			0
GO_Biological_Process_2023__Carbon Dioxide Transport (GO:0015670)	2.23	0.0	False			0
GO_Biological_Process_2023__MHC Class II Protein Complex Assembly (GO:0002399)	-2.21	0.0013542047826617	False			0
GO_Biological_Process_2023__Peptide Antigen Assembly With MHC Class II Protein Complex (GO:0002503)	-2.21	0.0013542047826617	False			0
GO_Biological_Process_2023__Immunoglobulin Production Involved In Immunoglobulin-Mediated Immune Response (GO:0002381)	-2.2	0.0018281764565933	False			0
Reactome_2022__Kinesins R-HSA-983189	2.19	0.0	False			0
GO_Biological_Process_2023__B Cell Proliferation (GO:0042100)	-2.18	0.0020172981589995	False		PTPRC(0.31);CD79A(0.30)	2
GO_Biological_Process_2023__Mitochondrial Electron Transport, Cytochrome C To Oxygen (GO:0006123)	-2.17	0.0023589373633462	False			0
KEGG_2021_Human__Allograft rejection	-2.17	0.0024375686087911	False			0
GO_Biological_Process_2023__One-Carbon Compound Transport (GO:0019755)	2.16	0.0	False			0
GO_Biological_Process_2023__Erythrocyte Differentiation (GO:0030218)	2.15	0.0	False		JAK2(0.31);IKZF1(0.29)	2
Reactome_2022__HDACs Deacetylate Histones R-HSA-3214815	2.15	0.0	False			0
Reactome_2022__Vitamin D (Calciferol) Metabolism R-HSA-196791	-2.14	0.0052560073127058	False			0
GO_Biological_Process_2023__Microtubule Cytoskeleton Organization Involved In Mitosis (GO:1902850)	2.14	0.0	False		SETD2(0.32)	1
GO_Biological_Process_2023__Oxygen Transport (GO:0015671)	2.13	0.0	False			0
KEGG_2021_Human__Cell cycle	2.12	0.0	False		CCND1(0.53);RB1(0.44);MDM2(0.33);CCND2(0.32);CDK6(0.32);ABL1(0.31);EP300(0.30);CREBBP(0.30);SMAD2(0.29)	9
Reactome_2022__Polo-like Kinase Mediated Events R-HSA-156711	2.11	0.0	False		EP300(0.30)	1
GO_Biological_Process_2023__Positive Regulation Of Cell Cycle Process (GO:0090068)	2.11	0.0	False			0
Reactome_2022__SARS-CoV-2 Modulates Host Translation Machinery R-HSA-9754678	-2.1	0.0088638858501494	False			0
GO_Biological_Process_2023__Proton Motive Force-Driven ATP Synthesis (GO:0015986)	-2.1	0.0090333424914023	False			0
Reactome_2022__G1/S-Specific Transcription R-HSA-69205	2.09	0.0	False			0
GO_Biological_Process_2023__Positive Regulation Of Attachment Of Spindle Microtubules To Kinetochore (GO:0051987)	2.09	0.0	False			0
KEGG_2021_Human__Autoimmune thyroid disease	-2.09	0.0094020503481942	False			0
GO_Biological_Process_2023__Sister Chromatid Segregation (GO:0000819)	2.08	0.0	False		TOP2A(0.59);RB1(0.44);TOP2B(0.41)	3
KEGG_2021_Human__Intestinal immune network for IgA production	-2.08	0.0100796755985145	False		TNFRSF13B(0.56);TNFSF13B(0.31)	2
Reactome_2022__DNA Damage/Telomere Stress Induced Senescence R-HSA-2559586	2.08	0.0	False		TP53(0.59);RB1(0.44)	2
GO_Biological_Process_2023__Mitochondrial Electron Transport, NADH To Ubiquinone (GO:0006120)	-2.08	0.0103596665873622	False			0
GO_Biological_Process_2023__Mitotic Cytokinesis (GO:0000281)	2.07	0.0005817514600545	False			0
GO_Biological_Process_2023__Mitotic Spindle Assembly (GO:0090307)	2.07	0.0	False			0
GO_Biological_Process_2023__Mitotic Spindle Elongation (GO:0000022)	2.06	0.0005235763140491	False			0
Reactome_2022__Transcriptional Regulation Of Granulopoiesis R-HSA-9616222	2.06	0.0004986441086182	False		EP300(0.30);CREB1(0.30)	2
Reactome_2022__TP53 Regulates Transcription Of Genes Involved In G2 Cell Cycle Arrest R-HSA-6804114	2.06	0.0004759784673173	False		EP300(0.30)	1
Reactome_2022__Factors Involved In Megakaryocyte Development And Platelet Production R-HSA-983231	2.06	0.000455283751347	False		TP53(0.59);ABL1(0.31);JAK2(0.31)	3
Reactome_2022__G0 And Early G1 R-HSA-1538133	2.06	0.0004363135950409	False		TOP2A(0.59)	1
