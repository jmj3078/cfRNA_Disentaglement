# PACKET: Tuberculosis (Chang)

## meta
```json
{
 "phenotype": "Tuberculosis (Chang)",
 "ot_disease": "tuberculosis",
 "n_sig_with_rare": 660,
 "n_novel": 117,
 "n_rare_led": 2,
 "only_nbi_sig": 571,
 "right_only": 117,
 "left_only": 28,
 "jaccard": 0.789,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
GO_Biological_Process_2023__Epithelium Development (GO:0060429)	1.67	True	KRT28	PGK1(0.37);WT1(0.25);TGFB1(0.11);TST(0.11)
Reactome_2022__TCF Dependent Signaling In Response To WNT R-HSA-201681	1.65	True	DKK4	UBC(0.47);UBB(0.46);UBA52(0.46)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
KEGG_2021_Human__Systemic lupus erythematosus	2.55	0.0	False		CTSG(0.37);FCGR3A(0.09);FCGR1A(0.09)	3
KEGG_2021_Human__Neutrophil extracellular trap formation	2.53	0.0	False		MAPK3(0.38);CTSG(0.37);MAPK1(0.37);CASP1(0.10);FCGR3A(0.09);FPR1(0.09);FCGR1A(0.09)	7
Reactome_2022__Interferon Alpha/Beta Signaling R-HSA-909733	2.43	0.0	False		KPNB1(0.50);SOCS3(0.10);HLA-B(0.10)	3
KEGG_2021_Human__Alcoholism	2.43	0.0	False		MAPK3(0.38);MAPK1(0.37)	2
GO_Biological_Process_2023__Glycolytic Process (GO:0006096)	2.42	0.0	False		ENO1(0.37);PGK1(0.37)	2
Reactome_2022__Regulation Of Actin Dynamics For Phagocytic Cup Formation R-HSA-2029482	2.42	0.0	False		MAPK3(0.38);MAPK1(0.37);FCGR1A(0.09)	3
GO_Biological_Process_2023__RNA Splicing, Via Transesterification Reactions With Bulged Adenosine As Nucleophile (GO:0000377)	-2.38	0.0066746820694613	False			0
GO_Biological_Process_2023__mRNA Splicing, Via Spliceosome (GO:0000398)	-2.33	0.0033373410347306	False		SFPQ(0.37)	1
GO_Biological_Process_2023__mRNA Processing (GO:0006397)	-2.32	0.0022248940231537	False		SFPQ(0.37)	1
Reactome_2022__RHO GTPases Activate WASPs And WAVEs R-HSA-5663213	2.3	0.0	False		MAPK3(0.38);MAPK1(0.37)	2
GO_Biological_Process_2023__Pyruvate Metabolic Process (GO:0006090)	2.29	0.0	False		ENO1(0.37);PGK1(0.37)	2
Reactome_2022__FCGR3A-mediated Phagocytosis R-HSA-9664422	2.28	0.0	False		MAPK3(0.38);MAPK1(0.37)	2
Reactome_2022__Interferon Signaling R-HSA-913531	2.27	0.0	False		KPNB1(0.50);UBC(0.47);UBB(0.46);UBA52(0.46);MAPK3(0.38);SOCS3(0.10);HLA-B(0.10);GBP5(0.10);FCGR1A(0.09);GBP1(0.08)	10
Reactome_2022__RHO GTPase Effectors R-HSA-195258	2.27	0.0	False		MAPK3(0.38);MAPK1(0.37);S100A9(0.09)	3
Reactome_2022__Signaling By SCF-KIT R-HSA-1433557	2.26	0.0	False		STAT3(0.11)	1
Reactome_2022__Mitotic G1 Phase And G1/S Transition R-HSA-453279	2.26	0.0	False		UBC(0.47);UBB(0.46);UBA52(0.46)	3
Reactome_2022__Assembly Of Pre-Replicative Complex R-HSA-68867	2.25	0.0	False		KPNB1(0.50);UBC(0.47);UBB(0.46);UBA52(0.46)	4
KEGG_2021_Human__Bacterial invasion of epithelial cells	2.24	0.0	False			0
KEGG_2021_Human__Fc gamma R-mediated phagocytosis	2.24	0.0	False		MAPK3(0.38);MAPK1(0.37);ASAP1(0.34);FCGR3A(0.09);FCGR1A(0.09)	5
GO_Biological_Process_2023__RNA Splicing (GO:0008380)	-2.23	0.0033373410347306	False		SFPQ(0.37)	1
KEGG_2021_Human__Viral carcinogenesis	2.21	0.00063533440625	False		MAPK3(0.38);MAPK1(0.37);STAT3(0.11);HLA-B(0.10)	4
Reactome_2022__Signaling By KIT In Disease R-HSA-9669938	2.21	0.0006727070183823	False		STAT3(0.11)	1
KEGG_2021_Human__Prolactin signaling pathway	2.21	0.0006018957532895	False		MAPK3(0.38);MAPK1(0.37);STAT3(0.11);SOCS3(0.10)	4
Reactome_2022__Cooperation Of Prefoldin And TriC/CCT In Actin And Tubulin Folding R-HSA-389958	2.19	0.000571800965625	False			0
Reactome_2022__RUNX1 Regulates Transcription Of Genes Involved In Differentiation Of HSCs R-HSA-8939236	2.19	0.0005445723482143	False		UBC(0.47);UBB(0.46);UBA52(0.46)	3
GO_Biological_Process_2023__Maturation Of SSU-rRNA (GO:0030490)	-2.18	0.0026698728277845	False			0
Reactome_2022__Interferon Gamma Signaling R-HSA-877300	2.18	0.0005198190596591	False		SOCS3(0.10);HLA-B(0.10);GBP5(0.10);FCGR1A(0.09);GBP1(0.08)	5
KEGG_2021_Human__Spliceosome	-2.17	0.0044497880463075	False			0
KEGG_2021_Human__Pancreatic cancer	2.17	0.0004574407725	False		MAPK3(0.38);MAPK1(0.37);STAT3(0.11);TGFB1(0.11)	4
Reactome_2022__Response Of Mtb To Phagocytosis R-HSA-9637690	2.17	0.0004972182309782	False		KPNB1(0.50);UBC(0.47);UBB(0.46);RAB7A(0.46);UBA52(0.46);CORO1A(0.38);MAPK3(0.38);CTSG(0.37);MAPK1(0.37);GSK3A(0.37)	13
Reactome_2022__Signaling By CSF3 (G-CSF) R-HSA-9674555	2.17	0.0004765008046875	False		UBC(0.47);UBB(0.46);UBA52(0.46);STAT3(0.11);SOCS3(0.10)	5
Reactome_2022__Infection With Mycobacterium Tuberculosis R-HSA-9635486	2.15	0.0004398468966346	False		LTF(0.52);KPNB1(0.50);UBC(0.47);UBB(0.46);RAB7A(0.46);UBA52(0.46);CORO1A(0.38);MAPK3(0.38);CTSG(0.37);MAPK1(0.37)	14
KEGG_2021_Human__Chronic myeloid leukemia	2.15	0.0004084292611607	False		MAPK3(0.38);MAPK1(0.37);TGFB1(0.11)	3
Reactome_2022__Transcriptional Regulation Of Granulopoiesis R-HSA-9616222	2.15	0.0004235562708333	False		STAT3(0.11)	1
Reactome_2022__Fcgamma Receptor (FCGR) Dependent Phagocytosis R-HSA-2029480	2.15	0.00038120064375	False		MAPK3(0.38);MAPK1(0.37);FCGR1A(0.09)	3
Reactome_2022__Paradoxical Activation Of RAF Signaling By Kinase Inactive BRAF R-HSA-6802955	2.15	0.0003943454935345	False		MAPK3(0.38);MAPK1(0.37)	2
Reactome_2022__MAP2K And MAPK Activation R-HSA-5674135	2.14	0.0003465460397727	False		MAPK3(0.38);MAPK1(0.37)	2
Reactome_2022__DNA Replication R-HSA-69306	2.14	0.0003573756035156	False		KPNB1(0.50);UBC(0.47);UBB(0.46);UBA52(0.46)	4
Reactome_2022__RHO GTPases Activate PKNs R-HSA-5625740	2.14	0.0003689038487903	False			0
Reactome_2022__Assembly Of ORC Complex At Origin Of Replication R-HSA-68616	2.13	0.0010090605275735	False		KPNB1(0.50)	1
Reactome_2022__G1/S Transition R-HSA-69206	2.13	0.0013069736357143	False		UBC(0.47);UBB(0.46);UBA52(0.46)	3
Reactome_2022__G2/M Checkpoints R-HSA-69481	2.12	0.0012706688125	False		UBC(0.47);UBB(0.46);UBA52(0.46)	3
GO_Biological_Process_2023__Positive Regulation Of Transcription By RNA Polymerase III (GO:0045945)	-2.12	0.0114423121190765	True			0
Reactome_2022__Activated PKN1 Stimulates Transcription Of Androgen Receptor Regulated KLK2 And KLK3 R-HSA-5625886	2.11	0.00114360193125	False			0
Reactome_2022__ER-Phagosome Pathway R-HSA-1236974	2.11	0.001203791506579	False		UBC(0.47);UBB(0.46);UBA52(0.46);HLA-B(0.10);S100A9(0.09)	5
KEGG_2021_Human__Acute myeloid leukemia	2.11	0.0012363264121622	False		MAPK3(0.38);MAPK1(0.37);STAT3(0.11);FCGR1A(0.09)	4
GO_Biological_Process_2023__Carbohydrate Catabolic Process (GO:0016052)	2.11	0.0011729250576923	False		ENO1(0.37);PGK1(0.37)	2
Reactome_2022__Formation Of Tubulin Folding Intermediates By CCT/TriC R-HSA-389960	2.1	0.0010891446964286	False			0
GO_Biological_Process_2023__Ribonucleoprotein Complex Biogenesis (GO:0022613)	-2.1	0.0133493641389226	True			0
Reactome_2022__RMTs Methylate Histone Arginines R-HSA-3214858	2.1	0.0010396381193182	False			0
Reactome_2022__Signaling To RAS R-HSA-167044	2.1	0.0011157092012195	False			0
Reactome_2022__Antigen processing-Cross Presentation R-HSA-1236975	2.1	0.00106381575	False		UBC(0.47);UBB(0.46);UBA52(0.46);HLA-B(0.10);S100A9(0.09);FCGR1A(0.09)	6
GO_Biological_Process_2023__Regulation Of Macrophage Derived Foam Cell Differentiation (GO:0010743)	2.09	0.0012430455774457	False		CRP(0.12);IL18(0.08)	2
GO_Biological_Process_2023__Spliceosomal Complex Assembly (GO:0000245)	-2.09	0.0140909954799738	False			0
Reactome_2022__DNA Replication Pre-Initiation R-HSA-69002	2.09	0.0012706688125	False		KPNB1(0.50);UBC(0.47);UBB(0.46);UBA52(0.46)	4
KEGG_2021_Human__Mitophagy	2.08	0.0011669407461735	False		UBC(0.47);UBB(0.46);RAB7A(0.46);UBA52(0.46);MFN2(0.21);HIF1A(0.09)	6
Reactome_2022__Senescence-Associated Secretory Phenotype (SASP) R-HSA-2559582	2.08	0.0011211783639706	False		UBC(0.47);UBB(0.46);UBA52(0.46);MAPK3(0.38);MAPK1(0.37);STAT3(0.11)	6
Reactome_2022__Signaling By RAF1 Mutants R-HSA-9656223	2.08	0.00114360193125	False		MAPK3(0.38);MAPK1(0.37)	2
Reactome_2022__Condensation Of Prophase Chromosomes R-HSA-2299718	2.08	0.0012165977992021	False			0
Reactome_2022__DNA Methylation R-HSA-5334118	2.08	0.0011912520117188	False			0
