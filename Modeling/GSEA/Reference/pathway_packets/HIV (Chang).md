# PACKET: HIV (Chang)

## meta
```json
{
 "phenotype": "HIV (Chang)",
 "ot_disease": "HIV infectious disease",
 "n_sig_with_rare": 437,
 "n_novel": 55,
 "n_rare_led": 3,
 "only_nbi_sig": 420,
 "right_only": 55,
 "left_only": 38,
 "jaccard": 0.804,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
KEGG_2021_Human__Leukocyte transendothelial migration	1.75	True	MYL10;CLDN17	RAC1(0.37)
KEGG_2021_Human__Relaxin signaling pathway	1.75	True	INSL5	
Reactome_2022__Interleukin-6 Family Signaling R-HSA-6783589	1.73	True	IL31	

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
Reactome_2022__Interferon Alpha/Beta Signaling R-HSA-909733	2.7	0.0	False		KPNB1(0.56);MX2(0.11)	2
GO_Biological_Process_2023__Negative Regulation Of Viral Process (GO:0048525)	2.7	0.0	False		PPIA(0.62);EIF2AK2(0.52)	2
Reactome_2022__Antiviral Mechanism By IFN-stimulated Genes R-HSA-1169410	2.6	0.0	False		UBC(0.57);UBB(0.57);NUP35(0.56);KPNB1(0.56);NUP98(0.56);NUP153(0.55);NUP214(0.55);NUP205(0.54);NUP58(0.54);NUP54(0.54)	13
GO_Biological_Process_2023__Ribosomal Small Subunit Biogenesis (GO:0042274)	-2.59	0.0	False		NPM1(0.54)	1
GO_Biological_Process_2023__Negative Regulation Of Viral Genome Replication (GO:0045071)	2.57	0.0	False		EIF2AK2(0.52)	1
Reactome_2022__ISG15 Antiviral Mechanism R-HSA-1169408	2.57	0.0	False		UBC(0.57);UBB(0.57);NUP35(0.56);KPNB1(0.56);NUP98(0.56);NUP153(0.55);NUP214(0.55);NUP205(0.54);NUP58(0.54);NUP54(0.54)	13
Reactome_2022__Interferon Signaling R-HSA-913531	2.55	0.0	False		UBC(0.57);UBB(0.57);NUP35(0.56);KPNB1(0.56);NUP98(0.56);NUP153(0.55);NUP214(0.55);NUP205(0.54);NUP58(0.54);NUP54(0.54)	14
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-2.54	0.0	False		RPS27A(0.57)	1
GO_Biological_Process_2023__Regulation Of Viral Genome Replication (GO:0045069)	2.51	0.0	False		PPIA(0.62);EIF2AK2(0.52)	2
GO_Biological_Process_2023__Defense Response To Symbiont (GO:0140546)	2.48	0.0	False		EIF2AK2(0.52);OPRK1(0.32);MX2(0.11)	3
KEGG_2021_Human__Neutrophil extracellular trap formation	2.4	0.0	False	H2AC1		0
Reactome_2022__Assembly Of ORC Complex At Origin Of Replication R-HSA-68616	2.39	0.0	False		KPNB1(0.56)	1
Reactome_2022__Transcriptional Regulation Of Granulopoiesis R-HSA-9616222	2.38	0.0	False			0
Reactome_2022__DNA Methylation R-HSA-5334118	2.36	0.0	False			0
GO_Biological_Process_2023__Defense Response To Virus (GO:0051607)	2.35	0.0	False		EIF2AK2(0.52);OPRK1(0.32);MX2(0.11);CXCL10(0.10)	4
Reactome_2022__RNA Polymerase I Promoter Opening R-HSA-73728	2.35	0.0	False			0
GO_Biological_Process_2023__Antiviral Innate Immune Response (GO:0140374)	2.32	0.0	False		EIF2AK2(0.52);CXCL10(0.10)	2
KEGG_2021_Human__Systemic lupus erythematosus	2.32	0.0	False			0
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-2.32	0.0022062090100672	False		RPS27A(0.57)	1
Reactome_2022__Translation R-HSA-72766	-2.31	0.0016546567575504	False		RPS27A(0.57)	1
Reactome_2022__Oxidative Stress Induced Senescence R-HSA-2559580	2.3	0.0	False		UBC(0.57);UBB(0.57)	2
Reactome_2022__Mitochondrial Translation Termination R-HSA-5419276	-2.3	0.0013237254060403	False			0
Reactome_2022__Condensation Of Prophase Chromosomes R-HSA-2299718	2.3	0.0	False			0
Reactome_2022__DNA Damage/Telomere Stress Induced Senescence R-HSA-2559586	2.29	0.0	False		HMGA1(0.58)	1
KEGG_2021_Human__Alcoholism	2.29	0.0	False			0
Reactome_2022__RHO GTPases Activate PKNs R-HSA-5625740	2.28	0.0	False			0
GO_Biological_Process_2023__Maturation Of SSU-rRNA From Tricistronic rRNA Transcript (SSU-rRNA, 5.8S rRNA, LSU-rRNA) (GO:0000462)	-2.28	0.0033093135151008	False			0
Reactome_2022__Cleavage Of Damaged Purine R-HSA-110331	2.27	0.0	False			0
Reactome_2022__Activated PKN1 Stimulates Transcription Of Androgen Receptor Regulated KLK2 And KLK3 R-HSA-5625886	2.27	0.0	False			0
Reactome_2022__RUNX1 Regulates Genes Involved In Megakaryocyte Differentiation And Platelet Function R-HSA-8936459	2.27	0.0	False			0
Reactome_2022__Packaging Of Telomere Ends R-HSA-171306	2.27	0.0	False			0
Reactome_2022__B-WICH Complex Positively Regulates rRNA Expression R-HSA-5250924	2.26	0.0	False		POLR2F(0.60)	1
Reactome_2022__Pre-NOTCH Transcription And Translation R-HSA-1912408	2.26	0.0	False			0
Reactome_2022__Mitochondrial Translation Initiation R-HSA-5368286	-2.26	0.0037820725886866	False			0
GO_Biological_Process_2023__Response To Interferon-Beta (GO:0035456)	2.25	0.0	False			0
KEGG_2021_Human__Ribosome	-2.23	0.004136641893876	False		RPS27A(0.57)	1
GO_Biological_Process_2023__Respiratory Chain Complex IV Assembly (GO:0008535)	-2.22	0.0036770150167787	False			0
KEGG_2021_Human__Hepatitis C	2.2	0.00038909563669	False		EIF2AK2(0.52);MX2(0.11);CXCL10(0.10)	3
Reactome_2022__Cleavage Of Damaged Pyrimidine R-HSA-110329	2.2	0.0003765441645388	False			0
Reactome_2022__RHO GTPase Effectors R-HSA-195258	2.2	0.0004025127276104	False	NOX3	NUP160(0.58);XPO1(0.56);NUP98(0.56);SEC13(0.54)	4
Reactome_2022__Transcriptional Regulation By Small RNAs R-HSA-5578749	2.18	0.0003647771593969	False		POLR2F(0.60);POLR2G(0.60);NUP160(0.58);NUP35(0.56);NUP98(0.56);NUP153(0.55);NUP214(0.55);NUP205(0.54);SEC13(0.54);NUP58(0.54)	13
GO_Biological_Process_2023__Clathrin-Dependent Endocytosis (GO:0072583)	2.18	0.0003537233060819	False		AP2M1(0.56);AP2B1(0.55);AP2S1(0.55)	3
Reactome_2022__rRNA Processing R-HSA-72312	-2.17	0.0059567643271815	False		RPS27A(0.57)	1
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-2.17	0.0054152402974377	False		RPS27A(0.57)	1
Reactome_2022__Processing Of DNA Double-Strand Break Ends R-HSA-5693607	2.16	0.0006670210914687	False		UBC(0.57);UBB(0.57)	2
Reactome_2022__Meiotic Recombination R-HSA-912446	2.16	0.0006866393588648	False			0
Reactome_2022__Deposition Of New CENPA-containing Nucleosomes At Centromere R-HSA-606279	2.16	0.0006484927278168	False			0
Reactome_2022__Platelet Sensitization By LDL R-HSA-432142	2.16	0.0006309658973352	False			0
Reactome_2022__SIRT1 Negatively Regulates rRNA Expression R-HSA-427359	2.15	0.0006143615316159	False			0
KEGG_2021_Human__Platelet activation	2.15	0.0008541123732221	False		FYN(0.37)	1
Reactome_2022__Mitochondrial Translation Elongation R-HSA-5389840	-2.15	0.0088248360402689	False			0
Reactome_2022__PRC2 Methylates Histones And DNA R-HSA-212300	2.15	0.0005986086718309	False			0
Reactome_2022__Meiotic Synapsis R-HSA-1221632	2.15	0.0005836434550351	False			0
Reactome_2022__RAS Processing R-HSA-9648002	2.15	0.0008337763643359	False			0
Reactome_2022__Base-Excision Repair, AP Site Formation R-HSA-73929	2.15	0.0008143862163281	False			0
KEGG_2021_Human__Chronic myeloid leukemia	2.14	0.0007958774386842	False			0
Reactome_2022__Mitochondrial Translation R-HSA-5368287	-2.14	0.0081460024987098	False			0
Reactome_2022__Defective Pyroptosis R-HSA-9710421	2.13	0.0007781912733801	False			0
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-2.13	0.0083835942382555	False		RPS27A(0.57)	1
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-2.13	0.008509663324545	False		RPS27A(0.57)	1
