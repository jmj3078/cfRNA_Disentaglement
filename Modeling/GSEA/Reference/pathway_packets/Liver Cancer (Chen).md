# PACKET: Liver Cancer (Chen)

## meta
```json
{
 "phenotype": "Liver Cancer (Chen)",
 "ot_disease": "hepatocellular carcinoma",
 "n_sig_with_rare": 706,
 "n_novel": 102,
 "n_rare_led": 3,
 "only_nbi_sig": 642,
 "right_only": 102,
 "left_only": 38,
 "jaccard": 0.812,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
KEGG_2021_Human__Pentose and glucuronate interconversions	1.77	True	UGT1A8	
GO_Biological_Process_2023__Positive Regulation Of Defense Response (GO:0031349)	-1.72	True	CCL1	CEBPA(0.33)
GO_Biological_Process_2023__DNA Repair (GO:0006281)	-1.69	True	TNP1	RRM2B(0.42);POLD3(0.41);ATM(0.36);MSH6(0.36);NBN(0.35);MSH2(0.35);BRCA1(0.35);ERCC5(0.35);NPM1(0.32)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-3.29	0.0	False		RPL22(0.36)	1
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-3.29	0.0	False		RPL22(0.36)	1
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-3.22	0.0	False		RPL22(0.36)	1
KEGG_2021_Human__Ribosome	-3.21	0.0	False		RPL22(0.36)	1
GO_Biological_Process_2023__Translation (GO:0006412)	-3.2	0.0	False		RPL22(0.36)	1
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-3.19	0.0	False		RPL22(0.36)	1
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-3.18	0.0	False		RPL22(0.36)	1
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-3.17	0.0	False		RPL22(0.36)	1
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-3.17	0.0	False		RPL22(0.36)	1
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-3.16	0.0	False		RPL22(0.36)	1
Reactome_2022__Selenocysteine Synthesis R-HSA-2408557	-3.16	0.0	False		RPL22(0.36)	1
Reactome_2022__rRNA Processing R-HSA-72312	-3.14	0.0	False		RPL22(0.36)	1
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-3.14	0.0	False		RPL22(0.36)	1
Reactome_2022__rRNA Processing In Nucleus And Cytosol R-HSA-8868773	-3.09	0.0	False		RPL22(0.36)	1
Reactome_2022__Response Of EIF2AK4 (GCN2) To Amino Acid Deficiency R-HSA-9633012	-3.07	0.0	False		RPL22(0.36)	1
Reactome_2022__Viral mRNA Translation R-HSA-192823	-3.06	0.0	False		RPL22(0.36)	1
Reactome_2022__Selenoamino Acid Metabolism R-HSA-2408522	-3.04	0.0	False		RPL22(0.36)	1
Reactome_2022__Major Pathway Of rRNA Processing In Nucleolus And Cytosol R-HSA-6791226	-3.03	0.0	False		RPL22(0.36)	1
Reactome_2022__Nonsense Mediated Decay (NMD) Enhanced By Exon Junction Complex (EJC) R-HSA-975957	-3.03	0.0	False		RPL22(0.36)	1
GO_Biological_Process_2023__Peptide Biosynthetic Process (GO:0043043)	-2.89	0.0	False		RPL22(0.36)	1
Reactome_2022__Formation Of Ternary Complex, And Subsequently, 43S Complex R-HSA-72695	-2.85	0.0	False			0
Reactome_2022__Translation R-HSA-72766	-2.78	0.0	False		RPL22(0.36)	1
GO_Biological_Process_2023__Macromolecule Biosynthetic Process (GO:0009059)	-2.77	0.0	False		TYMS(0.56);POLD3(0.41);RPL22(0.36)	3
Reactome_2022__mRNA Activation Upon Binding Of Cap-Binding Complex And eIFs, Subsequent Binding To 43S R-HSA-72662	-2.77	0.0	False			0
Reactome_2022__Regulation Of Expression Of SLITs And ROBOs R-HSA-9010553	-2.74	0.0	False		RBX1(0.38);RPL22(0.36)	2
GO_Biological_Process_2023__Ribosomal Small Subunit Biogenesis (GO:0042274)	-2.73	0.0	False		NPM1(0.32)	1
Reactome_2022__Translation Initiation Complex Formation R-HSA-72649	-2.72	0.0	False			0
Reactome_2022__Ribosomal Scanning And Start Codon Recognition R-HSA-72702	-2.71	0.0	False			0
Reactome_2022__Influenza Viral RNA Transcription And Replication R-HSA-168273	-2.69	0.0	False		RPL22(0.36)	1
Reactome_2022__Influenza Infection R-HSA-168255	-2.66	0.0	False		RPL22(0.36)	1
Reactome_2022__Nucleotide-binding Domain, Leucine Rich Repeat Containing NLR Signaling Pathways R-HSA-168643	-2.65	0.0	False		CASP8(0.43)	1
Reactome_2022__Negative Regulation Of NOTCH4 Signaling R-HSA-9604323	-2.65	0.0	False		RBX1(0.38)	1
Reactome_2022__FBXL7 Down-Regulates AURKA During Mitotic Entry And In Early Mitosis R-HSA-8854050	-2.64	0.0	False		RBX1(0.38)	1
Reactome_2022__Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By Uncoupling Proteins R-HSA-163200	-2.64	0.0	False			0
Reactome_2022__Regulation Of Apoptosis R-HSA-169911	-2.64	0.0	False			0
GO_Biological_Process_2023__rRNA Processing (GO:0006364)	-2.64	0.0	False		DDX10(0.32)	1
Reactome_2022__Cellular Response To Starvation R-HSA-9711097	-2.63	0.0	False		RPL22(0.36);MTOR(0.32)	2
Reactome_2022__AUF1 (hnRNP D0) Binds And Destabilizes mRNA R-HSA-450408	-2.63	0.0	False			0
Reactome_2022__Degradation Of AXIN R-HSA-4641257	-2.62	0.0	False		AXIN2(0.36)	1
GO_Biological_Process_2023__Gene Expression (GO:0010467)	-2.62	0.0	False		CASP8(0.43);RPL22(0.36)	2
Reactome_2022__Mitochondrial Translation Elongation R-HSA-5389840	-2.61	0.0	False			0
Reactome_2022__Cristae Formation R-HSA-8949613	-2.61	0.0	False			0
Reactome_2022__Vif-mediated Degradation Of APOBEC3G R-HSA-180585	-2.6	0.0	False		RBX1(0.38)	1
Reactome_2022__Mitochondrial Translation Initiation R-HSA-5368286	-2.6	0.0	False			0
GO_Biological_Process_2023__Ribosome Biogenesis (GO:0042254)	-2.59	0.0	False		DDX10(0.32);NPM1(0.32)	2
Reactome_2022__Mitochondrial Translation R-HSA-5368287	-2.59	0.0	False			0
Reactome_2022__Vpu Mediated Degradation Of CD4 R-HSA-180534	-2.59	0.0	False			0
Reactome_2022__SARS-CoV-2 Modulates Host Translation Machinery R-HSA-9754678	-2.58	0.0	False			0
Reactome_2022__Complex I Biogenesis R-HSA-6799198	-2.58	0.0	False			0
GO_Biological_Process_2023__ncRNA Processing (GO:0034470)	-2.57	0.0	False		DDX10(0.32)	1
Reactome_2022__Regulation Of Activated PAK-2p34 By Proteasome Mediated Degradation R-HSA-211733	-2.56	0.0	False			0
Reactome_2022__SCF-beta-TrCP Mediated Degradation Of Emi1 R-HSA-174113	-2.56	0.0	False			0
Reactome_2022__Downstream TCR Signaling R-HSA-202424	-2.56	0.0	False		PIK3CA(0.68)	1
Reactome_2022__Respiratory Electron Transport R-HSA-611105	-2.56	0.0	False			0
Reactome_2022__GSK3B And BTRC:CUL1-mediated-degradation Of NFE2L2 R-HSA-9762114	-2.56	0.0	False		NFE2L2(0.61);RBX1(0.38)	2
Reactome_2022__Mitochondrial Translation Termination R-HSA-5419276	-2.55	0.0	False			0
KEGG_2021_Human__Proteasome	-2.55	0.0	False			0
Reactome_2022__Cross-presentation Of Soluble Exogenous Antigens (Endosomes) R-HSA-1236978	-2.55	0.0	False			0
GO_Biological_Process_2023__Oxidative Phosphorylation (GO:0006119)	-2.54	0.0	False			0
Reactome_2022__GLI3 Is Processed To GLI3R By Proteasome R-HSA-5610785	-2.54	0.0	False		RBX1(0.38)	1
