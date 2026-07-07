# PACKET: ME_CFS (Gardella)

## meta
```json
{
 "phenotype": "ME_CFS (Gardella)",
 "ot_disease": "myalgic encephalomeyelitis/chronic fatigue syndrome",
 "n_sig_with_rare": 228,
 "n_novel": 91,
 "n_rare_led": 4,
 "only_nbi_sig": 149,
 "right_only": 91,
 "left_only": 12,
 "jaccard": 0.571,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
KEGG_2021_Human__Autophagy	-1.88	True	INS	PRKAA1(0.06);MAP2K1(0.04);MAPK1(0.04);INS(0.03);ATG13(0.03);MAPK3(0.03)
GO_Biological_Process_2023__Regulation Of Lipid Storage (GO:0010883)	-1.86	True	PLA2G10	FBXW7(0.30);TNF(0.08);NFKB1(0.03)
GO_Biological_Process_2023__Positive Regulation Of Autophagy (GO:0010508)	-1.8	True	LACRT	PRKAA1(0.06);IL4(0.04);ATG13(0.03)
GO_Biological_Process_2023__Positive Regulation Of Cell Differentiation (GO:0045597)	-1.79	True	PLA2G10;INS	TCF4(0.29);ADRA2B(0.10);ADRA2C(0.10);IL1B(0.07);IGF1(0.05);MAPK14(0.04);INS(0.03);SMAD4(0.03);NFKB1(0.03)

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-2.6	0.0	False			0
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-2.54	0.0	False			0
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-2.53	0.0	False			0
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-2.53	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Enhanced By Exon Junction Complex (EJC) R-HSA-975957	-2.52	0.0	False			0
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-2.52	0.0	False			0
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-2.52	0.0	False			0
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-2.48	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-2.46	0.0	False			0
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-2.44	0.0	False			0
Reactome_2022__Viral mRNA Translation R-HSA-192823	-2.43	0.0	False			0
Reactome_2022__Selenocysteine Synthesis R-HSA-2408557	-2.42	0.0	False			0
Reactome_2022__Response Of EIF2AK4 (GCN2) To Amino Acid Deficiency R-HSA-9633012	-2.41	0.0	False			0
KEGG_2021_Human__Coronavirus disease	-2.34	0.0	False		TLR3(0.33);TNF(0.08);IL1B(0.07);CXCL8(0.06);MAPK14(0.04);MAPK1(0.04);TLR4(0.04);MAPK3(0.03);EIF2AK2(0.03)	9
Reactome_2022__mRNA Activation Upon Binding Of Cap-Binding Complex And eIFs, Subsequent Binding To 43S R-HSA-72662	-2.29	0.0	False			0
Reactome_2022__Translation Initiation Complex Formation R-HSA-72649	-2.29	0.0	False			0
Reactome_2022__Selenoamino Acid Metabolism R-HSA-2408522	-2.28	0.0	False			0
GO_Biological_Process_2023__Growth Hormone Receptor Signaling Pathway (GO:0060396)	-2.25	0.0	False			0
GO_Biological_Process_2023__Cellular Response To Glucose Stimulus (GO:0071333)	-2.25	0.0	False		PRKAA1(0.06)	1
GO_Biological_Process_2023__Macromolecule Biosynthetic Process (GO:0009059)	-2.24	0.0006434402988148	False			0
Reactome_2022__Formation Of Ternary Complex, And Subsequently, 43S Complex R-HSA-72695	-2.24	0.0006128002845855	False			0
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-2.23	0.0005849457261952	False			0
Reactome_2022__Ribosomal Scanning And Start Codon Recognition R-HSA-72702	-2.22	0.0005595133033172	False			0
GO_Biological_Process_2023__Peptidyl-Threonine Modification (GO:0018210)	-2.22	0.0005362002490123	False		MAPK1(0.04)	1
GO_Biological_Process_2023__Peptidyl-Threonine Phosphorylation (GO:0018107)	-2.22	0.0005147522390518	False		MAPK1(0.04)	1
KEGG_2021_Human__FoxO signaling pathway	-2.21	0.0014848622280341	False		PRKAA1(0.06);IGF1(0.05);MAPK14(0.04);MAPK1(0.04);SMAD4(0.03);MAPK3(0.03)	6
KEGG_2021_Human__Ribosome	-2.19	0.0033363571049657	False			0
Reactome_2022__Regulation Of Expression Of SLITs And ROBOs R-HSA-9010553	-2.16	0.0036768017075132	False			0
KEGG_2021_Human__Growth hormone synthesis, secretion and action	-2.16	0.0035500154417369	False		CACNA1F(0.07);IGF1(0.05);MAPK14(0.04);MAPK1(0.04);MAPK3(0.03)	5
GO_Biological_Process_2023__Peptide Biosynthetic Process (GO:0043043)	-2.15	0.003431681593679	False			0
Reactome_2022__Influenza Infection R-HSA-168255	-2.14	0.0033209821874313	False		EIF2AK2(0.03)	1
Reactome_2022__Signaling By ROBO Receptors R-HSA-376176	-2.12	0.0044236520543518	False			0
GO_Biological_Process_2023__Gene Expression (GO:0010467)	-2.12	0.0042896019920987	False			0
Reactome_2022__Extra-nuclear Estrogen Signaling R-HSA-9009391	-2.12	0.0041634372276252	False		MAPK1(0.04);MAPK3(0.03)	2
Reactome_2022__Estrogen-dependent Nuclear Events Downstream Of ESR-membrane Signaling R-HSA-9634638	-2.11	0.0047798422197671	False		MAPK1(0.04);MAPK3(0.03)	2
Reactome_2022__Signaling By Non-Receptor Tyrosine Kinases R-HSA-9006927	-2.1	0.0053620024901234	False		NR3C1(0.05)	1
Reactome_2022__Apoptosis Induced DNA Fragmentation R-HSA-140342	-2.1	0.0052170835039039	False			0
GO_Biological_Process_2023__Central Nervous System Neuron Axonogenesis (GO:0021955)	-2.09	0.0050797918327485	False			0
GO_Biological_Process_2023__Peptidyl-Serine Modification (GO:0018209)	-2.09	0.0059394489121367	False		MAPK14(0.04);MAPK1(0.04)	2
Reactome_2022__Post NMDA Receptor Activation Events R-HSA-438064	-2.07	0.0070778432869629	False		PRKAA1(0.06);GRIN2B(0.06);GRIN2A(0.06);GRIN2C(0.06);MAPK1(0.04);MAPK3(0.03)	6
GO_Biological_Process_2023__Peptidyl-Serine Phosphorylation (GO:0018105)	-2.05	0.0078468329123757	False		MAPK14(0.04);MAPK1(0.04)	2
KEGG_2021_Human__Neutrophil extracellular trap formation	-2.05	0.0076600035573192	False		MPO(0.07);MAPK14(0.04);MAPK1(0.04)	3
Reactome_2022__Apoptotic Execution Phase R-HSA-75153	-2.04	0.0086789621700603	False		TJP1(0.05)	1
Reactome_2022__Cellular Response To Starvation R-HSA-9711097	-2.04	0.0084817130298316	False			0
KEGG_2021_Human__ErbB signaling pathway	-2.04	0.0082932305180576	False		MAP2K1(0.04);MAPK1(0.04);MAPK3(0.03)	3
GO_Biological_Process_2023__Regulation Of Endothelial Cell Migration (GO:0010594)	-2.03	0.0095117261563929	False		EDN1(0.07)	1
Reactome_2022__Interleukin-6 Signaling R-HSA-1059683	-2.03	0.0093093490041292	True			0
Reactome_2022__RHO GTPases Activate NADPH Oxidases R-HSA-5668599	-2.03	0.009383504357716	False		MAPK14(0.04);MAPK1(0.04);MAPK3(0.03)	3
Reactome_2022__Influenza Viral RNA Transcription And Replication R-HSA-168273	-2.03	0.0094546329621768	False			0
Reactome_2022__Activation Of NMDA Receptors And Postsynaptic Events R-HSA-442755	-2.01	0.0110671731396148	False		PRKAA1(0.06);GRIN2B(0.06);GRIN2A(0.06);GRIN2C(0.06);MAPK1(0.04);MAPK3(0.03)	6
Reactome_2022__Signaling To ERKs R-HSA-187687	-2.01	0.0116071583315613	False		MAPK14(0.04);MAP2K1(0.04);MAPK1(0.04);MAPK3(0.03)	4
GO_Biological_Process_2023__Phosphatidylinositol 3-Kinase Signaling (GO:0014065)	-2.01	0.0121263748622792	False		EDN1(0.07);IGF1(0.05)	2
Reactome_2022__Attenuation Phase R-HSA-3371568	-2.01	0.0118975753365758	False			0
GO_Biological_Process_2023__Epidermal Growth Factor Receptor Signaling Pathway (GO:0007173)	-2.0	0.0133454284198628	True			0
Reactome_2022__ESR-mediated Signaling R-HSA-8939211	-2.0	0.0135707408477306	False		MAPK1(0.04);MAPK3(0.03)	2
Reactome_2022__HSF1-dependent Transactivation R-HSA-3371571	-1.99	0.0142476066166137	False			0
Reactome_2022__Signaling By CSF3 (G-CSF) R-HSA-9674555	-1.99	0.0146749541834957	True			0
Reactome_2022__Platelet Sensitization By LDL R-HSA-432142	-1.99	0.0144219377320562	False		MAPK14(0.04)	1
Reactome_2022__Negative Regulation Of FGFR2 Signaling R-HSA-5654727	-1.99	0.0141774981094789	False		MAPK1(0.04);MAPK3(0.03)	2
GO_Biological_Process_2023__Response To UV-A (GO:0070141)	-1.98	0.0143701666735308	False			0
