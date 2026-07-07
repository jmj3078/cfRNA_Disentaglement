# PACKET: ICI-treated Cancer (Raissadati)

## meta
```json
{
 "phenotype": "ICI-treated Cancer (Raissadati)",
 "ot_disease": null,
 "n_sig_with_rare": 1395,
 "n_novel": 172,
 "n_rare_led": 3,
 "only_nbi_sig": 1278,
 "right_only": 172,
 "left_only": 55,
 "jaccard": 0.843,
 "sign_agree": 1.0
}
```

## rare-led NOVEL pathways (rare-branch gene directly in leading edge; strongest rare-attributable evidence)

Term	NES	novel	rare_lead	db_support
GO_Biological_Process_2023__Detection Of Chemical Stimulus Involved In Sensory Perception Of Taste (GO:0050912)	1.82	True	TAS2R42	
GO_Biological_Process_2023__Detection Of Chemical Stimulus Involved In Sensory Perception Of Smell (GO:0050911)	1.75	True	OR13F1;OR6K2;OR2K2;OR10G2;OR10H3;OR2S2	
GO_Biological_Process_2023__Detection Of Chemical Stimulus Involved In Sensory Perception (GO:0050907)	1.74	True	OR13F1;OR6K2;OR2K2;OR10G2;OR10H3;OR2S2	

## top significant pathways by |NES| (novel flag = new vs only_nbi)

Term	NES	FDR	novel	rare_lead	db_support	n_db
KEGG_2021_Human__Lysosome	2.97	0.0	False			0
Reactome_2022__Eukaryotic Translation Elongation R-HSA-156842	-2.9	0.0	False			0
GO_Biological_Process_2023__Cytoplasmic Translation (GO:0002181)	-2.86	0.0	False			0
Reactome_2022__Peptide Chain Elongation R-HSA-156902	-2.82	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Enhanced By Exon Junction Complex (EJC) R-HSA-975957	-2.81	0.0	False			0
Reactome_2022__Formation Of A Pool Of Free 40S Subunits R-HSA-72689	-2.8	0.0	False			0
Reactome_2022__Nonsense Mediated Decay (NMD) Independent Of Exon Junction Complex (EJC) R-HSA-975956	-2.8	0.0	False			0
Reactome_2022__Viral mRNA Translation R-HSA-192823	-2.8	0.0	False			0
Reactome_2022__Eukaryotic Translation Termination R-HSA-72764	-2.78	0.0	False			0
Reactome_2022__Selenocysteine Synthesis R-HSA-2408557	-2.78	0.0	False			0
Reactome_2022__L13a-mediated Translational Silencing Of Ceruloplasmin Expression R-HSA-156827	-2.76	0.0	False			0
Reactome_2022__GTP Hydrolysis And Joining Of 60S Ribosomal Subunit R-HSA-72706	-2.74	0.0	False			0
Reactome_2022__Response Of EIF2AK4 (GCN2) To Amino Acid Deficiency R-HSA-9633012	-2.74	0.0	False			0
Reactome_2022__Selenoamino Acid Metabolism R-HSA-2408522	-2.73	0.0	False			0
KEGG_2021_Human__Ribosome	-2.72	0.0	False			0
KEGG_2021_Human__Other glycan degradation	2.68	0.0	False			0
Reactome_2022__Influenza Viral RNA Transcription And Replication R-HSA-168273	-2.66	0.0	False			0
Reactome_2022__Cap-dependent Translation Initiation R-HSA-72737	-2.64	0.0	False			0
Reactome_2022__Influenza Infection R-HSA-168255	-2.59	0.0	False			0
Reactome_2022__Formation Of Ternary Complex, And Subsequently, 43S Complex R-HSA-72695	-2.58	0.0	False			0
GO_Biological_Process_2023__Macromolecule Biosynthetic Process (GO:0009059)	-2.56	0.0	False			0
Reactome_2022__Ribosomal Scanning And Start Codon Recognition R-HSA-72702	-2.56	0.0	False			0
Reactome_2022__Regulation Of Expression Of SLITs And ROBOs R-HSA-9010553	-2.56	0.0	False			0
GO_Biological_Process_2023__Peptide Biosynthetic Process (GO:0043043)	-2.55	0.0	False			0
Reactome_2022__Major Pathway Of rRNA Processing In Nucleolus And Cytosol R-HSA-6791226	-2.53	0.0	False			0
Reactome_2022__Signaling By ROBO Receptors R-HSA-376176	-2.52	0.0	False			0
Reactome_2022__Translation Initiation Complex Formation R-HSA-72649	-2.51	0.0	False			0
Reactome_2022__Cellular Response To Starvation R-HSA-9711097	-2.5	0.0	False			0
Reactome_2022__mRNA Activation Upon Binding Of Cap-Binding Complex And eIFs, Subsequent Binding To 43S R-HSA-72662	-2.5	0.0	False			0
GO_Biological_Process_2023__Vacuolar Acidification (GO:0007035)	2.5	0.0	False			0
Reactome_2022__rRNA Processing In Nucleus And Cytosol R-HSA-8868773	-2.47	0.0	False			0
GO_Biological_Process_2023__Translation (GO:0006412)	-2.47	0.0	False			0
GO_Biological_Process_2023__Lysosomal Lumen Acidification (GO:0007042)	2.46	0.0	False			0
GO_Biological_Process_2023__Regulation Of Lysosomal Lumen pH (GO:0035751)	2.45	0.0	False			0
Reactome_2022__rRNA Processing R-HSA-72312	-2.45	0.0	False			0
Reactome_2022__Cellular Senescence R-HSA-2559583	-2.44	0.0	False			0
KEGG_2021_Human__Allograft rejection	2.42	0.0	False			0
Reactome_2022__Immunoregulatory Interactions Between A Lymphoid And A non-Lymphoid Cell R-HSA-198933	2.41	0.0	False			0
Reactome_2022__SARS-CoV-2 Modulates Host Translation Machinery R-HSA-9754678	-2.41	0.0	False			0
Reactome_2022__Neutrophil Degranulation R-HSA-6798695	2.4	0.0	False			0
KEGG_2021_Human__Cell adhesion molecules	2.4	0.0	False			0
Reactome_2022__SRP-dependent Cotranslational Protein Targeting To Membrane R-HSA-1799339	-2.39	0.0	False			0
GO_Biological_Process_2023__Intracellular pH Reduction (GO:0051452)	2.36	0.0	False			0
GO_Biological_Process_2023__Positive Regulation Of T Cell Mediated Cytotoxicity (GO:0001916)	2.35	0.0	False			0
GO_Biological_Process_2023__Innate Immune Response-Activating Signaling Pathway (GO:0002758)	2.34	0.0	False			0
Reactome_2022__Oxidative Stress Induced Senescence R-HSA-2559580	-2.34	0.0	False			0
KEGG_2021_Human__Coronavirus disease	-2.34	0.0	False			0
Reactome_2022__PKMTs Methylate Histone Lysines R-HSA-3214841	-2.32	0.0	False			0
Reactome_2022__CDC42 GTPase Cycle R-HSA-9013148	-2.31	0.0	False			0
Reactome_2022__Activation Of HOX Genes During Differentiation R-HSA-5619507	-2.31	0.0	False			0
KEGG_2021_Human__Antigen processing and presentation	2.31	0.0	False			0
Reactome_2022__Deactivation Of Beta-Catenin Transactivating Complex R-HSA-3769402	-2.3	0.0	False			0
KEGG_2021_Human__Graft-versus-host disease	2.3	0.0	False			0
Reactome_2022__RHO GTPases Activate PKNs R-HSA-5625740	-2.3	0.0	False			0
Reactome_2022__Chromatin Modifying Enzymes R-HSA-3247509	-2.3	0.0	False			0
Reactome_2022__Cell Cycle Checkpoints R-HSA-69620	-2.3	0.0	False			0
GO_Biological_Process_2023__Protein Processing (GO:0016485)	2.29	0.0	False			0
KEGG_2021_Human__Pantothenate and CoA biosynthesis	2.29	0.0	False			0
GO_Biological_Process_2023__Negative Regulation Of T Cell Proliferation (GO:0042130)	2.28	0.0	False			0
Reactome_2022__RHOJ GTPase Cycle R-HSA-9013409	-2.28	0.0	False			0
