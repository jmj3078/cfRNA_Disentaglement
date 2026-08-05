# Pancreatic Cancer Pathway-Convergence Literature Review

Independent re-verification (paper-lookup skill, PubMed E-utilities) of the 8 selected pathways +
1 rejected pathway from the earlier conversational pathway-convergence analysis. All citations below
were confirmed by direct PubMed query on the date of this review; none are recalled from memory.

## Selected pathways

### 1. Extracellular Matrix Organization
PDAC's defining desmoplastic stroma (dense collagen/hyaluronan matrix) is a stronger determinant of
drug delivery and disease behavior in PDAC than in most solid tumors, making ECM-organization signal
biologically PDAC-specific rather than a generic cancer readout.

- Provenzano PP, Cuevas C, Chang AE, et al., *Cancer Cell*, 2012 (PMID: 22439937; DOI: 10.1016/j.ccr.2012.01.007) -- enzymatic ablation of tumor hyaluronan (PEGPH20) collapses interstitial pressure and re-opens collapsed tumor vasculature specifically in PDAC models.
- Original recall matched exactly (author, journal, year, topic all confirmed).

### 2. Neutrophil Extracellular Trap Formation
NETs are mechanistically implicated in PDAC-specific tumor-stroma crosstalk and hypercoagulability,
distinct from their more generic role in inflammation.

- Miller-Ocuin JL, Vitola A, et al. (review authorship confirmed under this search), *Cancers*, 2022 (PMID: 35884400) -- "Neutrophil Extracellular Traps and Pancreatic Cancer Development: A Vicious Cycle," review of NET-driven PDAC progression.
- Primary mechanistic paper also found and cross-checked: Miller-Ocuin JL, Liang X, Boone BA, et al., *Oncoimmunology*, 2019 (PMID: 31428515; DOI: 10.1080/2162402X.2019.1605822) -- NET-released DNA activates pancreatic stellate cells and enhances PDAC tumor growth.
- Original recall ("Miller-Ocuin et al. or a review, Cancers 2022") matched exactly on the review; confirmed correct.

### 3. Regulation Of IGF Transport And Uptake By IGFBPs
IGF axis dysregulation is specifically implicated in PDAC risk in prospective serum studies, distinct
from general growth-factor signaling.

- Gong Y, Zhang B, Liao Y, et al., *Nutrients*, 2017 (PMID: 28420208; DOI: 10.3390/nu9040394) -- "Serum Insulin-Like Growth Factor Axis and the Risk of Pancreatic Cancer: Systematic Review and Meta-Analysis."
- Original recall matched exactly.

### 4. Platelet Degranulation
PDAC has one of the highest cancer-associated-thrombosis rates of any solid tumor, and platelet
activation/degranulation is a documented PDAC-specific hypercoagulability mechanism (podoplanin/mucin-driven).

- Willems RAL, Biesmans C, Campello E, et al., *Seminars in Thrombosis and Hemostasis*, 2024 (PMID: 38049115; DOI: 10.1055/s-0043-1777304) -- "Cellular Components Contributing to the Development of Venous Thrombosis in Patients with Pancreatic Cancer": PDAC-specific narrative review covering platelet activation/aggregation (via podoplanin, mucins) alongside NETs and microvesicles in PDAC hypercoagulability.
- Note: the original recall ("2024 Seminars in Thrombosis and Hemostasis PDAC-specific review") was directionally correct but imprecise -- the actual 2024 issue of that journal contains several cancer-thrombosis reviews; this is the one that is PDAC-specific (title centers on venous thrombosis/cellular components generally, not literally "platelet degranulation," but platelets are a core focus within it). No paper titled specifically around "platelet degranulation in PDAC" was found in this journal/year; treat this as the closest genuine match, not a verbatim confirmation.

### 5. Stabilization Of P53
TP53 is one of the most frequently mutated genes in PDAC and defines the poor-prognosis "squamous"
molecular subtype, making p53-pathway signal disease-relevant beyond a generic tumor-suppressor readout.

- Bailey P, Chang DK, Nones K, et al., *Nature*, 2016 (PMID: 26909576; DOI: 10.1038/nature16965) -- "Genomic analyses identify molecular subtypes of pancreatic cancer": integrated genomic analysis of 456 PDACs; abstract confirms squamous subtype is "enriched for TP53 and KDM6A mutations" with poor prognosis. Note: the abstract's headline pathway list is KRAS/TGF-β/WNT/NOTCH/etc. (10 pathways from 32 recurrently mutated genes) -- the canonical "4 driver genes" framing (KRAS/TP53/CDKN2A/SMAD4) is well-established PDAC literature but is not the specific framing emphasized in this paper's abstract; it derives more directly from earlier work (e.g., Jones et al. 2008, Biankin et al. 2012). TP53's centrality to PDAC is nonetheless directly confirmed here.
- Original recall matched exactly on citation identity; the "four canonical driver genes" framing is accurate PDAC domain knowledge but not this paper's own headline claim.

### 6. Thermogenesis
PDAC cachexia includes a distinctive hypothermia/thermogenic-dysregulation phenotype mechanistically
linked to tumor-secreted Lcn2, differentiating it from generic cancer cachexia.

- Lemecha M, Chalise JP, Takamuku Y, et al., *Molecular Metabolism*, 2022 (PMID: 36243318; DOI: 10.1016/j.molmet.2022.101612) -- "Lcn2 mediates adipocyte-muscle-tumor communication and hypothermia in pancreatic cancer cachexia."
- Original recall matched exactly.

### 7. TCR Signaling
T-cell clonal expansion and localization within PDAC tumors has been directly characterized (not
inferred from pan-cancer TIL atlases), supporting PDAC-specific adaptive immune signal.

- Stromnes IM, Hulbert A, Pierce RH, et al., *Cancer Immunology Research*, 2017 (PMID: 29066497; DOI: 10.1158/2326-6066.CIR-16-0322) -- "T-cell Localization, Activation, and Clonal Expansion in Human Pancreatic Ductal Adenocarcinoma."
- Original recall matched exactly.

### 8. Signaling By B Cell Receptor (BCR)
Mature tertiary lymphoid structures (TLS) with functional B-cell compartments are a documented,
clinically relevant PDAC-specific immunotherapy-response niche.

- Kinker GS, Vitiello GAF, Diniz AB, et al., *Gut*, 2023 (PMID: 37230755; DOI: 10.1136/gutjnl-2022-328697) -- "Mature tertiary lymphoid structures are key niches of tumour-specific immune responses in pancreatic ductal adenocarcinomas": scRNA-seq/spatial characterization showing mature PDAC TLSs support B-cell proliferation/plasma-cell differentiation and tumor-reactive T-cell activity; mature-TLS gene signature associates with longer chemoimmunotherapy survival.
- Original recall matched exactly. Note: the paper's direct focus is TLS B/T-cell compartmentalization broadly, not "BCR signaling" as a named pathway per se -- treat the BCR-pathway link as the rationale's own inference from B-cell activity within TLS, which this paper supports but does not use that exact terminology for.

## Considered and rejected

### Systemic lupus erythematosus (SLE)
**Rejected** based on direct gene-set composition inspection: 57% of the KEGG "Systemic lupus
erythematosus" term's member genes are core histone genes (H1/H2A/H2B/H3/H4 family) -- a known
autoantigen-driven composition artifact of that specific KEGG term, not real PDAC biology. This
rejection reason is independent of literature and was not re-litigated here.

**Epidemiological claim re-verified**: CONFIRMED. Seo MS, Yeo J, Hwang IC, Shim JY, *Clinical
Rheumatology*, 2019 (PMID: 31270697; DOI: 10.1007/s10067-019-04660-9) -- "Risk of pancreatic cancer in
patients with systemic lupus erythematosus: a meta-analysis" (11 cohort studies; HR = 1.42, 95% CI
1.32-1.53, increased pancreatic cancer risk in SLE patients). Original recall (journal, year, direction
of association) matched exactly. The epidemiological premise is sound; the pathway itself was correctly
rejected for the separate, unrelated reason of KEGG gene-set composition bias.

## Raw search log

All PMIDs surfaced during PubMed searches for the 9 claims (best matches used above are marked *):

1. ECM/Provenzano: 22439937* (Cancer Cell 2012), 23299539 (Br J Cancer 2013, Provenzano/Hingorani, hyaluronan/fluid pressure), 31167451 (Cancers 2019, Maloney/Provenzano, stromal biophysics monitoring), 27166818 (Biophys J 2016, DuFort, interstitial pressure), 25026210 (Cancer Cell 2014, DelGiorno, response letter on interstitial pressure)
2. NETs: 35884400* (Cancers 2022 review), 31428515* (Oncoimmunology 2019, primary), 29929491 (BMC Cancer 2018, Boone/Miller-Ocuin, chloroquine/NETs/hypercoagulability), 42449662 (Cancers 2026, NET/hypercoagulability survival), 42449604 (Cancers 2026, NET/CD8 review), 34503307 (Cancers 2021, NETs in cancer metastasis, not PDAC-specific)
3. IGF/IGFBP: 28420208* (Nutrients 2017, only hit)
4. Platelet/thrombosis: 38049115* (Semin Thromb Hemost 2024, PDAC-specific), 38604227 (Semin Thromb Hemost 2024, ML for thrombosis prediction, not PDAC-specific), 31430786 (2019, microvesicles/cancer thrombosis, general), 34116580 (2021, splanchnic vein thrombosis, general), 33636745 (2021, cancer-thrombosis management, general)
5. TP53/Bailey: 26909576* (Nature 2016, exact match), 25719666 (Nature 2015, Waddell, whole-genome landscape, related but distinct paper)
6. Lcn2/thermogenesis: 36243318* (Mol Metab 2022, exact match), 39353236 (Transl Oncol 2024, GDF15/LCN2 biomarker), 37006254 (Front Immunol 2023, Lcn2/neutrophil activation cachexia), 38685046, 35190301, 34245812, 33824339 (related Lcn2/cachexia/appetite papers, not thermogenesis-specific)
7. TCR/Stromnes: 29066497* (Cancer Immunol Res 2017, exact match), 40072469 (Cancer Immunol Res 2025, unrelated IL-15/CD40 study)
8. TLS/BCR/Gut: 37230755* (Gut 2023, only hit, exact match)
9. SLE/pancreatic risk: 31270697* (Clin Rheumatol 2019, exact match), 35600353 (Front Oncol 2022, SLE/cancer cohort review), 40823502 (EClinicalMedicine 2025, autoimmune/digestive cancer meta-analysis), 30522515 (Arthritis Res Ther 2018, SLE/cancer risk meta-analysis), 34887857 (Front Immunol 2021, autoimmune/gastric cancer), 40322556 (J Immunol Res 2025, Mendelian randomization autoimmune/pan-cancer)
