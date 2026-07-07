# References — Esophagus Cancer (Chen) [carcinoma of esophagus]
Skill-retrieved evidence only (paper-lookup = PubMed E-utilities; database-lookup = Open Targets GraphQL v4). Retrieved 2026-07-06.

| Gene | Claim / disease | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| MMP3 | rare-led novel (ECM / External Encapsulating Structure Org); matrix metalloproteinase | ~36 (MMP3 AND esophageal cancer); ~2185 (MMP3 AND extracellular matrix) | 39741182 (scRNA-seq and spatial transcriptomics of esophageal squamous cell carcinoma with lymph node metastases, Exp Mol Med 2025); 30969151 | Term db_support NF1(0.47). Most literature-supported rare-led candidate (ECM invasion axis). |
| INS | rare-led novel (Aldosterone-regulated sodium reabsorption; Beta-cell gene expr.); pancreatic insulin | ~107 (INS AND esophageal cancer; largely non-specific) | (none esophagus-specific) | Beta-cell restricted; ectopic/low-expression artifact — recommend exclusion. |
| CCL26 | rare-led novel (Positive Reg. Endothelial Cell Proliferation); eotaxin-3, type-2 inflammation | ~96 (CCL26 AND cancer) | 34037993 (Dupilumab suppresses type 2 inflammatory biomarkers across atopic/allergic diseases, Clin Exp Allergy 2021) | Term db_support KDR(0.49)/AKT1(0.40)/ARNT(0.38). Weak candidate (eosinophilic inflammation context). |
| HTR3D | rare-led novel (Anterograde Trans-Synaptic Signaling); serotonin receptor subunit | ~4 (HTR3D AND cancer) | (none) | Neuronal-restricted low expression. Artifact caution. |
| AVP | rare-led novel (Water Transport); arginine vasopressin | ~465 (AVP AND cancer; mostly SIADH/ectopic) | (none esophagus-specific) | Hypothalamus-restricted. Artifact caution. |
| IFNA1 | rare-led novel (SARS-CoV-2-host Interactions, NES -1.61); interferon alpha 1 | (not separately counted) | n/a | Term db_support IKBKB(0.39). Conservative candidate. |
| FGFR2 | established (FGFR Downstream Signaling, non-novel) | ~81 (FGFR2 AND esophageal cancer) | 36441501 (Futibatinib: First Approval, Drugs 2022); 34224333 | Packet db_support FGFR2 0.51; OT (gastric 0.743, colorectal 0.731) confirms strong literature/genetic evidence. Established RTK axis. |

Notes:
- Highest artifact burden of the 5 cohorts: 4/6 rare leads (INS, HTR3D, AVP, partly CCL26) are tissue-restricted low-expression genes.
- MMP3 is the only rare-led with real esophageal-cancer literature support.
- Robust NON-novel axes: FGFR-PI3K signaling, cAMP, YAP1/WWTR1 up; translation/OXPHOS down.
