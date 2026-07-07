# ICI-m (Raissadati) — GSEA with_rare reference (skill-retrieved)

ot_disease: myocarditis. Sources: PubMed E-utilities, Open Targets GraphQL. Retrieved 2026-07-06.

| Gene | Pathway context | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| MYH7 / MYH6 | KEGG Cardiac muscle contraction (up, NES +2.28) | ~13 (MYH7 AND myocarditis) | 41746849 (Genetic testing for primary dilated cardiomyopathy and role of myocardial inflammation) | MYH7: hypertrophic cardiomyopathy 0.892, cardiomyopathy 0.758 (strong). packet MYH7 0.27/MYH6 0.06 for myocarditis. Cardiomyocyte-leak signal, coherent |
| Neutrophil/macrophage set (LGALS3, CD68, ITGAM, MMP9, S100A8/9, TLR2) | Reactome Neutrophil Degranulation (up, NES +3.0); ER-Phagosome / cross-presentation | ~466 (neutrophil AND myocarditis) | 42354881 (cardiovascular sequelae review) | 19 targets db_support (~0.03-0.05 each). Established inflammatory myocarditis axis |
| IFNA1 | KEGG HSV-1 infection (down, NES -1.75, rare lead) | ~16 (interferon AND immune checkpoint inhibitor myocarditis) | 42130414 (Cardiovascular toxicity from ICI: an inflammatory continuum), 42031428 (JAK inhibition in severe irAE) | db_support TNF/IRF7/IFNG/IL1B/CGAS (~0.03-0.10). Interferon axis established; IFNA1 itself low-expression cytokine → axis trusted, gene conservative |
| ATP6V1G3 | KEGG Collecting duct acid secretion (up, NES +1.83, rare lead); OxPhos; ROS in phagocytes; transferrin recycling | 0 (ATP6V1G3 AND myocarditis) | — | OT: NO cardiac/myocarditis association; top assoc type 2 diabetes 0.238 (n_assoc=88 all weak). Kidney collecting-duct restricted, low plasma expression → artifact candidate (possible v-ATPase mis-attribution onto real phagosome axis) |
| Mitochondrial/OXPHOS (axis) | KEGG Oxidative phosphorylation up (nbi) | ~211 (mitochondrial dysfunction AND myocarditis) | (count-level) | axis established; cell-of-origin uncertain in cfRNA |

Notes:
- DB-supported novel = 0; robust conclusions come from nbi (inflammation + cardiac contraction), not rare.
- Context: immune checkpoint inhibitor AND myocarditis ~1465 PubMed hits (well-established entity).
