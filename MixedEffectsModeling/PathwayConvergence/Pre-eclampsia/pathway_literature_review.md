# Pre-eclampsia — pathway literature review

Candidate list dominated by broad umbrella/[GENERIC] terms and an olfactory-receptor
gene-family cluster; PubMed check on `preeclampsia AND olfactory receptor` returned only
3 hits (38069004, 26574895, 35269385), none establishing a PE-specific olfactory-receptor
mechanism — so the olfactory/sensory-perception candidates are excluded along with the
auto-flagged [GENERIC] terms.

## Selected pathways

### Neutrophil Degranulation
n_sig=17/58, size=463, eff=5.77, hist_frac=0%
Neutrophil activation/degranulation is a long-established, PE-specific maternal
inflammatory feature (distinct from normal pregnancy neutrophil priming), reproduced
across decades of literature.
- Greer, Semin Reprod Endocrinol, 1998 (PMID: 9654608) — foundational review, "The neutrophil and preeclampsia."
- Xie et al., Cell Mol Immunol, 2023 (PMID: 36973485) — IL-32-driven neutrophil activation specifically in preeclampsia.

### Platelet Degranulation
n_sig=8/58, size=123, eff=4.40, hist_frac=0%
Platelet alpha-granule release and thrombo-inflammatory activation are a classic,
clinically used PE biomarker axis (thrombocytopenia, PF4/beta-TG release), not a generic
pregnancy finding.
- Socol et al., Blood, 1984 (PMID: 6230118) — fibrinogen proteolysis and platelet alpha-granule release specifically in preeclampsia/eclampsia.
- Han et al., Res Pract Thromb Haemost, 2023 (PMID: 36923708) — platelet procoagulant membrane dynamics and biomarkers in preeclampsia.
- Ashworth et al., Res Pract Thromb Haemost, 2025 (PMID: 40746439) — increased platelet activation/thrombo-inflammation differentiating early- vs late-onset preeclampsia.

### Th17 Cell Differentiation
n_sig=9/58, size=105, eff=4.00, hist_frac=0%
Th17/Treg imbalance (skewing toward Th17, loss of Treg tolerance) is a specific,
mechanistically studied driver of the maternal immune maladaptation seen in PE.
- Xiong et al., Reprod Sci, 2023 (PMID: 36155892) — PE-derived exosomes imbalance Th17/Treg activity in PBMCs from healthy pregnant women.
- Zhao et al., Int Immunopharmacol, 2025 (PMID: 40763478) — immunological dysregulation (incl. Th17/Treg) in PE pathogenesis.

### NOD-like Receptor Signaling Pathway (NLRP3 inflammasome)
n_sig=7/58, size=178, eff=4.24, hist_frac=0%
NLRP3 inflammasome activation in trophoblast/monocytes by DAMPs (uric acid, cholesterol
crystals) is a specific mechanistic hypothesis for PE-associated systemic inflammation
and hypertension, distinct from generic cytokine signaling.
- Weel et al., Front Endocrinol, 2020 (PMID: 32161574) — role of the NLRP3 inflammasome in preeclampsia.
- Stödle et al., Cells, 2020 (PMID: 32650532) — NLRP3 inflammasome role in pregnancy-induced hypertension and preeclampsia.

### Complex I Biogenesis
n_sig=9/58, size=51, eff=3.88, hist_frac=0%
Placental mitochondrial Complex I dysfunction, driven by hypoxia-induced miR-210
suppression of ISCU, is a specific, well-replicated PE placental mechanism (not generic
oxidative-phosphorylation loss).
- Colleoni et al., Placenta, 2012 (PMID: 22840297) — miR-210 modulates mitochondrial respiration in placenta with preeclampsia.
- Vaka et al., Mol Cell Biochem, 2022 (PMID: 35389182) — impaired mitochondrial respiration in platelets and placentas in preeclamptic pregnancies.
- 2025, Int J Mol Sci (PMID: 40362193) — mitochondrial OXPHOS alterations in early- vs late-onset PE placental tissue.

### Interferon Alpha/Beta Signaling
n_sig=9/58, size=71, eff=3.99, hist_frac=0%
A recent functional study shows type I interferon exposure directly impairs invasive
extravillous trophoblast function, giving a causal (not merely correlative) mechanism
linking this pathway to the trophoblast-invasion defect central to PE.
- 2025, Cell Rep Med (PMID: 40054459) — type I interferon exposure in an implantation-on-a-chip device alters invasive extravillous trophoblast function.
- 2024, bioRxiv preprint (PMID: 38559122) — same finding, preprint version.
- Mekinian et al., Autoimmun Rev, 2019 (PMID: 30772492) — enhanced type I IFN gene signature associated with earlier disease onset and preeclampsia in primary APS.

### RHO GTPase Cycle — weaker-tier
n_sig=8/58, size=439, eff=4.22, hist_frac=0%
Kept despite being a large, mechanistically generic signaling module (adjacent to the
[GENERIC]-flagged "Signaling By Rho GTPases") because RhoA/ROCK is specifically implicated
in PE trophoblast-derived endothelial injury; evidence base is thinner (single strong
mechanistic paper) than the other selections, so treat as lower confidence.
- Cheng et al., Oxid Med Cell Longev, 2022 (PMID: 36160709) — trophoblast exosomal UCA1 induces endothelial injury via PFN1-RhoA/ROCK pathway in preeclampsia.

### Neutrophil Extracellular Trap Formation — weaker-tier
n_sig=8/58, size=183, eff=3.90, hist_frac=40% (fails the hist_frac>15% filter)
Kept despite the histone-gene composition flag because NETosis is one of the most
actively studied PE-specific mechanisms right now (NET-driven endothelial dysfunction),
with multiple independent 2023-2024 papers explicitly targeting NETs in PE; the hist_frac
flag likely reflects genuine NET biology (histone citrullination/release is a defining
feature of NETosis) rather than a pure composition artifact, but is flagged as weaker-tier
per the rules given the elevated histone-gene fraction.
- Espino et al., Front Immunol, 2024 (PMID: 39735539) — circulating EVs and NETs contribute to endothelial dysfunction in preeclampsia.
- Cerdeira et al., Pharmaceuticals, 2024 (PMID: 38794175) — targeting NET formation as a therapeutic strategy in preeclampsia.

## Considered and rejected

- **Osteoclast Differentiation** (n_sig=10, eff=4.43) — only 3 PubMed hits for `preeclampsia AND osteoclast`, mostly generic feto-maternal bone-remodeling physiology (PMID 15979546) or an unrelated ceRNA network paper (PMID 33832098) that just happened to match the keyword; no genuine PE-specific osteoclast mechanism found. Rejected as likely a RANKL/TNF-superfamily gene-overlap artifact with cytokine pathways.
- **NF-kappa B Signaling Pathway** (n_sig=8, eff=4.22) — 257 PubMed hits reflect NF-kB's role as a near-universal downstream inflammatory/apoptotic node in essentially every disease; PE-specific hits found (PMID 34434269 Fas-NF-kB trophoblast apoptosis; PMID 34727826 CASP-3/NF-kB/miRNA) are mechanistically thin and not distinctively PE. Rejected as too broadly implicated to count as disease-specific.
- **Signaling By Interleukins** (n_sig=10, eff=4.51) — broad 449-gene umbrella; top relevance-ranked PubMed hit was an unrelated gestational-diabetes review (PMID 40076938), indicating low specificity of the search term itself. One specific sub-finding exists (IL-17 promoting trophoblast invasion via PPAR-γ/RXR-α/Wnt, PMID 35258399) but that is better captured by narrower, already-selected pathways (Th17, NF-kB-adjacent) than by this generic umbrella. Rejected.
- **Interferon Gamma Signaling** (n_sig=7, eff=4.58) — genuinely PE-relevant (PMID 36289172 IFN-γ role in EVT invasion/PE progression; PMID 25060131 IL-18/IFN-γ meta-analysis in PE) but represents the same underlying interferon-axis signal as the already-selected Interferon Alpha/Beta Signaling; kept the pathway with the stronger single mechanistic (functional, causal) paper rather than double-counting the IFN axis.
- **Response To Elevated Platelet Cytosolic Ca2+** / **Platelet Activation, Signaling And Aggregation** (n_sig=9/7, eff=4.42/4.17) — largely overlapping gene sets with the selected Platelet Degranulation pathway (same platelet-activation biology); kept the single best-evidenced representative to avoid redundant entries.
- **Olfactory Signaling Pathway / Olfactory Transduction / Expression And Translocation Of Olfactory Receptors / Sensory Perception** — explicitly checked per instructions; `preeclampsia AND olfactory receptor` returned only 3 PubMed hits (PMID 38069004, 26574895, 35269385) with no PE-specific olfactory-receptor mechanism identified. Treated as generic gene-family artifact and excluded.
- All other [GENERIC]-tagged candidates (Immune System, Innate/Adaptive Immune System, Cell Cycle, Metabolism Of RNA/Proteins, RNA Polymerase II Transcription, mRNA Splicing/Spliceosome, Cellular Responses To Stress/Stimuli, Generic Transcription Pathway, Infectious Disease/Disease, RNA transport, Processing Of Capped Intron-Containing Pre-mRNA, Cytokine Signaling In Immune System) — excluded per the auto-flag rule, no targeted search performed since these are umbrella terms not addressed by disease-specific literature by construction.
- Viral-infection pathways (SARS-CoV/-2, EBV, HSV1, HIV, Hepatitis B, Influenza A, Measles, Leishmaniasis, Salmonella) — Reactome/KEGG infectious-disease pathway gene sets are shared broadly across host antiviral/immune response genes and are not PE-specific; not deep-searched given the [GENERIC]-adjacent "Infectious Disease"/"Disease" umbrella already flagged.

## Raw search log

PubMed PMIDs retrieved across all searches (esearch/esummary via NCBI E-utilities), including those not ultimately cited:

- Neutrophil activation/degranulation: 9654608, 32973809, 36973485, 39735539, 41935260, 12908997, 11120528, 41106578
- NETosis: 37958788, 39735539, 38794175, 41935260, 41106578, 33562975, 41811413, 38913117
- NF-kB: 36374543, 33099123, 38008034, 41420525, 35218282, 33445783, 30084476, 38825645, 34434269, 34727826, 41959089, 23066738
- NLRP3/inflammasome: 32161574, 35704142, 32650532, 35976163, 34375018, 40867825, 36508916, 38677349
- Osteoclast: 15979546, 33832098, 31403127
- Type I interferon: 40054459, 38559122, 30772492, 35147251, 25603823, 8951773, 25605672, 30834654
- Platelet activation: 38537226, 40746439, 32986992, 31340711, 19464511, 30983478, 36923708, 28089907
- Th17: 32973809, 36155892, 40763478, 36237987, 24904591, 37573650, 31900289, 26135758
- Mitochondrial Complex I: 40362193, 35389182, 22840297, 27573305, 27915495, 22902742, 22858023
- RhoA/trophoblast: 35177224, 36160709, 31004838, 26653761, 18235104, 41216940, 34865521, 42123616
- Interleukin signaling: 40076938, 35258399, 40967465, 35505165, 36054229
- Interferon gamma: 36289172, 25060131, 24920727, 38672196, 40274232
- Platelet degranulation/PF4: 36923708, 6230118, 37996819, 1605673, 10452571
- Olfactory receptor + preeclampsia: 38069004, 26574895, 35269385

All searches performed via `esearch.fcgi`/`esummary.fcgi` (db=pubmed) with `sort=relevance`, no API key.
