# ME/CFS Pathway Literature Review

Candidate list source: `cand_ME-CFS.txt` (n=70 candidates, [GENERIC]-flagged and hist_frac>15% pathways excluded per protocol unless explicitly justified). PubMed searched via NCBI eutils; no unlisted pathways introduced.

## Selected pathways

### Platelet Activation, Signaling And Aggregation
n_sig=32/90, size=252, eff=4.33, hist_frac=0%

ME/CFS-specific hyperactivated-platelet and fibrinaloid-microclot pathology has been directly measured in ME/CFS patients (not just inferred from a generic "infection/inflammation" umbrella), with a distinct phenotype from healthy controls: thromboelastography-confirmed hypercoagulability, >10-fold greater microclot burden, and platelet hyperactivation (PAC-1/CD62P, spreading score) in ME/CFS plasma/hematocrit vs matched controls. This is mechanistically tied to a specific downstream symptom claim (microcapillary occlusion/ischemia contributing to fatigue), not a generic "immune activation" story.

- Nunes JM, Kruger A, Proal A, Kell DB, Pretorius E. *Pharmaceuticals (Basel)*, 2022 (PMID: 36015078; DOI: 10.3390/ph15080931) -- ME/CFS patients (vs matched HC) show significant whole-blood/PPP hypercoagulability, >10x greater fibrinaloid microclot area, and platelet hyperactivation (PAC-1/CD62P positivity, spreading score 2.72 vs 1.00).

Note: an earlier, smaller study found the opposite direction (reduced/unremarkable platelet aggregation, no hypercoagulability) -- see Kennedy et al. 2006 below, flagged in Considered/caveats. The two KEGG/Reactome sibling pathways "Platelet activation" (n_sig=33, eff=3.85) and "Hemostasis" (n_sig=30, eff=4.16) are gene-set supersets/near-duplicates of this same signal and are covered by the same evidence rather than treated as independent hits.

## Considered and rejected

- **Platelet activation (KEGG)** / **Hemostasis (Reactome)** -- not independently justified beyond the selected "Platelet Activation, Signaling And Aggregation" entry; same underlying gene/evidence overlap, kept as one pathway to avoid double-counting a single mechanism.
- **Complex I Biogenesis** (n_sig=24, size=51, eff=3.88, hist_frac=0%) -- mitochondrial/OXPHOS dysfunction is a genuine, heavily studied ME/CFS hypothesis (Tomas et al. PLoS One 2017, PMID 29065167: reduced OXPHOS parameters, esp. maximal respiration, in ME/CFS PBMCs vs HC). However, the one study that directly measured individual respiratory-chain complex activity (including Complex I) in ME/CFS found **no significant difference** in Complex I/II/IV activity vs controls in either PBMCs or myotubes, concluding the deficit lies upstream of the complexes themselves (Tomas et al., PeerJ 2019, PMID 30847260). A 2024 muscle-biopsy study found reduced Complex I OXPHOS capacity specifically in Post-COVID Syndrome, not in CFS (Bizjak et al., Int J Mol Sci 2024, PMID 38338957). Given the direct negative evidence for Complex I specifically, this candidate is excluded despite genuine broader mitochondrial-dysfunction literature -- claiming Complex I assembly specificity would overstate what the data support.
- **Herpes simplex virus 1 infection** (n_sig=26, size=491, eff=4.95, hist_frac=0%) -- herpesvirus reactivation is a real ME/CFS research thread, but the strongest/most specific literature is for HHV-6/EBV, not HSV-1. The one HSV-1-specific finding located is a symptom-correlation result (HSV-1 IgG titer associated with brain-fog severity within ME/CFS subgroups; Domingues et al., Heliyon 2023, PMID 37519635) -- suggestive but a single study, not a robust or mechanistic HSV-1-specific claim. The Reactome pathway itself is a large (491-gene) generic antiviral/innate-immune-signaling gene set that happens to be modeled on HSV-1; it functions similarly to the already-excluded [GENERIC] infection/immune pathways. Excluded as insufficiently HSV-1-specific.
- **HIV Infection** (n_sig=19, size=228, eff=4.52) -- same generic-antiviral-machinery composition issue as above; no ME/CFS-specific literature located, not pursued further as HIV is an unrelated disease entity.
- **Thermogenesis** (n_sig=31, size=219, eff=4.13, hist_frac=0%) -- searched for thermoregulatory/autonomic dysfunction literature in ME/CFS; hits were incidental (arousal-network review, unrelated thermal-dysregulation-post-surgery paper, T3-dosing case study) with no direct mechanistic link to this specific gene set. Not pursued.
- **Neutrophil Degranulation** (n_sig=25, size=463, eff=4.75, hist_frac=0%) -- only generic immune-activation review hits (innate/adaptive immune cell roles in ME/CFS) and unrelated CSF proteomics/PASC papers; no ME/CFS-specific neutrophil-degranulation mechanism found. Behaves like the already-excluded generic Immune System pathways; excluded.
- **Diabetic cardiomyopathy** (n_sig=29, size=189, eff=4.17) -- 1 PubMed co-occurrence hit, incidental, not mechanistic. Composition is dominated by generic OXPHOS/calcium-handling genes (same artifact pattern as the excluded "Non-alcoholic fatty liver disease" [GENERIC] entry). Excluded.
- **Focal adhesion** (n_sig=29, size=199, eff=4.29) -- 2 incidental PubMed co-occurrences, no mechanistic ME/CFS literature. Excluded.
- **Systemic lupus erythematosus** (hist_frac=57%) -- excluded per protocol; same histone-composition artifact previously identified in the Pancreatic Cancer round (histones as SLE autoantigens, not real disease signal). Not independently re-investigated.
- Cell-cycle/APC-C/mitotic-machinery candidates (M Phase, RHO GTPase Cycle/Effectors, various APC/C degradation pathways, etc.) -- not [GENERIC]-tagged in the candidate list but functionally indistinguishable from the excluded generic Cell Cycle/Signaling By Rho GTPases pathways (shared core mitotic/cytoskeletal machinery); no ME/CFS-specific literature was sought given the strong prior that these reflect generic proliferation/composition signal rather than disease-specific mechanism.

## Raw search log

PubMed (NCBI eutils esearch/esummary/efetch), searched 2026-08-05:

- PMID 32506340 -- Thakur et al., Neurotox Res 2020 (hemin/CFS mouse model, not used)
- PMID 28910366 -- Chen et al., PLoS One 2017 (Gulf War Illness mitochondrial DNA damage, not ME/CFS-specific, not used)
- PMID 28018972 -- Missailidis et al./JCI Insight 2016, pyruvate dehydrogenase in ME/CFS (mitochondrial thread, not directly cited)
- PMID 33669532 -- Missailidis et al., Int J Mol Sci 2021, oxidisable substrate provision to mitochondria in ME/CFS lymphoblasts (mitochondrial thread, not directly cited)
- PMID 24557875 -- Morris & Maes, Metab Brain Dis 2014, mitochondrial dysfunction via immuno-inflammatory/oxidative/nitrosative stress in ME/CFS (mitochondrial thread, not directly cited)
- PMID 38338957 -- Bizjak et al., Int J Mol Sci 2024 -- cited (Considered/rejected, Complex I Biogenesis)
- PMID 29065167 -- Tomas et al., PLoS One 2017 -- cited (Considered/rejected, Complex I Biogenesis)
- PMID 29420633 -- Correction to Tomas et al. 2017 (not independently used)
- PMID 30847260 -- Tomas et al., PeerJ 2019 -- cited (Considered/rejected, Complex I Biogenesis)
- PMID 32046336 -- Missailidis et al., Int J Mol Sci 2020, cell-based blood biomarkers ME/CFS (mitochondrial thread, not directly cited)
- PMID 36131342 -- Kruger/Pretorius et al., Cardiovasc Diabetol 2022, Long COVID microclot proteomics (background for platelet/microclot thread, not directly cited)
- PMID 36043493 -- Bulle et al., Biochem J 2022, ischaemia-reperfusion in RA/Long COVID/ME-CFS (background, not directly cited)
- PMID 36015078 -- Nunes et al., Pharmaceuticals 2022 -- **cited** (Selected, Platelet Activation)
- PMID 35195253 -- Kell/Pretorius, Biochem J 2022, amyloid fibrin microclots in Long COVID (background, not directly cited)
- PMID 16479189 -- Kennedy et al., Blood Coagul Fibrinolysis 2006 -- cited as contrasting/negative evidence (Selected pathway caveat)
- PMID 37519635 -- Domingues et al., Heliyon 2023 -- cited (Considered/rejected, HSV1)
- PMID 34422848 -- Legler et al., Front Med 2021, salivary HHV-6/7 DNA loads in ME/CFS (background herpesvirus thread, not directly cited)
- PMID 34291062 -- Herpesvirus serology subgroups in UK ME/CFS Biobank, Front Med 2021 (background, not directly cited)
- PMID 32793195 -- Commentary on herpesvirus antibodies in ME/CFS, Front Immunol 2020 (background, not directly cited)
- PMID 35203896 -- Barnden/Kwiatek review, Brain Sciences 2022, midbrain arousal network in ME/CFS (checked for Thermogenesis, not sufficiently specific, not cited)
- PMID 26005946 -- co-occurrence hit for "diabetic cardiomyopathy" (checked, incidental, not cited)
- PMID 42533331, 31759091 -- co-occurrence hits for "focal adhesion" (checked, incidental, not cited)
- PMID 41932997, 40987794, 33405100, 24343819 -- checked for "neutrophil degranulation" (incidental/generic immune, not cited)

Open Targets was not queried in this round -- primary literature (PubMed) was sufficient to confirm/reject each candidate, and per project precedent Open Targets scores alone are not treated as adequate justification.
