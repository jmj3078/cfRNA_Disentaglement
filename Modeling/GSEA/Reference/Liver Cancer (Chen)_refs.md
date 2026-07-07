# References — Liver Cancer (Chen) [hepatocellular carcinoma]
Skill-retrieved evidence only (paper-lookup = PubMed E-utilities; database-lookup = Open Targets GraphQL v4). Retrieved 2026-07-06.

| Gene | Claim / disease | PubMed magnitude | Representative PMIDs (title) | Open Targets |
|---|---|---|---|---|
| UGT1A8 | rare-led novel; glucuronidation (Pentose and glucuronate interconversions) in HCC | ~5 (UGT1A8 AND hepatocellular carcinoma) | 34147074 (A stemness-based eleven-gene signature correlates with clinical outcome of hepatocellular carcinoma, BMC Cancer 2021); 15057901 | No HCC association; only "Abnormality of the liver" score 0.08 (genetic_association 0.13). Conservative novel candidate. |
| CCL1 | rare-led novel; immune chemokine (Positive Regulation of Defense Response) | ~8 (CCL1 AND hepatocellular carcinoma) | (sparse) | Not retrieved as HCC-associated; term db_support = CEBPA(0.33). Conservative novel candidate. |
| TNP1 | rare lead of DNA Repair term; testis/spermatid-specific transition protein 1 | ~4 (TNP1 AND carcinoma; all spermatogenesis context) | (spermatogenesis only) | No cancer association. Low-expression artifact candidate. |
| DNA Repair term (db_support, non-rare) | ATM/BRCA1/MSH2/MSH6/RRM2B canonical repair genes | n/a | n/a | Packet OT scores: RRM2B 0.42, POLD3 0.41, ATM 0.36, BRCA1 0.35, MSH2/6 0.35 (term-level support; rare lead TNP1 not supported). |
| CTNNB1 | established HCC driver (context / positive control) | large | 38123979 (Targeting MMP9 in CTNNB1 mutant HCC restores CD8+ T-cell antitumour immunity, Gut 2024) | HCC association 0.780 (literature 0.99, somatic_mutation 0.86, genetic_association 0.85, animal_model 0.43). Established. |
| RPL22 | dominant translation-downregulation leading-edge gene (non-novel) | ~189 (RPL22 AND cancer) | n/a | Packet db_support 0.36. |

Notes:
- Core signal (translation/ribosome + OXPHOS downregulation, NES -3.3..-2.5) is NON-novel and robust.
- rare-led novel signals (UGT1A8, CCL1, TNP1) all have sparse literature; UGT1A8 is the only mechanistically plausible hepatic candidate; TNP1 flagged as testis-specific low-expression artifact.
