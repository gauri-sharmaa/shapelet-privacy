> **Superseded by `paper/main.tex` / `paper/main.pdf` (final 3-seed numbers).** Kept as the original outline.

# Interpretable by Construction, Identifiable by Construction: Privacy Leakage of Shapelet-Based Anti-Money-Laundering Models

*Draft for the ICAIF 2026 Workshop on Financial AI Security, Privacy and Safety (4 pages + refs). Numbers marked † are seed 0; final tables report mean ± std over 3 seeds (`results/RESULTS.md`).*

## Abstract

Regulators and investigators ask that anti-money-laundering (AML) models be explainable, and shapelet-based time-series classifiers are a natural answer: they explain a decision by pointing to a short, human-readable pattern in an account's activity. We show that this explanation is often *literally a customer's transaction history*. On SAML-D (9.5M synthetic transactions, 28 typologies), we train shapelet detectors on activity-matched weekly account series and release their shapelets to an adversary holding a pool of ~30k candidate accounts. For the classic shapelet transform, 17%† of released shapelets uniquely re-identify their source customer from shape alone, and 63%† of source accounts are flagged launderers, so the model also discloses who is under suspicion. For the state-of-the-art Random Dilated Shapelet Transform (RDST), which stores per-shapelet normalisation statistics, 83%† of shapelets single out exactly one customer (803† customers in total) and reveal their exact weekly amounts and counts. Standard membership-inference audits miss this entirely (AUC ≈ 0.50–0.52), while GBDT and MiniRocket baselines leak nothing measurable. Gaussian perturbation and k-anonymous averaging remove exact fingerprints at no accuracy cost but leave residual linkage; *learned* shapelets match RDST accuracy (PR-AUC 0.957† vs 0.952†) with no shapelet within distance 0.01 of any real account. We argue interpretable financial models need a privacy review of their explanations, not just their predictions.

## 1 Introduction

- AML transaction monitoring is high-stakes, regulated, and increasingly ML-driven (Tertychnyi et al. 2022 deploy CatBoost + Shapley explanations at a multinational bank). Investigators need explanations; supervisors increasingly expect model transparency.
- Shapelets (Ye & Keogh 2009; Hills et al. 2014) are among the most interpretable time-series classifiers and fit AML typologies (structuring, fan-in/out, rapid pass-through) that are *shapes in time*.
- But classic shapelets are subsequences copied from training data, and SOTA variants (RDST, Guillaume et al. 2022) additionally store normalisation statistics. Prior work notes this risk in passing (FedST, Liang & Wang 2024) and defends against pattern inference (PMP, 2023); explanation-leakage work (Shokri et al. 2021) studies feature attributions, not model-embedded explanations. To our knowledge no work measures what a released shapelet model reveals about specific customers.
- **Contributions.** (1) A threat model and attack suite (re-identification, attribute reconstruction, membership inference) for released shapelet models. (2) An empirical study on SAML-D showing large, *concentrated* leakage that aggregate MIA misses, including disclosure of flagged (suspicious) customers — a tipping-off / SAR-confidentiality hazard. (3) A privacy–utility comparison of mitigations, finding that learned shapelets give near-SOTA accuracy without copying customer data.

## 2 Threat model

A bank trains a shapelet-based AML detector and shares the fitted model with a third party — a vendor, auditor, regulator sandbox, consortium partner, or via model theft. The model includes its shapelets (and, for RDST, their normalisation statistics). The adversary also holds a pool of candidate account series (e.g. a partner institution's or a leaked extract) that contains some training members and non-members. Goals: (a) **re-identify** which customer a shapelet came from; (b) **reconstruct** that customer's transaction amounts/counts; (c) infer **membership** and, via the shapelet's class, **suspicion status**.

## 3 Setup

**Data.** SAML-D (Oztas et al. 2023): 9,504,852 transactions, 2022-10-07 → 2023-08-23, 0.104% suspicious, 17 suspicious + 11 normal typologies. We build per-account weekly series (46 weeks × 6 channels: log1p outgoing/incoming amount, outgoing/incoming count, cross-border outgoing amount, cash amount). Accounts involved in any laundering transaction are positive (7,902). Normal accounts are sampled 3:1 **matched on number of active weeks** (median 18 vs 19 active weeks), because unmatched normal accounts are mostly single-week and make detection trivial. 29,946 accounts total, split 50/50 by account (members/non-members), stratified.

**Detectors.** CatBoost on 40 aggregate features + flattened series (the deployed-industry baseline); MiniRocket (10k kernels) + logistic regression; classic binary shapelet transform (STC: 5,000 sampled non-constant z-normalised candidates, quality = class separation of min-distance profile, top-200 with self-similarity removal) + ExtraTrees; RDST (aeon, 1,000 shapelets) + ExtraTrees; learned shapelets (144 shapelets, 3 lengths × 6 channels × 8, soft-min distance, Adam, random init).

**Attacks.** *Re-ID*: for each released shapelet, compute its min sliding (dilated, for RDST) distance to every candidate account; report top-1 rate and the **unique-match rate** (the source is the only account within ε = 1e-4). *Reconstruction (RDST)*: raw window = values × std + mean; count exactly revealed non-zero weekly cells. *MIA*: score = −min distance to any released shapelet; for baselines, standard confidence attack.

## 4 Results

**Table 1 (seed 0†).**

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | MIA AUC | Shapelets uniquely re-identifying a customer |
|---|---|---|---|---|---|
| CatBoost (aggregates) | 0.983 | 0.995 | 0.829 | 0.505 | n/a |
| MiniRocket | 0.955 | 0.988 | 0.648 | 0.507 | n/a |
| Shapelet transform (STC, k=200) | 0.869 | 0.943 | 0.447 | 0.497 | **17.0%** (top-1 21.5%) |
| RDST (k=1000) | 0.952 | 0.987 | 0.618 | 0.521 | **83.4%** (803 customers) |
| Learned shapelets | 0.957 | 0.987 | 0.649 | 0.501 | 0% (no exact copies) |

**Findings.**
1. *Leakage is concentrated, so aggregate MIA misses it.* Only the few hundred source accounts are exposed out of ~15k members, so MIA AUC stays ≈0.5 — yet those accounts are exactly and uniquely identifiable.
2. *Shape alone identifies.* Even z-normalised STC shapelets (no amounts) uniquely re-identify 17% of sources among ~30k accounts.
3. *SOTA leaks more.* RDST stores means/stds, so 83% of its shapelets reconstruct an exact, unique slice of one customer's history (median 12 non-zero weekly amount/count cells; median peak weekly amount ≈ £24k).
4. *The model discloses suspicion.* 125/200 STC source accounts are positives: releasing the model reveals which customers were flagged — conflicting with SAR confidentiality / anti-tipping-off obligations.

**Table 2: mitigations on STC shapelets (seed 0†).**

| Release | PR-AUC | Unique re-ID | Top-1 re-ID |
|---|---|---|---|
| As-is | 0.869 | 17.0% | 21.5% |
| Gaussian noise σ=0.1 | 0.871 | 0% | 14.0% |
| σ=0.25 | 0.866 | 0% | 3.5% |
| σ=0.5 | 0.869 | 0% | 0.5% |
| σ=1.0 | 0.865 | 0% | 0% |
| k-anonymous mean, k=5 | 0.868 | 0.5% | 14.0% |
| k=20 | 0.869 | 0.5% | 15.5% |
| k=50 | 0.868 | 0% | 12.5% |
| Learned shapelets | 0.957 | 0% (none within 0.01 of any account) | – |

Noise removes exact fingerprints at no utility cost but needs σ ≥ 0.5 to stop nearest-neighbour linkage; k-anonymous averaging stops unique matching but the source's neighbourhood remains linkable (12–16% top-1). Learned shapelets dominate: best shapelet accuracy and no copied customer data — though they carry no formal guarantee (future: DP-SGD).

## 5 Discussion and limitations

- Synthetic data (SAML-D) with activity-matched sampling; absolute detection numbers are optimistic, but the leakage mechanism (copied subsequences, stored statistics) is data-independent.
- Our STC is a faithful but simplified re-implementation (AUC-based quality instead of information gain) because aeon's implementation degenerates on sparse series (85% zeros).
- The adversary's pool contains the source; real-world success depends on auxiliary data access — but consortium AML and model-sharing settings make that plausible.
- Recommendations: treat shapelets as personal data; do not release normalisation statistics; prefer learned or noised shapelets; audit with *per-shapelet* re-identification, not aggregate MIA.

## References (to format)

Ye & Keogh 2009 (shapelets); Hills et al. 2014 (shapelet transform); Bostrom & Bagnall 2017; Guillaume et al. 2022 (RDST); Grabocka et al. 2014 (learned shapelets); Dempster et al. 2021 (MiniRocket); Oztas et al. 2023 (SAML-D); Altman et al. 2023 (AMLworld); Tertychnyi et al. 2022 (AML monitoring); Liang & Wang 2024 (FedST); PMP 2023 (arXiv 2301.01838); Shokri et al. 2021 (explanation privacy); Carlini et al. 2022 (MIA at low FPR).
