# Building SMILES-Based Predictors for Protomer Formation and Protomer-Resolved CCS

## Goal restatement

Two models are needed:

- **Model 1 (classifier):** SMILES to a probability that the molecule forms resolvable protomers under TIMS conditions.
- **Model 2 (regressor):** charge-positioned SMILES (a candidate structure with the proton explicitly placed on one basic site) to predicted CCS for that specific protomer.

Both models must be trainable from the existing dataset (~185,000 molecules, ~97,000 with usable [M+H]+ data, ~48,500 with a resolved secondary CCS) plus a small literature-derived gold set. No DFT, 3D conformers, or orthogonal IM data currently exist in-house, so the design below treats those as things to generate for a validation subset, not as prerequisites.

## Why the two models are coupled

Model 1's positive label and Model 2's training pairs come from the same ambiguous source: an observed secondary CCS peak. A peak only counts as protomer evidence if Model 2 can assign it to a specific charge-positioned candidate with a defensible CCS match. Building them together, sharing features and possibly a representation, is more consistent than building them in isolation.

## Findings update (2026-07): what a ground-truth test changed

A validation exercise (matching_validation_breakdown.ipynb) plus a danofloxacin ground-truth test (two experimentally confirmed protomers: keto-O+ at 181.7 Å², aliphatic-amine-N+ at 193.7 Å²) revised several assumptions below. The chosen implementation strategy is "tight a-priori screen, then full QM validation on survivors" (see the accompanying plan). Key changes, folded into the steps:

- **The CCS engine works and is the trustworthy part.** ORCA-PM3-MD + MOBCAL (scripts/predict_candidate_ccs_md.py) reproduced danofloxacin's real ±12 Å² protomer split to ~1 Å² (predicted gap +13.1 vs observed +12.0), ordering correct. Ordinary CCS models (GraphXSection) under-predict this gap; closing it is the whole point of Model 2.
- **Enumeration must be N/O/S, not N-only (Step 1).** The original nitrogen-only enumeration could not represent danofloxacin's keto-O+ protomer at all. scripts/enumerate_protonation_candidates.py is now widened to enumerate every N/O/S site; dimorphite-dl only annotates solution pKa.
- **Match by predicted CCS, not by energy/pKa ranking (Step 2).** Nearest-predicted-CCS matching assigned danofloxacin's protomers to the right peaks; ranking cannot, because it assumes most-stable = most-intense, which kinetic trapping breaks.
- **Do NOT gate on gas-phase energy / proton-affinity gaps.** Danofloxacin's protomers differ by ~18 kcal/mol in gas-phase energy yet both are observed: populations are set in solution (pKa) and kinetically trapped into the gas phase, not re-equilibrated. A large gap does not mean "one protomer".
- **pKa (solution) ≠ proton affinity (gas phase).** The pipeline's `pka_mean` is a Dimorphite-DL solution-pKa lookup, NOT a proton affinity (this doc previously conflated them). Keep both as Model 1 descriptors, neither as a gate.
- **Model 2 trains on validated multi-peak protomer pairs only, NOT single-site molecules** (they carry no protomer signal and dilute the gap the model must learn).
- **A cheap a-priori screen (scripts/screen_protomer_candidates.py) decides which multi-peak molecules get expensive QM:** ≥2 enumerated sites AND ≥2 protomer-relevant sites (basic N or carbonyl O). Calibrated to keep danofloxacin/ciprofloxacin/benzocaine/PABA and drop single-site amines and all-ether-O sugars.

## Step 1: Generate charge-positioned candidate structures for every molecule

For each of the ~97,000 [M+H]+ molecules, enumerate distinct protonation-site candidates from the 2D SMILES:

- Identify basic/protonatable atoms with SMARTS patterns (aliphatic amine, aromatic N, amide/amidine, guanidine, pyridine-type N, etc.), or use an existing protonation-state enumerator (e.g., Dimorphite-DL, RDKit's `EnumerateProtonationStates`) restricted to generating one candidate per distinct basic site rather than per pKa-consistent microstate.
- Discard sites that are chemically equivalent by symmetry.
- Each candidate becomes a charge-positioned SMILES: the 2D graph with an explicit formal +1 charge on one specific atom.
- Molecules with only one plausible basic site (the large majority, per Kruve's structure-specific propensity finding, *Anal. Chem.* 2022, 10.1021/acs.analchem.2c00257) get one candidate and are expected, by construction, not to form protomers.

This enumeration is the single most important shared artifact: it defines the candidate space for Model 2 and the input for Model 1's basic-site features.

## Step 2: Label construction (the central difficulty)

**Model 1 labels**, from the CCS ratio bands already established:

| Ratio band | Interpretation | Label |
|---|---|---|
| > 1.25 | dimer / co-eluting species | excluded (not a monomer question) |
| ~1.08-1.20 | benzocaine-magnitude split (Warnke et al., *JACS* 2015, 10.1021/jacs.5b01338; Bull et al., *PCCP* 2016, 10.1039/C6CP04941A report 4-8% in He, >15% in N2) | positive (protomer) |
| ~1.03-1.08 | ambiguous: could be protomer or large conformer split | uncertain, hold out or soft-label |
| ~0.97-1.03 | suspected peak-picking artifact | negative, but noisy |
| single peak | no resolved secondary species | negative, but noisy (may hide an unresolved protomer below instrument resolving power) |

Both negative classes carry label noise: TIMS resolving power (R ~ 100-200) sets a detection floor, so "single peak" is really "no protomer above the resolvable threshold," not "no protomer." Treat this as a positive-unlabeled or noisy-label problem rather than clean binary classification. Practical options: exclude the 0.97-1.03 and single-peak-with-borderline-features rows from training and only use molecules with a clear peak or a clearly clean single peak (e.g., no other same-class molecule at similar features shows a split), or use label smoothing / a noise-robust loss (e.g., a Bernoulli mixture model, or training with soft targets derived from peak-shape QC confidence).

**Model 2 labels** require an additional step: assigning each observed CCS value (primary, secondary) to a specific charge-positioned candidate. This is not given by the data. Two workable strategies, usable together:

1. **Single-peak molecules with one basic site**: unambiguous. The one candidate gets the one observed CCS. This is the bulk of clean training data for Model 2 (tens of thousands of pairs) and anchors the model even before any protomer question is addressed.
2. **Multi-peak molecules**: use a fast, scalable proxy to rank candidates and match them to peaks, since full DFT is not feasible at this scale. A defensible proxy chain: (a) rank candidates by a fast semi-empirical proton affinity estimate (e.g., GFN2-xTB) rather than full DFT, (b) rank candidates by an ML-predicted CCS from their 2D structure alone (any of the standard tools below), (c) match observed peaks to candidates by nearest predicted CCS, solving the assignment as a small bipartite matching problem (trivial when there are only two candidates and two peaks). This assignment is itself uncertain and should be treated as weak supervision, not ground truth.

## Step 3: Model 1, the protomer-formation classifier

**Features** (all computable from 2D SMILES at N = 97,000 scale):
- number and type of distinct basic sites from Step 1
- presence of proton-hopping-blocking substituents adjacent to a basic site (Kruve's mechanism for why some sites lock in rather than equilibrate)
- gap between the top two candidates' fast proton-affinity estimates (small gap implies co-population is more likely)
- flexibility descriptors (rotatable bond count, ring count, presence of intramolecular H-bond donors/acceptors)
- standard 2D descriptors (molecular weight, TPSA, logP) as covariates
- optionally, difference in dipole moment between top candidates, following the correlation Boschmans et al. (*Analyst* 2016, 10.1039/c5an02456k) found between delta-dipole and CCS separation magnitude, computed from cheap semi-empirical geometries rather than DFT

**Architecture choice:**
- Start with gradient-boosted trees (LightGBM/XGBoost) on the engineered features above. At this dataset size, with noisy labels, a tabular model is easier to calibrate and inspect than a graph network, and feature importances double as a sanity check against the mechanistic literature (basic-site count and proton-affinity gap should dominate if the model has learned something real).
- As a second model, a graph neural network (e.g., via Chemprop or a custom PyTorch Geometric message-passing net) taking the raw molecular graph, to check whether it recovers the same signal or finds structure the engineered features miss. Treat agreement between the two as a confidence signal.
- A multi-task variant is worth trying once Model 2 exists: shared graph embedding, one head predicting protomer probability, another predicting CCS per candidate. This is close to what Ross et al. describe their 3D-descriptor model doing implicitly (*J. Am. Soc. Mass Spectrom.* 2022, 33, 1061, 10.1021/jasms.2c00111): a model that changes its CCS prediction by protonation site is already halfway to flagging when two sites are competitive enough to matter.

**Validation:**
- Held-out literature gold set: benzocaine, aminobenzoic acid isomers, caffeine/paraxanthine, quinazoline, melphalan, cefpodoxime proxetil, neonicotinoids, several anticancer drugs (palbociclib, copanlisib, olaparib, *J. Am. Soc. Mass Spectrom.* 2024, 10.1021/jasms.4c00233), all with literature-asserted protomer or conformer origin. Small (dozens of compounds) but mechanistically confirmed, so useful for calibration and sanity-checking rather than training.
- Internal stratified k-fold, but report performance separately by ratio band, since the 0.97-1.03 and single-peak classes are the noisiest.

## Step 4: Model 2, the protomer-resolved CCS predictor

**Input representation:** charge-positioned SMILES from Step 1, i.e., the 2D graph with the formal charge pinned to one atom. This is a stronger signal than the plain neutral SMILES most CCS predictors take, and is exactly the missing ingredient Ross et al. identify: standard 2D descriptors "do not allow the differentiation of such isomers," while a model that explicitly encodes which atom carries the charge can produce different CCS values for different protonation states of the same molecule.

**Architecture:**
- A message-passing GNN over the molecular graph, with the protonated atom flagged as a node feature (formal charge, or a dedicated one-hot "is-protonation-site" flag), predicting a scalar CCS. This mirrors the design intent of SigmaCCS/GraphCCS/DeepCCS but adds explicit charge-site conditioning at the input rather than relying on the network to infer it from an ambiguous neutral graph.
- Pretrain on the large, unambiguous single-site training pairs from Step 2 (tens of thousands of molecule-to-CCS pairs, one candidate each). This gives the model a solid general CCS surface before it has to resolve any protomer subtlety.
- Fine-tune, or add an auxiliary loss, on the multi-peak molecules using the matching-derived pairs from Step 2, with a permutation-invariant set loss (match predicted candidate CCS values to observed peaks by minimum total error, since which candidate corresponds to which peak is the very ambiguity being resolved).
- Where budget allows, compute true DFT-optimized-geometry + trajectory-method CCS (e.g., via HPCCS, following Boschmans et al. and the anticancer-drug HPCCS study above) for a few dozen representative multi-peak molecules across the tiers. Use these as a small, high-trust fine-tuning or evaluation set layered on top of the semi-empirical-matched pairs, since DFT-level proton affinity and trajectory-method CCS are the closest available substitute for ground truth without new experiments.

**Expected magnitude as a sanity check:** protomer-driven CCS separations of ~4-15%+ (larger in N2 than He, per Bull et al. 2016), versus ~1-2% for rigid-conformer differences. A trained model whose predicted candidate-to-candidate CCS gaps fall in this range for molecules flagged as protomers by Model 1, and near zero for molecules with one basic site, is behaving consistently with the literature even before any experimental validation.

## Step 5: Joint validation plan

1. Run both models on the full ~97,000-molecule set.
2. Pull a stratified subset (~30-100 molecules) spanning: Model 1 high-confidence positive, high-confidence negative, and disagreement cases (e.g., Model 1 says protomer but ratio is in the ambiguous 1.03-1.08 band, or Model 1 says no protomer but ratio is in the 0.97-1.03 band).
3. On this subset, compute DFT proton affinities and trajectory-method CCS for all candidates (as in Step 4), and, if any collaboration access exists, seek mobility-selected tandem-MS/MS fragmentation or IRMPD/UVPD action spectroscopy as a closer-to-definitive check (the only techniques in the literature that directly assign protonation site rather than infer it).
4. Use this subset to report calibrated per-tier accuracy for Model 1 and per-candidate CCS error for Model 2, rather than claiming per-molecule certainty anywhere in the full dataset.

## Practical notes for implementation

- Steps 1 and 2 (candidate enumeration, feature computation, weak-label assignment) are the bulk of the engineering effort and should be built as a reusable Python pipeline (RDKit for structure handling and SMARTS matching, GFN2-xTB via `xtb-python` or a subprocess call for fast proton affinity, a chosen 2D/3D CCS tool for the matching proxy).
- Keep the candidate-enumeration and matching steps auditable per molecule, since debugging mislabeled training pairs will matter more than most hyperparameter choices given how noisy the labels are.
- A natural visualization once this pipeline exists: per-molecule diagrams showing the 2D structure with candidate protonation sites marked, their predicted proton affinities and CCS values, and the observed peak(s) overlaid, alongside dataset-wide plots such as the CCS-ratio distribution split by Model 1 prediction, and predicted-vs-observed CCS scatter for Model 2 colored by training tier (single-site / matched-pair / DFT-validated).

## Caveats

- Neither model can be validated against true, experimentally confirmed per-molecule ground truth at scale; both rest on proxies (ratio bands for Model 1, fast proton-affinity and CCS-based matching for Model 2) that are reasonable but not definitive.
- Label noise from TIMS peak-picking artifacts and from unresolved sub-threshold protomers puts a ceiling on achievable classifier performance that no amount of modeling effort will remove; the DFT-validated subset is there to estimate that ceiling, not eliminate it.
- The matching step in Model 2 assumes the correct candidate is among the enumerated basic sites; unusual tautomeric or non-canonical protonation chemistry not captured by the SMARTS-based enumeration would be silently mismatched.
- All of the above is trained on one instrument (Bruker timsTOF Pro 2, N2 drift gas, positive ESI); transfer to other IM-MS platforms or drift gases is not implied, and the reported CCS separation magnitudes (protomer versus conformer) are themselves partly N2-specific per Bull et al. 2016.
