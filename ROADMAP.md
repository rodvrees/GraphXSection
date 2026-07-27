# Plan: protomer-aware CCS GNN (data, architecture, training) built from GraphXSection_torch

## Context

The end goal is a CCS predictor that is protonation-aware: given a molecule and adduct it should
predict a distinct CCS per protomer, reproducing the few-percent CCS gap between protomers that
ordinary CCS models (GraphXSection, any 2D model trained on one CCS per molecule) miss. The
current SigmaModel already ingests a charge-positioned SMILES, so it *can* differentiate protomers
through charge-sensitive atom features, but it has never been trained on data where the gap matters,
so it defaults to ~1 to 2% variation across sites. Enveda-180 is the only large dataset with the
multi-CCS (protomer) signal needed to fix that.

Two findings from prior work shape the design:
1. Per-site to per-peak assignment is unreliable at scale (a nearest-CCS matcher manufactures ~63%
   of the apparent gap concordance; only a weak structure-specific component survives). So the model
   must learn the gap **without** committing to which enumerated site is which observed peak.
2. The CCS engine is not the bottleneck and does not belong in the training loop; MD/DFT are for
   validation, not label generation.

---

## STATUS (2026-07-27)

The full pipeline is **implemented, trained, and evaluated**. Base-CCS accuracy is at the Enveda data
ceiling (not a method deficiency) -- now proven from BOTH directions: the good reference model also only
reaches ~2.4% on Enveda, and the current approach reaches ~1.4% on curated data. **The protomer gap
DIRECTION is NOT learned.** The headline Enveda gap_corr 0.73 is a matcher artifact: with a fixed,
matcher-free assignment the correlation collapses to 0.05, the model predicts a near-constant positive
gap, and the external gold directional check fails (1/3). See the GAP-DIRECTION ANALYSIS below. Net: base
CCS works and is data-limited. **DECIDED path for the gap: assignment-labelled supervision -- fix the
site->peak assignment by proton-affinity ranking (energy -> abundance -> intensity order), use the observed
CCS as the label, compute NO CCS. See the CHOSEN DIRECTION section.** Sections 1-4 below are the design
reference; this block records what was built, the results, and what deviated from the plan.

### Implemented
- **Corpus:** `scripts/build_training_corpus.py` -> `outputs/training_corpus/corpus_{train,valid,test}.parquet`.
  Enveda-180 only so far (the curated TIMS add-ons in Section 1 are not yet merged). Scaffold-split, 0 leakage.
- **Featurizers:** `GasteigerChargeBinned` + `IsProtonationSite` in `graphamole/.../featurizers/atom.py`;
  config-switchable via `configs/features_{baseline,protomer}.json` (15 vs 17 atom features).
- **Dataset/collate/datamodule:** `GraphXSection_torch/protomer_data.py`
  (`ProtomerCorpusDataset`, `collate_protomer`, `ProtomerDataModule`), with `graph_smiles_mode`
  (charged/neutral) and `primary_only` switches for ablations.
- **Loss:** `anchored_set_loss` + `ProtomerLightningModule` in `lightning_modules.py`.
- **Architecture:** charged-atom readout in `models/Sigma.py` (ablatable); adduct descriptor decoupled
  onto the neutral parent SMILES.
- **Scripts:** `train_protomer.py`, `eval_protomer_gap.py` (base parity, gap vs zero-gap, permutation
  null, gold spot-checks), `run_ablations.py` (6-variant leave-one-out), `build_matched_baseline_csv.py`.
- **Config:** `configs/sigma_config_protomer.json` and the generated `outputs/ablations/config_*.json`.

### Deviations from the original plan (all necessary)
- **`final_activation: none` (not relu).** The final ReLU is a dead-output trap at init: predictions
  collapse to 0, gradients vanish, metrics freeze after ~1 epoch. Linear output is required.
- **`learning_rate: 0.003` at `batch_size: 512`.** The sqrt-batch-scaled 0.02 diverged; 0.003 (near the
  original stable 0.00356) is clean.
- **Feature caching added.** Training was CPU-bound on per-item RDKit featurization (~5 ms/mol, GPU
  forward ~4 ms/batch, so bigger batches did nothing). Now precompute once to a disk cache
  (`~/.cache/protomer_precompute`, parallel shard-write ~1100 mol/s) and serve `num_workers=0`.
  Also fixed an O(N^2) Gasteiger recompute (once per atom -> once per molecule).
- **Adduct-parser robustness.** `_TOKEN_ALIASES` maps MS shorthands (FA/Hac/MeOH/...) to real formulas;
  `_VALID_ELEMENTS` pre-check stops an RDKit C++ banner that flooded logs.
- **Multimers dropped from the corpus** (`DROP_MULTIMERS` in the builder): 24.8% of rows were
  `[2M+...]` where the featurized monomer graph does not match the measured dimer ion. In-source
  fragment adducts (`[M-H2O+H]+`, ...) are kept per the user. Corpus now train 208,455 / valid 26,003
  / test 25,720 (gap rows unchanged 48,131). With-multimer backups: `corpus_{split}_withmultimer.parquet`.

### Results (100-epoch ablations, WITH-multimer test split; clean-data rerun pending)

| variant        | base med-rel | gap_corr | sign_acc | gap_MAE | zero-gap | null_corr |
|----------------|-------------:|---------:|---------:|--------:|---------:|----------:|
| full           | 2.51%        | **0.709**| **0.86** | **4.97**| 10.58    | 0.383     |
| no_isprot      | 2.34%        | 0.653    | 0.81     | 5.99    | 10.57    | 0.36      |
| no_gasteiger   | 2.17%        | 0.688    | 0.85     | 5.36    | 10.56    | 0.37      |
| no_readout     | 2.39%        | 0.642    | 0.81     | 6.05    | 10.57    | 0.36      |
| neutral_smiles | 4.00%        | 0.133    | 0.06     | 10.54   | 10.54    | 0.12      |
| primary_only   | **1.81%**    | 0.457    | 0.64     | 9.69    | 10.55    | 0.31      |

- **`full` gap_corr 0.709, sign_acc 0.86, gap_MAE 4.97 vs zero-gap 10.58, above null 0.383.**
  [SUPERSEDED] These are the matcher-based numbers; the GAP-DIRECTION ANALYSIS below shows they are
  matcher inflation (fixed-assignment corr 0.05). "Beats the null" is real but small; it does NOT mean
  the gap direction is learned. Read the ablation ORDERING (which feature helps) but not the absolute
  gap_corr as a capability.
- **`neutral_smiles` control:** strip charge from the graph and the gap signal collapses to the null.
  All gap capability comes from the charge-positioned SMILES.
- **`primary_only`** (no gap term) already has gap_corr 0.457 from the intrinsic charged-feature spread;
  the gap loss ~doubles it to 0.709. The gap objective does real work.
- **Feature ladder:** readout and `IsProtonationSite` each contribute ~0.06 gap_corr; Gasteiger ~0.02.
- **Base/gap tradeoff:** `gap_lambda=5` costs ~0.7 pt of base accuracy (full 2.51% vs primary_only 1.81%).

### Clean-data `full` retrain (multimers removed; evaluated on the clean 25,720-mol test set)

| metric | value | vs before |
|--------|-------|-----------|
| base med-rel (ALL) | **2.80%** | worse than old full 2.51% and reference 2.40% |
| base med-rel [M+H]+ / [M-H]- | 2.55% / 3.31% | [M+H]+ ~0.5 pt above the reference's 2.02% |
| Enveda gap_corr | **0.727** | held up (was 0.709) |
| Enveda sign_acc / gap_MAE | 0.871 / 4.81 (null 0.389) | held up |
| **gold directional (n=3)** | **1/3 correct** | benzocaine, danofloxacin WRONG sign; melphalan right but mag 1.51 vs obs 9.2 |

Two things to read from this:
1. **Base got worse, not better, after cleaning.** Removing 68k base-only multimer rows shrank the
   base-regression signal while the gap molecules and `gap_lambda=5`/`oversample=5` stayed fixed, so
   the gap term now dominates more and base pays for it. The fix is the `gap_lambda` sweep (Next-step
   #3) and/or lowering `gap_oversample_factor`; `primary_only` on clean data is still expected ~1.8-2%.
2. **The Enveda gap correlation does NOT transfer to the external gold set** -- because it is a matcher
   artifact. See the gap-direction analysis below.

### GAP-DIRECTION ANALYSIS (2026-07-27, `scripts/analyze_gap_direction.py`): the gap is NOT learned

For the 1,350 clean 2-peak/2-candidate test molecules, the signed predicted gap (secondary - base)
was scored two ways:

| assignment scheme | n | sign_acc | corr | gap_MAE |
|---|---|---|---|---|
| MATCHER (Hungarian; the headline metric) | 1284 | 0.831 | **+0.456** | 5.88 |
| FIXED (dominant heuristic = base; matcher-free) | 1350 | 0.708 | **+0.051** | 7.38 |

(The full-eval headline 0.73 is even higher than 0.456 here because it includes >2-candidate molecules,
where the matcher has more candidates to choose the error-minimising assignment from -- more freedom,
more inflation.)

**Conclusion: the gap direction is not learned; the correlation is manufactured by the matcher.**
- With a fixed, matcher-free assignment the correlation is **0.05 (zero)**.
- The model predicts a **near-constant positive gap** (pred_gap_mean ~+6 to +9 for every site-element
  pair, regardless of the observed gap). Observed gaps are positive 76% of the time, so this
  constant-positive rule scores sign_acc 0.708 -- *below* the trivial always-positive baseline (0.76).
  It learned the MEAN gap direction, not per-molecule direction.
- Root cause: the two protomers of a molecule are near-identical 2D charged graphs (they differ only
  at the proton position, and CCS is ~blind to protonation site), so the input carries too little
  signal to predict which protomer is larger. The anchored SET loss rewards matching the set of CCS
  magnitudes, and a constant spread satisfies that without learning direction. This is the founding
  concern (per-site assignment unreliable; matcher manufactures concordance), now proven at the model
  level, and it explains the gold failure (1/3): with true assignments and no matcher, the
  constant-positive rule fails whenever the real ordering differs.

**Implication:** base CCS is solved (data-limited); protomer-gap DIRECTION is very likely NOT learnable
from 2D charged-SMILES graphs with a set-matching objective. Advancing it needs either
direction-aware supervision from a set with KNOWN site->peak assignments (QM-derived labels, see the
protomer-QM-label pipeline, or the Waters 48-pair set with sites protonated) or a representation that
actually distinguishes protomers (3D/conformer or physics features) -- which trades against the
"no conformer generation at inference" constraint.

### CHOSEN DIRECTION (2026-07-27): assignment-labelled supervision by proton-affinity ranking

Decision: fix the site->peak assignment by ENERGY (proton affinity), not by CCS, and use the observed
CCS directly as the label. Every prior attempt (MD-CCS, DFT-static CCS, the GNN) assigned peaks by
COMPUTED CCS, and that matching is exactly what inflated the results (the MD pipeline's structure-specific
signal was only +0.092 above the permutation null; ~63% was nearest-CCS matcher selection artifact). The
key realisation: **we do NOT need to compute CCS at all.** The observed CCS is already in the data, so once
a protomer is assigned to a peak by abundance, that peak's observed CCS is the protomer's label.

Why energy/abundance is the right key: the observed peaks are intensity-ordered (`ccs_species`, base peak
first) and peak intensity tracks protomer abundance. Abundance is governed by protonation energetics, so a
proton-affinity ranking gives the assignment. This is orthogonal to CCS, so it breaks the matcher
degeneracy that CCS-based matching cannot escape (many candidates near the observed gap -> selection
artifact).

Pipeline (no MOBCAL / MD / CCS prediction anywhere):
1. **Purity gate: protomer vs conformer.** The 2-9% band is conformer-dominated (Enveda analysis:
   tertiary-amide OR 1.27, biaryl dose-response, cross-adduct non-transfer). A conformer has no site to
   rank, so restrict to a high-purity protomer subset first: cross-adduct non-transfer (Step 8), no
   tertiary-amide / biaryl rotor, >=2 competitive basic sites.
2. **Enumerate protomers** (Dimorphite / existing `enumerate_protonation_candidates.py`).
3. **Rank protomer abundance by energetics** — `scripts/rank_protomer_energies.py` (BUILT + validated
   2026-07-27, replaces `.archive/proton_affinity_calculator.py`). Ranks the enumerated microstates of
   each molecule by relative gas-phase energy of the CHARGED species (the neutral cancels, so it is NOT
   computed). Two-stage for efficiency over ~800k candidates: **Stage 1** PM3-opt every candidate (store
   energy + optimized geometry); **Stage 2** r2SCAN-3c single point only for MOLECULES whose ranking is
   uncertain (PM3 top-2 gap < 3 kcal), that carry a gap (`has_clean_secondary_ccs`), or are anions — and
   for ALL candidates of a selected molecule (never mix PM3 and DFT within one ranking); **Stage 3** rank
   by the homogeneous per-molecule energy, Boltzmann populations, + the protomer/conformer signals below.
   Reuses `embed_and_optimize`/`mol_to_xyz_block` (`rank_candidates_pm3.py`); ORCA callers inlined to
   avoid the isicle/MOBCAL chain. Resume/skip-existing built in. Runs on ALL molecules (not just gap
   ones) so the dominant protomer that sets each molecule's base-CCS structure is validated everywhere.
   - **Starts from the ORIGINAL data** (`--data data/Enveda-180/enveda-180.jsonl.gz`), not a pre-built
     intermediate. Candidate enumeration is delegated to the validated pipeline in
     `enumerate_protonation_candidates.py` (`load_spectra` -> `build_secondary_species_table` ->
     `build_molecule_table` -> `run_candidate_generation`, the exact chain that produced
     `enveda_widened.parquet`), then cached to `--candidates` (default
     `outputs/protomer_energies/candidates.parquet`) and reused on resume. Point `--candidates` at an
     existing candidate parquet to skip enumeration; `--force-enumerate` rebuilds the cache.
   - **Starts from the ORIGINAL data** (`--data data/Enveda-180/enveda-180.jsonl.gz`), not a pre-built
     intermediate. Candidate enumeration is delegated to the validated pipeline in
     `enumerate_protonation_candidates.py` (`load_spectra` -> `build_secondary_species_table` ->
     `build_molecule_table` -> `run_candidate_generation`, the exact chain that produced
     `enveda_widened.parquet`), then cached to `--candidates` (default
     `outputs/protomer_energies/candidates.parquet`) and reused on resume. Point `--candidates` at an
     existing candidate parquet to skip enumeration; `--force-enumerate` rebuilds the cache.
   - **Enumerate ALL competing sites; do NOT pre-filter by element.** The archived N>O>S first-tier-only
     `identify_protonation_sites` would enumerate only N whenever any N exists, discarding the O protomer
     that is actually observed (danofloxacin keto-O+ vs amine-N+; benzocaine carboxyl-O+ vs amine-N+).
     We enumerate every N/O/S (+1) / Dimorphite acidic (-1) site (symmetry-collapsed) and let energy rank.
   - **Negative mode ([M-H]-):** SOUND and symmetric (lowest-energy anion = most acidic = most abundant
     deprotomer). r2SCAN-3c's basis has NO diffuse functions, so anions use a diffuse-augmented level
     (`--dft-method-neg` default `r2SCAN def2-TZVPD D4 RIJCOSX def2/J`, confirmed to run + converge in the
     ORCA build); PM3 anion energies are not trusted (anions always go to DFT). SCF non-convergence flagged.
   - **Caveats (flagged, not fixed now):** electronic energy is a proxy for free energy (ZPE/entropy
     ignored, usually small between similar protomers); gas-phase energy != observed abundance under
     kinetic trapping, so keep Dimorphite `pka_mean` alongside energy and gate the assignment on
     energy+pKa agreement + a resolvable gap; single conformer per microstate (K-conformer ensemble is a
     later accuracy upgrade).
   - **Protomer-vs-conformer discovery output** (per molecule, orthogonal to the Enveda rotor/cross-adduct
     flags): `n_populated` (Boltzmann pop > threshold), `top2_energy_gap_kcal`, `cross_element_competition`,
     `protomer_plausible` (>=2 populated protomers). A `has_clean_secondary_ccs` molecule that is NOT
     protomer_plausible (one protomer dominates, large energy gap) has a secondary peak better explained as
     a CONFORMER. Validated on the smoke test: a [M+H]+ case with a 21 kcal gap -> conformer-suspect; a
     [M-H]- case with a 1.5 kcal gap -> protomer-plausible. Outputs `outputs/protomer_energies/ranked_{candidates,molecules}.parquet`.
4. **Align abundance ranking to intensity-ordered peaks:** most abundant protomer -> base (most intense)
   peak, next -> second peak. Confidence-gate on the proton-affinity gap: keep only molecules where the
   PA difference is large enough to trust the order; drop near-ties.
5. **Emit labels:** (protomer charged SMILES, observed CCS of its assigned peak). The observed CCS is the
   label; nothing is computed.
6. **VALIDATE the ranking on the Waters gold set BEFORE relabelling Enveda.** `data/Waters/other_ccs_measurements.xlsx`
   (6,172 rows; `CCS Protomer 1/2` + proton-site labels + neutral SMILES). Protonate at the labelled
   sites, run the PA ranking, and confirm it recovers the known site->CCS (danofloxacin keto-O+ 181.7 vs
   amine-N+ 193.7, etc.). Only trust Enveda labels once this passes.

Training and acceptance:
- **Direction-aware loss** on the confident labelled subset (penalise the wrong sign of
  pred[secondary] - pred[dominant] using the assigned direction), on top of the existing base-CCS anchor
  term trained on all data. Retire the anchored-offset CCS matcher from the loss for these molecules.
- **Acceptance metric = external gold directional accuracy** (expanded Waters set), NOT the matcher-based
  Enveda gap_corr, which is not trustworthy.

Retire for labelling: the MD-CCS (`predict_candidate_ccs_md.py`) and DFT-static
(`predict_candidate_ccs_dft.py`) CCS pipelines, and CCS-based anchored-offset matching. Keep the DFT
machinery only for the relative-energy / proton-affinity ranking.

Honest caveats (unchanged): the labelled set will be modest and high-confidence, not large; and a 2D GNN
may still fail to learn direction even from correct labels if the signal is inherently 3D. Decide whether
the deliverable is a direction-aware GNN or the physics ranking itself as the protomer assigner, with the
GNN as a base-CCS-only fast surrogate.

First concrete step: build the Waters-gold PA-ranking validation (step 6) -- prove the assignment method
recovers known site->CCS before any Enveda relabelling.

#### Waters-gold validation, attempted 2026-07-27 (findings)

Data reality: `other_ccs_measurements.xlsx` has only **7** fully-labelled protomer pairs (danofloxacin,
fentanyl, despropionyl-fentanyl, melphalan, para/meta-benzocaine, aniline), NOT 48. The site labels are
inconsistent free text ("N on chloroethyl side", "O carboxyl group", "N-22 (or N-19, less likely)"), and
there is **no intensity/abundance column** (the two CCS are ordered by CCS, Protomer 1 < Protomer 2, 7/7).
Consolidated table: `outputs/protomer_gold/waters_protomer_pairs.csv`.

POSITIVE finding (the direction is real and chemically interpretable): the higher-CCS (extended) protomer
is **N-protonated in 7/7**; the lower-CCS (compact) protomer is O-protonated (3), ring (1), or a
different/less-basic N (3). Charge solvation / intramolecular H-bonding compacts O- and ring-protonated
ions; N-ammonium is charge-localised and extends. This is a site-TYPE rule, which a 2D graph CAN encode,
so the model's failure was not learning this rule (it predicted a constant gap), not that direction is
unlearnable. Encouraging for assignment-labelled supervision.

PA machinery works and confirms PA != CCS-direction: `scripts/rank_gold_protomers_by_energy.py` ran real
ORCA PM3 energies on the 3 molecules with protonated SMILES. Most-stable (highest-PA, predicted base)
protomer = higher CCS for benzocaine and melphalan but = LOWER CCS for danofloxacin. So proton affinity
predicts ABUNDANCE (which peak), not CCS magnitude, as intended. (PM3 is crude -- benzocaine dE only 1.75
kcal/mol, a near-tie that could flip; production ranking must use r2SCAN-3c via the
`predict_candidate_ccs_dft.py` path.)

LIMITATION that changes the plan: because the gold set has no intensity, it **cannot validate the
abundance-assignment step** (PA -> most abundant -> base peak). It can only (a) confirm PA ranks abundance
not CCS, and (b) reveal the site-type direction rule. The core premise "most-abundant protomer = highest
PA = base peak" stays an assumption. To firm it up, either recover relative protomer intensities from the
source papers (the `Reference` DOIs; manual, small n) for a true (site, CCS, intensity) validation set, or
accept the premise and validate DOWNSTREAM: train on PA-assigned Enveda labels and test the model's
direction against the gold's known site->CCS and the N-vs-O rule. Note also: since direction is largely a
site-TYPE effect (N extended vs O/ring compact), QM PA is mainly needed to break N-vs-N ties (fentanyl-like)
and to set abundance; the coarse direction may be recoverable from site type alone.

### KEY FINDING: base-CCS accuracy is data-limited, not method-limited

Diagnostic (`configs/sigma_config_gen_adduct_fixed.json`, dims rebuilt from the checkpoint):
the good reference model `GeneralizedAdducts_SigmaCCS`, which gets ~1.3% on its own curated data,
reaches only **2.40% median-rel on the full Enveda test set** (25,720 molecules; run confirmed on the
whole set, not just a sample). Per adduct: **[M+H]+ 2.02%** (14,765), **[M-H]- 3.06%** (6,350), and the
minor adducts are worse ([M+Na]+ 3.5%, [M+NH4]+ 3.2%, [M+Cl]- 5.0%, [M+K]+ 4.6%). So Enveda has an
intrinsic ~2% ([M+H]+) to ~3% ([M-H]-) CCS ceiling; `primary_only` (1.81%) beats it and `full` (2.51%)
matches it. The "worse than expected" base CCS is an Enveda data-quality ceiling, not a defect in the
training/method. Do not chase base CCS below ~2% on this data.

Note on the reference run's gap section: it was run in neutral graph mode (a base-only, no-gap model),
so its gap_MAE equals the zero-gap baseline (10.54) and its gap_corr 0.27 ~= the permutation null 0.19.
That 0.27 is the anchored-offset matcher's floor, not a real gap signal, and is the correct control
value for "no gap capability" (contrast the trained `full` model: gap_corr 0.71).

**PROVEN FROM BOTH DIRECTIONS (2026-07-27):** the current approach trained on the curated SigmaCCS data
(`configs/sigma_config_curated.json`, single-CCS -> base regression only) reaches **1.40% median-rel on
the curated test set** ([M+H]+ 1.37%, [M-H]- 1.40%, [M+Na]+ 1.65%), matching the reference's ~1.3% on the
same data. So: reference-on-Enveda = 2.40% (a good model cannot beat the Enveda ceiling) AND
current-approach-on-curated = 1.40% (our method reaches curated-level accuracy when the data supports it).
The base-CCS "underperformance" on Enveda is definitively a data-quality ceiling, not a method deficiency.
The training procedure and architecture are sound for base CCS; the open work is entirely the gap
direction (Next-step #0).

### Next steps
0. **DECIDED: assignment-labelled supervision by proton-affinity ranking.** The gap direction is not
   learnable from CCS-matched labels (proven, `scripts/analyze_gap_direction.py`). The chosen path is to
   assign site->peak by ENERGY (proton affinity -> abundance -> intensity order) and use the observed CCS
   as the label, computing NO CCS. Full plan in the CHOSEN DIRECTION section above. **First concrete step:
   build the Waters-gold PA-ranking validation** -- protonate `data/Waters/other_ccs_measurements.xlsx` at
   its labelled sites, run the proton-affinity ranking, and confirm it recovers the known site->CCS before
   any Enveda relabelling. Then: purity gate -> rank -> align -> label -> direction-aware loss; acceptance
   = external gold directional accuracy, not the matcher-based Enveda gap_corr.
1. **`full` retrained on clean data: DONE** (base 2.80%, gap held at 0.727, gold 1/3). Still to run:
   **`primary_only` on the cleaned corpus** to confirm the base ceiling (~1.8-2%) with no gap-term tax.
2. **Data-vs-method control: train the CURRENT approach on the curated (old) SigmaCCS data. DONE.**
   Result: **1.40% median-rel** on the curated test ([M+H]+ 1.37%), matching the reference's ~1.3% on
   the same data. This confirms the method reaches curated-level base accuracy when the data supports
   it; Enveda's ~2% is purely a data ceiling. (`scripts/build_curated_corpus.py`,
   `configs/sigma_config_curated.json`; native SigmaCCS splits so the number is directly comparable to
   the reference; protomer 17-feature config, the two protomer features inert on neutral graphs.)
3. **`gap_lambda` sweep -- now mainly a base-recovery question.** Since the gap direction is not real
   (Next-step #0), the gap term is buying inflated metrics at the cost of base accuracy. Sweep toward
   LOWER `gap_lambda` (and consider `primary_only` as the shipping base model) to recover base toward
   1.8%. Do not tune to maximise the matcher-based gap_corr -- that metric is not trustworthy.
4. **Gold directional check** is now the acceptance metric for any gap work, but must be measured on the
   EXPANDED Waters set (Next-step #0), not the n=3 file. The current model fails it (1/3); do not
   re-report the 3-molecule spot-check as validation.
5. **Data-scale iteration (later):** merge the curated TIMS add-ons (CCSBase TIMS, PFAS); then the
   deferred non-TIMS sets + IM-technique/gas global feature to pool instruments (Section 1/2.5).

---

## Decisions locked with the user

- **Training objective:** anchored set loss (below). Base peak pinned to the dominant protomer;
  remaining observed peaks matched permutation-invariantly to remaining candidates.
- **Instrument scope:** timsTOF/N2 only for this first model (Enveda, plus the TIMS subset of
  CCSBase and PFAS). This is the regime where the protomer gap is meaningful and Enveda is the only
  multi-CCS source. Pooling other IM techniques (DT/TW) via an instrument/gas global feature is
  deferred to a later iteration (small-molecule CCS is reasonably transferable, so it is a data-scale
  add-on, not needed to prove the method). The cross-instrument Waters/literature gold protomer set
  is still used for evaluation only, as a directional gap check.
- **Adduct scope:** stay general-adduct. Protomer enumeration and the gap objective apply to
  [M+H]+ and [M-H]- only; metal/other adducts contribute base-CCS regression using the existing
  neutral-SMILES + adduct-descriptor path (no protomer enumeration).
- **Dominant-site choice:** Dimorphite-DL pKa plus a valence/basicity heuristic for O/S gaps
  (cheap, already in `enumerate_protonation_candidates.py`).

## 1. Data setup

### Unified per-(molecule, adduct) record
Build one canonical training table where each row is a (molecule, adduct) with:
- `parent_smiles` (neutral), `inchikey`, `adduct`, `charge_state`
- `observed_ccs_peaks`: list of observed CCS, primary (base peak) first. Single-CCS sources give a
  1-element list; Enveda [M+H]+/[M-H]- give the intensity-ordered `ccs_species` list after cleaning.
- `candidate_smiles` / `charged_atom_idx` / `site_names` / `pka_mean`: the enumerated protomer set
  (H+/- only); metal adducts carry a single charge-positioned or neutral+adduct representation.
- `im_type` (TIMS/DT/TW), `instrument`, `source`, `dominant_site_idx`.

### Sources and roles
| Source | rows / mols | role |
|---|---|---|
| Enveda-180 (timsTOF/N2) | 184,944 mols; 191,389 multi-CCS (SMILES,adduct) | protomer-gap signal (H+/-) + base CCS. **Core dataset.** |
| CCSBase (TIMS subset) | ~2,950 TIMS rows | base CCS (TIMS only) |
| PFAS | ~160 mols (TIMS, neg) | base CCS, negative-mode |
| CCSBase DT/TW, GraphCCS, SigmaCCS (DT), Waters (TWIMS) | | **deferred** (non-TIMS; add later with an instrument feature) |
| Waters `other_ccs_measurements.xlsx` | 48 protomer pairs, 7-10 proton-site labelled | **gold protomer eval only (held out)** |
| METLIN | no SMILES | excluded (unusable as-is) |

For this first (timsTOF-only) model, base CCS is essentially Enveda's primary peaks (184,944
molecules is ample), plus the TIMS subset of CCSBase and PFAS. Use
`data/Enveda-180/enveda-180-dedup.csv` (`ccs_species`, IM-peak intensity-ordered, base peak first)
as the core; filter `data/all_CCS_data_standardized.csv` to `CCS Type == "TIMS"` for the curated
TIMS add-ons. The non-TIMS curated sets and the IM-technique global feature are a later data-scale
iteration, not part of proving the method.

### Protomer candidate generation (reuse existing)
Run `scripts/enumerate_protonation_candidates.py` to produce, per H+/- molecule, the
symmetry-collapsed N/O/S (or Dimorphite acidic) candidate set with `candidate_smiles` already
charge-positioned via `build_candidate_smiles_generic` (+1) / `build_candidate_smiles` (-1). This is
the exact charged-SMILES bridge the model featurizes. Extend enumeration to the curated single-CCS
H+/- molecules too (not just Enveda) so base training uses a charged structure; pick
`dominant_site_idx` by max `pka_mean` with a basicity/valence tiebreak.

### Cleaning / filtering
- Drop dimers (`ratio = sp_ccs/ccs_f > 1.25`).
- Secondary peaks kept for the gap objective only inside the conformer/protomer band
  (`|ratio-1|` in [2%, 9%], the tightened band) AND for molecules with >=2 comparable-pKa/PA sites
  (credible-protomer filter), to avoid teaching the model that conformer/artifact gaps are
  protonation effects. Downweight/flag likely-conformer cases using the Model-1 mechanistic signals.
- Single-CCS everywhere else feeds base regression only.

### Splits (prevent leakage)
- Split on **molecule identity, scaffold-grouped** (Bemis-Murcko or InChIKey connectivity block),
  never per-row, so protomers/adducts/replicates of one molecule stay in one split.
- Deduplicate connectivity blocks across CCSBase/GraphCCS/SigmaCCS first (they are ~96 to 97% the
  same compound pool). Enveda is compound-disjoint from all curated sets (0 shared blocks), so it
  can be split independently.
- Hold out entirely for evaluation: the Waters gold protomer set and a stratified Enveda multi-CCS
  test set (by scaffold + gap size). Existing `protomer_eval/protomer_test_df.csv` (has
  `protonated_SMILES`) is reused as an additional held-out protomer benchmark.

## 2. Architecture changes (keep the SigmaModel/ECCConv backbone and the 1-scalar head)

All changes are additive; the message-passing core and output head are unchanged, satisfying "keep
the architecture reasonably similar" and "no conformer generation at inference".

1. **`is_protonation_site` boolean atom feature** (new featurizer class in
   `graphamole/graphamole/featurizers/atom.py`, registered in `featurizers/__init__.py`): flags the
   `charged_atom_idx` so the model can attend to where the proton sits, not just infer it from
   formal-charge deltas. Directly targets the Step-14 insensitivity finding.
2. **Re-enable the Gasteiger partial-charge atom featurizer** (already present but disabled in
   `ALL_FEATURIZERS`): propagates the protonation effect through the neighbourhood, not just the one
   atom. Continuous feature path already exists in the model.
3. **Charged-atom readout augmentation** (small change in `models/Sigma.py:forward`): concatenate the
   protonated atom's node embedding to the pooled graph embedding before the dense head, so the head
   is explicitly charge-site aware. Optional but recommended; ablate it.
4. **Decouple adduct-descriptor SMILES from graph SMILES** in `GraphXSection_torch/datasets.py`
   (`build_dataset_from_df`, currently feeds the same SMILES to both). Feed the **neutral** SMILES to
   `AdductFeatureExtractor.extract_features` (so mass/mz/Gasteiger-quantile terms are physical) and
   the **charged** SMILES to the graph featurizer.
5. **(Deferred) IM-technique/gas global feature**: append `im_type` (TIMS/DT/TW) and gas to the
   adduct global descriptor to pool instruments for base CCS. Not in this first timsTOF-only model;
   listed so the data schema (carry `im_type`) is ready for it.

Also fix the known config/checkpoint gotchas when starting fresh: set model dims explicitly in a new
config, and use the lowercase `adduct` column (the `sigma_config_gen_adduct.json` `"Adduct"` mismatch
would KeyError on the standardized CSV).

## 3. Training procedure

### Anchored set loss (the core new objective; in `lightning_modules.py`)
Batch is grouped by (molecule, adduct); each group carries its candidate predictions and its observed
peak set. Per group:
- **Anchor term:** the dominant candidate (`dominant_site_idx`) is regressed to the base (most
  intense) observed peak. This supervises absolute CCS and stabilises the match. Applies to every
  molecule, so single-CCS molecules reduce to standard regression.
- **Gap term (multi-CCS only):** remaining observed peaks are assigned to remaining candidates by a
  permutation-invariant best match (Hungarian on |pred - obs|), and the loss is on the matched pairs.
  This teaches the gap magnitude without asserting a specific unreliable site identity.
- Model head stays a single scalar per candidate; only batching (a grouping collate) and the loss
  change. Reuse `match_by_anchored_ccs` logic from `scripts/build_protomer_labels.py` for the match.

### Schedule: single-stage unified objective (recommended), two-stage optional

The anchored set loss already unifies both jobs: the **anchor term is exactly the base-CCS
regression** (every molecule has it), and the **gap term is added only for multi-CCS molecules**. So
the default is **one training stage on the full timsTOF corpus** (single-CCS + multi-CCS molecules
mixed together), optimizing anchor + gap jointly. There is no separate "pretrain then fine-tune"
step, and therefore no forgetting problem, because base-CCS examples are present in every batch
throughout training. Reuse the existing Adam + ReduceLROnPlateau + Huber setup.

One imbalance to manage: multi-CCS credible-protomer molecules are a minority of the corpus, so the
gap term is a minority signal. Handle it with a **gap-term weight (lambda)** and/or **oversampling of
multi-CCS molecules** in the sampler, tuned on the held-out gap metric, so the gap signal is not
drowned by the base-CCS majority. This is the only real knob the single-stage design adds.

**On "forgetting" (why two-stage is risky):** catastrophic forgetting is what happens if you first
train base CCS, then fine-tune on gap-only data (the multi-CCS subset alone). The weights drift to
fit that narrow subset and the model degrades on the broad base-CCS task it had learned, i.e. it
"forgets" general accuracy. The single-stage unified objective sidesteps this entirely. If a
two-stage curriculum is ever wanted (learn absolute CCS first, then emphasize the gap), it must use
**rehearsal**: keep a large fraction of base-CCS molecules in the stage-2 batches (not gap-only),
which is exactly what the single-stage design does by construction. So two-stage buys nothing here
and is not recommended.

### Config additions (new `configs/sigma_config_protomer.json`)
`observed_ccs_peaks` column, candidate-group key, `loss_function: "anchored_set"`, band and
credible-protomer filter thresholds, `im_feature: true`, `is_protonation_site_feature: true`,
`charged_atom_readout: true`, and explicit model dims.

## 4. Evaluation (this is where "protomer-aware" is proven)

**Matched-baseline principle (applies to every comparison below):** never compare the new model
against the pre-existing `GeneralizedAdducts_SigmaCCS` checkpoint directly, because that checkpoint was
trained on different (mixed-instrument, differently-split) data, so any difference confounds method
with data. Instead **retrain the old procedure** (current featurization, plain single-CCS regression,
no protomer changes) on the **same timsTOF training molecules and the same splits**, and evaluate it
on the **same held-out set**. Every claimed improvement is new-procedure vs matched-baseline on
identical data. Each architecture change (Section 2) is likewise A/B'd on the same data via ablation.

**Exception (data-ceiling diagnostic, not an improvement claim):** the pre-existing
`GeneralizedAdducts_SigmaCCS` checkpoint *is* run directly on the Enveda test set, precisely to
measure the data floor. Because it was trained on curated data and still only reaches ~2.4% on Enveda,
it tells us the ceiling is the data, not our method. This is a diagnostic about the data, so the
different-training-data confound is the point rather than a problem. See the STATUS block above.

1. **Base CCS accuracy** on the held-out molecule split: MAE / median relative error, per adduct.
   The new model must stay comparable to the matched baseline (do not regress base performance while
   adding protomer awareness).
2. **Protomer-gap evaluation** on the held-out Enveda multi-CCS set and the Waters gold set: for each
   molecule predict CCS for every enumerated protomer, form the predicted peak set, and compare to the
   observed set with a set-matched gap error. Report the gap correlation and, crucially, **beat the
   protonation-blind baseline** (GraphXSection / a zero-gap model) on the gap specifically. Also run
   the permutation null (wrong-structure candidate predictions) to confirm the trained model's gap
   signal is structure-specific, not matcher flexibility.
3. **Ablations:** charged vs neutral SMILES input; `is_protonation_site` on/off; Gasteiger on/off;
   charged-atom readout on/off. Each should move the gap metric if it matters.
4. **Directional spot-checks:** danofloxacin (keto-O+ 181.7 vs amine-N+ 193.7) and the fluoroquinolone
   / benzocaine / aminobenzoic-acid gold cases predicted in the correct direction and magnitude band.

## Files to create / modify

Create:
- `scripts/build_training_corpus.py` — assemble the unified per-(molecule, adduct) table (base +
  Enveda peak sets + enumerated candidates + dominant site), apply cleaning/band/credible-protomer
  filters, and write scaffold-grouped train/val/test splits with connectivity-block dedup.
- `configs/sigma_config_protomer.json` — new config with correct dims and the new keys.
- `scripts/eval_protomer_gap.py` (or a notebook under `protomer_eval/`) — the gap evaluation + null +
  ablation harness; reuse `protomer_eval/protomer_test_df.csv` and the Waters gold set.

Modify (all additive):
- `graphamole/graphamole/featurizers/atom.py` + `featurizers/__init__.py` — add `IsProtonationSite`,
  re-enable `GasteigerCharge`.
- `GraphXSection_torch/datasets.py` (+ `data_extract.py`) — decouple adduct-descriptor SMILES from
  graph SMILES; molecule-grouped/scaffold split function; per-molecule candidate-group dataset +
  grouping collate; `observed_ccs_peaks` support.
- `GraphXSection_torch/lightning_modules.py` — the `anchored_set` loss.
- `GraphXSection_torch/models/Sigma.py` — optional charged-atom readout augmentation.

Reuse as-is: `scripts/enumerate_protonation_candidates.py` (+ `build_candidate_smiles_generic`),
`AdductFeatureExtractor`, `GraphXSectionDataModule`, `train_lightning_GraphXSection`,
`evaluate_lightning_model`, `match_by_anchored_ccs`. Retire `.archive/proton_affinity_calculator.py`
in favour of the per-site enumeration.

## Verification

- **Featurizer unit checks:** a known protonated SMILES (e.g. an `[NH3+]` amine) yields a graph that
  differs from the neutral only at the expected atoms; `is_protonation_site` flags exactly
  `charged_atom_idx`.
- **Overfit one multi-CCS molecule:** the anchored set loss drives the predicted protomer gap to the
  observed gap (sanity that the loss + grouping work).
- **No-leakage assertion:** no InChIKey/connectivity block appears in more than one split.
- **Base parity:** the new model's base-CCS MAE on held-out molecules is within noise of the
  matched baseline (old procedure retrained on the identical timsTOF molecules/splits), not the
  differently-trained pre-existing checkpoint.
- **Gap payoff (the acceptance test):** on the held-out Enveda set and Waters gold set, the trained
  model's set-matched gap error beats the protonation-blind baseline and its gap signal survives the
  wrong-structure permutation null; danofloxacin and the fluoroquinolone gold cases are directionally
  correct.
