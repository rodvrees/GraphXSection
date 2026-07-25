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
