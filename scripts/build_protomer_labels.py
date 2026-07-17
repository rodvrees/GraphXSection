#!/usr/bin/env python3
"""
Step 2 of protomer_ccs_predictor_roadmap.md: label construction, from Step 1's
output (scripts/enumerate_protonation_candidates.py).

Produces two Parquet files:

1. Model 1 labels (protomer-formation classifier), one row per (inchikey,
   charge_state), from the roadmap's ratio-band table:

     spread > 1.25        dimer / co-eluting species        excluded
     1.08 <= spread <=1.20 benzocaine-magnitude split        positive
     1.03 <= spread < 1.08 ambiguous (protomer or conformer) hold out
     0.97 <= spread < 1.03 suspected peak-picking artifact   negative, noisy
     no secondary peak     --                                negative, noisy

   `spread = max(ratio, 1/ratio)` where ratio = sp_ccs / ccs_f: the roadmap's
   table is written for ratio >= 1 (secondary peak larger than primary), but
   ratio can go below 1 in the data (a genuinely more compact secondary
   conformer), and the literature this table is drawn from (Warnke et al.,
   Bull et al.) reports protomer/conformer CCS separation as a magnitude, not
   a signed value. Folding the ratio this way keeps every observed value
   inside the table's bands instead of silently falling outside it.

   The narrow 1.20-1.25 window isn't named in the roadmap's own table; it is
   treated as "excluded", extending predict_multi.ipynb's original ambiguous-
   gap precedent (1.15-1.25) rather than inventing a new rule.

2. Model 2 pairs (protomer-resolved CCS regressor training pairs), one row
   per (spectrum, matched candidate):
     - Unambiguous case (roadmap): a molecule with exactly one candidate for
       its polarity and exactly one observed CCS peak -> that candidate gets
       that CCS value directly, match_confidence="unambiguous_single_site".
     - Everything else (multiple candidates and/or multiple peaks) ranks
       candidates by a proton-affinity proxy (see rank_candidates()):
       Dimorphite-DL's tabulated pka_mean (descending, free, the default), or
       -- if --pm3-energies is given (scripts/rank_candidates_pm3.py's
       output) and every candidate for that molecule converged -- ORCA PM3
       energy (ascending; the roadmap's proxy (a), "a fast semi-empirical
       proton affinity estimate", using PM3 in place of GFN2-xTB since no
       xtb binary is available in this environment, see rank_candidates_pm3.py
       for why). Observed CCS peaks are ranked by intensity, most intense
       first, as given in the raw `ccs_species` field order (NOT sorted by
       CCS value -- confirmed directly against the data that the field's
       given order is intensity rank, with its first entry matching the
       reported `ccs`/primary value in effectively all rows). Peaks are then
       paired to candidates by rank: rank-1 (most intense/most populated)
       peak with rank-1 (most stable per the ranking source used) candidate,
       and so on. match_confidence="weak_rank_pka_mean" or
       "weak_rank_pm3_energy" (see `ranking_source` column). This is a
       documented, best-effort heuristic either way, NOT a validated
       peak-to-site assignment -- intensity plausibly tracks population
       fraction, and population plausibly favors the more stable site, but
       neither link is confirmed here. It exists to produce auditable,
       clearly-flagged weak supervision (per the roadmap's own framing), not
       ground truth. The roadmap's fuller proxy chain (rank by ML-predicted
       CCS, then bipartite-match to observed peaks) needs an existing CCS
       predictor scored on charge-positioned candidate SMILES, which this
       script does not attempt -- this repo's GraphXSection_torch checkpoints
       are trained on neutral SMILES + adduct strings, not charge-positioned
       candidates, so plugging one in is a separate adaptation task, not a
       drop-in call.

Usage:
    python scripts/build_protomer_labels.py \\
        --spectra-input data/Enveda-180/enveda-180.jsonl.gz \\
        --candidates-input outputs/protonation_candidates.parquet \\
        --model1-output outputs/model1_labels.parquet \\
        --model2-output outputs/model2_pairs.parquet \\
        --pm3-energies outputs/pm3_energies.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from enumerate_protonation_candidates import (
    DIMER_THRESH,
    build_secondary_species_table,
    load_spectra,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BAND_PROTOMER = "protomer_band"
BAND_AMBIGUOUS = "ambiguous"
BAND_AMBIGUOUS_GAP = "ambiguous_gap"
BAND_DIMER = "dimer"
BAND_ARTIFACT = "artifact"

ADDUCT_REGEX_BY_CHARGE = {1: r"\[M\+H\]", -1: r"\[M-H\]"}


# --------------------------------------------------------------------------- #
# Model 1: molecule-level protomer-formation labels
# --------------------------------------------------------------------------- #


def classify_ratio_band(ratio: float) -> str:
    spread = max(ratio, 1.0 / ratio)
    if spread > 1.25:
        return BAND_DIMER
    if spread > 1.20:
        return BAND_AMBIGUOUS_GAP
    if spread >= 1.08:
        return BAND_PROTOMER
    if spread >= 1.03:
        return BAND_AMBIGUOUS
    return BAND_ARTIFACT


def aggregate_model1_label(bands_seen: set[str], has_any_secondary: bool) -> tuple[int, str]:
    """Precedence: positive evidence wins outright; among negative outcomes,
    the least-resolved evidence (ambiguous) is reported before the more
    clearly-resolved ones, matching the roadmap's "hold out or soft-label"
    treatment of the ambiguous band as distinct from a flat exclusion."""
    if BAND_PROTOMER in bands_seen:
        return 1, "positive"
    if BAND_AMBIGUOUS in bands_seen:
        return -1, "ambiguous_hold_out"
    if BAND_DIMER in bands_seen or BAND_AMBIGUOUS_GAP in bands_seen:
        return -1, "excluded_dimer_or_gap"
    if BAND_ARTIFACT in bands_seen:
        return 0, "negative_artifact"
    if not has_any_secondary:
        return 0, "negative_no_secondary"
    return -1, "excluded_unclassified"  # pragma: no cover - defensive only


def build_model1_labels(spectra_df: pd.DataFrame, multi_df: pd.DataFrame, charge_state: int) -> pd.DataFrame:
    adduct_regex = ADDUCT_REGEX_BY_CHARGE[charge_state]
    df_pol = spectra_df[spectra_df["adduct"].str.contains(adduct_regex, regex=True, na=False)]
    multi_pol = multi_df[multi_df["adduct"].str.contains(adduct_regex, regex=True, na=False)].copy()
    multi_pol["band"] = multi_pol["ratio"].apply(classify_ratio_band)

    bands_by_mol = multi_pol.groupby("inchikey")["band"].apply(set)

    rows = []
    for inchikey in df_pol["inchikey"].unique():
        bands_seen = bands_by_mol.get(inchikey, set())
        has_secondary = inchikey in bands_by_mol.index
        label, detail = aggregate_model1_label(bands_seen, has_secondary)
        rows.append(
            {
                "inchikey": inchikey,
                "charge_state": charge_state,
                "model1_label": label,
                "model1_label_detail": detail,
                "ratio_bands_seen": ",".join(sorted(bands_seen)),
            }
        )

    result = pd.DataFrame(rows)
    logger.info(
        "Model 1 labels (charge=%+d): %d molecules -- %s",
        charge_state, len(result),
        result["model1_label_detail"].value_counts().to_dict(),
    )
    return result


# --------------------------------------------------------------------------- #
# Model 2: candidate-to-observed-CCS training pairs
# --------------------------------------------------------------------------- #


def parse_ccs_species_by_intensity(s: object) -> list[float]:
    """Parse `ccs_species` WITHOUT sorting by value. Unlike Step 1's
    parse_ccs_species (which sorts ascending -- fine for ratio-band
    computation, where order doesn't matter), the raw field order here is
    intensity rank: most intense peak first. Confirmed directly against the
    data: the first entry equals the reported `ccs` (primary/most intense)
    value in 345,825/345,829 CSV rows and 289,282/289,282 sampled jsonl.gz
    rows. Sorting by CCS value, as an earlier version of this script did,
    destroys that signal -- CCS magnitude has no relationship to which peak
    is most abundant."""
    if pd.isna(s) or not str(s).strip():
        return []
    return [float(x) for x in str(s).split(",") if x.strip()]


def build_peak_table(spectra_df: pd.DataFrame, charge_state: int) -> pd.DataFrame:
    """One row per spectrum (title) of the given polarity, with its observed
    CCS values in intensity order (most intense/primary first)."""
    adduct_regex = ADDUCT_REGEX_BY_CHARGE[charge_state]
    df_pol = spectra_df[spectra_df["adduct"].str.contains(adduct_regex, regex=True, na=False)].copy()
    df_pol = df_pol[df_pol["ccs_f"].notna() & (df_pol["ccs_f"] > 0)]

    n_primary_mismatch = 0

    def peak_values(row) -> list[float]:
        nonlocal n_primary_mismatch
        species = [v for v in parse_ccs_species_by_intensity(row["ccs_species"]) if v > 0]
        if not species:
            return [row["ccs_f"]]
        if abs(species[0] - row["ccs_f"]) > 1e-3:
            # Rare data-quality mismatch (~4 in 345,829 CSV rows) -- trust the
            # dedicated `ccs` column for which value is primary rather than
            # silently keeping whatever ccs_species happened to list first.
            n_primary_mismatch += 1
            species = [row["ccs_f"]] + [v for v in species if v != row["ccs_f"]]
        return species

    df_pol["peak_values"] = df_pol.apply(peak_values, axis=1)
    if n_primary_mismatch:
        logger.warning(
            "%d/%d spectra (charge=%+d) had ccs_species[0] != ccs; reordered to keep ccs primary",
            n_primary_mismatch, len(df_pol), charge_state,
        )
    return df_pol[["title", "inchikey", "ccs_f", "peak_values"]]


def rank_candidates(group: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rank one molecule's candidates, most-favored site first.

    Prefers ORCA PM3 energy (ascending -- lowest energy is the most stable
    protonation/deprotonation site, a real QM proxy for proton affinity; see
    scripts/rank_candidates_pm3.py) when EVERY candidate in the group has a
    converged ("ok") PM3 result. Falls back to Dimorphite-DL's tabulated
    pka_mean (descending) otherwise -- either because PM3 wasn't run for this
    molecule at all, or because it failed/didn't converge for at least one of
    its candidates. This is deliberately all-or-nothing per molecule: ranking
    some candidates by PM3 energy and others by pKa within the same group
    would compare two different, non-commensurate quantities."""
    has_pm3 = "pm3_energy_hartree" in group.columns
    if has_pm3 and (group["pm3_status"] == "ok").all():
        return group.sort_values("pm3_energy_hartree", ascending=True).reset_index(drop=True), "pm3_energy"
    return group.sort_values("pka_mean", ascending=False).reset_index(drop=True), "pka_mean"


# --------------------------------------------------------------------------- #
# Step 4 (revised plan): nearest-CCS matching. Instead of pairing peaks to
# candidates by a proton-affinity / pKa *ranking* (which assumes most-stable =
# most-intense -- broken by kinetic trapping, per the danofloxacin ground-truth
# finding), assign each observed peak to the candidate whose QM-predicted CCS is
# closest to it. Single-site molecules carry no protomer signal and are dropped.
# pKa (solution) and a within-molecule relative proton-affinity descriptor (from
# QM energies) are carried as *descriptor* columns, never as a matching gate.
# --------------------------------------------------------------------------- #

HARTREE_TO_KCAL = 627.5095


def relative_pa_kcal(energies_hartree: "pd.Series | list[float]") -> list[float]:
    """Within-molecule relative gas-phase proton affinity descriptor, in
    kcal/mol, one value per candidate.

    PA of a site = -(E_protonated - E_neutral) * 627.5095. The neutral reference
    is common to every site of the same molecule, so it cancels when comparing
    sites *within* a molecule: the most stable protonated site (lowest energy)
    has the highest PA. We report each site as its destabilization above that
    most stable site, i.e. delta = (E_i - E_min) * 627.5095 kcal/mol, so 0 marks
    the highest-PA site and larger values mark progressively lower-PA sites.
    This is a relative, within-molecule descriptor only; it is NOT an absolute
    cross-molecule PA and is deliberately NOT used to gate matching."""
    e = np.asarray(list(energies_hartree), dtype=float)
    if e.size == 0 or np.all(np.isnan(e)):
        return [float("nan")] * e.size
    return list((e - np.nanmin(e)) * HARTREE_TO_KCAL)


def match_by_nearest_ccs(
    predicted_ccs: "list[float]", observed_peaks: "list[float]"
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment of observed peaks to candidates by nearest
    predicted CCS. Returns (candidate_index, peak_index) pairs, min(n_cand,
    n_peak) of them, minimizing total |predicted - observed| (Hungarian
    algorithm). Candidates with a NaN predicted CCS are excluded from matching.

    Assignment is global (bipartite), not greedy per peak: two peaks are never
    assigned the same candidate, and the pairing minimizes total CCS residual
    rather than locking in whichever peak is processed first. For the common
    2-peak / >=2-candidate case this reduces to the intuitive nearest match but
    stays correct when candidate predictions are close together."""
    pred = np.asarray(predicted_ccs, dtype=float)
    obs = np.asarray(observed_peaks, dtype=float)
    valid = np.where(~np.isnan(pred))[0]
    if valid.size == 0 or obs.size == 0:
        return []
    cost = np.abs(pred[valid][:, None] - obs[None, :])
    rows, cols = linear_sum_assignment(cost)
    return [(int(valid[r]), int(c)) for r, c in zip(rows, cols)]


def match_by_anchored_ccs(
    predicted_ccs: "list[float]", observed_peaks: "list[float]"
) -> "tuple[list[tuple[int, int]], float, Optional[int]]":
    """Anchor-offset assignment. observed_peaks[0] is the anchor: the most
    intense / base IM peak (build_clean_peak_table puts the primary first).

      1. Assign the anchor peak to its nearest predicted candidate.
      2. offset = obs_anchor - pred_anchor; add it to every prediction. This is
         a pure translation -- it removes the model's absolute-CCS bias without
         changing any inter-candidate gap, so the offset's only effect is on
         which candidate a *secondary* peak is assigned to.
      3. Hungarian-assign the remaining candidates to the remaining peaks by
         nearest *shifted* prediction.

    Returns (matches, offset, anchor_cand_idx):
      matches          list of (candidate_index, peak_index) into the ORIGINAL
                       (unshifted) arrays, including the anchor pair; length
                       min(n_valid_cand, n_peak).
      offset           applied translation, obs_anchor - pred_anchor.
      anchor_cand_idx  candidate index chosen for the anchor peak.

    Candidates with a NaN predicted CCS are excluded. Returns ([], nan, None)
    when there is no valid prediction or no observed peak. Contrast
    match_by_nearest_ccs, which assigns on the raw (unshifted) predictions and is
    retained for provenance -- the two can disagree on the secondary assignment
    exactly when the model has a large absolute-CCS bias."""
    pred = np.asarray(predicted_ccs, dtype=float)
    obs = np.asarray(observed_peaks, dtype=float)
    valid = np.where(~np.isnan(pred))[0]
    if valid.size == 0 or obs.size == 0:
        return [], float("nan"), None

    # 1. anchor (observed peak 0) -> nearest valid prediction
    anchor_cand_idx = int(valid[int(np.argmin(np.abs(pred[valid] - obs[0])))])
    # 2. translation that pins the anchor prediction onto the anchor peak
    offset = float(obs[0] - pred[anchor_cand_idx])
    shifted = pred + offset
    matches = [(anchor_cand_idx, 0)]
    # 3. remaining candidates vs remaining peaks, on the shifted predictions
    rem_cand = np.array([c for c in valid if c != anchor_cand_idx], dtype=int)
    rem_peak = np.arange(1, obs.size)
    if rem_cand.size and rem_peak.size:
        cost = np.abs(shifted[rem_cand][:, None] - obs[rem_peak][None, :])
        rows, cols = linear_sum_assignment(cost)
        matches += [(int(rem_cand[r]), int(rem_peak[c])) for r, c in zip(rows, cols)]
    return matches, offset, anchor_cand_idx


def match_candidates_to_peaks(
    candidates_pol: pd.DataFrame, peak_table: pd.DataFrame, charge_state: int
) -> tuple[list[dict], dict[str, int]]:
    stats = {
        "unambiguous_single_site": 0, "weak_rank_pm3_energy": 0, "weak_rank_pka_mean": 0,
        "unmatched_candidates": 0, "unmatched_peaks": 0,
    }
    rows: list[dict] = []

    candidates_by_mol: dict[str, tuple[pd.DataFrame, str]] = {
        inchikey: rank_candidates(group) for inchikey, group in candidates_pol.groupby("inchikey")
    }

    for peak_row in peak_table.itertuples(index=False):
        entry = candidates_by_mol.get(peak_row.inchikey)
        if entry is None or entry[0].empty:
            stats["unmatched_peaks"] += len(peak_row.peak_values)
            continue
        candidates, ranking_source = entry

        # peak_row.peak_values is already in intensity order (most intense/
        # primary first, see build_peak_table) -- rank 1 peak (most abundant,
        # thus presumably most populated species) pairs with rank 1 candidate
        # (most stable per ranking_source, thus presumably the thermo-
        # dynamically favored/most populated site). Still a proxy, not a
        # validated match, per the match_confidence flag below.
        peaks_ranked = peak_row.peak_values
        n_pairs = min(len(candidates), len(peaks_ranked))
        confidence = (
            "unambiguous_single_site"
            if len(candidates) == 1 and len(peaks_ranked) == 1
            else f"weak_rank_{ranking_source}"
        )

        for i in range(n_pairs):
            candidate = candidates.iloc[i]
            observed_ccs = peaks_ranked[i]
            rows.append(
                {
                    "inchikey": peak_row.inchikey,
                    "title": peak_row.title,
                    "charge_state": charge_state,
                    "candidate_smiles": candidate["candidate_smiles"],
                    "charged_atom_idx": candidate["charged_atom_idx"],
                    "observed_ccs": observed_ccs,
                    "peak_role": "primary" if i == 0 else "secondary",
                    "match_confidence": confidence,
                    "ranking_source": ranking_source,
                    "pka_mean": candidate["pka_mean"],
                    "pm3_energy_hartree": candidate.get("pm3_energy_hartree"),
                    "candidate_rank": i + 1,
                }
            )
            stats[confidence] += 1

        stats["unmatched_candidates"] += max(0, len(candidates) - n_pairs)
        stats["unmatched_peaks"] += max(0, len(peaks_ranked) - n_pairs)

    return rows, stats


def build_model2_pairs(
    spectra_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    charge_state: int,
    pm3_energies_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    candidates_pol = candidates_df[
        (candidates_df["charge_state"] == charge_state) & candidates_df["candidate_smiles"].notna()
    ].copy()
    if pm3_energies_df is not None:
        candidates_pol = candidates_pol.merge(
            pm3_energies_df[["inchikey", "charge_state", "candidate_smiles", "pm3_energy_hartree", "pm3_status"]],
            on=["inchikey", "charge_state", "candidate_smiles"],
            how="left",
        )
    peak_table = build_peak_table(spectra_df, charge_state)

    rows, stats = match_candidates_to_peaks(candidates_pol, peak_table, charge_state)
    result = pd.DataFrame(rows)
    logger.info("Model 2 pairs (charge=%+d): %d pairs -- %s", charge_state, len(result), stats)
    return result, stats


# --------------------------------------------------------------------------- #
# Step 4 nearest-CCS Model 2 builder (supersedes the rank-based path above)
# --------------------------------------------------------------------------- #

# Conformer/protomer band on the secondary-to-primary CCS ratio, confirmed by
# visual inspection of the ratio distribution: keep secondary peaks whose
# |ratio - 1| is in [2%, 15%]. Below 2% is peak-picking noise / repeatability;
# above 15% (up to the 1.25 dimer threshold) is treated as ambiguous co-elution
# and left out of Model 2, since matching a candidate to a peak we do not trust
# as a real protomer/conformer would only inject bad labels. Danofloxacin's
# real split (6.6%) sits inside this band; the roadmap's older >=8% (spread
# 1.08) "protomer" floor would have excluded it.
CONFORMER_BAND_LOW = 0.02
CONFORMER_BAND_HIGH = 0.15
PEAK_CLUSTER_TOL = 0.01  # collapse secondary peaks within 1% of each other


def build_clean_peak_table(
    spectra_df: pd.DataFrame,
    charge_state: int,
    band_low: float = CONFORMER_BAND_LOW,
    band_high: float = CONFORMER_BAND_HIGH,
) -> pd.DataFrame:
    """One row per inchikey: observed peaks for nearest-CCS matching, primary
    first. Primary = median primary CCS across the molecule's spectra of this
    polarity. Secondary peaks = clean secondaries (ratio <= DIMER_THRESH, so no
    dimers) whose |ratio - 1| falls in the conformer band [band_low, band_high];
    near-duplicate secondaries (within PEAK_CLUSTER_TOL) are collapsed to their
    median. This replaces build_peak_table, which took every ccs_species value
    with NO dimer filter and so admitted dimer peaks as "secondary"."""
    adduct_regex = ADDUCT_REGEX_BY_CHARGE[charge_state]
    pol = spectra_df[spectra_df["adduct"].str.contains(adduct_regex, regex=True, na=False)].copy()
    pol = pol[pol["ccs_f"].notna() & (pol["ccs_f"] > 0)]
    primary_by_mol = pol.groupby("inchikey")["ccs_f"].median()

    multi = build_secondary_species_table(spectra_df)
    multi = multi[multi["adduct"].str.contains(adduct_regex, regex=True, na=False)].copy()
    multi["gap"] = (multi["ratio"] - 1.0).abs()
    band = multi[
        (multi["ratio"] <= DIMER_THRESH) & (multi["gap"] >= band_low) & (multi["gap"] <= band_high)
    ]
    # Carry each secondary's intensity_rank (from the source ccs_species order)
    # alongside its CCS, so the matched output can record how intense each
    # observed peak was relative to the base peak.
    sec_by_mol = {
        ik: list(zip(g["sp_ccs"].tolist(), g["intensity_rank"].tolist()))
        for ik, g in band.groupby("inchikey")
    }

    rows = []
    for inchikey, prim in primary_by_mol.items():
        # Sort by CCS value only to make near-duplicate clustering well-defined;
        # this is NOT the peak order used downstream (the anchor is the primary,
        # peak_values[0]). Each cluster keeps the strongest (lowest) rank of its
        # members as its representative intensity_rank.
        secs = sorted(sec_by_mol.get(inchikey, []), key=lambda t: t[0])
        clusters: list[list[tuple[float, int]]] = []
        for v, rk in secs:
            if clusters and abs(v - clusters[-1][-1][0]) / v <= PEAK_CLUSTER_TOL:
                clusters[-1].append((v, rk))
            else:
                clusters.append([(v, rk)])
        sec_vals = [float(np.median([t[0] for t in c])) for c in clusters]
        sec_ranks = [int(min(t[1] for t in c)) for c in clusters]
        rows.append(
            {
                "inchikey": inchikey,
                "peak_values": [float(prim)] + sec_vals,  # primary first (anchor)
                "peak_ranks": [0] + sec_ranks,  # intensity rank; 0 = base peak
                "n_secondary": len(sec_vals),
            }
        )
    return pd.DataFrame(rows)


def build_model2_pairs_ccs(
    spectra_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    predicted_ccs_df: pd.DataFrame,
    charge_state: int,
    ccs_column: str,
    band_low: float = CONFORMER_BAND_LOW,
    band_high: float = CONFORMER_BAND_HIGH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Anchored-CCS Model 2 pairing (Step 4). Each molecule's most intense (base)
    peak anchors a per-molecule CCS offset (match_by_anchored_ccs): the anchor
    peak is pinned to its nearest predicted candidate, that offset is applied to
    all predictions to remove the model's absolute-CCS bias, and the remaining
    peaks are then assigned to the remaining candidates by nearest *shifted*
    prediction. Assignment is NOT by any energy/pKa ranking. Drops single-site
    molecules and molecules with no in-band secondary peak. pka_mean and a
    within-molecule relative-PA descriptor (rel_pa_kcal, from dft_energy_hartree
    when present) are carried as descriptor columns only.

    Provenance: the non-shifted nearest-CCS assignment (match_by_nearest_ccs) is
    computed alongside and recorded per peak -- `raw_match_candidate_smiles` and
    `assignment_changed` flag exactly where anchoring reassigned a secondary. Both
    the shifted and raw residuals are kept (`ccs_residual` = |shifted - obs|,
    `ccs_residual_raw` = |raw - obs|) so the absolute-bias removal is auditable."""
    cand = candidates_df[
        (candidates_df["charge_state"] == charge_state) & candidates_df["candidate_smiles"].notna()
    ].copy()
    merge_cols = ["inchikey", "charge_state", "candidate_smiles"]
    pred_cols = merge_cols + [c for c in predicted_ccs_df.columns if c not in cand.columns]
    cand = cand.merge(predicted_ccs_df[pred_cols], on=merge_cols, how="left")

    clean = build_clean_peak_table(spectra_df, charge_state, band_low, band_high)
    peaks_by_mol = {r.inchikey: r.peak_values for r in clean.itertuples(index=False)}
    ranks_by_mol = {r.inchikey: r.peak_ranks for r in clean.itertuples(index=False)}

    stats = {
        "pairs": 0, "molecules_matched": 0, "molecules_assignment_changed": 0,
        "dropped_single_site": 0, "dropped_no_in_band_secondary": 0,
        "dropped_no_prediction": 0,
    }
    rows: list[dict] = []
    for inchikey, group in cand.groupby("inchikey"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            stats["dropped_single_site"] += 1
            continue
        peak_values = peaks_by_mol.get(inchikey)
        if peak_values is None or len(peak_values) < 2:
            stats["dropped_no_in_band_secondary"] += 1
            continue
        pred = group[ccs_column].tolist()
        matches, offset, _ = match_by_anchored_ccs(pred, peak_values)
        if not matches:
            stats["dropped_no_prediction"] += 1
            continue
        # Non-shifted nearest-CCS assignment, kept purely for provenance.
        raw_cand_by_peak = {p: c for c, p in match_by_nearest_ccs(pred, peak_values)}
        peak_ranks = ranks_by_mol.get(inchikey, [0] * len(peak_values))
        rel_pa = (
            relative_pa_kcal(group["dft_energy_hartree"])
            if "dft_energy_hartree" in group.columns else [float("nan")] * len(group)
        )
        stats["molecules_matched"] += 1
        molecule_changed = False
        for cand_idx, peak_idx in matches:
            c = group.iloc[cand_idx]
            obs = peak_values[peak_idx]
            raw_pred = c[ccs_column]
            shifted_pred = (raw_pred + offset) if pd.notna(raw_pred) else float("nan")
            raw_cand_idx = raw_cand_by_peak.get(peak_idx)
            raw_smiles = (
                group.iloc[raw_cand_idx]["candidate_smiles"] if raw_cand_idx is not None else None
            )
            changed = raw_smiles is not None and raw_smiles != c["candidate_smiles"]
            molecule_changed = molecule_changed or changed
            rows.append(
                {
                    "inchikey": inchikey,
                    "charge_state": charge_state,
                    "candidate_smiles": c["candidate_smiles"],
                    "charged_atom_idx": c["charged_atom_idx"],
                    "observed_ccs": obs,
                    "predicted_ccs": raw_pred,
                    "predicted_ccs_shifted": shifted_pred,
                    "anchor_offset": offset,
                    "is_anchor": peak_idx == 0,
                    "ccs_residual": abs(shifted_pred - obs) if pd.notna(shifted_pred) else float("nan"),
                    "ccs_residual_raw": abs(raw_pred - obs) if pd.notna(raw_pred) else float("nan"),
                    "peak_role": "primary" if peak_idx == 0 else "secondary",
                    "peak_intensity_rank": int(peak_ranks[peak_idx]) if peak_idx < len(peak_ranks) else -1,
                    "match_confidence": "ccs_anchored",
                    "raw_match_candidate_smiles": raw_smiles,
                    "assignment_changed": changed,
                    "pka_mean": c.get("pka_mean"),
                    "rel_pa_kcal": rel_pa[cand_idx],
                    "dft_energy_hartree": c.get("dft_energy_hartree"),
                    "site_names": c.get("site_names"),
                }
            )
            stats["pairs"] += 1
        if molecule_changed:
            stats["molecules_assignment_changed"] += 1

    result = pd.DataFrame(rows)
    logger.info("Model 2 anchored-CCS pairs (charge=%+d): %d pairs -- %s", charge_state, len(result), stats)
    return result, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


MODEL1_SCHEMA = pa.schema(
    [
        ("inchikey", pa.string()),
        ("charge_state", pa.int8()),
        ("model1_label", pa.int8()),
        ("model1_label_detail", pa.string()),
        ("ratio_bands_seen", pa.string()),
    ]
)

MODEL2_SCHEMA = pa.schema(
    [
        ("inchikey", pa.string()),
        ("title", pa.string()),
        ("charge_state", pa.int8()),
        ("candidate_smiles", pa.string()),
        ("charged_atom_idx", pa.int32()),
        ("observed_ccs", pa.float32()),
        ("peak_role", pa.string()),
        ("match_confidence", pa.string()),
        ("ranking_source", pa.string()),
        ("pka_mean", pa.float32()),
        ("pm3_energy_hartree", pa.float64()),
        ("candidate_rank", pa.int32()),
    ]
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--spectra-input", type=Path, default=Path("data/Enveda-180/enveda-180.jsonl.gz"),
        help="Path to enveda-180.jsonl.gz or enveda-180-dedup.csv (same file Step 1 was run on)",
    )
    parser.add_argument(
        "--candidates-input", type=Path, required=True,
        help="Step 1's output Parquet (scripts/enumerate_protonation_candidates.py)",
    )
    parser.add_argument("--model1-output", type=Path, required=True, help="Output Parquet for Model 1 labels")
    parser.add_argument("--model2-output", type=Path, required=True, help="Output Parquet for Model 2 pairs")
    parser.add_argument(
        "--pm3-energies", type=Path, default=None,
        help=(
            "Optional output of scripts/rank_candidates_pm3.py. When given, candidates "
            "for a molecule are ranked by PM3 energy instead of pka_mean, but only if "
            "EVERY candidate for that molecule has a converged PM3 result -- otherwise "
            "that molecule falls back to pka_mean ranking (see rank_candidates())."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on unique molecules read from --spectra-input (for smoke testing)",
    )
    args = parser.parse_args(argv)

    args.model1_output.parent.mkdir(parents=True, exist_ok=True)
    args.model2_output.parent.mkdir(parents=True, exist_ok=True)

    spectra_df = load_spectra(args.spectra_input, limit_molecules=args.limit)
    multi_df = build_secondary_species_table(spectra_df)
    candidates_df = pd.read_parquet(args.candidates_input)
    pm3_energies_df = pd.read_parquet(args.pm3_energies) if args.pm3_energies is not None else None
    if pm3_energies_df is not None:
        logger.info("Loaded %d PM3 energies from %s", len(pm3_energies_df), args.pm3_energies)

    model1_frames = [build_model1_labels(spectra_df, multi_df, charge_state) for charge_state in (1, -1)]
    model1_df = pd.concat(model1_frames, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(model1_df, schema=MODEL1_SCHEMA, preserve_index=False), args.model1_output)
    logger.info("Wrote Model 1 labels to %s", args.model1_output)

    model2_frames = []
    all_stats: dict[str, int] = {}
    for charge_state in (1, -1):
        frame, stats = build_model2_pairs(spectra_df, candidates_df, charge_state, pm3_energies_df)
        model2_frames.append(frame)
        for key, value in stats.items():
            all_stats[f"{key}_{charge_state:+d}"] = value
    model2_df = pd.concat(model2_frames, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(model2_df, schema=MODEL2_SCHEMA, preserve_index=False), args.model2_output)
    logger.info("Wrote Model 2 pairs to %s", args.model2_output)

    logger.info("Run summary:")
    for key in sorted(all_stats):
        logger.info("  %s: %d", key, all_stats[key])

    return 0


if __name__ == "__main__":
    sys.exit(main())
