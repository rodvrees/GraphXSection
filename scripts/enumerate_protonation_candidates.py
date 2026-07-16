#!/usr/bin/env python3
"""
Step 1 of protomer_ccs_predictor_roadmap.md: enumerate one charge-positioned
candidate structure per distinct ionizable site, for every molecule observed as
[M+H]+ (basic sites -> +1 candidates) or [M-H]- (acidic sites -> -1 candidates)
in the Enveda-180 dataset.

[M+H]+ enumeration is WIDENED to nitrogen, oxygen AND sulfur sites (every N/O/S
atom is a candidate protonation site; see enumerate_basic_heteroatom_sites and
build_candidate_smiles_generic). This was added after the danofloxacin
ground-truth test: its confirmed keto-O+ protomer is an oxygen protonation that
a nitrogen-only enumeration could never represent. Dimorphite-DL is still used
to *annotate* solution pKa on the basic-N sites it recognizes, but it no longer
defines the [M+H]+ candidate set, and no energy/pKa gating is applied here (a
large gas-phase energy gap does NOT rule out a second observed protomer, because
populations are set in solution and kinetically trapped -- danofloxacin's two
protomers differ by ~18 kcal/mol yet both are observed). Amide/vinylogous-amide
nitrogens are excluded (they protonate on the carbonyl O, which is enumerated).
[M-H]- (deprotonation) still uses Dimorphite-DL's acidic-site detection unchanged.

Uses Dimorphite-DL's SMARTS/pKa site-detection layer
(dimorphite_dl.protonate.detect.ProtonationSiteDetector), not its top-level
pH-window API: site detection is pH-independent, which is what "one candidate per
distinct basic site rather than per pKa-consistent microstate" requires. See
protomer_ccs_predictor_roadmap.md and the accompanying plan for the reasoning.

Dimer handling mirrors data/Enveda-180/predict_multi.ipynb:
  - adduct-string removal of "[2M...]" rows (cell 1)
  - ratio-based secondary-CCS dimer/co-eluting exclusion, DIMER_THRESH = 1.25
    (cells 2 and 4), reproduced here as molecule-level QC metadata plus the same
    diagnostic CCS-vs-m/z plot.

Input can be either the original enveda-180.jsonl.gz (title/adduct/ccs/
ccs_species/pepmass/inchikey/smiles) or enveda-180-dedup.csv (SMILES/adduct/ccs/
ccs_species/num_peaks -- no title, pepmass, or inchikey column). Format is
auto-detected from the file suffix. For the CSV: `inchikey` is computed from
SMILES via RDKit (Chem.MolToInchiKey), `pepmass_f` is the neutral exact mass
(RDKit CalcExactMolWt) rather than a true adduct-shifted precursor m/z since no
mass column exists, and `title` is a synthetic per-row id (each CSV row is
already a single aggregated spectrum, so no cross-row grouping is needed there).

Usage:
    python scripts/enumerate_protonation_candidates.py \\
        --input data/Enveda-180/enveda-180.jsonl.gz \\
        --output outputs/protonation_candidates.parquet

    python scripts/enumerate_protonation_candidates.py \\
        --input data/Enveda-180/enveda-180-dedup.csv \\
        --output outputs/protonation_candidates.parquet

    # Smoke test on a subset first:
    python scripts/enumerate_protonation_candidates.py --limit 500 \\
        --output outputs/protonation_candidates_smoke.parquet
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy import stats
from tqdm import tqdm

from dimorphite_dl.mol import MoleculeRecord
from dimorphite_dl.protonate.change import set_protonation_charge
from dimorphite_dl.protonate.data import PKaData
from dimorphite_dl.protonate.detect import ProtonationSiteDetector
from dimorphite_dl.protonate.site import ProtonationSite

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DIMER_THRESH = 1.25  # ratio threshold above which a secondary CCS is a dimer/co-eluting species (predict_multi.ipynb)

# Basic-site classes are derived from the rule (matched atom is N, SMARTS class
# name not "starred"), not hardcoded, since that rule is what the dimorphite-dl
# source (protonate/change.py) actually encodes. This constant is a regression
# guard: fail loudly if a future dimorphite-dl version's basic-site set drifts.
EXPECTED_BASIC_SITE_NAMES = {
    "AmidineGuanidine1",
    "AmidineGuanidine2",
    "Anilines_primary",
    "Anilines_secondary",
    "Anilines_tertiary",
    "Aromatic_nitrogen_unprotonated",
    "Amines_primary_secondary_tertiary",
    # Its pKa entry (idx_site=2) points at the pattern's [NH2:3] atom, not the
    # O -- confirmed by basic_site_names_from_table(), which is why that
    # function derives this set mechanically rather than only trusting a
    # hand-audited reading of site_substructures.smarts.
    "Primary_hydroxyl_amine",
}


# --------------------------------------------------------------------------- #
# Step 1: build the input molecule list (dimer removal + CCS-ratio diagnostic)
# --------------------------------------------------------------------------- #


def load_spectra(input_path: Path, limit_molecules: Optional[int] = None) -> pd.DataFrame:
    """Dispatch to the right loader by file suffix; both return the same
    canonical schema (title, adduct, ccs_f, pepmass_f, inchikey, smiles,
    ccs_species) so everything downstream is format-agnostic."""
    suffixes = "".join(input_path.suffixes).lower()
    if suffixes.endswith(".csv"):
        df = _load_spectra_csv(input_path, limit_molecules)
    else:
        df = _load_spectra_jsonl_gz(input_path, limit_molecules)

    # Adduct-string dimer removal (predict_multi.ipynb cell 1). Redundant with the
    # \[M\+H\]/\[M-H\] regexes below for those two subsets specifically (confirmed:
    # "[2M+H]+" never contains the literal substring "[M+H]"), kept anyway for
    # fidelity to the notebook and because other adduct subsets are not filtered
    # this way downstream.
    n_before = len(df)
    df = df[~df["adduct"].str.startswith("[2M", na=False)]
    logger.info("Dropped %d explicit dimer ([2M...]) rows", n_before - len(df))

    return df


def _load_spectra_jsonl_gz(input_path: Path, limit_molecules: Optional[int]) -> pd.DataFrame:
    """Stream enveda-180.jsonl.gz, keeping only the columns Step 1 needs."""
    keep = {"title", "adduct", "ccs", "ccs_species", "pepmass", "inchikey", "smiles"}
    rows: list[dict[str, Any]] = []
    seen_inchikeys: set[str] = set()

    with gzip.open(input_path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({k: d.get(k) for k in keep})
            if limit_molecules is not None:
                ik = d.get("inchikey")
                if ik:
                    seen_inchikeys.add(ik)
                if len(seen_inchikeys) >= limit_molecules:
                    break

    df = pd.DataFrame(rows)
    df["ccs_f"] = pd.to_numeric(df["ccs"], errors="coerce")
    df["pepmass_f"] = pd.to_numeric(df["pepmass"], errors="coerce")
    logger.info("Loaded %d spectra rows (%d unique inchikeys)", len(df), df["inchikey"].nunique())
    return df


def _load_spectra_csv(input_path: Path, limit_molecules: Optional[int]) -> pd.DataFrame:
    """Load enveda-180-dedup.csv (SMILES/adduct/ccs/ccs_species/num_peaks -- no
    title, pepmass, or inchikey column). Each row is already a single
    aggregated spectrum (ccs + ccs_species cover its peaks directly), so a
    synthetic per-row `title` is enough; no cross-row grouping is needed the
    way the jsonl loader's spectrum `title` implied. `inchikey` and
    `pepmass_f` are derived from SMILES via RDKit, computed once per unique
    SMILES (there are far fewer unique SMILES than rows -- about half, in the
    full file -- so this avoids redundant RDKit calls)."""
    df = pd.read_csv(input_path)
    df = df.rename(columns={"SMILES": "smiles"})
    df["title"] = df.index.astype(str)
    df["ccs_f"] = pd.to_numeric(df["ccs"], errors="coerce")

    if limit_molecules is not None:
        # Apply the cap on *raw SMILES strings* before the RDKit pass below,
        # not after computing inchikeys, since that pass is the expensive part
        # (tens of seconds over the full ~185k unique SMILES) and --limit
        # exists specifically to keep smoke tests fast. This can only differ
        # from a post-inchikey cap if two distinct SMILES strings in the file
        # resolve to the same InChIKey right at the cap boundary -- harmless
        # for a smoke test, and irrelevant when limit_molecules is None.
        first_n_smiles = df["smiles"].drop_duplicates().head(limit_molecules)
        df = df[df["smiles"].isin(set(first_n_smiles))]

    unique_smiles = df["smiles"].unique()
    logger.info("Computing InChIKey + neutral mass for %d unique SMILES (CSV has no inchikey/pepmass column)", len(unique_smiles))
    inchikey_by_smiles: dict[str, Optional[str]] = {}
    mass_by_smiles: dict[str, Optional[float]] = {}
    n_parse_failed = 0
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            inchikey_by_smiles[smi] = None
            mass_by_smiles[smi] = None
            n_parse_failed += 1
            continue
        try:
            inchikey_by_smiles[smi] = Chem.MolToInchiKey(mol)
        except Exception:  # noqa: BLE001
            inchikey_by_smiles[smi] = None
        try:
            mass_by_smiles[smi] = rdMolDescriptors.CalcExactMolWt(mol)
        except Exception:  # noqa: BLE001
            mass_by_smiles[smi] = None
    if n_parse_failed:
        logger.warning("RDKit failed to parse %d/%d unique SMILES from the CSV", n_parse_failed, len(unique_smiles))

    df["inchikey"] = df["smiles"].map(inchikey_by_smiles)
    # Neutral exact mass, NOT a true adduct-shifted precursor m/z -- see module
    # docstring. Only used for the diagnostic CCS-vs-mass trend/plot, never for
    # candidate enumeration itself.
    df["pepmass_f"] = df["smiles"].map(mass_by_smiles)
    df = df.dropna(subset=["inchikey"])

    logger.info("Loaded %d spectra rows (%d unique inchikeys) from CSV", len(df), df["inchikey"].nunique())
    return df


def parse_ccs_species(s: Any) -> list[float]:
    if pd.isna(s) or not str(s).strip():
        return []
    return sorted(float(x) for x in str(s).split(",") if x.strip())


def build_secondary_species_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce predict_multi.ipynb cell 2: explode multi-CCS spectra into (title, sp_ccs, ratio) rows."""
    df = df.copy()
    df["ccs_species_list"] = df["ccs_species"].apply(parse_ccs_species)
    df["n_species"] = df["ccs_species_list"].apply(len)

    multi = (
        df[df["n_species"] >= 2][
            ["title", "adduct", "ccs_f", "ccs_species_list", "inchikey", "pepmass_f"]
        ]
        .explode("ccs_species_list")
        .rename(columns={"ccs_species_list": "sp_ccs"})
        .assign(sp_ccs=lambda x: pd.to_numeric(x["sp_ccs"], errors="coerce"))
    )
    multi = multi[
        (multi["sp_ccs"] != multi["ccs_f"])
        & multi["ccs_f"].notna()
        & (multi["ccs_f"] > 0)
        & multi["sp_ccs"].notna()
        & (multi["sp_ccs"] > 0)
    ].copy()
    multi["ratio"] = multi["sp_ccs"] / multi["ccs_f"]
    multi["above_sep"] = multi["ratio"] > DIMER_THRESH
    logger.info("Secondary-species pairs: %d", len(multi))
    return multi


def fit_ccs_trend(df: pd.DataFrame) -> tuple[float, float]:
    """Reproduce predict_multi.ipynb cell 4's log(CCS) vs log(m/z) fit (visualization only)."""
    primary_clean = df[
        df["ccs_f"].notna() & df["pepmass_f"].notna() & (df["ccs_f"] > 0) & (df["pepmass_f"] > 0)
    ]
    slope, intercept, r, _, _ = stats.linregress(
        np.log(primary_clean["pepmass_f"].values), np.log(primary_clean["ccs_f"].values)
    )
    logger.info("CCS trend fit: log(CCS) = %.4f + %.4f x log(m/z)  (R2=%.4f)", intercept, slope, r**2)
    return slope, intercept


def build_molecule_table(df: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    """One row per unique inchikey: neutral SMILES, observed polarities, CCS-ratio QC flags."""
    smiles_by_inchikey = df.drop_duplicates("inchikey").set_index("inchikey")["smiles"]

    is_pos = df["adduct"].str.contains(r"\[M\+H\]", regex=True, na=False)
    is_neg = df["adduct"].str.contains(r"\[M-H\]", regex=True, na=False)

    pos_inchikeys = set(df.loc[is_pos, "inchikey"])
    neg_inchikeys = set(df.loc[is_neg, "inchikey"])

    multi_pos_neg = multi[multi["adduct"].str.contains(r"\[M\+H\]|\[M-H\]", regex=True, na=False)]
    clean_by_mol = multi_pos_neg.groupby("inchikey")["above_sep"].agg(lambda x: (~x).any())
    dimer_only_by_mol = multi_pos_neg.groupby("inchikey")["above_sep"].agg(lambda x: bool(x.all()))

    all_inchikeys = pos_inchikeys | neg_inchikeys
    mol_df = pd.DataFrame({"inchikey": sorted(all_inchikeys)})
    mol_df["smiles"] = mol_df["inchikey"].map(smiles_by_inchikey)
    mol_df["has_mh_pos"] = mol_df["inchikey"].isin(pos_inchikeys)
    mol_df["has_mh_neg"] = mol_df["inchikey"].isin(neg_inchikeys)
    mol_df["has_clean_secondary_ccs"] = mol_df["inchikey"].map(clean_by_mol).fillna(False).astype(bool)
    mol_df["has_dimer_only_secondary"] = mol_df["inchikey"].map(dimer_only_by_mol).fillna(False).astype(bool)
    mol_df = mol_df.dropna(subset=["smiles"])
    mol_df = mol_df[mol_df["smiles"].str.len() > 0]

    logger.info(
        "Molecule table: %d total (%d [M+H]+, %d [M-H]-, %d both)",
        len(mol_df),
        mol_df["has_mh_pos"].sum(),
        mol_df["has_mh_neg"].sum(),
        (mol_df["has_mh_pos"] & mol_df["has_mh_neg"]).sum(),
    )
    return mol_df


def plot_dimer_separator(
    df: pd.DataFrame,
    multi: pd.DataFrame,
    slope: float,
    intercept: float,
    output_path: Path,
    mass_label: str = "Precursor m/z",
) -> None:
    """Reproduce predict_multi.ipynb cell 4's CCS-vs-m/z dimer-separator scatter
    plot. mass_label should say "Neutral mass" instead of "Precursor m/z" when
    the input came from the CSV loader, whose pepmass_f is a neutral-mass proxy,
    not a true adduct-shifted precursor m/z (see module docstring)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
    surf, grid, ink, ink2, muted = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e", "#898781"
    plt.rcParams.update(
        {
            "figure.facecolor": surf,
            "axes.facecolor": surf,
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "xtick.color": ink2,
            "ytick.color": ink2,
            "grid.color": grid,
            "grid.linewidth": 0.6,
            "text.color": ink,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    primary_clean = df[
        df["ccs_f"].notna() & df["pepmass_f"].notna() & (df["ccs_f"] > 0) & (df["pepmass_f"] > 0)
    ]
    mz_grid = np.linspace(50, 1200, 400)
    ccs_trend = np.exp(intercept + slope * np.log(mz_grid))

    sample_prim = primary_clean.sample(min(20_000, len(primary_clean)), random_state=42)
    sample_sec = multi.sample(min(60_000, len(multi)), random_state=42)
    below = sample_sec[~sample_sec["above_sep"]]
    above = sample_sec[sample_sec["above_sep"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        sample_prim["pepmass_f"], sample_prim["ccs_f"], s=3, alpha=0.07, c=muted, linewidths=0,
        zorder=1, label="Primary ions",
    )
    ax.scatter(
        below["pepmass_f"], below["sp_ccs"], s=4, alpha=0.25, c=pal[1], linewidths=0, zorder=2,
        label=f"Secondary: ratio < {DIMER_THRESH} — retained (n={len(below):,})",
    )
    ax.scatter(
        above["pepmass_f"], above["sp_ccs"], s=4, alpha=0.30, c=pal[2], linewidths=0, zorder=3,
        label=f"Secondary: ratio > {DIMER_THRESH} — dimer, excluded (n={len(above):,})",
    )
    ax.plot(mz_grid, ccs_trend, color=ink, lw=1.4, ls="-", zorder=4, label="Fitted trend (visual)")
    ax.plot(
        mz_grid, ccs_trend * DIMER_THRESH, color=pal[5], lw=1.6, ls="--", zorder=4,
        label=f"Trend x {DIMER_THRESH} (visual guide only)",
    )
    ax.set_xlabel(mass_label, labelpad=6)
    ax.set_ylabel("CCS (Å²)")
    ax.set_title(
        f"CCS vs {mass_label.lower()}: dimer exclusion via ratio threshold\n"
        f"Separator: sp_ccs / ccs_f > {DIMER_THRESH}  (molecule-specific)",
        fontweight="bold", pad=10,
    )
    ax.set_xlim(50, 1200)
    ax.set_ylim(50, 700)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor=grid)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(
        "Secondary species excluded as dimers (ratio > %.2f): %.1f%%",
        DIMER_THRESH, multi["above_sep"].mean() * 100,
    )
    logger.info("Saved dimer-separator diagnostic plot to %s", output_path)


# --------------------------------------------------------------------------- #
# Step 2: per-molecule site detection and candidate construction
# --------------------------------------------------------------------------- #


def is_basic_site(atom: Chem.Atom, site_name: str) -> bool:
    return atom.GetAtomicNum() == 7 and not site_name.startswith("*")


def basic_site_names_from_table() -> set[str]:
    """Derive the actual basic-site class names from dimorphite-dl's own SMARTS/pKa
    table, independent of any real target molecule: a class is basic iff its name
    is not starred and the atom its pKa entry points at (in the *pattern* mol,
    which GetSubstructMatches indexes consistently) is nitrogen."""
    names: set[str] = set()
    for substructure in PKaData().get_substructures():
        if substructure.name.startswith("*") or substructure.mol is None:
            continue
        for pka in substructure.pkas:
            if pka.idx_site < substructure.mol.GetNumAtoms():
                if substructure.mol.GetAtomWithIdx(pka.idx_site).GetAtomicNum() == 7:
                    names.add(substructure.name)
    return names


def detect_sites(smiles: str, detector: ProtonationSiteDetector) -> Optional[tuple[Chem.Mol, list[ProtonationSite]]]:
    """Run the detector once per molecule; returns the prepared mol (atom indices
    stable across dimorphite-dl's internal AddHs/RemoveHs, confirmed by reading
    mol.py/change.py) and its detected sites, or None on failure."""
    try:
        mol_record = MoleculeRecord(smiles, "")
        mol_record, sites = detector.find_sites(mol_record)
        if mol_record.mol is None:
            return None
        return mol_record.mol, sites
    except Exception as exc:  # noqa: BLE001 - one bad molecule must not kill the run
        logger.debug("Site detection failed for %r: %s", smiles, exc)
        return None


def collapse_duplicate_atoms(
    mol: Chem.Mol, sites: list[ProtonationSite], polarity: str
) -> dict[int, dict[str, Any]]:
    """Flatten sites to per-atom records, keyed by atom_idx, filtered to the requested polarity."""
    by_atom: dict[int, dict[str, Any]] = defaultdict(lambda: {"site_names": set(), "pka_mean": None, "pka_stdev": None})

    for site in sites:
        for pka in site.pkas:
            atom_idx = site.idxs_match[pka.idx_site]
            atom = mol.GetAtomWithIdx(atom_idx)
            basic = is_basic_site(atom, site.name)
            if polarity == "+1" and not basic:
                continue
            if polarity == "-1" and basic:
                continue

            rec = by_atom[atom_idx]
            rec["site_names"].add(site.name)
            if rec["pka_mean"] is None or pka.mean > rec["pka_mean"]:
                rec["pka_mean"] = pka.mean
                rec["pka_stdev"] = pka.stdev

    return dict(by_atom)


def collapse_symmetry(mol: Chem.Mol, by_atom: dict[int, dict[str, Any]]) -> list[tuple[int, list[int]]]:
    """Group candidate atoms by RDKit canonical rank; keep the lowest index per rank."""
    if not by_atom:
        return []
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    by_rank: dict[int, list[int]] = defaultdict(list)
    for atom_idx in by_atom:
        by_rank[ranks[atom_idx]].append(atom_idx)

    representatives = []
    for idxs in by_rank.values():
        idxs_sorted = sorted(idxs)
        representatives.append((idxs_sorted[0], idxs_sorted[1:]))
    return representatives


def is_adjacent_to_carbonyl(mol: Chem.Mol, atom_idx: int) -> bool:
    """Flags amide/vinylogous-amide-like deactivation. Two distinct failure modes
    converge on this one check, found during verification on real Enveda-180
    molecules, not just the caffeine probe case:
    (1) fused aromatic ring N's flanked by ring carbonyls (caffeine-style) --
        these are usually self-correcting: dimorphite-dl's own degree-3
        fully-conjugated-aromatic-N special case (see build_candidate_smiles)
        already fails to assign a charge there, so they rarely reach this flag;
    (2) non-aromatic tertiary amide/lactam nitrogens matched by the
        `Amines_primary_secondary_tertiary` SMARTS ([C:1]-[NX3+0:2]), which does
        not exclude an adjacent carbonyl carbon. This is the dominant case in
        practice: on a 500-molecule real-data sample, 51% of all
        `Amines_primary_secondary_tertiary` candidates were flagged here --
        amide nitrogens (pKaH roughly 0) misassigned pKa_mean=8.16 and a +1
        candidate structure, which is chemically wrong, not a rare edge case."""
    atom = mol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        for bond in nbr.GetBonds():
            other = bond.GetOtherAtom(nbr)
            if other.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2.0:
                return True
    return False


# --------------------------------------------------------------------------- #
# Widened [M+H]+ site enumeration (N/O/S), added per the danofloxacin finding
# --------------------------------------------------------------------------- #
# The original enumeration was nitrogen-only (dimorphite-dl's basic-site SMARTS
# only fire on N), so an O-protonated protomer -- e.g. danofloxacin's keto-O+,
# a confirmed experimental protomer -- could never be represented. Protonation
# can also occur on O and S, so for [M+H]+ every N/O/S atom is now enumerated
# as a candidate (superset of dimorphite's basic-N sites). dimorphite-dl is
# still used, but only to *annotate* solution pKa on the atoms it recognizes;
# it no longer defines the candidate set. Ranking/gating by energy or pKa is
# deliberately NOT done here (kinetic trapping from solution means a large
# gas-phase energy gap does not rule out a second observed protomer -- see the
# danofloxacin ground-truth test), so all sites are emitted and left for the
# downstream CCS-based matching + QM validation to sort out.
PROTONATABLE_ELEMENTS_POS = (7, 8, 16)  # N, O, S


def site_type_name(mol: Chem.Mol, atom_idx: int) -> str:
    """A short, chemically-descriptive label for a generic protonation site
    (used when the atom is not one of dimorphite-dl's named basic-N classes)."""
    atom = mol.GetAtomWithIdx(atom_idx)
    el = atom.GetSymbol()
    if atom.GetIsAromatic():
        return f"aromatic_{el}"
    if atom.GetAtomicNum() == 8:
        if any(b.GetBondTypeAsDouble() == 2.0 for b in atom.GetBonds()):
            return "carbonyl_O"
        return "hydroxyl_O" if atom.GetTotalNumHs() > 0 else "ether_O"
    if atom.GetAtomicNum() == 7:
        return "amine_N"
    if atom.GetAtomicNum() == 16:
        return "sulfur_S"
    return f"other_{el}"


def enumerate_basic_heteroatom_sites(mol: Chem.Mol) -> list[int]:
    """Every N/O/S atom is a candidate [M+H]+ protonation site, except:
    already-cationic atoms (nothing to protonate), and amide/vinylogous-amide
    nitrogens (those protonate on the carbonyl O, not the N -- the carbonyl O
    is itself enumerated here, so the site is kept, just moved to the right
    atom; see is_adjacent_to_carbonyl and the danofloxacin amide-fix note)."""
    sites: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in PROTONATABLE_ELEMENTS_POS:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetAtomicNum() == 7 and is_adjacent_to_carbonyl(mol, atom.GetIdx()):
            continue
        sites.append(atom.GetIdx())
    return sites


def build_plus_site_records(
    mol: Chem.Mol, dimorphite_sites: list[ProtonationSite]
) -> dict[int, dict[str, Any]]:
    """Per-atom records for the widened [M+H]+ candidate set: all N/O/S sites
    (enumerate_basic_heteroatom_sites), annotated with dimorphite-dl's solution
    pKa wherever the atom is one of its recognized basic-N sites, else pKa=None
    (most O/S sites have no tabulated pKa -- that is expected, pKa is a
    descriptor here, not a filter)."""
    pka_lookup = collapse_duplicate_atoms(mol, dimorphite_sites, "+1")
    by_atom: dict[int, dict[str, Any]] = {}
    for atom_idx in enumerate_basic_heteroatom_sites(mol):
        if atom_idx in pka_lookup:
            rec = dict(pka_lookup[atom_idx])
            rec["site_names"] = set(rec["site_names"]) | {site_type_name(mol, atom_idx)}
        else:
            rec = {"site_names": {site_type_name(mol, atom_idx)}, "pka_mean": None, "pka_stdev": None}
        by_atom[atom_idx] = rec
    return by_atom


def build_candidate_smiles_generic(
    mol: Chem.Mol, atom_idx: int, target_charge: int
) -> tuple[Optional[str], Optional[str]]:
    """Element-agnostic protonation for the widened N/O/S site set: add one H
    and a +1 formal charge to the target atom, sanitize, and verify the total
    charge landed at +1. Unlike build_candidate_smiles (which delegates to
    dimorphite-dl's set_protonation_charge and only handles its named N site
    types), this works for O and S sites too, and -- deliberately -- does NOT
    silently drop fully-conjugated aromatic N's the way dimorphite-dl's special
    case does: an aromatic-N protomer is a legitimate candidate to hand to the
    downstream CCS/QM validation rather than to exclude a priori."""
    m = Chem.Mol(mol)
    atom = m.GetAtomWithIdx(atom_idx)
    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
    atom.SetNoImplicit(True)
    try:
        atom.UpdatePropertyCache(strict=True)
        Chem.SanitizeMol(m)
        # detect_sites' mol carries explicit Hs; strip them so the canonical
        # SMILES matches the rest of the pipeline's implicit-H form (and the
        # hand-built danofloxacin protomers already validated against MOBCAL).
        m = Chem.RemoveHs(m)
    except Exception as exc:  # noqa: BLE001 - invalid valence (e.g. hypervalent O) -> reject
        logger.debug("Generic protonation failed at atom %d: %s", atom_idx, exc)
        return None, "sanitize_error"
    if Chem.GetFormalCharge(m) != target_charge:
        return None, "wrong_charge"
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True), None


def build_candidate_smiles(
    mol: Chem.Mol, atom_idx: int, target_charge: int, site_name: str
) -> tuple[Optional[str], Optional[str]]:
    """Reuse dimorphite-dl's own tested charge-setting routine rather than
    hand-rolling explicit-H bookkeeping. state_charge is the pre-nitrogen-offset
    charge value set_protonation_charge expects (0 -> PROTONATED/+1 for bases,
    -1 -> DEPROTONATED/-1 for acids); see protonate/change.py.

    Returns (candidate_smiles, failure_reason); exactly one is None. The
    "wrong_charge" reason is not hypothetical: dimorphite-dl's own
    _set_nitrogen_properties silently no-ops (leaves the atom at charge 0,
    without raising) for degree-3, fully-conjugated aromatic nitrogens, a
    known edge case in its own source comments ("aromatics often detected on
    NADP") -- confirmed directly on caffeine's ring nitrogens during
    verification. The charge check below is what catches that silent failure.
    """
    state_charge = 0 if target_charge == 1 else -1
    try:
        results = set_protonation_charge(mols=[mol], idx=atom_idx, charges=[state_charge], prot_site_name=site_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Charge assignment raised at atom %d (%s): %s", atom_idx, site_name, exc)
        return None, "charge_assignment_error"
    if not results:
        return None, "charge_assignment_empty"

    candidate = results[0]
    try:
        Chem.SanitizeMol(candidate)
        total_charge = Chem.GetFormalCharge(candidate)
        if total_charge != target_charge:
            logger.debug(
                "Candidate at atom %d (%s) has charge %d, expected %d; discarding "
                "(likely dimorphite-dl's fully-conjugated-aromatic-N special case)",
                atom_idx, site_name, total_charge, target_charge,
            )
            return None, "wrong_charge"
        return Chem.MolToSmiles(candidate, isomericSmiles=True, canonical=True), None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sanitization/charge check failed at atom %d (%s): %s", atom_idx, site_name, exc)
        return None, "sanitize_error"


def generate_candidates_for_molecule(
    inchikey: str, smiles: str, has_pos: bool, has_neg: bool, detector: ProtonationSiteDetector, stats_counter: dict[str, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detected = detect_sites(smiles, detector)
    if detected is None:
        stats_counter["detection_failed"] += 1
        for charge_state in ([1] if has_pos else []) + ([-1] if has_neg else []):
            rows.append(_zero_candidate_row(inchikey, smiles, charge_state))
        return rows

    mol, sites = detected
    if len(sites) >= 50:  # matches ProtonationSiteDetector's default max_sites_per_molecule
        stats_counter["hit_max_sites_cap"] += 1

    for polarity, has_polarity, charge_state in (("+1", has_pos, 1), ("-1", has_neg, -1)):
        if not has_polarity:
            continue

        # [M+H]+ uses the widened N/O/S enumeration (build_plus_site_records +
        # build_candidate_smiles_generic); [M-H]- keeps dimorphite-dl's acidic
        # site detection + charge-setting unchanged. The widening applies to
        # protonation, which is where the missing O-protomers (danofloxacin's
        # keto-O+) were.
        if charge_state == 1:
            by_atom = build_plus_site_records(mol, sites)
        else:
            by_atom = collapse_duplicate_atoms(mol, sites, polarity)
        representatives = collapse_symmetry(mol, by_atom)

        if not representatives:
            stats_counter[f"no_site_found_{charge_state:+d}"] += 1
            rows.append(_zero_candidate_row(inchikey, smiles, charge_state))
            continue

        # Build every representative first; n_candidates must reflect the
        # surviving count (a generic protonation can still fail sanitization on
        # an invalid valence, and dimorphite-dl can silently fail to assign a
        # charge for some -1 sites -- so the pre-build representative count
        # would overstate it).
        built: list[dict[str, Any]] = []
        for atom_idx, symmetry_group in representatives:
            rec = by_atom[atom_idx]
            if charge_state == 1:
                candidate_smiles, failure_reason = build_candidate_smiles_generic(
                    mol, atom_idx, charge_state
                )
            else:
                candidate_smiles, failure_reason = build_candidate_smiles(
                    mol, atom_idx, charge_state, next(iter(rec["site_names"]))
                )
            if candidate_smiles is None:
                stats_counter[f"candidate_build_failed_{failure_reason}"] += 1
                continue
            built.append(
                {
                    "inchikey": inchikey,
                    "parent_smiles": smiles,
                    "charge_state": charge_state,
                    "candidate_smiles": candidate_smiles,
                    "charged_atom_idx": atom_idx,
                    "site_names": ",".join(sorted(rec["site_names"])),
                    "pka_mean": rec["pka_mean"],
                    "pka_stdev": rec["pka_stdev"],
                    "symmetry_group_atom_idxs": ",".join(str(i) for i in symmetry_group),
                    "adjacent_to_carbonyl": (
                        is_adjacent_to_carbonyl(mol, atom_idx) if charge_state == 1 else False
                    ),
                }
            )

        if not built:
            stats_counter[f"all_candidates_failed_to_build_{charge_state:+d}"] += 1
            rows.append(_zero_candidate_row(inchikey, smiles, charge_state))
            continue

        stats_counter[f"n_candidates_{charge_state:+d}_{min(len(built), 2)}"] += 1
        for r in built:
            r["n_candidates"] = len(built)
        rows.extend(built)

    return rows


def _zero_candidate_row(inchikey: str, smiles: str, charge_state: int) -> dict[str, Any]:
    return {
        "inchikey": inchikey,
        "parent_smiles": smiles,
        "charge_state": charge_state,
        "candidate_smiles": None,
        "charged_atom_idx": None,
        "site_names": None,
        "pka_mean": None,
        "pka_stdev": None,
        "symmetry_group_atom_idxs": None,
        "adjacent_to_carbonyl": False,
        "n_candidates": 0,
    }


OUTPUT_SCHEMA = pa.schema(
    [
        ("inchikey", pa.string()),
        ("parent_smiles", pa.string()),
        ("charge_state", pa.int8()),
        ("candidate_smiles", pa.string()),
        ("charged_atom_idx", pa.int32()),
        ("site_names", pa.string()),
        ("pka_mean", pa.float32()),
        ("pka_stdev", pa.float32()),
        ("symmetry_group_atom_idxs", pa.string()),
        ("adjacent_to_carbonyl", pa.bool_()),
        ("n_candidates", pa.int32()),
        ("has_clean_secondary_ccs", pa.bool_()),
        ("has_dimer_only_secondary", pa.bool_()),
    ]
)


def run_candidate_generation(mol_df: pd.DataFrame, output_path: Path, chunk_size: int) -> dict[str, int]:
    detector = ProtonationSiteDetector(validate_sites=True, max_sites_per_molecule=50)
    stats_counter: dict[str, int] = defaultdict(int)

    writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA)
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        chunk_df = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(chunk_df, schema=OUTPUT_SCHEMA, preserve_index=False)
        writer.write_table(table)
        buffer.clear()

    try:
        for row in tqdm(mol_df.itertuples(index=False), total=len(mol_df), desc="Enumerating candidates"):
            try:
                mol_rows = generate_candidates_for_molecule(
                    row.inchikey, row.smiles, row.has_mh_pos, row.has_mh_neg, detector, stats_counter
                )
            except Exception as exc:  # noqa: BLE001 - one bad molecule must not kill the run
                logger.warning("Unhandled error for %s (%r): %s", row.inchikey, row.smiles, exc)
                stats_counter["unhandled_error"] += 1
                continue

            for r in mol_rows:
                r["has_clean_secondary_ccs"] = row.has_clean_secondary_ccs
                r["has_dimer_only_secondary"] = row.has_dimer_only_secondary
            buffer.extend(mol_rows)
            stats_counter["molecules_processed"] += 1

            if len(buffer) >= chunk_size:
                flush()
        flush()
    finally:
        writer.close()

    return dict(stats_counter)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/Enveda-180/enveda-180.jsonl.gz"),
        help=(
            "Path to enveda-180.jsonl.gz or enveda-180-dedup.csv "
            "(format is auto-detected from the file suffix)"
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Parquet path; the diagnostic plot is saved alongside it as <stem>_dimer_separator.png",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000,
        help="Number of candidate rows to buffer before writing to Parquet (default: 5000)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on the number of unique molecules to process (for smoke testing)",
    )
    args = parser.parse_args(argv)

    actual_basic_names = basic_site_names_from_table()
    if actual_basic_names != EXPECTED_BASIC_SITE_NAMES:
        raise RuntimeError(
            "dimorphite-dl's basic-site SMARTS table no longer matches this "
            f"script's assumptions. Expected {sorted(EXPECTED_BASIC_SITE_NAMES)}, "
            f"got {sorted(actual_basic_names)}. Re-check is_basic_site() and the "
            "installed dimorphite-dl version before proceeding."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    spectra_df = load_spectra(args.input, limit_molecules=args.limit)
    multi = build_secondary_species_table(spectra_df)
    slope, intercept = fit_ccs_trend(spectra_df)
    mol_df = build_molecule_table(spectra_df, multi)

    is_csv = "".join(args.input.suffixes).lower().endswith(".csv")
    mass_label = "Neutral mass" if is_csv else "Precursor m/z"

    plot_path = args.output.with_name(args.output.stem + "_dimer_separator.png")
    plot_dimer_separator(spectra_df, multi, slope, intercept, plot_path, mass_label=mass_label)

    stats_counter = run_candidate_generation(mol_df, args.output, args.chunk_size)

    logger.info("Run summary:")
    for key in sorted(stats_counter):
        logger.info("  %s: %d", key, stats_counter[key])
    logger.info("Wrote candidates to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
