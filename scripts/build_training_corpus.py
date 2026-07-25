#!/usr/bin/env python3
"""
Build the unified training corpus for the protomer-aware CCS model
(protomer_ccs_predictor plan, Section 1). Produces one record per
(molecule, adduct):

  inchikey, parent_smiles (neutral), adduct, charge_state, source, im_type,
  observed_ccs_peaks : list[float]  -- base peak first, dimers/out-of-band
                                       secondaries removed
  candidate_smiles   : list[str]    -- charge-positioned protomer set (H+/- only)
  charged_atom_idx   : list[int]
  site_names         : list[str]
  pka_mean           : list[float]
  dominant_site_idx  : int          -- index into the candidate lists (highest
                                       pKa, basicity/valence tiebreak); -1 if none
  n_obs_peaks        : int
  n_candidates       : int
  has_gap            : bool          -- multi-CCS AND credible-protomer (>=2
                                        candidates), i.e. contributes the gap term

Splits are scaffold-grouped (Bemis-Murcko, InChIKey-connectivity-block fallback)
so no molecule/scaffold leaks across train/val/test.

This first version targets the Enveda-180 timsTOF core (the only multi-CCS
source). Curated TIMS single-CCS add-ons (CCSBase/SigmaCCS/PFAS TIMS subset) are
appended as base-only records when --with-curated-tims is given; they need
inline candidate enumeration and are a small contribution.

Usage:
    python scripts/build_training_corpus.py \\
        --enveda-dedup data/Enveda-180/enveda-180-dedup.csv \\
        --enveda-candidates outputs/protonation_candidates/enveda_widened.parquet \\
        --out-dir outputs/training_corpus \\
        [--limit N] [--seed 42]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DIMER_THRESH = 1.25            # ratio sp_ccs/ccs above which a secondary peak is a dimer
BAND_LOW, BAND_HIGH = 0.02, 0.09   # |ratio-1| conformer/protomer band for kept secondaries
PROTOMER_ADDUCTS = {"[M+H]+": 1, "[M-H]-": -1}   # only these get protomer enumeration
VALID_FRAC, TEST_FRAC = 0.1, 0.1


# --------------------------------------------------------------------------- #
# Observed-peak parsing / cleaning
# --------------------------------------------------------------------------- #
def parse_peaks(ccs_species: str, base_ccs: float) -> list[float]:
    """ccs_species is intensity-ordered (base peak first). Keep the base peak
    plus secondaries that are NOT dimers (ratio <= DIMER_THRESH) and fall inside
    the conformer/protomer band [BAND_LOW, BAND_HIGH]. Drops near-duplicate and
    out-of-band peaks; always keeps the base peak first."""
    try:
        vals = [float(x) for x in str(ccs_species).split(",") if x.strip()]
    except ValueError:
        vals = []
    if not vals:
        return [base_ccs] if base_ccs and base_ccs > 0 else []
    base = vals[0]
    if not (base and base > 0):
        return []
    kept = [base]
    for v in vals[1:]:
        if v <= 0:
            continue
        ratio = v / base
        gap = abs(ratio - 1.0)
        if ratio <= DIMER_THRESH and BAND_LOW <= gap <= BAND_HIGH:
            kept.append(v)
    return kept


# --------------------------------------------------------------------------- #
# InChIKey / scaffold helpers (cached)
# --------------------------------------------------------------------------- #
def inchikey_map(smiles_list: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        out[smi] = Chem.MolToInchiKey(mol) if mol is not None else None
    return out


def scaffold_key(smiles: str, inchikey: str) -> str:
    """Bemis-Murcko scaffold SMILES; falls back to the InChIKey connectivity
    block (first 14 chars) for acyclic molecules with an empty scaffold."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is not None:
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaf:
                return scaf
        except Exception:  # noqa: BLE001
            pass
    return inchikey[:14] if isinstance(inchikey, str) else "UNKNOWN"


# --------------------------------------------------------------------------- #
# Dominant protonation site
# --------------------------------------------------------------------------- #
def _site_element(site_name: str) -> str:
    """Protonation-site element from the generic site label (the last
    comma-part, e.g. 'amine_N' -> N, 'carbonyl_O' -> O, 'sulfur_S' -> S).
    Falls back to keyword inference for Dimorphite-only class names."""
    tag = str(site_name).split(",")[-1].strip()
    if tag.endswith("_N"):
        return "N"
    if tag.endswith("_O"):
        return "O"
    if tag.endswith("_S"):
        return "S"
    t = tag.lower()
    if any(k in t for k in ("amine", "aniline", "amide", "nitrogen", "pyridine", "imid", "azo")):
        return "N"
    if any(k in t for k in ("hydroxyl", "carboxyl", "carbonyl", "ether", "alcohol", "oxygen")):
        return "O"
    if "sulf" in t or "thio" in t:
        return "S"
    return "?"


def pick_dominant(pka_means: list[float], site_names: list[str]) -> int:
    """Dominant (base-peak) protomer index. Primary rule: highest Dimorphite
    solution pKa -- the correct criterion, since the dominant protomer population
    is set in solution (pKa) and kinetically trapped into the gas phase (roadmap
    finding), and a site Dimorphite does not annotate is a genuinely weak solution
    base that should lose. Fallback when no site has a pKa: the N > O > S element
    tier priority ported from .archive/proton_affinity_calculator.py's
    identify_protonation_sites (prefer any N, else O, else S; first within tier).
    Returns index into the candidate lists, or -1 if empty."""
    if not pka_means:
        return -1
    pka = np.array([p if p is not None and not np.isnan(p) else -np.inf for p in pka_means], dtype=float)
    if np.any(np.isfinite(pka)):
        return int(np.argmax(pka))
    # no pKa anywhere: N > O > S element tier
    elements = [_site_element(s) for s in site_names]
    for tier in ("N", "O", "S"):
        for i, e in enumerate(elements):
            if e == tier:
                return i
    return 0


# --------------------------------------------------------------------------- #
# Corpus assembly (Enveda core)
# --------------------------------------------------------------------------- #
def build_enveda_records(dedup: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    # candidates grouped by (inchikey, charge_state)
    cand_groups: dict[tuple[str, int], pd.DataFrame] = {
        key: g for key, g in cands.groupby(["inchikey", "charge_state"])
    }

    records = []
    n_no_cands = 0
    for row in dedup.itertuples(index=False):
        adduct = row.adduct
        charge_state = PROTOMER_ADDUCTS.get(adduct)
        peaks = parse_peaks(row.ccs_species, row.ccs)
        if len(peaks) == 0:
            continue

        rec = {
            "inchikey": row.inchikey,
            "parent_smiles": row.SMILES,
            "adduct": adduct,
            "charge_state": charge_state if charge_state is not None else 0,
            "source": "enveda",
            "im_type": "TIMS",
            "observed_ccs_peaks": peaks,
            "n_obs_peaks": len(peaks),
            "candidate_smiles": [],
            "charged_atom_idx": [],
            "site_names": [],
            "pka_mean": [],
            "dominant_site_idx": -1,
            "n_candidates": 0,
            "has_gap": False,
        }

        if charge_state is not None:  # H+/- -> attach protomer candidates
            g = cand_groups.get((row.inchikey, charge_state))
            if g is not None:
                g = g[g["candidate_smiles"].notna() & g["charged_atom_idx"].notna()]
            if g is not None and len(g) > 0:
                rec["candidate_smiles"] = g["candidate_smiles"].tolist()
                rec["charged_atom_idx"] = [int(x) for x in g["charged_atom_idx"].tolist()]
                rec["site_names"] = g["site_names"].tolist()
                rec["pka_mean"] = [float(x) if pd.notna(x) else float("nan") for x in g["pka_mean"].tolist()]
                rec["n_candidates"] = len(g)
                rec["dominant_site_idx"] = pick_dominant(rec["pka_mean"], rec["site_names"])
            else:
                n_no_cands += 1
        rec["has_gap"] = rec["n_obs_peaks"] >= 2 and rec["n_candidates"] >= 2
        records.append(rec)

    logger.info("Enveda records: %d (%d H+/- rows had no enumerated candidates)", len(records), n_no_cands)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Scaffold-grouped split
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    scaffs = df.drop_duplicates("inchikey")[["inchikey", "parent_smiles"]].copy()
    scaffs["scaffold"] = [scaffold_key(s, ik) for s, ik in zip(scaffs["parent_smiles"], scaffs["inchikey"])]
    ik_to_scaf = dict(zip(scaffs["inchikey"], scaffs["scaffold"]))

    unique_scaffolds = sorted(scaffs["scaffold"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_scaffolds)
    n = len(unique_scaffolds)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VALID_FRAC)
    test_s = set(unique_scaffolds[:n_test])
    val_s = set(unique_scaffolds[n_test:n_test + n_val])

    def split_of(ik: str) -> str:
        s = ik_to_scaf.get(ik)
        return "test" if s in test_s else "valid" if s in val_s else "train"

    df = df.copy()
    df["scaffold"] = df["inchikey"].map(ik_to_scaf)
    df["split"] = df["inchikey"].map(split_of)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enveda-dedup", type=Path, default=Path("data/Enveda-180/enveda-180-dedup.csv"))
    p.add_argument("--enveda-candidates", type=Path,
                   default=Path("outputs/protonation_candidates/enveda_widened.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/training_corpus"))
    p.add_argument("--limit", type=int, default=None, help="Cap unique enveda SMILES (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dedup = pd.read_csv(args.enveda_dedup)
    dedup["ccs"] = pd.to_numeric(dedup["ccs"], errors="coerce")
    dedup = dedup[dedup["ccs"].notna() & (dedup["ccs"] > 0)]
    if args.limit is not None:
        keep = dedup["SMILES"].drop_duplicates().head(args.limit)
        dedup = dedup[dedup["SMILES"].isin(set(keep))]

    logger.info("Computing InChIKeys for %d unique Enveda SMILES", dedup["SMILES"].nunique())
    ikmap = inchikey_map(dedup["SMILES"].unique().tolist())
    dedup["inchikey"] = dedup["SMILES"].map(ikmap)
    dedup = dedup[dedup["inchikey"].notna()]

    cands = pd.read_parquet(args.enveda_candidates)

    df = build_enveda_records(dedup, cands)
    df = assign_splits(df, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        out = args.out_dir / f"corpus_{split}.parquet"
        sub.to_parquet(out, index=False)
        logger.info(
            "%s: %d records | %d molecules | %d with gap | %d single-peak",
            split, len(sub), sub["inchikey"].nunique(),
            int(sub["has_gap"].sum()), int((sub["n_obs_peaks"] == 1).sum()),
        )

    # leakage assertion
    splits_by_scaf = df.groupby("scaffold")["split"].nunique()
    leaked = int((splits_by_scaf > 1).sum())
    logger.info("Scaffolds spanning >1 split (should be 0): %d", leaked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
