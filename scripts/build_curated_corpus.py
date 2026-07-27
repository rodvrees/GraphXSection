"""Convert the curated SigmaCCS data into the protomer-corpus schema so the CURRENT
approach (protomer pipeline + anchored set loss) can be trained on it directly.

This is the data-vs-method control: the curated data is single-CCS, so each row
becomes a base-only record (no protomer candidates, no gap). Training the current
pipeline on it therefore reduces to base-CCS regression on the neutral parent
graph + 41-D adduct descriptor -- the same task the reference model solves at
~1.3% on this data. If the current approach also reaches ~1.3% here, the ~2%
on Enveda is confirmed to be a data ceiling, not a method deficiency.

Input:  data/SigmaCCS/{train,valid,test}_data.csv  (ID, SMILES, adduct, CCS; neutral SMILES)
Output: outputs/curated_corpus/corpus_{train,valid,test}.parquet  (corpus schema, base-only)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path("/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/data/SigmaCCS")
OUT_DIR = Path("/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/outputs/curated_corpus")
CHARGE = {"[M+H]+": 1, "[M-H]-": -1}   # matches PROTOMER_ADDUCTS; others -> 0 (base only)


def _inchikey(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m) if m is not None else None


def _scaffold(smiles: str, inchikey: str | None) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return inchikey[:14] if inchikey else smiles
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        return s if s else (inchikey[:14] if inchikey else smiles)
    except Exception:
        return inchikey[:14] if inchikey else smiles


def convert(df: pd.DataFrame, split: str) -> pd.DataFrame:
    recs = []
    for r in df.itertuples(index=False):
        ik = _inchikey(r.SMILES)
        if ik is None:
            continue
        recs.append({
            "inchikey": ik,
            "parent_smiles": r.SMILES,          # neutral
            "adduct": r.adduct,
            "charge_state": CHARGE.get(r.adduct, 0),
            "source": "sigmaccs",
            "im_type": "DT",
            "observed_ccs_peaks": [float(r.CCS)],
            "n_obs_peaks": 1,
            "candidate_smiles": [],             # base-only: forwards the neutral parent
            "charged_atom_idx": [],
            "site_names": [],
            "pka_mean": [],
            "dominant_site_idx": -1,
            "n_candidates": 0,
            "has_gap": False,
            "scaffold": _scaffold(r.SMILES, ik),
            "split": split,
        })
    return pd.DataFrame(recs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        df = pd.read_csv(SRC_DIR / f"{split}_data.csv")
        out = convert(df, split)
        dst = OUT_DIR / f"corpus_{split}.parquet"
        out.to_parquet(dst, index=False)
        logger.info("%s: %d rows -> %s", split, len(out), dst)
    # leakage check on scaffolds (splits are pre-defined by the source files)
    scafs = {s: set(pd.read_parquet(OUT_DIR / f"corpus_{s}.parquet")["scaffold"])
             for s in ("train", "valid", "test")}
    leak = (scafs["train"] & scafs["test"]) | (scafs["train"] & scafs["valid"]) | (scafs["valid"] & scafs["test"])
    logger.info("scaffold overlap across curated splits: %d (note: source splits are not scaffold-grouped)", len(leak))


if __name__ == "__main__":
    main()
