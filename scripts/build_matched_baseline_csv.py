"""Flatten the protomer corpus into a matched-baseline CSV set.

The matched baseline is the OLD procedure (plain single-CCS regression, the
15 baseline atom features, no protomer changes) retrained on the SAME molecules
and the SAME scaffold splits as the protomer model, so any gap improvement is
attributable to the method and not to different data.

Each corpus row (one molecule, one adduct) becomes one baseline row:
  - SMILES : the dominant charged candidate (H+/- molecules) or the neutral
             parent (base-only adducts). This is exactly the single structure the
             protomer model's anchor term regresses to the base peak, so the
             baseline sees the same input, minus the gap term and the two added
             features.
  - Adduct : the adduct string.
  - CCS    : the base (most intense) observed peak.

Writes train/valid/test CSVs mirroring the corpus split column. Train the
baseline with the existing entrypoint, pointing feature-config at the 15-feature
baseline set, e.g.:

    python -m GraphXSection_torch \\
        --config configs/sigma_config_baseline_matched.json \\
        --feature-config configs/features_baseline.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CORPUS_DIR = Path("/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/outputs/training_corpus")
OUT_DIR = Path("/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/outputs/matched_baseline")


def dominant_smiles(row) -> str | None:
    """Charged dominant candidate, or the neutral parent for base-only adducts."""
    cands = row["candidate_smiles"]
    dom = row["dominant_site_idx"]
    if cands is None or len(cands) == 0 or dom is None or int(dom) < 0:
        return row["parent_smiles"]
    dom = int(dom)
    if dom >= len(cands):
        return row["parent_smiles"]
    return cands[dom]


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "SMILES": df.apply(dominant_smiles, axis=1),
            "Adduct": df["adduct"],
            "CCS": df["observed_ccs_peaks"].map(lambda p: float(p[0])),
            "inchikey": df["inchikey"],
            "scaffold": df["scaffold"],
            "has_gap": df["has_gap"],
        }
    )
    return out.dropna(subset=["SMILES", "Adduct", "CCS"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src = corpus_dir / f"corpus_{split}.parquet"
        df = pd.read_parquet(src)
        flat = flatten(df)
        dst = out_dir / f"baseline_{split}.csv"
        flat.to_csv(dst, index=False)
        print(
            f"{split:5s}: {len(df):>7d} corpus rows -> {len(flat):>7d} baseline rows "
            f"({int(flat['has_gap'].sum())} from gap molecules) -> {dst}"
        )


if __name__ == "__main__":
    main()
