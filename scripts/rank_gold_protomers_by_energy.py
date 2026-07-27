"""Proton-affinity (stability) ranking check on the labelled gold protomer pairs.

For each gold molecule with per-protomer charged SMILES, embed + PM3-opt each
protomer and rank by electronic energy. The lower-energy protomer is the more
stable / higher proton-affinity / more abundant one (the base peak under the
assignment-by-abundance approach). Report that ranking against the observed CCS.

NOTE: PM3 is a fast first pass; the QM-label memo established PM3 relative
energetics are crude and r2SCAN-3c is the production method. This is a pipeline
demonstration + preliminary read, not the final ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_candidates_pm3 import embed_and_optimize, mol_to_xyz_block, run_orca_pm3  # noqa: E402
from rdkit import Chem  # noqa: E402

ORCA = "/home/rodvrees/orca_6_0_0/orca"
HARTREE_KCAL = 627.5095


def pm3_energy(smiles: str):
    mol, st = embed_and_optimize(smiles)
    if mol is None:
        return None, st
    e, st = run_orca_pm3(mol_to_xyz_block(mol), Chem.GetFormalCharge(mol), ORCA, 300, "/dev/shm")
    return e, st


def main():
    gold = pd.read_csv("protomer_eval/protomer_test_df.csv",
                       usecols=lambda c: c in ["Chemical Name", "protonated_SMILES", "CCS"])
    gold["mol"] = gold["Chemical Name"].str.replace(r"_\d+$", "", regex=True)
    rows = []
    for name, g in gold.groupby("mol"):
        g = g.reset_index(drop=True)
        if len(g) != 2:
            continue
        es = []
        for _, r in g.iterrows():
            e, st = pm3_energy(r["protonated_SMILES"])
            es.append(e)
            print(f"  {name:12s} CCS={r['CCS']:.1f}  PM3_E={e}  ({st})")
        if None in es:
            print(f"  {name}: energy failed, skip"); continue
        # lower energy = more stable = more abundant = predicted base peak
        stable_i = 0 if es[0] < es[1] else 1
        other_i = 1 - stable_i
        rows.append({
            "molecule": name,
            "stable_CCS": g.loc[stable_i, "CCS"],          # predicted base (most abundant)
            "other_CCS": g.loc[other_i, "CCS"],
            "stable_is_higher_CCS": bool(g.loc[stable_i, "CCS"] > g.loc[other_i, "CCS"]),
            "dE_kcal": round(abs(es[0] - es[1]) * HARTREE_KCAL, 2),
        })
    print("\n=== stability ranking vs CCS (PM3, preliminary) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nReading: 'stable' = lower-energy protomer = predicted most-abundant = base peak under the")
    print("assignment-by-abundance rule. Whether that matches the true base peak needs peak INTENSITY,")
    print("which the gold set does not carry. dE_kcal shows whether the two protomers are separable at all.")


if __name__ == "__main__":
    main()
