"""Protomer-gap evaluation harness (the acceptance test).

Given a trained protomer model, this reports:

1. Base-CCS parity: the dominant candidate's predicted CCS vs the base observed
   peak, overall and per adduct (MAE, median relative error). Optionally compared
   against a matched-baseline predictions file on the same molecules.
2. Protomer-gap payoff: on multi-CCS (gap) molecules, all candidates are
   forwarded, the predicted peak set is matched to the observed set with the
   anchored-offset matcher (build_protomer_labels.match_by_anchored_ccs), and the
   set-matched gap error / correlation / sign accuracy is compared against the
   protonation-blind zero-gap baseline (which predicts no gap).
3. Permutation null: the same matcher is run with wrong-structure candidate
   predictions (predicted vectors shuffled across gap molecules of equal
   candidate count). The trained model's gap concordance must exceed this null,
   otherwise the signal is matcher artifact rather than structure-specific.
4. Directional spot-checks: optional gold cases (e.g. danofloxacin 181.7/193.7)
   checked for correct gap direction and magnitude band.

The model is loaded via ProtomerLightningModule so the SigmaModel backbone,
feature config and charged-atom readout all come from the training config. Import
order: SigmaModel is imported first (graphamole namespace-package gotcha).

Usage:
    python scripts/eval_protomer_gap.py \\
        --config configs/sigma_config_protomer.json \\
        --checkpoint outputs/Protomer_SigmaCCS_protomer_model.pth \\
        [--split test] [--baseline-preds path.parquet] [--gold-csv path.csv]
"""
from __future__ import annotations

from GraphXSection_torch.models.Sigma import SigmaModel  # noqa: F401 import first

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from GraphXSection_torch.feature_config import load_feature_config
from GraphXSection_torch.protomer_data import ProtomerCorpusDataset, collate_protomer
from GraphXSection_torch.lightning_modules import ProtomerLightningModule
from GraphXSection_torch.adduct_features import AdductFeatureExtractor
from GraphXSection_torch.data_extract import MoleculeIon

# reuse the exact anchored-offset matcher used to build the labels
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_protomer_labels import match_by_anchored_ccs  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("eval_protomer_gap")
log.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def load_model(config, collection):
    global_dim = int(len(AdductFeatureExtractor().extract_features("[M+H]+", smiles="CCO")))
    lm = ProtomerLightningModule(
        config,
        model_type="Sigma",
        node_feature_sizes=collection.node_feature_sizes,
        edge_feature_sizes=collection.edge_feature_sizes,
        global_feature_sizes=global_dim,
    )
    return lm


def load_weights(lm, checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = lm.load_state_dict(state, strict=False)
    if missing:
        log.warning("missing keys: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])
    return lm


@torch.no_grad()
def predict_records(lm, df, collection, batch_size=32, device="cpu",
                    graph_smiles_mode="charged"):
    """Return one record per molecule with all-candidate predictions."""
    ds = ProtomerCorpusDataset(df, collection, eval_all_candidates=True,
                               graph_smiles_mode=graph_smiles_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_protomer, num_workers=0)
    lm = lm.to(device).eval()

    records = []
    row_ptr = 0
    for batch in loader:
        batch_on = {
            "graph": batch["graph"].to(device),
            "global_feats": batch["global_feats"].to(device),
            "charged_atom_index": batch["charged_atom_index"].to(device),
        }
        preds = lm.model(batch_on).view(-1).cpu().numpy()
        mol_index = batch["mol_index"].numpy()
        dom_index = batch["dominant_index"].numpy()
        for m in range(batch["num_molecules"]):
            row = df.iloc[row_ptr]
            row_ptr += 1
            cand_positions = np.where(mol_index == m)[0]
            pred_ccs = preds[cand_positions].tolist()
            dom_global = int(dom_index[m])
            dom_local = int(np.where(cand_positions == dom_global)[0][0])
            obs = [float(x) for x in row["observed_ccs_peaks"]]
            records.append({
                "inchikey": row["inchikey"],
                "adduct": row["adduct"],
                "obs_peaks": obs,
                "pred_ccs": pred_ccs,
                "dominant_local": dom_local,
                "has_gap": bool(row["has_gap"]),
            })
    return records


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def base_parity(records) -> pd.DataFrame:
    rows = []
    for r in records:
        pred_base = r["pred_ccs"][r["dominant_local"]]
        obs_base = r["obs_peaks"][0]
        rows.append({"adduct": r["adduct"], "abs_err": abs(pred_base - obs_base),
                     "rel_err": abs(pred_base - obs_base) / obs_base})
    d = pd.DataFrame(rows)
    overall = pd.DataFrame([{"adduct": "ALL", "n": len(d),
                             "MAE": d["abs_err"].mean(),
                             "median_rel_%": 100 * d["rel_err"].median()}])
    per = (d.groupby("adduct")
             .agg(n=("abs_err", "size"), MAE=("abs_err", "mean"),
                  median_rel_pct=("rel_err", lambda s: 100 * s.median()))
             .reset_index()
             .rename(columns={"median_rel_pct": "median_rel_%"}))
    return pd.concat([overall, per], ignore_index=True)


def _matched_gap_pairs(records):
    """(pred_gap, obs_gap) over anchored-offset-matched secondary peaks."""
    pred_gaps, obs_gaps = [], []
    for r in records:
        if len(r["obs_peaks"]) < 2 or len(r["pred_ccs"]) < 2:
            continue
        matches, _offset, anchor_cand = match_by_anchored_ccs(r["pred_ccs"], r["obs_peaks"])
        if anchor_cand is None:
            continue
        pred_anchor = r["pred_ccs"][anchor_cand]
        obs_base = r["obs_peaks"][0]
        for cand_i, peak_j in matches:
            if peak_j == 0:
                continue  # the anchor pairing is the base peak, not a gap
            pred_gaps.append(r["pred_ccs"][cand_i] - pred_anchor)
            obs_gaps.append(r["obs_peaks"][peak_j] - obs_base)
    return np.array(pred_gaps), np.array(obs_gaps)


def gap_report(pred_gaps, obs_gaps, label="model"):
    n = len(pred_gaps)
    if n == 0:
        return {"label": label, "n_pairs": 0}
    gap_mae = float(np.mean(np.abs(pred_gaps - obs_gaps)))
    zero_mae = float(np.mean(np.abs(obs_gaps)))  # protonation-blind (no gap)
    corr = float(np.corrcoef(pred_gaps, obs_gaps)[0, 1]) if np.std(pred_gaps) > 1e-9 else float("nan")
    sign_acc = float(np.mean(np.sign(pred_gaps) == np.sign(obs_gaps)))
    return {"label": label, "n_pairs": n, "gap_MAE": gap_mae,
            "zero_gap_MAE": zero_mae, "beats_zero_gap": gap_mae < zero_mae,
            "gap_corr": corr, "sign_acc": sign_acc}


def permutation_null(records, seed=0):
    """Match wrong-structure predictions (shuffled among equal-n_cand gap mols)."""
    rng = np.random.default_rng(seed)
    gap = [r for r in records if len(r["obs_peaks"]) >= 2 and len(r["pred_ccs"]) >= 2]
    by_n = {}
    for i, r in enumerate(gap):
        by_n.setdefault(len(r["pred_ccs"]), []).append(i)
    perm = list(range(len(gap)))
    for _n, idxs in by_n.items():
        if len(idxs) < 2:
            continue
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        for src, dst in zip(idxs, shuffled):
            perm[dst] = src
    scrambled = []
    for i, r in enumerate(gap):
        donor = gap[perm[i]]
        scrambled.append({**r, "pred_ccs": donor["pred_ccs"],
                          "dominant_local": min(donor["dominant_local"], len(r["obs_peaks"]) - 1)})
    pg, og = _matched_gap_pairs(scrambled)
    return gap_report(pg, og, label="permutation_null")


@torch.no_grad()
def gold_spotcheck(lm, collection, gold_csv, device="cpu",
                   name_col="Chemical Name", smiles_col="protonated_SMILES",
                   adduct_col="Adduct", ccs_col="CCS"):
    """Directional check on a gold set with explicit per-protomer SMILES + CCS.

    Each molecule's protomers are rows sharing `name_col`; the true
    protomer->peak assignment is known, so we forward every protomer's charged
    SMILES and compare the predicted CCS ordering to the observed one. Uses the
    protomer SMILES for the adduct descriptor (no neutral parent in the file).
    """
    import re

    adduct_map = {"(H+)": "[M+H]+", "(H-)": "[M-H]-", "(-H)": "[M-H]-"}
    df = pd.read_csv(gold_csv, usecols=lambda c: c in
                     {name_col, smiles_col, adduct_col, ccs_col})
    # protomers of one molecule are keyed by a shared base name with a numeric
    # suffix (e.g. Danofloxacin_0 / Danofloxacin_1); group on the base name.
    df["_mol"] = df[name_col].astype(str).str.replace(r"_\d+$", "", regex=True)
    ext = AdductFeatureExtractor()
    lm = lm.to(device).eval()
    rows = []
    for name, g in df.groupby("_mol"):
        g = g.dropna(subset=[smiles_col, ccs_col])
        if len(g) < 2:
            continue
        smis = g[smiles_col].tolist()
        obs = [float(x) for x in g[ccs_col].tolist()]
        raw_adduct = str(g[adduct_col].iloc[0]).strip()
        adduct = adduct_map.get(raw_adduct, raw_adduct)
        graphs = [MoleculeIon(s).get_data(collection) for s in smis]
        gfeat = torch.as_tensor(ext.extract_features(adduct, smiles=smis[0]),
                                dtype=torch.float32)
        item = {"graphs": graphs, "global_feat": gfeat,
                "peaks": torch.as_tensor(obs, dtype=torch.float32),
                "dominant_local_idx": 0}
        batch = collate_protomer([item])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["graph"] = batch["graph"].to(device)
        preds = lm.model(batch).view(-1).cpu().numpy().tolist()
        # rank agreement (do predicted and observed order the protomers alike?)
        order_obs = np.argsort(obs)
        order_pred = np.argsort(preds)
        rank_match = bool(np.array_equal(order_obs, order_pred))
        # widest observed pair: direction + magnitude
        hi, lo = int(np.argmax(obs)), int(np.argmin(obs))
        obs_gap = obs[hi] - obs[lo]
        pred_gap = preds[hi] - preds[lo]
        rows.append({"molecule": name, "n_protomers": len(g),
                     "obs_range": round(obs_gap, 2), "pred_range": round(pred_gap, 2),
                     "dir_ok": bool(np.sign(pred_gap) == np.sign(obs_gap)),
                     "rank_match": rank_match})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--baseline-preds", default=None,
                    help="Optional parquet (inchikey, adduct, pred_ccs) for base-parity head-to-head")
    ap.add_argument("--gold-csv", default=None,
                    help="Optional gold set (Chemical Name, protonated_SMILES, Adduct, CCS) for directional spot-checks")
    ap.add_argument("--limit", type=int, default=None, help="Subsample N molecules (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.load(open(args.config))
    collection = load_feature_config(config["feature_config_path"])

    corpus_path = config[f"{args.split}_data_path"]
    df = pd.read_parquet(corpus_path)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    log.info("Loaded %d molecules from %s", len(df), corpus_path)

    lm = load_model(config, collection)
    lm = load_weights(lm, args.checkpoint)

    graph_mode = config.get("model", {}).get("graph_smiles_mode", "charged")
    records = predict_records(lm, df, collection, device=args.device,
                              graph_smiles_mode=graph_mode)

    print("\n=== 1. Base-CCS parity (dominant candidate vs base peak) ===")
    print(base_parity(records).to_string(index=False))

    print("\n=== 2. Protomer-gap payoff (anchored-offset matched) ===")
    pg, og = _matched_gap_pairs(records)
    model_rep = gap_report(pg, og, label="protomer_model")
    for k, v in model_rep.items():
        print(f"  {k}: {v}")

    print("\n=== 3. Permutation null (wrong-structure predictions) ===")
    null_rep = permutation_null(records)
    for k, v in null_rep.items():
        print(f"  {k}: {v}")
    if model_rep.get("n_pairs") and null_rep.get("n_pairs"):
        print(f"\n  --> model gap_corr {model_rep['gap_corr']:.3f} vs "
              f"null {null_rep['gap_corr']:.3f} "
              f"(structure-specific if model >> null)")

    if args.baseline_preds:
        print("\n=== Base-parity head-to-head vs matched baseline ===")
        bp = pd.read_parquet(args.baseline_preds)
        key = {(r["inchikey"], r["adduct"]): r["pred_ccs"] for _, r in bp.iterrows()}
        errs = []
        for r in records:
            k = (r["inchikey"], r["adduct"])
            if k in key:
                errs.append(abs(key[k] - r["obs_peaks"][0]))
        if errs:
            print(f"  matched-baseline base MAE on {len(errs)} shared molecules: "
                  f"{np.mean(errs):.3f}")

    if args.gold_csv:
        print("\n=== 4. Directional spot-checks (gold protomer set) ===")
        sc = gold_spotcheck(lm, collection, args.gold_csv, device=args.device)
        if len(sc):
            print(sc.to_string(index=False))
            print(f"\n  direction correct: {sc['dir_ok'].mean():.0%}   "
                  f"full rank match: {sc['rank_match'].mean():.0%}  "
                  f"(n={len(sc)} molecules)")
        else:
            print("  no multi-protomer molecules found in gold file")

    print("\nDone")


if __name__ == "__main__":
    main()
