"""Gap-direction analysis: is the Enveda gap_corr real directional signal or an
artifact of the anchored-offset matcher?

For every 2-peak / 2-candidate gap molecule in the test set we compute the signed
observed gap (secondary - base peak, by IM intensity order) and the model's signed
predicted gap under two assignment schemes:

  - MATCHER (Hungarian, the headline metric): the matcher is free to assign the
    base peak to whichever candidate the model predicts closest, then matches the
    rest. This can manufacture concordance.
  - FIXED (matcher-free): the dominant candidate (dominant_site_idx heuristic) is
    ALWAYS the base peak; the other candidate is the secondary. No prediction-based
    reassignment.

If sign-accuracy / correlation are high under MATCHER but collapse under FIXED, the
headline gap_corr is matcher-inflated and the model has not learned protomer
direction. Also breaks the FIXED result down by the site-element pair (does the
model impose a systematic, possibly wrong, N-vs-O ordering?).
"""
from __future__ import annotations

from scripts.eval_protomer_gap import load_model, load_weights, predict_records  # noqa
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from GraphXSection_torch.feature_config import load_feature_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_protomer_labels import match_by_anchored_ccs  # noqa: E402


def _elem(site_name: str) -> str:
    s = str(site_name)
    for suf, e in (("_N", "N"), ("_O", "O"), ("_S", "S")):
        if s.endswith(suf):
            return e
    for e in ("N", "O", "S"):
        if e.lower() in s.lower():
            return e
    return "?"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="outputs/ablations_clean/config_full.json")
    ap.add_argument("--checkpoint", default="outputs/ablations_clean/ablation_full_protomer_model.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = json.load(open(args.config))
    coll = load_feature_config(config["feature_config_path"])
    df = pd.read_parquet(config[f"{args.split}_data_path"])
    gap = df[df["has_gap"]].reset_index(drop=True)          # only gap molecules (fast)
    print(f"gap molecules in {args.split}: {len(gap)}")

    lm = load_weights(load_model(config, coll), args.checkpoint)
    recs = predict_records(lm, gap, coll, device=args.device,
                           graph_smiles_mode=config["model"].get("graph_smiles_mode", "charged"))

    obs_g, pred_matcher, pred_fixed, elem_pair = [], [], [], []
    for r, (_, row) in zip(recs, gap.iterrows()):
        obs = r["obs_peaks"]
        pred = r["pred_ccs"]
        if len(obs) < 2 or len(pred) != 2:            # restrict to the clean 2/2 case
            continue
        dom = r["dominant_local"]
        other = 1 - dom
        og = obs[1] - obs[0]                          # signed observed gap
        # FIXED: dominant is base, other is secondary
        pred_fixed.append(pred[other] - pred[dom])
        # MATCHER: anchored-offset assignment
        matches, _off, anchor = match_by_anchored_ccs(pred, obs)
        pm = np.nan
        if anchor is not None:
            for ci, pj in matches:
                if pj == 1:
                    pm = pred[ci] - pred[anchor]
        pred_matcher.append(pm)
        obs_g.append(og)
        sn = list(row["site_names"])
        elem_pair.append(tuple(sorted((_elem(sn[dom]), _elem(sn[other])))))

    obs_g = np.array(obs_g); pf = np.array(pred_fixed); pm = np.array(pred_matcher)
    ok = ~np.isnan(pm)

    def report(name, pg, og):
        pg, og = np.asarray(pg), np.asarray(og)
        m = ~np.isnan(pg)
        pg, og = pg[m], og[m]
        sign = np.mean(np.sign(pg) == np.sign(og))
        corr = np.corrcoef(pg, og)[0, 1] if np.std(pg) > 1e-9 else float("nan")
        mae = np.mean(np.abs(pg - og))
        print(f"  {name:24s} n={len(pg):5d}  sign_acc={sign:.3f}  corr={corr:+.3f}  gapMAE={mae:.2f}")

    print(f"\n2-peak/2-candidate molecules: {len(obs_g)}")
    print("Direction of the SIGNED gap (secondary - base):")
    report("MATCHER (Hungarian)", pm[ok], obs_g[ok])
    report("FIXED (heuristic anchor)", pf, obs_g)
    print(f"\n  observed gap sign balance: +{np.mean(obs_g>0):.2f} / -{np.mean(obs_g<0):.2f}"
          f"  (|obs_gap| median {np.median(np.abs(obs_g)):.2f})")

    # systematic site-element ordering under FIXED
    print("\nFIXED signed pred gap by site-element pair (dominant vs secondary):")
    ep = np.array([f"{a}/{b}" for a, b in elem_pair])
    rows = []
    for pair in sorted(set(ep)):
        mask = ep == pair
        rows.append({"pair": pair, "n": int(mask.sum()),
                     "obs_gap_mean": round(float(np.mean(obs_g[mask])), 2),
                     "pred_gap_mean": round(float(np.mean(pf[mask])), 2),
                     "sign_agree": round(float(np.mean(np.sign(pf[mask]) == np.sign(obs_g[mask]))), 2)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
