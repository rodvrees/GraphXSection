"""Ablation sweep for the protomer-aware CCS model.

Leave-one-out from the full model over the four switches the plan calls out:

  full            charged SMILES + IsProtonationSite + Gasteiger + charged-atom readout
  no_isprot       drop the IsProtonationSite atom feature
  no_gasteiger    drop the GasteigerChargeBinned atom feature
  no_readout      charged_atom_readout = false
  neutral_smiles  featurize the neutral parent instead of the charged candidate
  primary_only    train on the base (primary) CCS only: no secondary peaks, no gap
                  term. Same architecture/features/data; isolates the gap objective.

Each variant is the full model with exactly one component removed, so the change
in the gap metric attributes that component's contribution on identical data/splits.
Evaluation always forwards all candidates (even for primary_only) so the gap a
primary-only-trained model produces is measured on the same held-out set.

This script GENERATES a self-contained config + feature-config per variant (so the
variants can be trained on other compute), and can optionally train and/or evaluate
them and assemble one results table.

Modes:
  (default)     write per-variant configs and print the train/eval commands (no runs)
  --run         also train (subprocess) then evaluate each variant, build the table
  --eval-only   skip training; evaluate each variant's existing checkpoint, build the table

Checkpoint per variant (produced by train_protomer.py):
  <output-dir>/ablation_<name>_protomer_model.pth

Usage:
  python scripts/run_ablations.py                      # generate configs + commands
  python scripts/run_ablations.py --run --max-epochs 40
  python scripts/run_ablations.py --eval-only --split test
"""
from __future__ import annotations

# import order (graphamole namespace gotcha): SigmaModel first
from GraphXSection_torch.models.Sigma import SigmaModel  # noqa: F401

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from GraphXSection_torch.feature_config import load_feature_config

# reuse the eval building blocks
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_protomer_gap as ev  # noqa: E402

REPO = Path("/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol")
BASE_CONFIG = REPO / "configs/sigma_config_protomer.json"
FEATURES_PROTOMER = REPO / "configs/features_protomer.json"

# Each variant is the full model with one override. Recognised override keys:
#   drop_feature          atom feature to remove from features_protomer.json
#   charged_atom_readout  model.charged_atom_readout
#   graph_smiles_mode     model.graph_smiles_mode ("charged"|"neutral")
#   primary_only          model.primary_only (train on base peak only, no gap term)
#   gap_lambda            model.gap_lambda
#   mcp_monitor           checkpoint monitor metric
ABLATIONS = {
    "full":           {},
    "no_isprot":      {"drop_feature": "IsProtonationSite"},
    "no_gasteiger":   {"drop_feature": "GasteigerChargeBinned"},
    "no_readout":     {"charged_atom_readout": False},
    "neutral_smiles": {"graph_smiles_mode": "neutral"},
    # primary-only: same architecture/features/data, but trained on the base CCS
    # only (no secondary peaks, gap term never fires). Isolates the gap objective
    # itself. val_gap_mae is meaningless here (no gap pairs), so monitor val_loss.
    "primary_only":   {"primary_only": True, "gap_lambda": 0.0, "mcp_monitor": "val_loss"},
}


def make_feature_config(drop_feature, out_path: Path):
    cfg = json.load(open(FEATURES_PROTOMER))
    if drop_feature:
        cfg["atom_features"] = [f for f in cfg["atom_features"] if f["type"] != drop_feature]
    json.dump(cfg, open(out_path, "w"), indent=2)
    return out_path


def make_variant_config(name, overrides, out_dir: Path, epochs, use_wandb):
    feat_path = make_feature_config(overrides.get("drop_feature"),
                                    out_dir / f"features_{name}.json")
    cfg = copy.deepcopy(json.load(open(BASE_CONFIG)))
    cfg["run_name"] = f"ablation_{name}"
    cfg["feature_config_path"] = str(feat_path)
    cfg["output_path"] = str(out_dir)
    cfg["model"]["charged_atom_readout"] = overrides.get("charged_atom_readout", True)
    cfg["model"]["graph_smiles_mode"] = overrides.get("graph_smiles_mode", "charged")
    cfg["model"]["primary_only"] = overrides.get("primary_only", False)
    if "gap_lambda" in overrides:
        cfg["model"]["gap_lambda"] = overrides["gap_lambda"]
    if "mcp_monitor" in overrides:
        cfg["mcp"]["mcp_monitor"] = overrides["mcp_monitor"]
    if epochs is not None:
        cfg["model"]["epochs"] = epochs
    cfg["mcp"]["mcp_path"] = f"{out_dir}/ablation_{name}_best.pth"
    cfg["wandb"]["use"] = use_wandb
    path = out_dir / f"config_{name}.json"
    json.dump(cfg, open(path, "w"), indent=2)
    return feat_path, path


def evaluate_variant(config_path, ckpt_path, split, device, limit=None):
    """Run the eval building blocks and return a one-row metrics dict."""
    config = json.load(open(config_path))
    collection = load_feature_config(config["feature_config_path"])
    df = pd.read_parquet(config[f"{split}_data_path"])
    if limit:
        df = df.head(limit).reset_index(drop=True)
    lm = ev.load_model(config, collection)
    lm = ev.load_weights(lm, ckpt_path)
    graph_mode = config.get("model", {}).get("graph_smiles_mode", "charged")
    records = ev.predict_records(lm, df, collection, device=device,
                                 graph_smiles_mode=graph_mode)

    bp = ev.base_parity(records)
    base_all = bp[bp["adduct"] == "ALL"].iloc[0]
    pg, og = ev._matched_gap_pairs(records)
    model_rep = ev.gap_report(pg, og, "model")
    null_rep = ev.permutation_null(records)
    return {
        "base_MAE": round(float(base_all["MAE"]), 3),
        "base_med_rel_%": round(float(base_all["median_rel_%"]), 2),
        "n_gap_pairs": model_rep.get("n_pairs", 0),
        "gap_MAE": round(model_rep.get("gap_MAE", float("nan")), 3),
        "zero_gap_MAE": round(model_rep.get("zero_gap_MAE", float("nan")), 3),
        "beats_zero_gap": model_rep.get("beats_zero_gap"),
        "gap_corr": round(model_rep.get("gap_corr", float("nan")), 3),
        "null_corr": round(null_rep.get("gap_corr", float("nan")), 3),
        "sign_acc": round(model_rep.get("sign_acc", float("nan")), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/ablations"))
    ap.add_argument("--run", action="store_true", help="train (subprocess) then eval each variant")
    ap.add_argument("--eval-only", action="store_true", help="eval existing checkpoints only")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="subsample N eval molecules")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false",
                    help="disable W&B for training runs (on by default)")
    ap.set_defaults(wandb=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. generate configs
    variants = {}
    for name, overrides in ABLATIONS.items():
        _feat_path, cfg_path = make_variant_config(name, overrides, out_dir,
                                                   args.max_epochs, args.wandb)
        ckpt = out_dir / f"ablation_{name}_protomer_model.pth"
        variants[name] = {"config": cfg_path, "ckpt": ckpt}
    print(f"Wrote {len(variants)} variant configs to {out_dir}\n")

    # 2. train (optional)
    if args.run and not args.eval_only:
        for name, v in variants.items():
            print(f"=== training {name} ===")
            cmd = [sys.executable, str(REPO / "scripts/train_protomer.py"),
                   "--config", str(v["config"])]
            if args.max_epochs is not None:
                cmd += ["--max-epochs", str(args.max_epochs)]
            r = subprocess.run(cmd, cwd=str(REPO), env={**_env(), "PYTHONPATH": str(REPO)})
            if r.returncode != 0:
                print(f"  training {name} failed (rc={r.returncode}); skipping its eval")
                v["ckpt"] = None

    # dry-run: just print the commands the user should run on their compute
    if not args.run and not args.eval_only:
        print("To train each variant (e.g. on a cluster), run:")
        for name, v in variants.items():
            print(f"  PYTHONPATH=. python scripts/train_protomer.py --config {v['config']}")
        print("\nThen assemble the table with:")
        print(f"  PYTHONPATH=. python scripts/run_ablations.py --eval-only --split {args.split}")
        return

    # 3. eval + assemble table
    rows = []
    for name, v in variants.items():
        ckpt = v["ckpt"]
        if ckpt is None or not Path(ckpt).exists():
            print(f"  {name}: checkpoint missing ({ckpt}); skipping")
            continue
        print(f"=== evaluating {name} ===")
        metrics = evaluate_variant(v["config"], str(ckpt), args.split, args.device, args.limit)
        rows.append({"variant": name, **metrics})

    if not rows:
        print("No checkpoints evaluated. Train the variants first (see configs in %s)." % out_dir)
        return

    table = pd.DataFrame(rows)
    csv_path = out_dir / f"ablation_results_{args.split}.csv"
    md_path = out_dir / f"ablation_results_{args.split}.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text("# Protomer ablation results (%s split)\n\n" % args.split
                       + table.to_markdown(index=False) + "\n")
    print("\n=== Ablation results (%s) ===" % args.split)
    print(table.to_string(index=False))
    print(f"\nWrote {csv_path}\n      {md_path}")


def _env():
    import os
    return {k: v for k, v in os.environ.items()}


if __name__ == "__main__":
    main()
