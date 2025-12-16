#!/usr/bin/env python3
"""
Inspect molecular graph featurization in GraphXSection_torch for a given SMILES and adduct.
Prints node features, edge indices/attributes, and global features (adduct OHE).

Usage:
    python -m GraphXSection_torch.inspect_featurization --smiles "CCO" --adduct "(+H)" [--out torch.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import List, Tuple

import torch
from torch_geometric.data import Data

from GraphXSection_torch.encode_graph import MolecularGraphEncoder
from graphamole.featurizers import ALL_FEATURIZERS

# Use the same vocabulary in both scripts to make OHE comparable
ADDUCT_VOCAB: List[str] = [
    "(+H)",
    "(-H)",
    "(+Na)",
    "(+K)",
    "(+HCOOH-H)",
]


def one_hot_adduct(adduct: str, vocab: List[str]) -> Tuple[torch.Tensor, List[str]]:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    try:
        idx = vocab.index(adduct)
        vec[idx] = 1.0
    except ValueError:
        # Unknown adduct → keep zero vector
        pass
    names = [f"Adduct_{a}" for a in vocab]
    return vec, names


def summarize_graph(g: Data, max_show: int = 5) -> dict:
    summary = {}
    # Nodes
    x = getattr(g, "x", None)
    if x is not None:
        summary["x_shape"] = list(x.shape)
        summary["x_dtype"] = str(x.dtype)
        rows = min(max_show, x.size(0))
        summary["x_head"] = x[:rows].detach().cpu().tolist()
    # Edges
    ei = getattr(g, "edge_index", None)
    if ei is not None:
        summary["edge_index_shape"] = list(ei.shape)
        cols = min(max_show, ei.size(1))
        summary["edge_index_head"] = ei[:, :cols].detach().cpu().tolist()
    ea = getattr(g, "edge_attr", None)
    if ea is not None:
        summary["edge_attr_shape"] = list(ea.shape)
        rows = min(max_show, ea.size(0))
        summary["edge_attr_head"] = ea[:rows].detach().cpu().tolist()
    # Batch info if present
    b = getattr(g, "batch", None)
    if b is not None:
        summary["num_graphs"] = int(b.max().item()) + 1
    else:
        summary["num_graphs"] = 1
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Inspect GraphXSection_torch featurization"
    )
    parser.add_argument("--smiles", type=str, default="CCO", help="SMILES string")
    parser.add_argument("--adduct", type=str, default="(+H)", help="Adduct label")
    parser.add_argument("--max-show", type=int, default=5, help="Rows/cols to show")
    parser.add_argument(
        "--out", type=str, default=None, help="Optional path to write JSON output"
    )
    args = parser.parse_args()

    enc = MolecularGraphEncoder()
    graphs = enc([args.smiles])
    if not graphs:
        print("Failed to featurize SMILES.")
        sys.exit(1)
    g = graphs[0]

    # Summarize graph tensor shapes/heads
    summary = summarize_graph(g, max_show=args.max_show)

    # Build feature schemas aligned with Data.x and Data.edge_attr columns
    atom_types = ALL_FEATURIZERS.get_atom_feature_types()
    atom_feats = ALL_FEATURIZERS.atom_featurizers
    atom_schema = []
    for i, (ftype, fobj) in enumerate(zip(atom_types, atom_feats)):
        entry = {
            "col": i,
            "name": fobj.__class__.__name__,
            "type": ftype,
        }
        if ftype == "token" and hasattr(fobj, "vocab"):
            entry["vocab"] = list(fobj.vocab)
        atom_schema.append(entry)

    bond_types = ALL_FEATURIZERS.get_bond_feature_types()
    bond_feats = ALL_FEATURIZERS.bond_featurizers
    bond_schema = []
    for i, (ftype, fobj) in enumerate(zip(bond_types, bond_feats)):
        entry = {
            "col": i,
            "name": fobj.__class__.__name__,
            "type": ftype,
        }
        if ftype == "token" and hasattr(fobj, "vocab"):
            entry["vocab"] = list(fobj.vocab)
        bond_schema.append(entry)

    # Map a few sample rows to human-readable tokens
    node_samples = []
    if hasattr(g, "x") and g.x is not None:
        rows = min(args.max_show, g.x.size(0))
        for r in range(rows):
            row_map = {}
            for i, val in enumerate(g.x[r].tolist()):
                fdef = atom_schema[i]
                if fdef["type"] == "token" and "vocab" in fdef:
                    idx = int(val)
                    label = (
                        fdef["vocab"][idx] if 0 <= idx < len(fdef["vocab"]) else None
                    )
                    row_map[fdef["name"]] = {"index": idx, "label": label}
                else:
                    row_map[fdef["name"]] = float(val)
            node_samples.append(row_map)

    edge_samples = []
    if hasattr(g, "edge_attr") and g.edge_attr is not None and g.edge_attr.numel() > 0:
        rows = min(args.max_show, g.edge_attr.size(0))
        for r in range(rows):
            row_map = {}
            for i, val in enumerate(g.edge_attr[r].tolist()):
                fdef = bond_schema[i]
                if fdef["type"] == "token" and "vocab" in fdef:
                    idx = int(val)
                    label = (
                        fdef["vocab"][idx] if 0 <= idx < len(fdef["vocab"]) else None
                    )
                    row_map[fdef["name"]] = {"index": idx, "label": label}
                else:
                    row_map[fdef["name"]] = float(val)
            edge_samples.append(row_map)

    # Global features (adduct OHE)
    ohe, names = one_hot_adduct(args.adduct, ADDUCT_VOCAB)
    global_feats = {
        "names": names,
        "vector": ohe.tolist(),
        "sum": float(ohe.sum().item()),
    }

    # Build unified comparable structure
    node_names = [e["name"] for e in atom_schema]
    edge_names = [e["name"] for e in bond_schema]
    out = {
        "impl": "torch",
        "smiles": args.smiles,
        "adduct": args.adduct,
        "graph": summary,
        "node_features": {
            "names": node_names,
            "schema": atom_schema,
            "samples": node_samples,
        },
        "edge_features": {
            "names": edge_names,
            "schema": bond_schema,
            "samples": edge_samples,
        },
        "global_feats": global_feats,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print("=== GraphXSection_torch featurization ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
