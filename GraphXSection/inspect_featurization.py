#!/usr/bin/env python3
"""
Inspect molecular graph featurization in GraphXSection (TensorFlow/molgraph) for a given SMILES and adduct.
Prints node/edge feature shapes from the molgraph graph object and the global adduct OHE, plus the exact
atom/bond feature names configured by your encoder.

Usage:
    # Using a minimal default encoder
    python -m GraphXSection.inspect_featurization --smiles "CCO" --adduct "(+H)"

    # Using your project config to mirror the exact feature set
    python -m GraphXSection.inspect_featurization --smiles "CCO" --adduct "(+H)" --config path/to/config.json [--out tf.json]
"""
from __future__ import annotations
import argparse
import json
from typing import List, Tuple, Any, Mapping

import tensorflow as tf
from molgraph import layers, chemistry
from GraphXSection import encode_graph as enc_mod
import json as pyjson

# Matches the Torch script to compare apples-to-apples
ADDUCT_VOCAB: List[str] = [
    "(+H)",
    "(-H)",
    "(+Na)",
    "(+K)",
    "(+HCOOH-H)",
]

# Build a reasonable default encoder mirroring GraphXSection.encode_graph.get_encoder
DEFAULT_ATOM_FEATURES = [
    chemistry.features.Symbol(),
    chemistry.features.ChiralCenter(),
    chemistry.features.CIPCode(),
    chemistry.features.CrippenLogPContribution(),
    chemistry.features.CrippenMolarRefractivityContribution(),
    chemistry.features.Degree(),
    chemistry.features.FormalCharge(),
    chemistry.features.GasteigerCharge(),
    chemistry.features.Hybridization(),
    chemistry.features.Aromatic(),
    chemistry.features.HydrogenDonor(),
    chemistry.features.HydrogenAcceptor(),
    chemistry.features.Hetero(),
    chemistry.features.Ring(),
    chemistry.features.RingSize(),
    chemistry.features.LabuteASAContribution(),
    chemistry.features.TotalNumHs(),
    chemistry.features.NumRadicalElectrons(),
    chemistry.features.TotalValence(),
    chemistry.features.TPSAContribution(),
]

DEFAULT_BOND_FEATURES = [
    chemistry.features.BondType(),
    chemistry.features.Conjugated(),
    chemistry.features.Rotatable(),
    chemistry.features.Stereo(),
]


def build_encoder(config_path: str | None):
    """Build encoder either from provided project config or a reasonable default."""
    if config_path:
        with open(config_path, "r") as f:
            cfg = pyjson.load(f)
        # Ensure minimal keys expected by GraphXSection.encode_graph.get_encoder
        if "mol_encoder" not in cfg:
            cfg.setdefault("mol_encoder", {"positional_encoding_dim": 0})
        if "atom_encoder" not in cfg:
            # default to a comprehensive set similar to project defaults (booleans)
            cfg["atom_encoder"] = {k: True for k in enc_mod.ATOM_FEATURES.keys()}
        if "bond_encoder" not in cfg:
            cfg["bond_encoder"] = {k: True for k in enc_mod.BOND_FEATURES.keys()}
        encoder = enc_mod.get_encoder(cfg)
        # Also prepare human-readable feature name lists
        atom_feature_names = []
        for k, v in cfg["atom_encoder"].items():
            if v:
                cls = (
                    enc_mod.ATOM_FEATURES[k].__name__
                    if hasattr(enc_mod.ATOM_FEATURES[k], "__name__")
                    else enc_mod.ATOM_FEATURES[k].__class__.__name__
                )
                atom_feature_names.append({"key": k, "class": cls})
        bond_feature_names = []
        for k, v in cfg["bond_encoder"].items():
            if v:
                cls = (
                    enc_mod.BOND_FEATURES[k].__name__
                    if hasattr(enc_mod.BOND_FEATURES[k], "__name__")
                    else enc_mod.BOND_FEATURES[k].__class__.__name__
                )
                bond_feature_names.append({"key": k, "class": cls})
        return encoder, atom_feature_names, bond_feature_names
    else:
        atom_encoder = chemistry.Featurizer(DEFAULT_ATOM_FEATURES)
        bond_encoder = chemistry.Featurizer(DEFAULT_BOND_FEATURES)
        encoder = chemistry.MolecularGraphEncoder(
            atom_encoder,
            bond_encoder,
            positional_encoding_dim=0,
        )
        # Names from defaults
        atom_feature_names = [f.__class__.__name__ for f in DEFAULT_ATOM_FEATURES]
        bond_feature_names = [f.__class__.__name__ for f in DEFAULT_BOND_FEATURES]
        return encoder, atom_feature_names, bond_feature_names


def one_hot_adduct(adduct: str, vocab: List[str]):
    vec = tf.zeros((len(vocab),), dtype=tf.float32)
    try:
        idx = vocab.index(adduct)
        vec = tf.tensor_scatter_nd_update(vec, [[idx]], [1.0])
    except ValueError:
        pass
    names = [f"Adduct_{a}" for a in vocab]
    return vec, names


def _shape_and_head(arr: Any, max_rows: int = 5):
    try:
        if hasattr(arr, "shape"):
            shape = list(arr.shape)
        else:
            shape = None
        head = None
        if hasattr(arr, "numpy"):
            a = arr
            if len(a.shape) >= 1:
                head = a[: min(max_rows, a.shape[0])].numpy().tolist()
        return shape, head
    except Exception:
        return None, None


def summarize_molgraph(graph) -> dict:
    # graph may be a dict-like object from molgraph encoder; fall back to private attrs if needed
    summary = {}
    try:
        if isinstance(graph, Mapping):
            for k, v in graph.items():
                shape, head = _shape_and_head(v)
                summary[k] = {"shape": shape, "head": head}
        else:
            # Try common private attrs
            x = getattr(graph, "_x", None)
            e = getattr(graph, "_e", None)
            if x is not None:
                shape, head = _shape_and_head(x)
                summary["_x"] = {"shape": shape, "head": head}
            if e is not None:
                shape, head = _shape_and_head(e)
                summary["_e"] = {"shape": shape, "head": head}
            if hasattr(graph, "_edge_index"):
                ei = getattr(graph, "_edge_index")
                shape, head = _shape_and_head(ei)
                summary["_edge_index"] = {"shape": shape, "head": head}
    except Exception as err:
        summary["error"] = f"Failed to summarize molgraph: {err}"
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Inspect GraphXSection (TF) featurization"
    )
    parser.add_argument("--smiles", type=str, default="CCO", help="SMILES string")
    parser.add_argument("--adduct", type=str, default="(+H)", help="Adduct label")
    parser.add_argument("--max-show", type=int, default=5, help="Rows/cols to show")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config to build exact encoder",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Optional path to write JSON output"
    )
    args = parser.parse_args()

    enc, atom_feature_names, bond_feature_names = build_encoder(args.config)
    graph = enc([args.smiles])  # returns a molgraph graph structure (often dict-like)

    summary = summarize_molgraph(graph)
    ohe, names = one_hot_adduct(args.adduct, ADDUCT_VOCAB)
    global_feats = {
        "names": names,
        "vector": ohe.numpy().tolist(),
        "sum": float(tf.reduce_sum(ohe).numpy()),
    }

    out = {
        "impl": "tf",
        "smiles": args.smiles,
        "adduct": args.adduct,
        "graph": summary,
        "node_features": {
            "names": atom_feature_names,
        },
        "edge_features": {
            "names": bond_feature_names,
        },
        "global_feats": global_feats,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print("=== GraphXSection (TF/molgraph) featurization ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
