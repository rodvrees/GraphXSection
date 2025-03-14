"""
Module for predicting CCS of a small molecule using a trained model.

Input: 
    - Trained model
    - SMILES string, or file containing SMILES strings
    - Adduct string
    - Configuration dictionary
    - Optional: Path to save output file

Output:
    - Predicted CCS values for each SMILES string
    - Optional: Save output to file
"""

import argparse
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
from logging import getLogger
from pathlib import Path
import json
from typing import Any, Dict

from GraphXSection.encode_graph import get_encoder

logger = getLogger(__name__)


def _parse_config(config_path: Path) -> Dict[str, Any]:
    """
    Parse the configuration file.

    Args:
        config_path (Path): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        ValueError: If the config file is not a JSON file.
    """
    if Path(config_path).suffix.lower() != ".json":
        raise ValueError("Config file must be a JSON file.")

    with open(config_path, "r") as f:
        config = json.load(f)
        logger.debug(config)
    return config


def predict_ccs(model: tf.keras.Model, smiles: str, adduct: str, config: dict) -> float:
    """Predict CCS value for a given SMILES string.

    Args:
        model (tf.keras.Model): Trained model.
        smiles (str): SMILES string.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        float: Predicted CCS value.
    """
    encoder = get_encoder(config=config)
    test_x_adduct = tf.constant([1, 0, 0, 0, 0])
    test_x_mol = encoder(smiles)

    # Create dumy y_test
    y_test = [0]

    model_input = {
        "args_0": tf.reshape(test_x_mol.sizes, [-1]),
        "args_0_1": test_x_mol.node_feature,
        "args_0_2": test_x_mol.edge_src,
        "args_0_3": test_x_mol.edge_dst,
        "args_0_4": test_x_mol.edge_feature,
        "args_0_5": tf.reshape(test_x_adduct, (-1, 5)),
    }
    results = model(**model_input)

    output_key = list(results.keys())[0]
    return results[output_key].numpy()[0][0]


def main():
    parser = argparse.ArgumentParser(
        description="Predict CCS of a small molecule using a trained model."
    )
    parser.add_argument("model", type=str, help="Path to trained model.")
    parser.add_argument(
        "smiles",
        type=str,
        help="SMILES string, or path to file containing SMILES strings.",
    )
    parser.add_argument("adduct", type=str, help="Adduct string.")
    parser.add_argument("config", type=str, help="Path to configuration file.")
    parser.add_argument("--output", type=str, help="Path to save output file.")
    args = parser.parse_args()

    model = tf.saved_model.load(args.model)
    model = model.signatures["serving_default"]

    config = _parse_config(args.config)

    try:
        if os.path.isfile(args.smiles):
            with open(args.smiles, "r") as f:
                smiles = f.readlines()
        else:
            smiles = [args.smiles]
    except Exception as e:
        logger.error(
            f"Smiles should be a string or a path to a file containing SMILES strings. Error: {e}"
        )

    ccs_values = []
    for s in smiles:
        ccs = predict_ccs(model, s, args.adduct, config)
        ccs_values.append(ccs)

    if args.output:
        df = pd.DataFrame({"SMILES": smiles, "CCS": ccs_values})
        df.to_csv(args.output, index=False)
    else:
        for s, ccs in zip(smiles, ccs_values):
            print(f"{s.strip()} - Predicted CCS: {ccs:.2f}")


if __name__ == "__main__":
    main()
