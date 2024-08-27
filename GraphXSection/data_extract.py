"""Data extraction functions for GraphXSection."""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def _stratified_split(
    df: pd.DataFrame,
    test_split: float = 0.1,
    valid_split: float = 0.1,
    stratify_col: str = "Adduct",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified split of the datafram into training, validation, and test sets.

    Args:
        df (pd.DataFrame): Dataframe to split.
        test_split (float): Fraction of the data to use for testing.
        valid_split (float): Fraction of the data to use for validation.
        stratify_col (str): Column to stratify the data on.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Dataframes for training, validation, and testing.
    """

    # Initialize empty dataframes for train, valid, and test
    df_train = pd.DataFrame(columns=df.columns)
    df_valid = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)

    # Group by the stratify column
    grouped = df.groupby(stratify_col)

    for _, group in grouped:
        # Split the group into train and test
        train, test = train_test_split(group, test_size=test_split, random_state=42)
        # Split the train group into train and valid
        train, valid = train_test_split(train, test_size=valid_split, random_state=42)

        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, valid])
        df_test = pd.concat([df_test, test])

    return df_train, df_valid, df_test


def _get_adduct_OHE(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode the adducts.

    Args:
        df (pd.DataFrame): Dataframe with the adducts.

    Returns:
        np.ndarray: One-hot encoded adducts.
    """
    adduct_ohe = pd.get_dummies(df["Adduct"], prefix="Adduct").astype(int)
    return adduct_ohe.values


def _get_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract data necessary for training from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe with the data.

    Returns:
        Dict[str, Any]: Dictionary with extracted data, including SMILES, one-hot encoded adducts and CCS.
    """
    smiles = df["SMILES"].values
    adduct_OHE = _get_adduct_OHE(df)
    y = df["CCS"].values

    logger.info(f"Data shape: {smiles.shape}, {adduct_OHE.shape}, {y.shape}")

    data = {
        "smiles": smiles,
        "adduct_OHE": adduct_OHE,
        "y": y,
    }
    return data


def data_extract(
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Extract the data for training the model based on the configuration provided.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]: Training, validation, and test data, and the validation and test dataframes.
    """

    logger.info("Reading data file: %s", config["data_path"])
    data = pd.read_csv(config["data_path"])

    df_train, df_valid, df_test = _stratified_split(
        data,
        test_split=config["test_split"],
        valid_split=config["valid_split"],
        stratify_col="Adduct",
    )

    if config.get("save_dfs", False):
        df_train.to_csv(f"{config['output_path']}/train_df.csv", index=False)
        df_valid.to_csv(f"{config['output_path']}/valid_df.csv", index=False)
        df_test.to_csv(f"{config['output_path']}/test_df.csv", index=False)

    logger.debug(
        f"Training data shape: {df_train.shape}\n{df_valid.shape}\n{df_test.shape}"
    )

    logger.info("Extracting data")
    train_data = _get_data(df_train)
    valid_data = _get_data(df_valid)
    test_data = _get_data(df_test)
    logger.info("Data extraction complete")

    return train_data, valid_data, test_data, df_test, df_valid
