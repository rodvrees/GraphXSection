"""Data extraction functions for GraphXSection."""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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


def _get_global_feats(df: pd.DataFrame) -> np.ndarray:
    """Extract global features from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe with the data.

    Returns:
        np.ndarray: Global features.
    """
    feature_names = [
        "Molecular Weight",
        # "Van der Waals Volume",
        # "Radius of Gyration",
        "Asphericity",
        "Dipole Moment",
        # "Polarizability",
    ]

    global_feats = df[feature_names].values
    return global_feats


def _get_data(
    df: pd.DataFrame, smiles_col="SMILES", train=False, scaler=None
) -> Tuple[Dict[str, Any], MinMaxScaler]:
    """Extract data necessary for training from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe with the data.
        train (bool): If True, fit a new scaler. If False, use the provided scaler.
        scaler (MinMaxScaler): Scaler to apply to the features (None if training).

    Returns:
        Tuple[Dict[str, Any], MinMaxScaler]: Extracted data and the scaler (if train=True).
    """

    # Remove rows with NaN values in the SMILES column or the global features
    df = df.dropna(
        subset=[
            smiles_col,
            "Molecular Weight",
            # "Van der Waals Volume",
            # "Radius of Gyration",
            "Asphericity",
            "Dipole Moment",
            # "Polarizability",
            "Adduct",
            "CCS",
        ]
    )
    smiles = df[smiles_col].values
    adduct_OHE = _get_adduct_OHE(df)
    # TODO: if no features in the dataframe, calculate them
    global_feats = _get_global_feats(df)

    logger.debug(f"First few features before scaling: {global_feats[:5]}")

    # Min-max normalize the global features
    if train:
        scaler = MinMaxScaler()
        global_feats = scaler.fit_transform(global_feats)
        logger.debug(f"First few features after scaling: {global_feats[:5]}")
        logger.debug(
            f"Scaler fitted: {scaler}, which has seen {scaler.n_samples_seen_} samples"
        )
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when train=False")

        logger.debug(
            f"Features are scaled using {scaler}, which has seen {scaler.n_samples_seen_} samples"
        )
        global_feats = scaler.transform(global_feats)
        logger.debug(f"First few features after scaling: {global_feats[:5]}")

    # Concatenate the global features and the one-hot encoded adducts
    all_feats = np.concatenate([adduct_OHE, global_feats], axis=1)
    logger.debug(f"First few features after concatenation: {all_feats[:5]}")

    y = df["CCS"].values

    logger.info(f"Data shape: {smiles.shape}, {all_feats.shape}, {y.shape}")

    data = {
        "smiles": smiles,
        "global_feats": all_feats,
        "y": y,
    }
    return data, scaler if train else None  # Return scaler only if train=True


def data_extract(config: Dict[str, Any]) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Extract the data for training the model based on the configuration provided.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        Training, validation, and test data, the validation and test dataframes, and the fitted MinMaxScaler.
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
        f"Training data shape: {df_train.shape}, {df_valid.shape}, {df_test.shape}"
    )

    logger.info("Extracting training data")
    train_data, scaler = _get_data(
        df_train, smiles_col=config.get("smiles_col", "SMILES"), train=True
    )

    logger.info("Extracting validation and test data using fitted scaler")
    valid_data, _ = _get_data(
        df_valid,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=False,
        scaler=scaler,
    )
    test_data, _ = _get_data(
        df_test,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=False,
        scaler=scaler,
    )

    logger.info("Data extraction complete")

    return train_data, valid_data, test_data, df_test, df_valid
