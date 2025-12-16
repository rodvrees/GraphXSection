"""Data extraction functions for GraphXSection."""

import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any
from GraphXSection.models.Eddy import get_embeddings
from GraphXSection.feature_calc import get_feats

logger = logging.getLogger(__name__)


def _drop_constant_columns(df):
    def is_constant(col):
        # Drop NaNs, get unique values
        unique_vals = col.dropna().unique()
        # True if zero or one unique non-NaN value
        return len(unique_vals) <= 1

    cols_to_drop = [col for col in df.columns if is_constant(df[col])]
    logger.info(
        "Dropping constant columns: %s", cols_to_drop
    )  # Log the columns being dropped
    return df.drop(columns=cols_to_drop)


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


def _get_adduct_OHE(df: pd.DataFrame, force_OHE=True) -> np.ndarray:
    """One-hot encode the adducts, ensuring consistent columns across datasets.

    Args:
        df (pd.DataFrame): DataFrame with the 'Adduct' column.
        force_OHE (bool): If True, force one-hot encoding to include all expected adducts.

    Returns:
        np.ndarray: One-hot encoded adducts.
    """
    expected_adducts = ["(+H)", "(-H)", "(+Na)", "(+HCOOH-H)", "(K+)"]
    adduct_ohe = pd.get_dummies(df["Adduct"], prefix="Adduct").astype(int)

    if force_OHE:
        # Ensure all expected adduct columns are present
        for adduct in expected_adducts:
            col_name = f"Adduct_{adduct}"
            if col_name not in adduct_ohe.columns:
                adduct_ohe[col_name] = 0

        # Reorder columns to match expected order
        ordered_columns = [f"Adduct_{a}" for a in expected_adducts]
        adduct_ohe = adduct_ohe[ordered_columns]
        print(f"Adduct OHE columns: {adduct_ohe.columns.tolist()}")

    return adduct_ohe.values


def _get_data(
    df: pd.DataFrame,
    smiles_col="SMILES",
    train=False,
    scaler=None,
    embeddings=False,
    threeD=False,
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
            # "Polarizability Orca",
            # "Dipole moment Orca",
            "Adduct",
            "CCS",
        ]
    )
    smiles = df[smiles_col].values
    adduct_OHE = _get_adduct_OHE(df)

    if threeD:
        feature_cols = df.columns
        feature_id_idx = feature_cols.get_loc("Feature_Identifier")
        calculated_features = df.iloc[:, feature_id_idx + 1 :]

    logger.info("df.columns: %s", df.columns.to_list())

    # all_feats is calculated features + adduct OHE
    if threeD:
        all_feats = np.concatenate(
            (
                adduct_OHE,
                calculated_features.values,
            ),
            axis=1,
        )

    else:
        all_feats = adduct_OHE
    logger.info(f"all_feats shape: {all_feats.shape}")

    y = df["CCS"].values

    # Scale the features if threeD
    if threeD:
        if train:
            scaler = MinMaxScaler()
            all_feats = scaler.fit_transform(all_feats)
        else:
            all_feats = scaler.transform(all_feats)

    logger.info(f"Data shape: {smiles.shape}, {all_feats.shape}, {y.shape}")

    if not embeddings:
        data = {
            "smiles": smiles,
            "global_feats": all_feats,
            "y": y,
        }
    else:
        data = {
            "smiles": smiles,
            "global_feats": all_feats,
            "y": y,
            "embeddings": df["embeddings"].values,
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
    # # Sample first 20 rows for testing
    data = pd.read_csv(config["data_path"])

    if config["model"].get("type", None) == "Ginny3D":
        #     ions = data["Adduct"].unique()
        #     logger.debug(ions)
        #     features_df = get_feats(
        #         data, config, ions=ions, RDKit=True, mordred=True, mordred_3d=True
        #     )
        #     # features_df = features_df.fillna(0.0)
        #     logger.debug(features_df.index)

        #     data["Feature_Identifier"] = data.apply(
        #         lambda x: str(x.name) + "|" + x["Chemical Name"], axis=1
        #     )
        #     logger.debug(data["Feature_Identifier"])
        #     data = pd.merge(
        #         data,
        #         features_df,
        #         left_on="Feature_Identifier",
        #         right_index=True,
        #     )
        #     if config.get("save_dfs", False):
        #         data.to_csv(
        #             f"{config['output_path']}/{config['run_name']}_features_df.csv",
        #             index=False,
        #         )

        data = pd.read_csv(config["data_path"])
        data.drop(["delta_to_second_best"], axis=1, inplace=True)
        # Remove columns where all values are the same
        data.replace(-1000, np.nan, inplace=True)
        data = _drop_constant_columns(data)
        # Drop columns with NaN values
        logger.info("Dropping columns with NaN values")
        logger.debug(
            "Columns with NaN values: %s", len(data.columns[data.isna().any()])
        )
        data.dropna(axis=1, inplace=True)
        logger.info("Dropped columns with NaN values")
        logger.debug("Remaining columns: %s", len(data.columns))

    df_train, df_valid, df_test = _stratified_split(
        data,
        test_split=config["test_split"],
        valid_split=config["valid_split"],
        stratify_col="Adduct",
    )

    if config["model"].get("type", None) == "Eddy":
        df_train = get_embeddings(df_train, config)
        df_valid = get_embeddings(df_valid, config)
        df_test = get_embeddings(df_test, config)
        df_train["Adduct_OHE"] = _get_adduct_OHE(df_train).tolist()
        df_valid["Adduct_OHE"] = _get_adduct_OHE(df_valid).tolist()
        df_test["Adduct_OHE"] = _get_adduct_OHE(df_test).tolist()

        return df_train, df_valid, df_test, None, None

    if config.get("save_dfs", False):
        df_train.to_csv(f"{config['output_path']}/train_df.csv", index=False)
        df_valid.to_csv(f"{config['output_path']}/valid_df.csv", index=False)
        df_test.to_csv(f"{config['output_path']}/test_df.csv", index=False)

    logger.debug(
        f"Training data shape: {df_train.shape}, {df_valid.shape}, {df_test.shape}"
    )

    logger.info("Extracting training data")
    train_data, scaler = _get_data(
        df_train,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=True,
        embeddings=True if config["model"].get("type", None) == "Eddy" else False,
        threeD=True if config["model"].get("type", None) == "Ginny3D" else False,
    )

    logger.info("Extracting validation and test data using fitted scaler")
    valid_data, _ = _get_data(
        df_valid,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=False,
        scaler=scaler,
        embeddings=True if config["model"].get("type", None) == "Eddy" else False,
        threeD=True if config["model"].get("type", None) == "Ginny3D" else False,
    )
    test_data, _ = _get_data(
        df_test,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=False,
        scaler=scaler,
        embeddings=True if config["model"].get("type", None) == "Eddy" else False,
        threeD=True if config["model"].get("type", None) == "Ginny3D" else False,
    )

    logger.info("Data extraction complete")

    return train_data, valid_data, test_data, df_test, df_valid
