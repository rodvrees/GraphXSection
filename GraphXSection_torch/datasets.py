"""Lightweight dataset and DataLoader helpers (graphamole-style).

This module lets you: build a dataset from a DataFrame or CSV, split into
train/val/test using fractions, and get DataLoaders. It reuses the existing
GraphXSectionDataset and collate_fn from data_extract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import logging
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Subset
from pathlib import Path

from GraphXSection_torch.data_extract import (
    GraphXSectionDataset,
    collate_fn,
)
from GraphXSection_torch.adduct_features import AdductFeatureExtractor

logger = logging.getLogger(__name__)


def build_dataset_from_df(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    adduct_col: str = "Adduct",
    use_adduct_descriptors: bool = True,
    one_hot_encode_feats: bool = False,
    feature_config_path: Optional[Path] = None,
    feat_include_ref: bool = False,
) -> Tuple[GraphXSectionDataset, int]:
    """Create GraphXSectionDataset from a DataFrame.

    Drops rows missing SMILES/Adduct/CCS and extracts adduct features using
    physicochemical descriptors (default) or legacy one-hot encoding.

    Args:
        df: DataFrame with molecular data
        smiles_col: Column name for SMILES strings
        adduct_col: Column name for adduct information
        use_adduct_descriptors: Whether to use physicochemical descriptors (True, recommended)
                               or legacy one-hot encoding (False)
        one_hot_encode_feats: Whether to use one-hot encoding for molecular features
        feature_config_path: Path to custom feature configuration JSON

    Returns:
        Tuple of (dataset, adduct_feature_dimension)
    """
    df = df.dropna(subset=[smiles_col, adduct_col, "CCS"])  # minimal requirements
    smiles = df[smiles_col].tolist()
    adducts = df[adduct_col].tolist()
    targets = df["CCS"].values

    if use_adduct_descriptors:
        # Use physicochemical descriptors (vocab-free, generalization-friendly)
        logger.info("Extracting physicochemical adduct descriptors (vocab-free)")
        if not feat_include_ref:
            extractor = AdductFeatureExtractor()
            adduct_features = np.stack(
                [
                    extractor.extract_features(adduct, smiles=smi)
                    for adduct, smi in zip(adducts, smiles)
                ]
            ).astype(np.float32)
            logger.debug(f"Adduct descriptor shape: {adduct_features.shape}")
            logger.debug(f"Adduct descriptor sample:\n{adduct_features[:5]}")
        else:
            extractor = AdductFeatureExtractor()
            adduct_features = np.stack(
                [
                    extractor.extract_descriptor_with_reference(adduct, smiles=smi)
                    for adduct, smi in zip(adducts, smiles)
                ]
            ).astype(np.float32)
            logger.debug(f"Adduct descriptor (with ref) shape: {adduct_features.shape}")
            logger.debug(f"Adduct descriptor (with ref) sample:\n{adduct_features[:5]}")
    else:
        # Legacy one-hot encoding (requires vocabulary, not recommended for new work)
        logger.warning(
            "Using legacy one-hot encoding for adducts (not recommended for generalization)"
        )
        from GraphXSection_torch.data_extract import AdductEncoder

        adduct_encoder = AdductEncoder().fit([df[adduct_col]])
        adduct_features = adduct_encoder.transform(df[adduct_col])  # (N, V)
        logger.debug(f"Adduct OHE shape: {adduct_features.shape}")
        logger.debug(f"Adduct OHE sample:\n{adduct_features[:5]}")

    dataset = GraphXSectionDataset(
        smiles,
        adducts,
        adduct_features,
        targets,
        one_hot_encode=one_hot_encode_feats,
        feature_config_path=feature_config_path,
    )
    adduct_feat_dim = adduct_features.shape[1]
    dataset.summarize()
    return dataset, adduct_feat_dim


def build_dataset_from_csv(
    csv_path: str,
    smiles_col: str = "SMILES",
    adduct_col: str = "Adduct",
    use_adduct_descriptors: bool = True,
) -> Tuple[GraphXSectionDataset, pd.DataFrame, int]:
    """Load a CSV and return the dataset, cleaned DataFrame, and feature dimension.

    Args:
        csv_path: Path to CSV file
        smiles_col: Column name for SMILES strings
        adduct_col: Column name for adduct information
        use_adduct_descriptors: Whether to use physicochemical descriptors (True, default)
                               or legacy one-hot encoding (False)

    Returns:
        Tuple of (dataset, dataframe, adduct_feature_dimension)
    """
    df = pd.read_csv(csv_path)
    dataset, adduct_feat_dim = build_dataset_from_df(
        df,
        smiles_col=smiles_col,
        adduct_col=adduct_col,
        use_adduct_descriptors=use_adduct_descriptors,
    )
    return dataset, df, adduct_feat_dim


def split_indices_stratified(
    df: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    stratify_col: str = "Adduct",
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Return stratified train/val/test index lists based on a DataFrame column.

    We group by stratify_col and sample fractions per group for stable splits.
    """
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    grouped = df.groupby(stratify_col, dropna=False)
    for _, group in grouped:
        idxs = group.index.to_numpy()
        n = len(idxs)
        n_test = int(round(test_fraction * n))
        n_val = int(round(valid_fraction * (n - n_test)))
        rng.shuffle(idxs)
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())

    return train_idx, val_idx, test_idx


def build_datasets_from_separate_files(
    train_csv_path: str,
    valid_csv_path: str,
    test_csv_path: str,
    smiles_col: str = "SMILES",
    adduct_col: str = "Adduct",
    use_adduct_descriptors: bool = True,
    one_hot_encode_feats: bool = False,
    feature_config_path: Optional[Path] = None,
) -> Tuple[Subset, Subset, Subset, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Create datasets from separate train/valid/test CSV files.

    This function loads three separate CSV files and creates individual datasets
    for each split using physicochemical adduct descriptors (vocab-free by default).

    Args:
        train_csv_path: Path to training CSV file
        valid_csv_path: Path to validation CSV file
        test_csv_path: Path to test CSV file
        smiles_col: Column name for SMILES strings
        adduct_col: Column name for adduct information
        use_adduct_descriptors: Whether to use physicochemical descriptors (True, default)
                               or legacy one-hot encoding (False)
        one_hot_encode_feats: Whether to use one-hot encoding for molecular features
        feature_config_path: Path to custom feature configuration JSON

    Returns:
        Tuple of (train_subset, valid_subset, test_subset, df_train, df_valid, df_test, adduct_feat_dim)
    """
    logger.info("Loading datasets from separate CSV files...")

    # Load all DataFrames
    df_train = pd.read_csv(train_csv_path)
    df_valid = pd.read_csv(valid_csv_path)
    df_test = pd.read_csv(test_csv_path)

    logger.info(f"Loaded train: {len(df_train)} samples")
    logger.info(f"Loaded valid: {len(df_valid)} samples")
    logger.info(f"Loaded test: {len(df_test)} samples")

    # Create datasets for each split
    train_dataset, adduct_feat_dim = build_dataset_from_df(
        df_train,
        smiles_col=smiles_col,
        adduct_col=adduct_col,
        use_adduct_descriptors=use_adduct_descriptors,
        one_hot_encode_feats=one_hot_encode_feats,
        feature_config_path=feature_config_path,
        feat_include_ref=True,
    )

    valid_dataset, _ = build_dataset_from_df(
        df_valid,
        smiles_col=smiles_col,
        adduct_col=adduct_col,
        use_adduct_descriptors=use_adduct_descriptors,
        one_hot_encode_feats=one_hot_encode_feats,
        feature_config_path=feature_config_path,
        feat_include_ref=True,
    )

    test_dataset, _ = build_dataset_from_df(
        df_test,
        smiles_col=smiles_col,
        adduct_col=adduct_col,
        use_adduct_descriptors=use_adduct_descriptors,
        one_hot_encode_feats=one_hot_encode_feats,
        feature_config_path=feature_config_path,
        feat_include_ref=True,
    )

    # Create Subset objects with full indices (since each dataset is already the correct split)
    train_subset = Subset(train_dataset, list(range(len(train_dataset))))
    valid_subset = Subset(valid_dataset, list(range(len(valid_dataset))))
    test_subset = Subset(test_dataset, list(range(len(test_dataset))))

    logger.info("Successfully created datasets from separate files")

    return (
        train_subset,
        valid_subset,
        test_subset,
        df_train,
        df_valid,
        df_test,
        adduct_feat_dim,
    )


def split_train_val_test(
    dataset: GraphXSectionDataset,
    df: pd.DataFrame,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    stratify_col: str = "Adduct",
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create Subset splits for train/val/test along with their DataFrames.

    Returns (train_subset, val_subset, test_subset, df_train, df_val, df_test).
    """
    train_idx, val_idx, test_idx = split_indices_stratified(
        df, valid_fraction, test_fraction, stratify_col=stratify_col, seed=seed
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    return (
        train_subset,
        val_subset,
        test_subset,
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def get_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset | None,
    test_dataset: torch.utils.data.Dataset | None,
    batch_size: int = 32,
    num_workers: int = None,  # Changed from 0 to None for auto-detection
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader | None,
    torch.utils.data.DataLoader | None,
]:
    """Return train/val/test DataLoaders using the project collate_fn."""
    from torch.utils.data import DataLoader
    from torch_geometric.loader import (
        DataLoader as PyGDataLoader,
    )  # not used; using our collate

    # Set intelligent default for num_workers if not specified
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 20)
        logger.info(
            f"DataLoader configured with {num_workers} workers (auto-detected from {os.cpu_count()} CPU cores)"
        )
    else:
        logger.info(
            f"DataLoader configured with {num_workers} workers (user-specified)"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = (
        DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        if valid_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader
