"""PyTorch Lightning data module for GraphXSection_torch."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

from GraphXSection_torch.data_extract import collate_fn

logger = logging.getLogger(__name__)


class GraphXSectionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for GraphXSection datasets."""

    def __init__(
        self,
        train_dataset: Subset,
        valid_dataset: Subset,
        test_dataset: Subset,
        config: Dict[str, Any],
        global_feature_sizes: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        train_dataset : Subset
            Training dataset
        valid_dataset : Subset
            Validation dataset
        test_dataset : Subset
            Test dataset
        config : Dict[str, Any]
            Configuration dictionary
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        # store backing attribute to avoid assigning to the property
        self._global_feature_sizes = global_feature_sizes
        self.config = config

        # Store batch size and num workers
        self.batch_size = config["model"].get("batch_size", 32)

        # Set intelligent default for num_workers based on CPU count
        import os

        default_workers = min(os.cpu_count() or 1, 8)  # Cap at 8 for stability
        self.num_workers = config.get("num_workers", default_workers)

        logger.info(
            f"DataLoader configuration: batch_size={self.batch_size}, num_workers={self.num_workers}"
        )

    def train_dataloader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        # Ensure we pass only the dataset, not a tuple
        dataset = self.test_dataset
        if isinstance(dataset, (tuple, list)):
            # If test_dataset is accidentally a tuple/list, use first element
            dataset = dataset[0]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    @property
    def node_feature_sizes(self) -> Optional[Dict[str, int]]:
        """Return node feature sizes if available."""
        # If train_dataset is a torch.utils.data.Subset, access the underlying dataset.
        dataset = getattr(self.train_dataset, "dataset", self.train_dataset)
        featurizer = getattr(dataset, "featurizer_collection", None)
        if featurizer:
            return featurizer.node_feature_sizes
        return None

    @property
    def edge_feature_sizes(self) -> Optional[Dict[str, int]]:
        """Return edge feature sizes if available."""
        # If train_dataset is a torch.utils.data.Subset, access the underlying dataset.
        dataset = getattr(self.train_dataset, "dataset", self.train_dataset)
        featurizer = getattr(dataset, "featurizer_collection", None)
        if featurizer:
            return featurizer.edge_feature_sizes
        return None

    @property
    def global_feature_sizes(self) -> Optional[Dict[str, int]]:
        """Return global feature sizes if available."""
        logger.info(f"Global feature sizes: {self._global_feature_sizes}")
        return self._global_feature_sizes
