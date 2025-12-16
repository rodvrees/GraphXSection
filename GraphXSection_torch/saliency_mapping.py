"""Saliency mapping module for GraphXSection_torch - PyTorch implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import logging
from torch.utils.data import DataLoader
from GraphXSection_torch.data_extract import GraphXSectionDataset, collate_fn

logger = logging.getLogger(__name__)


class GradientSaliencyMapper:
    """Gradient-based saliency mapping for graph neural networks."""

    def __init__(self, model: nn.Module):
        """
        Initialize the saliency mapper.

        Parameters
        ----------
        model : nn.Module
            Trained PyTorch model
        """
        self.model = model
        self.model.eval()

    def get_node_attributions(self, batch, target_class=None):
        """
        Calculate node attributions using gradients.

        Parameters
        ----------
        batch : dict
            Batch of graph data
        target_class : int, optional
            Target class for attribution (not used in regression)

        Returns
        -------
        torch.Tensor
            Node attributions
        """
        # Enable gradient computation for node features
        batch["graph"].x.requires_grad_(True)

        # Forward pass
        output = self.model(batch)

        # For regression, we compute gradients w.r.t. the output
        if target_class is None:
            # Sum the output to get a scalar for backprop
            target = output.sum()
        else:
            target = output[target_class]

        # Backward pass
        target.backward()

        # Get gradients
        node_attributions = batch["graph"].x.grad

        # Clean up gradients
        batch["graph"].x.grad = None

        return node_attributions

    def get_edge_attributions(self, batch, target_class=None):
        """
        Calculate edge attributions using gradients.

        Parameters
        ----------
        batch : dict
            Batch of graph data
        target_class : int, optional
            Target class for attribution (not used in regression)

        Returns
        -------
        torch.Tensor
            Edge attributions
        """
        if not hasattr(batch["graph"], "edge_attr") or batch["graph"].edge_attr is None:
            logger.warning("No edge attributes found for edge attribution")
            return None

        # Enable gradient computation for edge features
        batch["graph"].edge_attr.requires_grad_(True)

        # Forward pass
        output = self.model(batch)

        # For regression, we compute gradients w.r.t. the output
        if target_class is None:
            target = output.sum()
        else:
            target = output[target_class]

        # Backward pass
        target.backward()

        # Get gradients
        edge_attributions = batch["graph"].edge_attr.grad

        # Clean up gradients
        batch["graph"].edge_attr.grad = None

        return edge_attributions

    def get_integrated_gradients(self, batch, baseline=None, steps=50):
        """
        Calculate integrated gradients for node features.

        Parameters
        ----------
        batch : dict
            Batch of graph data
        baseline : torch.Tensor, optional
            Baseline for integrated gradients (zeros by default)
        steps : int
            Number of integration steps

        Returns
        -------
        torch.Tensor
            Integrated gradients
        """
        if baseline is None:
            baseline = torch.zeros_like(batch["graph"].x)

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(batch["graph"].x.device)

        integrated_grads = torch.zeros_like(batch["graph"].x)

        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated_x = baseline + alpha * (batch["graph"].x - baseline)
            interpolated_x.requires_grad_(True)

            # Create interpolated batch
            batch_interp = batch.copy()
            batch_interp["graph"].x = interpolated_x

            # Forward pass
            output = self.model(batch_interp)
            target = output.sum()

            # Backward pass
            target.backward()

            # Accumulate gradients
            integrated_grads += interpolated_x.grad / steps

            # Clean up
            interpolated_x.grad = None

        # Multiply by input difference
        integrated_grads *= batch["graph"].x - baseline

        return integrated_grads


def get_saliency_maps(
    model: nn.Module, dataset: GraphXSectionDataset, config: Dict[str, Any] = None
) -> Dict[str, np.ndarray]:
    """
    Generate saliency maps for a given model and dataset.

    Args:
        model: Trained PyTorch model
        dataset: Dataset to generate saliency maps for
        config: Configuration dictionary

    Returns:
        Dictionary containing different types of saliency maps
    """
    logger.info("Generating saliency maps")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create saliency mapper
    saliency_mapper = GradientSaliencyMapper(model)

    # Create data loader (single batch for simplicity)
    dataloader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn
    )

    saliency_maps = {
        "node_attributions": [],
        "edge_attributions": [],
        "integrated_gradients": [],
    }

    with torch.enable_grad():  # Enable gradients for saliency computation
        for batch in dataloader:
            # Move batch to device
            for key in batch:
                if key == "graph":
                    batch[key] = batch[key].to(device)
                else:
                    batch[key] = batch[key].to(device)

            # Calculate node attributions
            node_attr = saliency_mapper.get_node_attributions(batch)
            saliency_maps["node_attributions"].append(node_attr.detach().cpu().numpy())

            # Calculate edge attributions
            edge_attr = saliency_mapper.get_edge_attributions(batch)
            if edge_attr is not None:
                saliency_maps["edge_attributions"].append(
                    edge_attr.detach().cpu().numpy()
                )

            # Calculate integrated gradients
            integrated_grads = saliency_mapper.get_integrated_gradients(batch)
            saliency_maps["integrated_gradients"].append(
                integrated_grads.detach().cpu().numpy()
            )

    # Convert lists to arrays
    for key in saliency_maps:
        if saliency_maps[key]:
            saliency_maps[key] = np.concatenate(saliency_maps[key], axis=0)
        else:
            saliency_maps[key] = None

    logger.info(f"Generated saliency maps with shapes:")
    for key, value in saliency_maps.items():
        if value is not None:
            logger.info(f"  {key}: {value.shape}")
        else:
            logger.info(f"  {key}: None")

    return saliency_maps
