import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from typing import Dict, Any, Optional, List, Union
from graphamole.graphamole.layers.GINConv import GINConv
from graphamole.graphamole.layers.AttentiveFPReadout import AttentiveFPReadout
import logging

# Import integrated readout classes
try:
    from .ReadoutIntegrated import (
        SetGatherReadoutAndConcatGlobal,
        SegmentPoolingReadoutAndConcatGlobal,
        AttentiveFPReadoutAndConcatGlobal,
    )
except ImportError:
    # Fallback - define simple versions inline if import fails
    class SetGatherReadoutAndConcatGlobal(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.pool_fn = global_mean_pool

        def forward(self, node_features, batch_idx, global_features=None):
            pooled = self.pool_fn(node_features, batch_idx)
            if global_features is not None:
                return torch.cat([pooled, global_features], dim=-1)
            return pooled

    class SegmentPoolingReadoutAndConcatGlobal(nn.Module):
        def __init__(self, mode="mean", **kwargs):
            super().__init__()
            if mode == "mean":
                self.pool_fn = global_mean_pool
            elif mode == "sum":
                self.pool_fn = global_add_pool
            else:
                self.pool_fn = global_max_pool

        def forward(self, node_features, batch_idx, global_features=None):
            pooled = self.pool_fn(node_features, batch_idx)
            if global_features is not None:
                return torch.cat([pooled, global_features], dim=-1)
            return pooled

    class AttentiveFPReadoutAndConcatGlobal(nn.Module):
        def __init__(self, node_dim, steps=4, **kwargs):
            super().__init__()
            self.readout = AttentiveFPReadout(node_dim=node_dim, steps=steps)

        def forward(self, node_features, edge_index, batch_idx, global_features=None):
            pooled = self.readout(node_features, edge_index, batch_idx)
            if global_features is not None:
                return torch.cat([pooled, global_features], dim=-1)
            return pooled


logger = logging.getLogger(__name__)


class GinnyModel(nn.Module):
    """
    PyTorch Geometric implementation of the Ginny model architecture.

    Based on the TensorFlow/Molgraph Ginny model that uses GINConv layers
    for molecular property prediction with optional global features.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        node_feature_sizes: Union[int, List[int]],
        edge_feature_sizes: Union[int, List[int]],
        global_feature_sizes: int = 0,
    ):
        super().__init__()

        self.config = config
        m = config["model"]

        # Handle feature dimensions
        if isinstance(node_feature_sizes, list):
            self.node_dim = (
                sum(node_feature_sizes)
                if config.get("one_hot_encode_feats", False)
                else len(node_feature_sizes)
            )
        else:
            self.node_dim = node_feature_sizes

        if isinstance(edge_feature_sizes, list):
            self.edge_dim = (
                sum(edge_feature_sizes)
                if config.get("one_hot_encode_feats", False)
                else len(edge_feature_sizes)
            )
        else:
            self.edge_dim = edge_feature_sizes

        self.global_dim = global_feature_sizes

        self.one_hot_encode_feats = config.get("one_hot_encode_feats", False)

        # Model hyperparameters
        self.gnn_units = m.get("gnn_units", 64)
        self.num_gnn_layers = m.get("num_GNN_layers", 3)
        self.dense_units = m.get("dense_units", 64)
        self.num_dense_layers = m.get("num_dense_layers", 3)
        self.dropout = m.get("dropout", 0.0)
        self.normalization = m.get("normalization", None)
        self.readout = m.get("readout", "mean")
        self.kernel_initializer = m.get("kernel_initializer", "truncated_normal")

        # AttentiveFP readout parameters
        self.attentive_fp_steps = m.get("attentive_fp_steps", 4)

        # Global feature processing configuration
        self.global_feature_mode = m.get(
            "global_feature_mode", "raw"
        )  # "processed", "raw", "none"

        # Log all extracted hyperparameters for verification
        logger.info("=" * 80)
        logger.info("Ginny Model Initialization - Hyperparameters:")
        logger.info(f"  Architecture:")
        logger.info(f"    - GNN layers: {self.num_gnn_layers}")
        logger.info(f"    - GNN units: {self.gnn_units}")
        logger.info(f"    - Dense layers: {self.num_dense_layers}")
        logger.info(f"    - Dense units: {self.dense_units}")
        logger.info(f"  Regularization:")
        logger.info(f"    - Dropout: {self.dropout}")
        logger.info(f"    - Normalization: {self.normalization}")
        logger.info(f"  Pooling:")
        logger.info(f"    - Readout method: {self.readout}")
        logger.info(f"    - AttentiveFP steps: {self.attentive_fp_steps}")
        logger.info(f"  GIN-specific:")
        logger.info(f"    - Train eps: {m.get('train_eps', True)}")
        logger.info(
            f"    - Apply self projection: {m.get('apply_self_projection', False)}"
        )
        logger.info(f"    - Jumping knowledge: {m.get('jumping_knowledge', None)}")
        logger.info(f"  Feature encoding:")
        logger.info(f"    - One-hot encode: {self.one_hot_encode_feats}")
        logger.info(f"    - Node dim: {self.node_dim}")
        logger.info(f"    - Edge dim: {self.edge_dim}")
        logger.info(f"    - Global dim: {self.global_dim}")
        logger.info(f"    - Global feature mode: {self.global_feature_mode}")
        logger.info("=" * 80)

        # Handle embeddings vs projections for different encoding types
        if not self.one_hot_encode_feats:
            # Use embeddings for categorical features
            if isinstance(node_feature_sizes, list):
                self.node_embeddings = nn.ModuleList(
                    [
                        nn.Embedding(vocab_size, self.gnn_units)
                        for vocab_size in node_feature_sizes
                    ]
                )
            else:
                self.node_embeddings = nn.Embedding(node_feature_sizes, self.gnn_units)

            if isinstance(edge_feature_sizes, list):
                self.edge_embeddings = nn.ModuleList(
                    [
                        nn.Embedding(vocab_size, self.gnn_units)
                        for vocab_size in edge_feature_sizes
                    ]
                )
            else:
                self.edge_embeddings = nn.Embedding(edge_feature_sizes, self.gnn_units)
        else:
            # Use projections for one-hot encoded features (created dynamically)
            self.node_projection = None
            self.edge_projection = None

        # GIN Convolution layers
        self.gin_layers = nn.ModuleList()

        # First layer (from input features to hidden)
        first_layer = GINConv(
            in_channels=self.gnn_units,
            units=self.gnn_units,
            edge_dim=self.gnn_units,  # Pass edge dimension
            normalization=self.normalization,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            use_edge_features=True,
            apply_relu_activation=m.get(
                "apply_relu_activation", False
            ),  # MolGraph compatibility
            train_eps=m.get("train_eps", True),  # Enable trainable epsilon
            self_projection=False,  # Match MolGraph's GIN model
        )
        self.gin_layers.append(first_layer)

        # Subsequent layers
        for i in range(self.num_gnn_layers - 1):
            layer = GINConv(
                in_channels=self.gnn_units,
                units=self.gnn_units,
                edge_dim=self.gnn_units,  # Pass edge dimension
                normalization=self.normalization,
                dropout=self.dropout,
                kernel_initializer=self.kernel_initializer,
                use_edge_features=True,
                apply_relu_activation=m.get("apply_relu_activation", False),
                train_eps=m.get("train_eps", True),
                self_projection=False,  # Match MolGraph's GIN model
            )
            self.gin_layers.append(layer)

        # Readout/Pooling
        self._setup_readout()

        # Global features processing (for Ginny3D-like functionality)
        self.global_dense_layers = None
        if self.global_dim > 0 and self.global_feature_mode == "processed":
            global_layers = []
            global_input_dim = self.global_dim
            num_global_layers = m.get("num_dense_layers_global", 2)
            global_units = m.get("dense_units_global", 64)

            for i in range(num_global_layers):
                # Reduce dimension progressively as in original
                out_dim = max(global_units // (i + 2), 16)
                global_layers.extend(
                    [
                        nn.Linear(global_input_dim, out_dim),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                    ]
                )
                global_input_dim = out_dim

            self.global_dense_layers = nn.Sequential(*global_layers)
            self.final_global_dim = global_input_dim
        elif self.global_dim > 0 and self.global_feature_mode == "raw":
            # Use raw global features without processing
            self.final_global_dim = self.global_dim
        else:
            # No global features used
            self.final_global_dim = 0

        # Dense prediction head - calculate input dimension based on readout type
        if self.readout_layer is not None:
            # Integrated readouts include global features in their output
            if isinstance(self.readout_layer, SetGatherReadoutAndConcatGlobal):
                # SetGather doubles the node dimension (like MolGraph)
                base_readout_dim = self.gnn_units * 2
            else:
                # Other readouts keep original dimension
                base_readout_dim = self.gnn_units

            if self.global_feature_mode == "processed" and self.global_dim > 0:
                # Global features are processed and concatenated by integrated readout
                input_dim = base_readout_dim + self.final_global_dim
            elif self.global_feature_mode == "raw" and self.global_dim > 0:
                # Raw global features concatenated by integrated readout
                input_dim = base_readout_dim + self.global_dim
            else:
                # No global features
                input_dim = base_readout_dim
        else:
            # Standard readouts - global features handled separately
            input_dim = self.gnn_units + self.final_global_dim

        dense_layers = []
        for i in range(self.num_dense_layers):
            dense_layers.extend(
                [
                    nn.Linear(input_dim, self.dense_units),
                    nn.ReLU(),
                ]
            )
            if self.dropout > 0:
                dense_layers.append(nn.Dropout(self.dropout))
            input_dim = self.dense_units

        # Final output layer (linear activation as in original)
        dense_layers.append(nn.Linear(input_dim, 1))

        self.dense_head = nn.Sequential(*dense_layers)

        # Initialize weights to match TensorFlow defaults
        self.apply(self._tf_style_init)

    def _setup_readout(self):
        """Setup the readout/pooling mechanism based on configuration"""
        if self.readout == "attentive_fp":
            # Use integrated AttentiveFP readout with global feature concatenation
            self.readout_layer = AttentiveFPReadoutAndConcatGlobal(
                node_dim=self.gnn_units, steps=self.attentive_fp_steps
            )
            self.pool_fn = None
        elif self.readout == "set_gather":
            # Use integrated SetGather readout with global feature concatenation
            self.readout_layer = SetGatherReadoutAndConcatGlobal()
            self.pool_fn = None
        elif self.readout == "segment_pooling":
            # Use integrated SegmentPooling readout with global feature concatenation
            readout_mode = self.config.get("model", {}).get("readout_mode", "mean")
            self.readout_layer = SegmentPoolingReadoutAndConcatGlobal(mode=readout_mode)
            self.pool_fn = None
        elif self.readout == "sum":
            self.pool_fn = global_add_pool
            self.readout_layer = None
        elif self.readout == "mean":
            self.pool_fn = global_mean_pool
            self.readout_layer = None
        elif self.readout == "max":
            self.pool_fn = global_max_pool
            self.readout_layer = None
        else:
            # Default to mean pooling
            self.pool_fn = global_mean_pool
            self.readout_layer = None
            logger.warning(
                f"Unknown readout method '{self.readout}', defaulting to mean pooling"
            )

    def _tf_style_init(self, module):
        """Initialize weights to match TensorFlow defaults more closely"""
        if isinstance(module, nn.Linear):
            # TensorFlow uses glorot_uniform by default
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_weights(self, module):
        """Initialize weights similar to TensorFlow/Molgraph defaults"""
        if isinstance(module, nn.Linear):
            # Use truncated normal with std=0.005 to match TF initialization
            nn.init.trunc_normal_(module.weight, std=0.005)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.005)

    def _compute_node_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings from input features"""
        if not self.one_hot_encode_feats:
            # Categorical embedding
            if isinstance(self.node_embeddings, nn.ModuleList):
                # Multiple categorical features - sum embeddings
                emb = torch.zeros(x.size(0), self.gnn_units, device=x.device)
                for i, embedding in enumerate(self.node_embeddings):
                    emb = emb + embedding(x[:, i].long())
                return emb
            else:
                # Single categorical feature
                return self.node_embeddings(x.long())
        else:
            # One-hot encoded features - create projection dynamically
            if self.node_projection is None:
                self.node_projection = nn.Linear(x.size(1), self.gnn_units).to(x.device)
                nn.init.trunc_normal_(self.node_projection.weight, std=0.005)
                nn.init.zeros_(self.node_projection.bias)
            return self.node_projection(x.float())

    def _compute_edge_embedding(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute edge embeddings from input features"""
        if not self.one_hot_encode_feats:
            # Categorical embedding
            if isinstance(self.edge_embeddings, nn.ModuleList):
                # Multiple categorical features - sum embeddings
                emb = torch.zeros(
                    edge_attr.size(0), self.gnn_units, device=edge_attr.device
                )
                for i, embedding in enumerate(self.edge_embeddings):
                    emb = emb + embedding(edge_attr[:, i].long())
                return emb
            else:
                # Single categorical feature
                return self.edge_embeddings(edge_attr.long())
        else:
            # One-hot encoded features - create projection dynamically
            if self.edge_projection is None:
                self.edge_projection = nn.Linear(edge_attr.size(1), self.gnn_units).to(
                    edge_attr.device
                )
                nn.init.trunc_normal_(self.edge_projection.weight, std=0.005)
                nn.init.zeros_(self.edge_projection.bias)
            return self.edge_projection(edge_attr.float())

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Ginny model.

        Args:
            batch: Dictionary containing:
                - "graph": PyG Data object with x, edge_index, edge_attr, batch
                - "global_feats": Optional global features [batch_size, global_dim]

        Returns:
            Predictions [batch_size, 1]
        """
        graph = batch["graph"]
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        batch_idx = graph.batch
        global_feats = batch.get("global_feats", None)

        # Compute embeddings
        x_emb = self._compute_node_embedding(x)
        edge_emb = self._compute_edge_embedding(edge_attr)

        # Apply GIN convolution layers
        h = x_emb
        for gin_layer in self.gin_layers:
            h = gin_layer(h, edge_index, edge_emb)

        # Apply readout based on configuration
        if self.readout_layer is not None:
            # Use integrated readout that handles global feature concatenation
            if isinstance(self.readout_layer, AttentiveFPReadoutAndConcatGlobal):
                # AttentiveFP needs edge_index
                h = self.readout_layer(h, edge_index, batch_idx, global_feats)
            else:
                # SetGather and SegmentPooling don't need edge_index
                h = self.readout_layer(h, batch_idx, global_feats)

            # Global features already concatenated by integrated readout
            global_feats = None  # Mark as handled
        else:
            # Use standard global pooling
            h = self.pool_fn(h, batch_idx)

        # Process global features if not already handled by integrated readout
        if global_feats is not None:
            if (
                self.global_feature_mode == "processed"
                and self.global_dense_layers is not None
            ):
                # Apply dense layers to process global features
                global_processed = self.global_dense_layers(global_feats)
                h = torch.cat([h, global_processed], dim=-1)
            elif self.global_feature_mode == "raw":
                # Concatenate raw global features without processing
                h = torch.cat([h, global_feats], dim=-1)
            # If global_feature_mode == "none", skip global features entirely

        # Dense prediction head
        out = self.dense_head(h)

        return out


# Alias for backward compatibility and consistency with naming convention
Ginny = GinnyModel
