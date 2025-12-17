import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from typing import Dict, Any, Optional, List, Union
from graphamole.graphamole.layers.ECCConv import ECCConv


class SigmaModel(nn.Module):
    """
    PyTorch Geometric implementation of the SigmaCCS model architecture.

    Based on the SigmaCCS paper model that uses ECCConv layers followed by
    dense layers for molecular property prediction.
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

        # Global features handling
        self.global_raw_dim = int(global_feature_sizes) if global_feature_sizes else 0
        self.global_emb_dim = int(
            m.get("global_emb_dim", 16)
        )  # Embedding dimension for descriptors
        self.one_hot_encode_feats = config.get("one_hot_encode_feats", False)

        # Model hyperparameters (matching SigmaCCS defaults)
        self.num_ecc_layers = m.get("num_ecc_layers", 3)
        self.ecc_channels = m.get("ecc_channels", 16)
        self.kernel_network = m.get("kernel_network", [64, 64, 64, 64])
        self.activation = m.get("activation", "relu")
        self.readout = m.get("readout", "sum")

        # Dense head parameters
        self.dense_units = m.get("dense_units", 384)
        self.num_dense_layers = m.get("num_dense_layers", 8)
        self.dropout = m.get("dropout", 0.0)
        self.final_activation = m.get("final_activation", "relu")

        # Log all extracted hyperparameters for verification
        import logging

        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("Sigma Model Initialization - Hyperparameters:")
        logger.info(f"  Architecture:")
        logger.info(f"    - ECC layers: {self.num_ecc_layers}")
        logger.info(f"    - ECC channels: {self.ecc_channels}")
        logger.info(f"    - Kernel network: {self.kernel_network}")
        logger.info(f"    - Dense layers: {self.num_dense_layers}")
        logger.info(f"    - Dense units: {self.dense_units}")
        logger.info(f"  Regularization:")
        logger.info(f"    - Dropout: {self.dropout}")
        logger.info(f"  Pooling:")
        logger.info(f"    - Readout method: {self.readout}")
        logger.info(f"  Activations:")
        logger.info(f"    - Activation: {self.activation}")
        logger.info(f"    - Final activation: {self.final_activation}")
        logger.info(f"  Feature encoding:")
        logger.info(f"    - One-hot encode: {self.one_hot_encode_feats}")
        logger.info(f"    - Node dim: {self.node_dim}")
        logger.info(f"    - Edge dim: {self.edge_dim}")
        logger.info(f"    - Global raw dim: {self.global_raw_dim}")
        logger.info(f"    - Global emb dim: {self.global_emb_dim}")
        logger.info("=" * 80)

        # Handle embeddings vs projections for different encoding types
        if not self.one_hot_encode_feats:
            # Use embeddings for categorical features
            if isinstance(node_feature_sizes, list):
                self.node_embeddings = nn.ModuleList(
                    [
                        nn.Embedding(vocab_size, self.ecc_channels)
                        for vocab_size in node_feature_sizes
                    ]
                )
            else:
                self.node_embeddings = nn.Embedding(
                    node_feature_sizes, self.ecc_channels
                )

            if isinstance(edge_feature_sizes, list):
                self.edge_embeddings = nn.ModuleList(
                    [
                        nn.Embedding(vocab_size, self.ecc_channels)
                        for vocab_size in edge_feature_sizes
                    ]
                )
            else:
                self.edge_embeddings = nn.Embedding(
                    edge_feature_sizes, self.ecc_channels
                )
        else:
            # Use projections for one-hot encoded features (created dynamically)
            self.node_projection = None
            self.edge_projection = None

        # ECC Convolution layers
        self.ecc_layers = nn.ModuleList()

        # First layer (from input features to hidden)
        first_layer = ECCConv(
            in_channels=self.ecc_channels,
            out_channels=self.ecc_channels,
            edge_dim=self.ecc_channels,
            kernel_network=self.kernel_network,
            activation=self.activation,
            use_bias=True,
            root_weight=True,
        )
        self.ecc_layers.append(first_layer)

        # Subsequent layers
        for _ in range(self.num_ecc_layers - 1):
            layer = ECCConv(
                in_channels=self.ecc_channels,
                out_channels=self.ecc_channels,
                edge_dim=self.ecc_channels,
                kernel_network=self.kernel_network,
                activation=self.activation,
                use_bias=True,
                root_weight=True,
            )
            self.ecc_layers.append(layer)

        # Pooling
        if self.readout == "sum":
            self.pool = global_add_pool
        elif self.readout == "mean":
            self.pool = global_mean_pool
        elif self.readout == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_add_pool

        # Global feature encoder (for adduct descriptors)
        if self.global_raw_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.global_raw_dim, self.global_emb_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.global_emb_dim * 2, self.global_emb_dim),
            )
            logger.info(
                f"Global encoder MLP: {self.global_raw_dim} -> {self.global_emb_dim * 2} -> {self.global_emb_dim}"
            )
        else:
            self.global_encoder = None

        dense_layers = []
        # Input: pooled graph embedding + encoded global features
        input_dim = self.ecc_channels + (
            self.global_emb_dim if self.global_raw_dim > 0 else 0
        )

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

        # Final output layer
        dense_layers.append(nn.Linear(input_dim, 1))
        if self.final_activation == "relu":
            dense_layers.append(nn.ReLU())

        self.dense_head = nn.Sequential(*dense_layers)

        # Initialize weights to match SigmaCCS
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights similar to TensorFlow/Spektral defaults"""
        if isinstance(module, nn.Linear):
            # Use truncated normal with std=0.005 to match typical TF initialization
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
                # Multiple categorical features
                emb = torch.zeros(x.size(0), self.ecc_channels, device=x.device)
                for i, embedding in enumerate(self.node_embeddings):
                    emb = emb + embedding(x[:, i].long())
                return emb
            else:
                # Single categorical feature
                return self.node_embeddings(x.long())
        else:
            # One-hot encoded features - create projection dynamically
            if self.node_projection is None:
                self.node_projection = nn.Linear(x.size(1), self.ecc_channels).to(
                    x.device
                )
                nn.init.trunc_normal_(self.node_projection.weight, std=0.005)
                nn.init.zeros_(self.node_projection.bias)
            return self.node_projection(x.float())

    def _compute_edge_embedding(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute edge embeddings from input features"""
        if not self.one_hot_encode_feats:
            # Categorical embedding
            if isinstance(self.edge_embeddings, nn.ModuleList):
                # Multiple categorical features
                emb = torch.zeros(
                    edge_attr.size(0), self.ecc_channels, device=edge_attr.device
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
                self.edge_projection = nn.Linear(
                    edge_attr.size(1), self.ecc_channels
                ).to(edge_attr.device)
                nn.init.trunc_normal_(self.edge_projection.weight, std=0.005)
                nn.init.zeros_(self.edge_projection.bias)
            return self.edge_projection(edge_attr.float())

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Sigma model.

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

        # Apply ECC convolution layers
        h = x_emb
        for ecc_layer in self.ecc_layers:
            h = ecc_layer(h, edge_index, edge_emb)

        # Global pooling
        h_pooled = self.pool(h, batch_idx)

        # Encode and concatenate global features (adduct descriptors) if available
        if global_feats is not None and self.global_encoder is not None:
            # Pass descriptors through embedding MLP
            adduct_emb = self.global_encoder(global_feats.float())
            h_pooled = torch.cat([h_pooled, adduct_emb], dim=-1)
        elif global_feats is not None:
            # Fallback: direct concatenation (legacy mode)
            h_pooled = torch.cat([h_pooled, global_feats], dim=-1)

        # Dense prediction head
        out = self.dense_head(h_pooled)

        return out
