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
        use_adduct_virtual_node: bool = False,
        adduct_feature_dim: int = 16,
    ):
        super().__init__()

        self.config = config
        m = config["model"]
        self.use_adduct_virtual_node = use_adduct_virtual_node
        self.adduct_feature_dim = adduct_feature_dim

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
        logger.info(f"    - Global dim: {self.global_dim}")
        logger.info(f"  Virtual Node:")
        logger.info(f"    - Use virtual node: {self.use_adduct_virtual_node}")
        if self.use_adduct_virtual_node:
            logger.info(f"    - Adduct feature dim: {self.adduct_feature_dim}")
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

        # Add virtual node attention layers if enabled
        if self.use_adduct_virtual_node:
            from graphamole.layers.VirtualNodeAttention import (
                VirtualNodeAttention,
            )

            self.virtual_node_attention_layers = nn.ModuleList(
                [
                    VirtualNodeAttention(
                        channels=self.ecc_channels,
                        heads=2,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_ecc_layers)
                ]
            )

            # Separate encoder for adduct node (MLP instead of single linear layer)
            self.adduct_encoder = nn.Sequential(
                nn.Linear(self.adduct_feature_dim, self.ecc_channels * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.ecc_channels * 2, self.ecc_channels),
            )
            logger.info(
                f"Adduct encoder MLP: {self.adduct_feature_dim} -> {self.ecc_channels * 2} -> {self.ecc_channels}"
            )

            # Node-type embeddings to distinguish atoms from virtual nodes
            self.node_type_embedding = nn.Embedding(
                2, self.ecc_channels
            )  # 0=atom, 1=virtual
            logger.info("Added node-type embeddings (atom vs virtual node)")

            # Edge-type embeddings: 0=bond edge, 1=adduct edge
            self.edge_type_embedding = nn.Embedding(2, self.ecc_channels)
            logger.info("Added edge-type embeddings (bond vs adduct edge)")

            logger.info(
                f"Initialized {self.num_ecc_layers} VirtualNodeAttention layers"
            )

        # Pooling
        if self.readout == "sum":
            self.pool = global_add_pool
        elif self.readout == "mean":
            self.pool = global_mean_pool
        elif self.readout == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_add_pool

        # Dense prediction head (matching SigmaCCS architecture)
        dense_layers = []
        # When using virtual nodes, adduct info is in the graph, so no global feats needed
        if self.use_adduct_virtual_node:
            input_dim = self.ecc_channels
        else:
            input_dim = self.ecc_channels + self.global_dim

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
        """Forward pass with proper heterogeneous node handling.

        Args:
            batch: Dictionary containing:
                - "graph": PyG Data object with x_atoms, x_vnode, node_types, edge_index, edge_attr, edge_types
                - "global_feats": Optional global features (may be None if using virtual node)

        Returns:
            Predictions [batch_size, 1]
        """
        graph = batch["graph"]
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        global_feats = batch.get("global_feats", None)

        # Sanity checks when virtual nodes are enabled (only during eval to save time)
        if (
            not self.training
            and self.use_adduct_virtual_node
            and hasattr(graph, "x_vnode")
            and graph.x_vnode is not None
        ):
            assert hasattr(
                graph, "batch"
            ), "Expected PyG Batch with graph.batch present"
            assert graph.batch.size(0) == graph.node_types.size(
                0
            ), f"Batch size mismatch: batch={graph.batch.size(0)}, node_types={graph.node_types.size(0)}"

            # Exactly one vnode per graph
            num_vnodes = int((graph.node_types == 1).sum().item())
            num_graphs = int(graph.num_graphs)
            assert (
                num_vnodes == num_graphs
            ), f"Expected {num_graphs} virtual nodes (one per graph), got {num_vnodes}"

            # Edge indices in range
            max_edge_idx = int(graph.edge_index.max().item())
            num_nodes = int(graph.num_nodes)
            assert (
                max_edge_idx < num_nodes
            ), f"Edge index {max_edge_idx} out of range for {num_nodes} nodes"

            # Vnode features align with num_graphs
            assert (
                graph.x_vnode.size(0) == num_graphs
            ), f"x_vnode size mismatch: got {graph.x_vnode.size(0)}, expected {num_graphs}"

        # Step 1: Encode atoms and virtual nodes separately
        if (
            self.use_adduct_virtual_node
            and hasattr(graph, "x_vnode")
            and graph.x_vnode is not None
        ):
            # In batched mode, graph.x_atoms is already concatenated across batch
            # graph.x_vnode is [num_graphs_in_batch, vnode_feat_dim]
            # graph.node_types is [total_nodes_in_batch] where virtual nodes are marked as 1

            # Reconstruct proper batch_idx for ALL nodes (atoms + virtual nodes)
            # PyG's Batch.from_data_list() creates batch indices, but with separate x_atoms and x_vnode,
            # we need to reconstruct the full batch assignment ourselves.

            # Get node types to identify atoms vs virtual nodes
            node_types = graph.node_types  # [total_nodes]
            num_total_nodes = node_types.size(0)
            num_graphs = graph.x_vnode.size(0)

            # Strategy: PyG batches x_atoms separately, creating x_atoms_batch
            # We need to find the batch index tensor that corresponds to atoms
            # Then extend it to include virtual nodes

            # Try to get batch info from PyG's batching
            x_atoms_batch_idx = None
            if hasattr(graph, "x_atoms_batch"):
                x_atoms_batch_idx = graph.x_atoms_batch
            elif hasattr(graph, "_slice_dict") and "x_atoms" in graph._slice_dict:
                # Reconstruct from slice information
                slices = graph._slice_dict["x_atoms"]
                x_atoms_batch_idx = torch.zeros(
                    graph.x_atoms.size(0), dtype=torch.long, device=graph.x_atoms.device
                )
                for i in range(len(slices) - 1):
                    x_atoms_batch_idx[slices[i] : slices[i + 1]] = i
            else:
                # Fallback: use num_real_atoms to reconstruct
                if hasattr(graph, "num_real_atoms"):
                    num_real_atoms_list = graph.num_real_atoms
                    if torch.is_tensor(num_real_atoms_list):
                        num_real_atoms_list = num_real_atoms_list.tolist()
                    elif not isinstance(num_real_atoms_list, list):
                        num_real_atoms_list = [num_real_atoms_list]

                    x_atoms_batch_idx = torch.zeros(
                        graph.x_atoms.size(0),
                        dtype=torch.long,
                        device=graph.x_atoms.device,
                    )
                    offset = 0
                    for graph_id, num_atoms in enumerate(num_real_atoms_list):
                        x_atoms_batch_idx[offset : offset + num_atoms] = graph_id
                        offset += num_atoms
                else:
                    raise ValueError(
                        "Cannot determine batch indices for atoms: no x_atoms_batch, _slice_dict, or num_real_atoms found"
                    )

            # Now build full batch_idx by matching node_types ordering (vectorized)
            atom_mask = node_types == 0
            vnode_mask = node_types == 1

            # Create batch_idx tensor
            batch_idx = torch.zeros(
                num_total_nodes, dtype=torch.long, device=graph.x_atoms.device
            )

            # Assign atom batch indices
            batch_idx[atom_mask] = x_atoms_batch_idx

            # Assign virtual node batch indices (0, 1, 2, ..., num_graphs-1)
            batch_idx[vnode_mask] = torch.arange(
                num_graphs, dtype=torch.long, device=graph.x_atoms.device
            )

            # Separate encoders for atoms and virtual nodes
            x_atoms_emb = self._compute_node_embedding(
                graph.x_atoms
            )  # [N_atoms_total_in_batch, ecc_channels]
            x_vnode_emb = self.adduct_encoder(
                graph.x_vnode.float()
            )  # [num_graphs, ecc_channels]

            # Build h tensor by combining embeddings in the correct order (vectorized)
            # Strategy: use node_types to place embeddings correctly without in-place ops
            atom_mask = node_types == 0
            vnode_mask = node_types == 1

            # Allocate h tensor
            h = torch.zeros(
                num_total_nodes,
                self.ecc_channels,
                device=x_atoms_emb.device,
                dtype=x_atoms_emb.dtype,
            )

            # Use advanced indexing to assign embeddings (vectorized, no loops)
            h[atom_mask] = x_atoms_emb
            h[vnode_mask] = x_vnode_emb

            # Add type embeddings
            type_emb = self.node_type_embedding(node_types)
            h = h + type_emb
        else:
            # No virtual node - standard processing
            h = self._compute_node_embedding(graph.x_atoms)
            virtual_node_indices = None
            batch_idx = (
                graph.batch
                if hasattr(graph, "batch")
                else torch.zeros(
                    graph.x_atoms.size(0), dtype=torch.long, device=graph.x_atoms.device
                )
            )

        # Step 2: Encode edges
        edge_emb = self._compute_edge_embedding(edge_attr)

        # Add edge-type embedding if present (does not change edge_emb dimensionality)
        if (
            self.use_adduct_virtual_node
            and hasattr(graph, "edge_types")
            and graph.edge_types is not None
        ):
            edge_emb = edge_emb + self.edge_type_embedding(graph.edge_types)

        # Pre-compute virtual node info once (not in every layer)
        vnode_per_graph = None
        if (
            self.use_adduct_virtual_node
            and hasattr(graph, "x_vnode")
            and graph.x_vnode is not None
        ):
            node_types = graph.node_types
            vnode_mask = node_types == 1
            virtual_node_indices = torch.where(vnode_mask)[0]

            # Vectorized lookup: batch_idx[virtual_node_indices] gives graph_id for each vnode
            # Since we have exactly one vnode per graph, this is our mapping
            vnode_per_graph = virtual_node_indices

        # Step 3: Message passing with virtual node attention
        for layer_idx, ecc_layer in enumerate(self.ecc_layers):
            # Standard ECC message passing (handles both bond and adduct edges)
            h = ecc_layer(h, edge_index, edge_emb)

            # Apply virtual node attention if enabled
            if vnode_per_graph is not None:
                h = self.virtual_node_attention_layers[layer_idx](
                    h, edge_index, vnode_per_graph, batch_idx
                )

        # Step 4: Pool atoms only (exclude virtual nodes)
        if (
            self.use_adduct_virtual_node
            and hasattr(graph, "x_vnode")
            and graph.x_vnode is not None
        ):
            # Get node types and create masks
            node_types = graph.node_types
            vnode_mask = node_types == 1
            virtual_node_indices = torch.where(vnode_mask)[0]
            atom_mask = node_types == 0

            # Pool only real atoms
            atom_batch_idx = batch_idx[atom_mask]
            h_atoms = h[atom_mask]

            # Verify we have atoms for all graphs
            num_graphs = graph.x_vnode.size(0)
            unique_batches = torch.unique(atom_batch_idx)

            # If any graph is missing atoms, this would cause dimension mismatch
            # In practice, this shouldn't happen with molecular graphs
            if len(unique_batches) != num_graphs:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Batch mismatch: Expected {num_graphs} graphs, but atom_batch_idx has {len(unique_batches)} unique indices"
                )
                logger.warning(f"Unique batch indices in atoms: {unique_batches}")
                logger.warning(f"Expected: 0 to {num_graphs-1}")

            h_pooled = self.pool(h_atoms, atom_batch_idx)
        else:
            h_pooled = self.pool(h, batch_idx)

        # Step 5: Optionally concatenate global features
        # Note: When using virtual node, global_feats may be redundant
        if global_feats is not None and not self.use_adduct_virtual_node:
            h_pooled = torch.cat([h_pooled, global_feats], dim=-1)

        # Step 6: Dense prediction head
        out = self.dense_head(h_pooled)

        return out
