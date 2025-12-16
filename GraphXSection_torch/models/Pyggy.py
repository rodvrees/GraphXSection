import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    JumpingKnowledge,
    AttentionalAggregation,
    MedianAggregation,
    MLPAggregation,
)

"""PyG GINEConv-based Pyggy model approximating the TF/Spektral SiGMA variant."""

logger = logging.getLogger(__name__)


ACTIVATION_FUNCTIONS = {
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "identity": nn.Identity(),
    "leaky_relu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


class Embedder(nn.Module):
    """Simple embedding layer for categorical node or edge features."""

    def __init__(self, num_embeddings, embedding_dim: int) -> None:
        super().__init__()
        # Expect num_embeddings to be an iterable of vocab sizes for each categorical feature.
        # Use a list comprehension to ensure ModuleList is populated correctly.
        self.embedding = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim) for vocab_size in num_embeddings]
        )

    def __getitem__(self, idx: int) -> nn.Embedding:
        # Allow indexing like embedder[i] to access individual embedding modules.
        return self.embedding[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support two use-cases:
        # - 1D tensor of indices -> apply the first embedding
        # - 2D tensor (N, F) -> return list of embeddings for each feature column
        if x.dim() == 1:
            return self.embedding[0](x.long())
        elif x.dim() == 2:
            return [emb(x[:, i].long()) for i, emb in enumerate(self.embedding)]
        else:
            raise ValueError("Unsupported input shape for Embedder.forward")


class GINEMLP(nn.Module):
    def __init__(self, in_dim, out_dim, apply_self_projection=True, normalization=None):
        super().__init__()
        self.apply_self_projection = apply_self_projection
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin_skip = nn.Linear(in_dim, out_dim) if apply_self_projection else None

        if normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(out_dim)
        elif normalization == "layer_norm":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        out = self.lin1(x)
        if self.lin_skip is not None:
            out = out + self.lin_skip(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.relu(out)
        out = self.lin2(out)
        return out


class GNNEncoder(nn.Module):
    """GNN Encoder using GNN layers."""

    def __init__(
        self,
        gnn_units: int = 64,
        n_layers: int = 3,
        hidden_activation: str = "relu",
        final_activation: str = "relu",
        use_batchnorm: bool = True,
        use_batchnorm_gin_mlp: bool = False,
        dropout: float = 0.0,
        jumping_knowledge: Optional[str] = None,
        residual: bool = False,
        apply_self_projection: bool = True,
        train_eps: bool = False,
    ) -> None:
        """
        GNN Encoder using GNN layers.
        """
        super().__init__()

        self.residual = residual
        self.apply_self_projection = apply_self_projection
        self.train_eps = train_eps

        self.conv_layers = nn.ModuleList()
        self.post_conv_layers = nn.ModuleList()
        self.resample_layers = nn.ModuleList()

        for layer_index in range(n_layers):
            is_final_layer = layer_index == n_layers - 1

            conv_layer = GINEConv(
                GINEMLP(
                    in_dim=gnn_units,
                    out_dim=gnn_units,
                    apply_self_projection=self.apply_self_projection,
                    normalization=(
                        use_batchnorm_gin_mlp if use_batchnorm_gin_mlp else None
                    ),
                ),
                eps=0.0,
                train_eps=self.train_eps,
            )
            post_conv_layer = nn.Sequential(
                nn.BatchNorm1d(gnn_units) if use_batchnorm else nn.Identity(),
                (
                    ACTIVATION_FUNCTIONS[hidden_activation]
                    if not is_final_layer
                    else nn.Identity()
                ),
                (
                    nn.Dropout(dropout)
                    if dropout and not is_final_layer
                    else nn.Identity()
                ),
            )
            self.conv_layers.append(conv_layer)
            self.post_conv_layers.append(post_conv_layer)

            self.resample_layers.append(nn.Identity())  # If widths vary later, alter

        self.jumping_knowledge = (
            nn.Sequential(
                JumpingKnowledge(
                    mode=jumping_knowledge,
                    channels=gnn_units,
                    num_layers=n_layers,
                ),
                (
                    nn.Linear(gnn_units * n_layers, gnn_units)
                    if jumping_knowledge == "cat"
                    else nn.Identity()
                ),
            )
            if jumping_knowledge
            else None
        )

        self.final_activation = ACTIVATION_FUNCTIONS[final_activation]

        # Log GNN encoder architecture
        logger.info(
            f"GNNEncoder initialized: {n_layers} layers, {gnn_units} units, dropout={dropout}, jumping_knowledge={jumping_knowledge}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x
            Node features.
        edge_index
            Graph connectivity.
        edge_attr
            Edge features.

        """
        outputs = []
        h = x
        for conv_layer, post_conv_layer, resample in zip(
            self.conv_layers, self.post_conv_layers, self.resample_layers
        ):
            h_in = h
            h = conv_layer(h, edge_index, edge_attr)
            h = post_conv_layer(h)
            if self.residual:
                h = h + resample(h_in)
            outputs.append(h)

        outputs = (
            self.jumping_knowledge(outputs) if self.jumping_knowledge else outputs[-1]
        )

        return self.final_activation(outputs)


class PredHead(nn.Module):
    def __init__(
        self,
        gnn_units: int = 64,
        global_units: int = 0,
        dense_units: int = 128,
        num_dense_layers: int = 2,
        dropout: float = 0.0,
        final_activation: str = "relu",
    ) -> None:
        """
        Dense prediction head.
        """
        super().__init__()

        layers = []
        cur_dim = gnn_units + global_units
        for i in range(num_dense_layers):
            layers.append(nn.Linear(cur_dim, dense_units))
            layers.append(ACTIVATION_FUNCTIONS.get(final_activation, nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur_dim = dense_units
        layers.append(nn.Linear(cur_dim, 1))  # Final output layer

        self.network = nn.Sequential(*layers)

        # Log prediction head architecture
        logger.info(
            f"PredHead initialized: {num_dense_layers} layers, {dense_units} units, input_dim={gnn_units + global_units}, dropout={dropout}"
        )

    def forward(
        self, x: torch.Tensor, global_feats: torch.Tensor | None
    ) -> torch.Tensor:
        if global_feats is not None:
            x = torch.cat([x, global_feats], dim=-1)
        return self.network(x)


class Pyggy(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        node_feature_sizes: int | list[int],
        edge_feature_sizes: int | list[int],
        global_feature_sizes: int = 0,
    ) -> None:
        """
        Pyggy model: Embedding --> GNN --> Pooling --> Dense head"""

        super().__init__()
        ohe = config.get("one_hot_encode_feats", False)
        m = config["model"]
        self.config = config

        # Handle both list and int inputs for feature sizes
        if isinstance(node_feature_sizes, list):
            self.input_node_dim = node_feature_sizes
            # For OHE, use sum of vocab sizes (each TokenFeaturizer contributes vocab_size, ContinuousFeaturizer contributes 1)
            # For categorical, use the number of features
            self.total_node_features = (
                sum(node_feature_sizes) if ohe else len(node_feature_sizes)
            )
        else:
            self.input_node_dim = node_feature_sizes
            self.total_node_features = node_feature_sizes

        if isinstance(edge_feature_sizes, list):
            self.edge_dim = edge_feature_sizes
            # For OHE, use sum of vocab sizes (each TokenFeaturizer contributes vocab_size, ContinuousFeaturizer contributes 1)
            # For categorical, use the number of features
            self.total_edge_features = (
                sum(edge_feature_sizes) if ohe else len(edge_feature_sizes)
            )
        else:
            self.edge_dim = edge_feature_sizes
            self.total_edge_features = edge_feature_sizes

        self.global_feat_dim = global_feature_sizes

        # Extract model parameters
        self.n_gnn_layers = int(m.get("num_layers", m.get("num_GNN_layers", 1)))
        self.gnn_units = int(m.get("hidden_dim", m.get("gnn_units", 64)))
        self.n_dense_layers = int(m.get("num_dense_layers", 2))
        self.dense_units = int(m.get("dense_units", 128))
        self.dropout = float(m.get("dropout", 0.0) or 0.0)
        self.readout = m.get("readout", "sum")
        self.activation = m.get("activation", "relu")
        self.final_activation = m.get("final_activation", "relu")
        self.use_batchnorm = bool(m.get("use_batchnorm", True))
        self.use_batchnorm_gin_mlp = bool(m.get("use_batchnorm_gin_mlp", False))
        self.hidden_dim_gin_mlp = m.get("hidden_dim_gin_mlp", self.gnn_units)
        self.jumping_knowledge = m.get("jumping_knowledge", None)
        self.one_hot_encode_feats = ohe
        self.residual = bool(m.get("residual", False))
        self.apply_self_projection = bool(m.get("apply_self_projection", False))
        self.train_eps = bool(m.get("train_eps", False))

        # Log all extracted hyperparameters for verification
        logger.info("=" * 80)
        logger.info("Pyggy Model Initialization - Hyperparameters:")
        logger.info(f"  Architecture:")
        logger.info(f"    - GNN layers: {self.n_gnn_layers}")
        logger.info(f"    - GNN units (hidden_dim): {self.gnn_units}")
        logger.info(f"    - Dense layers: {self.n_dense_layers}")
        logger.info(f"    - Dense units: {self.dense_units}")
        logger.info(f"  Regularization:")
        logger.info(f"    - Dropout: {self.dropout}")
        logger.info(f"  Pooling:")
        logger.info(f"    - Readout method: {self.readout}")
        if "pool_config" in m:
            logger.info(f"    - Pool config: {m['pool_config']}")
        logger.info(f"  Activations:")
        logger.info(f"    - Hidden activation: {self.activation}")
        logger.info(f"    - Final activation: {self.final_activation}")
        logger.info(f"  Normalization:")
        logger.info(f"    - Use BatchNorm: {self.use_batchnorm}")
        logger.info(f"    - Use BatchNorm in GIN MLP: {self.use_batchnorm_gin_mlp}")
        logger.info(f"  GIN-specific:")
        logger.info(f"    - Apply self projection: {self.apply_self_projection}")
        logger.info(f"    - Train eps: {self.train_eps}")
        logger.info(f"    - Jumping knowledge: {self.jumping_knowledge}")
        logger.info(f"    - Residual connections: {self.residual}")
        logger.info(f"  Feature encoding:")
        logger.info(f"    - One-hot encode: {self.one_hot_encode_feats}")
        logger.info(f"    - Node feature sizes: {self.input_node_dim}")
        logger.info(f"    - Edge feature sizes: {self.edge_dim}")
        logger.info(f"    - Global feature size: {self.global_feat_dim}")
        logger.info("=" * 80)

        # Embedders, if not one-hot encoding
        if not self.one_hot_encode_feats:
            self.node_embeddings = Embedder(
                num_embeddings=self.input_node_dim, embedding_dim=self.gnn_units
            )
            self.edge_embeddings = Embedder(
                num_embeddings=self.edge_dim, embedding_dim=self.gnn_units
            )

        # If one-hot encoding, set embedding lists to empty ModuleLists and use a linear layer to project
        else:
            self.node_embeddings = nn.ModuleList()
            self.edge_embeddings = nn.ModuleList()
            # For one-hot encoding, we'll create projection layers dynamically on first forward pass
            # to handle the actual tensor dimensions correctly
            self.node_projection = None
            self.edge_projection = None

        # GNN backbone
        self.GNNEncoder = GNNEncoder(
            gnn_units=self.gnn_units,
            n_layers=self.n_gnn_layers,
            hidden_activation=self.activation,
            final_activation=self.final_activation,
            use_batchnorm=self.use_batchnorm,
            use_batchnorm_gin_mlp=self.use_batchnorm_gin_mlp,
            dropout=self.dropout,
            jumping_knowledge=self.jumping_knowledge,
            residual=self.residual,
            apply_self_projection=self.apply_self_projection,
            train_eps=self.train_eps,
        )

        # Pooling function or aggregation module
        self.pool_fn, pool_module = self._get_pool_fn(pool_method=self.readout)
        # Register pooling module if it exists
        if pool_module is not None:
            self.pool_module = pool_module
        else:
            self.pool_module = None

        # Dense head
        self.PredHead = PredHead(
            gnn_units=self.gnn_units,
            global_units=self.global_feat_dim,
            dense_units=self.dense_units,
            num_dense_layers=self.n_dense_layers,
            dropout=self.dropout,
            final_activation=self.activation,
        )

    def _get_pool_fn(self, pool_method: str) -> tuple[callable, Optional[nn.Module]]:
        """Get pooling function and optional aggregation module.

        Returns:
            Tuple of (pooling_function, aggregation_module)
            - For simple pooling (mean, max, sum): (function, None)
            - For aggregation operators: (None, module)
        """
        pool_config = self.config["model"].get("pool_config", {})
        logger.info(
            f"Configuring pooling method '{pool_method}' with config: {pool_config}"
        )
        if pool_method == "mean":
            return global_mean_pool, None
        elif pool_method == "max":
            return global_max_pool, None
        elif pool_method == "sum":
            return global_add_pool, None
        elif pool_method == "median":
            return None, MedianAggregation()
        elif pool_method == "attention":
            # AttentionalAggregation requires gate_nn and nn parameters
            pool_gate_n_layers = pool_config.get("gate_nn_layers", 2)
            pool_nn_layers = pool_config.get("nn_layers", 2)
            pool_gate_hidden = pool_config.get("gate_nn_hidden", self.gnn_units // 2)
            pool_nn_hidden = pool_config.get("nn_hidden", self.gnn_units // 2)
            logger.info(
                f"  Attention pooling params: gate_layers={pool_gate_n_layers}, gate_hidden={pool_gate_hidden}, nn_layers={pool_nn_layers}, nn_hidden={pool_nn_hidden}"
            )

            def build_mlp(
                in_dim: int, hidden_dim: int, out_dim: int, n_layers: int
            ) -> nn.Sequential:
                layers = []
                if n_layers <= 1:
                    layers.append(nn.Linear(in_dim, out_dim))
                else:
                    # first hidden layer
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    # intermediate hidden layers (if any)
                    for _ in range(n_layers - 2):
                        layers.append(nn.Linear(hidden_dim, hidden_dim))
                        layers.append(nn.ReLU())
                    # final projection to out_dim
                    layers.append(nn.Linear(hidden_dim, out_dim))
                return nn.Sequential(*layers)

            gate_nn = build_mlp(self.gnn_units, pool_gate_hidden, 1, pool_gate_n_layers)
            nn_module = build_mlp(
                self.gnn_units, pool_nn_hidden, self.gnn_units, pool_nn_layers
            )
            return None, AttentionalAggregation(gate_nn=gate_nn, nn=nn_module)
        elif pool_method == "mlp":
            # MLPAggregation requires in_channels and max_num_elements parameters
            pool_max_num_elements = pool_config.get("max_num_elements", 128)
            pool_nn_layers = pool_config.get("nn_layers", 2)
            pool_nn_hidden = pool_config.get("nn_hidden", self.gnn_units // 2)
            logger.info(
                f"  MLP pooling params: max_num_elements={pool_max_num_elements}, nn_layers={pool_nn_layers}, nn_hidden={pool_nn_hidden}"
            )
            return None, MLPAggregation(
                in_channels=self.gnn_units,
                out_channels=self.gnn_units,
                max_num_elements=pool_max_num_elements,
                num_layers=pool_nn_layers,
                hidden_channels=pool_nn_hidden,
            )
        else:
            raise ValueError(f"Unknown pooling method: {pool_method}")

    def _compute_node_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Sum node embeddings across all feature dimensions."""
        emb = 0
        for i in range(x.size(1)):
            emb = emb + self.node_embeddings[i](x[:, i])
        return emb

    def _compute_edge_embedding(self, edge: torch.Tensor) -> torch.Tensor:
        """Sum edge embeddings across all feature dimensions."""
        emb = 0
        for i in range(edge.size(1)):
            emb = emb + self.edge_embeddings[i](edge[:, i])
        return emb

    def forward(self, data) -> torch.Tensor:
        graph = data["graph"]
        global_feats = data.get("global_feats", None)
        needs_edge_embedding = (
            hasattr(graph, "edge_attr") and graph.edge_attr is not None
        )

        node = graph.x

        # Handle node features for both embedding and one-hot (dense) cases
        if self.one_hot_encode_feats:
            # Expect node to be a dense tensor of shape (N, input_node_dim)
            if node is None:
                raise ValueError("Expected node features for one-hot encoding mode.")
            node_tensor = node.float()

            # Create node projection layer dynamically on first forward pass
            if self.node_projection is None:
                actual_node_dim = node_tensor.shape[1]
                self.node_projection = nn.Linear(actual_node_dim, self.gnn_units).to(
                    device=node_tensor.device
                )

            x_emb = self.node_projection(node_tensor)
        else:
            # Categorical features: compute embeddings by summing per-feature embeddings
            x_emb = self._compute_node_embedding(node)

        # Handle edge features similarly
        if needs_edge_embedding:
            edge = graph.edge_attr
            if self.one_hot_encode_feats:
                edge_tensor = edge.float()

                # Create edge projection layer dynamically on first forward pass
                if self.edge_projection is None:
                    actual_edge_dim = edge_tensor.shape[1]
                    self.edge_projection = nn.Linear(
                        actual_edge_dim, self.gnn_units
                    ).to(device=edge_tensor.device)

                edge_emb = self.edge_projection(edge_tensor)
            else:
                edge_emb = self._compute_edge_embedding(edge)
        else:
            edge_emb = None

        h = self.GNNEncoder(x_emb, graph.edge_index, edge_emb)

        # Apply pooling - use either simple function or aggregation module
        if self.pool_fn is not None:
            # Simple pooling functions (mean, max, sum)
            h = self.pool_fn(h, graph.batch)
        elif self.pool_module is not None:
            # Aggregation operators (median, attention, mlp)
            h = self.pool_module(h, graph.batch)
        else:
            raise RuntimeError("No pooling method configured")

        out = self.PredHead(h, global_feats)
        return out
