import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP
from GraphXSection_torch.models.Pyggy import PredHead


class AttentiveFPWrapper(nn.Module):
    def __init__(
        self, config, node_feature_sizes, edge_feature_sizes, global_feature_sizes
    ):
        super().__init__()
        self.config = config
        self.m = config["model"]
        self.one_hot_encode_feats = config.get("one_hot_encode_feats", False)

        # Handle both list and int inputs for feature sizes
        if isinstance(node_feature_sizes, list):
            self.node_feature_sizes = tuple(node_feature_sizes)
        else:
            self.node_feature_sizes = (node_feature_sizes,)

        if isinstance(edge_feature_sizes, list):
            self.edge_feature_sizes = tuple(edge_feature_sizes)
        else:
            self.edge_feature_sizes = (edge_feature_sizes,)

        # If you truly have a flat numeric global feature vector of width G, pass G (int) here:
        self.global_feature_dim = (
            int(global_feature_sizes)
            if not isinstance(global_feature_sizes, int)
            else global_feature_sizes
        )

        self.hidden_dim = self.m.get("hidden_dim", 64)
        self.num_layers = self.m.get("num_layers", 3)
        self.num_timesteps = self.m.get("num_timesteps", 2)
        self.dense_units = self.m.get("dense_units", 128)
        self.num_dense_layers = self.m.get("num_dense_layers", 2)
        self.dropout = self.m.get("dropout", 0.0)
        self.final_activation = self.m.get("final_activation", "relu")

        # --- Handle embeddings vs projections based on encoding type ---
        if not self.one_hot_encode_feats:
            # Categorical features: use embedding tables
            self.node_embeddings = nn.ModuleList(
                [
                    nn.Embedding(vocab_size, self.hidden_dim)
                    for vocab_size in self.node_feature_sizes
                ]
            )
            self.edge_embeddings = nn.ModuleList(
                [
                    nn.Embedding(vocab_size, self.hidden_dim)
                    for vocab_size in self.edge_feature_sizes
                ]
            )
            self.node_projection = None
            self.edge_projection = None
        else:
            # One-hot encoding: use dynamic projection layers
            self.node_embeddings = nn.ModuleList()
            self.edge_embeddings = nn.ModuleList()
            # For one-hot encoding, we'll create projection layers dynamically on first forward pass
            self.node_projection = None
            self.edge_projection = None

        # AttentiveFP expects in_channels=int and edge_dim=int (feature widths).
        # For one-hot encoding, edge features will be projected to hidden_dim
        # For categorical encoding, edge features are embedded to hidden_dim
        edge_dim = None
        if self.one_hot_encode_feats:
            # Edge dimension will be determined dynamically, but AttentiveFP needs hidden_dim
            edge_dim = self.hidden_dim if len(self.edge_feature_sizes) > 0 else None
        else:
            # Categorical: each edge field is embedded to hidden_dim, then summed
            edge_dim = self.hidden_dim if len(self.edge_feature_sizes) > 0 else None

        self.encoder = AttentiveFP(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            edge_dim=edge_dim,
            dropout=self.dropout,
        )

        self.pred_head = PredHead(
            gnn_units=self.hidden_dim,
            global_units=self.global_feature_dim,
            dense_units=self.dense_units,
            num_dense_layers=self.num_dense_layers,
            dropout=self.dropout,
            final_activation=self.final_activation,
        )

    @staticmethod
    def _ensure_long(x: torch.Tensor) -> torch.Tensor:
        # Embedding indices must be Long
        return x.long() if x.dtype != torch.long else x

    def _embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, num_node_fields] of integer ids per field (categorical)
           OR [num_nodes, feature_dim] of one-hot encoded features
        Returns: [num_nodes, hidden_dim]
        """
        if self.one_hot_encode_feats:
            # One-hot encoded features
            x_float = x.float()

            # Create node projection layer dynamically on first forward pass
            if self.node_projection is None:
                actual_node_dim = x_float.shape[1]
                self.node_projection = nn.Linear(actual_node_dim, self.hidden_dim)
                if x_float.is_cuda:
                    self.node_projection = self.node_projection.cuda()

            return self.node_projection(x_float)
        else:
            # Categorical features: use embeddings
            x = self._ensure_long(x)
            emb = 0
            for j in range(x.size(1)):
                emb = emb + self.node_embeddings[j](x[:, j])
            return emb

    def _embed_edges(self, edge_attr: torch.Tensor | None) -> torch.Tensor | None:
        """
        edge_attr: [num_edges, num_edge_fields] of integer ids per field (categorical)
                  OR [num_edges, feature_dim] of one-hot encoded features
                  OR None
        Returns: [num_edges, hidden_dim] or None
        """
        if edge_attr is None or edge_attr.numel() == 0:
            return None

        if self.one_hot_encode_feats:
            # One-hot encoded features
            edge_float = edge_attr.float()

            # Create edge projection layer dynamically on first forward pass
            if self.edge_projection is None:
                actual_edge_dim = edge_float.shape[1]
                self.edge_projection = nn.Linear(actual_edge_dim, self.hidden_dim)
                if edge_float.is_cuda:
                    self.edge_projection = self.edge_projection.cuda()

            return self.edge_projection(edge_float)
        else:
            # Categorical features: use embeddings
            if len(self.edge_embeddings) == 0:
                return None
            edge_attr = self._ensure_long(edge_attr)
            emb = 0
            for j in range(edge_attr.size(1)):
                emb = emb + self.edge_embeddings[j](edge_attr[:, j])
            return emb

    def forward(self, data):
        """
        Expects:
            data["graph"].x          : [N, num_node_fields] (integer ids) OR [N, feature_dim] (one-hot)
            data["graph"].edge_attr  : [E, num_edge_fields] (integer ids) OR [E, feature_dim] (one-hot) OR None
            data["graph"].edge_index : [2, E]
            data["graph"].batch      : [N]
            data.get("global_feats") : [B, G] or None
        """
        graph = data["graph"]
        global_feats = data.get("global_feats", None)

        x_emb = self._embed_nodes(graph.x)
        edge_emb = self._embed_edges(getattr(graph, "edge_attr", None))

        # Correct AttentiveFP forward: (x, edge_index, edge_attr, batch)
        h = self.encoder(x_emb, graph.edge_index, edge_emb, graph.batch)

        # Use the PredHead properly
        out = self.pred_head(h, global_feats)
        return out
