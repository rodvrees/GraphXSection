# Virtual adduct node for Sigma model: suggested implementation and code changes (updated)

This note consolidates recommended changes for a **correct and maintainable** adduct virtual-node implementation under the **Sigma** architecture, based on the modules provided:

- `Sigma.py` (Sigma model with ECCConv + optional virtual-node attention)
- `VirtualNodeAttention.py` (virtual-node attention implementation)
- `data_extract.py` / `data.py` (graph construction and PyG `Data`)
- `adduct_features.py` (adduct feature extraction)
- `ECCConv.py` (edge-conditioned convolution)
- `lightning_modules.py` (Sigma instantiation and configuration plumbing)

The focus is correctness under **PyG batching**, dimensional consistency, and preserving information content in adduct features.


## 1) Target invariants (“correctness contract”)

When you add exactly **one virtual node per molecule graph**:

1. **Graph node count**
   - `data.num_nodes == num_atoms + 1`
   - `data.virtual_node_idx == num_atoms` (per-graph, local index)

2. **Node-level tensors match `num_nodes`**
   - `node_types.shape == [num_atoms + 1]` with `0=atom`, `1=vnode`
   - any node mask is length `num_atoms + 1`

3. **Batching**
   - After `Batch.from_data_list`, `graph.batch` is defined for **all nodes** (atoms + vnodes)
   - `graph.num_graphs == batch_size`
   - `torch.sum(graph.node_types == 1) == graph.num_graphs`

4. **Heterogeneous raw features are preserved**
   - atoms: typically categorical integer fields (often `torch.long`)
   - vnode: continuous vector (`torch.float32`)
   - **no padding/truncation to match atom schema** at the raw level

---

## 2) High-impact issues found in the current code

### 2.1 `AdductFeatureExtractor` destroys information content

In `adduct_features.py`, the extractor computes normalized **continuous** features (charge magnitude, mass delta), but then returns `features.astype(np.int64)` “for embedding”. This collapses most values to 0/1 and discards the continuous signal.

It also has a parsing bug: `parse_adduct()` multiplies the computed charge by an overall `charge_sign` derived from the prefix, but the regex parsing already encodes sign per component; this leads to sign inversions for strings like `(-2H)`.

**Consequence:** the virtual node receives near-constant or incorrect features, and any adduct conditioning becomes unreliable.

### 2.2 `VirtualNodeAttention` assigns atoms to graphs using `batch == i`

In `VirtualNodeAttention.py`, attention loops over virtual nodes and selects atoms using `(batch == i)`, where `i` is the loop index. This assumes the i-th virtual node always belongs to graph i, which is not guaranteed under batching and reindexing.

It also uses `torch_geometric.utils.softmax(scores, index=zeros)` in a way that is easy to segment incorrectly; since you are already operating per-graph, plain `torch.softmax(scores, dim=0)` is the clean and correct choice.

### 2.3 Edge type handling is currently brittle

`ECCConv.py` explicitly does **not** use `edge_types`, and expects `edge_attr` to fully define the dynamic filter.

In your current graph builder, you sometimes append `edge_type` as an extra feature column to `edge_attr`. This is only safe if *all* downstream feature-size plumbing (e.g., `edge_feature_sizes`) is updated accordingly. If not, you will get silent embedding mismatches or runtime errors.

**Preferred approach:** keep `edge_types` separate and incorporate them into `edge_emb` inside the model, without changing the `edge_attr` schema.

---

## 3) Recommended implementation (minimal, correct, Sigma-only)

### 3.1 Fix `AdductFeatureExtractor` (return float features; correct charge parsing)

**Goals**
- Preserve continuous information (do not cast to int).
- Fix charge sign logic.
- Keep dimensionality stable (16 by default is fine, but keep it float).

#### Suggested patch (adduct_features.py)

```diff
@@ class AdductFeatureExtractor:
     def parse_adduct(self, adduct_str: str) -> Dict[str, any]:
@@
-        # Determine overall charge sign
-        charge_sign = 1 if adduct.startswith("+") else -1
+        # Do NOT apply an overall sign multiplier here.
+        # The regex already captures per-component signs (e.g., -2H).
@@
-        return {
-            "charge": total_charge * charge_sign,
-            "ions": ions,
-            "mass_delta": mass_delta,
-        }
+        return {
+            "charge": total_charge,
+            "ions": ions,
+            "mass_delta": mass_delta,
+        }

@@ def extract_features(self, adduct_str: str) -> np.ndarray:
-        return features.astype(np.int64)  # Convert to long for embedding
+        # Keep continuous features as float32 for the adduct_encoder MLP
+        return features.astype(np.float32)
```

**Note:** If you *really* want categorical embeddings for ion type, keep the ion-type multi-hot as float and let the adduct MLP learn from it. The current Sigma implementation already uses an MLP for `adduct_encoder`, so float features are the correct match.

---

### 3.2 Data construction: set `num_nodes`, store vnode as float, and keep edge types separate

In `data_extract.py` you already store `x_vnode` as float32 (good). The two required changes are:

1. **Set `data.num_nodes` explicitly** when virtual nodes are present.
2. **Stop mutating `edge_attr` shape by appending `edge_type`**, and instead keep `edge_types` as a separate tensor.

#### Suggested patch (data_extract.py and/or data.py)

```diff
@@ def get_data(..., add_virtual_node=False, virtual_node_features=None):
     if add_virtual_node:
         # atoms are long (categorical fields)
         x_atoms = torch.tensor(np.array(atom_features_list), dtype=torch.long)

         # vnode is continuous
         x_vnode = torch.tensor(virtual_node_features, dtype=torch.float32).unsqueeze(0)

         node_types = torch.zeros(num_atoms + 1, dtype=torch.long)
         node_types[virtual_idx] = 1

@@
-        # Encode edge_type into edge_attr so ECCConv can condition on it
-        edge_type_feature = edge_types_tensor.unsqueeze(1)
-        edge_attr = torch.cat([edge_attr, edge_type_feature], dim=1)
+        # Keep chemical edge attributes unchanged; keep edge_types separate.
+        # Edge-type conditioning will be handled in Sigma.py by embedding edge_types
+        # and adding it to edge_emb.

     data = Data(
         x_atoms=x_atoms,
         x_vnode=x_vnode,
         node_types=node_types,
         edge_index=edge_index,
         edge_attr=edge_attr,
         edge_types=edge_types_tensor,
     )

     if add_virtual_node:
         data.has_virtual_node = True
         data.virtual_node_idx = virtual_idx
         data.num_real_atoms = num_atoms
+        data.num_nodes = num_atoms + 1
     else:
         data.has_virtual_node = False
+        data.num_nodes = num_atoms
```

**Fail-fast improvement:** if `add_virtual_node=True` and `virtual_node_features is None`, raise an error or create a correctly-sized zero vector using the configured `adduct_feature_dim`.

---

### 3.3 Sigma model: embed `edge_types` and add to `edge_emb` (no `edge_attr` schema change)

Because `ECCConv` expects `edge_attr`-derived dynamic filters and does not understand `edge_types` explicitly, the cleanest solution is:

- keep `_compute_edge_embedding(edge_attr)` as-is (chemical features)
- add a learned `edge_type_embedding(edge_types)` and **sum** into `edge_emb`

This preserves `edge_dim == ecc_channels` as expected by `ECCConv`.

#### Suggested patch (Sigma.py)

```diff
@@ class SigmaModel.__init__(...):
     if self.use_adduct_virtual_node:
         ...
         self.node_type_embedding = nn.Embedding(2, self.ecc_channels)
+        # Edge type: 0=bond edge, 1=adduct edge
+        self.edge_type_embedding = nn.Embedding(2, self.ecc_channels)

@@ def forward(self, batch):
     edge_emb = self._compute_edge_embedding(edge_attr)
+
+    # Add edge-type embedding if present (does not change edge_emb dimensionality)
+    if self.use_adduct_virtual_node and hasattr(graph, "edge_types") and graph.edge_types is not None:
+        edge_emb = edge_emb + self.edge_type_embedding(graph.edge_types)
```

---

### 3.4 Fix `VirtualNodeAttention`: correct per-graph atom selection and softmax

Replace the `(batch == i)` assumption with `(batch == graph_id)` where `graph_id = batch[v_idx]`.

Also replace `torch_geometric.utils.softmax` with `torch.softmax(scores, dim=0)` inside the per-graph loop.

#### Suggested patch (VirtualNodeAttention.py)

```diff
@@ def forward(...):
-        for i, v_idx in enumerate(virtual_node_idx):
-            graph_mask = (batch == i) & atom_mask
+        for i, v_idx in enumerate(virtual_node_idx):
+            graph_id = int(batch[v_idx].item())
+            graph_mask = (batch == graph_id) & atom_mask
             atom_indices = torch.where(graph_mask)[0]
             if len(atom_indices) == 0:
                 continue
@@
-            scores = softmax(scores, index=torch.zeros_like(scores[:, 0], dtype=torch.long))
+            # normalize across atoms for each head (scores: [num_atoms, heads])
+            scores = torch.softmax(scores, dim=0)
```

**Also recommended:** remove the unused `edge_index` argument and drop `MessagePassing` inheritance, since this module does not call `propagate()`.

---

## 4) Sanity checks to add (before training)

Add these assertions early in `Sigma.forward()` when virtual nodes are enabled:

```python
assert hasattr(graph, "batch"), "Expected PyG Batch with graph.batch present"
assert graph.batch.size(0) == graph.node_types.size(0)

# exactly one vnode per graph
assert int((graph.node_types == 1).sum().item()) == int(graph.num_graphs)

# edge indices in range
assert int(graph.edge_index.max().item()) < int(graph.num_nodes)

# vnode features align with num_graphs
assert graph.x_vnode.size(0) == graph.num_graphs
```

If any of these fail, fix the `Data` construction first—do not patch around it in the model.

---

## 5) Why this version is “how it should happen”

This design matches PyG’s intended batching semantics:

- the vnode is a real graph node (indexable via `edge_index`, pooled/excluded via masks)
- `graph.batch` provides the authoritative mapping from node → graph
- heterogeneity is handled at the encoder level (atom encoder vs adduct encoder), not by forcing raw feature schemas to match
- edge-type information is incorporated **without** changing `edge_attr` dimensionality, so your categorical embedding plumbing remains stable

---

## 6) Minimal ablations to verify the vnode is used

1. **No vnode (baseline):** late concatenation of global adduct features (your old model)
2. **VNode + ECC only:** disable `VirtualNodeAttention` (or bypass it)
3. **VNode + ECC + attention:** full model

If (2) ≈ (3), the attention layer may be redundant. If (3) > (2) and both > (1), your vnode pathway is functioning and beneficial.
