# Feature Architecture: Molecular Atoms vs. Adduct Virtual Node

## Overview

This document explains how molecular features and adduct features are computed, their fundamental differences, and how the model handles heterogeneous node representations.

---

## Molecular Atom Features

### Feature Extraction

Molecular atom features are extracted using **graphamole's featurizer collection** from RDKit molecule objects. Each atom in a molecule is characterized by the following categorical features:

| Feature | Description | Type | Source |
|---------|-------------|------|--------|
| **AtomType** | Chemical element (C, N, O, H, etc.) | Categorical | RDKit atom.GetSymbol() |
| **Hybridization** | Orbital hybridization (sp, sp2, sp3, etc.) | Categorical | RDKit atom.GetHybridization() |
| **CIPCode** | Stereochemistry (R/S configuration) | Categorical | RDKit atom.GetPropsAsDict() |
| **FormalCharge** | Formal charge on atom (-2, -1, 0, +1, +2, etc.) | Categorical | RDKit atom.GetFormalCharge() |
| **TotalNumHs** | Total hydrogen count (explicit + implicit) | Categorical | RDKit atom.GetTotalNumHs() |
| **TotalValence** | Total valence of atom | Categorical | RDKit atom.GetTotalValence() |
| **NumRadicalElectrons** | Number of radical electrons | Categorical | RDKit atom.GetNumRadicalElectrons() |
| **Degree** | Number of bonded neighbors | Categorical | RDKit atom.GetDegree() |
| **RingSize** | Size of smallest ring containing atom (0 if not in ring) | Categorical | Custom ring detection |

### Feature Representation

**Non-One-Hot Encoding Mode** (default):
```python
# Each feature is an integer index
# Example for a carbon atom:
x = [
    6,   # AtomType: Carbon (atomic number)
    2,   # Hybridization: sp2
    0,   # CIPCode: none
    0,   # FormalCharge: neutral
    0,   # TotalNumHs: 0 hydrogens
    3,   # TotalValence: 3
    0,   # NumRadicalElectrons: 0
    2,   # Degree: 2 neighbors
    6    # RingSize: in 6-membered ring
]
# Shape: (num_features,) = (9,) typically
```

**One-Hot Encoding Mode**:
```python
# Each feature is one-hot encoded and concatenated
# Example dimensions:
# AtomType: ~120 elements
# Hybridization: ~8 types
# ... etc ...
# Total shape: (sum_of_all_vocab_sizes,) ≈ 200-300 dimensions
```

### Semantic Meaning

These features describe **local atomic properties**:
- Chemical identity (element type)
- Electronic structure (hybridization, valence)
- Structural context (degree, ring membership)
- Stereochemical properties

All features are **intrinsic to individual atoms** and their immediate bonding environment.

---

## Adduct Virtual Node Features

### Feature Extraction

Adduct features are extracted from adduct strings (e.g., "(+H)", "(+Na)", "(-2H)") using the **AdductFeatureExtractor** class. These features describe the **ionization state of the entire molecule**.

### Feature Calculation Pipeline

```python
from GraphXSection_torch.adduct_features import AdductFeatureExtractor

extractor = AdductFeatureExtractor(feature_dim=16)
features = extractor.extract_features("(+Na)")
```

**Step 1: Parse Adduct String**
```python
# Input: "(+Na)"
parsed = {
    'charge': +1,           # Net charge
    'ions': ['Na'],         # Ion types present
    'mass_delta': 22.990    # Mass change in Daltons
}
```

**Step 2: Construct Feature Vector**

| Component | Dimension | Description | Example (for +Na) |
|-----------|-----------|-------------|-------------------|
| **Charge Magnitude** | 1 | \|charge\| / 3.0 (normalized) | 0.33 |
| **Charge Sign** | 2 | One-hot [positive, negative] | [1, 0] |
| **Ion Type** | 9 | Multi-hot encoding | [0, 1, 0, 0, 0, 0, 0, 0, 0] |
| **Mass Delta** | 1 | mass_delta / 50.0 (normalized) | 0.46 |
| **Padding** | 3 | Zeros to reach feature_dim=16 | [0, 0, 0] |

**Ion Type Encoding**:
```python
ION_TYPES = [
    'H',      # Proton (index 0)
    'Na',     # Sodium (index 1)
    'K',      # Potassium (index 2)
    'NH4',    # Ammonium (index 3)
    'Li',     # Lithium (index 4)
    'Metal',  # Other metals (Mg, Ca, Fe, etc.) (index 5)
    'HCOO',   # Formate (index 6)
    'Halide', # Cl, Br (index 7)
    'Other',  # Unknown (index 8)
]
```

**Complete Example**:
```python
# "(+Na)" -> [0.33, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.46, 0, 0, 0, 0]
#            [mag, pos, neg, H, Na, K, NH4, Li, Met, HCOO, Hal, Oth, mass, pad...]

# "(+2H)" -> [0.67, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.04, 0, 0, 0]
#            [mag, pos, neg, H, Na, K, NH4, Li, Met, HCOO, Hal, Oth, mass, pad...]

# "(-H)" -> [0.33, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -0.02, 0, 0, 0]
#           [mag, pos, neg, H, Na, K, NH4, Li, Met, HCOO, Hal, Oth, mass, pad...]
```

### Semantic Meaning

These features describe **global molecular properties**:
- Overall charge state
- Type of ionizing agent
- Mass-to-charge ratio impact
- Adduction chemistry

All features are **extrinsic to the molecular structure** - they represent how the molecule interacts with the mass spectrometry ionization process.

---

## Critical Difference: Heterogeneous Features

### The Mismatch Problem

| Aspect | Atom Features | Adduct Features |
|--------|--------------|-----------------|
| **Semantic Domain** | Atomic/local chemistry | Molecular/global ionization |
| **Information Scale** | Individual atom properties | Whole-molecule properties |
| **Feature Types** | Element, bonds, geometry | Charge, ion type, mass |
| **Example** | "This is a sp2 carbon atom with 2 neighbors" | "This molecule has +1 charge from sodium" |
| **Typical Values** | Small integers (0-120) | Normalized floats (0.0-1.0) |
| **Vocabulary** | Chemical elements, hybridizations | Ion types, charges |

### Why This Is Problematic for GNNs

**Traditional GNNs assume homogeneous node features**:

1. **Message Passing**: Edge networks compute messages as:
   ```python
   message = EdgeNetwork(node_i_features, node_j_features, edge_features)
   ```
   When `node_i` is an atom (features = [element, hybridization, ...]) and `node_j` is the adduct virtual node (features = [charge, ion_type, mass, ...]), the EdgeNetwork receives **semantically incompatible inputs**.

2. **Attention Mechanisms**: Attention computes similarity/relevance:
   ```python
   attention_weight = Attention(query=vnode_features, key=atom_features)
   ```
   What does "similarity" mean between "sodium adduct with +1 charge" and "sp3 carbon atom"? The dot-product in attention space is **semantically undefined**.

3. **Learned Embeddings**: The model projects both feature types into a shared embedding space:
   ```python
   atom_embedding = AtomEmbedding(atom_features)  # [element, hybridization, ...]
   vnode_embedding = VNodeProjection(adduct_features)  # [charge, ion_type, ...]
   ```
   There's no inherent reason these should occupy the same geometric manifold.

---

## How the Model Handles This

### Current Implementation

The model uses a **forced homogenization** approach:

#### 1. Dimension Alignment (Graphamole)

```python
# In graphamole/graphamole/data.py
if add_virtual_node:
    # Pad or truncate adduct features to match atom feature length
    if len(virtual_node_features) < len(atom_features_list[0]):
        virtual_node_features = np.pad(
            virtual_node_features,
            (0, target_len - len(virtual_node_features))
        )
```

**Problem**: Padding with zeros is semantically meaningless and adds noise.

#### 2. Learned Projection (Sigma Model)

```python
# In GraphXSection_torch/models/Sigma.py
if self.use_adduct_virtual_node:
    self.virtual_node_projection = nn.Linear(
        self.adduct_feature_dim,  # 16
        self.ecc_channels          # 16
    )
```

**What this does**:
- Maps adduct features → shared embedding space
- Allows virtual node to participate in message passing
- Lets attention mechanism operate on projected features

**Limitations**:
- No guarantee the projection is semantically meaningful
- The model must learn that adduct features are "different"
- Inefficient use of model capacity

#### 3. Message Passing with Attention

```python
# Standard ECC message passing
for ecc_layer in self.ecc_layers:
    h = ecc_layer(h, edge_index, edge_emb)  # All nodes updated
    
    # Virtual node attention
    if self.use_adduct_virtual_node:
        h = virtual_node_attention(h, vnode_indices, batch)
```

**What happens**:
1. Virtual node receives messages from all atoms (standard message passing)
2. Virtual node sends messages back to all atoms (standard message passing)
3. Additional attention layer re-weights virtual node's representation based on atom features

**Issue**: Steps 1-2 operate on heterogeneous features despite the projection.

#### 4. Pooling (Excludes Virtual Node)

```python
# Pool only real atoms (exclude virtual nodes)
if self.use_adduct_virtual_node and virtual_node_indices is not None:
    real_atom_mask = torch.ones(h.size(0), dtype=torch.bool)
    real_atom_mask[virtual_node_indices] = False
    h_pooled = self.pool(h[real_atom_mask], batch[real_atom_mask])
```

**Why this is important**: The virtual node representation is NOT included in the final pooled graph representation. It only influences atom representations through message passing and attention.

---

## Architecture Flow Diagram

```
Input Graph
│
├─> Molecular Atoms                    ├─> Adduct Virtual Node
│   Features: [element, hyb, ...]      │   Features: [charge, ion, mass, ...]
│   Source: RDKit featurizers          │   Source: AdductFeatureExtractor
│   Dim: 9 categorical                 │   Dim: 16 float
│                                      │
│   ↓ Embedding Layer                  │   ↓ Linear Projection
│   [9] → [16] (ecc_channels)         │   [16] → [16] (ecc_channels)
│                                      │
└───┴──────────────────────────────────┘
             │
             ↓
    Combined Node Matrix [N+1, 16]
    (N atoms + 1 virtual node)
             │
             ↓
    ┌──────────────────────┐
    │  ECC Layer 1         │
    │  - Message passing   │ ← Heterogeneous features mixed here
    │  - Edge conditioning │
    └──────────────────────┘
             │
             ↓
    ┌──────────────────────┐
    │  Virtual Node        │
    │  Attention Layer 1   │ ← Attempts to learn meaningful weights
    │  - Multi-head attn   │
    └──────────────────────┘
             │
             ↓
    ┌──────────────────────┐
    │  ECC Layer 2         │
    └──────────────────────┘
             │
             ↓
    ┌──────────────────────┐
    │  Virtual Node        │
    │  Attention Layer 2   │
    └──────────────────────┘
             │
             ↓
         (repeat...)
             │
             ↓
    ┌──────────────────────┐
    │  Pooling             │
    │  (atoms only, no     │ ← Virtual node excluded
    │   virtual node)      │
    └──────────────────────┘
             │
             ↓
    Graph Embedding [batch, 16]
             │
             ↓
    ┌──────────────────────┐
    │  Concatenate         │
    │  + Global Features   │ ← Original adduct OHE also added
    └──────────────────────┘
             │
             ↓
    Dense Prediction Head
             │
             ↓
    CCS Prediction
```

---

## Theoretical Concerns

### 1. Semantic Inconsistency

The model treats "carbon atom with sp3 hybridization" and "sodium adduct with +1 charge" as interchangeable nodes in the graph, despite them representing fundamentally different types of information.

**Analogy**: It's like having a social network where some nodes are people (with features: age, occupation, location) and other nodes are events (with features: date, attendance, budget). The "friendship" edges don't make semantic sense between people and events.

### 2. Information Redundancy

The adduct information is present **twice**:
1. As virtual node features (processed through GNN)
2. As global features (concatenated after pooling)

This redundancy might help or hurt depending on whether the GNN can extract useful information from the virtual node architecture.

### 3. Attention Interpretability

While attention weights can show "which atoms the model focuses on for adduction," the interpretation is unclear because:
- The query (virtual node) and keys (atoms) are from different feature spaces
- The learned projection is a black box
- There's no chemical prior encoded

---

## Alternative Architectures

### Option 1: Pure Global Features (Baseline)

**Remove virtual node entirely**, use only global features:

```python
# After pooling atoms
h_pooled = pool(atom_embeddings, batch)

# Concatenate adduct features directly
h_combined = concat([h_pooled, adduct_ohe], dim=-1)

# Predict
y = dense_head(h_combined)
```

**Pros**: Simple, no semantic mismatch, proven effective  
**Cons**: No attention mechanism, adduct info only at end

### Option 2: Separate Processing Paths

**Process atoms and adduct independently, combine at prediction**:

```python
# Atom path
h_atoms = ECC_layers(atom_embeddings)
h_pooled = pool(h_atoms, batch)

# Adduct path (separate network)
h_adduct = AdductMLP(adduct_features)

# Attention mechanism
attention_weights = Attention(query=h_adduct, keys=h_atoms)
h_attended = weighted_sum(h_atoms, attention_weights)

# Combine
h_combined = concat([h_pooled, h_attended, h_adduct], dim=-1)
y = dense_head(h_combined)
```

**Pros**: No heterogeneous mixing, interpretable attention  
**Cons**: More complex architecture, harder to train

### Option 3: Make Virtual Node More Atom-Like

**Transform adduct features to look like atomic features**:

```python
# Example: Create a "pseudo-atom" for the adduct
vnode_features = [
    adduct_to_pseudo_element(adduct),  # Map Na+ → pseudo-element
    0,  # No hybridization
    formal_charge_from_adduct(adduct),
    # ... other atomic-like features
]
```

**Pros**: Semantic consistency improved  
**Cons**: Requires domain expertise, may lose information

---

## Recommendations

### For Training/Evaluation

1. **Baseline**: Train model **without virtual node** (use only global features)
2. **Compare**: Train model **with virtual node** (current implementation)
3. **Analyze**:
   - Does virtual node improve performance?
   - Are attention weights chemically meaningful?
   - Is the added complexity justified?

### For Understanding Results

If virtual node **improves** performance:
- The GNN is learning useful cross-modal interactions despite heterogeneous features
- Attention provides implicit feature engineering
- The projection layer successfully bridges semantic gap

If virtual node **degrades** performance:
- Heterogeneous features confuse the model
- Added parameters hurt generalization
- Simple concatenation (baseline) is sufficient

### For Future Work

Consider hybrid approaches:
- Use virtual node only for attention/interpretability
- Keep global features for prediction
- Experiment with different projection architectures (e.g., cross-attention instead of linear projection)

---

## Summary

| Component | Feature Type | Dimensions | Semantic Meaning | Processing |
|-----------|-------------|------------|------------------|------------|
| **Molecular Atoms** | Categorical (element, bonds, etc.) | 9 features | Local atomic properties | RDKit → Embedding |
| **Virtual Node** | Continuous (charge, ion, mass) | 16 features | Global ionization state | AdductExtractor → Projection |
| **Problem** | Heterogeneous features mixed in GNN | - | Semantic inconsistency | Forced homogenization |
| **Solution** | Linear projection + attention | - | Learn meaningful mapping | May or may not work |

**Key Takeaway**: The current architecture **forces heterogeneous features into a homogeneous framework**. This is theoretically questionable but may be practically effective. Empirical evaluation is essential to determine if the virtual node approach provides value over simpler baselines.
