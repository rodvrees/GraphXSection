# Implementing adduct generalization in Sigma via physicochemical adduct descriptors

This document describes the **minimal code changes** required to replace **adduct one-hot encoding** with a **continuous physicochemical adduct descriptor vector**, while keeping the **Sigma** architecture (ECCConv → pooling → MLP) and **not using a virtual node**.

The approach is:

1. Compute an adduct descriptor vector per example (float32).
2. Pass it through a small **adduct embedding MLP**.
3. Concatenate the resulting embedding with the pooled graph embedding.
4. Predict CCS with the existing dense head.

This makes inference on **unseen adduct strings** well-posed as long as the descriptor extractor can parse them.

---

## 1) Recommended adduct descriptor feature set

### 1.1 Minimum viable descriptor (strong baseline)
These are the features most directly tied to ion mobility differences and are easy to compute reliably.

**Core features**
- **Charge state**: `z_signed` (e.g., +1, +2, −1), and/or `z_abs = |z|`
- **Polarity/sign**: `polarity` ∈ {−1, +1} (redundant with `z_signed`, but often stabilizes learning)
- **Adduct mass shift**: `delta_mass` (Da) from the adduct composition
- **Ion m/z**: `mz = (neutral_exact_mass + delta_mass) / z_abs`

**Derived scalars (helpful regularizers)**
- `mass_over_charge = (neutral_exact_mass + delta_mass) / z_abs` (identical to `mz`; keep only one)
- `charge_density_proxy = z_abs / (neutral_exact_mass + delta_mass)` (helps scale the “charge effect” across mass range)

> You can start with: `[z_signed, z_abs, delta_mass, mz]` (+ optional `charge_density_proxy`).

### 1.2 Additional adduct composition features (recommended for better OOD behavior)
These features help the model distinguish *why* an adduct differs (protonation vs metal vs anion attachment).

- **Ion-type multi-hot** (already present in your current `AdductFeatureExtractor`):
  - proton, alkali metal, ammonium, acetate/formate, halide, other
- **Component counts**:
  - `n_proton_added`, `n_proton_removed`
  - `n_metal_added` (Li/Na/K/…)
  - `n_anion_added` (Cl/Br/…)
- **Primary charge carrier identity** (low-dimensional numeric):
  - `carrier_atomic_number` (e.g., Na → 11), `carrier_mass` (Da)
  - (optional) `carrier_ionic_radius` / `carrier_polarizability` from a small lookup table (for common ions)

> These features are cheap, stable, and allow generalization to new adduct strings that share carrier families (e.g., Na/K/Li).

### 1.3 Notes on “polarizability-related” descriptors
Direct polarizability of an ion–molecule complex is not typically available from SMILES alone without quantum chemistry. Practical proxies include:
- **m/z** (already)
- **charge density proxy** (above)
- **carrier ion polarizability / ionic radius** (lookup table; coarse but useful)
- If you already compute global molecular descriptors elsewhere, you can include a molecule-side proxy like **molar refractivity (MR)**, but that is *molecule* information rather than *adduct* information.

---

## 2) Extraction strategy and code changes

### 2.1 Fix and refactor `adduct_features.py` into a descriptor extractor (float32)

Your current `AdductFeatureExtractor.extract_features()`:
- pads/truncates to a configured dim (fine),
- **casts to int64** (bad for continuous features),
- and `parse_adduct()` applies an overall sign multiplier that can invert charges for strings like `(-2H)`.

**Change summary**
- Keep outputs as `float32`.
- Fix sign logic in parsing (do not apply an “overall sign multiplier” on top of signed components).
- Add an API that can compute **mz** given a molecule mass.

#### Patch sketch (adduct_features.py)

```diff
@@ class AdductFeatureExtractor:
-        # Determine overall charge sign
-        charge_sign = 1 if adduct.startswith("+") else -1
+        # Do NOT apply an overall sign multiplier.
+        # Signs are already captured per component by the regex.

@@
-        return {
-            "charge": total_charge * charge_sign,
-            "ions": ions,
-            "mass_delta": mass_delta,
-        }
+        return {"charge": total_charge, "ions": ions, "mass_delta": mass_delta}

@@ def extract_features(self, adduct_str: str) -> np.ndarray:
-        return features.astype(np.int64)  # Convert to long for embedding
+        return features.astype(np.float32)
```

#### Add a new method for mz-aware descriptors

```python
def extract_descriptor(self, adduct_str: str, neutral_exact_mass: float) -> np.ndarray:
    parsed = self.parse_adduct(adduct_str)
    z = int(parsed["charge"])
    z_abs = max(1, abs(z))  # avoid divide-by-zero if parsing fails
    delta_mass = float(parsed["mass_delta"])
    ion_mass = neutral_exact_mass + delta_mass
    mz = ion_mass / z_abs

    # Core features
    z_signed = float(z)
    polarity = 1.0 if z > 0 else -1.0
    charge_density = z_abs / max(1e-6, ion_mass)

    base = np.array([z_signed, float(z_abs), delta_mass, mz, polarity, charge_density], dtype=np.float32)

    # Optional: concatenate your existing ion multi-hot encoding
    ion_enc = self.encode_ion_types(parsed["ions"]).astype(np.float32)

    feat = np.concatenate([base, ion_enc], axis=0)
    return feat
```

> Recommendation: remove “pad to feature_dim” for this path and instead define the descriptor length explicitly from the chosen feature set.

---

### 2.2 Compute descriptors in `data_extract._get_data()` (replace OHE)

In `data_extract.py`, `_get_data()` currently builds `all_feats = adduct_OHE` via `AdductEncoder`. Replace this with adduct descriptors.

You need the molecule’s neutral exact mass from SMILES. Since you already use RDKit in `MoleculeIon`, the simplest is to compute it in `_get_data()` once per row.

#### Add an RDKit mass helper near `_get_data()`

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def _neutral_exact_mass_from_smiles(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return float(rdMolDescriptors.CalcExactMolWt(mol))
```

#### Patch sketch in `_get_data()` (data_extract.py)

```diff
@@ def _get_data(..., adduct_encoder: AdductEncoder | None = None) -> GraphXSectionDataset:
-    # Use AdductEncoder if provided, otherwise create a temporary one on this split
-    if adduct_encoder is not None:
-        adduct_OHE = adduct_encoder.transform(df["Adduct"])
-    else:
-        ...
-    all_feats = adduct_OHE
+    from GraphXSection_torch.adduct_features import AdductFeatureExtractor
+    extractor = AdductFeatureExtractor(feature_dim=0)  # feature_dim unused for descriptor path
+
+    neutral_masses = [ _neutral_exact_mass_from_smiles(s) for s in smiles ]
+    all_feats = np.stack(
+        [extractor.extract_descriptor(a, m) for a, m in zip(adducts, neutral_masses)],
+        axis=0
+    ).astype(np.float32)
```

**Important:** `GraphXSectionDataset` already stores `global_feats` as float32; no change needed there.

---

### 2.3 Remove `AdductEncoder` usage from `datasets.py` and `__main__.py`

Your pipeline currently instantiates and passes an `AdductEncoder` for OHE. With descriptors, it is no longer needed.

#### `datasets.py`
Where you currently branch on `use_adduct_virtual_node` and otherwise fit/transform with `AdductEncoder`, remove the encoder path and always compute descriptors (or gate by config).

#### `__main__.py`
Remove:

```python
from GraphXSection_torch.data_extract import AdductEncoder
encoder = AdductEncoder()
...
adduct_encoder=encoder
```

and rely on the descriptor computation in `_get_data()` / dataset creation.

---

## 3) Sigma model changes: embed descriptors with an MLP before concatenation

### 3.1 Add a global/adduct embedding MLP
Sigma currently assumes `global_feats` is already in the correct scale/space and concatenates it directly. Replace this with:

- `adduct_encoder_mlp: R^{F_raw} → R^{F_emb}` (e.g., 16 or 32 dims)
- concatenate `h_pooled || adduct_emb`

This matches the “embedding MLP” design you described.

#### Patch sketch (Sigma.py)

```diff
@@ class SigmaModel.__init__(..., global_feature_sizes, config, ...):
-    self.global_dim = global_feature_sizes
+    self.global_raw_dim = int(global_feature_sizes)
+    self.global_emb_dim = int(config.get("global_emb_dim", 16))
+
+    self.global_encoder = nn.Sequential(
+        nn.Linear(self.global_raw_dim, self.global_emb_dim * 2),
+        nn.ReLU(),
+        nn.Linear(self.global_emb_dim * 2, self.global_emb_dim),
+    )
+
+    # Dense head input dimension changes:
-    input_dim = self.ecc_channels + self.global_dim
+    input_dim = self.ecc_channels + self.global_emb_dim
```

#### Patch sketch in `forward()`

```diff
@@ def forward(self, batch):
     h_pooled = self.pool(h, batch_idx)

     if global_feats is not None:
-        h_pooled = torch.cat([h_pooled, global_feats], dim=-1)
+        adduct_emb = self.global_encoder(global_feats.float())
+        h_pooled = torch.cat([h_pooled, adduct_emb], dim=-1)
```

> This is the minimal “Sigma-compatible” change.

### 3.2 Feature scaling recommendation (important)
Your descriptor components (`mz`, `delta_mass`, `charge_density`) differ in magnitude. Do **one** of the following:

**Option A (recommended): standardize in the data pipeline**
- Fit `StandardScaler` on `all_feats` of the training split
- Apply to train/val/test consistently
- Store scaler parameters in the checkpoint / alongside the model

**Option B: normalize in-model**
- Add `nn.BatchNorm1d(global_raw_dim)` as the first layer of `global_encoder`
- This is simpler, but you must ensure stable batch statistics and evaluation mode

---

## 4) Configuration/hyperparameters

### 4.1 Set the raw descriptor dimension correctly
Because the model now uses a descriptor vector, you must set:

- `global_feature_sizes = descriptor_dim` (raw feature length)
- `model.global_emb_dim = 16` (or 32) in your config

The raw descriptor dim depends on which features you include. For example:

- core `[z_signed, z_abs, delta_mass, mz, polarity, charge_density]` → 6
- plus ion-type multi-hot (9) → 15 total

So `global_feature_sizes = 15` in that example.

### 4.2 Lightning module (`lightning_modules.py`)
`GraphXSectionModule` already accepts `global_feature_sizes` and forwards it into `SigmaModel`. Ensure the DataModule / training script passes the new descriptor dimension instead of the old OHE dimension.

---

## 5) Suggested validation checklist

1. **Descriptor sanity checks**
   - Print min/max for each dimension (especially `mz`, `delta_mass`)
   - Confirm `z_signed` matches adduct notation
2. **Ablation**
   - old one-hot baseline vs descriptor vector
   - descriptor only: `[z_signed, delta_mass, mz, polarity]`
   - descriptor + composition (ion multi-hot, counts)
3. **OOD adduct test**
   - Hold out one adduct family (e.g., Na adducts) and evaluate generalization

---

## 6) Summary of files to edit

- `adduct_features.py`
  - fix parsing sign logic
  - return float32 (no int casting)
  - add `extract_descriptor(adduct_str, neutral_exact_mass)`
- `data_extract.py`
  - in `_get_data()`: replace `AdductEncoder` OHE with descriptor computation (using RDKit exact mass)
- `datasets.py`, `__main__.py`
  - remove `AdductEncoder` fitting/usage paths (or gate by config)
- `Sigma.py`
  - add `global_encoder` MLP
  - change dense head input dim from `ecc_channels + global_dim` to `ecc_channels + global_emb_dim`
  - embed `global_feats` before concatenation
- Config / DataModule
  - set `global_feature_sizes = descriptor_dim`
  - set `model.global_emb_dim` (e.g., 16)
