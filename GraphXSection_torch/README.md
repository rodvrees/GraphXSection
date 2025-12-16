# GraphXSection_torch

PyTorch implementation of GraphXSection using graphamole for graph neural networks to predict collisional cross-section (CCS) of small molecules. Now includes **PyTorch Lightning** support for streamlined training and advanced features.

## Overview

This is a PyTorch port of the original GraphXSection package, which was implemented in TensorFlow with molgraph. The main changes include:

- **Framework**: TensorFlow → PyTorch
- **Graph library**: molgraph → graphamole  
- **Data handling**: Custom PyTorch datasets with PyTorch Geometric
- **Training**: Native PyTorch training loops + **PyTorch Lightning support**
- **Models**: All models (Eddy, Ginny, Gato) ported to PyTorch

## Lightning Features ⚡

PyTorch Lightning provides several advantages:

- **Simplified Training**: Automatic handling of training loops, validation, and testing
- **Advanced Callbacks**: Built-in model checkpointing, early stopping, and learning rate monitoring
- **Multi-GPU Support**: Easy scaling to multiple GPUs
- **Experiment Tracking**: Seamless integration with Wandb, TensorBoard, and other loggers
- **Code Organization**: Cleaner separation of research code from engineering code

## Installation

1. **Install PyTorch and PyTorch Geometric:**
   ```bash
   pip install torch torch-geometric
   ```

2. **Install PyTorch Lightning:**
   ```bash
   pip install pytorch-lightning
   ```

3. **Install graphamole:**
   ```bash
   cd ../graphamole
   pip install -e .
   ```

4. **Install other dependencies:**
   ```bash
   cd ../GraphXSection_torch
   pip install -r requirements.txt
   ```

## Key Changes from Original

### Architecture Changes

1. **Data Pipeline**:
   - `GraphXSectionDataset`: Custom PyTorch dataset class
   - `collate_fn`: Custom collation for batching molecular graphs
   - Integration with graphamole's `Molecule` class for graph generation

2. **Models**:
   - **Base Model** (`model.py`): Core GraphXSection model using graphamole's GINEEncoder
   - **Eddy** (`models/Eddy.py`): MLP model for pre-computed embeddings
   - **Ginny** (`models/Ginny.py`): GIN-based model using graphamole
   - **Gato** (`models/Gato.py`): GAT-based model with attention mechanisms

3. **Training** (`train.py`):
   - PyTorch training loops with automatic mixed precision support
   - Custom callbacks for model checkpointing, early stopping, and wandb logging
   - Learning rate scheduling

4. **Evaluation** (`evaluate.py`):
   - PyTorch-based model evaluation
   - Metric calculation and visualization
   - Support for both graph-based and embedding-based models

### Usage

The API supports both standard PyTorch and Lightning training:

#### Standard PyTorch Training
```python
from GraphXSection_torch.data_extract import data_extract
from GraphXSection_torch.train import train_GraphXSection
from GraphXSection_torch.evaluate import evaluate

# Load and process data
train_data, valid_data, test_data, test_df, valid_df = data_extract(config)

# Train model
model = train_GraphXSection(train_data, valid_data, config)

# Evaluate
predictions, rmedae = evaluate(model, test_data, test_df, config)
```

#### PyTorch Lightning Training ⚡
```python
from GraphXSection_torch.data_extract import data_extract
from GraphXSection_torch.train_lightning import train_lightning_GraphXSection
from GraphXSection_torch.evaluate_lightning import evaluate_lightning_model

# Load and process data
train_data, valid_data, test_data, test_df, valid_df = data_extract(config)

# Train with Lightning
lightning_module = train_lightning_GraphXSection(train_data, valid_data, config)

# Evaluate
predictions, rmedae = evaluate_lightning_model(lightning_module, test_data, test_df, config)
```

#### Command Line Usage
```bash
# Standard PyTorch training
python -m GraphXSection_torch config.json

# PyTorch Lightning training
python -m GraphXSection_torch config.json --use-lightning
```

### Configuration

The configuration format supports both standard PyTorch and Lightning-specific settings:

```json
{
  "model": {
    "type": "Ginny",
    "gnn_units": 128,
    "num_GNN_layers": 3,
    "dense_units": 256,
    "num_dense_layers": 2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout": 0.1,
    "use_batchnorm": true,
    "epochs": 100
  },
  "mcp": {
    "mcp_path": "./checkpoints/best_model.pth",
    "mcp_monitor": "val_loss",
    "mcp_mode": "min"
  },
  "early_stopping": {
    "monitor": "val_loss",
    "patience": 15,
    "min_delta": 0.001
  },
  "lr_scheduler": {
    "factor": 0.5,
    "patience": 10,
    "min_lr": 1e-6
  },
  "lightning": {
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1,
    "precision": 32
  },
  "wandb": {
    "use": true,
    "project": "graphxsection-torch"
  }
}
```

## Model Types

1. **QSAR_linear**: Linear QSAR model with graph features
2. **Ginny**: Graph Isomorphism Network (GIN) based model
3. **Ginny3D**: GIN model with 3D molecular features
4. **Gato**: Graph Attention Network (GAT) based model
5. **Eddy**: MLP model using pre-computed molecular embeddings

## Weights & Biases Integration 📊

GraphXSection_torch includes comprehensive Weights & Biases (wandb) integration, matching the original TensorFlow implementation:

### Features
- **Automatic metric logging**: Training loss, validation metrics, learning rate
- **Best validation MAE tracking**: Tracks and logs the best validation MAE across epochs
- **Hyperparameter sweeps**: Full support for wandb sweep configuration
- **Model artifacts**: Optional model checkpointing to wandb
- **Real-time monitoring**: Live training and validation plots

### Configuration
```json
{
  "wandb": {
    "use": true,
    "project": "graphxsection-torch",
    "entity": "your-wandb-username",
    "tags": ["pytorch", "gnn", "molecular-properties"],
    "notes": "GraphXSection PyTorch implementation"
  }
}
```

### Usage
```bash
# Install wandb
pip install wandb

# Login (first time only)
wandb login

# Run with wandb logging
python -m GraphXSection_torch example_wandb_config.json --use-lightning
```

### Hyperparameter Sweeps
```bash
# Create sweep
wandb sweep sweep_config.yaml

# Run sweep agents
wandb agent SWEEP_ID
```

### Logged Metrics
The implementation logs the same metrics as the original GraphXSection:
- `train_loss`, `val_loss`: Training and validation losses
- `train_mae`, `val_mae`: Mean Absolute Error
- `best_val_mae`: Best validation MAE seen so far
- `test_mean_absolute_error`: Final test MAE
- `test_medAE`: Test median absolute error
- `test_RMAE`: Test relative mean absolute error
- `test_RMedAE`: Test relative median absolute error
- `test_corr`: Test correlation coefficient

## Features

- **Modern PyTorch**: Uses latest PyTorch features and best practices
- **Graphamole Integration**: Leverages graphamole for robust molecular graph handling
- **Flexible Architecture**: Easy to extend with new model types
- **Comprehensive Evaluation**: Detailed metrics and visualization
- **Experiment Tracking**: Built-in wandb integration
- **Saliency Mapping**: Gradient-based interpretability methods

## Key Files

- `__main__.py`: Main entry point with Lightning support (`--use-lightning`)
- `data_extract.py`: Data loading and preprocessing with PyTorch datasets
- `model.py`: Base model classes and utilities

### Standard PyTorch Files
- `train.py`: Training pipeline with PyTorch
- `evaluate.py`: Model evaluation and metrics

### PyTorch Lightning Files ⚡  
- `lightning_modules.py`: Lightning module implementations
- `lightning_data.py`: Lightning data modules
- `train_lightning.py`: Lightning training pipeline
- `evaluate_lightning.py`: Lightning evaluation

### Models
- `models/`: Individual model implementations
  - `Eddy.py`: Embedding-based MLP
  - `Ginny.py`: GIN-based graph model
  - `Gato.py`: GAT-based graph model

### Utilities
- `feature_calc.py`: Molecular feature calculation utilities
- `saliency_mapping.py`: Gradient-based interpretability
- `encode_graph.py`: Graph encoding utilities

## TODO / Notes

1. **Pretrained Embeddings**: The Eddy model currently uses placeholder embeddings. You'll need to implement the actual pretrained model loading.

2. **Feature Vocabularies**: The categorical feature vocabulary sizes are currently hardcoded. These should be determined from the actual data or featurizers.

3. **3D Features**: The 3D molecular feature calculation is partially implemented and may need refinement.

4. **OpenBabel Integration**: The `obabel_wrapper.py` is a placeholder and needs actual OpenBabel integration if required.

5. **Testing**: Comprehensive testing should be added to ensure parity with the original implementation.

## Dependencies

- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.2.0
- **PyTorch Lightning >= 1.8.0**
- Graphamole (local installation)
- RDKit
- NumPy, Pandas, Scikit-learn
- Wandb (for experiment tracking)
- Rich, Click (for CLI)

## Lightning Advantages ⚡

Using PyTorch Lightning provides several benefits:

1. **Automatic Multi-GPU**: Scale to multiple GPUs without code changes
2. **Professional Logging**: Built-in integration with Wandb, TensorBoard
3. **Robust Callbacks**: Model checkpointing, early stopping, learning rate scheduling
4. **Less Boilerplate**: Focus on research, not engineering
5. **Reproducibility**: Built-in seeding and deterministic training
6. **Mixed Precision**: Automatic mixed precision training
7. **Debugging**: Built-in debugging tools and fast dev run

## Migration from Original

If you're migrating from the original TensorFlow version:

1. Update your configuration files (minimal changes needed)
2. Choose between standard PyTorch or Lightning training
3. **Recommended**: Use Lightning for production workflows
4. Retrain your models with the new PyTorch implementation
5. Update any custom model extensions to use PyTorch
6. Verify results match the original implementation

### Lightning vs Standard PyTorch

**Use Lightning when:**
- You want professional-grade training infrastructure
- You need multi-GPU training
- You want advanced callbacks and logging
- You're doing production ML workflows

**Use Standard PyTorch when:**
- You need maximum control over training loops
- You're doing research requiring custom training logic
- You're learning or experimenting

The PyTorch version provides similar or better performance while offering more flexibility and easier debugging. Lightning adds production-ready features on top.
