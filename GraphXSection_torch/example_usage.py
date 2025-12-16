#!/usr/bin/env python3
"""
Example script demonstrating both standard PyTorch and Lightning training
for GraphXSection_torch.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from GraphXSection_torch.data_extract import data_extract
from GraphXSection_torch.train import train_GraphXSection, train_Eddy
from GraphXSection_torch.evaluate import evaluate, evaluate_eddy

# Lightning imports
try:
    from GraphXSection_torch.train_lightning import (
        train_lightning_GraphXSection,
        train_lightning_Eddy,
    )
    from GraphXSection_torch.evaluate_lightning import (
        evaluate_lightning_model,
        evaluate_lightning_eddy,
    )

    LIGHTNING_AVAILABLE = True
except ImportError:
    print(
        "Warning: PyTorch Lightning not available. Install with: pip install pytorch-lightning"
    )
    LIGHTNING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_config():
    """Create an example configuration for testing."""
    return {
        "run_name": "example_run",
        "data_path": "path/to/your/data.csv",  # Update this path
        "output_path": "./outputs",
        "smiles_col": "SMILES",
        "test_split": 0.1,
        "valid_split": 0.1,
        "save_dfs": False,
        "use_best_model": False,
        "save_model": True,
        "create_plots": True,
        "saliency_mapping": False,
        "model": {
            "type": "Ginny",
            "gnn_units": 64,  # Smaller for quick testing
            "num_GNN_layers": 2,
            "dense_units": 128,
            "num_dense_layers": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "dropout": 0.1,
            "use_batchnorm": True,
            "activation": "relu",
            "readout": "sum",
            "global_feat_dim": 5,
            "epochs": 5,  # Very short for testing
        },
        "mcp": {
            "mcp_path": "./checkpoints/test_model.pth",
            "mcp_monitor": "val_loss",
            "mcp_mode": "min",
        },
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 3,
            "min_delta": 0.001,
            "mode": "min",
        },
        "lr_scheduler": {"factor": 0.5, "patience": 2, "min_lr": 1e-6},
        "wandb": {
            "use": False,  # Disable for testing
            "project": "graphxsection-torch-test",
        },
        "num_workers": 0,  # 0 for debugging
    }


def run_standard_pytorch_training(config):
    """Run training with standard PyTorch."""
    logger.info("=== Running Standard PyTorch Training ===")

    # Load data
    logger.info("Loading data...")
    train_data, valid_data, test_data, test_df, valid_df = data_extract(config)

    # Train model
    logger.info("Training model...")
    if config["model"].get("type", None) != "Eddy":
        model = train_GraphXSection(train_data, valid_data, config)
        # Evaluate
        logger.info("Evaluating model...")
        predictions, rmedae = evaluate(model, test_data, test_df, config)
    else:
        model = train_Eddy(train_data, valid_data, config)
        predictions, rmedae = evaluate_eddy(model, test_data, config)

    logger.info(f"Standard PyTorch - RMedAE: {rmedae:.2f}%")
    return model, predictions, rmedae


def run_lightning_training(config):
    """Run training with PyTorch Lightning."""
    if not LIGHTNING_AVAILABLE:
        logger.error("PyTorch Lightning not available. Skipping Lightning training.")
        return None, None, None

    logger.info("=== Running PyTorch Lightning Training ===")

    # Load data
    logger.info("Loading data...")
    train_data, valid_data, test_data, test_df, valid_df = data_extract(config)

    # Train model
    logger.info("Training model with Lightning...")
    if config["model"].get("type", None) != "Eddy":
        lightning_module = train_lightning_GraphXSection(train_data, valid_data, config)
        # Evaluate
        logger.info("Evaluating model...")
        predictions, rmedae = evaluate_lightning_model(
            lightning_module, test_data, test_df, config
        )
    else:
        lightning_module = train_lightning_Eddy(train_data, valid_data, config)
        predictions, rmedae = evaluate_lightning_eddy(
            lightning_module, test_data, config
        )

    logger.info(f"PyTorch Lightning - RMedAE: {rmedae:.2f}%")
    return lightning_module, predictions, rmedae


def main():
    """Main example function."""
    logger.info("GraphXSection_torch Training Comparison Example")

    # Create example config
    config = example_config()

    # Check if data file exists
    if not Path(config["data_path"]).exists():
        logger.error(f"Data file not found: {config['data_path']}")
        logger.error(
            "Please update the 'data_path' in the config to point to your actual data file."
        )
        return

    # Create output directory
    Path(config["output_path"]).mkdir(exist_ok=True)
    Path("./checkpoints").mkdir(exist_ok=True)

    # Run standard PyTorch training
    try:
        pytorch_model, pytorch_predictions, pytorch_rmedae = (
            run_standard_pytorch_training(config.copy())
        )
    except Exception as e:
        logger.error(f"Standard PyTorch training failed: {e}")
        pytorch_model, pytorch_predictions, pytorch_rmedae = None, None, None

    # Run Lightning training
    try:
        lightning_module, lightning_predictions, lightning_rmedae = (
            run_lightning_training(config.copy())
        )
    except Exception as e:
        logger.error(f"Lightning training failed: {e}")
        lightning_module, lightning_predictions, lightning_rmedae = None, None, None

    # Compare results
    logger.info("=== Training Comparison ===")
    if pytorch_rmedae is not None:
        logger.info(f"Standard PyTorch RMedAE: {pytorch_rmedae:.2f}%")
    if lightning_rmedae is not None:
        logger.info(f"PyTorch Lightning RMedAE: {lightning_rmedae:.2f}%")

    if pytorch_rmedae is not None and lightning_rmedae is not None:
        diff = abs(pytorch_rmedae - lightning_rmedae)
        logger.info(f"Difference: {diff:.2f}%")
        if diff < 1.0:
            logger.info("✅ Results are very similar!")
        else:
            logger.info(
                "⚠️  Results differ - this might be due to different random seeds or training procedures"
            )

    logger.info("Example completed!")


if __name__ == "__main__":
    main()
