"""
Hyperparameter Run Naming and Utilities for GraphXSection_torch

This module provides utilities for intelligent run naming, parameter validation,
and enhanced sweep configuration management.
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RunNamingSystem:
    """Intelligent naming system for W&B runs based on hyperparameters."""

    @staticmethod
    def format_value(value: Any, key: str) -> str:
        """Format parameter values for run names."""
        if isinstance(value, float):
            if "learning_rate" in key or "weight_decay" in key:
                # Scientific notation for rates
                return f"{value:.0e}"
            elif value < 0.01:
                return f"{value:.3f}"
            else:
                return f"{value:.2f}"
        elif isinstance(value, bool):
            return "T" if value else "F"
        elif isinstance(value, str):
            # Abbreviate common strings
            abbrev_map = {
                "batch_norm": "BN",
                "layer_norm": "LN",
                "adamw": "AdW",
                "adam": "Ad",
                "relu": "R",
                "gelu": "G",
                "set_gather": "SG",
                "attention": "Att",
                "sum": "S",
                "mean": "M",
                "max": "X",
            }
            return abbrev_map.get(value.lower(), value[:3])
        else:
            return str(value)

    @staticmethod
    def generate_run_name(config: Dict[str, Any], model_type: str) -> str:
        """Generate intelligent run name capturing essential parameters.

        Args:
            config: Full configuration dictionary
            model_type: Model type (ginny, sigma, pyggy, attentivefp)

        Returns:
            Formatted run name
        """
        model_config = config.get("model", {})

        # Essential parameters by model type
        essential_params = {
            "ginny": [
                ("learning_rate", "lr"),
                ("gnn_units", "dim"),
                ("num_GNN_layers", "lay"),
                ("dropout", "do"),
                ("readout", "pool"),
                ("optimizer", "opt"),
            ],
            "sigma": [
                ("learning_rate", "lr"),
                ("gnn_units", "dim"),
                ("num_GNN_layers", "lay"),
                ("pool_out_dim", "pdim"),
                ("dropout", "do"),
                ("readout", "pool"),
            ],
            "pyggy": [
                ("learning_rate", "lr"),
                ("hidden_dim", "dim"),
                ("num_layers", "lay"),
                ("dense_units", "dense"),
                ("dropout", "do"),
                ("readout", "pool"),
            ],
            "attentivefp": [
                ("learning_rate", "lr"),
                ("hidden_dim", "dim"),
                ("num_layers", "lay"),
                ("num_timesteps", "ts"),
                ("dense_units", "dense"),
                ("dropout", "do"),
            ],
        }

        # Get parameters for this model type
        params_to_include = essential_params.get(
            model_type.lower(), essential_params["ginny"]
        )

        # Build name components
        name_parts = [model_type.upper()[:4]]  # Model type prefix

        for param_key, short_name in params_to_include:
            value = model_config.get(param_key)
            if value is not None:
                formatted_value = RunNamingSystem.format_value(value, param_key)
                name_parts.append(f"{short_name}-{formatted_value}")

        # Add batch size if specified
        if model_config.get("batch_size"):
            bs = RunNamingSystem.format_value(model_config["batch_size"], "batch_size")
            name_parts.append(f"bs-{bs}")

        # Add normalization if used
        if model_config.get("normalization") or model_config.get("use_batchnorm"):
            norm = model_config.get("normalization") or (
                "BN" if model_config.get("use_batchnorm") else None
            )
            if norm:
                norm_str = RunNamingSystem.format_value(norm, "normalization")
                name_parts.append(f"norm-{norm_str}")

        return "_".join(name_parts)


class ParameterValidator:
    """Validates hyperparameter configurations for compatibility."""

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model configuration for common issues.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        model_config = config.get("model", {})
        model_type = model_config.get("type", "").lower()

        # Check required parameters by model type
        required_params = {
            "ginny": ["gnn_units", "num_GNN_layers", "learning_rate"],
            "sigma": ["gnn_units", "num_GNN_layers", "pool_out_dim", "learning_rate"],
            "pyggy": ["hidden_dim", "num_layers", "learning_rate"],
            "attentivefp": [
                "hidden_dim",
                "num_layers",
                "num_timesteps",
                "learning_rate",
            ],
        }

        if model_type in required_params:
            for param in required_params[model_type]:
                if param not in model_config:
                    issues.append(f"Missing required parameter: model.{param}")

        # Validate parameter ranges
        if "learning_rate" in model_config:
            lr = model_config["learning_rate"]
            if not (1e-6 <= lr <= 1e-1):
                issues.append(
                    f"Learning rate {lr} outside recommended range [1e-6, 1e-1]"
                )

        if "dropout" in model_config:
            dropout = model_config["dropout"]
            if not (0.0 <= dropout <= 0.9):
                issues.append(f"Dropout {dropout} outside valid range [0.0, 0.9]")

        # Check architecture consistency
        if "num_GNN_layers" in model_config and model_config["num_GNN_layers"] < 1:
            issues.append("Number of GNN layers must be at least 1")

        if "num_dense_layers" in model_config and model_config["num_dense_layers"] < 1:
            issues.append("Number of dense layers must be at least 1")

        # Model-specific validations
        if model_type == "attentivefp":
            if model_config.get("num_timesteps", 1) < 1:
                issues.append("AttentiveFP num_timesteps must be at least 1")

        return len(issues) == 0, issues

    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training-related configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check data paths - either single file or separate files
        has_single_data = "data_path" in config
        has_separate_data = all(
            key in config
            for key in ["train_data_path", "valid_data_path", "test_data_path"]
        )

        if not has_single_data and not has_separate_data:
            issues.append(
                "Must specify either 'data_path' or all three separate dataset paths ('train_data_path', 'valid_data_path', 'test_data_path')"
            )
        elif has_single_data and has_separate_data:
            issues.append(
                "Cannot specify both 'data_path' and separate dataset paths. Choose one approach."
            )

        # Validate file existence
        if has_single_data:
            data_path = Path(config["data_path"])
            if not data_path.exists():
                issues.append(f"Data file not found: {config['data_path']}")

        if has_separate_data:
            for key in ["train_data_path", "valid_data_path", "test_data_path"]:
                data_path = Path(config[key])
                if not data_path.exists():
                    issues.append(f"Data file not found: {config[key]}")

        # Check output path
        if "output_path" in config:
            output_path = Path(config["output_path"])
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")

        # Validate splits (only for single data file mode)
        if has_single_data:
            test_split = config.get("test_split", 0.1)
            valid_split = config.get("valid_split", 0.1)

            if not (0.0 < test_split < 1.0):
                issues.append(f"Test split {test_split} must be between 0 and 1")

            if not (0.0 < valid_split < 1.0):
                issues.append(f"Valid split {valid_split} must be between 0 and 1")

            if (test_split + valid_split) >= 1.0:
                issues.append(
                    f"Test + valid splits ({test_split + valid_split}) must be < 1.0"
                )

        return len(issues) == 0, issues


class SweepConfigEnhancer:
    """Enhances sweep configurations with advanced features."""

    @staticmethod
    def add_early_termination(
        sweep_config: Dict[str, Any], termination_type: str = "hyperband"
    ) -> Dict[str, Any]:
        """Add early termination to sweep config.

        Args:
            sweep_config: Base sweep configuration
            termination_type: Type of early termination

        Returns:
            Enhanced sweep configuration
        """
        if termination_type == "hyperband":
            sweep_config["early_terminate"] = {
                "type": "hyperband",
                "min_iter": 10,
                "eta": 2,
                "max_iter": 100,
            }
        elif termination_type == "median":
            sweep_config["early_terminate"] = {
                "type": "median_stopping",
                "min_iter": 15,
                "percentage": 50,
            }

        return sweep_config

    @staticmethod
    def add_metric_optimization(
        sweep_config: Dict[str, Any],
        primary_metric: str = "val_loss",
        secondary_metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """Add multi-metric optimization to sweep config.

        Args:
            sweep_config: Base sweep configuration
            primary_metric: Primary metric to optimize
            secondary_metrics: Additional metrics to track

        Returns:
            Enhanced sweep configuration
        """
        sweep_config["metric"] = {"goal": "minimize", "name": primary_metric}

        if secondary_metrics:
            # Note: W&B currently only supports single-objective optimization
            # But we can log these for analysis
            sweep_config["parameters"]["_secondary_metrics"] = {
                "value": secondary_metrics
            }

        return sweep_config

    @staticmethod
    def optimize_for_model_size(
        sweep_config: Dict[str, Any], target_size: str = "medium"
    ) -> Dict[str, Any]:
        """Optimize parameter ranges based on target model size.

        Args:
            sweep_config: Base sweep configuration
            target_size: Target model size (small, medium, large)

        Returns:
            Optimized sweep configuration
        """
        size_configs = {
            "small": {
                "gnn_units": [32, 64],
                "hidden_dim": [32, 64],
                "dense_units": [32, 64],
                "num_GNN_layers": [2, 3],
                "num_layers": [2, 3],
            },
            "medium": {
                "gnn_units": [64, 128],
                "hidden_dim": [64, 128],
                "dense_units": [64, 128],
                "num_GNN_layers": [3, 4],
                "num_layers": [3, 4],
            },
            "large": {
                "gnn_units": [128, 256, 512],
                "hidden_dim": [128, 256, 512],
                "dense_units": [128, 256, 512],
                "num_GNN_layers": [4, 5, 6],
                "num_layers": [4, 5, 6],
            },
            "all": {
                "gnn_units": [32, 64, 128, 256, 512],
                "hidden_dim": [32, 64, 128, 256, 512],
                "dense_units": [32, 64, 128, 256, 512],
                "num_GNN_layers": [2, 3, 4, 5, 6],
                "num_layers": [2, 3, 4, 5, 6, 7, 8],
            },
        }

        if target_size in size_configs:
            size_params = size_configs[target_size]

            for param, values in size_params.items():
                param_key = f"model.{param}"
                if param_key in sweep_config["parameters"]:
                    sweep_config["parameters"][param_key]["values"] = values

        return sweep_config


def create_sweep_script(
    sweep_config: Dict[str, Any], output_path: str = "run_sweep.sh"
) -> str:
    """Create a bash script to run the W&B sweep.

    Args:
        sweep_config: Sweep configuration dictionary
        output_path: Path for the output script

    Returns:
        Path to created script
    """
    script_content = f"""#!/bin/bash

# GraphXSection_torch Hyperparameter Optimization Sweep
# Generated automatically by hyperopt_runner.py

set -e  # Exit on any error

echo "Starting GraphXSection_torch hyperparameter optimization..."
echo "Sweep method: {sweep_config.get('method', 'bayes')}"
echo "Program: {sweep_config.get('program', 'GraphXSection_torch/__main__.py')}"

# Check if wandb is logged in
if ! wandb status > /dev/null 2>&1; then
    echo "Error: Please login to wandb first:"
    echo "  wandb login"
    exit 1
fi

# Check if GraphXSection_torch is importable
python -c "import GraphXSection_torch" 2>/dev/null || {{
    echo "Error: Cannot import GraphXSection_torch. Please check your Python environment."
    exit 1
}}

# Create sweep and get ID
echo "Creating W&B sweep..."
SWEEP_ID=$(wandb sweep --quiet sweep_config.yaml 2>&1 | grep -o 'Created sweep with ID: .*' | cut -d' ' -f5)

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep"
    exit 1
fi

echo "Created sweep: $SWEEP_ID"
echo "View sweep at: https://wandb.ai/{sweep_config.get('entity', 'YOUR_ENTITY')}/YOUR_PROJECT/sweeps/$SWEEP_ID"

# Run sweep agent
echo "Starting sweep agent..."
wandb agent $SWEEP_ID

echo "Sweep completed!"
"""

    with open(output_path, "w") as f:
        f.write(script_content)

    # Make script executable
    os.chmod(output_path, 0o755)

    logger.info(f"Created sweep execution script: {output_path}")
    return output_path


def check_environment() -> Tuple[bool, List[str]]:
    """Comprehensive environment check for GraphXSection_torch.

    Returns:
        Tuple of (is_ready, list_of_issues)
    """
    issues = []

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(
            f"Python {python_version.major}.{python_version.minor} is too old. Requires Python 3.8+"
        )

    # Check CUDA availability if using GPU
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
        else:
            logger.info("CUDA not available, will use CPU")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check critical packages and versions
    critical_packages = {
        "torch": "1.9.0",
        "pytorch_lightning": "1.5.0",
        "wandb": "0.12.0",
        "rdkit": "2020.09.1",
        "numpy": "1.19.0",
        "pandas": "1.1.0",
    }

    for package, min_version in critical_packages.items():
        try:
            module = __import__(package)
            if hasattr(module, "__version__"):
                version = module.__version__
                logger.info(f"{package}: {version}")
            else:
                logger.warning(f"{package}: version unknown")
        except ImportError:
            issues.append(
                f"Missing package: {package} (minimum version: {min_version})"
            )

    # Check GraphXSection_torch specific components
    try:
        from GraphXSection_torch.models import Ginny, Sigma, Pyggy, AttentiveFP

        logger.info("All GraphXSection_torch models importable")
    except ImportError as e:
        issues.append(f"GraphXSection_torch models not importable: {e}")

    # Check W&B authentication
    try:
        import wandb

        if wandb.api.api_key:
            logger.info("W&B authentication configured")
        else:
            issues.append("W&B not logged in (run 'wandb login')")
    except Exception:
        issues.append("W&B configuration issue")

    return len(issues) == 0, issues


def log_sweep_config_summary(sweep_config: Dict[str, Any], model_type: str) -> None:
    """Log a comprehensive summary of all hyperparameters in the sweep config.

    Args:
        sweep_config: The complete sweep configuration dictionary
        model_type: The model type being optimized
    """
    logger.info("=" * 80)
    logger.info(f"HYPERPARAMETER SWEEP CONFIGURATION SUMMARY - {model_type.upper()}")
    logger.info("=" * 80)

    # Log basic sweep info
    method = sweep_config.get("method", "unknown")
    metric = sweep_config.get("metric", {})
    metric_name = metric.get("name", "unknown")
    metric_goal = metric.get("goal", "unknown")

    logger.info(f"Sweep Method: {method}")
    logger.info(f"Optimization Metric: {metric_name} ({metric_goal})")
    logger.info(f"Program: {sweep_config.get('program', 'unknown')}")

    # Extract and categorize parameters
    parameters = sweep_config.get("parameters", {})

    # Categorize parameters
    model_params = {}
    training_params = {}
    data_params = {}
    other_params = {}

    for param_key, param_config in parameters.items():
        if param_key.startswith("model."):
            model_params[param_key] = param_config
        elif param_key in [
            "learning_rate",
            "batch_size",
            "max_epochs",
            "optimizer",
            "scheduler",
        ]:
            training_params[param_key] = param_config
        elif param_key in [
            "data_path",
            "train_data_path",
            "valid_data_path",
            "test_data_path",
            "test_split",
            "valid_split",
            "separate_datasets",
        ]:
            data_params[param_key] = param_config
        else:
            other_params[param_key] = param_config

    # Log model architecture parameters
    if model_params:
        logger.info("\n🏗️  MODEL ARCHITECTURE PARAMETERS:")
        for param_key, param_config in sorted(model_params.items()):
            param_name = param_key.replace("model.", "")
            if "values" in param_config:
                values = param_config["values"]
                logger.info(f"  • {param_name}: {values} ({len(values)} options)")
            elif "min" in param_config and "max" in param_config:
                logger.info(
                    f"  • {param_name}: [{param_config['min']}, {param_config['max']}] (continuous)"
                )
            elif "value" in param_config:
                logger.info(f"  • {param_name}: {param_config['value']} (fixed)")
            else:
                logger.info(f"  • {param_name}: {param_config}")

    # Log training parameters
    if training_params:
        logger.info("\n🎯 TRAINING PARAMETERS:")
        for param_key, param_config in sorted(training_params.items()):
            if "values" in param_config:
                values = param_config["values"]
                logger.info(f"  • {param_key}: {values} ({len(values)} options)")
            elif "min" in param_config and "max" in param_config:
                logger.info(
                    f"  • {param_key}: [{param_config['min']}, {param_config['max']}] (continuous)"
                )
            elif "value" in param_config:
                logger.info(f"  • {param_key}: {param_config['value']} (fixed)")
            else:
                logger.info(f"  • {param_key}: {param_config}")

    # Log data parameters
    if data_params:
        logger.info("\n📊 DATA PARAMETERS:")
        for param_key, param_config in sorted(data_params.items()):
            if "values" in param_config:
                values = param_config["values"]
                logger.info(f"  • {param_key}: {values} ({len(values)} options)")
            elif "value" in param_config:
                logger.info(f"  • {param_key}: {param_config['value']} (fixed)")
            else:
                logger.info(f"  • {param_key}: {param_config}")

    # Log other parameters
    if other_params:
        logger.info("\n⚙️  OTHER PARAMETERS:")
        for param_key, param_config in sorted(other_params.items()):
            if "values" in param_config:
                values = param_config["values"]
                logger.info(f"  • {param_key}: {values} ({len(values)} options)")
            elif "min" in param_config and "max" in param_config:
                logger.info(
                    f"  • {param_key}: [{param_config['min']}, {param_config['max']}] (continuous)"
                )
            elif "value" in param_config:
                logger.info(f"  • {param_key}: {param_config['value']} (fixed)")
            else:
                logger.info(f"  • {param_key}: {param_config}")

    # Calculate and log total parameter combinations
    total_combinations = 1
    variable_params = 0

    for param_config in parameters.values():
        if "values" in param_config:
            total_combinations *= len(param_config["values"])
            variable_params += 1
        elif "min" in param_config and "max" in param_config:
            variable_params += 1
            # For continuous parameters, we can't calculate exact combinations
            total_combinations = "∞ (continuous parameters present)"
            break

    logger.info(f"\n📈 SWEEP STATISTICS:")
    logger.info(f"  • Total parameters: {len(parameters)}")
    logger.info(f"  • Variable parameters: {variable_params}")
    logger.info(f"  • Fixed parameters: {len(parameters) - variable_params}")

    if isinstance(total_combinations, int):
        logger.info(f"  • Total parameter combinations: {total_combinations:,}")

        if method == "grid" and total_combinations > 1000:
            logger.warning(
                f"  ⚠️  Grid search with {total_combinations:,} combinations may be very slow!"
            )
        elif method == "bayes" and total_combinations < 10:
            logger.warning(
                f"  ⚠️  Bayesian optimization with only {total_combinations} combinations may be suboptimal!"
            )
    else:
        logger.info(f"  • Total parameter combinations: {total_combinations}")

    logger.info("=" * 80)
    input("Press Enter to continue...")


if __name__ == "__main__":
    # Example usage of utility functions
    logging.basicConfig(level=logging.INFO)

    print("Checking environment...")
    ready, issues = check_environment()

    if ready:
        print("✅ Environment is ready for hyperparameter optimization!")
    else:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
