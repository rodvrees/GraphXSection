#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Optimization Script for GraphXSection_torch

This script creates and executes Weights & Biases sweeps for all GraphXSection_torch models
with intelligent parameter ranges, automatic run naming, and package compatibility checks.

Usage:
    python hyperopt_runner.py --model ginny --sweep-type bayes --count 50
    python hyperopt_runner.py --model all --sweep-type grid --config custom_base_config.json
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import importlib.util
import logging
from datetime import datetime

# Import utility functions
from hyperopt_utils import log_sweep_config_summary

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphXSectionHyperOptRunner:
    """Main class for running hyperparameter optimization sweeps."""

    def __init__(self, base_config_path: Optional[str] = None):
        """Initialize the hyperopt runner.

        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = (
            base_config_path or "configs/example_lightning_config.json"
        )
        self.script_dir = Path(__file__).parent
        self.sweep_configs = {}
        self.compatibility_issues = []

    def check_package_compatibility(self) -> bool:
        """Check if the package is properly set up for hyperparameter optimization.

        Returns:
            True if compatible, False otherwise
        """
        logger.info("Checking package compatibility...")
        issues = []

        # Check if required files exist
        required_files = [
            "GraphXSection_torch/__main__.py",
            "GraphXSection_torch/train_lightning.py",
            "GraphXSection_torch/lightning_modules.py",
            "GraphXSection_torch/models/__init__.py",
        ]

        for file_path in required_files:
            full_path = self.script_dir / file_path
            if not full_path.exists():
                issues.append(f"Missing required file: {file_path}")

        # Check if base config exists
        base_config_full_path = self.script_dir / self.base_config_path
        if not base_config_full_path.exists():
            issues.append(f"Base config file not found: {self.base_config_path}")

        # Check Python dependencies
        required_packages = [
            ("torch", "PyTorch"),
            ("pytorch_lightning", "PyTorch Lightning"),
            ("wandb", "Weights & Biases"),
            ("rdkit", "RDKit"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
        ]

        for package_name, display_name in required_packages:
            try:
                importlib.import_module(package_name)
            except ImportError:
                issues.append(
                    f"Missing required package: {display_name} ({package_name})"
                )

        # Check if we can import GraphXSection_torch
        try:
            sys.path.insert(0, str(self.script_dir))
            import GraphXSection_torch
        except ImportError as e:
            issues.append(f"Cannot import GraphXSection_torch: {e}")

        # Check wandb login status
        try:
            import wandb

            # Try to get wandb API key
            if not wandb.api.api_key:
                issues.append(
                    "Weights & Biases not logged in. Run 'wandb login' first."
                )
        except Exception as e:
            logger.warning(f"Could not check wandb login status: {e}")

        self.compatibility_issues = issues

        if issues:
            logger.error("Compatibility issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("Package compatibility check passed!")
            return True

    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file.

        Returns:
            Configuration dictionary
        """
        config_path = self.script_dir / self.base_config_path

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded base config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load base config: {e}")
            raise

    def generate_run_name_pattern(self, model_type: str, sweep_type: str) -> str:
        """Generate intelligent run name pattern for W&B runs.

        Args:
            model_type: Model type (ginny, sigma, pyggy, attentivefp)
            sweep_type: Type of sweep (bayes, grid, random)

        Returns:
            Run name pattern with parameter placeholders
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Essential parameters for naming based on model type
        if model_type.lower() == "ginny":
            pattern = f"ginny_{sweep_type}_{timestamp}_lr-${{model.learning_rate:.0e}}_dim-${{model.gnn_units}}_lay-${{model.num_GNN_layers}}_do-${{model.dropout}}"
        elif model_type.lower() == "sigma":
            pattern = f"sigma_{sweep_type}_{timestamp}_lr-${{model.learning_rate:.0e}}_dim-${{model.gnn_units}}_lay-${{model.num_GNN_layers}}_pool-${{model.pool_out_dim}}"
        elif model_type.lower() == "pyggy":
            pattern = f"pyggy_{sweep_type}_{timestamp}_lr-${{model.learning_rate:.0e}}_dim-${{model.hidden_dim}}_lay-${{model.num_layers}}_dense-${{model.dense_units}}"
        elif model_type.lower() == "attentivefp":
            pattern = f"attfp_{sweep_type}_{timestamp}_lr-${{model.learning_rate:.0e}}_dim-${{model.hidden_dim}}_lay-${{model.num_layers}}_ts-${{model.num_timesteps}}"
        else:
            pattern = f"{model_type}_{sweep_type}_{timestamp}_lr-${{model.learning_rate:.0e}}_arch-${{model.gnn_units}}"

        return pattern

    def _ensure_wandb_enabled_in_sweep(
        self, sweep_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure W&B logging is enabled in sweep configuration.

        This is critical for sweeps to work properly - we override any base config
        setting that might have wandb disabled.
        """
        # Force enable W&B logging for sweeps
        sweep_config["parameters"]["wandb.use"] = {"value": True}

        # Ensure W&B project is set if not already specified
        if "wandb.project" not in sweep_config["parameters"]:
            sweep_config["parameters"]["wandb.project"] = {
                "value": "graphxsection-sweeps"
            }

        return sweep_config

    def create_ginny_sweep_config(self, sweep_type: str = "bayes") -> Dict[str, Any]:
        """Create sweep configuration for Ginny model."""
        base_config = self.load_base_config()
        run_name_pattern = self.generate_run_name_pattern("ginny", sweep_type)

        sweep_config = {
            "method": sweep_type,
            "program": "GraphXSection_torch/__main__.py",
            "parameters": {
                # Update run name with sweep parameters
                "run_name": {"value": run_name_pattern},
                # Model type - explicitly set to Ginny
                "model.type": {"value": "Ginny"},
                # Model architecture - Critical parameters
                "model.gnn_units": {"values": [32, 64, 128, 256]},
                "model.num_GNN_layers": {"values": [2, 3, 4, 5]},
                "model.dense_units": {"values": [32, 64, 128, 256]},
                "model.num_dense_layers": {"values": [1, 2, 3, 4]},
                # Regularization - Critical for generalization
                "model.dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
                "model.weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2,
                },
                # Optimization - Most critical parameter
                "model.learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-2,
                },
                "model.optimizer": {"values": ["adamw", "adam"]},
                # Batch size affects training dynamics
                "model.batch_size": {"values": [16, 32]},
                # Readout mechanism
                "model.readout": {
                    "values": ["sum", "mean", "max", "attention", "set_gather"]
                },
                # Normalization
                "model.normalization": {"values": [None, "batch_norm", "layer_norm"]},
                # Loss function
                "model.loss_function": {"values": ["mae", "mse", "huber"]},
                # GIN-specific parameters
                "model.train_eps": {"values": [True, False]},
                "model.apply_self_projection": {"values": [True, False]},
                "model.jumping_knowledge": {"values": [None, "cat", "max", "lstm"]},
                # Learning rate scheduler (ReduceLROnPlateau)
                "lr_scheduler.factor": {"values": [0.1, 0.2, 0.5]},
                "lr_scheduler.patience": {"values": [5, 10, 15, 20]},
                "lr_scheduler.min_lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-7,
                    "max": 1e-5,
                },
            },
            "metric": {"goal": "minimize", "name": "best_val_mae"},
            "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        }

        # Add base config parameters that shouldn't be swept
        for key, value in base_config.items():
            if key not in ["model", "run_name"] and not key.startswith("wandb"):
                sweep_config["parameters"][key] = {"value": value}

        # Ensure W&B is enabled for sweep
        sweep_config = self._ensure_wandb_enabled_in_sweep(sweep_config)

        return sweep_config

    def create_sigma_sweep_config(self, sweep_type: str = "bayes") -> Dict[str, Any]:
        """Create sweep configuration for Sigma model."""
        base_config = self.load_base_config()
        run_name_pattern = self.generate_run_name_pattern("sigma", sweep_type)

        sweep_config = {
            "method": sweep_type,
            "program": "GraphXSection_torch/__main__.py",
            "parameters": {
                "run_name": {"value": run_name_pattern},
                # Model type
                "model.type": {"value": "Sigma"},
                # Sigma-specific architecture
                "model.gnn_units": {"values": [32, 64, 128, 256]},
                "model.num_GNN_layers": {"values": [2, 3, 4]},
                "model.pool_out_dim": {"values": [32, 64, 128, 256]},
                "model.dense_units": {"values": [32, 64, 128, 256]},
                "model.num_dense_layers": {"values": [2, 3, 4]},
                # Regularization
                "model.dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
                "model.weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2,
                },
                # Optimization
                "model.learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 5e-4,
                    "max": 1e-2,
                },
                "model.optimizer": {"values": ["adamw", "adam"]},
                "model.batch_size": {"values": [16, 32]},
                # Sigma-specific readout
                "model.readout": {"values": ["sum", "median", "attention"]},
                "model.pool_config.gate_nn_layers": {"values": [1, 2, 3]},
                "model.pool_config.nn_layers": {"values": [1, 2, 3]},
                "model.pool_config.gate_nn_hidden": {"values": [16, 32, 64, 128]},
                "model.pool_config.nn_hidden": {"values": [16, 32, 64, 128]},
                # Activation and normalization
                "model.final_activation": {"values": ["gelu", "elu", "selu"]},
                "model.loss_function": {"values": ["mae", "mse", "huber"]},
                # Learning rate scheduler (ReduceLROnPlateau)
                "lr_scheduler.factor": {"values": [0.1, 0.2]},
                "lr_scheduler.patience": {"values": [15, 20, 30]},
                "lr_scheduler.min_lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-7,
                    "max": 1e-5,
                },
            },
            "metric": {"goal": "minimize", "name": "best_val_mae"},
            "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        }

        # Add base config parameters
        for key, value in base_config.items():
            if key not in ["model", "run_name"] and not key.startswith("wandb"):
                sweep_config["parameters"][key] = {"value": value}

        # Ensure W&B is enabled for sweep
        sweep_config = self._ensure_wandb_enabled_in_sweep(sweep_config)

        return sweep_config

    def create_pyggy_sweep_config(self, sweep_type: str = "bayes") -> Dict[str, Any]:
        """Create sweep configuration for Pyggy model."""
        base_config = self.load_base_config()
        run_name_pattern = self.generate_run_name_pattern("pyggy", sweep_type)

        sweep_config = {
            "method": sweep_type,
            "program": "GraphXSection_torch/__main__.py",
            "parameters": {
                "run_name": {"value": run_name_pattern},
                # Model type - explicitly set to Pyggy
                "model.type": {"value": "Pyggy"},
                # Pyggy architecture
                "model.hidden_dim": {"values": [32, 64, 128, 256]},
                "model.num_layers": {"values": [5, 6, 7, 8, 9, 10]},
                "model.dense_units": {"values": [128, 256, 512]},
                "model.num_dense_layers": {"values": [1, 2, 3, 4]},
                # Regularization
                "model.dropout": {"values": [0.0, 0.05, 0.1, 0.2]},
                "model.weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2,
                },
                # Optimization
                "model.learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-2,
                },
                "model.optimizer": {"values": ["adamw", "adam"]},
                "model.batch_size": {"values": [16, 32, 64, 128]},
                # Activation functions
                "model.hidden_activation": {"values": ["relu", "gelu", "elu", "selu"]},
                "model.final_activation": {"values": ["relu", "gelu", "elu", "selu"]},
                # Readout (Pyggy supports: sum, mean, max, median, attention, mlp)
                "model.readout": {"values": ["sum", "mean", "mlp"]},
                # Pool configuration (used for attention and MLP readout)
                # These will always be set, but only matter when readout=attention or mlp
                "model.pool_config.nn_layers": {"values": [1, 2, 3]},
                "model.pool_config.nn_hidden": {"values": [16, 32, 64, 128]},
                "model.pool_config.gate_nn_layers": {"values": [1, 2, 3]},
                "model.pool_config.gate_nn_hidden": {"values": [16, 32, 64, 128]},
                "model.pool_config.max_num_elements": {"values": [64, 128, 256, 512]},
                # Normalization options
                "model.use_batchnorm": {"values": [True, False]},
                "model.use_batchnorm_gin_mlp": {
                    "values": [None, "batch_norm", "layer_norm"]
                },
                # Jumping Knowledge
                "model.jumping_knowledge": {"values": [None, "cat", "lstm"]},
                # GIN-specific parameters
                "model.train_eps": {"values": [False]},
                "model.apply_self_projection": {"values": [True, False]},
                "model.loss_function": {"values": ["mae", "mse", "huber"]},
                # Learning rate scheduler (ReduceLROnPlateau)
                "lr_scheduler.factor": {"values": [0, 0.1, 0.2, 0.5]},
                "lr_scheduler.patience": {"values": [5, 10, 15, 20]},
                "lr_scheduler.min_lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-7,
                    "max": 1e-5,
                },
            },
            "metric": {"goal": "minimize", "name": "best_val_mae"},
            "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        }

        # Add base config parameters
        for key, value in base_config.items():
            if key not in ["model", "run_name"] and not key.startswith("wandb"):
                sweep_config["parameters"][key] = {"value": value}

        # Ensure W&B is enabled for sweep
        sweep_config = self._ensure_wandb_enabled_in_sweep(sweep_config)

        return sweep_config

    def create_attentivefp_sweep_config(
        self, sweep_type: str = "bayes"
    ) -> Dict[str, Any]:
        """Create sweep configuration for AttentiveFP model."""
        base_config = self.load_base_config()
        run_name_pattern = self.generate_run_name_pattern("attentivefp", sweep_type)

        sweep_config = {
            "method": sweep_type,
            "program": "GraphXSection_torch/__main__.py",
            "parameters": {
                "run_name": {"value": run_name_pattern},
                # Model type
                "model.type": {"value": "AttentiveFP"},
                # AttentiveFP architecture
                "model.hidden_dim": {"values": [32, 64, 128]},
                "model.num_layers": {"values": [2, 3, 4, 5]},
                "model.num_timesteps": {"values": [1, 2, 3, 4, 5]},
                "model.dense_units": {"values": [64, 128, 256, 512]},
                "model.num_dense_layers": {"values": [1, 2, 3, 4]},
                # Regularization
                "model.dropout": {
                    "values": [
                        0.0,
                        0.1,
                    ]
                },
                "model.weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2,
                },
                # Optimization
                "model.learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 5e-3,
                },
                "model.optimizer": {"values": ["adamw", "adam"]},
                "model.batch_size": {"values": [16, 32]},
                # Activation
                "model.final_activation": {"values": ["relu", "gelu", "selu", "tanh"]},
                "model.loss_function": {"values": ["mae", "mse", "huber"]},
                # Learning rate scheduler (ReduceLROnPlateau)
                "lr_scheduler.factor": {"values": [0.1, 0.2, 0.5]},
                "lr_scheduler.patience": {"values": [5, 10, 15]},
                "lr_scheduler.min_lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-7,
                    "max": 1e-6,
                },
            },
            "metric": {"goal": "minimize", "name": "best_val_mae"},
            "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        }

        # Add base config parameters
        for key, value in base_config.items():
            if key not in ["model", "run_name"] and not key.startswith("wandb"):
                sweep_config["parameters"][key] = {"value": value}

        # Ensure W&B is enabled for sweep
        sweep_config = self._ensure_wandb_enabled_in_sweep(sweep_config)

        return sweep_config

    def create_sweep_config(
        self, model_type: str, sweep_type: str = "bayes"
    ) -> Dict[str, Any]:
        """Create sweep configuration for specified model type.

        Args:
            model_type: Model type (ginny, sigma, pyggy, attentivefp)
            sweep_type: Sweep method (bayes, grid, random)

        Returns:
            Sweep configuration dictionary
        """
        model_type = model_type.lower()

        if model_type == "ginny":
            sweep_config = self.create_ginny_sweep_config(sweep_type)
        elif model_type == "sigma":
            sweep_config = self.create_sigma_sweep_config(sweep_type)
        elif model_type == "pyggy":
            sweep_config = self.create_pyggy_sweep_config(sweep_type)
        elif model_type == "attentivefp":
            sweep_config = self.create_attentivefp_sweep_config(sweep_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Log comprehensive summary of all hyperparameters
        log_sweep_config_summary(sweep_config, model_type)

        return sweep_config

    def save_sweep_config(
        self,
        sweep_config: Dict[str, Any],
        model_type: str,
        output_dir: str = "sweep_configs",
    ) -> str:
        """Save sweep configuration to YAML file.

        Args:
            sweep_config: Sweep configuration dictionary
            model_type: Model type for filename
            output_dir: Directory to save config files

        Returns:
            Path to saved config file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sweep_{model_type}_{timestamp}.yaml"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            yaml.dump(sweep_config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved sweep config to {filepath}")
        return str(filepath)

    def run_sweep(
        self,
        model_type: str,
        sweep_type: str = "bayes",
        count: int = 50,
        project_name: str = "graphxsection-hyperopt",
    ) -> str:
        """Create and run a W&B sweep for the specified model.

        Args:
            model_type: Model type to optimize
            sweep_type: Type of sweep (bayes, grid, random)
            count: Number of runs in the sweep
            project_name: W&B project name

        Returns:
            Sweep ID
        """
        # Check compatibility first
        if not self.check_package_compatibility():
            raise RuntimeError("Package compatibility check failed")

        # Create sweep configuration
        sweep_config = self.create_sweep_config(model_type, sweep_type)

        # Save configuration for reference
        config_path = self.save_sweep_config(sweep_config, model_type)

        # Initialize W&B sweep
        try:
            import wandb

            # Initialize wandb if not already done
            if not wandb.api.api_key:
                logger.error("Please login to wandb first: wandb login")
                raise RuntimeError("W&B not logged in")

            # Create the sweep
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            logger.info(f"Created W&B sweep: {sweep_id}")

            # Print instructions
            logger.info(f"To run the sweep, execute:")
            logger.info(f"  wandb agent {sweep_id}")
            logger.info(f"Or run it automatically with count={count}:")

            # Optionally run the agent automatically
            if count > 0:
                logger.info(f"Running sweep agent with {count} runs...")
                wandb.agent(sweep_id, count=count)

            return sweep_id

        except Exception as e:
            logger.error(f"Failed to create/run W&B sweep: {e}")
            raise

    def run_all_models(
        self,
        sweep_type: str = "bayes",
        count_per_model: int = 25,
        project_name: str = "graphxsection-hyperopt-all",
    ) -> Dict[str, str]:
        """Run hyperparameter optimization for all model types.

        Args:
            sweep_type: Type of sweep
            count_per_model: Number of runs per model
            project_name: W&B project name

        Returns:
            Dictionary mapping model types to sweep IDs
        """
        models = ["ginny", "sigma", "pyggy", "attentivefp"]
        sweep_ids = {}

        for model in models:
            try:
                logger.info(f"Creating sweep for {model} model...")
                sweep_id = self.run_sweep(
                    model_type=model,
                    sweep_type=sweep_type,
                    count=count_per_model,
                    project_name=project_name,
                )
                sweep_ids[model] = sweep_id
                logger.info(f"Successfully created sweep for {model}: {sweep_id}")
            except Exception as e:
                logger.error(f"Failed to create sweep for {model}: {e}")

        return sweep_ids


def main():
    """Main entry point for the hyperparameter optimization script."""
    parser = argparse.ArgumentParser(
        description="GraphXSection_torch Hyperparameter Optimization"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ginny", "sigma", "pyggy", "attentivefp", "all"],
        help="Model type to optimize",
    )

    parser.add_argument(
        "--sweep-type",
        type=str,
        default="bayes",
        choices=["bayes", "grid", "random"],
        help="W&B sweep method",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of runs in the sweep (0 to just create sweep)",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="graphxsection-torch-hyperopt",
        help="W&B project name",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Base configuration file (default: configs/example_lightning_config.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="sweep_configs",
        help="Directory to save sweep configuration files",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run compatibility check without creating sweeps",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = GraphXSectionHyperOptRunner(base_config_path=args.config)

    # Run compatibility check
    if args.check_only:
        compatible = runner.check_package_compatibility()
        if compatible:
            logger.info("✅ Package is ready for hyperparameter optimization!")
            return 0
        else:
            logger.error("❌ Package has compatibility issues")
            return 1

    # Run hyperparameter optimization
    try:
        if args.model == "all":
            sweep_ids = runner.run_all_models(
                sweep_type=args.sweep_type,
                count_per_model=args.count // 4,  # Distribute runs across models
                project_name=args.project,
            )

            logger.info("All sweeps created successfully!")
            for model, sweep_id in sweep_ids.items():
                logger.info(f"  {model}: {sweep_id}")

        else:
            sweep_id = runner.run_sweep(
                model_type=args.model,
                sweep_type=args.sweep_type,
                count=args.count,
                project_name=args.project,
            )

            logger.info(f"Sweep created successfully: {sweep_id}")

        return 0

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
