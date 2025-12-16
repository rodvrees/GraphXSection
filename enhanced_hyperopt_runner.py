#!/usr/bin/env python3
"""
Enhanced Hyperparameter Optimization Runner with W&B Integration

This script provides a complete hyperparameter optimization solution for GraphXSection_torch
with intelligent run naming, compatibility validation, and seamless W&B integration.

Usage Examples:
    # Run Bayesian optimization for Ginny model
    python enhanced_hyperopt_runner.py --model ginny --count 50

    # Run grid search for all models with custom config
    python enhanced_hyperopt_runner.py --model all --sweep-type grid --config configs/my_config.json

    # Just check compatibility without running sweeps
    python enhanced_hyperopt_runner.py --check-only

    # Create sweep configuration files without running
    python enhanced_hyperopt_runner.py --model pyggy --create-only --output-dir my_sweeps
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

# Import our enhanced utilities
from hyperopt_utils import (
    RunNamingSystem,
    ParameterValidator,
    SweepConfigEnhancer,
    create_sweep_script,
    check_environment,
    log_sweep_config_summary,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hyperopt.log", mode="a")],
)
logger = logging.getLogger(__name__)


class EnhancedHyperOptRunner:
    """Enhanced hyperparameter optimization runner with advanced features."""

    def __init__(self, base_config_path: Optional[str] = None):
        """Initialize the enhanced hyperopt runner.

        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = (
            base_config_path or "configs/example_lightning_config.json"
        )
        self.script_dir = Path(__file__).parent
        self.output_dir = self.script_dir / "sweep_configs"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.naming_system = RunNamingSystem()
        self.validator = ParameterValidator()
        self.enhancer = SweepConfigEnhancer()

    def comprehensive_environment_check(self) -> bool:
        """Run comprehensive environment and compatibility checks."""
        logger.info("🔍 Running comprehensive environment check...")

        # Use utility function for basic checks
        env_ready, env_issues = check_environment()

        if not env_ready:
            logger.error("❌ Environment check failed:")
            for issue in env_issues:
                logger.error(f"   {issue}")
            return False

        # Additional GraphXSection_torch specific checks
        logger.info("🔍 Checking GraphXSection_torch specific components...")

        try:
            # Check if we can load the main module
            sys.path.insert(0, str(self.script_dir))
            from GraphXSection_torch import __main__ as gxs_main

            logger.info("✅ GraphXSection_torch main module loadable")

            # Check model imports
            from GraphXSection_torch.models import Ginny, Sigma, Pyggy, AttentiveFP

            logger.info("✅ All model classes importable")

            # Check training components
            from GraphXSection_torch.train_lightning import (
                train_lightning_GraphXSection,
            )
            from GraphXSection_torch.lightning_modules import (
                GraphXSectionLightningModule,
            )

            logger.info("✅ Training components importable")

        except ImportError as e:
            logger.error(f"❌ GraphXSection_torch import failed: {e}")
            return False

        # Check base config file
        base_config_path = self.script_dir / self.base_config_path
        if not base_config_path.exists():
            logger.error(f"❌ Base config file not found: {self.base_config_path}")
            return False

        # Try to load and validate base config
        try:
            with open(base_config_path, "r") as f:
                base_config = json.load(f)

            config_valid, config_issues = self.validator.validate_model_config(
                base_config
            )
            if not config_valid:
                logger.warning("⚠️  Base config has validation issues:")
                for issue in config_issues:
                    logger.warning(f"   {issue}")
            else:
                logger.info("✅ Base configuration validated")

        except Exception as e:
            logger.error(f"❌ Failed to load/validate base config: {e}")
            return False

        logger.info("🎉 All environment checks passed!")
        return True

    def load_and_enhance_base_config(self) -> Dict[str, Any]:
        """Load base config and apply enhancements for sweeping."""
        config_path = self.script_dir / self.base_config_path

        with open(config_path, "r") as f:
            config = json.load(f)

        # Ensure W&B is enabled for sweeps
        if "wandb" not in config:
            config["wandb"] = {}
        config["wandb"]["use"] = True

        # Add sweep-specific enhancements
        if "mcp" not in config:
            config["mcp"] = {
                "mcp_path": "./checkpoints/sweep_model.ckpt",
                "mcp_monitor": "val_loss",
                "mcp_mode": "min",
            }

        logger.info("✅ Base configuration loaded and enhanced for sweeping")
        return config

    def create_enhanced_sweep_config(
        self, model_type: str, sweep_type: str = "bayes", target_size: str = "medium"
    ) -> Dict[str, Any]:
        """Create enhanced sweep configuration with all optimizations."""
        base_config = self.load_and_enhance_base_config()

        # Generate base sweep config using original hyperopt_runner logic
        from hyperopt_runner import GraphXSectionHyperOptRunner

        base_runner = GraphXSectionHyperOptRunner(self.base_config_path)
        sweep_config = base_runner.create_sweep_config(model_type, sweep_type)

        # CRITICAL FIX: Ensure W&B is enabled in sweep parameters
        # Override any wandb settings from base config to ensure sweep logging works
        sweep_config["parameters"]["wandb.use"] = {"value": True}

        # Also ensure the W&B project and other W&B settings are properly configured
        if "wandb.project" not in sweep_config["parameters"]:
            sweep_config["parameters"]["wandb.project"] = {
                "value": "graphxsection-sweeps"
            }

        if "wandb.entity" not in sweep_config["parameters"]:
            # Let W&B use default entity, but ensure parameter exists
            sweep_config["parameters"]["wandb.entity"] = {"value": None}

        # Enhance with utilities
        sweep_config = self.enhancer.add_early_termination(sweep_config, "hyperband")
        sweep_config = self.enhancer.add_metric_optimization(
            sweep_config, "best_val_mae", ["val_mae", "test_mae", "train_loss"]
        )
        # sweep_config = self.enhancer.optimize_for_model_size(sweep_config, target_size)

        # Add enhanced run naming
        # Note: W&B doesn't support dynamic naming in sweep config, so we'll handle this in the training script

        # Add metadata
        sweep_config["description"] = (
            f"Enhanced hyperparameter optimization for {model_type.upper()} model"
        )
        sweep_config["tags"] = ["graphxsection", model_type, sweep_type, "enhanced"]

        return sweep_config

    def create_training_wrapper_script(self, model_type: str) -> str:
        """Create a wrapper script that handles intelligent naming during sweep execution."""
        wrapper_content = f'''#!/usr/bin/env python3
"""
Training wrapper script for W&B sweeps with intelligent run naming.
Generated for {model_type} model hyperparameter optimization.
"""

import sys
import os
from pathlib import Path
import wandb

# Add proper paths - wrapper is in sweep_configs/, need to go up to parent
script_dir = Path(__file__).parent
parent_dir = script_dir.parent  # Go up one level to main directory
sys.path.insert(0, str(parent_dir))  # Add main directory with hyperopt_utils
sys.path.insert(0, str(parent_dir))  # Also ensure GraphXSection_torch is findable

from hyperopt_utils import RunNamingSystem
from GraphXSection_torch.__main__ import main as gxs_main

def wrapped_main():
    """Main function that sets intelligent run name before training."""
    
    import psutil
    import torch
    
    # Log system resources at start
    print(f"System memory: {{psutil.virtual_memory().total / (1024**3):.1f}}GB available")
    if torch.cuda.is_available():
        print(f"GPU memory: {{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}}GB available")
    
    # Initialize wandb run if not already done
    if wandb.run is None:
        wandb.init()
    
    # Generate intelligent run name based on sweep parameters
    naming_system = RunNamingSystem()
    
    # Get config from wandb
    config = dict(wandb.config)
    model_type = "{model_type}"
    
    # Validate critical parameters before starting
    try:
        lr = config.get('model.learning_rate', 0.001)
        batch_size = config.get('model.batch_size', 32)
        
        # Basic sanity checks
        if lr <= 0 or lr > 1:
            raise ValueError(f"Invalid learning rate: {{lr}}")
        if batch_size <= 0 or batch_size > 1024:
            raise ValueError(f"Invalid batch size: {{batch_size}}")
        
        # Memory estimation
        model_size = config.get('model.gnn_units', config.get('model.hidden_dim', 128))
        estimated_memory_gb = (batch_size * model_size * model_size) / (1024**3)
        
        if torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if estimated_memory_gb > available_memory_gb * 0.8:
                print(f"⚠️  WARNING: Estimated memory usage ({{estimated_memory_gb:.1f}}GB) may exceed available GPU memory ({{available_memory_gb:.1f}}GB)")
        
    except Exception as e:
        print(f"❌ Parameter validation failed: {{e}}")
        if wandb.run:
            wandb.log({{"parameter_validation_error": str(e)}})
        sys.exit(1)
    
    # Generate intelligent name
    try:
        run_name = naming_system.generate_run_name(config, model_type)
        
        # Update wandb run name
        wandb.run.name = run_name
        # Note: wandb.run.save() is for saving files, not run metadata
        # The name change is automatically synced
        
        print(f"Updated run name to: {{run_name}}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not generate intelligent name: {{e}}")
        # Continue with default name
    
    # Now run the original GraphXSection_torch main
    # The config will be merged by the original _overwrite_with_wandb function
    
    # Run original main with the base config file
    import click
    from click.testing import CliRunner
    
    # We need to pass the base config file path (relative to parent directory)
    base_config_path = str(parent_dir / "{self.base_config_path}")
    
    print(f"Starting training with config: {{base_config_path}}")
    print(f"W&B config parameters: {{dict(wandb.config)}}")
    
    # Create a context and invoke the original main
    runner = CliRunner()
    result = runner.invoke(gxs_main, [base_config_path])
    
    if result.exit_code != 0:
        print(f"❌ Training failed with exit code {{result.exit_code}}")
        print(f"Error output:")
        print(result.output)
        
        # Try to extract more specific error information
        if result.exception:
            print(f"Exception: {{result.exception}}")
            import traceback
            print(f"Traceback: {{traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__)}}")
        
        # Log to wandb if available
        if wandb.run:
            wandb.log({{"training_error": str(result.exception or result.output)}})
        
        sys.exit(result.exit_code)
    else:
        print(f"✅ Training completed successfully")
        print(f"Output: {{result.output}}")

if __name__ == "__main__":
    wrapped_main()
'''

        wrapper_path = self.output_dir / f"train_wrapper_{model_type}.py"
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)

        os.chmod(wrapper_path, 0o755)
        logger.info(f"Created training wrapper: {wrapper_path}")
        return str(wrapper_path)

    def run_enhanced_sweep(
        self,
        model_type: str,
        sweep_type: str = "bayes",
        count: int = 50,
        project_name: str = "graphxsection-enhanced",
        target_size: str = "medium",
        create_only: bool = False,
    ) -> Optional[str]:
        """Run enhanced hyperparameter sweep with all optimizations."""

        if not create_only and not self.comprehensive_environment_check():
            raise RuntimeError("Environment check failed")

        # Create enhanced sweep configuration
        sweep_config = self.create_enhanced_sweep_config(
            model_type, sweep_type, target_size
        )

        # Create wrapper script for intelligent naming
        wrapper_script = self.create_training_wrapper_script(model_type)

        # Update sweep config to use wrapper script
        sweep_config["program"] = wrapper_script

        # Save sweep configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_filename = (
            f"enhanced_sweep_{model_type}_{sweep_type}_{target_size}_{timestamp}.yaml"
        )
        config_path = self.output_dir / config_filename

        with open(config_path, "w") as f:
            yaml.dump(sweep_config, f, default_flow_style=False, indent=2)

        logger.info(f"💾 Saved enhanced sweep config: {config_path}")

        # Create execution script
        script_path = create_sweep_script(
            sweep_config, str(self.output_dir / f"run_{model_type}_sweep.sh")
        )

        if create_only:
            logger.info(f"📋 Created sweep configuration files (create-only mode)")
            logger.info(f"   Config: {config_path}")
            logger.info(f"   Script: {script_path}")
            logger.info(f"   Wrapper: {wrapper_script}")
            return None

        # Execute the sweep
        try:
            import wandb

            logger.info(f"🚀 Creating W&B sweep for {model_type} model...")
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            logger.info(f"✅ Created sweep: {sweep_id}")

            if count > 0:
                logger.info(f"🏃 Running {count} sweep iterations...")
                wandb.agent(sweep_id, count=count)
                logger.info("🎉 Sweep completed successfully!")
            else:
                logger.info(f"🔗 To run the sweep manually: wandb agent {sweep_id}")

            return sweep_id

        except Exception as e:
            logger.error(f"❌ Sweep execution failed: {e}")
            raise

    def run_all_models_enhanced(
        self,
        sweep_type: str = "bayes",
        count_per_model: int = 25,
        project_name: str = "graphxsection-all-enhanced",
        target_size: str = "medium",
        create_only: bool = False,
    ) -> Dict[str, Optional[str]]:
        """Run enhanced hyperparameter optimization for all models."""

        models = ["ginny", "sigma", "pyggy", "attentivefp"]
        results = {}

        for model in models:
            try:
                logger.info(f"🎯 Processing {model.upper()} model...")
                sweep_id = self.run_enhanced_sweep(
                    model_type=model,
                    sweep_type=sweep_type,
                    count=count_per_model,
                    project_name=project_name,
                    target_size=target_size,
                    create_only=create_only,
                )
                results[model] = sweep_id

                if not create_only:
                    logger.info(f"✅ {model.upper()} sweep completed: {sweep_id}")
                else:
                    logger.info(f"📋 {model.upper()} configuration created")

            except Exception as e:
                logger.error(f"❌ {model.upper()} failed: {e}")
                results[model] = None

        return results


def main():
    """Enhanced main entry point with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Enhanced GraphXSection_torch Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Bayesian optimization for Ginny with 50 iterations
    python enhanced_hyperopt_runner.py --model ginny --count 50
    
    # Create sweep configs for all models without running 
    python enhanced_hyperopt_runner.py --model all --create-only
    
    # Run grid search with medium-sized models
    python enhanced_hyperopt_runner.py --model pyggy --sweep-type grid --target-size medium
    
    # Just check environment compatibility
    python enhanced_hyperopt_runner.py --check-only
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ginny",
        choices=["ginny", "sigma", "pyggy", "attentivefp", "all"],
        help="Model type to optimize",
    )

    parser.add_argument(
        "--sweep-type",
        type=str,
        default="random",
        choices=["bayes", "grid", "random"],
        help="W&B sweep optimization method",
    )  # Should be random

    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of runs per sweep (0 to create without running)",
    )

    parser.add_argument(
        "--project", type=str, default="graphxsection-enhanced", help="W&B project name"
    )

    parser.add_argument("--config", type=str, help="Base configuration file path")

    parser.add_argument(
        "--target-size",
        type=str,
        default="all",
        choices=["small", "medium", "large", "all"],
        help="Target model size for parameter optimization",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="sweep_configs",
        help="Directory for saving sweep configuration files",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run environment compatibility check",
    )

    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create sweep configurations without running",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize enhanced runner
    runner = EnhancedHyperOptRunner(base_config_path=args.config)
    runner.output_dir = Path(args.output_dir)
    runner.output_dir.mkdir(exist_ok=True)

    # Handle check-only mode
    if args.check_only:
        logger.info("🔍 Running environment compatibility check...")
        if runner.comprehensive_environment_check():
            logger.info("🎉 Environment is ready for hyperparameter optimization!")
            return 0
        else:
            logger.error("❌ Environment has compatibility issues")
            return 1

    try:
        # Run hyperparameter optimization
        if args.model == "all":
            logger.info(
                "🚀 Starting enhanced hyperparameter optimization for all models..."
            )
            results = runner.run_all_models_enhanced(
                sweep_type=args.sweep_type,
                count_per_model=args.count // 4 if not args.create_only else 0,
                project_name=args.project,
                target_size=args.target_size,
                create_only=args.create_only,
            )

            if args.create_only:
                logger.info("📋 All sweep configurations created successfully!")
            else:
                logger.info("🎉 All sweeps completed successfully!")

            for model, result in results.items():
                if result:
                    logger.info(f"   {model.upper()}: {result}")
                else:
                    logger.warning(f"   {model.upper()}: FAILED")
        else:
            logger.info(
                f"🚀 Starting enhanced hyperparameter optimization for {args.model.upper()}..."
            )
            sweep_id = runner.run_enhanced_sweep(
                model_type=args.model,
                sweep_type=args.sweep_type,
                count=args.count,
                project_name=args.project,
                target_size=args.target_size,
                create_only=args.create_only,
            )

            if args.create_only:
                logger.info(f"📋 Sweep configuration created for {args.model.upper()}")
            else:
                logger.info(f"🎉 Sweep completed successfully: {sweep_id}")

        return 0

    except Exception as e:
        logger.error(f"❌ Hyperparameter optimization failed: {e}")
        if args.verbose:
            import traceback

            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
