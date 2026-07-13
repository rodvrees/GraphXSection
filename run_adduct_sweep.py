#!/usr/bin/env python3
"""
Focused Hyperparameter Optimization for Adduct Generalization

This script runs a targeted hyperparameter sweep focusing only on parameters
related to the new generalized adduct descriptor features.

Usage:
    python run_adduct_sweep.py --count 50
    python run_adduct_sweep.py --count 100 --project my-custom-project
    python run_adduct_sweep.py --dry-run  # Just create the sweep without running
"""

import argparse
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_wandb_login():
    """Check if user is logged into W&B"""
    try:
        import wandb

        if wandb.api.api_key:
            logger.info("✓ W&B authentication found")
            return True
        else:
            logger.error("✗ W&B not authenticated. Please run: wandb login")
            return False
    except Exception as e:
        logger.error(f"✗ W&B check failed: {e}")
        return False


def check_environment():
    """Verify environment is ready"""
    logger.info("Checking environment...")

    # Check Python packages
    required_packages = ["torch", "pytorch_lightning", "wandb", "rdkit"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"✗ Missing packages: {', '.join(missing)}")
        return False

    logger.info("✓ All required packages found")

    # Check if GraphXSection_torch is importable
    try:
        import GraphXSection_torch

        logger.info("✓ GraphXSection_torch importable")
    except ImportError as e:
        logger.error(f"✗ Cannot import GraphXSection_torch: {e}")
        return False

    # Check W&B
    if not check_wandb_login():
        return False

    logger.info("✓ Environment check passed!")
    return True


def summarize_sweep_config(config_path: Path):
    """Print a summary of the sweep configuration"""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("\n" + "=" * 80)
    logger.info("ADDUCT GENERALIZATION HYPERPARAMETER SWEEP")
    logger.info("=" * 80)
    logger.info(f"Sweep method: {config.get('method', 'unknown').upper()}")
    logger.info(
        f"Optimization metric: {config['metric']['name']} ({config['metric']['goal']})"
    )

    # Count swept parameters
    swept_params = []
    fixed_params = []

    for param, spec in config.get("parameters", {}).items():
        if "values" in spec or "distribution" in spec:
            swept_params.append(param)
        else:
            fixed_params.append(param)

    logger.info(f"\n📊 SWEEP STATISTICS:")
    logger.info(f"  • Total parameters: {len(config.get('parameters', {}))}")
    logger.info(f"  • Swept parameters: {len(swept_params)}")
    logger.info(f"  • Fixed parameters: {len(fixed_params)}")

    logger.info(f"\n🎯 SWEPT PARAMETERS (Adduct-related):")
    for param in sorted(swept_params):
        spec = config["parameters"][param]
        if "values" in spec:
            logger.info(f"  • {param}: {len(spec['values'])} discrete values")
        elif "distribution" in spec:
            logger.info(
                f"  • {param}: {spec['distribution']} ({spec.get('min')} to {spec.get('max')})"
            )

    logger.info("\n💡 FOCUS AREAS:")
    logger.info("  • Dense head architecture (layers, units, activation)")
    logger.info("  • Global feature encoder (adduct descriptors)")
    logger.info("  • Training dynamics (learning rate, weight decay, dropout)")
    logger.info("  • Learning rate scheduling")
    logger.info("=" * 80 + "\n")


def create_sweep(config_path: Path, project: str):
    """Create a W&B sweep and return the sweep ID"""
    import wandb
    import yaml

    logger.info(f"Creating sweep from: {config_path}")

    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Update project name if specified
    if "parameters" not in sweep_config:
        sweep_config["parameters"] = {}

    sweep_config["parameters"]["wandb.project"] = {"value": project}

    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
        logger.info(f"✓ Created sweep: {sweep_id}")
        logger.info(
            f"  View at: https://wandb.ai/{wandb.api.default_entity}/{project}/sweeps/{sweep_id}"
        )
        return sweep_id
    except Exception as e:
        logger.error(f"✗ Failed to create sweep: {e}")
        return None


def run_sweep(sweep_id: str, count: int, project: str):
    """Run the sweep agent"""
    import wandb
    
    # Import the wrapper function
    sys.path.insert(0, str(Path(__file__).parent / "sweep_configs"))
    from wandb_sweep_wrapper import run_training_with_config

    logger.info(f"\n🚀 Starting sweep agent...")
    logger.info(f"  Sweep ID: {sweep_id}")
    logger.info(f"  Count: {count} runs")
    logger.info(f"  Project: {project}")
    logger.info("\n" + "=" * 80)

    try:
        # Pass the function directly to avoid the program/command complexity
        wandb.agent(sweep_id, function=run_training_with_config, project=project, count=count)
        logger.info("\n✓ Sweep completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n⚠ Sweep interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Sweep failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run focused hyperparameter optimization for adduct generalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run sweep with 50 iterations
    python run_adduct_sweep.py --count 50
    
    # Run sweep with custom project name
    python run_adduct_sweep.py --count 100 --project my-adduct-sweep
    
    # Create sweep without running (to run agents separately)
    python run_adduct_sweep.py --dry-run
    
    # Quick test with just 5 runs
    python run_adduct_sweep.py --count 5 --project test-adduct-sweep
        """,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of sweep runs to execute (default: 50)",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="graphxsection-adduct-generalization",
        help="W&B project name (default: graphxsection-adduct-generalization)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create sweep but don't run agent (useful for manual agent launch)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="sweep_configs/sigma_adduct_generalization_sweep.yaml",
        help="Path to sweep configuration file",
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config

    if not config_path.exists():
        logger.error(f"✗ Sweep config not found: {config_path}")
        return 1

    # Show configuration summary
    summarize_sweep_config(config_path)

    # Check environment
    if not check_environment():
        logger.error("\n✗ Environment check failed. Please fix the issues above.")
        return 1

    # Create sweep
    sweep_id = create_sweep(config_path, args.project)
    if not sweep_id:
        return 1

    # Run sweep (unless dry-run)
    if args.dry_run:
        logger.info("\n✓ Sweep created (dry-run mode)")
        logger.info(f"\nTo run the sweep agent manually, use:")
        logger.info(f"  wandb agent {sweep_id}")
        return 0

    # Ask for confirmation
    try:
        response = input(
            f"\n▶ Ready to run {args.count} sweep iterations. Continue? [Y/n] "
        )
        if response.lower() in ["n", "no"]:
            logger.info("Sweep cancelled by user")
            return 0
    except KeyboardInterrupt:
        logger.info("\nSweep cancelled by user")
        return 0

    # Run the sweep
    try:
        run_sweep(sweep_id, args.count, args.project)
        return 0
    except Exception as e:
        logger.error(f"Sweep execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
