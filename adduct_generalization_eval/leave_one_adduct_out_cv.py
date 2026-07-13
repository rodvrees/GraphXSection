"""Leave-One-Adduct-Out Cross-Validation for adduct generalization evaluation.

This script trains n models on the GraphCCS data, where n is the number of unique
adducts. Each model is trained on all data except molecules with one specific 
adduct, which are held out for testing. This evaluates model generalization to 
unseen adducts.

For each fold, two test sets are created:
1. Original adduct test: Molecules with the held-out adduct (ground truth)
2. Baseline adduct test: Same molecules but with adduct replaced by [M+H]+ (positive)
   or [M-H]- (negative) to evaluate performance against a simple baseline

Models are trained from scratch unless a base model is provided for transfer learning.

Usage:
    python adduct_generalization_eval/leave_one_adduct_out_cv.py \\
        --config configs/example_lightning_config.json \\
        --data-path data/GraphCCS.csv \\
        --output-dir outputs/adduct_cv/
    
    # Optional: with transfer learning
    python adduct_generalization_eval/leave_one_adduct_out_cv.py \\
        --config configs/example_lightning_config.json \\
        --data-path data/GraphCCS.csv \\
        --base-model checkpoints/best_model.ckpt \\
        --output-dir outputs/adduct_cv/ \\
        --freeze-layers
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Set matplotlib to non-interactive backend to avoid threading issues
import matplotlib

matplotlib.use("Agg")  # Must be before importing pyplot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphXSection_torch.train_lightning import train_lightning_GraphXSection
from GraphXSection_torch.evaluate_lightning import evaluate_lightning_model
from GraphXSection_torch.lightning_data import GraphXSectionDataModule


def load_graphccs_data(data_path: str) -> pd.DataFrame:
    """Load GraphCCS data and validate required columns."""
    df = pd.read_csv(data_path)
    required_cols = ["SMILES", "CCS", "Adduct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} samples from {data_path}")
    logger.info(f"Found {df['Adduct'].nunique()} unique adducts")

    # Log adduct distribution
    logger.info("Adduct distribution:")
    for adduct, count in df["Adduct"].value_counts().items():
        logger.info(f"  {adduct}: {count}")

    return df


def infer_adduct_charge(adduct: str) -> str:
    """Infer if an adduct is positive or negative based on its notation.

    Args:
        adduct: Adduct string (e.g., '[M+H]+', '[M-H]-')

    Returns:
        '+' for positive, '-' for negative
    """
    if adduct.endswith("+"):
        return "+"
    elif adduct.endswith("-"):
        return "-"
    else:
        # Default to positive if unclear
        logger.warning(
            f"Could not infer charge for adduct: {adduct}, defaulting to positive"
        )
        return "+"


def create_fold_data(
    df: pd.DataFrame,
    test_adduct: str,
    valid_fraction: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/valid/test splits for one fold.

    Args:
        df: Full GraphCCS dataset
        test_adduct: Adduct to hold out for testing
        valid_fraction: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_df, valid_df, test_original_df, test_baseline_df
    """
    # Split test adduct molecules
    test_original_df = df[df["Adduct"] == test_adduct].copy()
    train_pool = df[df["Adduct"] != test_adduct].copy()

    # Create baseline test set: replace adduct with [M+H]+ or [M-H]-
    test_baseline_df = test_original_df.copy()
    adduct_charge = infer_adduct_charge(test_adduct)
    baseline_adduct = "[M+H]+" if adduct_charge == "+" else "[M-H]-"
    test_baseline_df["Adduct"] = baseline_adduct

    logger.info(
        f"Test adduct '{test_adduct}' is {adduct_charge}, baseline: '{baseline_adduct}'"
    )

    # Split validation from training pool
    train_pool = train_pool.sample(frac=1, random_state=random_seed).reset_index(
        drop=True
    )
    valid_size = int(len(train_pool) * valid_fraction)
    valid_df = train_pool[:valid_size].copy()
    train_df = train_pool[valid_size:].copy()

    logger.info(
        f"Fold '{test_adduct}': train={len(train_df)}, valid={len(valid_df)}, "
        f"test_original={len(test_original_df)}, test_baseline={len(test_baseline_df)}"
    )

    return train_df, valid_df, test_original_df, test_baseline_df


def save_fold_csvs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_original_df: pd.DataFrame,
    test_baseline_df: pd.DataFrame,
    fold_dir: Path,
) -> Tuple[str, str, str, str]:
    """Save fold data to CSV files.

    Returns:
        train_path, valid_path, test_original_path, test_baseline_path
    """
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(fold_dir / "train.csv")
    valid_path = str(fold_dir / "valid.csv")
    test_original_path = str(fold_dir / "test_original.csv")
    test_baseline_path = str(fold_dir / "test_baseline.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_original_df.to_csv(test_original_path, index=False)
    test_baseline_df.to_csv(test_baseline_path, index=False)

    return train_path, valid_path, test_original_path, test_baseline_path


def train_fold(
    config: Dict[str, Any],
    train_path: str,
    valid_path: str,
    test_original_path: str,
    fold_name: str,
    output_dir: Path,
    base_model_path: str = None,
) -> Any:
    """Train model for one fold using transfer learning.

    Args:
        config: Base configuration dict
        train_path: Path to training CSV
        valid_path: Path to validation CSV
        test_original_path: Path to test CSV (for config, not used in training)
        fold_name: Name of the fold (sanitized adduct name)
        output_dir: Output directory for all folds
        base_model_path: Path to base model checkpoint for transfer learning

    Returns:
        trained_model (LightningModule), adduct_feat_dim
    """
    # Update config for this fold
    fold_config = config.copy()
    fold_config["data_path"] = train_path
    fold_config["valid_data_path"] = valid_path
    fold_config["test_data_path"] = test_original_path
    fold_config["output_path"] = str(output_dir / fold_name)
    fold_config["run_name"] = f"{config.get('run_name', 'adduct_cv')}_{fold_name}"

    # Update checkpoint path to be fold-specific
    if "mcp" in fold_config:
        fold_config["mcp"]["mcp_path"] = str(output_dir / fold_name / "best_model.ckpt")

    logger.info(f"Training fold: {fold_name}")

    # Build datasets from CSV files
    logger.info(f"Building datasets from CSVs: train={train_path}, valid={valid_path}")

    # Get feature configuration from config
    use_adduct_descriptors = fold_config.get("use_adduct_descriptors", True)
    one_hot_encode = fold_config.get("one_hot_encode_feats", False)
    feature_config_path = fold_config.get("feature_config_path", None)

    from GraphXSection_torch.datasets import build_dataset_from_df

    train_df = pd.read_csv(train_path)
    train_dataset, adduct_feat_dim = build_dataset_from_df(
        train_df,
        smiles_col=fold_config.get("smiles_col", "SMILES"),
        adduct_col=fold_config.get("adduct_col", "Adduct"),
        use_adduct_descriptors=use_adduct_descriptors,
        one_hot_encode_feats=one_hot_encode,
        feature_config_path=feature_config_path,
    )

    valid_df = pd.read_csv(valid_path)
    valid_dataset, _ = build_dataset_from_df(
        valid_df,
        smiles_col=fold_config.get("smiles_col", "SMILES"),
        adduct_col=fold_config.get("adduct_col", "Adduct"),
        use_adduct_descriptors=use_adduct_descriptors,
        one_hot_encode_feats=one_hot_encode,
        feature_config_path=feature_config_path,
    )

    # Build test dataset for model initialization (won't be used in training)
    test_df = pd.read_csv(test_original_path)
    test_dataset, _ = build_dataset_from_df(
        test_df,
        smiles_col=fold_config.get("smiles_col", "SMILES"),
        adduct_col=fold_config.get("adduct_col", "Adduct"),
        use_adduct_descriptors=use_adduct_descriptors,
        one_hot_encode_feats=one_hot_encode,
        feature_config_path=feature_config_path,
    )

    if base_model_path:
        logger.info(f"Using transfer learning from: {base_model_path}")
        model = train_lightning_GraphXSection(
            fold_config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            global_feature_sizes=adduct_feat_dim,
            base_model_path=base_model_path,
        )
    else:
        logger.info("Training from scratch (no base model provided)")
        model = train_lightning_GraphXSection(
            fold_config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            global_feature_sizes=adduct_feat_dim,
        )

    return model, adduct_feat_dim


def evaluate_fold(
    model: Any,
    test_csv: str,
    config: Dict[str, Any],
    adduct_feat_dim: Dict[str, int],
    fold_name: str,
    fold_dir: Path,
    test_type: str = "original",
) -> Dict[str, float]:
    """Evaluate model on a test set.

    Args:
        model: Trained LightningModule
        test_csv: Path to test CSV
        config: Configuration dict
        adduct_feat_dim: Adduct feature dimension
        fold_name: Name of the fold
        fold_dir: Directory for this fold's outputs
        test_type: 'original' or 'baseline' to distinguish test sets

    Returns:
        test_metrics
    """
    logger.info(f"Evaluating fold: {fold_name} on {test_type} test set")

    # Build data module for test set
    from GraphXSection_torch.datasets import build_dataset_from_df

    test_df = pd.read_csv(test_csv)
    test_dataset, _ = build_dataset_from_df(
        test_df,
        smiles_col=config.get("smiles_col", "SMILES"),
        adduct_col=config.get("adduct_col", "Adduct"),
        use_adduct_descriptors=config.get("use_adduct_descriptors", True),
        one_hot_encode_feats=config.get("one_hot_encode_feats", False),
        feature_config_path=config.get("feature_config_path", None),
    )

    data_module = GraphXSectionDataModule(
        train_dataset=test_dataset,  # Dummy, won't be used
        valid_dataset=test_dataset,  # Dummy, won't be used
        test_dataset=test_dataset,
        config=config,
        global_feature_sizes=adduct_feat_dim,
    )

    # Update config for this fold's output
    fold_config = config.copy()
    fold_config["output_path"] = str(fold_dir / test_type)
    fold_config["run_name"] = (
        f"{config.get('run_name', 'adduct_cv')}_{fold_name}_{test_type}"
    )

    # Run evaluation
    predictions, rmedae = evaluate_lightning_model(
        lightning_module=model,
        data_module_or_dataset=data_module,
        test_df=test_df,
        config=fold_config,
    )

    # Calculate additional metrics
    targets = test_df["CCS"].values
    absolute_errors = np.abs(predictions - targets)
    relative_errors = absolute_errors / targets

    mae = np.mean(absolute_errors)
    medae = np.median(absolute_errors)
    rmae = np.mean(relative_errors) * 100
    correlation = np.corrcoef(predictions, targets)[0, 1]

    metrics = {
        "mae": mae,
        "medae": medae,
        "rmae": rmae,
        "rmedae": rmedae,
        "correlation": correlation,
        "n_samples": len(test_df),
    }

    logger.info(
        f"Fold {fold_name} ({test_type}) metrics: MAE={mae:.4f}, MedAE={medae:.4f}, "
        f"RMAE={rmae:.2f}%, RMedAE={rmedae:.2f}%, Corr={correlation:.4f}, N={len(test_df)}"
    )

    return metrics


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def aggregate_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all folds.

    Returns:
        aggregated_metrics with mean/std for each metric type
    """
    # Separate original and baseline metrics
    original_metrics = {}
    baseline_metrics = {}

    for fold_result in fold_results:
        # Original test set metrics
        for key, value in fold_result["test_original_metrics"].items():
            if key not in original_metrics:
                original_metrics[key] = []
            original_metrics[key].append(value)

        # Baseline test set metrics
        for key, value in fold_result["test_baseline_metrics"].items():
            if key not in baseline_metrics:
                baseline_metrics[key] = []
            baseline_metrics[key].append(value)

    # Compute mean and std for original
    aggregated = {"original": {}, "baseline": {}}

    for key, values in original_metrics.items():
        aggregated["original"][f"{key}_mean"] = float(np.mean(values))
        aggregated["original"][f"{key}_std"] = float(np.std(values))
        aggregated["original"][f"{key}_all"] = values

    for key, values in baseline_metrics.items():
        aggregated["baseline"][f"{key}_mean"] = float(np.mean(values))
        aggregated["baseline"][f"{key}_std"] = float(np.std(values))
        aggregated["baseline"][f"{key}_all"] = values

    # Calculate improvement: (baseline - original) / baseline * 100
    # Positive values mean original (with true adduct) is better
    for metric in ["mae", "medae", "rmae", "rmedae"]:
        if (
            f"{metric}_mean" in aggregated["original"]
            and f"{metric}_mean" in aggregated["baseline"]
        ):
            baseline_val = aggregated["baseline"][f"{metric}_mean"]
            original_val = aggregated["original"][f"{metric}_mean"]
            if baseline_val != 0:
                improvement = ((baseline_val - original_val) / baseline_val) * 100
                aggregated[f"{metric}_improvement_pct"] = improvement

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Leave-One-Adduct-Out Cross-Validation for adduct generalization"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config JSON file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to GraphCCS data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/adduct_cv/",
        help="Output directory for all folds",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-adducts",
        type=str,
        nargs="*",
        default=None,
        help="Adducts to skip (optional)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum number of samples required for an adduct to be included in CV",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to base model checkpoint for transfer learning (optional)",
    )
    parser.add_argument(
        "--freeze-layers",
        action="store_true",
        default=False,
        help="Freeze GNN layers during transfer learning (only train head, requires --base-model)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name for logging (optional)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional, defaults to CV timestamp)",
    )

    args = parser.parse_args()

    # Load base config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    # Configure transfer learning settings if base model provided
    if args.base_model:
        if not os.path.exists(args.base_model):
            raise FileNotFoundError(f"Base model not found: {args.base_model}")
        logger.info(f"Using base model for transfer learning: {args.base_model}")
        config["transfer_learn"] = True
        if args.freeze_layers:
            config["freeze_layers"] = True
            logger.info("Freeze layers enabled: Only head will be trained")
        else:
            logger.info("Freeze layers disabled: Full model will be fine-tuned")
    else:
        logger.info("No base model provided: Training from scratch")
        config["transfer_learn"] = False
        if args.freeze_layers:
            logger.warning(
                "--freeze-layers specified but no --base-model provided, ignoring"
            )

    # Initialize wandb parent run for CV experiment
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        run_name = (
            args.wandb_run_name
            or f"adduct_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        tags = ["cross-validation", "adduct-generalization"]
        if args.base_model:
            tags.append("transfer-learning")
        else:
            tags.append("from-scratch")

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "config_file": args.config,
                "data_path": args.data_path,
                "base_model": args.base_model if args.base_model else "none",
                "freeze_layers": args.freeze_layers if args.base_model else False,
                "valid_fraction": args.valid_fraction,
                "random_seed": args.random_seed,
                "min_samples": args.min_samples,
                "cv_type": "leave_one_adduct_out",
            },
            tags=tags,
        )
        logger.info(f"Initialized W&B run: {wandb_run.name}")
    elif args.wandb_project and not WANDB_AVAILABLE:
        logger.warning("W&B project specified but wandb not available")

    # Load GraphCCS data
    logger.info(f"Loading GraphCCS data from {args.data_path}")
    df = load_graphccs_data(args.data_path)

    # Get unique adducts with sufficient samples
    adduct_counts = df["Adduct"].value_counts()
    unique_adducts = sorted(
        [adduct for adduct, count in adduct_counts.items() if count >= args.min_samples]
    )

    if args.skip_adducts:
        unique_adducts = [a for a in unique_adducts if a not in args.skip_adducts]
        logger.info(f"Skipping adducts: {args.skip_adducts}")

    logger.info(f"Running CV with {len(unique_adducts)} folds: {unique_adducts}")
    logger.info(
        f"Filtered out {len(adduct_counts) - len(unique_adducts)} adducts with < {args.min_samples} samples"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run CV
    fold_results = []
    for i, test_adduct in enumerate(unique_adducts, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {i}/{len(unique_adducts)}: Held-out adduct = {test_adduct}")
        logger.info(f"{'='*80}\n")

        fold_name = (
            test_adduct.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("+", "plus")
            .replace("-", "minus")
        )
        fold_dir = output_dir / fold_name

        # if fold_dir exists, check if baseline subdir exists to skip
        if fold_dir.exists() and (fold_dir / "baseline").exists():
            logger.info(
                f"Fold directory {fold_dir} with baseline results exists, skipping fold"
            )
            continue

        # Create fold data splits
        train_df, valid_df, test_original_df, test_baseline_df = create_fold_data(
            df=df,
            test_adduct=test_adduct,
            valid_fraction=args.valid_fraction,
            random_seed=args.random_seed,
        )

        # Save fold CSVs
        train_path, valid_path, test_original_path, test_baseline_path = save_fold_csvs(
            train_df=train_df,
            valid_df=valid_df,
            test_original_df=test_original_df,
            test_baseline_df=test_baseline_df,
            fold_dir=fold_dir,
        )

        # Train model with transfer learning
        model, adduct_feat_dim = train_fold(
            config=config,
            train_path=train_path,
            valid_path=valid_path,
            test_original_path=test_original_path,
            fold_name=fold_name,
            output_dir=output_dir,
            base_model_path=args.base_model,
        )

        # Evaluate on held-out adduct (original)
        test_original_metrics = evaluate_fold(
            model=model,
            test_csv=test_original_path,
            config=config,
            adduct_feat_dim=adduct_feat_dim,
            fold_name=fold_name,
            fold_dir=fold_dir,
            test_type="original",
        )

        # Evaluate on baseline adduct ([M+H]+ or [M-H]-)
        test_baseline_metrics = evaluate_fold(
            model=model,
            test_csv=test_baseline_path,
            config=config,
            adduct_feat_dim=adduct_feat_dim,
            fold_name=fold_name,
            fold_dir=fold_dir,
            test_type="baseline",
        )

        # Store results
        fold_result = {
            "fold": i,
            "test_adduct": test_adduct,
            "baseline_adduct": test_baseline_df["Adduct"].iloc[0],
            "train_size": len(train_df),
            "valid_size": len(valid_df),
            "test_size": len(test_original_df),
            "test_original_metrics": test_original_metrics,
            "test_baseline_metrics": test_baseline_metrics,
        }
        fold_results.append(fold_result)

        logger.info(f"Fold {i} original metrics: {test_original_metrics}")
        logger.info(f"Fold {i} baseline metrics: {test_baseline_metrics}")

        # Log fold metrics to wandb
        if wandb_run:
            fold_log = {}
            # Original test metrics
            for k, v in test_original_metrics.items():
                fold_log[f"fold_{i}/original_{k}"] = v
            # Baseline test metrics
            for k, v in test_baseline_metrics.items():
                fold_log[f"fold_{i}/baseline_{k}"] = v
            # Metadata
            fold_log[f"fold_{i}/test_adduct"] = test_adduct
            fold_log[f"fold_{i}/train_size"] = len(train_df)
            fold_log[f"fold_{i}/valid_size"] = len(valid_df)
            fold_log[f"fold_{i}/test_size"] = len(test_original_df)
            wandb.log(fold_log)

    # Aggregate results
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATING RESULTS")
    logger.info(f"{'='*80}\n")

    aggregated = aggregate_results(fold_results)

    # Save results
    results_file = output_dir / "cv_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "data_path": args.data_path,
        "base_model": args.base_model if args.base_model else "none",
        "transfer_learn": args.base_model is not None,
        "freeze_layers": args.freeze_layers if args.base_model else False,
        "num_folds": len(unique_adducts),
        "min_samples": args.min_samples,
        "fold_results": fold_results,
        "aggregated_metrics": aggregated,
    }

    # Convert numpy types to native Python types for JSON serialization
    results = convert_to_json_serializable(results)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info("\nORIGINAL ADDUCT TEST SET:")
    for key, value in aggregated["original"].items():
        if key.endswith("_mean") or key.endswith("_std"):
            logger.info(f"  {key}: {value:.4f}")

    logger.info("\nBASELINE ADDUCT TEST SET ([M+H]+ / [M-H]-):")
    for key, value in aggregated["baseline"].items():
        if key.endswith("_mean") or key.endswith("_std"):
            logger.info(f"  {key}: {value:.4f}")

    logger.info("\nIMPROVEMENT (Original vs Baseline):")
    for key, value in aggregated.items():
        if key.endswith("_improvement_pct"):
            logger.info(f"  {key}: {value:+.2f}%")

    logger.info("=" * 80 + "\n")

    # Log aggregated results to wandb
    if wandb_run:
        # Log summary metrics
        summary_metrics = {}
        for k, v in aggregated["original"].items():
            if k.endswith("_mean") or k.endswith("_std"):
                summary_metrics[f"cv/original_{k}"] = v
        for k, v in aggregated["baseline"].items():
            if k.endswith("_mean") or k.endswith("_std"):
                summary_metrics[f"cv/baseline_{k}"] = v
        for k, v in aggregated.items():
            if k.endswith("_improvement_pct"):
                summary_metrics[f"cv/{k}"] = v

        wandb.log(summary_metrics)

        # Create and log results table
        table_data = []
        for fold_result in fold_results:
            row = [
                fold_result["fold"],
                fold_result["test_adduct"],
                fold_result["baseline_adduct"],
                fold_result["train_size"],
                fold_result["valid_size"],
                fold_result["test_size"],
            ]
            # Add original metrics
            for metric in ["mae", "medae", "rmae", "rmedae", "correlation"]:
                row.append(fold_result["test_original_metrics"][metric])
            # Add baseline metrics
            for metric in ["mae", "medae", "rmae", "rmedae", "correlation"]:
                row.append(fold_result["test_baseline_metrics"][metric])
            table_data.append(row)

        columns = [
            "Fold",
            "Test Adduct",
            "Baseline Adduct",
            "Train Size",
            "Valid Size",
            "Test Size",
            "Orig MAE",
            "Orig MedAE",
            "Orig RMAE",
            "Orig RMedAE",
            "Orig Corr",
            "Base MAE",
            "Base MedAE",
            "Base RMAE",
            "Base RMedAE",
            "Base Corr",
        ]

        results_table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"cv_results_table": results_table})

        # Log results file as artifact
        artifact = wandb.Artifact(
            name="cv_results",
            type="results",
            description="Leave-One-Adduct-Out CV results for adduct generalization",
        )
        artifact.add_file(str(results_file))
        wandb.log_artifact(artifact)

        logger.info("Logged results to W&B")
        wandb.finish()

    logger.info("Leave-One-Adduct-Out CV complete!")


if __name__ == "__main__":
    main()
