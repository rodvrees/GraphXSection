"""Evaluation module for GraphXSection_torch - PyTorch implementation."""

import logging
import wandb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from GraphXSection_torch.data_extract import GraphXSectionDataset, collate_fn

logger = logging.getLogger(__name__)


def plot(
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    absolute_errors: pd.Series,
    relative_errors: pd.Series,
) -> None:
    """Plot the predicted vs. actual CCS values"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()

    # Plot a diagonal line representing perfect predictions
    ax.plot(
        [min(test_df["CCS"]), max(test_df["CCS"])],
        [min(test_df["CCS"]), max(test_df["CCS"])],
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    ax.set(xlabel="Observed CCS", ylabel="Predicted CCS")

    # Normalize monoisotopic mass for color scaling if available
    if "monoisotopic mass" in test_df.columns:
        norm = plt.Normalize(
            test_df["monoisotopic mass"].min(), test_df["monoisotopic mass"].max()
        )
        scatter = ax.scatter(
            x=test_df["CCS"],
            y=test_df["Predicted CCS"],
            c=test_df["monoisotopic mass"],
            norm=norm,
            s=6,
            cmap="viridis_r",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Monoisotopic mass")
    else:
        ax.scatter(
            x=test_df["CCS"],
            y=test_df["Predicted CCS"],
            s=6,
            alpha=0.7,
        )

    RMAE = relative_errors.mean() * 100
    RmedAE = relative_errors.median() * 100
    corr = test_df["CCS"].corr(test_df["Predicted CCS"])

    plt.title("RMAE: {:.2f}, RmedAE: {:.2f}, Corr: {:.2f}".format(RMAE, RmedAE, corr))
    plt.savefig(f"{config['output_path']}/{config['run_name']}_predictions_plot.png")
    plt.close()


def evaluate(
    model: nn.Module,
    test_dataset: GraphXSectionDataset,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    batched=True,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate the model on the test set

    Args:
        model: Trained PyTorch model
        test_dataset: Test dataset
        test_df: Test dataframe
        config: Configuration dictionary
        batched: Whether to use batched evaluation

    Returns:
        Tuple containing predictions and median relative error
    """
    logger.info("Evaluating model on test set")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create data loader
    if batched:
        batch_size = config["model"].get("batch_size", 32)
    else:
        batch_size = len(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
    )

    # Collect predictions
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            for key in batch:
                if key == "graph":
                    batch[key] = batch[key].to(device)
                else:
                    batch[key] = batch[key].to(device)

            # Get predictions
            predictions = model(batch)
            targets = batch["target"]

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate metrics
    absolute_errors = np.abs(predictions - targets)
    relative_errors = absolute_errors / targets

    # Calculate statistics
    mae = np.mean(absolute_errors)
    medae = np.median(absolute_errors)
    rmae = np.mean(relative_errors) * 100
    rmedae = np.median(relative_errors) * 100
    correlation = np.corrcoef(predictions, targets)[0, 1]

    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test MedAE: {medae:.4f}")
    logger.info(f"Test RMAE: {rmae:.2f}%")
    logger.info(f"Test RMedAE: {rmedae:.2f}%")
    logger.info(f"Test Correlation: {correlation:.4f}")

    # Log to wandb if enabled
    if config.get("wandb", {}).get("use", False):
        wandb.log(
            {
                "test_mae": mae,
                "test_medae": medae,
                "test_rmae": rmae,
                "test_rmedae": rmedae,
                "test_correlation": correlation,
            }
        )

    # Add predictions to test dataframe and save
    test_df_copy = test_df.copy()
    test_df_copy["Predicted CCS"] = predictions
    test_df_copy["Absolute Error"] = absolute_errors
    test_df_copy["Relative Error"] = relative_errors

    # Save predictions
    output_path = f"{config['output_path']}/{config['run_name']}_test_predictions.csv"
    test_df_copy.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    # Create plot if enabled
    if config.get("create_plots", True):
        plot(
            test_df_copy, config, pd.Series(absolute_errors), pd.Series(relative_errors)
        )

    return predictions, rmedae


def evaluate_eddy(
    model: nn.Module,
    test_data: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """
    Evaluate the Eddy model on the test set

    Args:
        model: Trained Eddy model
        test_data: Test data with embeddings
        config: Configuration dictionary

    Returns:
        Tuple containing predictions and median relative error
    """
    logger.info("Evaluating Eddy model on test set")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Prepare test data
    test_embeddings = torch.tensor(
        np.stack(test_data["embeddings"]), dtype=torch.float32
    )
    test_adduct_ohe = torch.tensor(
        np.stack(test_data["Adduct_OHE"]), dtype=torch.float32
    )
    test_targets = torch.tensor(test_data["CCS"].values, dtype=torch.float32)

    # Create data loader
    from torch.utils.data import TensorDataset, DataLoader

    test_dataset = TensorDataset(test_embeddings, test_adduct_ohe, test_targets)
    test_loader = DataLoader(
        test_dataset, batch_size=config["model"].get("batch_size", 32), shuffle=False
    )

    # Collect predictions
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for embeddings, adduct_ohe, targets in test_loader:
            embeddings = embeddings.to(device)
            adduct_ohe = adduct_ohe.to(device)
            targets = targets.to(device)

            # Get predictions
            predictions = model(embeddings, adduct_ohe)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate metrics
    absolute_errors = np.abs(predictions - targets)
    relative_errors = absolute_errors / targets

    # Calculate statistics
    mae = np.mean(absolute_errors)
    medae = np.median(absolute_errors)
    rmae = np.mean(relative_errors) * 100
    rmedae = np.median(relative_errors) * 100
    correlation = np.corrcoef(predictions, targets)[0, 1]

    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test MedAE: {medae:.4f}")
    logger.info(f"Test RMAE: {rmae:.2f}%")
    logger.info(f"Test RMedAE: {rmedae:.2f}%")
    logger.info(f"Test Correlation: {correlation:.4f}")

    # Log to wandb if enabled
    if config.get("wandb", {}).get("use", False):
        wandb.log(
            {
                "test_mae": mae,
                "test_medae": medae,
                "test_rmae": rmae,
                "test_rmedae": rmedae,
                "test_correlation": correlation,
            }
        )

    # Create dataframe with predictions
    test_df = pd.DataFrame(
        {
            "CCS": targets,
            "Predicted CCS": predictions,
            "Absolute Error": absolute_errors,
            "Relative Error": relative_errors,
        }
    )

    # Save predictions
    output_path = (
        f"{config['output_path']}/{config['run_name']}_eddy_test_predictions.csv"
    )
    test_df.to_csv(output_path, index=False)
    logger.info(f"Eddy predictions saved to {output_path}")

    # Create plot if enabled
    if config.get("create_plots", True):
        plot(test_df, config, pd.Series(absolute_errors), pd.Series(relative_errors))

    return predictions, rmedae
