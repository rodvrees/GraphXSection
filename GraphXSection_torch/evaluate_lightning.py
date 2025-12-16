"""PyTorch Lightning evaluation module for GraphXSection_torch."""

import logging
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

from torch.utils.data import DataLoader

from GraphXSection_torch.lightning_modules import GraphXSectionLightningModule
from GraphXSection_torch.data_extract import GraphXSectionDataset
from GraphXSection_torch.lightning_data import GraphXSectionDataModule

logger = logging.getLogger(__name__)


def plot_lightning_results(
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    absolute_errors: pd.Series,
    relative_errors: pd.Series,
) -> None:
    """Plot the predicted vs. actual CCS values for Lightning model results."""
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
    plt.savefig(
        f"{config['output_path']}/{config['run_name']}_lightning_predictions_plot.png"
    )
    plt.close()


def evaluate_lightning_model(
    lightning_module,
    data_module_or_dataset,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a Lightning model on the test set.

    Accepts either:
      - (lightning_module, data_module, test_df, config) where data_module has .test_dataloader()
      - (lightning_module, test_dataset, test_df, config) where test_dataset is a Dataset

    Returns:
        Tuple containing predictions and median relative error
    """
    logger.info("Evaluating Lightning model on test set")

    # If a DataModule was passed in as data_module_or_dataset, use its test_dataloader
    dataloader = None

    # Detect DataModule-like object (has test_dataloader method)
    if hasattr(data_module_or_dataset, "test_dataloader") and callable(
        getattr(data_module_or_dataset, "test_dataloader")
    ):
        logger.info("Received DataModule for evaluation. Using its test_dataloader().")
        data_module = data_module_or_dataset
        try:
            dataloader = data_module.test_dataloader()
        except Exception as e:
            logger.exception("Failed to get test_dataloader() from DataModule: %s", e)
            raise

    else:
        # We were passed a raw dataset; build a DataLoader with correct collate_fn
        from GraphXSection_torch.data_extract import collate_fn

        logger.info(
            "Received raw Dataset for evaluation. Building DataLoader with collate_fn."
        )

        test_dataset = data_module_or_dataset

        # If test_dataset is accidentally a tuple/list, unwrap it
        if isinstance(test_dataset, (tuple, list)):
            logger.warning(
                "test_dataset is a tuple/list; using first element as dataset"
            )
            test_dataset = test_dataset[0]

        batch_size = config.get("test_batch_size") or config.get("model", {}).get(
            "batch_size", 16
        )
        num_workers = config.get("num_workers", 0)

        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Set model to evaluation mode
    lightning_module.eval()

    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator="auto", devices="auto", logger=False, enable_progress_bar=True
    )

    # Get predictions using trainer.predict
    predictions_list = trainer.predict(lightning_module, dataloader)

    # Collect all predictions and targets
    all_predictions = []
    all_targets = []

    test_dataloader = dataloader
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to same device as model
            device = next(lightning_module.parameters()).device
            for key in batch:
                if key == "graph":
                    batch[key] = batch[key].to(device)
                else:
                    batch[key] = batch[key].to(device)

            # Get predictions
            predictions = lightning_module(batch)
            targets = batch["target"]

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all predictions and ensure 1D shape
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    # Flatten (some models output shape [N, 1])
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

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

    # Log to wandb if Lightning logger is WandbLogger
    if hasattr(lightning_module.logger, "experiment"):
        try:
            lightning_module.logger.experiment.log(
                {
                    "test_mean_absolute_error": mae,  # Match original naming
                    "test_medAE": medae,  # Match original naming
                    "test_RMAE": rmae / 100,  # Convert back to decimal like original
                    "test_RMedAE": rmedae
                    / 100,  # Convert back to decimal like original
                    "test_corr": correlation,  # Match original naming
                }
            )
            logger.info("Logged evaluation metrics to wandb")
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    # Also check if wandb is available globally (like original does)
    try:
        import wandb

        if wandb.run is not None and config.get("wandb", {}).get("use", False):
            wandb.log(
                {
                    "test_mean_absolute_error": mae,
                    "test_medAE": medae,
                    "test_RMAE": rmae / 100,
                    "test_RMedAE": rmedae / 100,
                    "test_corr": correlation,
                }
            )
    except:
        pass  # Ignore if wandb is not available

    # Add predictions to test dataframe and save
    test_df_copy = test_df.copy()
    test_df_copy["Predicted CCS"] = predictions
    test_df_copy["Absolute Error"] = absolute_errors
    test_df_copy["Relative Error"] = relative_errors

    # Save predictions
    output_path = (
        f"{config['output_path']}/{config['run_name']}_lightning_test_predictions.csv"
    )

    os.makedirs(config["output_path"], exist_ok=True)
    test_df_copy.to_csv(output_path, index=False)
    logger.info(f"Lightning predictions saved to {output_path}")

    # Create plot if enabled
    if config.get("create_plots", True):
        plot_lightning_results(
            test_df_copy, config, pd.Series(absolute_errors), pd.Series(relative_errors)
        )

    return predictions, rmedae
