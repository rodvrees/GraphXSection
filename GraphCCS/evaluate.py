"""Evaluate the performance of the model on the test set"""

import logging
import wandb
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Tuple

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

    # Normalize monoisotopic mass for color scaling
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

    RMAE = relative_errors.mean() * 100
    RmedAE = relative_errors.median() * 100
    corr = test_df["CCS"].corr(test_df["Predicted CCS"])

    plt.title("RMAE: {:.2f}, RmedAE: {:.2f}, Corr: {:.2f}".format(RMAE, RmedAE, corr))
    plt.savefig(f"{config['output_path']}/{config['run_name']}_predictions_plot.png")
    plt.close()


def evaluate(
    model: tf.keras.Model,
    test_data: Dict[str, Any],
    test_df: pd.DataFrame,
    mol_encoder: Any,
    config: Dict[str, Any],
) -> Tuple[tf.data.Dataset, float]:
    """Evaluate the model on the test set

    Args:
        model (tf.keras.Model): Trained model.
        test_data (Dict[str, Any]): Test data.
        test_df (pd.DataFrame): Test dataframe.
        mol_encoder (Any): Molecular encoder.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[tf.data.Dataset, float]: Test dataset and median relative error."""

    logger.info("Evaluating model on test set")
    test_smiles = test_data["smiles"]
    test_x_adduct = test_data["adduct_OHE"]
    test_x_mol = mol_encoder(test_smiles)
    y_test = test_data["y"]

    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(
        ((test_x_mol, test_x_adduct), y_test)
    )
    test_dataset = test_dataset.batch(len(test_smiles)).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    test_dataset = test_dataset.map(
        lambda x_and_x_adduct, y: (
            x_and_x_adduct[0].update({"_x_adduct": x_and_x_adduct[1]}),
            y,
        )
    )

    # Evaluate the model on the test dataset
    results = model.evaluate(test_dataset)
    logger.info(f"Test loss: {results[0]}")
    logger.info(f"Test accuracy: {results[1]}")

    # Log results to wandb
    if config["wandb"].get("use", False):
        wandb.log({"test_loss": results[0], "test_mean_absolute_error": results[1]})

    logger.info("Generating predictions on test set")
    predictions = model.predict(test_dataset)
    test_df["Predicted CCS"] = predictions

    # Calculate the median absolute and relative errors
    absolute_errors = abs(test_df["CCS"] - test_df["Predicted CCS"])
    relative_errors = absolute_errors / test_df["CCS"]

    logger.info(f"Median absolute error: {absolute_errors.median()}")
    logger.info(f"Mean relative error: {relative_errors.mean() * 100:.2f}%")
    logger.info(f"Median relative error: {relative_errors.median() * 100:.2f}%")

    # Calculate Pearson correlation
    corr = test_df["CCS"].corr(test_df["Predicted CCS"])
    logger.info(f"Pearson correlation: {corr}")

    if config.get("wandb", False):
        if config["wandb"].get("use", False):
            wandb.log(
                {
                    "test_medAE": absolute_errors.median(),
                    "test_RMAE": relative_errors.mean(),
                    "test_RMedAE": relative_errors.median(),
                    "test_corr": corr,
                }
            )

    if config.get("save_predictions", False):
        test_df.to_csv(
            f"{config['output_path']}/{config['run_name']}_test_predictions.csv",
            index=False,
        )

    if config.get("save_plot", False):
        logger.info("Generating plots")
        plot(test_df, config, absolute_errors, relative_errors)
    logger.info("Model evaluation complete")

    return test_dataset, relative_errors.median()
