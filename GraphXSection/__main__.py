"""Graph neural networks for predicting the collisional cross-section of small molecules."""

import logging
import click
from rich.logging import RichHandler
from rich.console import Console
import json
from pathlib import Path
import wandb
from typing import Dict, Any, Tuple


LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

console = Console(record=True)
logger = logging.getLogger(__name__)


def _setup_logging(log_level: str) -> None:
    """
    Set up logging with the specified log level.

    Args:
        log_level (str): The log level for logging, must be one of
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    print("Setting up logging with log level", log_level)
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOGGING_LEVELS[log_level],
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                console=console,
                show_level=True,
                show_path=True,
                show_time=True,
            )
        ],
    )
    logger.info("Logging setup complete")
    logger.info(
        "Effective logging level is {}".format(
            logging.getLevelName(logger.getEffectiveLevel())
        )
    )


def _setup_wandb(config: Dict[str, Any]) -> None:
    """
    Set up Weights and Biases for experiment tracking.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    logger.info("Setting up Weights and Biases")
    wandb.init(
        project=config["wandb"]["project"], name=config["run_name"], config=config
    )


def _parse_config(config_path: Path) -> Dict[str, Any]:
    """
    Parse the configuration file.

    Args:
        config_path (Path): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        ValueError: If the config file is not a JSON file.
    """
    if Path(config_path).suffix.lower() != ".json":
        raise ValueError("Config file must be a JSON file.")

    with open(config_path, "r") as f:
        config = json.load(f)
        logger.debug(config)
    return config


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--log-level", type=click.Choice(LOGGING_LEVELS.keys()), default="INFO")
def main(config: str, log_level: str) -> None:
    """
    Main entry point for GraphXSection.

    Args:
        config (str): Path to the JSON configuration file.
        log_level (str): Log level for logging.
    """

    _setup_logging(log_level)

    # Import after logging setup to avoid tensorflow logging conflicts
    from data_extract import data_extract
    from encode_graph import get_encoder
    from train import train_GraphXSection
    from evaluate import evaluate
    from saliency_mapping import get_saliency_maps

    logger.info("Starting GraphXSection")
    config = _parse_config(config)

    if config.get("wandb", False) and config["wandb"].get("use", False):
        _setup_wandb(config)

    train_data, valid_data, test_data, test_df, valid_df = data_extract(config)
    mol_encoder = get_encoder(config)
    model = train_GraphXSection(train_data, valid_data, mol_encoder, config)

    # Evaluate with best model
    if config["use_best_model"]:
        logger.info("Evaluating with best model")
        model.load_weights(config["mcp"]["mcp_path"])
        logger.debug(f"Model loaded from {config['mcp']['mcp_path']}")
        test_x_mol, RMedAE = evaluate(model, test_data, test_df, mol_encoder, config)
    else:
        test_x_mol, RMedAE = evaluate(model, test_data, test_df, mol_encoder, config)

    # Save model
    if config["save_model"]:
        logger.info(
            f"Saving model to {config['output_path']}/{config['run_name']}_model.h5"
        )

        # TODO: Probably not working?
        model.save(
            f"{config['output_path']}/{config['run_name']}_model.tf",
            save_format="tf",
        )
        logger.info("Model saved")

    # Create saliency maps
    if config.get("saliency_mapping", False):
        import pickle

        logger.info("Generating saliency maps")
        saliency_maps = get_saliency_maps(model, test_x_mol)
        with open(
            f"{config['output_path']}/{config['run_name']}_saliency_maps.pkl", "wb"
        ) as f:
            pickle.dump(saliency_maps, f)
        logger.debug(f"Saliency maps shape: {saliency_maps.shape}")
        logger.info("Saliency maps generated")

    logger.info("GraphXSection Done")


if __name__ == "__main__":
    main()
