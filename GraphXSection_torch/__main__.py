"""Graph neural networks for predicting the collisional cross-section of small molecules - PyTorch implementation."""

import logging
import copy
import sys
import click
from rich.logging import RichHandler
from rich.console import Console
import json
from pathlib import Path
import wandb

import torch
from typing import Dict, Any, Tuple


LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

console = Console(record=True)
log = logging.getLogger(__name__)


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
    log.info("Logging setup complete")
    log.info(
        "Effective logging level is {}".format(
            logging.getLevelName(log.getEffectiveLevel())
        )
    )


def _setup_wandb(config: Dict[str, Any]) -> None:
    """
    Set up Weights and Biases for experiment tracking.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    log.info("Setting up Weights and Biases")

    # Ensure W&B runs in online mode for sweeps
    import os

    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "online"
        log.info("Set W&B to online mode for cloud syncing")

    # Ensure wandb config exists
    if "wandb" not in config:
        config["wandb"] = {}

    # Set default project if not specified
    project = config["wandb"].get("project", "graphxsection-sweeps")
    run_name = config.get("run_name", "graphxsection-run")

    log.info(f"Initializing W&B with project: {project}, run: {run_name}")

    wandb.init(
        project=project,
        name=run_name,
        config=config,
        mode="online",  # Explicitly set online mode
        settings=wandb.Settings(init_timeout=300),
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
        log.debug(config)
    return config


def _overwrite_with_wandb(config_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overwrite the config with wandb settings.

    Args:
        config_json (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Updated configuration dictionary.
    """
    log.info("Overwriting config with wandb settings")
    log.debug("Original config:")
    log.debug(config_json)

    # Start with a copy of the original config
    config = copy.deepcopy(config_json)

    # Check if we have W&B sweep config available
    try:
        import wandb

        if hasattr(wandb, "config") and wandb.config:
            sweep_config = wandb.config
            log.info(f"Found W&B sweep config with {len(sweep_config)} parameters")

            # Ensure wandb section exists in config
            if "wandb" not in config:
                config["wandb"] = {}

            for key, value in sweep_config.items():
                if "." in key:
                    keys = key.split(".")
                    sub_config = config
                    for k in keys[:-1]:
                        if k not in sub_config:
                            sub_config[k] = {}
                        sub_config = sub_config[k]
                    sub_config[keys[-1]] = value
                    log.debug(f"Set {key} = {value}")
                else:
                    config[key] = value
                    log.debug(f"Set {key} = {value}")

            log.info("Updated config with wandb settings")
            log.debug("Updated config:")
            log.debug(config)
    except Exception as e:
        log.debug(f"No W&B sweep config available: {e}")

    return config


def _merge_flat_overrides(base: dict, overrides: dict) -> dict:
    """
    Merge flattened overrides (like 'model.learning_rate': 1e-3) into base nested dict.
    Non-dotted keys overwrite top-level keys.
    """
    cfg = copy.deepcopy(base)
    if not overrides:
        return cfg
    for k, v in overrides.items():
        if not isinstance(k, str):
            cfg[k] = v
            continue
        if "." in k:
            parts = k.split(".")
            cur = cfg
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = v
        else:
            # top-level override
            cfg[k] = v
    return cfg


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--log-level", type=click.Choice(LOGGING_LEVELS.keys()), default="INFO")
@click.option(
    "-f",
    "--feature-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to feature configuration JSON file for custom vocabularies",
)
@click.option(
    "--base-model",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to pre-trained model checkpoint (.pth file) for transfer learning",
)
@click.option(
    "--transfer-learn",
    is_flag=True,
    default=False,
    help="Enable transfer learning mode - load base model and fine-tune on new dataset",
)
@click.option(
    "--inference",
    is_flag=True,
    default=False,
    help="Run inference-only: load saved model and run predictions on a test CSV",
)
@click.option(
    "--test-csv",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to CSV file to run inference on (required with --inference)",
)
@click.option(
    "--inference-model",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to saved model checkpoint (.pth) to use for inference (required with --inference)",
)
def main(
    config: str,
    log_level: str,
    feature_config: str = None,
    base_model: str = None,
    transfer_learn: bool = False,
    inference: bool = False,
    test_csv: str | None = None,
    inference_model: str | None = None,
) -> None:
    """
    Main entry point for GraphXSection_torch.

    Args:
        config (str): Path to the JSON configuration file.
        log_level (str): Log level for logging.
        feature_config (str): Path to feature configuration JSON file.
        base_model (str): Path to pre-trained model checkpoint for transfer learning.
        transfer_learn (bool): Enable transfer learning mode.
    """

    _setup_logging(log_level)

    # Validate transfer learning options
    if transfer_learn and base_model is None:
        log.error("Transfer learning mode requires --base-model to be specified")
        raise ValueError(
            "--base-model must be provided when --transfer-learn is enabled"
        )

    if base_model and not transfer_learn:
        log.warning(
            "--base-model specified but --transfer-learn not enabled. Use --transfer-learn to enable transfer learning mode."
        )

    if transfer_learn:
        log.info(f"Transfer learning mode enabled. Base model: {base_model}")

    # Validate inference options
    if inference:
        if test_csv is None and config.get("test_data_path") is None:
            log.error(
                "Inference mode requires --test-csv or config['test_data_path'] to be set"
            )
            raise ValueError("--test-csv must be provided when --inference is enabled")
        if inference_model is None and base_model is None:
            log.error(
                "Inference mode requires --inference-model or --base-model to be provided"
            )
            raise ValueError(
                "--inference-model or --base-model must be provided when --inference is enabled"
            )
        log.info(
            f"Inference mode enabled. Test CSV: {test_csv or config.get('test_data_path')}, model: {inference_model or base_model}"
        )

    from GraphXSection_torch.train_lightning import (
        train_lightning_GraphXSection,
    )
    from GraphXSection_torch.evaluate_lightning import (
        evaluate_lightning_model,
    )

    from GraphXSection_torch.saliency_mapping import get_saliency_maps

    log.info("Starting GraphXSection_torch")
    config = _parse_config(config)

    # Check for W&B usage - handle both nested config and sweep parameters
    wandb_enabled = False
    if config.get("wandb", False) and config["wandb"].get("use", False):
        wandb_enabled = True

    # Also check if we're in a W&B sweep (sweep parameters are available)
    try:
        import wandb

        if hasattr(wandb, "config") and wandb.config:
            # We're in a sweep, check if wandb.use is set to True
            if wandb.config.get("wandb.use", False):
                wandb_enabled = True
                log.info("W&B sweep detected - enabling W&B logging")
    except:
        pass

    if wandb_enabled:
        _setup_wandb(config)

    config = _overwrite_with_wandb(config)
    log.debug("Final config:")
    log.debug(config)

    # Graphamole-style dataset + splits + loaders
    import pandas as pd
    from GraphXSection_torch.datasets import (
        build_dataset_from_df,
        split_train_val_test,
        build_datasets_from_separate_files,
        build_dataset_from_csv,
    )

    # Check if separate dataset files are specified
    if all(
        key in config
        for key in ["train_data_path", "valid_data_path", "test_data_path"]
    ):
        log.info("Using separate train/valid/test CSV files")
        (
            train_data,
            valid_data,
            test_data,
            df_train,
            df_valid,
            df_test,
            adduct_feature_sizes,
        ) = build_datasets_from_separate_files(
            train_csv_path=config["train_data_path"],
            valid_csv_path=config["valid_data_path"],
            test_csv_path=config["test_data_path"],
            smiles_col=config.get("smiles_col", "SMILES"),
            adduct_col=config.get("adduct_col", "Adduct"),
            use_adduct_descriptors=config.get("use_adduct_descriptors", True),
            one_hot_encode_feats=config.get("one_hot_encode_feats", False),
            feature_config_path=Path(feature_config) if feature_config else None,
        )
    else:
        log.info("Using single dataset file with automatic splitting")
        df_all = pd.read_csv(config["data_path"])  # Keep raw df for saving/plots

        full_dataset, adduct_feature_sizes = build_dataset_from_df(
            df_all,
            smiles_col=config.get("smiles_col", "SMILES"),
            adduct_col=config.get("adduct_col", "Adduct"),
            use_adduct_descriptors=config.get("use_adduct_descriptors", True),
            one_hot_encode_feats=config.get("one_hot_encode_feats", False),
            feature_config_path=Path(feature_config) if feature_config else None,
        )

        train_data, valid_data, test_data, df_train, df_valid, df_test = (
            split_train_val_test(
                full_dataset,
                df_all,
                valid_fraction=config.get("valid_split", 0.1),
                test_fraction=config.get("test_split", 0.1),
                stratify_col=config.get("adduct_col", "Adduct"),
                seed=42,
            )
        )

    # PyTorch Lightning training
    # If inference-only mode is requested, skip training and run evaluation on provided CSV
    if inference:
        from GraphXSection_torch.lightning_modules import GraphXSectionLightningModule
        from GraphXSection_torch.lightning_data import GraphXSectionDataModule

        # Build dataset from provided test CSV (or fallback to config)
        test_csv_path = test_csv or config.get("test_data_path")
        test_dataset, test_df, _ = build_dataset_from_csv(
            test_csv_path,
            smiles_col=config.get("smiles_col", "SMILES"),
            adduct_col=config.get("adduct_col", "Adduct"),
            use_adduct_descriptors=config.get("use_adduct_descriptors", True),
        )

        # Create a temporary DataModule from existing datasets so we can extract
        # node/edge/global feature sizes and propagate them into the config.
        # train_data/valid_data/test_data are available from earlier dataset construction.
        try:
            (
                train_data,
                valid_data,
                test_data,
                df_train,
                df_valid,
                df_test,
                adduct_feature_sizes,
            ) = build_datasets_from_separate_files(
                train_csv_path=config.get("train_data_path"),
                valid_csv_path=config.get("valid_data_path"),
                test_csv_path=test_csv_path,
                smiles_col=config.get("smiles_col", "SMILES"),
                adduct_col=config.get("adduct_col", "Adduct"),
                use_adduct_descriptors=config.get("use_adduct_descriptors", True),
                one_hot_encode_feats=config.get("one_hot_encode_feats", False),
                feature_config_path=Path(feature_config) if feature_config else None,
            )
            global_feature_sizes = adduct_feature_sizes
            log.info(
                f"Propagating global feature sizes to config: {global_feature_sizes}"
            )
            if global_feature_sizes is not None:
                # Inject so LightningModule / model constructors can read sizes
                config["global_feature_sizes"] = global_feature_sizes
        except Exception as e:
            log.warning(f"Could not build DataModule to infer feature sizes: {e}")
            # proceed without injecting sizes; module should handle missing values
            global_feature_sizes = None

        data_module = GraphXSectionDataModule(
            train_dataset=train_data,
            valid_dataset=valid_data,
            test_dataset=test_dataset,  # Ensure this is just the dataset, not (dataset, df)
            config=config,
            global_feature_sizes=global_feature_sizes,
        )

        # Create LightningModule instance matching the config
        model_type = config["model"].get("type", "Ginny")
        lightning_module = GraphXSectionLightningModule(
            config,
            model_type,
            node_feature_sizes=data_module.node_feature_sizes,
            edge_feature_sizes=data_module.edge_feature_sizes,
            global_feature_sizes=global_feature_sizes,
        )

        # Load checkpoint: prefer --inference-model, then --base-model
        ckpt_path = inference_model or base_model
        log.info(f"Loading checkpoint for inference: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # load_state_dict returns None or raises; use strict=False to be permissive
        lightning_module.load_state_dict(state, strict=False)

        # Run evaluation which will save predictions to disk
        # Pass the data_module which has proper test_dataloader()
        test_predictions, RMedAE = evaluate_lightning_model(
            lightning_module, data_module, test_df, config
        )

        log.info("Inference completed; exiting")
        return

    # Normal training path
    lightning_module = train_lightning_GraphXSection(
        config,
        train_data,
        valid_data,
        test_data,
        global_feature_sizes=adduct_feature_sizes,
        base_model_path=base_model if transfer_learn else None,
    )
    # Lightning evaluation
    test_predictions, RMedAE = evaluate_lightning_model(
        lightning_module, test_data, df_test, config
    )

    model = lightning_module.model  # Extract the underlying model for saliency mapping

    # Save model
    if config.get("save_model", False):
        output_suffix = (
            "_transfer_learned_model.pth" if transfer_learn else "_lightning_model.pth"
        )
        output_filename = f"{config['output_path']}/{config['run_name']}{output_suffix}"
        log.info(f"Saving Lightning model to {output_filename}")
        torch.save(
            lightning_module.state_dict(),
            output_filename,
        )
        log.info("Model saved")
        if transfer_learn:
            log.info(f"Transfer learned model saved with base model: {base_model}")

    # Create saliency maps
    if config.get("saliency_mapping", False):
        import pickle

        log.info("Generating saliency maps")
        saliency_maps = get_saliency_maps(model, test_data)
        with open(
            f"{config['output_path']}/{config['run_name']}_saliency_maps.pkl", "wb"
        ) as f:
            pickle.dump(saliency_maps, f)
        log.debug(f"Saliency maps shape: {saliency_maps.shape}")
        log.info("Saliency maps generated")

    log.info("GraphXSection_torch Done")


if __name__ == "__main__":
    main()
