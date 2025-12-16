"""PyTorch Lightning training module for GraphXSection_torch."""

import logging
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import Subset
from typing import Dict, Any
import os

from GraphXSection_torch.lightning_modules import GraphXSectionLightningModule
from GraphXSection_torch.lightning_data import GraphXSectionDataModule
from GraphXSection_torch.data_extract import GraphXSectionDataset

logger = logging.getLogger(__name__)


def _merge_flat_overrides(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge flattened override keys (like 'model.loss_function') into a nested config copy.

    Non-dotted keys overwrite top-level entries. Returns a new merged dict.
    """
    cfg = copy.deepcopy(base) if base is not None else {}
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
            cfg[k] = v
    return cfg


def create_lightning_callbacks(config: Dict[str, Any]):
    """Create PyTorch Lightning callbacks based on configuration."""
    callbacks = []

    # Model checkpoint
    if config.get("mcp", False):
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.dirname(config["mcp"]["mcp_path"]),
            filename=os.path.basename(config["mcp"]["mcp_path"]).replace(".pth", ""),
            monitor=config["mcp"]["mcp_monitor"],
            mode=config["mcp"]["mcp_mode"],
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping
    if config.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor=config["early_stopping"].get("monitor", "val_loss"),
            patience=config["early_stopping"].get("patience", 10),
            min_delta=config["early_stopping"].get("min_delta", 0.0),
            mode=config["early_stopping"].get("mode", "min"),
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


def create_lightning_logger(config: Dict[str, Any]):
    """Create PyTorch Lightning logger based on configuration."""
    logger.info(f"Creating lightning logger. W&B config: {config.get('wandb', {})}")

    if config.get("wandb", {}).get("use", False):
        # Check if wandb is already initialized (like in original GraphXSection)
        try:
            import wandb
            import os

            # Ensure W&B runs in online mode
            if "WANDB_MODE" not in os.environ:
                os.environ["WANDB_MODE"] = "online"
                logger.info("Set W&B to online mode for Lightning logger")

            if wandb.run is not None:
                # Use existing wandb run
                wandb_logger = WandbLogger(experiment=wandb.run)
                logger.info("Using existing wandb run")
            else:
                # Create new wandb logger
                project = config["wandb"].get("project", "graphxsection-sweeps")
                run_name = config.get("run_name", "graphxsection-run")

                logger.info(
                    f"Creating new wandb logger - project: {project}, name: {run_name}"
                )
                wandb_logger = WandbLogger(
                    project=project,
                    name=run_name,
                    config=config,
                    mode="online",  # Explicitly set online mode
                )
                logger.info("✅ Created new wandb logger successfully")
            return wandb_logger
        except Exception as e:
            logger.error(f"❌ Failed to create wandb logger: {e}")
            import traceback

            traceback.print_exc()
            return True  # Fall back to default logger
    else:
        logger.info("W&B logging disabled - using default logger")
        return True  # Use default TensorBoardLogger


def train_lightning_GraphXSection(
    config: Dict[str, Any],
    train_dataset,
    valid_dataset,
    test_dataset=None,
    global_feature_sizes=None,
    base_model_path=None,
):
    """Train a GraphXSection model using PyTorch Lightning.

    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        global_feature_sizes: Global feature sizes
        base_model_path: Path to pre-trained model checkpoint for transfer learning (optional)
    """
    logger.info("Starting Lightning-based model training")
    if base_model_path:
        logger.info(
            f"Transfer learning enabled - will load weights from: {base_model_path}"
        )
    # Merge in any W&B / sweep flattened overrides into the nested config before use
    try:
        import wandb

        wb_cfg = dict(getattr(getattr(wandb, "run", None), "config", {}) or {})
        if wb_cfg:
            logger.info("Detected W&B run config overrides: %s", list(wb_cfg.keys()))
            config = _merge_flat_overrides(config, wb_cfg)
            logger.info(
                "Merged config keys after applying W&B overrides. loss_function: %s",
                (
                    config.get("model", {}).get("loss_function")
                    or config.get("loss_function")
                ),
            )
            # Log if learning rate scheduler is configured
            if "lr_scheduler" in config and isinstance(config["lr_scheduler"], dict):
                logger.info(
                    "Learning rate scheduler configured: %s", config["lr_scheduler"]
                )
    except Exception:
        # No W&B or no overrides; proceed with original config
        pass

    # Create data module
    data_module = GraphXSectionDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset if test_dataset is not None else valid_dataset,
        config=config,
        global_feature_sizes=global_feature_sizes,
    )

    # Create Lightning module
    model_type = config["model"].get("type", "Ginny")

    lightning_module = GraphXSectionLightningModule(
        config,
        model_type,
        node_feature_sizes=data_module.node_feature_sizes,
        edge_feature_sizes=data_module.edge_feature_sizes,
        global_feature_sizes=data_module.global_feature_sizes,
    )

    # Load pre-trained weights for transfer learning
    if base_model_path:
        logger.info(f"Loading pre-trained weights from {base_model_path}")
        try:
            checkpoint = torch.load(base_model_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # PyTorch Lightning checkpoint format
                state_dict = checkpoint["state_dict"]
                logger.info("Detected PyTorch Lightning checkpoint format")
            else:
                # Plain state dict
                state_dict = checkpoint
                logger.info("Detected plain state dict format")

            # Load weights with strict=False to allow for differences in architecture
            # (e.g., different output layers, different feature dimensions)
            missing_keys, unexpected_keys = lightning_module.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

            logger.info("Successfully loaded pre-trained weights for transfer learning")

            # Optionally freeze GNN layers for transfer learning
            # Check both 'freeze_layers' and 'freeze_base_model' for backwards compatibility
            freeze_layers = config.get(
                "freeze_layers", config.get("freeze_base_model", False)
            )

            if freeze_layers:
                logger.info(
                    "Freezing GNN encoder layers - attempting to keep prediction head trainable"
                )

                # Collect parameter names to keep trainable
                trainable_param_names = set()

                # Heuristic 1: module name contains head/pred keywords
                for mod_name, module in lightning_module.model.named_modules():
                    lname = mod_name.lower()
                    if any(
                        k in lname for k in ["pred", "head", "dense", "output", "final"]
                    ):
                        for p_name, _ in module.named_parameters(recurse=False):
                            full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                            trainable_param_names.add(full_name)

                # Heuristic 2: keep Linear layers that output a single value (regression head)
                import torch.nn as nn

                for mod_name, module in lightning_module.model.named_modules():
                    if isinstance(module, nn.Linear):
                        outf = getattr(module, "out_features", None)
                        if outf == 1:
                            for p_name, _ in module.named_parameters(recurse=False):
                                full_name = (
                                    f"{mod_name}.{p_name}" if mod_name else p_name
                                )
                                trainable_param_names.add(full_name)

                # Heuristic 3: parameter name contains typical head keywords
                for p_name, _ in lightning_module.model.named_parameters():
                    lname = p_name.lower()
                    if any(
                        k in lname
                        for k in [
                            "predhead",
                            "pred_head",
                            "prediction_head",
                            "pred",
                            "head",
                            "dense_layers",
                            "dense",
                        ]
                    ):
                        trainable_param_names.add(p_name)

                # Apply freezing: freeze everything except the trainable_param_names
                frozen_count = 0
                trainable_count = 0
                for p_name, param in lightning_module.model.named_parameters():
                    should_train = p_name in trainable_param_names
                    param.requires_grad = should_train
                    if should_train:
                        trainable_count += 1
                        logger.debug(f"Keeping trainable: {p_name}")
                    else:
                        frozen_count += 1
                        logger.debug(f"Froze parameter: {p_name}")

                # If no parameters are trainable, try to unfreeze the last Linear layer (safe fallback)
                if trainable_count == 0:
                    logger.warning(
                        "No trainable parameters detected after freezing heuristics. Trying to unfreeze the final linear layer."
                    )
                    last_linear = None
                    for mod_name, module in lightning_module.model.named_modules():
                        if isinstance(module, nn.Linear):
                            last_linear = (mod_name, module)
                    if last_linear is not None:
                        mod_name, module = last_linear
                        for p_name, _ in module.named_parameters(recurse=False):
                            full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                            for (
                                pname,
                                param,
                            ) in lightning_module.model.named_parameters():
                                if pname == full_name:
                                    param.requires_grad = True
                                    trainable_count += 1
                                    logger.info(
                                        f"Unfroze parameter (final linear): {pname}"
                                    )
                                    break

                # Final fallback: if still nothing trainable, unfreeze all parameters to avoid training crash
                if trainable_count == 0:
                    logger.error(
                        "Unable to detect any prediction-head parameters to train. Unfreezing all parameters to avoid training failure."
                    )
                    for p_name, param in lightning_module.model.named_parameters():
                        param.requires_grad = True
                    frozen_count = 0
                    trainable_count = sum(
                        1 for _ in lightning_module.model.named_parameters()
                    )

                logger.info(
                    f"Frozen {frozen_count} parameters, {trainable_count} parameters remain trainable"
                )
                logger.info(
                    "Transfer learning mode: GNN layers frozen (where possible), prediction head will be fine-tuned"
                )

        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            raise RuntimeError(
                f"Transfer learning failed - could not load base model: {e}"
            )

    # Create callbacks
    callbacks = create_lightning_callbacks(config)

    # Create logger
    lightning_logger = create_lightning_logger(config)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["model"].get("epochs", 100),
        accelerator="auto",  # Automatically choose GPU/CPU
        devices="auto",  # Use all available devices
        callbacks=callbacks,
        logger=lightning_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=config.get("gradient_clip_val", None),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        precision=config.get("precision", 32),
        deterministic=config.get("deterministic", False),
        enable_checkpointing=config.get("mcp", False),
        fast_dev_run=config.get("fast_dev_run", False),  # For debugging
    )

    # Train the model
    trainer.fit(lightning_module, data_module)

    # Optionally restore best checkpoint weights before returning
    if config.get("use_best_model", False) and config.get("mcp", False):
        best_path = None
        # Prefer explicit ModelCheckpoint callback if present
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", None):
                if cb.best_model_path:
                    best_path = cb.best_model_path
                    break
        # Fallback to trainer.checkpoint_callback
        if not best_path and getattr(trainer, "checkpoint_callback", None):
            best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)

        if best_path and os.path.exists(best_path):
            try:
                ckpt = torch.load(best_path, map_location="cpu")
                lightning_module.load_state_dict(ckpt["state_dict"], strict=True)
                logger.info(f"Loaded best checkpoint for evaluation: {best_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load best checkpoint at {best_path}; using last-epoch weights. Error: {e}"
                )
        else:
            logger.warning(
                "use_best_model is True but no best checkpoint was found; proceeding with last-epoch weights."
            )

    logger.info("Lightning-based model training completed")
    return lightning_module
