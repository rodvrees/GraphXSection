import logging
import tensorflow as tf
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger
from model import build_QSAR_model, get_qsar_model
from models.Ginny import Ginny
from models.Gato import Gato
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


def _configure_callbacks(config: Dict[str, Any]) -> List[keras.callbacks.Callback]:
    """Configure the callbacks for the model training.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        List[keras.callbacks.Callback]: List of callbacks
    """

    callbacks = []
    if config["mcp"]:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=config["mcp"]["mcp_path"],
                monitor=config["mcp"]["mcp_monitor"],
                save_best_only=True,
                mode=config["mcp"]["mcp_mode"],
            )
        )

    if config.get("lr_scheduler", False):
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=config["lr_scheduler"]["monitor"],
                factor=config["lr_scheduler"]["factor"],
                patience=config["lr_scheduler"]["patience"],
                min_lr=config["lr_scheduler"]["min_lr"],
                verbose=1,
            )
        )

    if config.get("wandb", False):
        if config["wandb"].get("use", False):
            callbacks.append(WandbMetricsLogger())

    return callbacks


def train_GraphXSection(
    train_data: Dict[str, Any],
    valid_data: Dict[str, Any],
    mol_encoder: Any,
    config: Dict[str, Any],
) -> keras.Model:
    """Train the GraphXSection model.

    Args:
        train_data (Dict[str, Any]): Training data.
        valid_data (Dict[str, Any]): Validation data.
        mol_encoder (Any): Molecular encoder.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        keras.Model: Trained model.
    """

    logger.info("Model training started")

    train_smiles = train_data["smiles"]
    train_global_feats = train_data["global_feats"]
    logger.debug(f"Train global features: {train_global_feats}")
    logger.debug(f"Type of train global features: {type(train_global_feats)}")
    logger.debug(f"Shape of train global features: {train_global_feats.shape}")
    train_global_feats = tf.convert_to_tensor(train_global_feats, dtype=tf.float32)
    logger.debug(
        f"Type of train global features after conversion: {type(train_global_feats)}"
    )
    logger.debug(
        f"Shape of train global features after conversion: {train_global_feats.shape}"
    )
    logger.debug(f"Train global features: {train_global_feats}")

    train_x_mol = mol_encoder(train_smiles)
    y_train = train_data["y"]

    valid_smiles = valid_data["smiles"]
    valid_global_feats = valid_data["global_feats"]
    valid_global_feats = tf.convert_to_tensor(valid_global_feats, dtype=tf.float32)
    valid_x_mol = mol_encoder(valid_smiles)
    y_valid = valid_data["y"]

    logger.debug(
        "Training on device: %s", tf.config.experimental.list_physical_devices("GPU")
    )

    print(
        "Checking for NaNs in training features:",
        np.isnan(train_global_feats.numpy()).sum(),
    )
    print(
        "Checking for NaNs in validation features:",
        np.isnan(valid_global_feats.numpy()).sum(),
    )

    # Identify nan values in the global features
    if np.isnan(train_global_feats.numpy()).sum() > 0:
        logger.warning("NaN values found in training global features")
        logger.warning("Here are the indices of NaN values:")
        nan_indices = np.argwhere(np.isnan(train_global_feats.numpy()))
        logger.warning(nan_indices)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_x_mol, train_global_feats), y_train)
    )
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        ((valid_x_mol, valid_global_feats), y_valid)
    )

    train_dataset = (
        train_dataset.shuffle(buffer_size=len(train_data["smiles"]))
        .batch(config["model"]["batch_size"])
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    valid_dataset = valid_dataset.batch(config["model"]["batch_size"]).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    train_dataset = train_dataset.map(
        lambda x_and_global_feats, y: (
            x_and_global_feats[0].update({"_x_global": x_and_global_feats[1]}),
            y,
        )
    )

    valid_dataset = valid_dataset.map(
        lambda x_and_global_feats, y: (
            x_and_global_feats[0].update({"_x_global": x_and_global_feats[1]}),
            y,
        )
    )

    if config["model"]["type"] == "QSAR_linear":
        logger.warning(
            'DeprationWarning: "QSAR_linear" model type is deprecated, and will be removed in future versions. Use "QSAR_sequential" instead.'
        )
        model = build_QSAR_model(config, train_x_mol, train_dataset)

    elif config["model"]["type"] == "QSAR_sequential":
        model = build_QSAR_model(config, train_x_mol, train_dataset)

    elif config["model"]["type"] == "Ginny":
        model = Ginny(train_dataset, config)

    elif config["model"]["type"] == "Gato":
        model = Gato(train_dataset, config)

    else:
        raise ValueError(
            f"Invalid model type: {config['model']['type']}. Only 'QSAR_sequential' is supported for now."
        )

    metrics = config["model"]["metrics"]
    if config["model"]["optimizer"] == "adam":
        opt = keras.optimizers.Adam(learning_rate=config["model"]["learning_rate"])
    elif config["model"]["optimizer"] == "adamw":
        opt = keras.optimizers.AdamW(
            learning_rate=config["model"]["learning_rate"],
            weight_decay=config["model"]["weight_decay"],
        )
    else:
        raise ValueError(
            f"Invalid optimizer: {config['model']['optimizer']}. Only 'adam' is supported for now."
        )

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=config["model"]["loss"],
        metrics=metrics,
    )

    # Train the model
    model.fit(
        train_dataset,
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        validation_data=valid_dataset,
        verbose=2,
        callbacks=_configure_callbacks(config),
    )
    logger.info(model.summary())

    logger.info("Model training complete")
    return model
