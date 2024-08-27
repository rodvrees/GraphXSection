import logging
import wandb
import tensorflow as tf
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger
from GraphXSection.model import SigmaCCSMimic, build_QSAR_model
from typing import Dict, Any, List

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
    train_x_adduct = train_data["adduct_OHE"]
    train_x_adduct = tf.convert_to_tensor(train_x_adduct, dtype=tf.int32)
    train_x_mass = train_data["mass"]
    train_x_mol = mol_encoder(train_smiles)
    y_train = train_data["y"]

    valid_smiles = valid_data["smiles"]
    valid_x_adduct = valid_data["adduct_OHE"]
    valid_x_adduct = tf.convert_to_tensor(valid_x_adduct, dtype=tf.int32)
    valid_x_mass = valid_data["mass"]
    valid_x_mol = mol_encoder(valid_smiles)
    y_valid = valid_data["y"]

    logger.debug(
        "Training on device: %s", tf.config.experimental.list_physical_devices("GPU")
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_x_mol, train_x_adduct), y_train)
    )
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        ((valid_x_mol, valid_x_adduct), y_valid)
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
        lambda x_and_x_adduct, y: (
            x_and_x_adduct[0].update({"_x_adduct": x_and_x_adduct[1]}),
            y,
        )
    )

    valid_dataset = valid_dataset.map(
        lambda x_and_x_adduct, y: (
            x_and_x_adduct[0].update({"_x_adduct": x_and_x_adduct[1]}),
            y,
        )
    )

    # Model selection based on the configuration
    if config["model"]["type"] == "SigmaCCS":
        model = SigmaCCSMimic(config)
    elif config["model"]["type"] == "QSAR":
        model = build_QSAR_model(config, train_x_mol)
    elif config["model"]["type"] == "QSAR_linear":
        model = build_QSAR_model(config, train_x_mol)
    elif config["model"]["type"] == "QSAR_sequential":
        model = build_QSAR_model(config, train_x_mol)

    else:
        raise ValueError(
            f"Invalid model type: {config['model']['type']}. Only 'SigmaCCS' and 'QSAR' are supported for now."
        )

    metrics = config["model"]["metrics"]
    if config["model"]["optimizer"] == "adam":
        opt = keras.optimizers.Adam(learning_rate=config["model"]["learning_rate"])
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
