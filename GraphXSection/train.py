import logging
import tensorflow as tf
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger
from GraphXSection.model import build_QSAR_model, get_qsar_model
from GraphXSection.models.Ginny import Ginny, Ginny3D
from GraphXSection.models.Gato import Gato
from GraphXSection.models.Eddy import Eddy
from typing import Dict, Any, List
import numpy as np
import wandb
from keras.callbacks import Callback

logger = logging.getLogger(__name__)


class WandbBestValMAELogger(Callback):
    def __init__(self, monitor="val_mean_absolute_error", log_name="best_val_mae"):
        super().__init__()
        self.monitor = monitor
        self.best_val = float("inf")
        self.log_name = log_name

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None and current < self.best_val:
            self.best_val = current

        # Log best value seen so far at this epoch
        wandb.log({self.log_name: self.best_val})


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
            callbacks.append(WandbBestValMAELogger())

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

    # Get the index of "NC1=CC=CC2=C1C=CC=C2N" in the training set
    if "NC1=CC=CC2=C1C=CC=C2N" in train_smiles:
        idx = train_smiles.index("NC1=CC=CC2=C1C=CC=C2N")
        logger.debug(f"Index of NC1=CC=CC2=C1C=CC=C2N in training set: {idx}")
        not_in = False
    else:
        logger.debug("NC1=CC=CC2=C1C=CC=C2N not found in training set")
        idx = 0
        not_in = True

    train_x_mol = mol_encoder(train_smiles)
    if not not_in:
        logger.info(f"First SMILES: {train_smiles[idx]}")
        logger.info(f"First molecular features: {train_x_mol[idx]}")
        logger.info(f"First node features:")
        for feature in train_x_mol[idx].node_feature:
            logger.info(feature)
        logger.info(f"First edge features:")
        for feature in train_x_mol[idx].edge_feature:
            logger.info(feature)
    y_train = train_data["y"]

    valid_smiles = valid_data["smiles"]
    if "NC1=CC=CC2=C1C=CC=C2N" in valid_smiles:
        idx = valid_smiles.index("NC1=CC=CC2=C1C=CC=C2N")
        logger.debug(f"Index of NC1=CC=CC2=C1C=CC=C2N in validation set: {idx}")
        not_in = False
    else:
        logger.debug("NC1=CC=CC2=C1C=CC=C2N not found in validation set")
        idx = 0
        not_in = True

    if not not_in:
        logger.info(f"First SMILES: {valid_smiles[idx]}")
        logger.info(f"First molecular features: {valid_smiles[idx]}")
        logger.info(f"First node features:")
        for feature in valid_smiles[idx].node_feature:
            logger.info(feature)
        logger.info(f"First edge features:")
        for feature in valid_smiles[idx].edge_feature:
            logger.info(feature)
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

    if not config["model"]["type"] == "Ginny3D":
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

    elif config["model"]["type"] == "Ginny3D":
        model = Ginny3D(train_dataset, config)

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


def train_Eddy(train_data, valid_data, config):
    """
    Train the Eddy model.

    Args:
        train_data (tf.data.Dataset): Training dataset.
        valid_data (tf.data.Dataset): Validation dataset.

    Returns:
        keras.Model: Trained model.
    """
    logger.info("Training Eddy model")

    x_embeddings_train = np.array(
        [np.array(embedding) for embedding in train_data["embeddings"]]
    )
    logger.debug(f"Shape of train embeddings: {x_embeddings_train.shape}")

    x_embeddings_valid = np.array(
        [np.array(embedding) for embedding in valid_data["embeddings"]]
    )
    logger.debug(f"Shape of valid embeddings: {x_embeddings_valid.shape}")

    x_adducts_train = np.array(
        [np.array(adduct_ohe) for adduct_ohe in train_data["Adduct_OHE"]]
    )
    logger.debug(f"Shape of train adducts: {x_adducts_train.shape}")
    x_adducts_valid = np.array(
        [np.array(adduct_ohe) for adduct_ohe in valid_data["Adduct_OHE"]]
    )
    logger.debug(f"Shape of valid adducts: {x_adducts_valid.shape}")
    y_train = train_data["CCS"]
    y_valid = valid_data["CCS"]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.concat([x_embeddings_train, x_adducts_train], axis=1), y_train)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.concat([x_embeddings_valid, x_adducts_valid], axis=1), y_valid)
    )

    train_dataset = train_dataset.batch(config["model"]["batch_size"]).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    valid_dataset = valid_dataset.batch(config["model"]["batch_size"]).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    metrics = config["model"]["metrics"]
    model = Eddy(config)
    model.compile(
        optimizer=config["model"]["optimizer"],
        loss=config["model"]["loss"],
        metrics=metrics,
    )
    model.fit(
        train_dataset,
        epochs=100,
        validation_data=valid_dataset,
        callbacks=_configure_callbacks(config),
    )

    return model
