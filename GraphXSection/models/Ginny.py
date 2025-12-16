from molgraph import layers
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


class SetGatherReadoutAndConcatGlobal(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readout = layers.SetGatherReadout()
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat(
            [self.readout(inputs), tf.cast(inputs._x_global, tf.float32)]
        )


class SegmentPoolingReadoutAndConcatGlobal(keras.layers.Layer):
    def __init__(self, mode="mean", **kwargs):
        super().__init__(**kwargs)
        self.readout = layers.SegmentPoolingReadout(mode=mode)
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat(
            [self.readout(inputs), tf.cast(inputs._x_global, tf.float32)]
        )


class AttentiveFPReadoutAndConcatGlobal(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readout = layers.AttentiveFPReadout()
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat(
            [self.readout(inputs), tf.cast(inputs._x_global, tf.float32)]
        )


def Ginny(train_dataset, config):
    """
    Train the Ginny model.

    Args:
        train_spec (Dict[str, Any]): Training specification.

    Returns:
        keras.Model: Trained model.
    """
    # Define the model
    input_spec = train_dataset.element_spec[0]

    graph_input = layers.GNNInput(input_spec)

    graph_layers = [
        layers.GINConv(
            units=config["model"]["gnn_units"],
            name=f"gin_{i+1}",
            normalization=config["model"].get("normalization", None),
            dropout=config["model"].get("dropout", 0.0),
            kernel_initializer=config["model"].get(
                "kernel_initializer",
                tf.keras.initializers.TruncatedNormal(stddev=0.005),
            ),
        )
        for i in range(config["model"]["num_GNN_layers"])
    ]

    graph_layers = layers.GNN(graph_layers)

    if config["model"].get("readout", "set_gather") == "set_gather":
        readout = SetGatherReadoutAndConcatGlobal()
    elif config["model"].get("readout", "set_gather") == "segment_pooling":
        readout = SegmentPoolingReadoutAndConcatGlobal(
            mode=config["model"].get("readout_mode", "mean")
        )
    elif config["model"].get("readout", "set_gather") == "attentive_fp":
        readout = AttentiveFPReadoutAndConcatGlobal()
    else:
        raise ValueError(
            "Invalid readout type. Must be 'set_gather' or 'segment_pooling'."
        )

    dense_layers = [
        keras.layers.Dense(
            units=config["model"]["dense_units"], activation="relu", name=f"dense_{i+1}"
        )
        for i in range(config["model"]["num_dense_layers"])
    ]

    output_layer = keras.layers.Dense(units=1, activation="linear", name="output")

    # Define the model
    model = keras.Sequential(
        [graph_input, graph_layers, readout] + dense_layers + [output_layer]
    )
    return model


def Ginny3D(train_dataset, config):
    """
    Train the Ginny3D model.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        keras.Model: Trained model.
    """
    # Train dataset is a tf.data.Dataset consisting of inputs and labels
    # inputs consist of smiles and global features
    # Extract both inputs
    input_spec = train_dataset.element_spec[0]
    # input_spec is a tuple of (smiles, global_features)
    smiles_spec = input_spec[0]
    global_feats_spec = input_spec[1]

    graph_input = layers.GNNInput(smiles_spec)

    graph_layers = [
        layers.GINConv(
            units=config["model"]["gnn_units"],
            name=f"gin_{i+1}",
            normalization=config["model"].get("normalization", None),
            dropout=config["model"].get("dropout", 0.0),
            kernel_initializer=config["model"].get(
                "kernel_initializer",
                tf.keras.initializers.TruncatedNormal(stddev=0.005),
            ),
        )
        for i in range(config["model"]["num_GNN_layers"])
    ]

    graph_layers = layers.GNN(graph_layers)
    graph_output = graph_layers(graph_input)

    # Run global features through a dense network
    global_feats_input = keras.layers.Input(
        shape=global_feats_spec.shape[1:], name="global_feats"
    )

    global_feats_dense = global_feats_input
    for i in range(config["model"]["num_dense_layers_global"]):
        global_feats_dense = keras.layers.Dense(
            units=config["model"]["dense_units_global"] / (i + 2),
            activation="relu",
            name=f"global_dense_{i+1}",
        )(global_feats_dense)
        global_feats_dense = keras.layers.Dropout(
            rate=config["model"].get("dropout", 0.0),
            name=f"global_dropout_{i+1}",
        )(global_feats_dense)

    if config["model"].get("readout", "set_gather") == "set_gather":
        readout = layers.SetGatherReadout()
    elif config["model"].get("readout", "set_gather") == "segment_pooling":
        readout = layers.SegmentPoolingReadout(
            mode=config["model"].get("readout_mode", "mean")
        )
    elif config["model"].get("readout", "set_gather") == "attentive_fp":
        readout = layers.AttentiveFPReadout()
    else:
        raise ValueError(
            "Invalid readout type. Must be 'set_gather' or 'segment_pooling'."
        )

    # Concatenate the graph readout and processed global features
    readout_output = readout(graph_output)
    logger.debug(f"readout type: {type(readout)}")
    concatenated = keras.layers.Concatenate(axis=-1)(
        [readout_output, global_feats_dense]
    )

    dense_layers = concatenated
    for i in range(config["model"]["num_dense_layers"]):
        dense_layers = keras.layers.Dense(
            units=config["model"]["dense_units"],
            activation="relu",
            name=f"dense_{i+1}",
        )(dense_layers)

    output_layer = keras.layers.Dense(units=1, activation="linear", name="output")(
        dense_layers
    )

    # Define the model
    model = keras.Model(inputs=[graph_input, global_feats_input], outputs=output_layer)
    return model
