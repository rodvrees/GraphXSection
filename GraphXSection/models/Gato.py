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


class TransformerEncoderReadoutAndConcatGlobal(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.readout = layers.TransformerEncoderReadout(hidden_units=units)
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat(
            [self.readout(inputs), tf.cast(inputs._x_global, tf.float32)]
        )


def Gato(train_dataset, config):
    """
    Train the Gato model.

    Args:
        train_spec (Dict[str, Any]): Training specification.

    Returns:
        keras.Model: Trained model.
    """
    # Define the model
    input_spec = train_dataset.element_spec[0]

    graph_input = layers.GNNInput(input_spec)

    graph_layers = [
        layers.GATConv(
            units=config["model"]["gnn_units"],
            num_heads=config["model"]["num_heads"],
            name=f"gin_{i+1}",
            normalization=config["model"].get("normalization", None),
            dropout=config["model"].get("dropout", 0.0),
            kernel_initializer=config["model"].get(
                "kernel_initializer",
                tf.keras.initializers.TruncatedNormal(stddev=0.005),
            ),
            activation=config["model"].get("activation", "relu"),
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
    elif config["model"].get("readout", "set_gather") == "transformer_encoder":
        readout = TransformerEncoderReadoutAndConcatGlobal(
            units=config["model"]["readout_units"]
        )
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
