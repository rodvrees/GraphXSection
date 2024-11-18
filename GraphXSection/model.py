from molgraph import layers
import tensorflow as tf
from tensorflow import keras
import logging
from typing import Any, Dict
from molgraph.layers import GNN

logger = logging.getLogger(__name__)


class DoNothing(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs


class GNN_old(keras.Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._layers = [
            layers.FeatureProjection(units=config["model"]["gnn_units"], name="proj")
        ] + [
            layers.GINConv(units=config["model"]["gnn_units"], name=f"gin_{i+1}")
            for i in range(config["model"]["num_GNN_layers"])
        ]

        self.readout = layers.Readout("sum")

    def call(self, inputs):
        x = inputs
        outputs = []
        for layer in self._layers:
            x = layer(x)
            outputs.append(x.node_feature)

        return x.update({"node_feature": tf.concat(outputs, axis=-1)})


class ReadoutAndConcatAdduct(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readout = layers.Readout("sum")

    def call(self, inputs):
        x = self.readout(inputs)
        x = tf.concat([x, tf.cast(inputs._x_adduct, tf.float32)], axis=1)
        return x


class QSAR_linear(keras.Model):
    def __init__(
        self, config: Dict[str, Any], variance_threshold: layers.VarianceThreshold
    ):
        super().__init__()
        self.config = config
        self.variance_threshold = variance_threshold
        self.donothing = DoNothing(name="donothing")
        self.gnn = GNN_old(self.config)
        self.readoutandconcatadduct = ReadoutAndConcatAdduct(
            name="readoutandconcatadduct"
        )
        for i in range(self.config["model"]["num_dense_layers"]):
            setattr(
                self,
                f"dense_{i+1}",
                keras.layers.Dense(
                    self.config["model"]["dense_units"],
                    activation="relu",
                    kernel_initializer=self.config["model"]["kernel_initializer"],
                    kernel_regularizer=tf.keras.regularizers.L1(
                        self.config["model"]["l1"]
                    ),
                ),
            )

        self.dense_output = keras.layers.Dense(1)

    def call(self, inputs):
        # x_mol is inputs without the "_x_adduct" key

        x_mol = inputs
        h0 = self.variance_threshold(x_mol)
        h0 = self.donothing(h0)
        x = self.gnn(h0)
        x = self.readoutandconcatadduct(x)

        for i in range(self.config["model"]["num_dense_layers"]):
            x = getattr(self, f"dense_{i+1}")(x)

        x = self.dense_output(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        config.update({"variance_threshold": self.variance_threshold})
        return config


def build_QSAR_model(
    config: Dict[str, Any], train_graph: Any, train_dataset: Any
) -> keras.Model:
    variance_threshold = layers.VarianceThreshold()
    variance_threshold.adapt(train_graph)
    if config["model"]["type"] == "QSAR_linear":
        model = QSAR_linear(config, variance_threshold)
    elif config["model"]["type"] == "QSAR_sequential":
        model = get_qsar_model(train_graph, config, variance_threshold, train_dataset)
    return model


class ReadoutAndConcatAdduct_sequential(keras.layers.Layer):

    def __init__(self, mode="mean", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.readout = layers.Readout(mode=self.mode)
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat(
            [self.readout(inputs), tf.cast(inputs._x_adduct, tf.float32)]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


def get_qsar_model(x_graph, config, variance_threshold, train_dataset):
    input_spec = train_dataset.element_spec[0]

    graph_input = layers.GNNInput(input_spec)

    graph_layers = [
        layers.FeatureProjection(units=config["model"]["gnn_units"], name="proj")
    ] + [
        layers.GINConv(units=config["model"]["gnn_units"], name=f"gin_{i+1}")
        for i in range(config["model"]["num_GNN_layers"])
    ]

    graph_layers = GNN(graph_layers)

    readout = ReadoutAndConcatAdduct_sequential(mode="sum")

    dense_layers = [
        keras.layers.Dense(
            config["model"]["dense_units"],
            activation="relu",
            kernel_initializer=config["model"]["kernel_initializer"],
            kernel_regularizer=tf.keras.regularizers.L1(config["model"]["l1"]),
        )
        for _ in range(config["model"]["num_dense_layers"])
    ]
    dense_layers += [keras.layers.Dense(1)]

    return keras.Sequential(
        [graph_input, variance_threshold, graph_layers, readout, *dense_layers]
    )
