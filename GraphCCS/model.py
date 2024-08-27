from molgraph import layers
import tensorflow as tf
from tensorflow import keras
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SigmaCCSMimic(keras.Model):
    def __init__(self, config: Dict[str, Any]):

        super().__init__()
        self.config = config
        for i in range(self.config["model"]["num_GNN_layers"]):
            setattr(
                self,
                f"gin_conv{i+1}",
                layers.GCNConv(
                    self.config["model"]["gnn_units"],
                    use_edge_features=self.config["model"]["use_edge_features"],
                    kernel_regularizer="l2",
                    kernel_initializer=self.config["model"]["kernel_initializer"],
                ),
            )
        self.readout = layers.Readout("sum")
        for i in range(self.config["model"]["num_dense_layers"]):
            setattr(
                self,
                f"dense_{i+1}",
                keras.layers.Dense(
                    self.config["model"]["dense_units"],
                    activation="relu",
                    kernel_regularizer="l2",
                ),
            )

        self.dense_output = keras.layers.Dense(1)

    def call(self, inputs):

        x_mol = inputs[0]
        x_adduct = inputs[1]
        x_adduct = tf.cast(x_adduct, tf.float32)

        x1 = x_mol
        for i in range(self.config["model"]["num_GNN_layers"]):
            x1 = getattr(self, f"gin_conv{i+1}")(x1)

        r = self.readout(x1)
        x = tf.concat([r, x_adduct], axis=1)
        for i in range(self.config["model"]["num_dense_layers"]):
            x = getattr(self, f"dense_{i+1}")(x)
        x = self.dense_output(x)
        return x


class QSARMimic(keras.Model):
    def __init__(
        self, config: Dict[str, Any], variance_threshold: layers.VarianceThreshold
    ):
        super().__init__()
        self.config = config
        self.variance_threshold = variance_threshold
        self.feature_projection = layers.FeatureProjection(
            self.config["model"].get("gnn_units", 128),
        )
        for i in range(self.config["model"]["num_GNN_layers"]):
            setattr(
                self,
                f"gin_conv{i+1}",
                layers.GINConv(
                    self.config["model"]["gnn_units"],
                    normalization=self.config["model"]["normalization"],
                    kernel_initializer=self.config["model"]["kernel_initializer"],
                    kernel_regularizer=tf.keras.regularizers.L1(
                        self.config["model"]["l1"]
                    ),
                ),
            )
        for i in range(self.config["model"]["num_GNN_layers"] + 1):
            setattr(self, f"readout_{i+1}", layers.Readout("sum"))
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
        x_mol = inputs[0]
        x_adduct = inputs[1]
        x_adduct = tf.cast(x_adduct, tf.float32)

        h0 = self.variance_threshold(x_mol)
        h0 = self.feature_projection(h0)
        h1 = self.gin_conv1(h0)
        h2 = self.gin_conv2(h1)
        h3 = self.gin_conv3(h2)
        # h4 = self.gin_conv4(h3)

        z0 = self.readout_1(h0)
        z1 = self.readout_2(h1)
        z2 = self.readout_3(h2)
        z3 = self.readout_4(h3)
        # z4 = self.readout_5(h4)

        x = tf.concat([z0, z1, z2, z3, x_adduct], axis=1)

        for i in range(self.config["model"]["num_dense_layers"]):
            x = getattr(self, f"dense_{i+1}")(x)

        x = self.dense_output(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config})
        config.update({"variance_threshold": self.variance_threshold})
        return config


class DoNothing(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs


class GNN(keras.Model):
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
        self.gnn = GNN(self.config)
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


def build_QSAR_model(config: Dict[str, Any], train_graph: Any) -> keras.Model:
    variance_threshold = layers.VarianceThreshold()
    variance_threshold.adapt(train_graph)
    if config["model"]["type"] == "QSAR":
        model = QSARMimic(config, variance_threshold)
    elif config["model"]["type"] == "QSAR_linear":
        model = QSAR_linear(config, variance_threshold)

    return model
