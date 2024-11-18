import molgraph
import tensorflow as tf
from logging import getLogger
from typing import Dict, Any

logger = getLogger(__name__)


def get_saliency_maps(model: tf.keras.Model, dataset: tf.data.Dataset) -> Any:
    """Generate saliency maps for a given model and dataset.

    Args:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Dataset to generate saliency maps for.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Any: Saliency maps.
    """
    logger.debug(model.summary())
    logger.debug(dataset)
    # Test dataset has a single batch
    for batch in dataset:
        x_all, y = batch

    saliency = molgraph.models.GradientActivationMapping(model)
    saliency_maps = saliency(x_all.separate())
    return saliency_maps
