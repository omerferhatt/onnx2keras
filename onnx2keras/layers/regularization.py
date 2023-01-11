from functools import partial

from tensorflow import keras

from . import register_converter
from ..utils.tensorflow_helpers import ensure_tf_type

register_converter = partial(register_converter, converter_type=__name__)


@register_converter("Dropout")
def convert_dropout(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Dropout layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """

    # In ONNX Dropout returns dropout mask as well.
    if isinstance(keras_name, list) and len(keras_name) > 1:
        keras_name = keras_name[0]

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    ratio = params["ratio"] if "ratio" in params else 0.0
    lambda_layer = keras.layers.Dropout(ratio, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
