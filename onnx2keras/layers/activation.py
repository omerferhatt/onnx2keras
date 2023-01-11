from functools import partial

from tensorflow import keras

from . import register_converter
from ..utils.numpy_helpers import ensure_numpy_type
from ..utils.tensorflow_helpers import ensure_tf_type


register_converter = partial(register_converter, converter_type=__name__)


@register_converter("Relu")
def convert_relu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    relu = keras.layers.Activation("relu", name=keras_name)
    layers[node_name] = relu(input_0)


@register_converter("Elu")
def convert_elu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    elu = keras.layers.ELU(alpha=params["alpha"], name=keras_name)
    layers[node_name] = elu(input_0)


@register_converter("LeakyRelu")
def convert_lrelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert LeakyReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    leakyrelu = keras.layers.LeakyReLU(alpha=params["alpha"], name=keras_name)
    layers[node_name] = leakyrelu(input_0)


@register_converter("Sigmoid")
def convert_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Sigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    sigmoid = keras.layers.Activation("sigmoid", name=keras_name)
    layers[node_name] = sigmoid(input_0)


@register_converter("Tanh")
def convert_tanh(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Tanh activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    tanh = keras.layers.Activation("tanh", name=keras_name)
    layers[node_name] = tanh(input_0)


@register_converter("Selu")
def convert_selu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    selu = keras.layers.Activation("selu", name=keras_name)
    layers[node_name] = selu(input_0)


@register_converter("Softmax")
def convert_softmax(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert softmax activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x, axis=params["axis"]):
        import tensorflow as tf

        return tf.nn.softmax(x, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)
    lambda_func[keras_name] = target_layer


@register_converter("PRelu")
def convert_prelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert PReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """

    if len(node.input) != 2:
        assert AttributeError("Activation layer PReLU should have 2 inputs.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    W = ensure_numpy_type(layers[node.input[1]])

    shared_axes = [2, 3]

    # for case when W.shape (n,). When activation is used for single dimension vector.
    shared_axes = shared_axes if len(W.shape) > 1 else None

    prelu = keras.layers.PReLU(weights=[W], shared_axes=shared_axes, name=keras_name)
    layers[node_name] = prelu(input_0)
