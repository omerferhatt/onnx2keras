from functools import partial

from . import register_converter


register_converter = partial(register_converter, converter_type=__name__)


@register_converter("Constant")
def convert_constant(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    layers[node_name] = params["value"]
