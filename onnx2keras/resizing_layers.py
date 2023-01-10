from tensorflow import keras
import numpy as np
import logging


def convert_resizing(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert resize.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.resize')
    logger.warning('!!! EXPERIMENTAL SUPPORT (resize) !!!')

    if 'mode' in params:
        mode = params['mode'].decode('utf-8')
        if mode == 'linear':
            mode = 'bilinear'
    else:
        mode = 'bilinear'

    if 'cubic_coeff_a' in params:
        cubic_coeff = params['cubic_coeff_a']
    else:
        cubic_coeff = -0.5
    
    # roi = layers[node.input[1]]
    scales = layers[node.input[2]]
    if len(node.input) > 3:
        sizes = layers[node.input[3]]
    else:
        sizes = []

    if any(sizes):
        size_x, size_y = sizes[-2:]
    elif any(scales):
        size_x, size_y = scales[-2:] * np.array(layers[node.input[0]].shape[1:3])
    else:
        raise ValueError('No sizes or scales provided for resizing.')

    resizing = keras.layers.Resizing(
        height=int(size_x), width=int(size_y), name=keras_name, 
    )

    layers[node_name] = resizing(layers[node.input[0]])
