"""
The ONNX to keras converter module
"""

from tensorflow import keras
import logging

from .layers import AVAILABLE_CONVERTERS
from .utils.onnx_helpers import onnx_node_attributes_to_dict
from .utils.onnx_helpers import onnx_node_initializers_to_dict


def onnx_to_keras(
    onnx_model,
    input_names,
    verbose=True,
):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param verbose: verbose output
    :return: Keras model
    """

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger("onnx2keras")

    logger.info("Converter is called.")

    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug("List inputs:")
    for i, input in enumerate(onnx_inputs):
        logger.debug("Input {0} -> {1}.".format(i, input.name))

    logger.debug("List outputs:")
    for i, output in enumerate(onnx_outputs):
        logger.debug("Output {0} -> {1}.".format(i, output))

    weights = onnx_node_initializers_to_dict(onnx_model)

    layers = dict()
    lambda_funcs = dict()
    keras_outputs = []
    keras_inputs = []

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                channels, *size = [
                    i.dim_value for i in onnx_i.type.tensor_type.shape.dim
                ][1:]

                layers[input_name] = keras.layers.InputLayer(
                    input_shape=[*size, channels], batch_size=1, name=input_name
                ).output

                keras_inputs.append(layers[input_name])

                logger.debug(
                    "Found input {0} with shape {1}".format(
                        input_name, [*size, channels]
                    )
                )
                logger.debug("Shape transposed to channel last")

    # Convert every operation separable
    node_names = []
    for node in onnx_nodes:
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        for i, _input_name in enumerate(node.input):
            if _input_name.startswith("/"):
                node.input[i] = _input_name[1:]
        for i, _output_name in enumerate(node.output):
            if _output_name.startswith("/"):
                node.output[i] = _output_name[1:]

        node_name = str(node.output[0])
        keras_names = []
        for output in node.output:
            keras_names.append(output)

        if len(node.output) != 1:
            logger.warning("Trying to convert multi-output node")
            node_params["_outputs"] = list(node.output)
            node_names.extend(keras_names)
        else:
            keras_names = keras_names[0]
            node_names.append(keras_names)

        logger.debug("######")
        logger.debug("...")
        logger.debug("Converting ONNX operation")
        logger.debug("type: %s", node_type)
        logger.debug("node_name: %s", node_name)
        logger.debug("node_params: %s", node_params)
        logger.debug("...")

        logger.debug("Check if all inputs are available:")
        if len(node.input) == 0 and node_type != "Constant":
            raise AttributeError("Operation doesn't have an input. Aborting.")

        for i, node_input in enumerate(node.input):
            logger.debug("Check input %i (name %s).", i, node_input)
            if node_input not in layers:
                logger.debug("The input not found in layers / model inputs.")

                if node_input in weights:
                    logger.debug("Found in weights, add as a numpy constant.")
                    layers[node_input] = weights[node_input]
                else:
                    raise AttributeError(
                        "Current node is not in weights / model inputs / layers."
                    )
        else:
            logger.debug("... found all, continue")

        AVAILABLE_CONVERTERS[node_type](
            node, node_params, layers, lambda_funcs, node_name, keras_names
        )
        if isinstance(keras_names, list):
            keras_names = keras_names[0]

        try:
            logger.debug("Output TF Layer -> " + str(layers[keras_names]))
        except KeyError:
            pass

    # Check for terminal nodes
    for layer in onnx_outputs:
        if layer in layers:
            keras_outputs.append(layers[layer])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)

    return model
