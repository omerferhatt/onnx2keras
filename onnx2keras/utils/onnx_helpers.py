import logging

from onnx import numpy_helper


def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """

    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField("t"):
            return numpy_helper.to_array(getattr(onnx_attr, "t"))

        for attr_type in ["f", "i", "s"]:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ["floats", "ints", "strings"]:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))

    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def onnx_node_initializers_to_dict(onnx_model):
    onnx_weights = onnx_model.graph.initializer
    logging.debug("Gathering weights to dictionary.")

    weights = {}
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]

            if onnx_extracted_weights_name.startswith("/"):
                onnx_extracted_weights_name = onnx_extracted_weights_name[1:]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        except:
            if onnx_extracted_weights_name.startswith("/"):
                onnx_extracted_weights_name = onnx_extracted_weights_name[1:]
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

        logging.debug(
            "Found weight {0} with shape {1}.".format(
                onnx_extracted_weights_name, weights[onnx_extracted_weights_name].shape
            )
        )
    return weights
