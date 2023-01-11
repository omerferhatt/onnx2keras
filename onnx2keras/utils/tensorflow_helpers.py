import numpy as np
from tensorflow import keras

from .numpy_helpers import is_numpy


def ensure_tf_type(obj, fake_input_layer=None, name=None):
    """
    Convert to Keras Constant if needed
    :param obj: numpy / tf type
    :param fake_input_layer: fake input layer to add constant
    :return: tf type
    """
    if is_numpy(obj):
        if obj.dtype == np.int64:
            obj = np.int32(obj)

        def target_layer(_, inp=obj, dtype=obj.dtype.name):
            import numpy as np
            import tensorflow as tf

            if not isinstance(inp, (np.ndarray, np.generic)):
                inp = np.array(inp, dtype=dtype)
            return tf.constant(inp, dtype=inp.dtype)

        lambda_layer = keras.layers.Lambda(target_layer, name=name)
        return lambda_layer(fake_input_layer)
    else:
        return obj
