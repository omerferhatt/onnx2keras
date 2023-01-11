from tensorflow import keras
import numpy as np


def check_torch_keras_error(
    model, k_model, input_np, epsilon=1e-5, change_ordering=False
):
    """
    Check difference between Torch and Keras models
    :param model: torch model
    :param k_model: keras model
    :param input_np: input data as numpy array or list of numpy array
    :param epsilon: allowed difference
    :param change_ordering: change ordering for keras input
    :return: actual difference
    """

    from torch import FloatTensor
    from torch.autograd import Variable

    initial_keras_image_format = keras.backend.image_data_format()

    if isinstance(input_np, np.ndarray):
        input_np = [input_np.astype(np.float32)]

    input_var = [Variable(FloatTensor(i)) for i in input_np]
    pytorch_output = model(*input_var)
    if not isinstance(pytorch_output, tuple):
        pytorch_output = [pytorch_output.data.numpy()]
    else:
        pytorch_output = [p.data.numpy() for p in pytorch_output]

    if change_ordering:
        # change image data format

        _input_np = []
        for i in input_np:
            axes = list(range(len(i.shape)))
            axes = axes[0:1] + axes[2:] + axes[1:2]
            _input_np.append(np.transpose(i, axes))
        input_np = _input_np

        # run keras model
        keras_output = k_model.predict(input_np)
        if not isinstance(keras_output, list):
            keras_output = [keras_output]

        # change image data format if output shapes are different (e.g. the same for global_avgpool2d)
        _koutput = []
        for i, k in enumerate(keras_output):
            if k.shape != pytorch_output[i].shape:
                axes = list(range(len(k.shape)))
                axes = axes[0:1] + axes[-1:] + axes[1:-1]
                k = np.transpose(k, axes)
            _koutput.append(k)
        keras_output = _koutput
    else:
        keras_output = k_model.predict(input_np)
        if not isinstance(keras_output, list):
            keras_output = [keras_output]

    max_error = 0
    for p, k in zip(pytorch_output, keras_output):
        error = np.max(np.abs(p - k))
        np.testing.assert_allclose(p, k, atol=epsilon, rtol=0.0)
        if error > max_error:
            max_error = error

    return max_error
