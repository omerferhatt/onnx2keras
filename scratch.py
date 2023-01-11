import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
from tensorflow import keras

onnx_model = onnx.load("model.onnx")

k_model: keras.Model = onnx_to_keras(onnx_model, ["input"])
k_model.summary()
k_model.save("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
