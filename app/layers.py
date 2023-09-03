# Custom L1 Distance layer module

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance layer


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, anchor_embedding, validation_embedding):
        return tf.math.abs(anchor_embedding - validation_embedding)
