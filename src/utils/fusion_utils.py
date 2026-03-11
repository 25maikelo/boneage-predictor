"""
Capas personalizadas para el modelo de fusión.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Reshape


@tf.keras.utils.register_keras_serializable()
class AttentionFusion(tf.keras.layers.Layer):
    """Mecanismo de atención entre segmentos para el modelo de fusión."""

    def __init__(self, num_segments, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.dense = Dense(1)
        self.softmax = Softmax()
        self.reshape1 = Reshape((num_segments,))
        self.reshape2 = Reshape((num_segments, 1))

    def call(self, inputs):
        concatenated = tf.stack(inputs, axis=1)           # (batch, num_seg, feat_dim)
        logits = self.dense(concatenated)                  # (batch, num_seg, 1)
        logits = self.reshape1(logits)                     # (batch, num_seg)
        weights = self.softmax(logits)                     # (batch, num_seg)
        weights_exp = self.reshape2(weights)               # (batch, num_seg, 1)
        weighted = concatenated * weights_exp
        fused = tf.reduce_sum(weighted, axis=1)
        return fused, weights

    def get_config(self):
        cfg = super().get_config()
        cfg["num_segments"] = self.num_segments
        return cfg


class SavedModelWrapper(tf.keras.layers.Layer):
    """Envuelve un SavedModel como capa Keras."""

    def __init__(self, saved_model_path, **kwargs):
        super().__init__(**kwargs)
        self.saved_model = tf.saved_model.load(saved_model_path)
        self.infer = self.saved_model.signatures.get("serving_default")
        if self.infer is None:
            raise ValueError(
                f"El SavedModel en {saved_model_path} no tiene la firma 'serving_default'"
            )
        self.input_key = list(self.infer.structured_input_signature[1].keys())[0]

    def call(self, inputs):
        outputs = self.infer(**inputs)
        return list(outputs.values())[0]
