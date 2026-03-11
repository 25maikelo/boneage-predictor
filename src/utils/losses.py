"""
Funciones de pérdida personalizadas para el predictor de edad ósea.
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def attention_loss(y_true, y_pred):
    """
    Pérdida de atención: pondera el error absoluto según la discrepancia relativa.
    loss = mean( |y_true - y_pred| * |(y_pred / y_true) - 1| )
    """
    error = tf.abs(y_true - y_pred)
    alpha = tf.abs((y_pred / y_true) - 1)
    return tf.reduce_mean(alpha * error)


@tf.keras.utils.register_keras_serializable()
def dynamic_attention_loss(y_true, y_pred, k=3.0):
    """
    Pérdida de atención con componente dinámico.
    loss = mean( |y_true - y_pred| * (|(y_pred / (y_true + eps)) - 1|) ** k )
    """
    epsilon = 1e-7
    error = tf.abs(y_true - y_pred)
    alpha = tf.abs((y_pred / (y_true + epsilon)) - 1)
    return tf.reduce_mean(tf.pow(alpha, k) * error)


@tf.keras.utils.register_keras_serializable()
def custom_mse_loss(y_true, y_pred):
    """Error cuadrático medio personalizado."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def custom_huber_loss(y_true, y_pred, delta=1.0):
    """
    Pérdida de Huber: cuadrática para errores pequeños, lineal para grandes.
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    is_small = abs_error <= delta
    small_loss = 0.5 * tf.square(error)
    large_loss = delta * abs_error - 0.5 * tf.square(delta)
    return tf.reduce_mean(tf.where(is_small, small_loss, large_loss))


LOSS_MAP = {
    "attention_loss": attention_loss,
    "dynamic_attention_loss": dynamic_attention_loss,
    "custom_mse_loss": custom_mse_loss,
    "custom_huber_loss": custom_huber_loss,
}
