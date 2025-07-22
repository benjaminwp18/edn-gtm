from keras import backend
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model

from .config import IMG_SHAPE


# Deprecated Keras util function
def keras_mean(x, axis=None, keepdims=False):
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, backend.floatx())
    return tf.reduce_mean(x, axis, keepdims)


def l2_loss(y_true, y_pred):
    return keras_mean((y_pred - y_true) ** 2)


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
    loss_model.trainable = False
    return keras_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return keras_mean(y_true * y_pred)


def perceptual_and_l2_loss(y_true, y_pred):
    return 0.5 * perceptual_loss(y_true, y_pred) + 0.5 * l2_loss(y_true, y_pred)
