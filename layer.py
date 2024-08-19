from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *
import tensorflow as tf
import numpy as np

def conv_unit(feat_dim, kernel_size, x_in, padding="CONSTANT"):
    """
    Conv unit: x_in --> Conv k x k + relu --> Conv 1 x 1 + relu --> output
    Parameter: 
                - x_in (tensor): input tensor
                - feat_dim (int): number of channels
                - kernel_size (k) (int): size of convolution kernel
                - padding (str): padding method to use
    Return:
                - (tensor): output of the conv unit
    """
    x = Conv2D(feat_dim, kernel_size, activation=LeakyReLU(0.2), padding="same")(x_in)
    x = Conv2D(feat_dim, 1, activation=LeakyReLU(0.2), padding="same")(x)
    return x

def conv_block_down(x, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    if mode == 'down':
        x = MaxPooling2D(2,2)(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x, padding)
    return x

def conv_block_up_w_concat(x, x1, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    if mode == 'up':
        x = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = Concatenate()([x,x1])
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x, padding)
    return x

def conv_block_up_wo_concat(x, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    if mode == 'up':
        x = UpSampling2D((2,2),interpolation='bilinear')(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x, padding)
    return x

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class Binary2RGB(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)