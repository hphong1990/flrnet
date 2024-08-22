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

class FourierFeature(tf.keras.layers.Layer):

    def __init__(self, gaussian_projection: int, gaussian_scale: float = 1.0, **kwargs):
        """
        Fourier Feature Projection layer from the paper:
        [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)

        Add this layer immediately after the input layer.

        Args:
            gaussian_projection: Projection dimension for the gaussian kernel in fourier feature
                projection layer. Can be negative or positive integer.
                If <=0, uses identity matrix (basic projection) without gaussian kernel.
                If >=1, uses gaussian projection matrix of specified dim.
            gaussian_scale: Scale of the gaussian kernel in fourier feature projection layer.
                Note: If the scale is too small, convergence will slow down and obtain poor results.
                If the scale is too large (>50), convergence will be fast but results will be grainy.
                Try grid search for scales in the range [10 - 50].
        """
        super().__init__(**kwargs)

        if 'dtype' in kwargs:
            self._kernel_dtype = kwargs['dtype']
        else:
            self._kernel_dtype = None

        gaussian_projection = int(gaussian_projection)
        gaussian_scale = float(gaussian_scale)

        self.gauss_proj = gaussian_projection
        self.gauss_scale = gaussian_scale

    def build(self, input_shape):
        # assume channel dim is always at last location
        input_dim = input_shape[-1]

        if self.gauss_proj <= 0:
            # Assume basic projection
            self.proj_kernel = tf.keras.layers.Conv2D(input_dim, 1, padding="same",
                                                       use_bias=False, trainable=False,
                                                     kernel_initializer='identity', dtype=self._kernel_dtype)

        else:
            initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=self.gauss_scale)
            self.proj_kernel = tf.keras.layers.Conv2D(self.gauss_proj, 1, padding = "same", 
                                                     use_bias=False, trainable=False,
                                                     kernel_initializer=initializer, dtype=self._kernel_dtype)

        self.built = True

    def call(self, inputs, **kwargs):
        x_proj = 2.0 * np.pi * inputs
        x_proj = self.proj_kernel(x_proj)

        x_proj_sin = tf.sin(x_proj)
        x_proj_cos = tf.cos(x_proj)

        output = tf.concat([x_proj_sin, x_proj_cos], axis=-1)
        return output

    def get_config(self):
        config = {
            'gaussian_projection': self.gauss_proj,
            'gaussian_scale': self.gauss_scale
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))