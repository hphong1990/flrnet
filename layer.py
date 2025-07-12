from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow as tf
import numpy as np

# Create a custom Conv block layer class instead of using functions
class ConvBlock(layers.Layer):
    def __init__(self, feat_dim, reps=1, kernel_size=3, mode='normal', **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.reps = reps
        self.kernel_size = kernel_size
        self.mode = mode
        
        self.pool = None
        if self.mode == 'down':
            self.pool = layers.MaxPooling2D(2, 2)
        elif self.mode == 'up':
            self.up = layers.UpSampling2D((2, 2), interpolation='bilinear')
            
        # Create layers for each repetition
        self.conv_layers = []
        self.leaky_relus = []
        self.conv1x1_layers = []
        self.leaky_relu1x1s = []
        
        for _ in range(self.reps):
            self.conv_layers.append(layers.Conv2D(self.feat_dim, self.kernel_size, padding="same"))
            self.leaky_relus.append(layers.LeakyReLU(0.2))
            self.conv1x1_layers.append(layers.Conv2D(self.feat_dim, 1, padding="same"))
            self.leaky_relu1x1s.append(layers.LeakyReLU(0.2))
    
    def call(self, inputs, training=None):
        x = inputs
        if self.mode == 'down' and self.pool is not None:
            x = self.pool(x)
        elif self.mode == 'up':
            x = self.up(x)
            
        # Apply conv blocks
        for i in range(self.reps):
            x = self.conv_layers[i](x)
            x = self.leaky_relus[i](x)
            x = self.conv1x1_layers[i](x)
            x = self.leaky_relu1x1s[i](x)
        
        return x

# Define functions that use the layer
def conv_block_down(x, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    """Use ConvBlock class with down mode"""
    # Create the block with an explicit name scope to avoid variable creation issues
    block = ConvBlock(feat_dim=feat_dim, reps=reps, kernel_size=kernel_size, 
                     mode='down' if mode == 'down' else 'normal', 
                     name=f"down_block_{feat_dim}")
    return block(x)

# Create a class for up blocks with concatenation
class UpBlockWithConcat(layers.Layer):
    def __init__(self, feat_dim, reps=1, kernel_size=3, mode='normal', **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.reps = reps
        self.kernel_size = kernel_size
        self.mode = mode
        
        if self.mode == 'up':
            self.up = layers.UpSampling2D((2, 2), interpolation='bilinear')
            
        self.concat = layers.Concatenate()
            
        # Create layers for each repetition
        self.conv_layers = []
        self.leaky_relus = []
        self.conv1x1_layers = []
        self.leaky_relu1x1s = []
        
        for _ in range(self.reps):
            self.conv_layers.append(layers.Conv2D(self.feat_dim, self.kernel_size, padding="same"))
            self.leaky_relus.append(layers.LeakyReLU(0.2))
            self.conv1x1_layers.append(layers.Conv2D(self.feat_dim, 1, padding="same"))
            self.leaky_relu1x1s.append(layers.LeakyReLU(0.2))
    
    def call(self, inputs, training=None):
        x, x1 = inputs
        if self.mode == 'up':
            x = self.up(x)
            
        x = self.concat([x, x1])
            
        # Apply conv blocks
        for i in range(self.reps):
            x = self.conv_layers[i](x)
            x = self.leaky_relus[i](x)
            x = self.conv1x1_layers[i](x)
            x = self.leaky_relu1x1s[i](x)
        
        return x

# Create a class for up blocks without concatenation
class UpBlockWithoutConcat(layers.Layer):
    def __init__(self, feat_dim, reps=1, kernel_size=3, mode='normal', **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.reps = reps
        self.kernel_size = kernel_size
        self.mode = mode
        
        if self.mode == 'up':
            self.up = layers.UpSampling2D((2, 2), interpolation='bilinear')
            
        # Create layers for each repetition
        self.conv_layers = []
        self.leaky_relus = []
        self.conv1x1_layers = []
        self.leaky_relu1x1s = []
        
        for _ in range(self.reps):
            self.conv_layers.append(layers.Conv2D(self.feat_dim, self.kernel_size, padding="same"))
            self.leaky_relus.append(layers.LeakyReLU(0.2))
            self.conv1x1_layers.append(layers.Conv2D(self.feat_dim, 1, padding="same"))
            self.leaky_relu1x1s.append(layers.LeakyReLU(0.2))
    
    def call(self, inputs, training=None):
        x = inputs
        if self.mode == 'up':
            x = self.up(x)
            
        # Apply conv blocks
        for i in range(self.reps):
            x = self.conv_layers[i](x)
            x = self.leaky_relus[i](x)
            x = self.conv1x1_layers[i](x)
            x = self.leaky_relu1x1s[i](x)
        
        return x

# Define functions that use these layers
def conv_block_up_w_concat(x, x1, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    """Use UpBlockWithConcat class"""
    block = UpBlockWithConcat(feat_dim=feat_dim, reps=reps, kernel_size=kernel_size, 
                             mode='up' if mode == 'up' else 'normal',
                             name=f"up_concat_block_{feat_dim}")
    return block([x, x1])

def conv_block_up_wo_concat(x, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    """Use UpBlockWithoutConcat class"""
    block = UpBlockWithoutConcat(feat_dim=feat_dim, reps=reps, kernel_size=kernel_size, 
                                mode='up' if mode == 'up' else 'normal',
                                name=f"up_block_{feat_dim}")
    return block(x)

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