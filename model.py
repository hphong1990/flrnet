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

# Encoder
def vgg_encoder(latent_dims = 4, input_shape = (128,256,1), n_base_features = 64):
    inputs = keras.Input(shape = input_shape)
    conv1 = conv_block_down(inputs,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv2 = conv_block_down(conv1,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv3 = conv_block_down(conv2,
                            feat_dim = n_base_features*2,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')
    conv4 = conv_block_down(conv3,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')
    conv5 = conv_block_down(conv4,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')   
    
    z_mean = layers.Conv2D(latent_dims,3, padding="same",name="z_mean")(conv5)
    z_log_var = layers.Conv2D(latent_dims,3, padding="same",name="z_log_var")(conv5)
    z = Sampling()([z_mean,z_log_var])
    encoder = keras.Model(inputs, [z_mean,z_log_var,z])
    return encoder

# Decoder
def vgg_decoder(input_shape = (4,8,4), n_base_features = 64):
    inputs = keras.Input(shape = input_shape)
    conv_in = layers.Conv2D(n_base_features*4, 3, activation = LeakyReLU(0.2), padding="same")(inputs)

    conv1 = conv_block_up_wo_concat(conv_in,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'up')
    conv2 = conv_block_up_wo_concat(conv1,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'up')
    conv3 = conv_block_up_wo_concat(conv2,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv4 = conv_block_up_wo_concat(conv3,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv5 = conv_block_up_wo_concat(conv4,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv_out = layers.Conv2D(1, 3, padding="same")(conv5)
    decoder = keras.Model(inputs, conv_out)
    return decoder

def create_mapping_operator(no_of_sensor = 8, latent_dim = (4,8,4)):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(128, activation=LeakyReLU(0.2))(inputs)
    fc_2 = Dense(256, activation=LeakyReLU(0.2))(fc_1)
    fc_3 = Dense(512, activation=LeakyReLU(0.2))(fc_2)
    fc_3 = Dense(256, activation=LeakyReLU(0.2))(fc_2)
    fc_4 = Dense(128)(fc_3)
    latent_var = Reshape(target_shape=latent_dim)(fc_4)
    z_mean = layers.Conv2D(latent_dim[2],3, padding="same",name="z_mean")(latent_var)
    z_log_var = layers.Conv2D(latent_dim[2],3, padding="same",name="z_log_var")(latent_var)
    z = Sampling()([z_mean,z_log_var])
    mapping = keras.Model(inputs, [z_mean,z_log_var,z])
    return mapping

class Binary2RGB(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)
    
def vgg():
    inputs = keras.Input(shape = (128, 256,1))
    rgb = Binary2RGB()(inputs)
    vgg = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                       weights='imagenet',
                                                       input_shape = (128,256,3),   
                                                       pooling=max)
    feature = vgg(rgb)
    vgg_model = keras.Model(inputs, feature)
    vgg_model.trainable = False
    return vgg_model

# Trainer class
class FLRNet(keras.Model):
    def __init__(self,  n_sensor = 8, **kwargs):
        super().__init__(**kwargs)
        self.encoder = vgg_encoder()
        self.decoder = vgg_decoder()
        self.sens_mapping = create_mapping_operator(no_of_sensor=n_sensor)
        # self.vgg19 = vgg()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss_ae"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss_ae")
        self.sens_reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss_sens"
        )
        self.sens_kl_loss_tracker = keras.metrics.Mean(name="kl_loss_sens")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.sens_reconstruction_loss_tracker,
            self.sens_kl_loss_tracker,
        ]
    def kld(self, mean_1, mean_2, z_log_var_1, z_log_var_2):
        var_1 = tf.exp(z_log_var_1)
        var_2 = tf.exp(z_log_var_2)
        kl_loss = ( 
            tf.math.log((var_2 / var_1) ** 0.5) 
              + (var_1 + (mean_1 - mean_2) ** 2) / (2 * var_2) 
              - 0.5
           )
        return kl_loss
    
    def perceptual_loss(self, y_pred, gt):
        # Pred perceptaul
        pred_feature = self.vgg19(y_pred)
        # GT perceptual
        gt_feature = self.vgg19(gt)
        return tf.keras.losses.MeanSquaredError(reduction = 'sum')(pred_feature,gt_feature)
    
    def train_step(self, data):
        sens_inp = tf.cast(data[0], dtype = tf.float32)
        img_inp = tf.cast(data[1],dtype = tf.float32)
        with tf.GradientTape() as tape:
            # Autoencoder
            z_mean_ae, z_log_var_ae, z_ae = self.encoder(img_inp)
            reconstruction_ae = self.decoder(z_ae)
            reconstruction_loss_ae = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(reconstruction_ae,img_inp)
            
            kl_loss_ae = -0.5 * (1 + z_log_var_ae - tf.square(z_mean_ae) - tf.exp(z_log_var_ae))
            kl_loss_ae = (tf.reduce_sum(kl_loss_ae, axis=(1,2,3)))


            # # Metric loss
            # u_dot = img_inp[:,:,:,1] - img_inp[:,:,:,0] #Compute u_dot
            # z_dot_enc = z_mean_ae_1 - z_mean_ae_2
            # with tf.autodiff.ForwardAccumulator(
            #     primals= self.encoder.trainable_weights,
            #     tangents= u_dot) as acc:
            #     z_mean_metric, _, _ = self.encoder(img_inp)
            # z_dot_compute = acc.jvp(z_mean_metric)
            # metric_loss = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(z_dot_compute, z_dot_enc)

            # perceptual_loss_ae = self.perceptual_loss(reconstruction_ae,img_inp)

            # Sens recon
            z_mean_sens, z_log_var_sens, z_sens = self.sens_mapping(sens_inp)
            reconstruction_sens = self.decoder(z_sens)
            reconstruction_loss_sens = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(reconstruction_sens,img_inp)
            
            kl_loss_sens = self.kld(z_mean_sens,z_mean_ae,z_log_var_sens, z_log_var_ae)
            kl_loss_sens = (tf.reduce_sum(kl_loss_sens, axis=(1,2,3)))

            # perceptual_loss_sens = self.perceptual_loss(reconstruction_sens,img_inp)

            total_loss = reconstruction_loss_ae + kl_loss_ae + 2*reconstruction_loss_sens + 2*kl_loss_sens 
            # + perceptual_loss_ae + perceptual_loss_sens
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_ae)
        self.kl_loss_tracker.update_state(kl_loss_ae)
        self.sens_reconstruction_loss_tracker.update_state(reconstruction_loss_sens)
        self.sens_kl_loss_tracker.update_state(kl_loss_sens)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_ae": self.reconstruction_loss_tracker.result(),
            "kl_loss_ae": self.kl_loss_tracker.result(),
            "reconstruction_loss_sens": self.sens_reconstruction_loss_tracker.result(),
            "kl_loss_sens": self.sens_kl_loss_tracker.result(),
        }