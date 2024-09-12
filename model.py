from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *
import tensorflow as tf
import numpy as np
import layer as flr_layer

# Encoder
def encoder(latent_dims = 4, input_shape = (128,256,1), n_base_features = 64):
    inputs = keras.Input(shape = input_shape)
    conv1 = flr_layer.conv_block_down(inputs,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv2 = flr_layer.conv_block_down(conv1,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv3 = flr_layer.conv_block_down(conv2,
                            feat_dim = n_base_features*2,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')
    conv4 = flr_layer.conv_block_down(conv3,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')
    conv5 = flr_layer.conv_block_down(conv4,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')   
    
    z_mean = layers.Conv2D(latent_dims,3, padding="same",name="z_mean")(conv5)
    z_log_var = layers.Conv2D(latent_dims,3, padding="same",name="z_log_var")(conv5)
    z = flr_layer.Sampling()([z_mean,z_log_var])
    encoder = keras.Model(inputs, [z_mean,z_log_var,z])
    return encoder

# Decoder
def decoder(input_shape = (4,8,4), n_base_features = 64):
    inputs = keras.Input(shape = input_shape)
    conv_in = layers.Conv2D(n_base_features*4, 3, activation = LeakyReLU(0.2), padding="same")(inputs)

    conv1 = flr_layer.conv_block_up_wo_concat(conv_in,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'up')
    conv2 = flr_layer.conv_block_up_wo_concat(conv1,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'up')
    conv3 = flr_layer.conv_block_up_wo_concat(conv2,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv4 = flr_layer.conv_block_up_wo_concat(conv3,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv5 = flr_layer.conv_block_up_wo_concat(conv4,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv_out = layers.Conv2D(1, 3, padding="same")(conv5)
    decoder = keras.Model(inputs, conv_out)
    return decoder

def sensor_mapping(no_of_sensor = 8, latent_dim = (4,8,4)):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(128, activation=LeakyReLU(0.2))(inputs)
    bn_1 = BatchNormalization()(fc_1)
    fc_2 = Dense(256, activation=LeakyReLU(0.2))(bn_1)
    bn_2 = BatchNormalization()(fc_2)

    fc_3 = Dense(512, activation=LeakyReLU(0.2))(bn_2)
    bn_3 = BatchNormalization()(fc_3)


    fc_4 = Dense(256, activation=LeakyReLU(0.2))(bn_3)
    bn_4 = BatchNormalization()(fc_4)

    fc_5 = Dense(latent_dim[0]*latent_dim[1]*latent_dim[2])(bn_4)
    latent_var = Reshape(target_shape=latent_dim)(fc_5)
    z_mean = layers.Conv2D(latent_dim[2],3, padding="same",name="z_mean")(latent_var)
    z_log_var = layers.Conv2D(latent_dim[2],3, padding="same",name="z_log_var")(latent_var)
    z = flr_layer.Sampling()([z_mean,z_log_var])
    mapping = keras.Model(inputs, [z_mean,z_log_var,z])
    return mapping
  
def vgg():
    inputs = keras.Input(shape = (128, 256,1))
    rgb = flr_layer.Binary2RGB()(inputs)
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
    def __init__(self,  n_sensor = 8, input_shape = (128, 256, 1), latent_shape = (4,8,4), **kwargs): # Latent shape = input shape /32, and has to be 4 feature
        super().__init__(**kwargs)
        self.encoder = encoder(input_shape = input_shape)
        self.encoder.trainable = False
        self.decoder = decoder(input_shape = latent_shape)
        self.decoder.trainable = False
        self.sens_mapping = sensor_mapping(no_of_sensor = n_sensor)
        self.vgg19 = vgg()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.sens_reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss_sens"
        )
        self.sens_kl_loss_tracker = keras.metrics.Mean(name="kl_loss_sens")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
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
            z_mean_ae, z_log_var_ae, _ = self.encoder(img_inp)
            
            # Sens recon
            z_mean_sens, z_log_var_sens, z_sens = self.sens_mapping(sens_inp)
            reconstruction_sens = self.decoder(z_sens)
            reconstruction_loss_sens = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(reconstruction_sens,img_inp)
            
            kl_loss_sens = self.kld(z_mean_sens,z_mean_ae,z_log_var_sens, z_log_var_ae)
            kl_loss_sens = (tf.reduce_sum(kl_loss_sens, axis=(1,2,3)))

            perceptual_loss_sens = self.perceptual_loss(reconstruction_sens,img_inp)

            total_loss = reconstruction_loss_sens + kl_loss_sens + perceptual_loss_sens
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.sens_reconstruction_loss_tracker.update_state(reconstruction_loss_sens)
        self.sens_kl_loss_tracker.update_state(kl_loss_sens)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_sens": self.sens_reconstruction_loss_tracker.result(),
            "kl_loss_sens": self.sens_kl_loss_tracker.result(),
        }
    

# Trainer class
class VAE(keras.Model):
    def __init__(self, input_shape = (128, 256, 1), latent_shape = (4,8,4),**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder(input_shape = input_shape)
        self.decoder = decoder(input_shape = latent_shape)
        self.vgg19 = vgg()
    
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss_ae"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss_ae")

        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
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
        # sens_inp = tf.cast(data[0], dtype = tf.float32)

        img_inp = tf.cast(data,dtype = tf.float32)
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

            perceptual_loss_ae = self.perceptual_loss(reconstruction_ae,img_inp)

            total_loss = reconstruction_loss_ae + kl_loss_ae + perceptual_loss_ae
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_ae)
        self.kl_loss_tracker.update_state(kl_loss_ae)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_ae": self.reconstruction_loss_tracker.result(),
            "kl_loss_ae": self.kl_loss_tracker.result(),
        }