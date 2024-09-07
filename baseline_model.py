from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *
import tensorflow as tf
import numpy as np
import layer as flr_layer

# def MLP_CNN(no_of_sensor = 8, latent_dim = (4,8,32), n_base_features = 64):
#     inputs = keras.Input(shape = (no_of_sensor))
#     fc_1 = Dense(128, activation=LeakyReLU(0.2))(inputs)
#     fc_2 = Dense(256, activation=LeakyReLU(0.2))(fc_1)
#     fc_3 = Dense(512, activation=LeakyReLU(0.2))(fc_2)
#     fc_4 = Dense(1024, activation=LeakyReLU(0.2))(fc_3)
#     # fc_5 = Dense(2048, activation=LeakyReLU(0.2))(fc_4)
#     # fc_6 = Dense(4096, activation=LeakyReLU(0.2))(fc_5)
#     # fc_7 = Dense(8192, activation=LeakyReLU(0.2))(fc_6)

#     latent_var = Reshape(target_shape=latent_dim)(fc_4)
#     conv_in = layers.Conv2D(n_base_features*4, 3, activation = LeakyReLU(0.2), padding="same")(latent_var)

#     conv1 = flr_layer.conv_block_up_wo_concat(conv_in,
#                             feat_dim = n_base_features*4,
#                             reps = 2,
#                             kernel_size = 3,
#                             mode = 'up')
#     conv2 = flr_layer.conv_block_up_wo_concat(conv1,
#                             feat_dim = n_base_features*4,
#                             reps = 2,
#                             kernel_size = 3,
#                             mode = 'up')
#     conv3 = flr_layer.conv_block_up_wo_concat(conv2,
#                             feat_dim = n_base_features*2,
#                             reps = 1,
#                             kernel_size = 3,
#                             mode = 'up')
#     conv4 = flr_layer.conv_block_up_wo_concat(conv3,
#                             feat_dim = n_base_features*2,
#                             reps = 1,
#                             kernel_size = 3,
#                             mode = 'up')
#     conv5 = flr_layer.conv_block_up_wo_concat(conv4,
#                             feat_dim = n_base_features,
#                             reps = 1,
#                             kernel_size = 3,
#                             mode = 'up')
#     conv_out = layers.Conv2D(1, 3, padding="same")(conv5)
#     decoder = keras.Model(inputs, conv_out)
#     return decoder

def MLP(no_of_sensor = 8, output_shape = (128,256,1), n_base_features = 64):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(64, activation='relu', 
                 kernel_regularizer=regularizers.L2(1e-3), 
                 bias_regularizer=regularizers.L2(1e-3))(inputs)
    bn_1 = BatchNormalization()(fc_1)
    fc_2 = Dense(128, activation='relu',
                 kernel_regularizer=regularizers.L2(1e-3), 
                 bias_regularizer=regularizers.L2(1e-3))(bn_1)
    bn_2 = BatchNormalization()(fc_2)

    fc_3 = Dense(256, activation='relu',
                 kernel_regularizer=regularizers.L2(1e-3), 
                 bias_regularizer=regularizers.L2(1e-3))(bn_2)
    bn_3 = BatchNormalization()(fc_3)

    fc_5 = Dense(output_shape[0]*output_shape[1]*output_shape[2])(bn_3)

    output = Reshape(target_shape=output_shape)(fc_5)
    
    mlp = keras.Model(inputs, output)
    return mlp
  
## Autoencoder baseline

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
    
    encoder = keras.Model(inputs, z_mean)
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

def mapping_operator(no_of_sensor = 8, latent_dim = (4,8,4)):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(128, activation=LeakyReLU(0.2))(inputs)
    bn_1 = BatchNormalization()(fc_1)
    fc_2 = Dense(256, activation=LeakyReLU(0.2))(bn_1)
    bn_2 = BatchNormalization()(fc_2)

    fc_3 = Dense(512, activation=LeakyReLU(0.2))(bn_2)
    bn_3 = BatchNormalization()(fc_3)


    fc_4 = Dense(256, activation=LeakyReLU(0.2))(bn_3)
    bn_4 = BatchNormalization()(fc_4)

    fc_5 = Dense(128)(bn_4)
    latent_var = Reshape(target_shape=latent_dim)(fc_5)
    z_mean = layers.Conv2D(latent_dim[2],3, padding="same",name="z_mean")(latent_var)

    mapping = keras.Model(inputs, z_mean)
    return mapping
  

# Trainer class
class AE(keras.Model):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder()
        self.decoder = decoder()
    
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss_ae"
        )

        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]
    
    def train_step(self, data):
        # sens_inp = tf.cast(data[0], dtype = tf.float32)

        img_inp = tf.cast(data[1],dtype = tf.float32)
        with tf.GradientTape() as tape:
            # Autoencoder
            z = self.encoder(img_inp)
            reconstruction_ae = self.decoder(z)
            reconstruction_loss_ae = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(reconstruction_ae,img_inp)
            total_loss = reconstruction_loss_ae 
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_ae)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_ae": self.reconstruction_loss_tracker.result(),
        }

# Trainer class
class FLRNetAE(keras.Model):
    def __init__(self,  n_sensor = 8, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder()
        self.encoder.trainable = False
        self.decoder = decoder()
        self.decoder.trainable = False
        self.sens_mapping = mapping_operator(no_of_sensor=n_sensor)   

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.sens_reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss_sens"
        )
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.sens_reconstruction_loss_tracker,
        ]
    
    def train_step(self, data):
        sens_inp = tf.cast(data[0], dtype = tf.float32)
        img_inp = tf.cast(data[1],dtype = tf.float32)
        with tf.GradientTape() as tape:
            # Autoencoder
            z = self.encoder(img_inp)
            
            # Sens recon
            z_sens = self.sens_mapping(sens_inp)
            reconstruction_sens = self.decoder(z_sens)
            reconstruction_loss_sens = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(reconstruction_sens,img_inp)
            
            latent_loss = tf.keras.losses.MeanSquaredError(reduction = 'sum')(z, z_sens)
            total_loss = reconstruction_loss_sens + latent_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.sens_reconstruction_loss_tracker.update_state(reconstruction_loss_sens)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_sens": self.sens_reconstruction_loss_tracker.result(),
        }
    
# POD-based method
def MLP_POD(no_of_sensor = 8, output_shape = 128, n_base_features = 64):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(128, activation=LeakyReLU(0.2))(inputs)
    bn_1 = BatchNormalization()(fc_1)
    fc_2 = Dense(256, activation=LeakyReLU(0.2))(bn_1)
    bn_2 = BatchNormalization()(fc_2)

    fc_3 = Dense(512, activation=LeakyReLU(0.2))(bn_2)
    bn_3 = BatchNormalization()(fc_3)


    fc_4 = Dense(256, activation=LeakyReLU(0.2))(bn_3)
    bn_4 = BatchNormalization()(fc_4)

    fc_5 = Dense(output_shape)(bn_4)
    
    mlp_pod = keras.Model(inputs, fc_5)
    return mlp_pod