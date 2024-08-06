from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *
import tensorflow as tf
import numpy as np
import layer as flr_layer

def MLP_CNN(no_of_sensor = 8, latent_dim = (4,8,32), n_base_features = 64):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(128, activation=LeakyReLU(0.2))(inputs)
    fc_2 = Dense(256, activation=LeakyReLU(0.2))(fc_1)
    fc_3 = Dense(512, activation=LeakyReLU(0.2))(fc_2)
    fc_4 = Dense(1024, activation=LeakyReLU(0.2))(fc_3)
    # fc_5 = Dense(2048, activation=LeakyReLU(0.2))(fc_4)
    # fc_6 = Dense(4096, activation=LeakyReLU(0.2))(fc_5)
    # fc_7 = Dense(8192, activation=LeakyReLU(0.2))(fc_6)

    latent_var = Reshape(target_shape=latent_dim)(fc_4)
    conv_in = layers.Conv2D(n_base_features*4, 3, activation = LeakyReLU(0.2), padding="same")(latent_var)

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

def MLP(no_of_sensor = 8, output_shape = (128,256,1), n_base_features = 64):
    inputs = keras.Input(shape = (no_of_sensor))
    fc_1 = Dense(64, activation=LeakyReLU(0.2))(inputs)
    fc_2 = Dense(128, activation=LeakyReLU(0.2))(fc_1)
    fc_5 = Dense(output_shape[0]*output_shape[1]*output_shape[2])(fc_2)
    output = Reshape(target_shape=output_shape)(fc_5)
    
    mlp = keras.Model(inputs, output)
    return mlp
  
