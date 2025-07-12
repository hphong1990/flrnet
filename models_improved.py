"""
Improved Flow Field Reconstruction Models with TensorFlow 2.x Best Practices

This module contains optimized VAE and FLRNet models with:
- Modern TensorFlow 2.x architecture
- Automated training pipeline with validation
- Callback support for model checkpointing
- Support for both standard and Fourier-enhanced versions
- Unified training workflow

Implementation now matches the architecture from model_fourier.py:
- FLREncoder: 5 blocks with conv_block_down approach
- FLRDecoder: 5 blocks with conv_block_up_wo_concat approach
- Latent space dimensions: (4,8,4) for 128x256 input (32x downsampling)
- SensorMapping: Exact architecture from model_fourier.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any
import json
import layer as flr_layer


class NaNMonitorCallback(callbacks.Callback):
    """Custom callback to monitor and log NaN/Inf values during training."""
    
    def on_batch_end(self, batch, logs=None):
        """Check for NaN/Inf values after each batch."""
        if logs:
            for key, value in logs.items():
                if np.isnan(value) or np.isinf(value):
                    print(f"\n⚠️  Warning: {key} is {value} at batch {batch}")
                    print("🛑 Training will be terminated due to NaN/Inf loss")
                    self.model.stop_training = True
                    break


class FLREncoder(layers.Layer):
    """Encoder layer for flow field data with modern TF 2.x practices."""
    
    def __init__(self, 
                 latent_dims: int = 4,
                 n_base_features: int = 64,
                 use_fourier: bool = False,
                 name: str = "flr_encoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dims = latent_dims
        self.n_base_features = n_base_features
        self.use_fourier = use_fourier
        
        # Build encoder blocks
        self._build_layers()
    
    def _build_layers(self):
        """Build encoder layers."""
        # Using 5 downsampling blocks to match model_fourier.py
        # This gives 32x reduction: 128/32=4, 256/32=8
        
        # Latent space layers
        self.z_mean_layer = layers.Conv2D(self.latent_dims, 3, padding='same', name='z_mean')
        self.z_log_var_layer = layers.Conv2D(self.latent_dims, 3, padding='same', name='z_log_var')
        self.sampling_layer = flr_layer.Sampling()
        
        # Pre-create ConvBlock instances to avoid variable creation issues
        self.conv_block1 = flr_layer.ConvBlock(feat_dim=self.n_base_features, 
                                             reps=1, kernel_size=3, mode='down', name='conv_block_1')
        self.conv_block2 = flr_layer.ConvBlock(feat_dim=self.n_base_features*2, 
                                             reps=1, kernel_size=3, mode='down', name='conv_block_2')
        self.conv_block3 = flr_layer.ConvBlock(feat_dim=self.n_base_features*2, 
                                             reps=1, kernel_size=3, mode='down', name='conv_block_3')
        self.conv_block4 = flr_layer.ConvBlock(feat_dim=self.n_base_features*4, 
                                             reps=1, kernel_size=3, mode='down', name='conv_block_4')
        self.conv_block5 = flr_layer.ConvBlock(feat_dim=self.n_base_features*4, 
                                             reps=1, kernel_size=3, mode='down', name='conv_block_5')
        
        # Create concatenation layers (used in Fourier mode)
        self.concat1 = layers.Concatenate(name='concat_1')
        self.concat2 = layers.Concatenate(name='concat_2')
        self.concat3 = layers.Concatenate(name='concat_3')
        self.concat4 = layers.Concatenate(name='concat_4')
        self.concat5 = layers.Concatenate(name='concat_5')
        
        # Average pooling for Fourier features
        self.avg_pool = layers.AveragePooling2D()
        
        if self.use_fourier:
            self.fourier_layer = flr_layer.FourierFeature(gaussian_projection=4, gaussian_scale=15)
    
    def call(self, inputs, training=None):
        """Forward pass following model_fourier.py architecture EXACTLY."""
        if self.use_fourier:
            img_input, coord_input = inputs
            fourier_feat = self.fourier_layer(coord_input)
            
            # Block 1: EXACTLY like model_fourier.py - concat FIRST, then conv_block_down
            concat1 = self.concat1([fourier_feat, img_input])  # Original resolution concat
            conv1 = self.conv_block1(concat1)  # Use pre-created ConvBlock instance
            
            # Block 2: Pool fourier features, then concat
            fourier_feat2 = self.avg_pool(fourier_feat)
            concat2 = self.concat2([fourier_feat2, conv1])
            conv2 = self.conv_block2(concat2)
            
            # Block 3
            fourier_feat3 = self.avg_pool(fourier_feat2)
            concat3 = self.concat3([fourier_feat3, conv2])
            conv3 = self.conv_block3(concat3)
            
            # Block 4
            fourier_feat4 = self.avg_pool(fourier_feat3)
            concat4 = self.concat4([fourier_feat4, conv3])
            conv4 = self.conv_block4(concat4)
            
            # Block 5
            fourier_feat5 = self.avg_pool(fourier_feat4)
            concat5 = self.concat5([fourier_feat5, conv4])
            conv5 = self.conv_block5(concat5)
            
            # Latent space
            z_mean = self.z_mean_layer(conv5)
            z_log_var = self.z_log_var_layer(conv5)
            
        else:
            # Standard mode without Fourier features - use pre-created ConvBlock instances
            x = inputs
            
            # Block 1
            x = self.conv_block1(x)
            
            # Block 2
            x = self.conv_block2(x)
            
            # Block 3
            x = self.conv_block3(x)
            
            # Block 4
            x = self.conv_block4(x)
            
            # Block 5
            x = self.conv_block5(x)
            
            # Latent space
            z_mean = self.z_mean_layer(x)
            z_log_var = self.z_log_var_layer(x)
            
        # Sample z from latent distribution
        z = self.sampling_layer([z_mean, z_log_var])
        
        return z_mean, z_log_var, z
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dims': self.latent_dims,
            'n_base_features': self.n_base_features,
            'use_fourier': self.use_fourier
        })
        return config


class FLRDecoder(layers.Layer):
    """Decoder layer for flow field data with modern TF 2.x practices."""
    
    def __init__(self, 
                 n_base_features: int = 64,
                 use_fourier: bool = False,
                 output_channels: int = 1,
                 name: str = "flr_decoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_base_features = n_base_features
        self.use_fourier = use_fourier
        self.output_channels = output_channels
        
        self._build_layers()
    
    def _build_layers(self):
        """Build decoder layers following model_fourier.py."""
        # In model_fourier.py, we use Conv2D + LeakyReLU for initial processing
        self.conv_in = layers.Conv2D(self.n_base_features * 4, 3, padding='same', name='conv_in')
        self.leaky_in = layers.LeakyReLU(0.2)
        self.conv_out = layers.Conv2D(self.output_channels, 3, padding='same', name='conv_out')
        
        # Pre-create UpBlockWithoutConcat instances to avoid variable creation issues
        # For Fourier mode (with concatenation)
        self.up_block1 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*4, 
                                                      reps=1, kernel_size=3, mode='up', name='up_block_1')
        self.up_block2 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*4, 
                                                      reps=1, kernel_size=3, mode='up', name='up_block_2')
        self.up_block3 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*2, 
                                                      reps=1, kernel_size=3, mode='up', name='up_block_3')
        self.up_block4 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*2, 
                                                      reps=1, kernel_size=3, mode='up', name='up_block_4')
        self.up_block5 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features, 
                                                      reps=1, kernel_size=3, mode='up', name='up_block_5')
        
        # For standard mode (without concatenation) - different configuration
        self.std_up_block1 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*4, 
                                                          reps=1, kernel_size=3, mode='up', name='std_up_block_1')
        self.std_up_block2 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*4, 
                                                          reps=1, kernel_size=3, mode='up', name='std_up_block_2')
        self.std_up_block3 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*2, 
                                                          reps=1, kernel_size=3, mode='up', name='std_up_block_3')
        self.std_up_block4 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features*2, 
                                                          reps=1, kernel_size=3, mode='up', name='std_up_block_4')
        self.std_up_block5 = flr_layer.UpBlockWithoutConcat(feat_dim=self.n_base_features, 
                                                          reps=1, kernel_size=3, mode='up', name='std_up_block_5')
        
        # Create concatenation layers (used in both Fourier and standard modes)
        self.concat1 = layers.Concatenate(name='decoder_concat_1')
        self.concat2 = layers.Concatenate(name='decoder_concat_2')
        self.concat3 = layers.Concatenate(name='decoder_concat_3')
        self.concat4 = layers.Concatenate(name='decoder_concat_4')
        self.concat5 = layers.Concatenate(name='decoder_concat_5')
        self.concat6 = layers.Concatenate(name='decoder_concat_6')
        
        # Average pooling for Fourier features
        self.avg_pool = layers.AveragePooling2D()
        
        if self.use_fourier:
            self.fourier_layer = flr_layer.FourierFeature(gaussian_projection=4, gaussian_scale=15)
    
    def call(self, inputs, training=None):
        """Forward pass using upsampling blocks following model_fourier.py EXACTLY."""
        if self.use_fourier:
            latent_input, coord_input = inputs
            fourier_feat = self.fourier_layer(coord_input)
            
            # Create 6 levels of downsampled fourier features for 5-block architecture
            # fourier_feat: (128, 256, 8) - Level 0: original resolution
            # fourier_feat2: (64, 128, 8) - Level 1
            # fourier_feat3: (32, 64, 8) - Level 2  
            # fourier_feat4: (16, 32, 8) - Level 3
            # fourier_feat5: (8, 16, 8) - Level 4
            # fourier_feat6: (4, 8, 8) - Level 5 <- matches latent space (4, 8, 4)
            
            fourier_feat2 = self.avg_pool(fourier_feat)
            fourier_feat3 = self.avg_pool(fourier_feat2)
            fourier_feat4 = self.avg_pool(fourier_feat3)
            fourier_feat5 = self.avg_pool(fourier_feat4)
            fourier_feat6 = self.avg_pool(fourier_feat5)
            
            # Initial processing - match model_fourier.py exactly
            conv_in = self.conv_in(latent_input)
            x = self.leaky_in(conv_in)
            
            # Block 1: concat with most downsampled fourier features (level 6)
            concat1 = self.concat1([fourier_feat6, x])
            conv1 = self.up_block1(concat1)
            
            # Block 2
            concat2 = self.concat2([fourier_feat5, conv1])
            conv2 = self.up_block2(concat2)
            
            # Block 3
            concat3 = self.concat3([fourier_feat4, conv2])
            conv3 = self.up_block3(concat3)
            
            # Block 4
            concat4 = self.concat4([fourier_feat3, conv3])
            conv4 = self.up_block4(concat4)
            
            # Block 5
            concat5 = self.concat5([fourier_feat2, conv4])
            conv5 = self.up_block5(concat5)
            
            # Final concatenation with original fourier features
            concat6 = self.concat6([fourier_feat, conv5])
            x = self.conv_out(concat6)
            
        else:
            # Standard mode without Fourier features - use pre-created up blocks
            conv_in = self.conv_in(inputs)
            x = self.leaky_in(conv_in)
            
            # Block 1
            x = self.std_up_block1(x)
            
            # Block 2
            x = self.std_up_block2(x)
            
            # Block 3
            x = self.std_up_block3(x)
            
            # Block 4
            x = self.std_up_block4(x)
            
            # Block 5
            x = self.std_up_block5(x)
            
            # Final output layer
            x = self.conv_out(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_base_features': self.n_base_features,
            'use_fourier': self.use_fourier,
            'output_channels': self.output_channels
        })
        return config


class SensorMapping(layers.Layer):
    """Enhanced sensor mapping with optional attention mechanism (2 blocks)."""
    
    def __init__(self, 
                 n_sensors: int = 8,
                 latent_dim: Tuple[int, int, int] = (4, 8, 4),
                 use_attention: bool = True,
                 attention_heads: int = 2,
                 hidden_dim: int = 128,
                 name: str = "sensor_mapping",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.hidden_dim = hidden_dim
        
        self._build_layers()
    
    def _build_layers(self):
        """Build enhanced sensor mapping with optional attention."""
        # Input processing
        self.noise_layer = layers.GaussianNoise(0.1)
        
        if self.use_attention:
            # === ATTENTION-BASED ARCHITECTURE ===
            
            # 1. Sensor embedding and positional encoding
            self.sensor_embedding = layers.Dense(self.hidden_dim, name='sensor_embedding')
            self.position_embedding = layers.Embedding(
                self.n_sensors, self.hidden_dim, name='position_embedding'
            )
            
            # 2. Two attention blocks (as suggested)
            self.attention_blocks = []
            for i in range(2):  # Only 2 attention blocks
                # Multi-head self-attention
                attention_layer = layers.MultiHeadAttention(
                    num_heads=self.attention_heads,
                    key_dim=self.hidden_dim // self.attention_heads,
                    name=f'sensor_attention_{i}'
                )
                
                # Feed-forward network
                ff_layer_1 = layers.Dense(self.hidden_dim * 2, activation='gelu', name=f'ff_1_{i}')
                ff_layer_2 = layers.Dense(self.hidden_dim, name=f'ff_2_{i}')
                
                # Layer normalizations
                norm_1 = layers.LayerNormalization(name=f'norm_1_{i}')
                norm_2 = layers.LayerNormalization(name=f'norm_2_{i}')
                
                # Dropout for regularization
                dropout_1 = layers.Dropout(0.1, name=f'dropout_1_{i}')
                dropout_2 = layers.Dropout(0.1, name=f'dropout_2_{i}')
                
                self.attention_blocks.append({
                    'attention': attention_layer,
                    'norm_1': norm_1,
                    'norm_2': norm_2,
                    'ff_1': ff_layer_1,
                    'ff_2': ff_layer_2,
                    'dropout_1': dropout_1,
                    'dropout_2': dropout_2
                })
            
            # 3. Global attention pooling
            self.global_attention = layers.MultiHeadAttention(
                num_heads=1,
                key_dim=self.hidden_dim,
                name='global_attention'
            )
            
            # 4. Final projection layers
            self.final_norm = layers.LayerNormalization(name='final_norm')
            self.final_projection = layers.Dense(512, activation='gelu', name='final_projection')
            self.final_dropout = layers.Dropout(0.1)
            
        else:
            # === FALLBACK: ORIGINAL DENSE ARCHITECTURE ===
            self.fc_1 = layers.Dense(256)
            self.leaky_1 = layers.LeakyReLU(0.2)
            self.bn_1 = layers.BatchNormalization()
            
            self.fc_2 = layers.Dense(512)
            self.leaky_2 = layers.LeakyReLU(0.2)
            self.bn_2 = layers.BatchNormalization()
        
            # === COMMON LAYERS (used in both modes) ===
            self.fc_3 = layers.Dense(512)
            self.leaky_3 = layers.LeakyReLU(0.2)
            self.bn_3 = layers.BatchNormalization()
            
            self.fc_4 = layers.Dense(512)
            self.leaky_4 = layers.LeakyReLU(0.2)
            self.bn_4 = layers.BatchNormalization()
        
        # Final projection to latent space
        latent_size = self.latent_dim[0] * self.latent_dim[1] * self.latent_dim[2]
        self.fc_5 = layers.Dense(latent_size)
        self.reshape_layer = layers.Reshape(self.latent_dim)
        
        # Latent space parameters
        self.z_mean_conv = layers.Conv2D(self.latent_dim[2], 3, padding='same', name='z_mean')
        self.z_log_var_conv = layers.Conv2D(self.latent_dim[2], 3, padding='same', name='z_log_var')
        self.sampling_layer = flr_layer.Sampling()
    
    def call(self, inputs, training=None):
        """Forward pass with optional attention mechanism."""
        x = self.noise_layer(inputs, training=training)
        
        if self.use_attention:
            # === ATTENTION-BASED PROCESSING ===
            batch_size = tf.shape(x)[0]
            
            # 1. Convert to sequence format for attention
            # x shape: (batch, n_sensors) → (batch, n_sensors, 1)
            x = tf.expand_dims(x, axis=-1)
            
            # 2. Sensor embedding
            x = self.sensor_embedding(x)  # (batch, n_sensors, hidden_dim)
            
            # 3. Add positional encoding
            positions = tf.range(self.n_sensors)
            positions = tf.expand_dims(positions, 0)  # (1, n_sensors)
            positions = tf.tile(positions, [batch_size, 1])  # (batch, n_sensors)
            pos_encoding = self.position_embedding(positions)  # (batch, n_sensors, hidden_dim)
            
            x = x + pos_encoding  # Add positional information
            
            # 4. Apply 2 attention blocks (as suggested)
            for block in self.attention_blocks:
                # Self-attention with residual connection
                attention_output = block['attention'](x, x, training=training)
                attention_output = block['dropout_1'](attention_output, training=training)
                x = block['norm_1'](x + attention_output)  # Residual + LayerNorm
                
                # Feed-forward with residual connection
                ff_output = block['ff_2'](block['ff_1'](x))
                ff_output = block['dropout_2'](ff_output, training=training)
                x = block['norm_2'](x + ff_output)  # Residual + LayerNorm
            
            # 5. Global attention pooling (learnable aggregation)
            global_query = tf.reduce_mean(x, axis=1, keepdims=True)  # (batch, 1, hidden_dim)
            global_output = self.global_attention(
                query=global_query, 
                key=x, 
                value=x, 
                training=training
            )  # (batch, 1, hidden_dim)
            
            # Squeeze to remove sequence dimension
            x = tf.squeeze(global_output, axis=1)  # (batch, hidden_dim)
            
            # 6. Final processing
            x = self.final_norm(x)
            x = self.final_projection(x)
            x = self.final_dropout(x, training=training)
            
        else:
            # === ORIGINAL DENSE PROCESSING ===
            x = self.fc_1(x)
            x = self.leaky_1(x)
            x = self.bn_1(x, training=training)
            
            x = self.fc_2(x)
            x = self.leaky_2(x)
            x = self.bn_2(x, training=training)
        
            # === COMMON PROCESSING ===
            x = self.fc_3(x)
            x = self.leaky_3(x)
            x = self.bn_3(x, training=training)
            
            x = self.fc_4(x)
            x = self.leaky_4(x)
            x = self.bn_4(x, training=training)
        
        x = self.fc_5(x)
        x = self.reshape_layer(x)
        
        # Generate latent parameters
        z_mean = self.z_mean_conv(x)
        z_log_var = self.z_log_var_conv(x)
        z = self.sampling_layer([z_mean, z_log_var])
        
        return z_mean, z_log_var, z
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_sensors': self.n_sensors,
            'latent_dim': self.latent_dim,
            'use_attention': self.use_attention,
            'attention_heads': self.attention_heads,
            'hidden_dim': self.hidden_dim
        })
        return config


class FLRVAE(keras.Model):
    """Variational Autoencoder for flow field reconstruction."""
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (128, 256, 1),
                 latent_dims: int = 4,
                 n_base_features: int = 64,
                 use_fourier: bool = False,
                 use_perceptual_loss: bool = True,
                 gradient_clip_norm: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape_custom = input_shape
        self.latent_dims = latent_dims
        self.n_base_features = n_base_features
        self.use_fourier = use_fourier
        self.use_perceptual_loss = use_perceptual_loss
        self.gradient_clip_norm = gradient_clip_norm
        
        # Calculate latent shape (32x downsampling for 5-block architecture)
        h, w = input_shape[0] // 32, input_shape[1] // 32
        self.latent_shape = (h, w, latent_dims)
        
        # Build model components
        self.encoder = FLREncoder(
            latent_dims=latent_dims,
            n_base_features=n_base_features,
            use_fourier=use_fourier,
            name='vae_encoder'
        )
        
        self.decoder = FLRDecoder(
            n_base_features=n_base_features,
            use_fourier=use_fourier,
            output_channels=input_shape[2],
            name='vae_decoder'
        )
        
        # Perceptual loss model
        if self.use_perceptual_loss:
            self._build_perceptual_model()
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        if self.use_perceptual_loss:
            self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
    
    def _build_perceptual_model(self):
        """Build perceptual loss model."""
        inputs = keras.Input(shape=self.input_shape_custom)
        rgb = flr_layer.Binary2RGB()(inputs)
        vgg = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(self.input_shape_custom[0], self.input_shape_custom[1], 3),
            pooling='avg'
        )
        vgg.trainable = False
        features = vgg(rgb)
        self.perceptual_model = keras.Model(inputs, features)
    
    def call(self, inputs, training=None):
        """Forward pass."""
        if self.use_fourier:
            img_input, coord_input = inputs
            z_mean, z_log_var, z = self.encoder([img_input, coord_input], training=training)
            reconstruction = self.decoder([z, coord_input], training=training)
        else:
            z_mean, z_log_var, z = self.encoder(inputs, training=training)
            reconstruction = self.decoder(z, training=training)
        
        return reconstruction
    
    def train_step(self, data):
        """Custom training step."""
        if self.use_fourier:
            if isinstance(data, tuple) and len(data) == 2:
                # Data format: ((img, coord), _) or (img, coord)
                if isinstance(data[0], tuple):
                    img_input, coord_input = data[0]
                else:
                    img_input, coord_input = data
            else:
                # Data format: [img, coord] (for some dataset formats)
                img_input, coord_input = data[0], data[1]
            inputs = [img_input, coord_input]
            target = img_input
        else:
            if isinstance(data, tuple) and len(data) == 2:
                inputs = data[0] if not isinstance(data[0], tuple) else data[0][0]
            else:
                inputs = data
            target = inputs
        
        with tf.GradientTape() as tape:
            # Forward pass
            if self.use_fourier:
                z_mean, z_log_var, z = self.encoder(inputs, training=True)
                reconstruction = self.decoder([z, coord_input], training=True)
            else:
                z_mean, z_log_var, z = self.encoder(inputs, training=True)
                reconstruction = self.decoder(z, training=True)
            
            # Compute losses
            reconstruction_loss = tf.reduce_sum(
                tf.keras.losses.mean_absolute_error(target, reconstruction)
            )
            
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(tf.clip_by_value(z_log_var, -20, 20))
            )
            
            total_loss = reconstruction_loss + kl_loss
            
            if self.use_perceptual_loss:
                perceptual_loss = tf.reduce_sum(
                    tf.square(self.perceptual_model(target) - self.perceptual_model(reconstruction))
                )
                total_loss += perceptual_loss
                self.perceptual_loss_tracker.update_state(perceptual_loss)
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Apply gradient clipping to prevent NaN loss
        gradients = [tf.clip_by_norm(grad, self.gradient_clip_norm) if grad is not None else None for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        metrics = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
        if self.use_perceptual_loss:
            metrics["perceptual_loss"] = self.perceptual_loss_tracker.result()
        
        return metrics
    
    def test_step(self, data):
        """Validation step - similar to train_step but without gradient updates."""
        if self.use_fourier:
            if isinstance(data, tuple) and len(data) == 2:
                # Data format: ((img, coord), _) or (img, coord)
                if isinstance(data[0], tuple):
                    img_input, coord_input = data[0]
                else:
                    img_input, coord_input = data
            else:
                # Data format: [img, coord] (for some dataset formats)
                img_input, coord_input = data[0], data[1]
            inputs = [img_input, coord_input]
            target = img_input
        else:
            if isinstance(data, tuple) and len(data) == 2:
                inputs = data[0] if not isinstance(data[0], tuple) else data[0][0]
            else:
                inputs = data
            target = inputs
        
        # Forward pass (no training=True, no GradientTape)
        if self.use_fourier:
            z_mean, z_log_var, z = self.encoder(inputs, training=False)
            reconstruction = self.decoder([z, coord_input], training=False)
        else:
            z_mean, z_log_var, z = self.encoder(inputs, training=False)
            reconstruction = self.decoder(z, training=False)
        
        # Compute losses
        reconstruction_loss = tf.reduce_sum(
            tf.keras.losses.mean_absolute_error(target, reconstruction)
        )
        
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        total_loss = reconstruction_loss + kl_loss
        
        if self.use_perceptual_loss:
            perceptual_loss = tf.reduce_sum(
                tf.square(self.perceptual_model(target) - self.perceptual_model(reconstruction))
            )
            total_loss += perceptual_loss
        
        # Return validation metrics - Keras will automatically add "val_" prefix
        metrics = {
            "loss": total_loss,  # Will become "val_loss"
            "reconstruction_loss": reconstruction_loss,  # Will become "val_reconstruction_loss"
            "kl_loss": kl_loss,  # Will become "val_kl_loss"
        }
        
        if self.use_perceptual_loss:
            metrics["perceptual_loss"] = perceptual_loss  # Will become "val_perceptual_loss"
        
        return metrics
    
    @property
    def metrics(self):
        """Return list of metrics."""
        metrics_list = [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        if self.use_perceptual_loss and hasattr(self, 'perceptual_loss_tracker'):
            metrics_list.append(self.perceptual_loss_tracker)
        return metrics_list


class FLRNet(keras.Model):
    """FLRNet model for sensor-based flow field reconstruction."""
    
    def __init__(self,
                 n_sensors: int = 8,
                 input_shape: Tuple[int, int, int] = (128, 256, 1),
                 latent_dims: int = 4,
                 n_base_features: int = 64,
                 use_fourier: bool = False,
                 use_perceptual_loss: bool = True,
                 pretrained_vae: Optional[FLRVAE] = None,
                 freeze_autoencoder: bool = True,
                 gradient_clip_norm: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_sensors = n_sensors
        self.input_shape_custom = input_shape
        self.latent_dims = latent_dims
        self.n_base_features = n_base_features
        self.use_fourier = use_fourier
        self.use_perceptual_loss = use_perceptual_loss
        self.freeze_autoencoder = freeze_autoencoder
        self.gradient_clip_norm = gradient_clip_norm
        
        # Calculate latent shape (32x downsampling for 5-block architecture)
        h, w = input_shape[0] // 32, input_shape[1] // 32
        self.latent_shape = (h, w, latent_dims)
        
        # Build or load encoder/decoder
        if pretrained_vae is not None:
            self.encoder = pretrained_vae.encoder
            self.decoder = pretrained_vae.decoder
        else:
            self.encoder = FLREncoder(
                latent_dims=latent_dims,
                n_base_features=n_base_features,
                use_fourier=use_fourier,
                name='flr_encoder'
            )
            
            self.decoder = FLRDecoder(
                n_base_features=n_base_features,
                use_fourier=use_fourier,
                output_channels=input_shape[2],
                name='flr_decoder'
            )
        
        # Freeze encoder/decoder if specified
        if self.freeze_autoencoder:
            self.encoder.trainable = False
            self.decoder.trainable = False
        
        # Sensor mapping network
        self.sensor_mapping = SensorMapping(
            n_sensors=n_sensors,
            latent_dim=self.latent_shape,
            name='sensor_mapping'
        )
        
        # Perceptual loss model
        if self.use_perceptual_loss:
            self._build_perceptual_model()
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        if self.use_perceptual_loss:
            self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
    
    def _build_perceptual_model(self):
        """Build perceptual loss model."""
        inputs = keras.Input(shape=self.input_shape_custom)
        rgb = flr_layer.Binary2RGB()(inputs)
        vgg = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(self.input_shape_custom[0], self.input_shape_custom[1], 3),
            pooling='avg'
        )
        vgg.trainable = False
        features = vgg(rgb)
        self.perceptual_model = keras.Model(inputs, features)
    
    def _compute_perceptual_loss(self, target, reconstruction):
        """Compute perceptual loss if enabled."""
        if self.use_perceptual_loss and hasattr(self, 'perceptual_model'):
            return tf.reduce_sum(
                tf.square(self.perceptual_model(target) - self.perceptual_model(reconstruction))
            )
        else:
            return tf.constant(0.0, dtype=tf.float32)
    
    def call(self, inputs, training=None):
        """Forward pass."""
        if self.use_fourier:
            # Check if we have multiple inputs or just sensor data
            if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
                # Training mode: sensor_data, field_data, coord_data
                sensor_input, field_input, coord_input = inputs
            elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                # Prediction mode with coordinates: sensor_data, coord_data
                sensor_input, coord_input = inputs
            else:
                # Prediction mode with just sensor data - need to create dummy coordinates
                sensor_input = inputs
                batch_size = tf.shape(sensor_input)[0]
                # Create dummy coordinate grid for reconstruction
                coord_input = tf.zeros((batch_size, self.input_shape_custom[0], self.input_shape_custom[1], 2))
            
            z_mean_sensor, z_log_var_sensor, z_sensor = self.sensor_mapping(sensor_input, training=training)
            reconstruction = self.decoder([z_sensor, coord_input], training=training)
        else:
            # Standard mode - only sensor input
            sensor_input = inputs
            z_mean_sensor, z_log_var_sensor, z_sensor = self.sensor_mapping(sensor_input, training=training)
            reconstruction = self.decoder(z_sensor, training=training)
        
        return reconstruction
    
    def kl_divergence(self, mean1, logvar1, mean2, logvar2):
        """Compute KL divergence between two Gaussian distributions with numerical stability."""
        # Clip log variances to prevent overflow/underflow
        logvar1_clipped = tf.clip_by_value(logvar1, -20, 20)
        logvar2_clipped = tf.clip_by_value(logvar2, -20, 20)
        
        var1 = tf.exp(logvar1_clipped)
        var2 = tf.exp(logvar2_clipped)
        
        # Standard KL for multivariate Gaussians: 0.5 * [log(det(Σ2)/det(Σ1)) + tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - d]
        # where d is the dimension
        eps = 1e-6
        kl = 0.5 * (
            logvar2_clipped - logvar1_clipped +  # log(det(Σ2)/det(Σ1))
            (var1 + tf.square(mean1 - mean2)) / (var2 + eps) - 1.0  # tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - d
        )
        
        # Ensure KL is non-negative (though theoretically it should be already)
        kl = tf.maximum(kl, 0)
        
        return tf.reduce_sum(kl)
    
    def train_step(self, data):
        """Custom training step."""
        if self.use_fourier:
            # Data format: (sensor_input, field_input, coord_input)
            if len(data) == 3:
                sensor_input, field_input, coord_input = data
            else:
                # Alternative format: ((sensor, field, coord),) or nested tuples
                if isinstance(data[0], (list, tuple)) and len(data[0]) == 3:
                    sensor_input, field_input, coord_input = data[0]
                else:
                    raise ValueError(f"Expected 3 inputs for Fourier mode, got {len(data)}")
            
            field_inputs = [field_input, coord_input]
            target = field_input
        else:
            # Data format: (sensor_input, field_input)
            if len(data) == 2:
                sensor_input, field_input = data
            else:
                sensor_input, field_input = data[0], data[1]
            
            field_inputs = field_input
            target = field_input
        
        with tf.GradientTape() as tape:
            # Get encoder latent representation (frozen)
            if self.use_fourier:
                z_mean_ae, z_log_var_ae, _ = self.encoder(field_inputs, training=False)
            else:
                z_mean_ae, z_log_var_ae, _ = self.encoder(field_input, training=False)
            
            # Sensor mapping
            z_mean_sensor, z_log_var_sensor, z_sensor = self.sensor_mapping(sensor_input, training=True)
            
            # Reconstruction
            if self.use_fourier:
                reconstruction = self.decoder([z_sensor, coord_input], training=False)
            else:
                reconstruction = self.decoder(z_sensor, training=False)
            
            # Compute losses
            reconstruction_loss = tf.reduce_sum(
                tf.keras.losses.mean_absolute_error(target, reconstruction)
            )
            
            kl_loss = self.kl_divergence(z_mean_sensor, z_log_var_sensor, z_mean_ae, z_log_var_ae)
            
            perceptual_loss = self._compute_perceptual_loss(target, reconstruction)
            
            # Use proper loss weighting: reconstruction + 2*kl + perceptual
            total_loss = reconstruction_loss + kl_loss + perceptual_loss
        
        # Only update sensor mapping weights
        trainable_vars = self.sensor_mapping.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Apply gradient clipping to prevent NaN loss
        gradients = [tf.clip_by_norm(grad, self.gradient_clip_norm) if grad is not None else None for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if self.use_perceptual_loss:
            self.perceptual_loss_tracker.update_state(perceptual_loss)
        
        metrics = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
        if self.use_perceptual_loss:
            metrics["perceptual_loss"] = self.perceptual_loss_tracker.result()
        
        return metrics
    
    def test_step(self, data):
        """Validation step - similar to train_step but without gradient updates."""
        if self.use_fourier:
            # Data format: (sensor_input, field_input, coord_input)
            if len(data) == 3:
                sensor_input, field_input, coord_input = data
            else:
                # Alternative format: ((sensor, field, coord),) or nested tuples
                if isinstance(data[0], (list, tuple)) and len(data[0]) == 3:
                    sensor_input, field_input, coord_input = data[0]
                else:
                    raise ValueError(f"Expected 3 inputs for Fourier mode, got {len(data)}")
            
            field_inputs = [field_input, coord_input]
            target = field_input
        else:
            # Data format: (sensor_input, field_input)
            if len(data) == 2:
                sensor_input, field_input = data
            else:
                sensor_input, field_input = data[0], data[1]
            
            field_inputs = field_input
            target = field_input
        
        # Forward pass (no training=True, no GradientTape)
        # Get encoder latent representation (frozen)
        if self.use_fourier:
            z_mean_ae, z_log_var_ae, _ = self.encoder(field_inputs, training=False)
        else:
            z_mean_ae, z_log_var_ae, _ = self.encoder(field_input, training=False)
        
        # Sensor mapping
        z_mean_sensor, z_log_var_sensor, z_sensor = self.sensor_mapping(sensor_input, training=False)
        
        # Reconstruction
        if self.use_fourier:
            reconstruction = self.decoder([z_sensor, coord_input], training=False)
        else:
            reconstruction = self.decoder(z_sensor, training=False)
        
        # Compute losses
        reconstruction_loss = tf.reduce_sum(
            tf.keras.losses.mean_absolute_error(target, reconstruction)
        )
        
        kl_loss = self.kl_divergence(z_mean_sensor, z_log_var_sensor, z_mean_ae, z_log_var_ae)
        
        perceptual_loss = self._compute_perceptual_loss(target, reconstruction)
        
        # Use proper loss weighting: reconstruction + 2*kl + perceptual
        total_loss = reconstruction_loss + 2.0 * kl_loss + perceptual_loss
        
        # Return validation metrics - Keras will automatically add "val_" prefix
        metrics = {
            "loss": total_loss,  # Will become "val_loss"
            "reconstruction_loss": reconstruction_loss,  # Will become "val_reconstruction_loss"
            "kl_loss": kl_loss,  # Will become "val_kl_loss"
        }
        
        if self.use_perceptual_loss:
            metrics["perceptual_loss"] = perceptual_loss  # Will become "val_perceptual_loss"
        
        return metrics
    
    @property
    def metrics(self):
        """Return list of metrics."""
        base_metrics = [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        
        if self.use_perceptual_loss:
            base_metrics.append(self.perceptual_loss_tracker)
        
        return base_metrics


class FLRTrainer:
    """Unified trainer for VAE and FLRNet models with callbacks and validation."""
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (128, 256, 1),
                 use_fourier: bool = False,
                 checkpoint_dir: str = "./checkpoints",
                 logs_dir: str = "./logs",
                 model_name: str = None,
                 save_best_model: bool = True,
                 save_last_model: bool = True,
                 gradient_clip_norm: float = 1.0):
        self.input_shape = input_shape
        self.use_fourier = use_fourier
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name or "default_model"
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.gradient_clip_norm = gradient_clip_norm
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.vae_model = None
        self.flr_model = None
    
    def create_callbacks(self, 
                        model_type: str,
                        monitor: str = 'val_reconstruction_loss',
                        patience: int = 10,
                        reduce_lr_patience: int = 5) -> List[callbacks.Callback]:
        """Create training callbacks with proper naming convention."""
        callback_list = []
        
        # Custom ModelCheckpoint that only starts saving from epoch 31 (more conservative for Windows)
        class ConditionalModelCheckpoint(callbacks.ModelCheckpoint):
            def __init__(self, *args, start_saving_after_epoch=30, **kwargs):
                super().__init__(*args, **kwargs)
                self.start_saving_after_epoch = start_saving_after_epoch
                print(f"🛡️ Checkpoint saving will be disabled for epochs 1-{start_saving_after_epoch}, enabled from epoch {start_saving_after_epoch + 1}")
            
            def on_epoch_end(self, epoch, logs=None):
                if epoch >= self.start_saving_after_epoch:
                    # Normal checkpoint saving (epoch 31+ in 1-indexed terms)
                    try:
                        super().on_epoch_end(epoch, logs)
                    except Exception as e:
                        print(f"⚠️ Checkpoint saving failed at epoch {epoch + 1}: {e}")
                        print("   Continuing training without saving this checkpoint...")
                else:
                    # Skip saving but still monitor the metric for the first 30 epochs
                    current = logs.get(self.monitor)
                    if current is None:
                        return
                    
                    if self.monitor_op(current, self.best):
                        print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f} (saving disabled for epochs 1-{self.start_saving_after_epoch})")
                        self.best = current
                    else:
                        print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.5f} (saving disabled)")
        
        # Model checkpoint for best model - save weights only for subclassed models
        if self.save_best_model:
            best_model_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_{model_type}_best"
            callback_list.append(
                ConditionalModelCheckpoint(
                    filepath=str(best_model_path),
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=True,  # Use weights only for subclassed models
                    verbose=1,
                    start_saving_after_epoch=30  # Start saving from epoch 31 (0-indexed: epoch 30)
                )
            )
        
        # Additional checkpoint for last model (also conditional)
        if self.save_last_model:
            last_model_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_{model_type}_last"
            callback_list.append(
                ConditionalModelCheckpoint(
                    filepath=str(last_model_path),
                    monitor=monitor,
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=0,
                    save_freq='epoch',  # Save every epoch
                    start_saving_after_epoch=30  # Start saving from epoch 31 (0-indexed: epoch 30)
                )
            )
        
        # Early stopping
        callback_list.append(
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Learning rate reduction
        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # TensorBoard logging
        callback_list.append(
            callbacks.TensorBoard(
                log_dir=self.logs_dir / f"{self.model_name}_{model_type}",
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
        # CSV logging
        callback_list.append(
            callbacks.CSVLogger(
                filename=self.logs_dir / f"{self.model_name}_{model_type}_training.csv",
                append=True
            )
        )
        
        # NaN termination callbacks
        callback_list.append(callbacks.TerminateOnNaN())
        callback_list.append(NaNMonitorCallback())
        
        return callback_list
    
    def train_vae(self,
                  train_dataset: tf.data.Dataset,
                  val_dataset: tf.data.Dataset,
                  epochs: int = 100,
                  learning_rate: float = 1e-4,
                  latent_dims: int = 4,
                  n_base_features: int = 64,
                  use_perceptual_loss: bool = True,
                  patience: int = 10,
                  reduce_lr_patience: int = 5,
                  **kwargs) -> FLRVAE:
        """Train VAE model."""
        print("🚀 Training VAE Model...")
        
        # Separate model kwargs from training kwargs
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['name', 'trainable']}  # Only pass valid keras.Model kwargs
        
        # Create VAE model
        self.vae_model = FLRVAE(
            input_shape=self.input_shape,
            latent_dims=latent_dims,
            n_base_features=n_base_features,
            use_fourier=self.use_fourier,
            use_perceptual_loss=use_perceptual_loss,
            gradient_clip_norm=self.gradient_clip_norm,
            **model_kwargs
        )
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.vae_model.compile(
            optimizer=optimizer,
            # Explicitly define metrics so callbacks can track them
            metrics=[]  # We'll handle metrics in custom train/test steps
        )
        
        # Create callbacks
        callback_list = self.create_callbacks("vae", patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        # Train model
        history = self.vae_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Save final model using weights if save_last_model is enabled
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_vae_final_weights"
            self.vae_model.save_weights(str(final_weights_path))
        
        print(f"✅ VAE training completed. Models saved to {self.checkpoint_dir}")
        return self.vae_model
    
    def train_flr_net(self,
                      train_dataset: tf.data.Dataset,
                      val_dataset: tf.data.Dataset,
                      n_sensors: int = 32,
                      epochs: int = 1000,
                      learning_rate: float = 5e-5,
                      pretrained_vae: Optional[FLRVAE] = None,
                      latent_dims: int = 4,
                      n_base_features: int = 64,
                      use_perceptual_loss: bool = True,
                      freeze_autoencoder: bool = True,
                      patience: int = 15,
                      reduce_lr_patience: int = 5,
                      **kwargs) -> FLRNet:
        """Train FLRNet model."""
        print("🚀 Training FLRNet Model...")
        
        # Use provided VAE or load from checkpoint
        if pretrained_vae is None:
            if self.vae_model is None:
                # Try to load from best checkpoint
                vae_best_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_vae_best"
                if vae_best_path.exists():
                    print(f"📁 Loading pretrained VAE from {vae_best_path}")
                    # Create VAE model and load weights
                    self.vae_model = FLRVAE(
                        input_shape=self.input_shape,
                        latent_dims=latent_dims,
                        n_base_features=n_base_features,
                        use_fourier=self.use_fourier,
                        use_perceptual_loss=use_perceptual_loss
                    )
                    # Build the model by calling it once
                    dummy_input = tf.zeros((1,) + self.input_shape)
                    if self.use_fourier:
                        dummy_coord = tf.zeros((1, self.input_shape[0], self.input_shape[1], 2))
                        _ = self.vae_model([dummy_input, dummy_coord])
                    else:
                        _ = self.vae_model(dummy_input)
                    self.vae_model.load_weights(str(vae_best_path))
                else:
                    raise ValueError("No pretrained VAE found. Train VAE first or provide pretrained_vae.")
            pretrained_vae = self.vae_model
        
        # Separate model kwargs from training kwargs
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['name', 'trainable']}  # Only pass valid keras.Model kwargs
        
        # Create FLRNet model
        self.flr_model = FLRNet(
            n_sensors=n_sensors,
            input_shape=self.input_shape,
            latent_dims=latent_dims,
            n_base_features=n_base_features,
            use_fourier=self.use_fourier,
            use_perceptual_loss=use_perceptual_loss,
            pretrained_vae=pretrained_vae,
            freeze_autoencoder=freeze_autoencoder,
            gradient_clip_norm=self.gradient_clip_norm,
            **model_kwargs
        )
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.flr_model.compile(optimizer=optimizer)
        
        # Create callbacks
        callback_list = self.create_callbacks("flrnet", patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        # Train model
        history = self.flr_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Save final model using weights if save_last_model is enabled
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_flrnet_final_weights"
            self.flr_model.save_weights(str(final_weights_path))
        
        print(f"✅ FLRNet training completed. Models saved to {self.checkpoint_dir}")
        return self.flr_model
    
    def continue_vae_training(self,
                             train_dataset: tf.data.Dataset,
                             val_dataset: tf.data.Dataset,
                             epochs: int = 50,
                             learning_rate: float = 1e-6,  # Lower LR for fine-tuning
                             patience: int = 10,
                             reduce_lr_patience: int = 5,
                             use_perceptual_loss: Optional[bool] = None,  # Override perceptual loss setting
                             **kwargs) -> FLRVAE:
        """
        Continue training an existing VAE model from checkpoint.
        
        This method preserves the loaded model weights and continues training,
        unlike train_vae() which creates a new model and resets weights.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of additional epochs to train
            learning_rate: Learning rate for continued training (usually lower)
            patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            use_perceptual_loss: Override perceptual loss setting (None = keep original)
            **kwargs: Additional training arguments
            
        Returns:
            Continued VAE model
            
        Raises:
            ValueError: If no VAE model is loaded
        """
        if self.vae_model is None:
            raise ValueError("No VAE model loaded! Load a model from checkpoint first using load_vae_from_checkpoint()")
        
        print("🔄 Continuing VAE training from loaded checkpoint...")
        print("=" * 60)
        
          
        print(f"📋 Current Model Configuration:")
        print(f"   - Input shape: {self.input_shape}")
        print(f"   - Latent dims: {getattr(self.vae_model, 'latent_dims', 'Unknown')}")
        print(f"   - Base features: {getattr(self.vae_model, 'n_base_features', 'Unknown')}")
        print(f"   - Use Fourier: {getattr(self.vae_model, 'use_fourier', self.use_fourier)}")
        
        # Handle perceptual loss override WITHOUT resetting model
        original_perceptual_loss = getattr(self.vae_model, 'use_perceptual_loss', True)
        if use_perceptual_loss is not None and use_perceptual_loss != original_perceptual_loss:
            print(f"🔄 Perceptual loss: {original_perceptual_loss} → {use_perceptual_loss} (OVERRIDDEN)")
            self.vae_model.use_perceptual_loss = use_perceptual_loss
            
            # Rebuild metrics trackers if needed
            if use_perceptual_loss and not hasattr(self.vae_model, 'perceptual_loss_tracker'):
                self.vae_model.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
                if not hasattr(self.vae_model, 'perceptual_model'):
                    self.vae_model._build_perceptual_model()
                print("   - Perceptual loss tracker initialized")
            elif not use_perceptual_loss and hasattr(self.vae_model, 'perceptual_loss_tracker'):
                delattr(self.vae_model, 'perceptual_loss_tracker')
                print("   - Perceptual loss tracker removed")
        else:
            perceptual_status = "ENABLED" if original_perceptual_loss else "DISABLED"
            print(f"🔄 Perceptual loss: {perceptual_status} (KEPT from checkpoint)")
        
        print(f"\n📋 Training Configuration:")
        print(f"   - Additional epochs: {epochs}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Perceptual loss: {'✅ ENABLED' if self.vae_model.use_perceptual_loss else '❌ DISABLED'}")
        print(f"   - Model weights: ✅ PRESERVED from checkpoint")
        
        # CRITICAL: Reset ONLY metric trackers, NOT model weights
        print(f"\n🔄 Resetting metric trackers (preserving model weights)...")
        if hasattr(self.vae_model, 'total_loss_tracker'):
            self.vae_model.total_loss_tracker.reset_state()
        if hasattr(self.vae_model, 'reconstruction_loss_tracker'):
            self.vae_model.reconstruction_loss_tracker.reset_state()
        if hasattr(self.vae_model, 'kl_loss_tracker'):
            self.vae_model.kl_loss_tracker.reset_state()
        if hasattr(self.vae_model, 'perceptual_loss_tracker'):
            self.vae_model.perceptual_loss_tracker.reset_state()
        
        # Create new optimizer but DON'T recompile in a way that resets weights
        print(f"🔧 Setting up optimizer (learning rate: {learning_rate})...")
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        
        # IMPORTANT: Use compile carefully to avoid weight reset
        self.vae_model.compile(optimizer=optimizer, run_eagerly=False)
        
        # Create callbacks (will use existing model_name)
        callback_list = self.create_callbacks("vae", patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        print(f"\n🚀 Starting training continuation with preserved weights...")
        
        # Continue training
        history = self.vae_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
   
        # Save final model using weights if save_last_model is enabled
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_vae_continued_weights"
            self.vae_model.save_weights(str(final_weights_path))
            print(f"💾 Continued model weights saved to: {final_weights_path}")
        
        print(f"✅ VAE training continuation completed with preserved weights!")
        return self.vae_model

    def continue_flrnet_training(self,
                               train_dataset: tf.data.Dataset,
                               val_dataset: tf.data.Dataset,
                               epochs: int = 100,
                               learning_rate: float = 1e-6,  # Lower LR for fine-tuning
                               patience: int = 15,
                               reduce_lr_patience: int = 5,
                               use_perceptual_loss: Optional[bool] = None,  # Override perceptual loss setting
                               **kwargs) -> FLRNet:
        """
        Continue training an existing FLRNet model from checkpoint.
        
        This method preserves the loaded model weights and continues training,
        unlike train_flr_net() which creates a new model and resets weights.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of additional epochs to train
            learning_rate: Learning rate for continued training (usually lower)
            patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            use_perceptual_loss: Override perceptual loss setting (None = keep original)
            **kwargs: Additional training arguments
            
        Returns:
            Continued FLRNet model
            
        Raises:
            ValueError: If no FLRNet model is loaded
        """
        if self.flr_model is None:
            raise ValueError("No FLRNet model loaded! Load a model from checkpoint first using load_flrnet_from_checkpoint()")
        
        print("🔄 Continuing FLRNet training from loaded checkpoint...")
        print("=" * 60)
        print(f"📋 Current Model Configuration:")
        print(f"   - Input shape: {self.input_shape}")
        print(f"   - Number of sensors: {getattr(self.flr_model, 'n_sensors', 'Unknown')}")
        print(f"   - Latent dims: {getattr(self.flr_model, 'latent_dims', 'Unknown')}")
        print(f"   - Use Fourier: {getattr(self.flr_model, 'use_fourier', self.use_fourier)}")
        
        # Handle perceptual loss override for FLRNet (affects VAE component)
        if hasattr(self.flr_model, 'autoencoder') and self.flr_model.autoencoder is not None:
            original_perceptual_loss = getattr(self.flr_model.autoencoder, 'use_perceptual_loss', True)
            if use_perceptual_loss is not None:
                # Override the perceptual loss setting in the VAE component
                self.flr_model.autoencoder.use_perceptual_loss = use_perceptual_loss
                print(f"   - Perceptual loss: {use_perceptual_loss} ({'OVERRIDDEN' if use_perceptual_loss != original_perceptual_loss else 'KEPT from checkpoint'})")
                
                # Ensure perceptual loss tracker exists if needed
                if use_perceptual_loss and not hasattr(self.flr_model.autoencoder, 'perceptual_loss_tracker'):
                    self.flr_model.autoencoder.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
                    if not hasattr(self.flr_model.autoencoder, 'perceptual_model'):
                        self.flr_model.autoencoder._build_perceptual_model()
                    print("   - Perceptual loss tracker initialized for VAE component")
                elif not use_perceptual_loss and hasattr(self.flr_model.autoencoder, 'perceptual_loss_tracker'):
                    # Remove perceptual loss tracker if disabling
                    delattr(self.flr_model.autoencoder, 'perceptual_loss_tracker')
                    print("   - Perceptual loss tracker removed from VAE component")
            else:
                print(f"   - Perceptual loss: {original_perceptual_loss} (KEPT from checkpoint)")
        else:
            print(f"   - Perceptual loss: Not applicable (no VAE component found)")
        
        print(f"\n🎯 Training Configuration:")
        print(f"   - Additional epochs: {epochs}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Patience: {patience}")
        print(f"   - Model weights: ✅ PRESERVED from checkpoint")
        
        # Compile model with new optimizer (preserves weights, resets optimizer state)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.flr_model.compile(optimizer=optimizer)
        
        # Create callbacks
        callback_list = self.create_callbacks("flrnet", patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        # Continue training
        print(f"\n🚀 Starting training continuation...")
        history = self.flr_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Save final model using weights if save_last_model is enabled
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_flrnet_continued_weights"
            self.flr_model.save_weights(str(final_weights_path))
            print(f"💾 Continued model weights saved to: {final_weights_path}")
        
        print(f"✅ FLRNet training continuation completed!")
        return self.flr_model

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FLRTrainer':
        """
        Create FLRTrainer from configuration dictionary.
        
        Args:
            config: Configuration dictionary loaded from YAML
            
        Returns:
            Configured FLRTrainer instance
        """
        model_config = config['model']
        output_config = config['output']
        
        return cls(
            input_shape=tuple(model_config['input_shape']),
            use_fourier=model_config['use_fourier'],
            checkpoint_dir=config['checkpoint_dir'],
            logs_dir=config['logs_dir'],
            model_name=config['model_name'],
            save_best_model=output_config['save_best_model'],
            save_last_model=output_config['save_last_model']
        )
    
    def load_model(self, model_path: str) -> keras.Model:
        """Load a saved model."""
        return keras.models.load_model(model_path, compile=False)
    
    def export_model_summary(self, model: keras.Model, filename: str):
        """Export model architecture summary."""
        # Handle both relative and absolute paths
        if os.path.isabs(filename):
            # If absolute path is provided, use it directly
            filepath = Path(filename)
        else:
            # If relative path, combine with logs_dir
            filepath = self.logs_dir / filename
        
        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    def load_vae_from_checkpoint(self, 
                                 checkpoint_dir: Optional[Union[str, Path]] = None, 
                                 latent_dims: int = 4,
                                 n_base_features: int = 64,
                                 use_perceptual_loss: bool = True,
                                 verbose: bool = True) -> Optional[FLRVAE]:
        """
        Load VAE model from checkpoint with robust error handling.
        
        Args:
            checkpoint_dir: Path to checkpoint directory (defaults to self.checkpoint_dir)
            latent_dims: Latent dimensions for VAE model
            n_base_features: Number of base features for VAE model
            use_perceptual_loss: Whether to use perceptual loss
            verbose: Whether to print detailed loading information
            
        Returns:
            Loaded VAE model or None if loading fails
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        if verbose:
            print(f"🔍 Loading VAE model from checkpoint directory: {checkpoint_dir}")
        
        # Check if checkpoint directory exists
        if not checkpoint_dir.exists():
            if verbose:
                print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            return None
        
        # Priority order for checkpoint types (best -> last -> final_weights)
        checkpoint_types = ['vae_best', 'vae_last', 'vae_final_weights']
        
        for checkpoint_type in checkpoint_types:
            # Find checkpoint files matching the pattern
            pattern = f"*{checkpoint_type}*"
            checkpoint_files = list(checkpoint_dir.glob(pattern))
            
            if not checkpoint_files:
                if verbose:
                    print(f"⚠️  No {checkpoint_type} checkpoint found")
                continue
            
            # Look for .index files to identify valid checkpoints
            index_files = [f for f in checkpoint_files if f.suffix == '.index']
            
            if not index_files:
                if verbose:
                    print(f"⚠️  No valid .index files found for {checkpoint_type}")
                continue
            
            # Use the first valid checkpoint (remove .index extension for TensorFlow)
            checkpoint_path = str(index_files[0]).replace('.index', '')
            
            if verbose:
                print(f"✅ Found {checkpoint_type} checkpoint: {checkpoint_path}")
            
            try:
                # Create VAE model with the same approach as train_vae method
                vae_model = FLRVAE(
                    input_shape=self.input_shape,
                    latent_dims=latent_dims,
                    n_base_features=n_base_features,
                    use_fourier=self.use_fourier,
                    use_perceptual_loss=use_perceptual_loss,
                    gradient_clip_norm=self.gradient_clip_norm
                )
                
                # Build the model by calling it once with dummy input
                dummy_input = tf.zeros((1,) + self.input_shape)
                if self.use_fourier:
                    dummy_coord = tf.zeros((1, self.input_shape[0], self.input_shape[1], 2))
                    _ = vae_model([dummy_input, dummy_coord])
                    if verbose:
                        print("🌊 VAE model built for Fourier features")
                else:
                    _ = vae_model(dummy_input)
                    if verbose:
                        print("🔄 VAE model built for standard features")
                
                if verbose:
                    print(f"📋 VAE Model Architecture:")
                    print(f"   - Input shape: {self.input_shape}")
                    print(f"   - Latent dims: {latent_dims}")
                    print(f"   - Base features: {n_base_features}")
                    print(f"   - Use Fourier: {self.use_fourier}")
                    print(f"   - Perceptual loss: {use_perceptual_loss}")
                
                # Load weights from checkpoint
                vae_model.load_weights(checkpoint_path)
                
                if verbose:
                    print(f"✅ Successfully loaded VAE model from {checkpoint_type} checkpoint!")
                
                # Store the loaded model
                self.vae_model = vae_model
                return vae_model
                
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to load {checkpoint_type} checkpoint: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if verbose:
            print(f"❌ Failed to load VAE model from any available checkpoints in {checkpoint_dir}")
            print("Available files:")
            for file in sorted(checkpoint_dir.iterdir()):
                print(f"   - {file.name}")
        
        return None
    
    def load_flrnet_from_checkpoint(self, 
                                   n_sensors: int,
                                   checkpoint_dir: Optional[Union[str, Path]] = None, 
                                   pretrained_vae: Optional[FLRVAE] = None,
                                   latent_dims: int = 4,
                                   n_base_features: int = 64,
                                   use_perceptual_loss: bool = True,
                                   freeze_autoencoder: bool = True,
                                   verbose: bool = True) -> Optional[FLRNet]:
        """
        Load FLRNet model from checkpoint with robust error handling.
        
        Args:
            n_sensors: Number of sensors for the FLRNet
            checkpoint_dir: Path to checkpoint directory (defaults to self.checkpoint_dir)
            pretrained_vae: Pretrained VAE model (if None, will try to load from checkpoint)
            latent_dims: Latent dimensions for VAE model
            n_base_features: Number of base features for models
            use_perceptual_loss: Whether to use perceptual loss
            freeze_autoencoder: Whether to freeze autoencoder weights
            verbose: Whether to print detailed loading information
            
        Returns:
            Loaded FLRNet model or None if loading fails
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        if verbose:
            print(f"🔍 Loading FLRNet model from checkpoint directory: {checkpoint_dir}")
        
        # Check if checkpoint directory exists
        if not checkpoint_dir.exists():
            if verbose:
                print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            return None
        
        # Ensure we have a VAE model
        if pretrained_vae is None:
            if self.vae_model is None:
                if verbose:
                    print("🔄 Loading VAE model first...")
                pretrained_vae = self.load_vae_from_checkpoint(
                    checkpoint_dir, 
                    latent_dims=latent_dims,
                    n_base_features=n_base_features,
                    use_perceptual_loss=use_perceptual_loss,
                    verbose=verbose
                )
                if pretrained_vae is None:
                    if verbose:
                        print("❌ Cannot load FLRNet without VAE model")
                    return None
            else:
                pretrained_vae = self.vae_model
        
        # Priority order for checkpoint types (best -> last -> final_weights)
        checkpoint_types = ['flrnet_best', 'flrnet_last', 'flrnet_final_weights']
        
        for checkpoint_type in checkpoint_types:
            # Find checkpoint files matching the pattern
            pattern = f"*{checkpoint_type}*"
            checkpoint_files = list(checkpoint_dir.glob(pattern))
            
            if not checkpoint_files:
                if verbose:
                    print(f"⚠️  No {checkpoint_type} checkpoint found")
                continue
            
            # Look for .index files to identify valid checkpoints
            index_files = [f for f in checkpoint_files if f.suffix == '.index']
            
            if not index_files:
                if verbose:
                    print(f"⚠️  No valid .index files found for {checkpoint_type}")
                continue
            
            # Use the first valid checkpoint (remove .index extension for TensorFlow)
            checkpoint_path = str(index_files[0]).replace('.index', '')
            
            if verbose:
                print(f"✅ Found {checkpoint_type} checkpoint: {checkpoint_path}")
            
            try:
                # Create FLRNet model with the same approach as train_flr_net method
                flrnet_model = FLRNet(
                    n_sensors=n_sensors,
                    input_shape=self.input_shape,
                    pretrained_vae=pretrained_vae,
                    use_fourier=self.use_fourier,
                    freeze_autoencoder=freeze_autoencoder,
                    latent_dims=latent_dims,
                    n_base_features=n_base_features,
                    gradient_clip_norm=self.gradient_clip_norm
                )
                
                # Build the model by calling it once with dummy input
                dummy_sensor_input = tf.zeros((1, n_sensors))
                _ = flrnet_model(dummy_sensor_input)
                
                if verbose:
                    print(f"📋 FLRNet Model Architecture:")
                    print(f"   - Input shape: {self.input_shape}")
                    print(f"   - Number of sensors: {n_sensors}")
                    print(f"   - Latent dims: {latent_dims}")
                    print(f"   - Base features: {n_base_features}")
                    print(f"   - Use Fourier: {self.use_fourier}")
                    print(f"   - Freeze autoencoder: {freeze_autoencoder}")
                
                # Load weights from checkpoint
                flrnet_model.load_weights(checkpoint_path)
                
                if verbose:
                    print(f"✅ Successfully loaded FLRNet model from {checkpoint_type} checkpoint!")
                
                # Store the loaded model
                self.flr_model = flrnet_model
                return flrnet_model
                
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to load {checkpoint_type} checkpoint: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if verbose:
            print(f"❌ Failed to load FLRNet model from any available checkpoints in {checkpoint_dir}")
            print("Available files:")
            for file in sorted(checkpoint_dir.iterdir()):
                print(f"   - {file.name}")
        
        return None
    
    def load_models_from_checkpoint(self, 
                                   n_sensors: int,
                                   checkpoint_dir: Optional[Union[str, Path]] = None,
                                   latent_dims: int = 4,
                                   n_base_features: int = 64,
                                   use_perceptual_loss: bool = True,
                                   freeze_autoencoder: bool = True,
                                   verbose: bool = True) -> Tuple[Optional[FLRVAE], Optional[FLRNet]]:
        """
        Load both VAE and FLRNet models from checkpoint directory.
        
        Args:
            n_sensors: Number of sensors for the FLRNet
            checkpoint_dir: Path to checkpoint directory (defaults to self.checkpoint_dir)
            latent_dims: Latent dimensions for models
            n_base_features: Number of base features for models
            use_perceptual_loss: Whether to use perceptual loss
            freeze_autoencoder: Whether to freeze autoencoder weights in FLRNet
            verbose: Whether to print detailed loading information
            
        Returns:
            Tuple of (VAE model, FLRNet model) - either can be None if loading fails
        """
        if verbose:
            print("🚀 Loading both VAE and FLRNet models from checkpoint...")
            print("=" * 60)
        
        # Load VAE first
        vae_model = self.load_vae_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            latent_dims=latent_dims,
            n_base_features=n_base_features,
            use_perceptual_loss=use_perceptual_loss,
            verbose=verbose
        )
        
        # Load FLRNet (will use the loaded VAE)
        flrnet_model = self.load_flrnet_from_checkpoint(
            n_sensors=n_sensors,
            checkpoint_dir=checkpoint_dir, 
            pretrained_vae=vae_model,
            latent_dims=latent_dims,
            n_base_features=n_base_features,
            use_perceptual_loss=use_perceptual_loss,
            freeze_autoencoder=freeze_autoencoder,
            verbose=verbose
        )
        
        if verbose:
            print("\n📊 Loading Summary:")
            print(f"   VAE model: {'✅ Loaded' if vae_model is not None else '❌ Failed'}")
            print(f"   FLRNet model: {'✅ Loaded' if flrnet_model is not None else '❌ Failed'}")
        
        return vae_model, flrnet_model

def create_sample_datasets(batch_size: int = 8, 
                          use_fourier: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create sample datasets for testing."""
    # This is a placeholder - replace with your actual data loading logic
    if use_fourier:
        # Sample data with coordinates
        img_data = np.random.random((100, 128, 256, 1)).astype(np.float32)
        coord_data = np.random.random((100, 128, 256, 2)).astype(np.float32)
        sensor_data = np.random.random((100, 32)).astype(np.float32)
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((sensor_data[:80], img_data[:80], coord_data[:80]))
        val_ds = tf.data.Dataset.from_tensor_slices((sensor_data[80:], img_data[80:], coord_data[80:]))
    else:
        # Standard data
        img_data = np.random.random((100, 128, 256, 1)).astype(np.float32)
        sensor_data = np.random.random((100, 32)).astype(np.float32)
        
        train_ds = tf.data.Dataset.from_tensor_slices((sensor_data[:80], img_data[:80]))
        val_ds = tf.data.Dataset.from_tensor_slices((sensor_data[80:], img_data[80:]))
    
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


# Example usage
if __name__ == "__main__":
    # Configuration
    USE_FOURIER = True  # Set to False for standard version
    N_SENSORS = 32
    BATCH_SIZE = 8
    
    # Create trainer
    trainer = FLRTrainer(
        input_shape=(128, 256, 1),
        use_fourier=USE_FOURIER,
        checkpoint_dir="./checkpoints",
        logs_dir="./logs"
    )
    
    # Create sample datasets (replace with your actual data loading)
    train_ds, val_ds = create_sample_datasets(BATCH_SIZE, USE_FOURIER)
    
    # Step 1: Train VAE
    vae_model = trainer.train_vae(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=50,
        learning_rate=1e-4
    )
    
    # Step 2: Train FLRNet
    flr_model = trainer.train_flr_net(
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_sensors=N_SENSORS,
        epochs=100,
        learning_rate=5e-5,
        pretrained_vae=vae_model
    )
    
    print("🎉 Training pipeline completed!")
