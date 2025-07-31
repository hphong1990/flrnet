"""
MLP Models for Flow Field Reconstruction

This module contains MLP models for sensor-based flow field reconstruction with:
- Modern TensorFlow 2.x architecture
- Automated training pipeline with validation
- Callback support for model checkpointing
- Unified training workflow similar to FLRNet
- Direct sensor-to-field reconstruction without VAE dependency
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any
import os
import json


class MLPReconstructionModel(keras.Model):
    """MLP model for direct sensor-to-field reconstruction."""
    
    def __init__(self,
                 n_sensors: int = 32,
                 input_shape: Tuple[int, int, int] = (128, 256, 1),
                 hidden_layers: List[int] = [512, 1024, 2048, 4096],
                 activation: str = 'leaky_relu',
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 gradient_clip_norm: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.n_sensors = n_sensors
        self.input_shape_custom = input_shape
        self.hidden_layers = hidden_layers
        
        # FIXED: Proper activation handling
        if activation == 'leaky_relu':
            self.activation = layers.LeakyReLU()
        elif activation == 'relu':
            self.activation = 'relu'
        elif activation == 'tanh':
            self.activation = 'tanh'
        elif activation == 'sigmoid':
            self.activation = 'sigmoid'
        elif activation == 'swish':
            self.activation = 'swish'
        else:
            # Default to relu for unknown activations
            print(f"⚠️  Unknown activation '{activation}', using 'relu'")
            self.activation = 'relu'
        
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.gradient_clip_norm = gradient_clip_norm
        
        # Calculate output size
        self.output_size = np.prod(input_shape)
        
        # Build MLP layers
        self._build_mlp()
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.mse_tracker = keras.metrics.Mean(name="mse")
        self.mae_tracker = keras.metrics.Mean(name="mae")
    
    def _build_mlp(self):
        """Build MLP architecture."""
        self.mlp_layers = []
        
        # Input layer
        self.mlp_layers.append(layers.Dense(
            self.hidden_layers[0], 
            activation=self.activation,
            name='dense_input'
        ))
        
        if self.use_batch_norm:
            self.mlp_layers.append(layers.BatchNormalization(name='bn_input'))
        
        if self.dropout_rate > 0:
            self.mlp_layers.append(layers.Dropout(self.dropout_rate, name='dropout_input'))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers[1:], 1):
            self.mlp_layers.append(layers.Dense(
                units, 
                activation=self.activation,
                name=f'dense_hidden_{i}'
            ))
            
            if self.use_batch_norm:
                self.mlp_layers.append(layers.BatchNormalization(name=f'bn_hidden_{i}'))
            
            if self.dropout_rate > 0:
                self.mlp_layers.append(layers.Dropout(self.dropout_rate, name=f'dropout_hidden_{i}'))
        
        # Output layer - FIXED: Use linear activation for final layer
        self.mlp_layers.append(layers.Dense(
            self.output_size,
            activation='linear',  # Changed from self.activation to 'linear'
            name='dense_output'
        ))
        
        # Reshape to field dimensions
        self.reshape_layer = layers.Reshape(
            self.input_shape_custom, 
            name='reshape_output'
        )
    
    def call(self, inputs, training=None):
        """Forward pass."""
        x = inputs
        
        # Apply MLP layers
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        
        # Reshape to field dimensions
        output = self.reshape_layer(x)
        
        return output
    
    def train_step(self, data):
        """Custom training step."""
        sensor_data, field_data = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction = self(sensor_data, training=True)
            
            # Compute losses
            mse_loss = tf.reduce_sum(tf.square(field_data - reconstruction))  # Changed to reduce_mean
            mae_loss = tf.reduce_sum(tf.abs(field_data - reconstruction))     # Changed to reduce_mean
            
            # Total loss (using MSE as primary loss)
            total_loss = mae_loss
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Apply gradient clipping to prevent NaN loss
        gradients = [tf.clip_by_norm(grad, self.gradient_clip_norm) if grad is not None else None for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.mae_tracker.update_state(mae_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "mse": self.mse_tracker.result(),
            "mae": self.mae_tracker.result(),
        }
    
    def test_step(self, data):
        """Validation step."""
        sensor_data, field_data = data
        
        # Forward pass (no training=True, no GradientTape)
        reconstruction = self(sensor_data, training=False)
        
        # Compute losses
        mse_loss = tf.reduce_sum(tf.square(field_data - reconstruction))  # Changed to reduce_mean
        mae_loss = tf.reduce_sum(tf.abs(field_data - reconstruction))     # Changed to reduce_mean
        total_loss = mse_loss  # Changed to use MSE instead of MAE
        
        # Return validation metrics - Keras will automatically add "val_" prefix
        return {
            "loss": total_loss,
            "mse": mse_loss,
            "mae": mae_loss,
        }
    
    @property
    def metrics(self):
        """Return list of metrics."""
        return [
            self.total_loss_tracker,
            self.mse_tracker,
            self.mae_tracker,
        ]


class NaNMonitorCallback(callbacks.Callback):
    """Custom callback to monitor and log NaN/Inf values during training."""
    
    def on_batch_end(self, batch, logs=None):
        """Check for NaN/Inf values after each batch."""
        if logs:
            for key, value in logs.items():
                if tf.math.is_nan(value) or tf.math.is_inf(value):
                    print(f"⚠️ Warning: {key} is {value} at batch {batch}")


class MLPTrainer:
    """Unified trainer for MLP models with callbacks and validation."""
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (128, 256, 1),
                 checkpoint_dir: str = "./checkpoints",
                 logs_dir: str = "./logs",
                 model_name: str = None,
                 save_best_model: bool = True,
                 save_last_model: bool = True,
                 gradient_clip_norm: float = 1.0):
        self.input_shape = input_shape
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name or "default_mlp_model"
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.gradient_clip_norm = gradient_clip_norm
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.mlp_model = None
    
    def create_callbacks(self, 
                        model_type: str = "mlp",
                        monitor: str = 'val_loss',
                        patience: int = 15,
                        reduce_lr_patience: int = 5) -> List[callbacks.Callback]:
        """Create training callbacks with proper naming convention."""
        callback_list = []
        
        # Custom ModelCheckpoint that starts saving from epoch 31
        class ConditionalModelCheckpoint(callbacks.ModelCheckpoint):
            def __init__(self, *args, start_saving_after_epoch=10, **kwargs):
                super().__init__(*args, **kwargs)
                self.start_saving_after_epoch = start_saving_after_epoch
                print(f"🛡️ Checkpoint saving will be disabled for epochs 1-{start_saving_after_epoch}, enabled from epoch {start_saving_after_epoch + 1}")
            
            def on_epoch_end(self, epoch, logs=None):
                if epoch >= self.start_saving_after_epoch:
                    try:
                        super().on_epoch_end(epoch, logs)
                    except Exception as e:
                        print(f"⚠️ Checkpoint saving failed at epoch {epoch + 1}: {e}")
                        print("   Continuing training without saving this checkpoint...")
                else:
                    current = logs.get(self.monitor)
                    if current is None:
                        return
                    
                    if self.monitor_op(current, self.best):
                        print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f} (saving disabled for epochs 1-{self.start_saving_after_epoch})")
                        self.best = current
                    else:
                        print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.5f} (saving disabled)")
        
        # Model checkpoint for best model
        if self.save_best_model:
            best_model_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_{model_type}_best"
            callback_list.append(
                ConditionalModelCheckpoint(
                    filepath=str(best_model_path),
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                    start_saving_after_epoch=10
                )
            )
        
        # Additional checkpoint for last model
        if self.save_last_model:
            last_model_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_{model_type}_last"
            callback_list.append(
                ConditionalModelCheckpoint(
                    filepath=str(last_model_path),
                    monitor=monitor,
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=0,
                    save_freq='epoch',
                    start_saving_after_epoch=10
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
    
    def train_mlp(self,
                  train_dataset: tf.data.Dataset,
                  val_dataset: tf.data.Dataset,
                  n_sensors: int = 32,
                  epochs: int = 100,
                  learning_rate: float = 5e-5,
                  hidden_layers: List[int] = [512, 1024, 2048, 4096],
                  activation: str = 'relu',
                  dropout_rate: float = 0.2,
                  use_batch_norm: bool = True,
                  patience: int = 15,
                  reduce_lr_patience: int = 5,
                  **kwargs) -> MLPReconstructionModel:
        """Train MLP model."""
        print("🚀 Training MLP Model...")
        
        # Separate model kwargs from training kwargs
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['name', 'trainable']}
        
        # Create MLP model
        self.mlp_model = MLPReconstructionModel(
            n_sensors=n_sensors,
            input_shape=self.input_shape,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            gradient_clip_norm=self.gradient_clip_norm,
            **model_kwargs
        )
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.mlp_model.compile(optimizer=optimizer)
        
        # Create callbacks
        callback_list = self.create_callbacks("mlp", patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        # Train model
        history = self.mlp_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Save final model using weights if save_last_model is enabled
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_mlp_final_weights"
            self.mlp_model.save_weights(str(final_weights_path))
        
        print(f"✅ MLP training completed. Models saved to {self.checkpoint_dir}")
        return self.mlp_model
    
    def load_mlp_from_checkpoint(self, 
                                n_sensors: int,
                                checkpoint_dir: Optional[Union[str, Path]] = None,
                                hidden_layers: List[int] = [512, 1024, 2048, 4096],
                                activation: str = 'relu',
                                dropout_rate: float = 0.2,
                                use_batch_norm: bool = True,
                                verbose: bool = True) -> Optional[MLPReconstructionModel]:
        """Load MLP model from checkpoint with robust error handling."""
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        if verbose:
            print(f"🔍 Loading MLP model from checkpoint directory: {checkpoint_dir}")
        
        # Check if checkpoint directory exists
        if not checkpoint_dir.exists():
            if verbose:
                print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            return None
        
        # Priority order for checkpoint types
        checkpoint_types = ['mlp_best', 'mlp_last', 'mlp_final_weights']
        
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
            
            # Use the first valid checkpoint
            checkpoint_path = str(index_files[0]).replace('.index', '')
            
            if verbose:
                print(f"✅ Found {checkpoint_type} checkpoint: {checkpoint_path}")
            
            try:
                # Create MLP model
                mlp_model = MLPReconstructionModel(
                    n_sensors=n_sensors,
                    input_shape=self.input_shape,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    gradient_clip_norm=self.gradient_clip_norm
                )
                
                # Build the model by calling it once with dummy input
                dummy_sensor_input = tf.zeros((1, n_sensors))
                _ = mlp_model(dummy_sensor_input)
                
                if verbose:
                    print(f"📋 MLP Model Architecture:")
                    print(f"   - Input shape: {self.input_shape}")
                    print(f"   - Number of sensors: {n_sensors}")
                    print(f"   - Hidden layers: {hidden_layers}")
                    print(f"   - Activation: {activation}")
                    print(f"   - Dropout rate: {dropout_rate}")
                    print(f"   - Use batch norm: {use_batch_norm}")
                
                # Load weights from checkpoint
                mlp_model.load_weights(checkpoint_path)
                
                if verbose:
                    print(f"✅ Successfully loaded MLP model from {checkpoint_type} checkpoint!")
                
                # Store the loaded model
                self.mlp_model = mlp_model
                return mlp_model
                
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to load {checkpoint_type} checkpoint: {str(e)}")
                continue
        
        if verbose:
            print(f"❌ Failed to load MLP model from any available checkpoints in {checkpoint_dir}")
        
        return None
    
    def continue_mlp_training(self,
                             train_dataset: tf.data.Dataset,
                             val_dataset: tf.data.Dataset,
                             epochs: int = 50,
                             learning_rate: float = 1e-6,
                             patience: int = 15,
                             reduce_lr_patience: int = 5,
                             **kwargs) -> MLPReconstructionModel:
        """Continue training an existing MLP model from checkpoint."""
        if self.mlp_model is None:
            raise ValueError("No MLP model loaded! Load a model from checkpoint first using load_mlp_from_checkpoint()")
        
        print("🔄 Continuing MLP training from loaded checkpoint...")
        print("=" * 60)
        print(f"📋 Current Model Configuration:")
        print(f"   - Input shape: {self.input_shape}")
        print(f"   - Number of sensors: {getattr(self.mlp_model, 'n_sensors', 'Unknown')}")
        print(f"   - Hidden layers: {getattr(self.mlp_model, 'hidden_layers', 'Unknown')}")
        print(f"   - Activation: {getattr(self.mlp_model, 'activation', 'Unknown')}")
        
        print(f"\n🎯 Training Configuration:")
        print(f"   - Additional epochs: {epochs}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Patience: {patience}")
        print(f"   - Model weights: ✅ PRESERVED from checkpoint")
        
        # Compile model with new optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.mlp_model.compile(optimizer=optimizer)
        
        # Create callbacks
        callback_list = self.create_callbacks("mlp", patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        # Continue training
        print(f"\n🚀 Starting training continuation...")
        history = self.mlp_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Save final model using weights if save_last_model is enabled
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_mlp_continued_weights"
            self.mlp_model.save_weights(str(final_weights_path))
            print(f"💾 Continued model weights saved to: {final_weights_path}")
        
        print(f"✅ MLP training continuation completed!")
        return self.mlp_model
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MLPTrainer':
        """Create MLPTrainer from configuration dictionary."""
        return cls(
            input_shape=tuple(config['input_shape']),
            checkpoint_dir=config['checkpoint_dir'],
            logs_dir=config['logs_dir'],
            model_name=config['model_name'],
            save_best_model=config.get('save_best_model', True),
            save_last_model=config.get('save_last_model', True),
            gradient_clip_norm=config.get('gradient_clip_norm', 1.0)
        )