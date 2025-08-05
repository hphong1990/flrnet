# POD-based Flow Field Reconstruction Models
# Implementation following the same pattern as MLP and FLRNet

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import modred as mr
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json

class PODReconstructionModel(keras.Model):
    """POD-based reconstruction model: Sensor → POD coefficients → Field reconstruction"""
    
    def __init__(self,
                 n_sensors: int = 32,
                 input_shape: Tuple[int, int, int] = (128, 256, 1),
                 n_pod_modes: int = 128,
                 hidden_layers: List[int] = [256, 512, 512, 256],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 gradient_clip_norm: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.n_sensors = n_sensors
        self.input_shape_custom = input_shape
        self.n_pod_modes = n_pod_modes
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.gradient_clip_norm = gradient_clip_norm
        
        # POD modes will be loaded externally
        self.pod_modes = None
        
        # Build MLP for sensor → POD coefficients mapping
        self._build_mlp()
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.coefficient_loss_tracker = keras.metrics.Mean(name="coefficient_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
    
    def _build_mlp(self):
        """Build MLP to map sensor readings to POD coefficients."""
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
        
        # Output layer: POD coefficients
        self.mlp_layers.append(layers.Dense(
            self.n_pod_modes,
            activation='linear',  # Linear for coefficient prediction
            name='dense_pod_coefficients'
        ))
    
    def set_pod_modes(self, pod_modes: np.ndarray):
        """Set POD modes for reconstruction."""
        self.pod_modes = tf.constant(pod_modes, dtype=tf.float32)
        print(f"✅ POD modes set: {pod_modes.shape}")
    
    def call(self, inputs, training=None):
        """Forward pass: Sensor → POD coefficients → Field reconstruction."""
        # Step 1: Sensor → POD coefficients
        x = inputs
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        
        pod_coefficients = x  # Shape: (batch_size, n_pod_modes)
        
        # Step 2: POD coefficients → Field reconstruction
        if self.pod_modes is not None:
            # Reconstruct field: coefficients × modes^T
            flattened_field = tf.matmul(pod_coefficients, self.pod_modes, transpose_b=True)
            # Reshape to original field dimensions
            reconstructed_field = tf.reshape(flattened_field, [-1] + list(self.input_shape_custom))
            return reconstructed_field
        else:
            # If no POD modes, return coefficients
            return pod_coefficients
    
    def predict_coefficients(self, sensor_inputs, training=None):
        """Predict POD coefficients only (without field reconstruction)."""
        x = sensor_inputs
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        return x
    
    def reconstruct_from_coefficients(self, pod_coefficients):
        """Reconstruct field from POD coefficients."""
        if self.pod_modes is None:
            raise ValueError("POD modes not set! Call set_pod_modes() first.")
        
        flattened_field = tf.matmul(pod_coefficients, self.pod_modes, transpose_b=True)
        reconstructed_field = tf.reshape(flattened_field, [-1] + list(self.input_shape_custom))
        return reconstructed_field
    
    def train_step(self, data):
        """Custom training step."""
        sensor_data, pod_coefficients_target = data
        
        with tf.GradientTape() as tape:
            # Forward pass: predict POD coefficients
            predicted_coefficients = self.predict_coefficients(sensor_data, training=True)
            
            # Coefficient loss (MSE between predicted and target coefficients)
            coefficient_loss = tf.reduce_mean(tf.square(pod_coefficients_target - predicted_coefficients))
            
            total_loss = coefficient_loss
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(grad, self.gradient_clip_norm) if grad is not None else None for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.coefficient_loss_tracker.update_state(coefficient_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "coefficient_loss": self.coefficient_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """Validation step."""
        sensor_data, pod_coefficients_target = data
        
        # Forward pass
        predicted_coefficients = self.predict_coefficients(sensor_data, training=False)
        
        # Compute losses
        coefficient_loss = tf.reduce_mean(tf.square(pod_coefficients_target - predicted_coefficients))
        total_loss = coefficient_loss
        
        return {
            "loss": total_loss,
            "coefficient_loss": coefficient_loss,
        }
    
    @property
    def metrics(self):
        """Return list of metrics."""
        return [
            self.total_loss_tracker,
            self.coefficient_loss_tracker,
        ]

class PODTrainer:
    """Trainer for POD-based reconstruction models."""
    
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
        self.model_name = model_name or "default_pod_model"
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.gradient_clip_norm = gradient_clip_norm
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.pod_model = None
        self.pod_modes = None
    
    def extract_pod_modes(self, 
                         field_data: np.ndarray,
                         n_modes: int = 128,
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        Extract POD modes from field data.
        
        Args:
            field_data: Field data array (n_samples, height, width, channels)
            n_modes: Number of POD modes to extract
            save_path: Path to save POD modes
            
        Returns:
            POD modes array (n_spatial_points, n_modes)
        """
        print(f"🔧 Extracting {n_modes} POD modes from field data...")
        print(f"📊 Field data shape: {field_data.shape}")
        
        # Step 1: Flatten field data for POD analysis
        original_shape = field_data.shape
        n_samples = original_shape[0]
        spatial_dims = original_shape[1:]  # (height, width, channels)
        n_spatial_points = np.prod(spatial_dims)
        
        # Reshape to (n_spatial_points, n_samples) for POD
        field_data_flattened = field_data.reshape(n_samples, n_spatial_points).T
        print(f"📊 Flattened shape for POD: {field_data_flattened.shape}")
        
        # Step 2: Compute POD using ModRed
        POD_res = mr.compute_POD_arrays_snaps_method(
            field_data_flattened, 
            list(range(n_modes))
        )
        pod_modes = POD_res.modes  # Shape: (n_spatial_points, n_modes)
        
        print(f"✅ POD modes extracted: {pod_modes.shape}")
        print(f"📈 POD modes statistics:")
        print(f"   - Mean: {np.mean(pod_modes):.6f}")
        print(f"   - Std: {np.std(pod_modes):.6f}")
        print(f"   - Range: [{np.min(pod_modes):.6f}, {np.max(pod_modes):.6f}]")
        
        # Step 3: Save POD modes if requested
        if save_path:
            pod_modes_path = Path(save_path)
            pod_modes_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pod_modes_path, pod_modes)
            print(f"💾 POD modes saved to: {pod_modes_path}")
        
        self.pod_modes = pod_modes
        return pod_modes
    
    def compute_pod_coefficients(self, field_data: np.ndarray, pod_modes: np.ndarray) -> np.ndarray:
        """
        Compute POD coefficients from field data and POD modes.
        
        Args:
            field_data: Field data array (n_samples, height, width, channels)
            pod_modes: POD modes array (n_spatial_points, n_modes)
            
        Returns:
            POD coefficients array (n_samples, n_modes)
        """
        print(f"🔧 Computing POD coefficients...")
        
        # Flatten field data
        n_samples = field_data.shape[0]
        n_spatial_points = np.prod(field_data.shape[1:])
        field_data_flattened = field_data.reshape(n_samples, n_spatial_points)
        
        # Compute coefficients: coefficients = field_data @ modes
        pod_coefficients = np.matmul(field_data_flattened, pod_modes)
        
        print(f"✅ POD coefficients computed: {pod_coefficients.shape}")
        print(f"📈 POD coefficients statistics:")
        print(f"   - Mean: {np.mean(pod_coefficients):.6f}")
        print(f"   - Std: {np.std(pod_coefficients):.6f}")
        print(f"   - Range: [{np.min(pod_coefficients):.6f}, {np.max(pod_coefficients):.6f}]")
        
        return pod_coefficients
    
    def create_callbacks(self, 
                        monitor: str = 'val_loss',
                        patience: int = 15,
                        reduce_lr_patience: int = 5) -> List[keras.callbacks.Callback]:
        """Create training callbacks (same pattern as MLP)."""
        callback_list = []
        
        # Model checkpoint for best model
        if self.save_best_model:
            best_model_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_pod_best"
            callback_list.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=str(best_model_path),
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            )
        
        # Early stopping
        callback_list.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Learning rate reduction
        callback_list.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # TensorBoard logging
        callback_list.append(
            keras.callbacks.TensorBoard(
                log_dir=self.logs_dir / f"{self.model_name}_pod",
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
        return callback_list
    
    def train_pod_model(self,
                       train_dataset: tf.data.Dataset,
                       val_dataset: tf.data.Dataset,
                       n_sensors: int = 32,
                       n_pod_modes: int = 128,
                       pod_modes: np.ndarray = None,
                       epochs: int = 200,
                       learning_rate: float = 5e-5,
                       hidden_layers: List[int] = [256, 512, 512, 256],
                       activation: str = 'relu',
                       dropout_rate: float = 0.2,
                       use_batch_norm: bool = True,
                       patience: int = 15,
                       reduce_lr_patience: int = 5,
                       **kwargs) -> PODReconstructionModel:
        """Train POD-based reconstruction model."""
        print("🚀 Training POD-based reconstruction model...")
        
        # Create POD model
        self.pod_model = PODReconstructionModel(
            n_sensors=n_sensors,
            input_shape=self.input_shape,
            n_pod_modes=n_pod_modes,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            gradient_clip_norm=self.gradient_clip_norm
        )
        
        # Set POD modes
        if pod_modes is not None:
            self.pod_model.set_pod_modes(pod_modes)
        elif self.pod_modes is not None:
            self.pod_model.set_pod_modes(self.pod_modes)
        else:
            print("⚠️  No POD modes provided. Model will only predict coefficients.")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.pod_model.compile(optimizer=optimizer)
        
        # Create callbacks
        callback_list = self.create_callbacks(patience=patience, reduce_lr_patience=reduce_lr_patience)
        
        # Train model
        print(f"📊 Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  POD modes: {n_pod_modes}")
        print(f"  Activation: {activation}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Use batch norm: {use_batch_norm}")
        
        history = self.pod_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Save final model
        if self.save_last_model:
            final_weights_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_pod_final_weights"
            self.pod_model.save_weights(str(final_weights_path))
            print(f"💾 Final model weights saved to: {final_weights_path}")
        
        print(f"✅ POD model training completed!")
        return self.pod_model
    
    def load_pod_from_checkpoint(self,
                                n_sensors: int = 32,
                                n_pod_modes: int = 128,
                                pod_modes: np.ndarray = None,
                                hidden_layers: List[int] = [256, 512, 512, 256],
                                activation: str = 'relu',
                                dropout_rate: float = 0.2,
                                use_batch_norm: bool = True,
                                checkpoint_name: str = "best") -> Optional[PODReconstructionModel]:
        """Load POD model from checkpoint."""
        print(f"🔄 Loading POD model from checkpoint...")
        
        # Create model with same architecture
        pod_model = PODReconstructionModel(
            n_sensors=n_sensors,
            input_shape=self.input_shape,
            n_pod_modes=n_pod_modes,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            gradient_clip_norm=self.gradient_clip_norm
        )
        
        # Set POD modes
        if pod_modes is not None:
            pod_model.set_pod_modes(pod_modes)
        elif self.pod_modes is not None:
            pod_model.set_pod_modes(self.pod_modes)
        
        # Load weights
        if checkpoint_name == "best":
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_pod_best"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.model_name}_pod_{checkpoint_name}"
        
        try:
            # Build model by running a dummy forward pass
            dummy_input = tf.zeros((1, n_sensors))
            _ = pod_model(dummy_input)
            
            # Load weights
            pod_model.load_weights(str(checkpoint_path))
            print(f"✅ POD model loaded from: {checkpoint_path}")
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
            pod_model.compile(optimizer=optimizer)
            
            return pod_model
            
        except Exception as e:
            print(f"❌ Failed to load POD model: {e}")
            return None
