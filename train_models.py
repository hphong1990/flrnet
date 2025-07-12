#!/usr/bin/env python3
"""
Command-line training script for Flow Field Reconstruction models.

This script provides a simple interface to train both VAE and FLRNet models
using the improved architecture with callbacks and validation.

Usage:
    python train_models.py --config config.json
    python train_models.py --use_fourier --n_sensors 32 --vae_epochs 100 --flr_epochs 500
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from models_improved import FlowFieldTrainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_data(data_config: Dict[str, Any]) -> tuple:
    """
    Load training and validation datasets.
    
    Args:
        data_config: Configuration for data loading
        
    Returns:
        Tuple of (train_dataset, val_dataset, normalization_params)
    """
    data_path = data_config['data_path']
    re_train = data_config['reynolds_train']
    re_val = data_config['reynolds_val']
    n_sensors = data_config['n_sensors']
    sensor_layout = data_config.get('sensor_layout', 'random')
    use_fourier = data_config['use_fourier']
    batch_size = data_config['batch_size']
    
    print(f"📁 Loading data from: {data_path}")
    print(f"🔢 Reynolds numbers - Train: {len(re_train)}, Val: {len(re_val)}")
    print(f"📡 Sensors: {n_sensors} ({sensor_layout} layout)")
    
    # Load sensor and field data
    def load_reynolds_data(reynolds_list):
        sensor_data_list = []
        field_data_list = []
        
        for Re in reynolds_list:
            # Sensor data file paths
            if sensor_layout == 'random':
                sensor_file = f"{data_path}/random_sensor_data/sensor_data_{n_sensors}_{Re}.npy"
            elif sensor_layout == 'circular':
                sensor_file = f"{data_path}/circular_sensor_pos_data/sensor_data_cir_{n_sensors}_{Re}.npy"
            elif sensor_layout == 'edge':
                sensor_file = f"{data_path}/edge_sensor_pos_data/sensor_data_edge_{n_sensors}_{Re}.npy"
            
            # Field data file path
            field_file = f"{data_path}/full_field_data/full_field_data_{Re}.npy"
            
            try:
                sensor_data = np.load(sensor_file)
                field_data = np.load(field_file)
                sensor_data_list.append(sensor_data)
                field_data_list.append(field_data)
            except FileNotFoundError as e:
                print(f"⚠️ Warning: {e}")
                continue
        
        if not sensor_data_list:
            return None, None
        
        # Concatenate and reshape
        sensor_array = np.swapaxes(np.concatenate(sensor_data_list, axis=-1), 0, 1)
        field_array = np.swapaxes(
            np.expand_dims(np.concatenate(field_data_list, axis=-1), axis=0), 0, -1
        )
        
        return sensor_array, field_array
    
    # Load training and validation data
    sensor_train, field_train = load_reynolds_data(re_train)
    sensor_val, field_val = load_reynolds_data(re_val)
    
    if sensor_train is None or sensor_val is None:
        raise ValueError("Failed to load data files. Check data_path and file structure.")
    
    # Normalization
    min_val = np.min(field_train)
    max_val = np.max(field_train)
    
    field_train_norm = (field_train - min_val) / (max_val - min_val)
    field_val_norm = (field_val - min_val) / (max_val - min_val)
    sensor_train_norm = (sensor_train - min_val) / (max_val - min_val)
    sensor_val_norm = (sensor_val - min_val) / (max_val - min_val)
    
    print(f"✅ Data loaded and normalized:")
    print(f"   Train - Sensors: {sensor_train_norm.shape}, Fields: {field_train_norm.shape}")
    print(f"   Val - Sensors: {sensor_val_norm.shape}, Fields: {field_val_norm.shape}")
    print(f"   Normalization range: [{min_val:.3f}, {max_val:.3f}]")
    
    # Create coordinate data for Fourier version
    if use_fourier:
        def create_coordinates(field_shape):
            dim = field_shape
            x = np.linspace(0, 1, dim[2])
            y = np.linspace(0, 1, dim[1])
            coordX, coordY = np.meshgrid(x, y)
            coordX = np.expand_dims(coordX, axis=-1)
            coordY = np.expand_dims(coordY, axis=-1)
            coord_data = np.concatenate([coordX, coordY], axis=-1)
            coord_data = np.repeat(coord_data[np.newaxis, :, :, :], dim[0], axis=0)
            return coord_data.astype(np.float32)
        
        coord_train = create_coordinates(field_train_norm.shape)
        coord_val = create_coordinates(field_val_norm.shape)
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            sensor_train_norm, field_train_norm, coord_train
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            sensor_val_norm, field_val_norm, coord_val
        ))
    else:
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            sensor_train_norm, field_train_norm
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            sensor_val_norm, field_val_norm
        ))
    
    # Batch and prefetch datasets
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    normalization_params = {'min_val': float(min_val), 'max_val': float(max_val)}
    
    return train_dataset, val_dataset, normalization_params


def main():
    parser = argparse.ArgumentParser(description='Train Flow Field Reconstruction Models')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    
    # Model parameters
    parser.add_argument('--use_fourier', action='store_true', help='Use Fourier-enhanced architecture')
    parser.add_argument('--n_sensors', type=int, default=32, help='Number of sensors')
    parser.add_argument('--input_shape', nargs=3, type=int, default=[128, 256, 1], help='Input shape (H W C)')
    parser.add_argument('--latent_dims', type=int, default=4, help='Latent space dimensions')
    parser.add_argument('--n_base_features', type=int, default=64, help='Base number of features')
    
    # Training parameters
    parser.add_argument('--vae_epochs', type=int, default=100, help='VAE training epochs')
    parser.add_argument('--vae_lr', type=float, default=1e-4, help='VAE learning rate')
    parser.add_argument('--flr_epochs', type=int, default=500, help='FLRNet training epochs')
    parser.add_argument('--flr_lr', type=float, default=5e-5, help='FLRNet learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='E:/Research/Data/flow_field_recon', 
                       help='Path to data directory')
    parser.add_argument('--sensor_layout', type=str, default='random', 
                       choices=['random', 'circular', 'edge'], help='Sensor layout type')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='Logs directory')
    parser.add_argument('--save_config', type=str, help='Save configuration to file')
    
    # Training control
    parser.add_argument('--skip_vae', action='store_true', help='Skip VAE training (load pretrained)')
    parser.add_argument('--skip_flr', action='store_true', help='Skip FLRNet training')
    parser.add_argument('--gpu_memory_growth', action='store_true', help='Enable GPU memory growth')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        config = load_config(args.config)
        print(f"📄 Loaded configuration from: {args.config}")
    else:
        # Create configuration from command line arguments
        config = {
            'model': {
                'use_fourier': args.use_fourier,
                'n_sensors': args.n_sensors,
                'input_shape': args.input_shape,
                'latent_dims': args.latent_dims,
                'n_base_features': args.n_base_features
            },
            'training': {
                'vae_epochs': args.vae_epochs,
                'vae_learning_rate': args.vae_lr,
                'flr_epochs': args.flr_epochs,
                'flr_learning_rate': args.flr_lr,
                'batch_size': args.batch_size
            },
            'data': {
                'data_path': args.data_path,
                'sensor_layout': args.sensor_layout,
                'reynolds_train': [300, 400, 450, 500, 600, 650, 700, 800, 850, 900, 1000],
                'reynolds_val': [350, 550, 750, 950],
                'use_fourier': args.use_fourier,
                'n_sensors': args.n_sensors,
                'batch_size': args.batch_size
            },
            'output': {
                'checkpoint_dir': args.checkpoint_dir,
                'logs_dir': args.logs_dir
            },
            'control': {
                'skip_vae': args.skip_vae,
                'skip_flr': args.skip_flr
            }
        }
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
        print(f"💾 Configuration saved to: {args.save_config}")
    
    # Configure GPU
    if args.gpu_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"❌ GPU setup error: {e}")
    
    print(f"🚀 Starting training with configuration:")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {len(tf.config.list_physical_devices('GPU'))} devices")
    print(f"   Use Fourier: {config['model']['use_fourier']}")
    print(f"   Sensors: {config['model']['n_sensors']}")
    print(f"   VAE epochs: {config['training']['vae_epochs']}")
    print(f"   FLRNet epochs: {config['training']['flr_epochs']}")
    
    # Load data
    try:
        train_dataset, val_dataset, norm_params = load_data(config['data'])
        
        # Save normalization parameters
        norm_file = Path(config['output']['checkpoint_dir']) / 'normalization_params.json'
        norm_file.parent.mkdir(exist_ok=True)
        save_config(norm_params, str(norm_file))
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print(f"💡 Check data path and file structure")
        return 1
    
    # Create trainer
    trainer = FlowFieldTrainer(
        input_shape=tuple(config['model']['input_shape']),
        use_fourier=config['model']['use_fourier'],
        checkpoint_dir=config['output']['checkpoint_dir'],
        logs_dir=config['output']['logs_dir']
    )
    
    # Train VAE
    if not config['control']['skip_vae']:
        print(f"\n🏗️ Training VAE...")
        vae_model = trainer.train_vae(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=config['training']['vae_epochs'],
            learning_rate=config['training']['vae_learning_rate'],
            latent_dims=config['model']['latent_dims'],
            n_base_features=config['model']['n_base_features']
        )
    else:
        print(f"⏭️ Skipping VAE training")
        vae_model = None
    
    # Train FLRNet
    if not config['control']['skip_flr']:
        print(f"\n🎯 Training FLRNet...")
        flr_model = trainer.train_flr_net(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_sensors=config['model']['n_sensors'],
            epochs=config['training']['flr_epochs'],
            learning_rate=config['training']['flr_learning_rate'],
            pretrained_vae=vae_model
        )
    else:
        print(f"⏭️ Skipping FLRNet training")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Models saved to: {config['output']['checkpoint_dir']}")
    print(f"📊 Logs available at: {config['output']['logs_dir']}")
    print(f"🔍 View training progress: tensorboard --logdir {config['output']['logs_dir']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
