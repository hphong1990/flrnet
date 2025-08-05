"""
Configuration Management for Flow Field Reconstruction Training

This module provides utilities for loading and managing YAML configuration files
for training flow field reconstruction models with different sensor layouts,
architectures, and hyperparameters.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation for FLR training."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing YAML configuration files
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config file (with or without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        # Add .yaml extension if not present
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate and process the configuration
            config = self._process_config(config)
            
            logger.info(f"✅ Loaded configuration: {config_name}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration {config_path}: {e}")
    
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate the configuration.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Processed configuration dictionary
        """
        # Generate dataset path
        sensors = config['sensors']
        layout = sensors['layout']
        n_sensors = sensors['n_sensors']
        dataset_filename = f"dataset_{layout}_{n_sensors}.npz"
        config['dataset_path'] = os.path.join(config['data']['dataset_base_path'], dataset_filename)
        
        # Generate checkpoint and log directories with naming convention
        model_config = config['model']
        use_fourier = model_config['use_fourier']
        use_perceptual = model_config['use_perceptual_loss']
        
        # Naming convention: checkpoint_<fourierTrue/False>_<percepTrue/False>_<type>_<n_sensors>
        fourier_str = "fourierTrue" if use_fourier else "fourierFalse"
        percep_str = "percepTrue" if use_perceptual else "percepFalse"
        
        model_name = f"{fourier_str}_{percep_str}_{layout}_{n_sensors}"
        
        # Create full paths
        base_checkpoint_dir = config['output']['checkpoint_base_dir']
        base_logs_dir = config['output']['logs_base_dir']
        
        config['checkpoint_dir'] = os.path.join(base_checkpoint_dir, model_name)
        config['logs_dir'] = os.path.join(base_logs_dir, model_name)
        config['model_name'] = model_name
        
        # Validate required fields
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration.
        
        Args:
            config: Configuration dictionary to validate
        """
        required_sections = ['model', 'sensors', 'training', 'data', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate sensor layout
        valid_layouts = ['edge', 'circular', 'random']
        layout = config['sensors']['layout']
        if layout not in valid_layouts:
            raise ValueError(f"Invalid sensor layout: {layout}. Must be one of {valid_layouts}")
        
        # Validate sensor counts
        valid_counts = [8, 16, 32]
        n_sensors = config['sensors']['n_sensors']
        if n_sensors not in valid_counts:
            raise ValueError(f"Invalid sensor count: {n_sensors}. Must be one of {valid_counts}")
        
        # Validate model parameters
        model = config['model']
        if not isinstance(model['use_fourier'], bool):
            raise ValueError("model.use_fourier must be a boolean")
        if not isinstance(model['use_perceptual_loss'], bool):
            raise ValueError("model.use_perceptual_loss must be a boolean")
    
    def list_configs(self) -> list:
        """
        List all available configuration files.
        
        Returns:
            List of configuration file names (without .yaml extension)
        """
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.stem)
        return sorted(config_files)
    
    def create_config_summary(self, config: Dict[str, Any]) -> str:
        """
        Create a summary string of the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Formatted summary string
        """
        model = config['model']
        sensors = config['sensors']
        training = config['training']
        
        summary = f"""
📋 Configuration Summary
{'='*50}
🏗️  Model Architecture:
   - Fourier Enhancement: {model['use_fourier']}
   - Perceptual Loss: {model['use_perceptual_loss']}
   - Input Shape: {model['input_shape']}
   - Latent Dimensions: {model['latent_dims']}
   - Base Features: {model['n_base_features']}

📡 Sensor Configuration:
   - Layout: {sensors['layout']}
   - Number of Sensors: {sensors['n_sensors']}
   - Dataset: {config['dataset_path']}

🚀 Training Parameters:
   - VAE Epochs: {training['vae_epochs']}
   - FLRNet Epochs: {training['flr_epochs']}
   - VAE Learning Rate: {training['vae_learning_rate']}
   - FLRNet Learning Rate: {training['flr_learning_rate']}
   - Batch Size: {training['batch_size']}
   - Test Split: {training['test_split']}

💾 Output Configuration:
   - Model Name: {config['model_name']}
   - Checkpoints: {config['checkpoint_dir']}
   - Logs: {config['logs_dir']}
   - Save Best Model: {config['output']['save_best_model']}
   - Save Last Model: {config['output']['save_last_model']}
"""
        return summary


def load_training_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Convenience function to load a training configuration.
    
    Args:
        config_name: Name of the configuration file to load
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_config(config_name)


def list_available_configs() -> list:
    """
    Convenience function to list available configurations.
    
    Returns:
        List of available configuration names
    """
    manager = ConfigManager()
    return manager.list_configs()


def flatten_config_for_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the hierarchical config into a flat dictionary with uppercase keys
    for backward compatibility with existing training code.
    
    Args:
        config: Hierarchical configuration dictionary
        
    Returns:
        Flattened configuration dictionary with uppercase keys
    """
    flattened = {}
    
    # Model parameters
    model = config['model']
    flattened['USE_FOURIER'] = model['use_fourier']
    flattened['USE_PERCEPTUAL_LOSS'] = model['use_perceptual_loss']
    flattened['INPUT_SHAPE'] = tuple(model['input_shape'])
    flattened['LATENT_DIMS'] = model['latent_dims']
    flattened['N_BASE_FEATURES'] = model['n_base_features']
    
    # Sensor parameters
    sensors = config['sensors']
    flattened['SENSOR_LAYOUT'] = sensors['layout']
    flattened['N_SENSORS'] = sensors['n_sensors']
    
    # Training parameters
    training = config['training']
    flattened['VAE_EPOCHS'] = training['vae_epochs']
    flattened['FLR_EPOCHS'] = training['flr_epochs']
    flattened['VAE_LEARNING_RATE'] = training['vae_learning_rate']
    flattened['FLR_LEARNING_RATE'] = training['flr_learning_rate']
    flattened['BATCH_SIZE'] = training['batch_size']
    flattened['TEST_SPLIT'] = training['test_split']
    flattened['SHUFFLE_BUFFER'] = training['shuffle_buffer']
    flattened['PATIENCE'] = training['patience']
    flattened['REDUCE_LR_PATIENCE'] = training['reduce_lr_patience']
    flattened['GRADIENT_CLIP_NORM'] = training.get('gradient_clip_norm', 1.0)
    
    # Data parameters
    flattened['DATASET_PATH'] = config['dataset_path']
    
    # Output parameters
    flattened['CHECKPOINT_DIR'] = config['checkpoint_dir']
    flattened['LOGS_DIR'] = config['logs_dir']
    flattened['MODEL_NAME'] = config['model_name']
    flattened['SAVE_BEST_MODEL'] = config['output']['save_best_model']
    flattened['SAVE_LAST_MODEL'] = config['output']['save_last_model']
    
    # Random seed
    flattened['RANDOM_SEED'] = config.get('random_seed', 42)
    
    return flattened


# Example usage
if __name__ == "__main__":
    # Load default configuration
    config = load_training_config("default")
    
    # Create and print summary
    manager = ConfigManager()
    print(manager.create_config_summary(config))
    
    # List available configurations
    print("\n📁 Available Configurations:")
    for config_name in list_available_configs():
        print(f"   - {config_name}")
