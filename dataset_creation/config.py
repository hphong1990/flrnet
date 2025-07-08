"""
Configuration file for FlowFieldDatasetCreator.

This file contains all the configuration parameters for dataset creation.
Modify these parameters according to your specific requirements.
"""

import os
from pathlib import Path

# =============================================================================
# DATA PATHS
# =============================================================================

# Path to the raw Navier-Stokes data files
# Update this path according to your system
RAW_DATA_PATH = "D:/data/Navier-Stokes/Navier-Stokes/"

# Alternative path for full field data (if using processed data)
FULL_FIELD_DATA_PATH = "E:/Research/Data/flow_field_recon/full_field_data/"

# Output path for generated datasets and sensor layouts
OUTPUT_PATH = "./dataset_creation/"

# =============================================================================
# DOMAIN PARAMETERS
# =============================================================================

# Shape of the flow domain (height, width)
DOMAIN_SHAPE = (128, 256)

# Number of time steps in the data
TIME_STEPS = 39

# Flow region parameters
FLOW_REGION_Y_START = 80  # Start of flow region in y-direction

# Obstacle parameters (for circular sensor placement)
OBSTACLE_CENTER = (64, 66)  # Center of circular obstacle
OBSTACLE_RADIUS = 21        # Radius of circular obstacle

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Reynolds numbers to process
REYNOLDS_NUMBERS = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# Alternative: subset for testing
# REYNOLDS_NUMBERS = [300, 500, 750, 1000]

# =============================================================================
# SENSOR CONFIGURATION
# =============================================================================

# Number of sensors for each configuration
SENSOR_COUNTS = [8, 16, 32, 64]

# Alternative: subset for testing
# SENSOR_COUNTS = [8, 16, 32]

# Sensor layout types
SENSOR_LAYOUT_TYPES = ['random', 'circular', 'edge']

# =============================================================================
# TENSORFLOW DATASET PARAMETERS
# =============================================================================

# Default batch size for TensorFlow datasets
DEFAULT_BATCH_SIZE = 32

# Default test split ratio
DEFAULT_TEST_SPLIT = 0.2

# Whether to shuffle data by default
DEFAULT_SHUFFLE = True

# Buffer size for shuffling
SHUFFLE_BUFFER_SIZE = 1000

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Figure size for visualizations
FIGURE_SIZE = (12, 6)

# DPI for saved figures
FIGURE_DPI = 300

# Colormap for velocity field visualization
VELOCITY_COLORMAP = 'viridis'

# Velocity magnitude range for visualization
VELOCITY_RANGE = (0, 2)

# Sensor marker properties
SENSOR_MARKER_SIZE = 50
SENSOR_MARKER_COLOR = 'red'
SENSOR_MARKER_EDGE_COLOR = 'white'
SENSOR_MARKER_EDGE_WIDTH = 1

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

# Naming patterns for different file types
SENSOR_LAYOUT_FILENAME_PATTERN = "sensor_layout_{layout_type}_{n_sensors}.npy"
DATASET_FILENAME_PATTERN = "dataset_{layout_type}_{n_sensors}.npz"
VISUALIZATION_FILENAME_PATTERN = "sensor_layout_{layout_type}_{n_sensors}.png"

# Raw data filename pattern
RAW_DATA_FILENAME_PATTERN = "Re_{reynolds_number}.npy"

# =============================================================================
# QUALITY CONTROL PARAMETERS
# =============================================================================

# Minimum and maximum sensor positions (to ensure they're within domain)
MIN_SENSOR_X = 0
MAX_SENSOR_X = DOMAIN_SHAPE[0] - 1
MIN_SENSOR_Y = FLOW_REGION_Y_START
MAX_SENSOR_Y = DOMAIN_SHAPE[1] - 1

# Random seed for reproducibility (set to None for random behavior)
RANDOM_SEED = 42

# =============================================================================
# ADVANCED PARAMETERS
# =============================================================================

# Memory optimization settings
USE_MEMORY_OPTIMIZATION = True

# Compression level for saved datasets (0-9, higher = more compression)
COMPRESSION_LEVEL = 6

# Whether to save intermediate results
SAVE_INTERMEDIATE_RESULTS = True

# Whether to create visualizations by default
CREATE_VISUALIZATIONS = True

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Whether to validate sensor positions
VALIDATE_SENSOR_POSITIONS = True

# Whether to validate data ranges
VALIDATE_DATA_RANGES = True

# Expected data range for velocity magnitudes
EXPECTED_VELOCITY_RANGE = (0, 10)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_dict():
    """
    Get all configuration parameters as a dictionary.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    return {
        # Data paths
        'raw_data_path': RAW_DATA_PATH,
        'full_field_data_path': FULL_FIELD_DATA_PATH,
        'output_path': OUTPUT_PATH,
        
        # Domain parameters
        'domain_shape': DOMAIN_SHAPE,
        'time_steps': TIME_STEPS,
        'flow_region_y_start': FLOW_REGION_Y_START,
        'obstacle_center': OBSTACLE_CENTER,
        'obstacle_radius': OBSTACLE_RADIUS,
        
        # Simulation parameters
        'reynolds_numbers': REYNOLDS_NUMBERS,
        
        # Sensor configuration
        'sensor_counts': SENSOR_COUNTS,
        'sensor_layout_types': SENSOR_LAYOUT_TYPES,
        
        # TensorFlow dataset parameters
        'default_batch_size': DEFAULT_BATCH_SIZE,
        'default_test_split': DEFAULT_TEST_SPLIT,
        'default_shuffle': DEFAULT_SHUFFLE,
        'shuffle_buffer_size': SHUFFLE_BUFFER_SIZE,
        
        # Visualization parameters
        'figure_size': FIGURE_SIZE,
        'figure_dpi': FIGURE_DPI,
        'velocity_colormap': VELOCITY_COLORMAP,
        'velocity_range': VELOCITY_RANGE,
        
        # Quality control
        'random_seed': RANDOM_SEED,
        'validate_sensor_positions': VALIDATE_SENSOR_POSITIONS,
        'validate_data_ranges': VALIDATE_DATA_RANGES,
        'expected_velocity_range': EXPECTED_VELOCITY_RANGE,
        
        # Advanced parameters
        'use_memory_optimization': USE_MEMORY_OPTIMIZATION,
        'compression_level': COMPRESSION_LEVEL,
        'save_intermediate_results': SAVE_INTERMEDIATE_RESULTS,
        'create_visualizations': CREATE_VISUALIZATIONS,
    }


def validate_config():
    """
    Validate configuration parameters.
    
    Returns:
        List of validation errors (empty if no errors)
    """
    errors = []
    
    # Check paths
    if not os.path.exists(RAW_DATA_PATH):
        errors.append(f"Raw data path does not exist: {RAW_DATA_PATH}")
    
    # Check domain parameters
    if len(DOMAIN_SHAPE) != 2:
        errors.append("DOMAIN_SHAPE must be a tuple of length 2")
    
    if TIME_STEPS <= 0:
        errors.append("TIME_STEPS must be positive")
    
    # Check sensor parameters
    if not SENSOR_COUNTS:
        errors.append("SENSOR_COUNTS cannot be empty")
    
    if not SENSOR_LAYOUT_TYPES:
        errors.append("SENSOR_LAYOUT_TYPES cannot be empty")
    
    # Check Reynolds numbers
    if not REYNOLDS_NUMBERS:
        errors.append("REYNOLDS_NUMBERS cannot be empty")
    
    # Check batch size
    if DEFAULT_BATCH_SIZE <= 0:
        errors.append("DEFAULT_BATCH_SIZE must be positive")
    
    # Check test split
    if not (0 < DEFAULT_TEST_SPLIT < 1):
        errors.append("DEFAULT_TEST_SPLIT must be between 0 and 1")
    
    return errors


def print_config():
    """
    Print current configuration parameters.
    """
    config = get_config_dict()
    
    print("="*60)
    print("FLOW FIELD DATASET CREATOR CONFIGURATION")
    print("="*60)
    
    for section, params in [
        ("Data Paths", ['raw_data_path', 'output_path']),
        ("Domain Parameters", ['domain_shape', 'time_steps', 'flow_region_y_start']),
        ("Simulation Parameters", ['reynolds_numbers']),
        ("Sensor Configuration", ['sensor_counts', 'sensor_layout_types']),
        ("TensorFlow Parameters", ['default_batch_size', 'default_test_split']),
        ("Quality Control", ['random_seed', 'validate_sensor_positions'])
    ]:
        print(f"\n{section}:")
        print("-" * len(section))
        for param in params:
            if param in config:
                print(f"  {param}: {config[param]}")
    
    print("="*60)


if __name__ == "__main__":
    # Print configuration
    print_config()
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\n✓ Configuration is valid!")
