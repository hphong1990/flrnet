#!/usr/bin/env python3
"""
Flow Field Dataset Generation Script

This script generates datasets for flow field reconstruction using different sensor placement strategies.
It extracts the core dataset creation functionality from the Jupyter notebook without visualizations.

Usage:
    python data_generation.py [options]

Features:
- Three sensor placement strategies: random, circular, edge
- Multiple sensor configurations: 8, 16, 32, 64 sensors
- Automated dataset creation for all configurations
- Support for custom data paths and output directories
"""

import argparse
import sys
import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the dataset creation system to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Import the dataset creation system
try:
    from data import FlowFieldDatasetCreator
    from data.utils import (
        validate_sensor_positions, 
        calculate_dataset_statistics,
        inspect_data_directory
    )
except ImportError as e:
    print(f"❌ Error importing dataset creation modules: {e}")
    print("💡 Make sure the 'data' folder contains the required modules.")
    sys.exit(1)


def setup_logging():
    """Setup basic logging for the script."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate flow field datasets for machine learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="E:/Research/Data/NavierStokes/train/",
        help="Path to the raw Navier-Stokes data files"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/",
        help="Path to save generated datasets and sensor layouts"
    )
    
    parser.add_argument(
        "--domain-shape",
        type=int,
        nargs=2,
        default=[128, 256],
        help="Domain shape as height width"
    )
    
    parser.add_argument(
        "--time-steps",
        type=int,
        default=39,
        help="Number of time steps in the data"
    )
    
    parser.add_argument(
        "--sensor-counts",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="List of sensor counts to generate datasets for"
    )
    
    parser.add_argument(
        "--layouts",
        type=str,
        nargs="+",
        choices=["random", "circular", "edge"],
        default=["random", "circular", "edge"],
        help="Sensor layout types to generate"
    )
    
    parser.add_argument(
        "--reynolds-numbers",
        type=int,
        nargs="*",
        help="Specific Reynolds numbers to process (if not provided, will read from data folder)"
    )
    
    parser.add_argument(
        "--obstacle-center",
        type=int,
        nargs=2,
        default=[64, 128],
        help="Obstacle center coordinates as x y"
    )
    
    parser.add_argument(
        "--obstacle-radius",
        type=int,
        default=22,
        help="Obstacle radius"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data and setup, don't generate datasets"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def validate_data_directory(data_path, logger):
    """Validate and inspect the data directory."""
    logger.info(f"🔍 Inspecting data directory: {data_path}")
    
    # Create configuration for test data creation
    creator_config = {
        'time_steps': 39,
        'domain_shape': (128, 256),
        'obstacle_center': (64, 128),
        'obstacle_radius': 22
    }
    
    # Inspect the data directory
    inspection_results = inspect_data_directory(
        data_path=data_path,
        visualize=False,  # No visualization in CLI
        create_test_data=True,
        creator_config=creator_config
    )
    
    # Process results
    if inspection_results['directory_exists'] and inspection_results['file_count'] > 0:
        logger.info(f"✅ Found {inspection_results['file_count']} data files")
        if inspection_results['data_info']:
            sample_data_shape = inspection_results['data_info']['shape']
            logger.info(f"📊 Sample data shape: {sample_data_shape}")
            return True, sample_data_shape
    elif inspection_results['test_data_created']:
        logger.info(f"✅ Test data created at: {inspection_results['test_data_path']}")
        sample_data_shape = inspection_results['data_info']['shape']
        logger.info(f"📊 Test data shape: {sample_data_shape}")
        return True, sample_data_shape
    else:
        logger.error("❌ No data found and no test data created")
        return False, None


def detect_data_format(data_path, logger):
    """Detect data format and return appropriate configuration."""
    data_dir = Path(data_path)
    
    # Find Reynolds number files
    reynolds_files = list(data_dir.glob("Re_*.npy"))
    if not reynolds_files:
        logger.error("❌ No Reynolds number files found in the data directory")
        return None, None, None, None
    
    reynolds_numbers = sorted([int(file.stem.split('_')[1]) for file in reynolds_files])
    logger.info(f"🔢 Found Reynolds numbers: {reynolds_numbers}")
    logger.info(f"📊 Total Reynolds numbers: {len(reynolds_numbers)}")
    
    # Inspect actual data format
    sample_file = data_dir / f"Re_{reynolds_numbers[0]}.npy"
    sample_data = np.load(sample_file)
    logger.info(f"📊 Sample data shape: {sample_data.shape}")
    
    # Determine the correct dimensions based on the actual data
    if len(sample_data.shape) == 4:
        # Data format: (time_steps, height, width, channels)
        time_steps, height, width, channels = sample_data.shape
        domain_shape = (height, width)
        logger.info(f"✅ 4D data format detected: (time_steps={time_steps}, height={height}, width={width}, channels={channels})")
    elif len(sample_data.shape) == 3:
        # Data format: (time_steps, height, channels) - assuming domain width
        time_steps, height, channels = sample_data.shape
        width = 256  # Use the full width as specified
        domain_shape = (height, width)
        logger.info(f"✅ 3D data format detected: (time_steps={time_steps}, height={height}, channels={channels})")
        logger.info(f"🔧 Assuming width={width} based on field size specification")
    else:
        logger.error(f"❌ Unexpected data format: {sample_data.shape}")
        return None, None, None, None
    
    logger.info(f"🎯 Using domain shape: {domain_shape}")
    logger.info(f"⏱️ Using time steps: {time_steps}")
    
    return reynolds_numbers, domain_shape, time_steps, sample_data.shape


def create_dataset_creator(args, reynolds_numbers, domain_shape, time_steps, logger):
    """Create and configure the FlowFieldDatasetCreator."""
    logger.info("🚀 Initializing Dataset Creator...")
    
    creator = FlowFieldDatasetCreator(
        data_path=args.data_path,
        output_path=args.output_path,
        domain_shape=domain_shape,
        time_steps=time_steps,
        reynolds_numbers=reynolds_numbers
    )
    
    # Update obstacle parameters
    creator.obstacle_center = tuple(args.obstacle_center)
    creator.obstacle_radius = args.obstacle_radius
    creator.sensor_counts = args.sensor_counts
    
    logger.info("✅ Creator configuration:")
    logger.info(f"   - Data path: {creator.data_path}")
    logger.info(f"   - Output path: {creator.output_path}")
    logger.info(f"   - Domain shape: {creator.domain_shape}")
    logger.info(f"   - Time steps: {creator.time_steps}")
    logger.info(f"   - Obstacle center: {creator.obstacle_center}")
    logger.info(f"   - Obstacle radius: {creator.obstacle_radius}")
    logger.info(f"   - Sensor counts: {creator.sensor_counts}")
    logger.info(f"   - Reynolds numbers: {len(creator.reynolds_numbers)}")
    
    return creator


def generate_datasets(creator, layouts, sensor_counts, logger):
    """Generate datasets for all specified configurations."""
    logger.info("🏗️ Starting Dataset Generation...")
    logger.info("=" * 50)
    
    # Define configurations to create
    configurations = []
    for layout_type in layouts:
        for n_sensors in sensor_counts:
            configurations.append((layout_type, n_sensors))
    
    logger.info(f"📋 Will create {len(configurations)} datasets:")
    for layout_type, n_sensors in configurations:
        logger.info(f"   - {layout_type} layout with {n_sensors} sensors")
    
    created_datasets = {}
    success_count = 0
    error_count = 0
    
    for layout_type, n_sensors in configurations:
        logger.info(f"\n📊 Creating {layout_type} dataset with {n_sensors} sensors...")
        
        start_time = time.time()
        
        try:
            # Create dataset
            dataset = creator.create_dataset_for_layout(layout_type, n_sensors)
            
            # Save dataset
            creator.save_dataset(dataset, layout_type, n_sensors)
            
            # Store for statistics
            created_datasets[f'{layout_type}_{n_sensors}'] = dataset
            
            # Calculate statistics
            stats = calculate_dataset_statistics(dataset)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(f"✅ Successfully created {layout_type} dataset!")
            logger.info(f"⏱️ Time taken: {elapsed_time:.2f} seconds")
            logger.info(f"📈 Dataset Statistics:")
            logger.info(f"   - Sensor data: {stats['sensor_data']['shape']}")
            logger.info(f"   - Field data: {stats['field_data']['shape']}")
            logger.info(f"   - Sensor data range: [{stats['sensor_data']['min']:.3f}, {stats['sensor_data']['max']:.3f}]")
            logger.info(f"   - Field data range: [{stats['field_data']['min']:.3f}, {stats['field_data']['max']:.3f}]")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error creating {layout_type} dataset: {str(e)}")
            logger.error(f"   Error details: {type(e).__name__}")
            error_count += 1
            
            if logger.level <= 10:  # DEBUG level
                import traceback
                logger.debug(f"   Full traceback: {traceback.format_exc()}")
    
    # Final summary
    logger.info(f"\n📋 Dataset Generation Summary:")
    logger.info("=" * 50)
    logger.info(f"✅ Successfully created: {success_count} datasets")
    if error_count > 0:
        logger.info(f"❌ Failed to create: {error_count} datasets")
    logger.info(f"📁 Total datasets: {len(created_datasets)}")
    
    return created_datasets


def validate_generated_datasets(creator, created_datasets, logger):
    """Validate the generated datasets."""
    logger.info("\n🔍 Validating Generated Datasets...")
    logger.info("-" * 40)
    
    # Get information about available datasets
    info = creator.get_dataset_info()
    logger.info(f"📈 Available datasets: {len(info['available_datasets'])}")
    logger.info(f"🎯 Dataset list: {info['available_datasets']}")
    
    # Validate each created dataset
    for dataset_name, dataset in created_datasets.items():
        logger.info(f"\n📊 Validating {dataset_name}:")
        
        # Validate sensor positions
        sensor_positions = dataset['sensor_positions']
        errors = validate_sensor_positions(sensor_positions, creator.domain_shape)
        
        if errors:
            logger.warning(f"   ⚠️ Validation warnings: {errors}")
        else:
            logger.info(f"   ✅ Sensor positions valid")
        
        # Basic data consistency checks
        sensor_data = dataset['sensor_data']
        field_data = dataset['field_data']
        
        logger.info(f"   📊 Sensor data shape: {sensor_data.shape}")
        logger.info(f"   📊 Field data shape: {field_data.shape}")
        logger.info(f"   📊 Layout type: {dataset['layout_type']}")
        logger.info(f"   📊 Number of sensors: {dataset['n_sensors']}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(sensor_data)) or np.any(np.isinf(sensor_data)):
            logger.warning(f"   ⚠️ Found NaN/Inf values in sensor data")
        
        if np.any(np.isnan(field_data)) or np.any(np.isinf(field_data)):
            logger.warning(f"   ⚠️ Found NaN/Inf values in field data")


def main():
    """Main function."""
    args = parse_arguments()
    logger = setup_logging()
    
    if args.verbose:
        logger.setLevel(10)  # DEBUG level
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    logger.info(f"🎲 Random seed set to: {args.seed}")
    
    logger.info("🚀 Flow Field Dataset Generation Script")
    logger.info("=" * 50)
    logger.info(f"📁 Data path: {args.data_path}")
    logger.info(f"📂 Output path: {args.output_path}")
    logger.info(f"📐 Domain shape: {args.domain_shape}")
    logger.info(f"⏱️ Time steps: {args.time_steps}")
    logger.info(f"🎯 Sensor counts: {args.sensor_counts}")
    logger.info(f"📋 Layouts: {args.layouts}")
    
    # Step 1: Validate data directory
    data_valid, sample_shape = validate_data_directory(args.data_path, logger)
    if not data_valid:
        logger.error("❌ Data validation failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Detect data format and configuration
    reynolds_numbers, domain_shape, time_steps, data_shape = detect_data_format(args.data_path, logger)
    if reynolds_numbers is None:
        logger.error("❌ Failed to detect data format. Exiting.")
        sys.exit(1)
    
    # Override with command line arguments if provided
    if args.reynolds_numbers:
        reynolds_numbers = args.reynolds_numbers
        logger.info(f"🔧 Using provided Reynolds numbers: {reynolds_numbers}")
    
    # Use detected values or command line overrides
    final_domain_shape = tuple(args.domain_shape) if args.domain_shape != [128, 256] else domain_shape
    final_time_steps = args.time_steps if args.time_steps != 39 else time_steps
    
    logger.info(f"🎯 Final configuration:")
    logger.info(f"   - Domain shape: {final_domain_shape}")
    logger.info(f"   - Time steps: {final_time_steps}")
    logger.info(f"   - Reynolds numbers: {len(reynolds_numbers)}")
    
    # Step 3: Create dataset creator
    creator = create_dataset_creator(args, reynolds_numbers, final_domain_shape, final_time_steps, logger)
    
    # Step 4: Validation only mode
    if args.validate_only:
        logger.info("✅ Validation completed successfully. Exiting (--validate-only mode).")
        return
    
    # Step 5: Generate datasets
    total_start_time = time.time()
    created_datasets = generate_datasets(creator, args.layouts, args.sensor_counts, logger)
    total_end_time = time.time()
    
    # Step 6: Validate generated datasets
    if created_datasets:
        validate_generated_datasets(creator, created_datasets, logger)
    
    # Final summary
    total_time = total_end_time - total_start_time
    logger.info(f"\n🎉 Dataset Generation Complete!")
    logger.info("=" * 50)
    logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
    logger.info(f"📁 Created {len(created_datasets)} datasets")
    logger.info(f"📂 Output directory: {args.output_path}")
    logger.info(f"🎯 Datasets available for machine learning training!")


if __name__ == "__main__":
    main()
