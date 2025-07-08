"""
Example usage of the FlowFieldDatasetCreator class.

This script demonstrates how to use the FlowFieldDatasetCreator to create
datasets for flow field reconstruction with different sensor layouts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flow_field_dataset import FlowFieldDatasetCreator
import numpy as np


def main():
    """
    Main function demonstrating the usage of FlowFieldDatasetCreator.
    """
    
    # Initialize the dataset creator
    # Update these paths according to your system
    data_path = "D:/data/Navier-Stokes/Navier-Stokes/"  # Path to your raw data
    output_path = "./dataset_creation/"  # Output path for datasets
    
    creator = FlowFieldDatasetCreator(
        data_path=data_path,
        output_path=output_path,
        domain_shape=(128, 256),
        time_steps=39,
        reynolds_numbers=[300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    )
    
    print("FlowFieldDatasetCreator initialized!")
    print(f"Domain shape: {creator.domain_shape}")
    print(f"Time steps: {creator.time_steps}")
    print(f"Sensor counts: {creator.sensor_counts}")
    print(f"Reynolds numbers: {creator.reynolds_numbers}")
    
    # Option 1: Create all datasets at once
    print("\n" + "="*60)
    print("OPTION 1: Creating all datasets")
    print("="*60)
    
    # Uncomment the following line to create all datasets
    # creator.create_all_datasets()
    
    # Option 2: Create datasets individually
    print("\n" + "="*60)
    print("OPTION 2: Creating individual datasets")
    print("="*60)
    
    # Create datasets for specific configurations
    layouts_to_create = [
        ('random', 32),
        ('circular', 16),
        ('edge', 8)
    ]
    
    for layout_type, n_sensors in layouts_to_create:
        print(f"\nCreating dataset: {layout_type} layout with {n_sensors} sensors")
        
        try:
            # Create dataset
            dataset = creator.create_dataset_for_layout(layout_type, n_sensors)
            
            # Save dataset
            creator.save_dataset(dataset, layout_type, n_sensors)
            
            # Visualize sensor layout
            creator.visualize_sensor_layout(layout_type, n_sensors)
            
            print(f"✓ Successfully created {layout_type} dataset with {n_sensors} sensors")
            
        except Exception as e:
            print(f"✗ Error creating {layout_type} dataset with {n_sensors} sensors: {e}")
    
    # Option 3: Create TensorFlow datasets
    print("\n" + "="*60)
    print("OPTION 3: Creating TensorFlow datasets")
    print("="*60)
    
    # Load a previously created dataset
    try:
        dataset = creator.load_dataset('random', 32)
        
        # Create TensorFlow datasets
        train_dataset, test_dataset = creator.create_tensorflow_dataset(
            dataset, 
            batch_size=16, 
            shuffle=True, 
            test_split=0.2
        )
        
        print("TensorFlow datasets created successfully!")
        print(f"Train dataset: {train_dataset}")
        print(f"Test dataset: {test_dataset}")
        
        # Example: Iterate through a few batches
        print("\nSample batch information:")
        for i, batch in enumerate(train_dataset.take(2)):
            print(f"Batch {i+1}:")
            print(f"  Sensor data shape: {batch['sensor_data'].shape}")
            print(f"  Field data shape: {batch['field_data'].shape}")
            
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please create datasets first using Option 1 or 2.")
    
    # Option 4: Get dataset information
    print("\n" + "="*60)
    print("OPTION 4: Dataset information")
    print("="*60)
    
    info = creator.get_dataset_info()
    print("Dataset Information:")
    print(f"  Available datasets: {info['available_datasets']}")
    print(f"  Sensor counts: {info['sensor_counts']}")
    print(f"  Reynolds numbers: {info['reynolds_numbers']}")
    print(f"  Domain shape: {info['domain_shape']}")
    print(f"  Time steps: {info['time_steps']}")


def example_model_training():
    """
    Example function showing how to use the datasets for model training.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Model Training Setup")
    print("="*60)
    
    # Initialize dataset creator
    creator = FlowFieldDatasetCreator()
    
    try:
        # Load dataset
        dataset = creator.load_dataset('random', 32)
        
        # Create TensorFlow datasets
        train_dataset, test_dataset = creator.create_tensorflow_dataset(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            test_split=0.2
        )
        
        # Example model architecture (placeholder)
        import tensorflow as tf
        
        # Input shape: (batch_size, n_sensors * time_steps)
        input_shape = (32 * 39,)  # 32 sensors * 39 time steps
        
        # Output shape: (batch_size, height * width * time_steps)
        output_shape = (128 * 256 * 39,)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(output_shape[0], activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("Model created successfully!")
        print(f"Model summary:")
        model.summary()
        
        # Example training (commented out to avoid actual training)
        # history = model.fit(
        #     train_dataset.map(lambda x: (x['sensor_data'], x['field_data'])),
        #     epochs=10,
        #     validation_data=test_dataset.map(lambda x: (x['sensor_data'], x['field_data'])),
        #     verbose=1
        # )
        
    except Exception as e:
        print(f"Error in model training example: {e}")


if __name__ == "__main__":
    main()
    
    # Uncomment to run model training example
    # example_model_training()
