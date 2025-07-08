"""
Flow Field Dataset Creation Class for TensorFlow

This module provides a comprehensive dataset creation class for flow field reconstruction
using different sensor placement strategies: random, edge, and circular arrangements.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict, Optional
import random
from pathlib import Path


class FlowFieldDatasetCreator:
    """
    A comprehensive dataset creation class for flow field reconstruction using TensorFlow.
    
    Supports three sensor placement strategies:
    1. Random placement using Latin Hypercube Sampling
    2. Edge placement (sensors along horizontal boundaries)
    3. Circular placement (sensors around obstacle)
    
    Supports sensor counts: 8, 16, 32, 64
    """
    
    def __init__(self, 
                 data_path: str = "D:/data/Navier-Stokes/Navier-Stokes/",
                 output_path: str = "./dataset_creation/",
                 domain_shape: Tuple[int, int] = (128, 256),
                 time_steps: int = 39,
                 reynolds_numbers: List[int] = None,
                 test_data_file: str = None,
                 use_synthetic_data: bool = False):
        """
        Initialize the FlowFieldDatasetCreator.
        
        Args:
            data_path: Path to the raw Navier-Stokes data files
            output_path: Path to save generated datasets and sensor layouts
            domain_shape: Shape of the flow domain (height, width)
            time_steps: Number of time steps in the data
            reynolds_numbers: List of Reynolds numbers to process
            test_data_file: Path to a test data file to use instead of data_path
            use_synthetic_data: If True, use synthetic data for testing/development
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.domain_shape = domain_shape
        self.time_steps = time_steps
        self.reynolds_numbers = reynolds_numbers or [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        self.sensor_counts = [8, 16, 32, 64]
        
        # Setup for obstacle (used in circular sensor placement and synthetic data)
        self.obstacle_center = [64, 64]  # Default obstacle center
        self.obstacle_radius = 8        # Default obstacle radius
        
        # Test data handling
        self.test_data_file = test_data_file
        self.test_data = None
        self.use_synthetic_data = use_synthetic_data
        
        # Load test data if specified
        if test_data_file and Path(test_data_file).exists():
            try:
                self.test_data = np.load(test_data_file)
                print(f"✅ Test data loaded from {test_data_file}")
                print(f"📊 Test data shape: {self.test_data.shape}")
            except Exception as e:
                print(f"❌ Error loading test data: {str(e)}")
        
        # Create synthetic data if requested
        if use_synthetic_data:
            self.create_synthetic_data()
        
        # Create output directories
        self.sensor_layouts_path = self.output_path / "sensor_layouts"
        self.datasets_path = self.output_path / "datasets"
        self.sensor_layouts_path.mkdir(parents=True, exist_ok=True)
        self.datasets_path.mkdir(parents=True, exist_ok=True)
        
        # Flow domain parameters
        self.obstacle_center = (64, 66)  # Center of circular obstacle
        self.obstacle_radius = 21
        self.flow_region_y_start = 80  # Start of flow region in y-direction
        
    def latin_hypercube_sampling(self, n: int, k: int) -> np.ndarray:
        """
        Generate Latin Hypercube Sampling points.
        
        Args:
            n: Number of samples (rows)
            k: Number of variables (columns)
            
        Returns:
            numpy array of shape (n, k) containing the LHS points
        """
        samples = np.zeros((n, k))
        for j in range(k):
            # Divide the interval [0, 1] into n equal intervals
            seg_size = 1.0 / n
            seg_starts = np.arange(0, 1, seg_size)
            
            # Randomly shuffle the starting points
            random.shuffle(seg_starts)
            
            # Fill each row with one of the random segments
            for i in range(n):
                samples[i, j] = random.uniform(seg_starts[i], seg_starts[i] + seg_size)
        
        return samples
    
    def generate_random_sensor_positions(self, n_sensors: int) -> np.ndarray:
        """
        Generate random sensor positions using Latin Hypercube Sampling.
        
        Args:
            n_sensors: Number of sensors to place
            
        Returns:
            Array of sensor positions with shape (n_sensors, 2)
        """
        # Generate LHS samples in [0, 1] x [0, 1]
        lhs_samples = self.latin_hypercube_sampling(n_sensors, 2)
        
        # Scale to domain dimensions
        sensor_positions = np.zeros((n_sensors, 2))
        sensor_positions[:, 0] = lhs_samples[:, 0] * self.domain_shape[0]  # x-coordinate
        sensor_positions[:, 1] = (lhs_samples[:, 1] * self.domain_shape[1]) + self.flow_region_y_start  # y-coordinate
        
        # Ensure positions are within bounds
        sensor_positions[:, 1] = np.clip(sensor_positions[:, 1], 
                                       self.flow_region_y_start, 
                                       self.domain_shape[1] - 1)
        
        return sensor_positions
    
    def generate_circular_sensor_positions(self, n_sensors: int) -> np.ndarray:
        """
        Generate sensor positions around the circular obstacle.
        
        Args:
            n_sensors: Number of sensors to place
            
        Returns:
            Array of sensor positions with shape (n_sensors, 2)
        """
        # Generate angular positions using LHS
        angles_lhs = self.latin_hypercube_sampling(n_sensors, 1)
        angles = angles_lhs.flatten() * 360  # Convert to degrees
        
        # Convert to Cartesian coordinates around obstacle
        sensor_positions = np.zeros((n_sensors, 2))
        sensor_positions[:, 0] = np.cos(np.deg2rad(angles)) * self.obstacle_radius + self.obstacle_center[0]
        sensor_positions[:, 1] = np.sin(np.deg2rad(angles)) * self.obstacle_radius + self.obstacle_center[1]
        
        return sensor_positions
    
    def generate_edge_sensor_positions(self, n_sensors: int) -> np.ndarray:
        """
        Generate sensor positions along the horizontal edges (top and bottom).
        
        Args:
            n_sensors: Number of sensors to place
            
        Returns:
            Array of sensor positions with shape (n_sensors, 2)
        """
        n_per_edge = n_sensors // 2
        remaining = n_sensors % 2
        
        # Generate positions for top edge
        top_positions_lhs = self.latin_hypercube_sampling(n_per_edge, 1)
        top_positions = np.zeros((n_per_edge, 2))
        top_positions[:, 0] = 1  # Top edge
        top_positions[:, 1] = top_positions_lhs.flatten() * self.domain_shape[1] + self.flow_region_y_start
        
        # Generate positions for bottom edge
        bottom_n = n_per_edge + remaining
        bottom_positions_lhs = self.latin_hypercube_sampling(bottom_n, 1)
        bottom_positions = np.zeros((bottom_n, 2))
        bottom_positions[:, 0] = self.domain_shape[0] - 1  # Bottom edge
        bottom_positions[:, 1] = bottom_positions_lhs.flatten() * self.domain_shape[1] + self.flow_region_y_start
        
        # Combine positions
        sensor_positions = np.vstack([top_positions, bottom_positions])
        
        return sensor_positions
    
    def save_sensor_layout(self, positions: np.ndarray, layout_type: str, n_sensors: int):
        """
        Save sensor layout to file.
        
        Args:
            positions: Sensor positions array
            layout_type: Type of layout ('random', 'circular', 'edge')
            n_sensors: Number of sensors
        """
        filename = f"sensor_layout_{layout_type}_{n_sensors}.npy"
        filepath = self.sensor_layouts_path / filename
        np.save(filepath, positions)
        print(f"Saved sensor layout: {filepath}")
    
    def load_field_data(self, reynolds_number: int) -> np.ndarray:
        """
        Load field data for a given Reynolds number.
        
        Args:
            reynolds_number: Reynolds number
            
        Returns:
            Field data array with shape (height, width, time_steps * 3)
        """
        filename = f"Re_{reynolds_number}.npy"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Field data file not found: {filepath}")
            
        return np.load(filepath)
    
    def extract_velocity_magnitude(self, field_data: np.ndarray) -> np.ndarray:
        """
        Extract velocity magnitude from field data.
        
        Args:
            field_data: Raw field data with shape (height, width, time_steps * 3)
            
        Returns:
            Velocity magnitude data with shape (height, width, time_steps)
        """
        velocity_magnitude = np.zeros((self.domain_shape[0], self.domain_shape[1], self.time_steps))
        
        for t in range(self.time_steps):
            u_vel = field_data[:, :, t * 3]      # u-velocity
            v_vel = field_data[:, :, t * 3 + 1]  # v-velocity
            velocity_magnitude[:, :, t] = np.sqrt(u_vel**2 + v_vel**2)
        
        return velocity_magnitude
    
    def extract_sensor_measurements(self, field_data: np.ndarray, sensor_positions: np.ndarray) -> np.ndarray:
        """
        Extract sensor measurements from field data.
        
        Args:
            field_data: Velocity magnitude data with shape (height, width, time_steps)
            sensor_positions: Sensor positions with shape (n_sensors, 2)
            
        Returns:
            Sensor measurements with shape (n_sensors, time_steps)
        """
        n_sensors = sensor_positions.shape[0]
        sensor_measurements = np.zeros((n_sensors, self.time_steps))
        
        for i, (x, y) in enumerate(sensor_positions):
            x_idx, y_idx = int(x), int(y)
            # Ensure indices are within bounds
            x_idx = np.clip(x_idx, 0, self.domain_shape[0] - 1)
            y_idx = np.clip(y_idx, 0, self.domain_shape[1] - 1)
            
            sensor_measurements[i, :] = field_data[x_idx, y_idx, :]
        
        return sensor_measurements
    
    def create_dataset_for_layout(self, layout_type: str, n_sensors: int) -> Dict[str, np.ndarray]:
        """
        Create dataset for a specific sensor layout configuration.
        
        Args:
            layout_type: Type of sensor layout ('random', 'circular', 'edge')
            n_sensors: Number of sensors
            
        Returns:
            Dictionary containing sensor data and full field data
        """
        # Generate sensor positions
        if layout_type == 'random':
            sensor_positions = self.generate_random_sensor_positions(n_sensors)
        elif layout_type == 'circular':
            sensor_positions = self.generate_circular_sensor_positions(n_sensors)
        elif layout_type == 'edge':
            sensor_positions = self.generate_edge_sensor_positions(n_sensors)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
        
        # Save sensor layout
        self.save_sensor_layout(sensor_positions, layout_type, n_sensors)
        
        # Initialize data containers
        all_sensor_data = []
        all_field_data = []
        reynolds_labels = []
        
        # Process each Reynolds number
        for re_num in self.reynolds_numbers:
            print(f"Processing Re = {re_num} for {layout_type} layout with {n_sensors} sensors")
            
            # Load and process field data
            raw_field_data = self.load_field_data(re_num)
            velocity_magnitude = self.extract_velocity_magnitude(raw_field_data)
            
            # Extract sensor measurements
            sensor_measurements = self.extract_sensor_measurements(velocity_magnitude, sensor_positions)
            
            # Store data
            all_sensor_data.append(sensor_measurements)
            all_field_data.append(velocity_magnitude)
            reynolds_labels.append(re_num)
        
        # Convert to numpy arrays
        dataset = {
            'sensor_data': np.array(all_sensor_data),      # (n_re, n_sensors, time_steps)
            'field_data': np.array(all_field_data),        # (n_re, height, width, time_steps)
            'sensor_positions': sensor_positions,           # (n_sensors, 2)
            'reynolds_numbers': np.array(reynolds_labels),  # (n_re,)
            'layout_type': layout_type,
            'n_sensors': n_sensors
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, np.ndarray], layout_type: str, n_sensors: int):
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset dictionary
            layout_type: Type of sensor layout
            n_sensors: Number of sensors
        """
        filename = f"dataset_{layout_type}_{n_sensors}.npz"
        filepath = self.datasets_path / filename
        
        np.savez_compressed(filepath, **dataset)
        print(f"Saved dataset: {filepath}")
        print(f"Dataset shapes:")
        print(f"  Sensor data: {dataset['sensor_data'].shape}")
        print(f"  Field data: {dataset['field_data'].shape}")
        print(f"  Sensor positions: {dataset['sensor_positions'].shape}")
        print(f"  Reynolds numbers: {dataset['reynolds_numbers'].shape}")
    
    def create_tensorflow_dataset(self, dataset: Dict[str, np.ndarray], 
                                batch_size: int = 32, 
                                shuffle: bool = True,
                                test_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets for training and testing.
        
        Args:
            dataset: Dataset dictionary
            batch_size: Batch size for TensorFlow dataset
            shuffle: Whether to shuffle the data
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Prepare input-output pairs
        sensor_data = dataset['sensor_data']    # (n_re, n_sensors, time_steps)
        field_data = dataset['field_data']      # (n_re, height, width, time_steps)
        
        # Flatten field data for output
        n_samples = sensor_data.shape[0]
        field_data_flat = field_data.reshape(n_samples, -1)  # (n_re, height*width*time_steps)
        sensor_data_flat = sensor_data.reshape(n_samples, -1)  # (n_re, n_sensors*time_steps)
        
        # Split data
        n_train = int(n_samples * (1 - test_split))
        
        if shuffle:
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices({
            'sensor_data': sensor_data_flat[train_indices],
            'field_data': field_data_flat[train_indices]
        })
        
        test_dataset = tf.data.Dataset.from_tensor_slices({
            'sensor_data': sensor_data_flat[test_indices],
            'field_data': field_data_flat[test_indices]
        })
        
        # Apply batching
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=1000)
        
        return train_dataset, test_dataset
    
    def visualize_sensor_layout(self, layout_type: str, n_sensors: int, reynolds_number: int = 750):
        """
        Visualize sensor layout on a flow field.
        
        Args:
            layout_type: Type of sensor layout
            n_sensors: Number of sensors
            reynolds_number: Reynolds number for background flow field
        """
        # Load sensor positions
        filename = f"sensor_layout_{layout_type}_{n_sensors}.npy"
        filepath = self.sensor_layouts_path / filename
        
        if not filepath.exists():
            print(f"Sensor layout file not found: {filepath}")
            return
        
        sensor_positions = np.load(filepath)
        
        # Load and process field data for visualization
        raw_field_data = self.load_field_data(reynolds_number)
        velocity_magnitude = self.extract_velocity_magnitude(raw_field_data)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot velocity magnitude at first time step
        plt.imshow(velocity_magnitude[:, :, 0], cmap='viridis', vmin=0, vmax=2)
        plt.colorbar(label='Velocity Magnitude')
        
        # Plot sensor positions
        plt.scatter(sensor_positions[:, 1], sensor_positions[:, 0], 
                   c='red', s=50, marker='o', edgecolors='white', linewidth=1)
        
        plt.title(f'Sensor Layout: {layout_type.capitalize()} ({n_sensors} sensors)')
        plt.xlabel('Y-coordinate')
        plt.ylabel('X-coordinate')
        plt.tight_layout()
        
        # Save figure
        fig_filename = f"sensor_layout_{layout_type}_{n_sensors}.png"
        fig_filepath = self.sensor_layouts_path / fig_filename
        plt.savefig(fig_filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved: {fig_filepath}")
    
    def create_all_datasets(self):
        """
        Create all datasets for all sensor layouts and sensor counts.
        """
        layout_types = ['random', 'circular', 'edge']
        
        print("Creating all datasets...")
        print(f"Layout types: {layout_types}")
        print(f"Sensor counts: {self.sensor_counts}")
        print(f"Reynolds numbers: {self.reynolds_numbers}")
        print("-" * 50)
        
        for layout_type in layout_types:
            for n_sensors in self.sensor_counts:
                print(f"\n--- Creating dataset: {layout_type} layout with {n_sensors} sensors ---")
                
                try:
                    # Create dataset
                    dataset = self.create_dataset_for_layout(layout_type, n_sensors)
                    
                    # Save dataset
                    self.save_dataset(dataset, layout_type, n_sensors)
                    
                    # Create visualization
                    self.visualize_sensor_layout(layout_type, n_sensors)
                    
                    print(f"✓ Successfully created dataset for {layout_type} layout with {n_sensors} sensors")
                    
                except Exception as e:
                    print(f"✗ Error creating dataset for {layout_type} layout with {n_sensors} sensors: {e}")
                    
        print("\n" + "=" * 50)
        print("Dataset creation completed!")
    
    def load_dataset(self, layout_type: str, n_sensors: int) -> Dict[str, np.ndarray]:
        """
        Load a previously created dataset.
        
        Args:
            layout_type: Type of sensor layout
            n_sensors: Number of sensors
            
        Returns:
            Loaded dataset dictionary
        """
        filename = f"dataset_{layout_type}_{n_sensors}.npz"
        filepath = self.datasets_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        data = np.load(filepath)
        dataset = {key: data[key] for key in data.files}
        
        print(f"Loaded dataset: {filepath}")
        print(f"Dataset shapes:")
        print(f"  Sensor data: {dataset['sensor_data'].shape}")
        print(f"  Field data: {dataset['field_data'].shape}")
        
        return dataset
    
    def get_dataset_info(self) -> Dict[str, List[str]]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        available_datasets = []
        
        for layout_type in ['random', 'circular', 'edge']:
            for n_sensors in self.sensor_counts:
                filename = f"dataset_{layout_type}_{n_sensors}.npz"
                filepath = self.datasets_path / filename
                
                if filepath.exists():
                    available_datasets.append(f"{layout_type}_{n_sensors}")
        
        return {
            'available_datasets': available_datasets,
            'sensor_counts': self.sensor_counts,
            'reynolds_numbers': self.reynolds_numbers,
            'domain_shape': self.domain_shape,
            'time_steps': self.time_steps
        }
    
    def create_synthetic_data(self):
        """
        Create synthetic data for testing and development purposes.
        
        This method generates a synthetic flow field dataset with a circular obstacle
        and wake pattern similar to real Navier-Stokes data, but for testing only.
        
        Returns:
            None (sets self.test_data)
        """
        print("🔧 Creating synthetic test data...")
        
        # Create synthetic data similar to what we might expect
        test_data_shape = (self.time_steps, *self.domain_shape)
        synthetic_data = np.random.randn(*test_data_shape) * 0.1
        
        # Add circular obstacle pattern to make it look like flow data
        center_y, center_x = self.obstacle_center
        radius = self.obstacle_radius
        
        y, x = np.ogrid[:self.domain_shape[0], :self.domain_shape[1]]
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Create a flow-like pattern around the obstacle
        for t in range(synthetic_data.shape[0]):
            # Set zero inside obstacle
            mask = dist_from_center <= radius
            synthetic_data[t][mask] = 0
            
            # Add wake pattern behind obstacle
            wake_mask = (x > center_x) & (np.abs(y - center_y) < radius * 1.5)
            pattern = np.sin(x[wake_mask] * 0.1 + t * 0.2) * np.exp(-(x[wake_mask] - center_x) * 0.01)
            synthetic_data[t][wake_mask] = pattern * 0.5
        
        self.test_data = synthetic_data
        print(f"✅ Created synthetic test data with shape {synthetic_data.shape}")
        
        # Save the synthetic data for future use
        try:
            synthetic_data_path = self.output_path / "synthetic_flow_data.npy"
            np.save(synthetic_data_path, synthetic_data)
            print(f"✅ Saved synthetic test data to {synthetic_data_path}")
        except Exception as e:
            print(f"❌ Error saving synthetic data: {str(e)}")
    
    def load_flow_field_data(self, reynolds_number=None):
        """
        Load flow field data either from the specified data directory or test data.
        
        This method handles multiple data sources in this priority order:
        1. Test data file if specified
        2. Synthetic data if generated
        3. External data directory with actual simulation data
        
        Args:
            reynolds_number: Reynolds number to load (used only for external data)
            
        Returns:
            numpy array containing the flow field data
        """
        # Case 1: Use test data if available
        if self.test_data is not None:
            print(f"📊 Using pre-loaded test data with shape {self.test_data.shape}")
            return self.test_data
        
        # Case 2: Use external data directory
        if self.data_path and Path(self.data_path).exists():
            try:
                # Try to load specific file for reynolds number if provided
                if reynolds_number:
                    file_path = Path(self.data_path) / f"flow_field_re_{reynolds_number}.npy"
                    if file_path.exists():
                        data = np.load(str(file_path))
                        print(f"✅ Loaded data for Re={reynolds_number} from {file_path}")
                        print(f"📊 Data shape: {data.shape}")
                        return data
                
                # If no specific reynolds number or file not found, find any .npy file
                npy_files = list(Path(self.data_path).glob("*.npy"))
                if npy_files:
                    file_path = npy_files[0]
                    data = np.load(str(file_path))
                    print(f"✅ Loaded data from {file_path}")
                    print(f"📊 Data shape: {data.shape}")
                    return data
            except Exception as e:
                print(f"❌ Error loading external data: {str(e)}")
        
        # Case 3: Create synthetic data as fallback
        print("⚠️ No data available. Creating synthetic data as fallback.")
        self.create_synthetic_data()
        return self.test_data
