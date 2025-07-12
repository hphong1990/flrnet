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
        self.obstacle_radius = 22
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
        Generate random sensor positions using Latin Hypercube Sampling,
        ensuring no sensors are placed inside the circular obstacle and
        only placing sensors downstream (after the obstacle center).

        Args:
            n_sensors: Number of sensors to place

        Returns:
            Array of sensor positions with shape (n_sensors, 2)
        """
        def is_inside_obstacle(positions):
            """Check if positions are inside the obstacle"""
            distances = np.sqrt(
                (positions[:, 0] - self.obstacle_center[0])**2 +
                (positions[:, 1] - self.obstacle_center[1])**2
            )
            return distances <= self.obstacle_radius
        
        def is_downstream(positions):
            """Check if positions are downstream (after obstacle center)"""
            return positions[:, 1] > self.obstacle_center[1]

        max_attempts = 100
        attempt = 0
        valid_positions = []

        while len(valid_positions) < n_sensors and attempt < max_attempts:
            # Generate more positions than needed to account for rejections
            extra_count = int(2.0 * (n_sensors - len(valid_positions))) or 1
            # Generate LHS samples in [0, 1] x [0, 1]
            lhs_samples = self.latin_hypercube_sampling(extra_count, 2)
            # Scale to domain dimensions - only downstream region
            positions = np.zeros((extra_count, 2))
            positions[:, 0] = lhs_samples[:, 0] * (self.domain_shape[0] - 1)  # x-coordinate (height) - full range
            # y-coordinate (width) - only downstream from obstacle center
            downstream_width = self.domain_shape[1] - self.obstacle_center[1] - 1
            positions[:, 1] = self.obstacle_center[1] + lhs_samples[:, 1] * downstream_width
            
            # Filter out positions inside the obstacle and ensure downstream
            valid_mask = ~is_inside_obstacle(positions) & is_downstream(positions)
            valid_positions.extend(positions[valid_mask])
            attempt += 1
            # Trim to exact number needed
            if len(valid_positions) >= n_sensors:
                valid_positions = valid_positions[:n_sensors]
                break

        if len(valid_positions) < n_sensors:
            raise RuntimeError(f"Could not generate {n_sensors} valid sensor positions downstream of obstacle after {max_attempts} attempts")

        sensor_positions = np.array(valid_positions)
        # Final bounds check
        sensor_positions[:, 0] = np.clip(sensor_positions[:, 0], 0, self.domain_shape[0] - 1)
        sensor_positions[:, 1] = np.clip(sensor_positions[:, 1], self.obstacle_center[1] + 1, self.domain_shape[1] - 1)
        return sensor_positions
    
    def generate_circular_sensor_positions(self, n_sensors: int) -> np.ndarray:
        """
        Generate sensor positions around the circular obstacle in the downstream wake region.
        Places sensors at angles -80° to 80° relative to the positive y-direction (downstream).

        Args:
            n_sensors: Number of sensors to place

        Returns:
            Array of sensor positions with shape (n_sensors, 2)
        """
        # Generate uniform angular positions from -80° to 80° relative to y-direction
        # This focuses sensors on the downstream wake region
        # Convert to radians and adjust for coordinate system
        angle_min = -80 * np.pi / 180  # -80 degrees in radians
        angle_max = 80 * np.pi / 180   # 80 degrees in radians
        angles = np.linspace(angle_min, angle_max, n_sensors, endpoint=False)

        # Convert to Cartesian coordinates around obstacle
        # For y-direction reference: 0° = positive y-direction (downstream)
        # x-coordinate uses sin(angle), y-coordinate uses cos(angle) for y-direction reference
        sensor_positions = np.zeros((n_sensors, 2))
        sensor_positions[:, 0] = np.sin(angles) * self.obstacle_radius + self.obstacle_center[0]
        sensor_positions[:, 1] = np.cos(angles) * self.obstacle_radius + self.obstacle_center[1]

        # Ensure positions are within domain bounds
        sensor_positions[:, 0] = np.clip(sensor_positions[:, 0], 0, self.domain_shape[0] - 1)
        sensor_positions[:, 1] = np.clip(sensor_positions[:, 1], 0, self.domain_shape[1] - 1)

        return sensor_positions
    
    def generate_edge_sensor_positions(self, n_sensors: int) -> np.ndarray:
        """
        Generate sensor positions along the horizontal edges (top and bottom),
        but only downstream of the obstacle (y > obstacle_center[1]).

        Args:
            n_sensors: Number of sensors to place

        Returns:
            Array of sensor positions with shape (n_sensors, 2)
        """
        n_per_edge = n_sensors // 2
        remaining = n_sensors % 2

        # Calculate downstream region width
        downstream_width = self.domain_shape[1] - self.obstacle_center[1] - 1

        # Generate positions for top edge - only downstream
        top_positions_lhs = self.latin_hypercube_sampling(n_per_edge, 1)
        top_positions = np.zeros((n_per_edge, 2))
        top_positions[:, 0] = 1  # Top edge (x=1)
        # y coordinates only in downstream region
        top_positions[:, 1] = self.obstacle_center[1] + top_positions_lhs.flatten() * downstream_width

        # Generate positions for bottom edge - only downstream
        bottom_n = n_per_edge + remaining
        bottom_positions_lhs = self.latin_hypercube_sampling(bottom_n, 1)
        bottom_positions = np.zeros((bottom_n, 2))
        bottom_positions[:, 0] = self.domain_shape[0] - 2  # Bottom edge (near bottom but not at edge)
        # y coordinates only in downstream region
        bottom_positions[:, 1] = self.obstacle_center[1] + bottom_positions_lhs.flatten() * downstream_width

        # Combine positions
        sensor_positions = np.vstack([top_positions, bottom_positions])

        # Final bounds check - ensure downstream placement
        sensor_positions[:, 0] = np.clip(sensor_positions[:, 0], 0, self.domain_shape[0] - 1)
        sensor_positions[:, 1] = np.clip(sensor_positions[:, 1], self.obstacle_center[1] + 1, self.domain_shape[1] - 1)

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
            Field data array with shape (time_steps, height, width, channels) or processed format
        """
        filename = f"Re_{reynolds_number}.npy"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Field data file not found: {filepath}")
        
        # Load the raw data
        raw_data = np.load(filepath)
        
        # Handle different data formats
        if len(raw_data.shape) == 4:
            # Data is already in (time_steps, height, width, channels) format
            return raw_data.astype(np.float32)
        elif len(raw_data.shape) == 3:
            # Data is in (time_steps, height, channels) format
            # Need to expand to match the expected domain width
            time_steps, height, channels = raw_data.shape
            target_width = self.domain_shape[1]  # Get target width from domain_shape
            
            # Create expanded data by replicating or interpolating
            expanded_data = np.zeros((time_steps, height, target_width, channels), dtype=np.float32)
            
            # For now, replicate the height dimension to create width
            # This assumes the data represents a 2D slice that needs to be expanded
            for t in range(time_steps):
                for c in range(channels):
                    # Interpolate or replicate to match target width
                    if target_width > height:
                        # Interpolate to expand
                        expanded_data[t, :, :, c] = np.interp(
                            np.linspace(0, height-1, target_width),
                            np.arange(height),
                            raw_data[t, :, c]
                        ).reshape(1, -1).repeat(height, axis=0)
                    else:
                        # Simple replication for smaller width
                        expanded_data[t, :, :target_width, c] = np.tile(
                            raw_data[t, :, c:c+1], (1, target_width)
                        )
            
            return expanded_data
        else:
            raise ValueError(f"Unsupported data format: {raw_data.shape}")
            
        return np.load(filepath)
    
    def extract_velocity_magnitude(self, field_data: np.ndarray) -> np.ndarray:
        """
        Extract velocity magnitude from field data.
        
        Args:
            field_data: Raw field data with shape (time_steps, height, width, channels)
            
        Returns:
            Velocity magnitude data with shape (height, width, time_steps)
        """
        if len(field_data.shape) == 4:
            # Field data is in (time_steps, height, width, channels) format
            time_steps, height, width, channels = field_data.shape
            velocity_magnitude = np.zeros((height, width, time_steps))
            
            for t in range(time_steps):
                if channels >= 2:
                    # Use u and v velocity components to calculate magnitude
                    u_vel = field_data[t, :, :, 0]  # u-velocity (first channel)
                    v_vel = field_data[t, :, :, 1]  # v-velocity (second channel)
                    velocity_magnitude[:, :, t] = np.sqrt(u_vel**2 + v_vel**2)
                else:
                    # If only one channel, use it directly as magnitude
                    velocity_magnitude[:, :, t] = field_data[t, :, :, 0]
        else:
            raise ValueError(f"Unsupported field data shape: {field_data.shape}")
        
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
                                batch_size: int = 16, 
                                shuffle: bool = True,
                                test_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets for training and testing.
        
        The data is reshaped to combine all cases and time steps into the batch dimension:
        - Sensor data: (batch, sensor) where batch = n_cases * n_time_steps
        - Field data: (batch, H, W, 1) where batch = n_cases * n_time_steps
        
        Args:
            dataset: Dataset dictionary with:
                - sensor_data: (n_re, n_sensors, time_steps)
                - field_data: (n_re, height, width, time_steps)
            batch_size: Batch size for TensorFlow dataset
            shuffle: Whether to shuffle the data
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Get data arrays
        sensor_data = dataset['sensor_data']    # (n_re, n_sensors, time_steps)
        field_data = dataset['field_data']      # (n_re, height, width, time_steps)
        
        n_cases, n_sensors, time_steps = sensor_data.shape
        n_cases_f, height, width, time_steps_f = field_data.shape
        
        # Verify shapes are consistent
        assert n_cases == n_cases_f, f"Inconsistent number of cases: {n_cases} vs {n_cases_f}"
        assert time_steps == time_steps_f, f"Inconsistent time steps: {time_steps} vs {time_steps_f}"
        
        # Reshape sensor data: (n_re, n_sensors, time_steps) -> (batch, n_sensors)
        # where batch = n_re * time_steps
        sensor_data_reshaped = sensor_data.transpose(0, 2, 1)  # (n_re, time_steps, n_sensors)
        sensor_data_reshaped = sensor_data_reshaped.reshape(-1, n_sensors)  # (batch, n_sensors)
        
        # Reshape field data: (n_re, height, width, time_steps) -> (batch, height, width, 1)
        # where batch = n_re * time_steps
        field_data_reshaped = field_data.transpose(0, 3, 1, 2)  # (n_re, time_steps, height, width)
        field_data_reshaped = field_data_reshaped.reshape(-1, height, width, 1)  # (batch, height, width, 1)
        
        # Total number of samples (all cases and time steps combined)
        n_samples = sensor_data_reshaped.shape[0]
        
        print(f"Dataset reshaped:")
        print(f"  Original sensor data: {sensor_data.shape}")
        print(f"  Reshaped sensor data: {sensor_data_reshaped.shape}")
        print(f"  Original field data: {field_data.shape}")
        print(f"  Reshaped field data: {field_data_reshaped.shape}")
        print(f"  Total samples: {n_samples}")
        
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
            'sensor_data': sensor_data_reshaped[train_indices].astype(np.float32),
            'field_data': field_data_reshaped[train_indices].astype(np.float32)
        })
        
        test_dataset = tf.data.Dataset.from_tensor_slices({
            'sensor_data': sensor_data_reshaped[test_indices].astype(np.float32),
            'field_data': field_data_reshaped[test_indices].astype(np.float32)
        })
        
        # Apply batching
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=1000)
        
        print(f"TensorFlow datasets created:")
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Test samples: {len(test_indices)}")
        
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
    
    def visualize_sensor_layouts(self, 
                                layouts_dict: Dict[str, np.ndarray] = None,
                                background_data: np.ndarray = None,
                                figsize: Tuple[int, int] = (18, 6),
                                save_plot: bool = False,
                                output_path: str = None) -> None:
        """
        Visualize sensor layouts on flow field background.
        
        This method creates comprehensive visualizations of different sensor layouts,
        showing sensor positions overlaid on flow field data or domain background.
        
        Args:
            layouts_dict: Dictionary of layout_name -> positions arrays. 
                         If None, generates default layouts with 16 sensors each.
            background_data: Flow field data to use as background. If None, uses dummy field.
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            output_path: Path to save the plot (if save_plot=True)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        # Generate default layouts if none provided
        if layouts_dict is None:
            n_sensors = 16
            layouts_dict = {
                'Random': self.generate_random_sensor_positions(n_sensors),
                'Circular': self.generate_circular_sensor_positions(n_sensors),
                'Edge': self.generate_edge_sensor_positions(n_sensors)
            }
        
        # Create figure with subplots
        n_layouts = len(layouts_dict)
        fig, axes = plt.subplots(1, n_layouts, figsize=figsize)
        
        # Ensure axes is always a list
        if n_layouts == 1:
            axes = [axes]
        
        # Define colors for different layouts
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        layout_names = list(layouts_dict.keys())
        
        for i, (layout_name, positions) in enumerate(layouts_dict.items()):
            ax = axes[i]
            color = colors[i % len(colors)]
            
            # Create background
            if background_data is not None:
                # Use provided flow field data
                if len(background_data.shape) == 4:  # (time, height, width, channels)
                    # Use velocity magnitude from first time step
                    u_velocity = background_data[0, :, :, 0]
                    v_velocity = background_data[0, :, :, 1] if background_data.shape[3] > 1 else np.zeros_like(u_velocity)
                    velocity_magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
                elif len(background_data.shape) == 3:  # (time, height, width)
                    velocity_magnitude = background_data[0, :, :]
                else:  # (height, width)
                    velocity_magnitude = background_data
                
                # Plot velocity magnitude as background
                im = ax.imshow(velocity_magnitude, 
                             extent=[0, velocity_magnitude.shape[1], velocity_magnitude.shape[0], 0],
                             cmap='viridis', 
                             alpha=0.8)
            else:
                # Create dummy field for background
                dummy_field = np.zeros(self.domain_shape)
                im = ax.imshow(dummy_field, cmap='gray', alpha=0.3, 
                             extent=[0, self.domain_shape[1], self.domain_shape[0], 0])
            
            # Plot sensor positions
            # Note: positions are in (x, y) format where x is height, y is width
            ax.scatter(positions[:, 1], positions[:, 0], 
                      c=color, s=100, alpha=0.8, 
                      edgecolors='white', linewidth=2, 
                      label=f'{layout_name} ({len(positions)} sensors)')
            
            # Add obstacle circle for reference
            obstacle_circle = Circle((self.obstacle_center[1], self.obstacle_center[0]), 
                                   self.obstacle_radius, color='black', fill=False, 
                                   linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(obstacle_circle)
            
            # Formatting
            ax.set_title(f'{layout_name} Sensor Layout', fontsize=14, fontweight='bold')
            ax.set_xlabel('Y-coordinate (pixels)', fontsize=12)
            ax.set_ylabel('X-coordinate (pixels)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add colorbar to the last subplot if using flow field background
            if background_data is not None and i == n_layouts - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Velocity Magnitude', fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            if output_path is None:
                output_path = self.output_path / "sensor_layouts_visualization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Sensor layout visualization saved to: {output_path}")
        
        plt.show()
        
        print("✅ Sensor layout visualizations completed!")
    
    def visualize_sensor_positions_on_field(self,
                                           layout_types: List[str] = None,
                                           n_sensors: int = 16,
                                           background_data: np.ndarray = None,
                                           load_from_saved: bool = True,
                                           figsize: Tuple[int, int] = (18, 5)) -> None:
        """
        Visualize sensor positions on flow field with saved layouts.
        
        This method loads sensor positions from saved numpy files and visualizes them
        on flow field background data.
        
        Args:
            layout_types: List of layout types to visualize. If None, uses ['random', 'circular', 'edge']
            n_sensors: Number of sensors for visualization
            background_data: Flow field data to use as background
            load_from_saved: Whether to load from saved layout files
            figsize: Figure size for the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from pathlib import Path
        
        print(f"6️⃣ Sensor Positions on Flow Field")
        print("-" * 55)
        
        if layout_types is None:
            layout_types = ['random', 'circular', 'edge']
        
        # Load sensor positions
        sensor_layouts = {}
        
        if load_from_saved:
            # Load from saved numpy files
            sensor_layouts_dir = self.sensor_layouts_path
            
            for layout_name in layout_types:
                layout_file = sensor_layouts_dir / f"sensor_layout_{layout_name}_{n_sensors}.npy"
                if layout_file.exists():
                    sensor_layouts[layout_name] = np.load(layout_file)
                    print(f"✅ Loaded {layout_name} layout: {sensor_layouts[layout_name].shape}")
                else:
                    print(f"❌ File not found: {layout_file}")
                    # Fallback to generating new positions
                    if layout_name == 'random':
                        sensor_layouts[layout_name] = self.generate_random_sensor_positions(n_sensors)
                    elif layout_name == 'circular':
                        sensor_layouts[layout_name] = self.generate_circular_sensor_positions(n_sensors)
                    elif layout_name == 'edge':
                        sensor_layouts[layout_name] = self.generate_edge_sensor_positions(n_sensors)
        else:
            # Generate new positions
            for layout_name in layout_types:
                if layout_name == 'random':
                    sensor_layouts[layout_name] = self.generate_random_sensor_positions(n_sensors)
                elif layout_name == 'circular':
                    sensor_layouts[layout_name] = self.generate_circular_sensor_positions(n_sensors)
                elif layout_name == 'edge':
                    sensor_layouts[layout_name] = self.generate_edge_sensor_positions(n_sensors)
        
        # Create visualization
        fig, axes = plt.subplots(1, len(layout_types), figsize=figsize)
        if len(layout_types) == 1:
            axes = [axes]
        
        titles = [f'{layout.title()} Layout' for layout in layout_types]
        
        # Prepare background data
        if background_data is not None:
            if len(background_data.shape) == 4:  # (time, height, width, channels)
                sample_data = background_data[0, :, :, 0]  # First time step, first channel
                if background_data.shape[3] > 1:
                    velocity_magnitude = np.sqrt(sample_data**2 + background_data[0, :, :, 1]**2)
                else:
                    velocity_magnitude = sample_data
            else:
                velocity_magnitude = background_data
        else:
            # Create dummy background
            velocity_magnitude = np.zeros(self.domain_shape)
        
        for idx, (layout_name, title) in enumerate(zip(layout_types, titles)):
            ax = axes[idx]
            
            # Plot velocity magnitude as background
            im = ax.imshow(velocity_magnitude, 
                          extent=[0, velocity_magnitude.shape[1], velocity_magnitude.shape[0], 0],
                          cmap='viridis', 
                          alpha=0.8)
            
            # Get sensor positions for this layout
            if layout_name in sensor_layouts:
                positions = sensor_layouts[layout_name]
                
                # Plot sensor positions (note: positions are in (x, y) format where x is height, y is width)
                ax.scatter(positions[:, 1], positions[:, 0], 
                          c='red', s=100, marker='o', 
                          edgecolors='white', linewidth=2, 
                          label=f'{n_sensors} sensors', alpha=0.9)
            
            # Add obstacle representation (white dashed circle)
            obstacle_circle = Circle((self.obstacle_center[1], self.obstacle_center[0]), 
                                   self.obstacle_radius, fill=False, 
                                   color='white', linestyle='--', linewidth=2)
            ax.add_patch(obstacle_circle)
            
            # Formatting
            ax.set_title(f'{title}\n({n_sensors} sensors - {"saved" if load_from_saved else "generated"} layout)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Y-coordinate (pixels)', fontsize=12)
            ax.set_ylabel('X-coordinate (pixels)', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to the last subplot
            if idx == len(layout_types) - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Velocity Magnitude', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # Print sensor statistics
        self._print_sensor_layout_statistics(layout_types, load_from_saved)
        
        print("\n✅ Sensor position visualization from saved files complete!")
    
    def _print_sensor_layout_statistics(self, layout_types: List[str], load_from_saved: bool = True) -> None:
        """Print detailed statistics for sensor layouts."""
        print("\n📋 Sensor Layout Analysis")
        print("=" * 80)
        print(f"{'Layout':<10} {'Sensors':<8} {'Mean_X':<8} {'Mean_Y':<8} {'Std_X':<8} {'Std_Y':<8} {'Coverage':<10}")
        print("-" * 80)
        
        # Analyze all available sensor layouts
        for layout in layout_types:
            for n_sens in [8, 16, 32]:
                if load_from_saved:
                    layout_file = self.sensor_layouts_path / f"sensor_layout_{layout}_{n_sens}.npy"
                    if layout_file.exists():
                        positions = np.load(layout_file)
                    else:
                        print(f"{layout:<10} {n_sens:<8} {'FILE NOT FOUND':<40}")
                        continue
                else:
                    # Generate positions dynamically
                    if layout == 'random':
                        positions = self.generate_random_sensor_positions(n_sens)
                    elif layout == 'circular':
                        positions = self.generate_circular_sensor_positions(n_sens)
                    elif layout == 'edge':
                        positions = self.generate_edge_sensor_positions(n_sens)
                    else:
                        continue
                
                # Calculate statistics
                mean_x = np.mean(positions[:, 0])
                mean_y = np.mean(positions[:, 1])
                std_x = np.std(positions[:, 0])
                std_y = np.std(positions[:, 1])
                
                # Calculate coverage as percentage of domain covered
                x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
                y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
                coverage = (x_range * y_range) / (self.domain_shape[0] * self.domain_shape[1]) * 100
                
                print(f"{layout:<10} {n_sens:<8} {mean_x:<8.1f} {mean_y:<8.1f} {std_x:<8.1f} {std_y:<8.1f} {coverage:<10.1f}%")
        
        if load_from_saved:
            print("\n📁 Available Sensor Layout Files:")
            sensor_files = list(self.sensor_layouts_path.glob("sensor_layout_*.npy"))
            for file in sorted(sensor_files):
                file_size = file.stat().st_size
                print(f"   • {file.name} ({file_size} bytes)")
        
        print("\n🔍 Key Insights from Sensor Layouts:")
        print("   • Layouts are persistent and reproducible")
        print("   • Random layouts use Latin Hypercube Sampling for optimal coverage")
        print("   • Circular layouts focus on obstacle wake region") 
        print("   • Edge layouts capture boundary layer effects")
        if load_from_saved:
            print("   • Saved layouts ensure consistency across experiments")
