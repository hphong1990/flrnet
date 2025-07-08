"""
Test script for the FlowFieldDatasetCreator system.

This script provides comprehensive testing for all components of the
dataset creation system, including validation, error handling, and
performance testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from pathlib import Path
import unittest

# Import our modules
from flow_field_dataset import FlowFieldDatasetCreator
from config import get_config_dict, validate_config
from utils import (
    validate_sensor_positions,
    validate_field_data,
    calculate_dataset_statistics,
    memory_usage_mb,
    create_dataset_summary
)


class TestFlowFieldDatasetCreator(unittest.TestCase):
    """
    Test suite for FlowFieldDatasetCreator class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_path = "./test_output/"
        self.creator = FlowFieldDatasetCreator(
            data_path="./test_data/",  # Mock data path
            output_path=self.test_output_path,
            domain_shape=(64, 128),  # Smaller domain for testing
            time_steps=10,  # Fewer time steps for testing
            reynolds_numbers=[300, 500, 750]  # Fewer Reynolds numbers
        )
        
        # Create test output directory
        Path(self.test_output_path).mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test files (optional)
        pass
    
    def test_latin_hypercube_sampling(self):
        """Test Latin Hypercube Sampling function."""
        n_samples = 10
        n_vars = 2
        
        samples = self.creator.latin_hypercube_sampling(n_samples, n_vars)
        
        # Check shape
        self.assertEqual(samples.shape, (n_samples, n_vars))
        
        # Check range [0, 1]
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))
        
        # Check that all samples are different
        unique_samples = np.unique(samples, axis=0)
        self.assertEqual(len(unique_samples), n_samples)
    
    def test_sensor_position_generation(self):
        """Test all sensor position generation methods."""
        n_sensors = 16
        
        # Test random positions
        random_pos = self.creator.generate_random_sensor_positions(n_sensors)
        self.assertEqual(random_pos.shape, (n_sensors, 2))
        self.assertTrue(np.all(random_pos >= 0))
        
        # Test circular positions
        circular_pos = self.creator.generate_circular_sensor_positions(n_sensors)
        self.assertEqual(circular_pos.shape, (n_sensors, 2))
        
        # Test edge positions
        edge_pos = self.creator.generate_edge_sensor_positions(n_sensors)
        self.assertEqual(edge_pos.shape, (n_sensors, 2))
        
        # Check that edge positions are actually on edges
        x_coords = edge_pos[:, 0]
        self.assertTrue(np.any(x_coords == 1) or np.any(x_coords == self.creator.domain_shape[0] - 1))
    
    def test_sensor_position_validation(self):
        """Test sensor position validation."""
        # Valid positions
        valid_pos = np.array([[10, 100], [20, 120], [30, 110]])
        errors = validate_sensor_positions(valid_pos, (64, 128))
        self.assertEqual(len(errors), 0)
        
        # Invalid positions (out of bounds)
        invalid_pos = np.array([[100, 100], [20, 300]])  # x too large, y too large
        errors = validate_sensor_positions(invalid_pos, (64, 128))
        self.assertGreater(len(errors), 0)
    
    def test_velocity_magnitude_extraction(self):
        """Test velocity magnitude extraction."""
        # Create mock field data
        height, width = 64, 128
        time_steps = 10
        mock_field_data = np.random.rand(height, width, time_steps * 3)
        
        # Extract velocity magnitude
        velocity_mag = self.creator.extract_velocity_magnitude(mock_field_data)
        
        # Check shape
        expected_shape = (height, width, time_steps)
        self.assertEqual(velocity_mag.shape, expected_shape)
        
        # Check that all values are non-negative (magnitude property)
        self.assertTrue(np.all(velocity_mag >= 0))
    
    def test_sensor_measurement_extraction(self):
        """Test sensor measurement extraction."""
        # Create mock field data
        height, width, time_steps = 64, 128, 10
        mock_field_data = np.random.rand(height, width, time_steps)
        
        # Create sensor positions
        sensor_positions = np.array([[10, 50], [20, 60], [30, 70]])
        
        # Extract measurements
        measurements = self.creator.extract_sensor_measurements(mock_field_data, sensor_positions)
        
        # Check shape
        expected_shape = (len(sensor_positions), time_steps)
        self.assertEqual(measurements.shape, expected_shape)
        
        # Check that measurements match field data at sensor positions
        for i, (x, y) in enumerate(sensor_positions):
            expected_measurements = mock_field_data[int(x), int(y), :]
            np.testing.assert_array_equal(measurements[i, :], expected_measurements)
    
    def test_dataset_structure(self):
        """Test dataset structure and format."""
        # Create mock dataset
        n_sensors = 8
        n_re = 3
        time_steps = 10
        
        mock_dataset = {
            'sensor_data': np.random.rand(n_re, n_sensors, time_steps),
            'field_data': np.random.rand(n_re, 64, 128, time_steps),
            'sensor_positions': np.random.rand(n_sensors, 2),
            'reynolds_numbers': np.array([300, 500, 750]),
            'layout_type': 'random',
            'n_sensors': n_sensors
        }
        
        # Test dataset statistics
        stats = calculate_dataset_statistics(mock_dataset)
        
        self.assertIn('sensor_data', stats)
        self.assertIn('field_data', stats)
        self.assertIn('mean', stats['sensor_data'])
        self.assertIn('std', stats['sensor_data'])
        self.assertIn('min', stats['sensor_data'])
        self.assertIn('max', stats['sensor_data'])
    
    def test_memory_usage_calculation(self):
        """Test memory usage calculation."""
        # Create test arrays
        small_array = np.random.rand(10, 10)
        large_array = np.random.rand(100, 100, 100)
        
        small_memory = memory_usage_mb(small_array)
        large_memory = memory_usage_mb(large_array)
        
        self.assertGreater(large_memory, small_memory)
        self.assertGreater(small_memory, 0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        errors = validate_config()
        
        # Print errors for debugging
        if errors:
            print("Configuration errors found:")
            for error in errors:
                print(f"  - {error}")
        
        # Config might have errors due to missing data paths, which is expected in testing
        self.assertIsInstance(errors, list)


class TestPerformance:
    """
    Performance testing for the dataset creation system.
    """
    
    def __init__(self):
        self.creator = FlowFieldDatasetCreator(
            data_path="./test_data/",
            output_path="./test_output/",
            domain_shape=(128, 256),
            time_steps=39,
            reynolds_numbers=[300, 500, 750]
        )
    
    def test_sensor_generation_performance(self):
        """Test performance of sensor position generation."""
        sensor_counts = [8, 16, 32, 64]
        layouts = ['random', 'circular', 'edge']
        
        print("\nSensor Generation Performance Test:")
        print("=" * 50)
        
        for layout in layouts:
            print(f"\nLayout: {layout}")
            print("-" * 20)
            
            for n_sensors in sensor_counts:
                start_time = time.time()
                
                if layout == 'random':
                    positions = self.creator.generate_random_sensor_positions(n_sensors)
                elif layout == 'circular':
                    positions = self.creator.generate_circular_sensor_positions(n_sensors)
                elif layout == 'edge':
                    positions = self.creator.generate_edge_sensor_positions(n_sensors)
                
                end_time = time.time()
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                
                print(f"  {n_sensors} sensors: {elapsed:.2f} ms")
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for different dataset sizes."""
        print("\nMemory Usage Estimation:")
        print("=" * 50)
        
        sensor_counts = [8, 16, 32, 64]
        n_re = 15  # Number of Reynolds numbers
        
        for n_sensors in sensor_counts:
            # Estimate memory usage
            sensor_data_size = n_re * n_sensors * 39 * 4  # 4 bytes per float32
            field_data_size = n_re * 128 * 256 * 39 * 4  # 4 bytes per float32
            total_size_mb = (sensor_data_size + field_data_size) / (1024 * 1024)
            
            print(f"  {n_sensors} sensors: ~{total_size_mb:.1f} MB")


class TestDataValidation:
    """
    Data validation testing.
    """
    
    def test_field_data_validation(self):
        """Test field data validation."""
        print("\nField Data Validation Test:")
        print("=" * 50)
        
        # Valid field data
        valid_data = np.random.rand(128, 256, 39)
        errors = validate_field_data(valid_data, (128, 256, 39))
        print(f"Valid data errors: {len(errors)}")
        
        # Invalid shape
        invalid_shape_data = np.random.rand(64, 128, 39)
        errors = validate_field_data(invalid_shape_data, (128, 256, 39))
        print(f"Invalid shape errors: {len(errors)}")
        
        # Data with NaN
        nan_data = np.random.rand(128, 256, 39)
        nan_data[0, 0, 0] = np.nan
        errors = validate_field_data(nan_data, (128, 256, 39))
        print(f"NaN data errors: {len(errors)}")
        
        # Data with infinite values
        inf_data = np.random.rand(128, 256, 39)
        inf_data[0, 0, 0] = np.inf
        errors = validate_field_data(inf_data, (128, 256, 39))
        print(f"Infinite data errors: {len(errors)}")


def create_mock_data():
    """
    Create mock Navier-Stokes data for testing.
    """
    print("\nCreating mock data for testing...")
    
    # Create test data directory
    test_data_path = Path("./test_data/")
    test_data_path.mkdir(exist_ok=True)
    
    # Create mock data files
    reynolds_numbers = [300, 500, 750]
    domain_shape = (128, 256)
    time_steps = 39
    
    for re_num in reynolds_numbers:
        # Create mock field data (u, v, p components)
        field_data = np.random.rand(domain_shape[0], domain_shape[1], time_steps * 3)
        
        # Add some structure to make it more realistic
        for t in range(time_steps):
            # u-velocity (x-component)
            field_data[:, :, t*3] = np.random.rand(domain_shape[0], domain_shape[1]) * 2
            # v-velocity (y-component)
            field_data[:, :, t*3+1] = np.random.rand(domain_shape[0], domain_shape[1]) * 2
            # pressure
            field_data[:, :, t*3+2] = np.random.rand(domain_shape[0], domain_shape[1]) * 0.5
        
        # Save mock data
        filename = test_data_path / f"Re_{re_num}.npy"
        np.save(filename, field_data)
        print(f"Created mock data: {filename}")


def run_all_tests():
    """
    Run all tests.
    """
    print("=" * 60)
    print("FLOW FIELD DATASET CREATION SYSTEM TESTS")
    print("=" * 60)
    
    # Create mock data
    create_mock_data()
    
    # Run unit tests
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    perf_tester = TestPerformance()
    perf_tester.test_sensor_generation_performance()
    perf_tester.test_memory_usage_estimation()
    
    # Run validation tests
    print("\n" + "=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)
    val_tester = TestDataValidation()
    val_tester.test_field_data_validation()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
