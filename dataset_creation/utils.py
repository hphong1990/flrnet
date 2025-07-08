"""
Utility functions for the FlowFieldDatasetCreator.

This module contains helper functions for data processing, visualization,
and validation that are used by the main dataset creation class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union
import os
from pathlib import Path


def validate_sensor_positions(positions: np.ndarray, 
                            domain_shape: Tuple[int, int],
                            flow_region_y_start: int = 80) -> List[str]:
    """
    Validate sensor positions to ensure they are within the domain bounds.
    
    Args:
        positions: Array of sensor positions with shape (n_sensors, 2)
        domain_shape: Shape of the flow domain (height, width)
        flow_region_y_start: Start of flow region in y-direction
        
    Returns:
        List of validation errors (empty if no errors)
    """
    errors = []
    
    if positions.shape[1] != 2:
        errors.append("Sensor positions must have shape (n_sensors, 2)")
        return errors
    
    # Check x-coordinates
    x_coords = positions[:, 0]
    if np.any(x_coords < 0) or np.any(x_coords >= domain_shape[0]):
        errors.append(f"X-coordinates must be between 0 and {domain_shape[0] - 1}")
    
    # Check y-coordinates
    y_coords = positions[:, 1]
    if np.any(y_coords < flow_region_y_start) or np.any(y_coords >= domain_shape[1]):
        errors.append(f"Y-coordinates must be between {flow_region_y_start} and {domain_shape[1] - 1}")
    
    return errors


def validate_field_data(field_data: np.ndarray,
                       expected_shape: Tuple[int, int, int],
                       expected_range: Tuple[float, float] = (0, 10)) -> List[str]:
    """
    Validate field data to ensure it meets expected criteria.
    
    Args:
        field_data: Field data array
        expected_shape: Expected shape of the field data
        expected_range: Expected range of values (min, max)
        
    Returns:
        List of validation errors (empty if no errors)
    """
    errors = []
    
    # Check shape
    if field_data.shape != expected_shape:
        errors.append(f"Field data shape {field_data.shape} does not match expected shape {expected_shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(field_data)):
        errors.append("Field data contains NaN values")
    
    if np.any(np.isinf(field_data)):
        errors.append("Field data contains infinite values")
    
    # Check value range
    data_min, data_max = np.min(field_data), np.max(field_data)
    if data_min < expected_range[0] or data_max > expected_range[1]:
        errors.append(f"Field data range [{data_min:.2f}, {data_max:.2f}] is outside expected range {expected_range}")
    
    return errors


def calculate_dataset_statistics(dataset: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for a dataset.
    
    Args:
        dataset: Dataset dictionary containing sensor_data and field_data
        
    Returns:
        Dictionary containing statistics for each data type
    """
    stats = {}
    
    # Sensor data statistics
    if 'sensor_data' in dataset:
        sensor_data = dataset['sensor_data']
        stats['sensor_data'] = {
            'mean': np.mean(sensor_data),
            'std': np.std(sensor_data),
            'min': np.min(sensor_data),
            'max': np.max(sensor_data),
            'shape': sensor_data.shape
        }
    
    # Field data statistics
    if 'field_data' in dataset:
        field_data = dataset['field_data']
        stats['field_data'] = {
            'mean': np.mean(field_data),
            'std': np.std(field_data),
            'min': np.min(field_data),
            'max': np.max(field_data),
            'shape': field_data.shape
        }
    
    return stats


def plot_sensor_time_series(sensor_data: np.ndarray, 
                           sensor_idx: int = 0,
                           reynolds_idx: int = 0,
                           title: str = "Sensor Time Series",
                           save_path: Optional[str] = None):
    """
    Plot time series data for a specific sensor.
    
    Args:
        sensor_data: Sensor data array with shape (n_re, n_sensors, time_steps)
        sensor_idx: Index of the sensor to plot
        reynolds_idx: Index of the Reynolds number to plot
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 4))
    
    time_series = sensor_data[reynolds_idx, sensor_idx, :]
    time_steps = np.arange(len(time_series))
    
    plt.plot(time_steps, time_series, 'b-', linewidth=2, label=f'Sensor {sensor_idx}')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity Magnitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_field_snapshots(field_data: np.ndarray,
                        reynolds_idx: int = 0,
                        time_indices: List[int] = None,
                        title: str = "Field Snapshots",
                        save_path: Optional[str] = None):
    """
    Plot multiple snapshots of the flow field.
    
    Args:
        field_data: Field data array with shape (n_re, height, width, time_steps)
        reynolds_idx: Index of the Reynolds number to plot
        time_indices: List of time indices to plot
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if time_indices is None:
        time_indices = [0, field_data.shape[3] // 2, field_data.shape[3] - 1]
    
    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, t_idx in enumerate(time_indices):
        im = axes[i].imshow(field_data[reynolds_idx, :, :, t_idx], 
                           cmap='viridis', vmin=0, vmax=2)
        axes[i].set_title(f'Time Step {t_idx}')
        axes[i].set_xlabel('Y-coordinate')
        axes[i].set_ylabel('X-coordinate')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Velocity Magnitude')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_sensor_layouts(sensor_positions_dict: Dict[str, np.ndarray],
                          domain_shape: Tuple[int, int] = (128, 256),
                          save_path: Optional[str] = None):
    """
    Compare different sensor layouts in a single plot.
    
    Args:
        sensor_positions_dict: Dictionary with layout names as keys and positions as values
        domain_shape: Shape of the flow domain
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, len(sensor_positions_dict), 
                            figsize=(6 * len(sensor_positions_dict), 5))
    
    if len(sensor_positions_dict) == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (layout_name, positions) in enumerate(sensor_positions_dict.items()):
        # Create a dummy field for background
        dummy_field = np.zeros(domain_shape)
        
        axes[i].imshow(dummy_field, cmap='gray', alpha=0.3)
        axes[i].scatter(positions[:, 1], positions[:, 0], 
                       c=colors[i % len(colors)], s=50, 
                       label=f'{layout_name} ({len(positions)} sensors)')
        axes[i].set_title(f'{layout_name.capitalize()} Layout')
        axes[i].set_xlabel('Y-coordinate')
        axes[i].set_ylabel('X-coordinate')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_reconstruction_metrics(true_field: np.ndarray, 
                                   predicted_field: np.ndarray) -> Dict[str, float]:
    """
    Calculate reconstruction metrics between true and predicted fields.
    
    Args:
        true_field: True field data
        predicted_field: Predicted field data
        
    Returns:
        Dictionary containing various reconstruction metrics
    """
    # Mean Squared Error
    mse = np.mean((true_field - predicted_field) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_field - predicted_field))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Relative Error
    relative_error = np.mean(np.abs(true_field - predicted_field) / (np.abs(true_field) + 1e-8))
    
    # Correlation coefficient
    correlation = np.corrcoef(true_field.flatten(), predicted_field.flatten())[0, 1]
    
    # R-squared
    ss_res = np.sum((true_field - predicted_field) ** 2)
    ss_tot = np.sum((true_field - np.mean(true_field)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'relative_error': relative_error,
        'correlation': correlation,
        'r_squared': r_squared
    }


def create_dataset_summary(dataset_path: str) -> Dict[str, Union[str, int, float]]:
    """
    Create a summary of a dataset file.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dictionary containing dataset summary information
    """
    if not os.path.exists(dataset_path):
        return {'error': f'Dataset file not found: {dataset_path}'}
    
    try:
        data = np.load(dataset_path)
        
        summary = {
            'file_path': dataset_path,
            'file_size_mb': os.path.getsize(dataset_path) / (1024 * 1024),
            'keys': list(data.keys()),
        }
        
        # Add information about each array
        for key in data.keys():
            if hasattr(data[key], 'shape'):
                summary[f'{key}_shape'] = data[key].shape
                summary[f'{key}_dtype'] = str(data[key].dtype)
                if data[key].dtype in [np.float32, np.float64]:
                    summary[f'{key}_min'] = float(np.min(data[key]))
                    summary[f'{key}_max'] = float(np.max(data[key]))
                    summary[f'{key}_mean'] = float(np.mean(data[key]))
            else:
                summary[f'{key}_value'] = data[key]
        
        return summary
        
    except Exception as e:
        return {'error': f'Error reading dataset: {str(e)}'}


def memory_usage_mb(obj) -> float:
    """
    Calculate memory usage of an object in MB.
    
    Args:
        obj: Object to calculate memory usage for
        
    Returns:
        Memory usage in MB
    """
    if isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    elif isinstance(obj, dict):
        total_size = 0
        for value in obj.values():
            total_size += memory_usage_mb(value)
        return total_size
    else:
        return 0


def optimize_memory_usage(dataset: Dict[str, np.ndarray], 
                         target_dtype: str = 'float32') -> Dict[str, np.ndarray]:
    """
    Optimize memory usage of a dataset by converting to smaller data types.
    
    Args:
        dataset: Dataset dictionary
        target_dtype: Target data type for optimization
        
    Returns:
        Optimized dataset dictionary
    """
    optimized_dataset = {}
    
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            # Convert float64 to float32 to save memory
            optimized_dataset[key] = value.astype(target_dtype)
        else:
            optimized_dataset[key] = value
    
    return optimized_dataset


def generate_batch_report(datasets_path: str) -> str:
    """
    Generate a comprehensive report of all datasets in a directory.
    
    Args:
        datasets_path: Path to the datasets directory
        
    Returns:
        Formatted report string
    """
    datasets_path = Path(datasets_path)
    
    if not datasets_path.exists():
        return f"Directory not found: {datasets_path}"
    
    # Find all dataset files
    dataset_files = list(datasets_path.glob("dataset_*.npz"))
    
    if not dataset_files:
        return f"No dataset files found in: {datasets_path}"
    
    report = []
    report.append("=" * 80)
    report.append("DATASET BATCH REPORT")
    report.append("=" * 80)
    report.append(f"Directory: {datasets_path}")
    report.append(f"Number of datasets: {len(dataset_files)}")
    report.append("")
    
    total_size = 0
    
    for dataset_file in sorted(dataset_files):
        summary = create_dataset_summary(str(dataset_file))
        
        if 'error' in summary:
            report.append(f"ERROR - {dataset_file.name}: {summary['error']}")
            continue
        
        report.append(f"Dataset: {dataset_file.name}")
        report.append("-" * 40)
        report.append(f"  File size: {summary['file_size_mb']:.2f} MB")
        
        if 'sensor_data_shape' in summary:
            report.append(f"  Sensor data shape: {summary['sensor_data_shape']}")
        
        if 'field_data_shape' in summary:
            report.append(f"  Field data shape: {summary['field_data_shape']}")
        
        if 'n_sensors' in summary:
            report.append(f"  Number of sensors: {summary['n_sensors']}")
        
        if 'layout_type' in summary:
            report.append(f"  Layout type: {summary['layout_type']}")
        
        total_size += summary['file_size_mb']
        report.append("")
    
    report.append("=" * 80)
    report.append(f"Total size: {total_size:.2f} MB")
    report.append("=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage of utility functions
    print("Flow Field Dataset Utilities")
    print("=" * 40)
    
    # Example: validate sensor positions
    dummy_positions = np.array([[10, 100], [50, 150], [90, 200]])
    errors = validate_sensor_positions(dummy_positions, (128, 256))
    
    if errors:
        print("Sensor position validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Sensor positions are valid")
    
    # Example: memory usage calculation
    dummy_array = np.random.rand(100, 100, 100)
    memory_mb = memory_usage_mb(dummy_array)
    print(f"Memory usage of dummy array: {memory_mb:.2f} MB")
