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


def reorganize_data_for_tensorflow(data_path: str, 
                                 channels: List[str] = ['u_velocity', 'v_velocity'],
                                 visualize: bool = True,
                                 save_back: bool = True) -> Dict:
    """
    Reorganize flow field data for TensorFlow compatibility.
    
    This function processes .npy files in the specified directory, converting them from
    the format (time, channels, height, width) to TensorFlow-compatible format 
    (time, height, width, channels), keeping only the specified velocity channels.
    
    Args:
        data_path: Path to directory containing .npy files
        channels: List of channel names to keep (default: u and v velocity)
        visualize: Whether to create visualizations of the processed data
        save_back: Whether to save the modified data back to original files
        
    Returns:
        Dictionary containing data information and statistics
    """
    print("🔄 Reorganizing Data Format for TensorFlow...")
    print("=" * 50)

    # Global variables to store data info
    data_info = {
        'original_shape': None,
        'reorganized_shape': None,
        'channels': channels,
        'data_range': None,
        'sample_files': [],
        'channel_ranges': {}
    }

    try:
        # Try to access the actual data directory first
        data_dir = Path(data_path)
        
        if data_dir.exists():
            npy_files = list(data_dir.glob("*.npy"))
            
            if npy_files:
                # Process all files in the directory
                for file_idx, sample_file in enumerate(npy_files):
                    print(f"\n📁 Processing file {file_idx + 1}/{len(npy_files)}: {sample_file.name}")
                    
                    original_data = np.load(str(sample_file))
                    
                    if file_idx == 0:  # Only print details for first file
                        data_info['original_shape'] = original_data.shape
                        print(f"📊 Original data shape: {original_data.shape}")
                        print(f"📈 Data type: {original_data.dtype}")
                        print(f"📉 Original data range: [{np.min(original_data):.6f}, {np.max(original_data):.6f}]")
                    
                    # Check if this matches the expected format (time, channels, height, width)
                    if len(original_data.shape) == 4 and original_data.shape[1] == 2:
                        # Transpose to TensorFlow format: (time, height, width, channels)
                        reorganized_data = np.transpose(original_data, (0, 2, 3, 1))
                        
                        if file_idx == 0:  # Only print details for first file
                            data_info['reorganized_shape'] = reorganized_data.shape
                            data_info['data_range'] = [float(np.min(reorganized_data)), float(np.max(reorganized_data))]
                            
                            print(f"✅ Selected channels u and v velocities")
                            print(f"📊 New data shape: {reorganized_data.shape}")
                            print(f"📉 New data range: [{data_info['data_range'][0]:.6f}, {data_info['data_range'][1]:.6f}]")
                            
                            # Analyze each channel separately
                            print(f"\n📊 Detailed Channel Analysis:")
                            for i, channel_name in enumerate(data_info['channels']):
                                channel_data = reorganized_data[:, :, :, i]
                                channel_min = np.min(channel_data)
                                channel_max = np.max(channel_data)
                                channel_mean = np.mean(channel_data)
                                channel_std = np.std(channel_data)
                                
                                data_info['channel_ranges'][channel_name] = {
                                    'min': float(channel_min),
                                    'max': float(channel_max),
                                    'mean': float(channel_mean),
                                    'std': float(channel_std)
                                }
                                
                                print(f"   Channel {i} ({channel_name}):")
                                print(f"     Range: [{channel_min:.3f}, {channel_max:.3f}]")
                                print(f"     Mean: {channel_mean:.3f}, Std: {channel_std:.3f}")
                        
                        # Save the modified data back to the original file if requested
                        if save_back:
                            np.save(str(sample_file), reorganized_data.astype(np.float32))
                            
                            if file_idx == 0:
                                print(f"\n✅ Saved modified data back to: {sample_file}")
                                print(f"📊 Final shape: {reorganized_data.shape} (time, height, width, channels)")
                                print(f"📈 Data type: float32")
                                
                                # Store reorganized data for use in creator (TensorFlow format)
                                data_info['reorganized_data'] = reorganized_data
                    
                    else:
                        print(f"⚠️ Unexpected data shape in {sample_file.name}: {original_data.shape}")
                        print(f"💡 Expected shape: (time_steps, 2, height, width)")
                
                if npy_files:
                    print(f"\n✅ Processed {len(npy_files)} files successfully!")
                    
                    # Create visualization for the first file's data
                    if visualize and 'reorganized_data' in data_info:
                        print(f"\n🖼️ Creating visualization of u and v velocity channels...")
                        
                        # Select multiple time steps for better understanding
                        time_steps_to_plot = [0, 10, 20, 30]
                        reorganized_data = data_info['reorganized_data']
                        
                        # Create a figure with subplots
                        fig, axes = plt.subplots(len(time_steps_to_plot), 2, figsize=(12, 16))
                        
                        for t_idx, t in enumerate(time_steps_to_plot):
                            for ch_idx, channel_name in enumerate(data_info['channels']):
                                channel_data = reorganized_data[t, :, :, ch_idx]
                                
                                # Use appropriate colormap for velocity channels
                                cmap = 'RdBu_r'
                                vmin, vmax = np.percentile(channel_data, [2, 98])
                                
                                im = axes[t_idx, ch_idx].imshow(channel_data, cmap=cmap, 
                                                               vmin=vmin, vmax=vmax,
                                                               aspect='auto')
                                
                                # Add titles and labels
                                if t_idx == 0:
                                    axes[t_idx, ch_idx].set_title(f'{channel_name.replace("_", " ").title()}', 
                                                                 fontsize=14, fontweight='bold')
                                
                                if ch_idx == 0:
                                    axes[t_idx, ch_idx].set_ylabel(f'Time Step {t}', fontsize=12)
                                
                                # Add colorbar
                                plt.colorbar(im, ax=axes[t_idx, ch_idx], shrink=0.8)
                                
                                # Remove tick labels for cleaner look
                                axes[t_idx, ch_idx].set_xticks([])
                                axes[t_idx, ch_idx].set_yticks([])
                        
                        plt.tight_layout()
                        plt.suptitle('U and V Velocity Channels Across Time Steps', fontsize=16, fontweight='bold', y=0.98)
                        plt.show()
                        
                        # Create a summary plot showing the evolution of each channel's statistics
                        print(f"\n📈 Temporal Evolution of Velocity Channel Statistics:")
                        
                        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                        
                        for ch_idx, channel_name in enumerate(data_info['channels']):
                            channel_data = reorganized_data[:, :, :, ch_idx]
                            
                            # Calculate statistics over time
                            time_means = np.mean(channel_data, axis=(1, 2))
                            time_stds = np.std(channel_data, axis=(1, 2))
                            time_mins = np.min(channel_data, axis=(1, 2))
                            time_maxs = np.max(channel_data, axis=(1, 2))
                            
                            time_steps = np.arange(len(time_means))
                            
                            # Plot mean with error bars
                            axes[ch_idx].plot(time_steps, time_means, 'b-', linewidth=2, label='Mean')
                            axes[ch_idx].fill_between(time_steps, time_means - time_stds, time_means + time_stds, 
                                                    alpha=0.3, color='blue', label='±1 Std')
                            axes[ch_idx].plot(time_steps, time_mins, 'r--', alpha=0.7, label='Min')
                            axes[ch_idx].plot(time_steps, time_maxs, 'g--', alpha=0.7, label='Max')
                            
                            axes[ch_idx].set_title(f'{channel_name.replace("_", " ").title()}', fontweight='bold')
                            axes[ch_idx].set_xlabel('Time Step')
                            axes[ch_idx].set_ylabel('Velocity')
                            axes[ch_idx].legend()
                            axes[ch_idx].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.suptitle('Temporal Evolution of Velocity Statistics', fontsize=16, fontweight='bold', y=1.05)
                        plt.show()
                        
                        print(f"\n✅ Data reorganization completed successfully!")
                        print(f"🎯 Ready to update FlowFieldDatasetCreator with new data format")
                        
                        # Summary of what we found
                        print(f"\n📋 Data Summary:")
                        print(f"   - Original format: (time, 2_channels, height, width)")
                        print(f"   - TensorFlow format: (time, height, width, 2_channels)")
                        print(f"   - Time steps: {reorganized_data.shape[0]}")
                        print(f"   - Spatial dimensions: {reorganized_data.shape[1]} x {reorganized_data.shape[2]}")
                        print(f"   - Number of channels: {reorganized_data.shape[3]}")
                        print(f"   - Data type: float32 (TensorFlow compatible)")
                        print(f"   - Files processed: {len(npy_files)}")
            else:
                print(f"❌ No .npy files found in {data_path}")
        else:
            print(f"❌ Data directory not accessible: {data_path}")
            
    except Exception as e:
        print(f"❌ Error accessing external data: {str(e)}")
        print(f"💡 Make sure the data directory exists and contains .npy files")
    
    print("\n✅ Data format reorganization completed!")
    return data_info


def inspect_data_directory(data_path: str, 
                           visualize: bool = True,
                           create_test_data: bool = True,
                           creator_config: Optional[Dict] = None) -> Dict:
    """
    Inspect data directory and analyze file formats.
    
    This function examines .npy files in a directory, loads sample files to understand
    their format, shows statistics, and optionally creates visualizations and test data.
    
    Args:
        data_path: Path to directory containing .npy files
        visualize: Whether to create visualizations of the data
        create_test_data: Whether to create test data if directory is not accessible
        creator_config: Optional configuration for test data creation
        
    Returns:
        Dictionary containing inspection results and file information
    """
    print("📁 Inspecting Data Directory...")
    print("=" * 50)
    
    inspection_results = {
        'directory_exists': False,
        'file_count': 0,
        'sample_files': [],
        'data_info': {},
        'test_data_created': False,
        'test_data_path': None
    }
    
    try:
        # List files using pathlib
        data_dir = Path(data_path)
        if data_dir.exists():
            inspection_results['directory_exists'] = True
            print(f"✅ Data directory exists: {data_path}")
            
            # Get all numpy files
            npy_files = list(data_dir.glob("*.npy"))
            inspection_results['file_count'] = len(npy_files)
            inspection_results['sample_files'] = [f.name for f in npy_files[:5]]
            
            print(f"📊 Found {len(npy_files)} .npy files")
            
            # Show the first 5 files
            print("\n📋 First 5 data files:")
            for i, file in enumerate(npy_files[:5]):
                print(f"  {i+1}. {file.name}")
            
            # Load the first file to inspect its format
            if npy_files:
                sample_file_path = npy_files[0]
                print(f"\n🔍 Loading sample file: {sample_file_path.name}")
                
                sample_data = np.load(str(sample_file_path))
                
                # Store data information
                data_info = {
                    'shape': sample_data.shape,
                    'dtype': str(sample_data.dtype),
                    'min': float(np.min(sample_data)),
                    'max': float(np.max(sample_data)),
                    'mean': float(np.mean(sample_data)),
                    'std': float(np.std(sample_data)),
                    'median': float(np.median(sample_data))
                }
                inspection_results['data_info'] = data_info
                
                print(f"📊 Data shape: {data_info['shape']}")
                print(f"📈 Data type: {data_info['dtype']}")
                print(f"📉 Data range: [{data_info['min']:.6f}, {data_info['max']:.6f}]")
                
                # Show data statistics
                print(f"\n📊 Data Statistics:")
                print(f"  - Mean: {data_info['mean']:.6f}")
                print(f"  - Std: {data_info['std']:.6f}")
                print(f"  - Median: {data_info['median']:.6f}")
                
                # Visualize a slice of the data if it's 3D or 4D
                if visualize:
                    dims = len(sample_data.shape)
                    
                    if dims > 2:
                        print("\n🖼️ Visualizing a slice of the data:")
                        
                        plt.figure(figsize=(10, 5))
                        
                        if dims == 3:  # Assuming (time, height, width) format
                            time_step = 0  # First time step
                            plt.imshow(sample_data[time_step], cmap='viridis')
                            plt.colorbar(label='Value')
                            plt.title(f'Data Slice at Time Step {time_step}')
                            
                        elif dims == 4:  # Assuming (time, height, width, channels) format
                            time_step = 0  # First time step
                            channel = 0  # First channel
                            plt.imshow(sample_data[time_step, :, :, channel], cmap='viridis')
                            plt.colorbar(label='Value')
                            plt.title(f'Data Slice at Time Step {time_step}, Channel {channel}')
                        
                        plt.tight_layout()
                        plt.show()
        else:
            print(f"❌ Data directory not found: {data_path}")
            print("💡 Please check the path and make sure it exists.")

    except Exception as e:
        print(f"❌ Error accessing data directory: {str(e)}")
        print("💡 This might be due to permission issues or the path being outside the workspace boundary.")
        print("\n⚠️ Alternative approach: Upload a sample file to the workspace or specify the exact file path.\n")
        
        if create_test_data:
            # Check if we can create a simple test file to validate our approach
            print("🔧 Creating a test file to validate our approach:")
            
            # Use provided creator config or defaults
            if creator_config:
                test_data_shape = (creator_config.get('time_steps', 39), 
                                 creator_config.get('domain_shape', (128, 256))[0],
                                 creator_config.get('domain_shape', (128, 256))[1])
                obstacle_center = creator_config.get('obstacle_center', (64, 64))
                obstacle_radius = creator_config.get('obstacle_radius', 22)
            else:
                test_data_shape = (39, 128, 256)  # Expected shape based on notebook configuration
                obstacle_center = (64, 64)
                obstacle_radius = 22
            
            print(f"📊 Creating test data with shape {test_data_shape}")
            
            # Create synthetic data similar to what we might expect
            test_data = np.random.randn(*test_data_shape) * 0.1
            
            # Add circular obstacle pattern to make it look like flow data
            center_y, center_x = obstacle_center
            
            y, x = np.ogrid[:test_data_shape[1], :test_data_shape[2]]
            dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            
            # Create a flow-like pattern around the obstacle
            for t in range(test_data.shape[0]):
                mask = dist_from_center <= obstacle_radius + 5
                test_data[t][mask] = 0  # Zero inside obstacle
                
                # Add wake pattern behind obstacle
                wake_mask = (x > center_x) & (np.abs(y - center_y) < obstacle_radius * 1.5)
                pattern = np.sin(x[wake_mask] * 0.1 + t * 0.2) * np.exp(-(x[wake_mask] - center_x) * 0.01)
                test_data[t][wake_mask] = pattern * 0.5
            
            # Save the test data
            test_file_path = './dataset_creation/test_flow_data.npy'
            np.save(test_file_path, test_data)
            
            inspection_results['test_data_created'] = True
            inspection_results['test_data_path'] = test_file_path
            inspection_results['data_info'] = {
                'shape': test_data.shape,
                'dtype': str(test_data.dtype),
                'min': float(np.min(test_data)),
                'max': float(np.max(test_data)),
                'mean': float(np.mean(test_data)),
                'std': float(np.std(test_data)),
                'median': float(np.median(test_data))
            }
            
            print(f"✅ Created test flow data at: {test_file_path}")
            print(f"📊 Test data shape: {test_data.shape}")
            print(f"📈 Test data range: [{np.min(test_data):.6f}, {np.max(test_data):.6f}]")
            
            # Visualize the test data
            if visualize:
                print("\n🖼️ Visualizing the test data:")
                
                plt.figure(figsize=(10, 5))
                time_step = 10  # Middle time step for better pattern
                plt.imshow(test_data[time_step], cmap='viridis')
                plt.colorbar(label='Value')
                plt.title(f'Test Data Slice at Time Step {time_step}')
                plt.tight_layout()
                plt.show()
            
            print("\n✅ Test data created successfully!")
            print("🚀 Now we can adjust our dataset creator to use this test file.")

    print("\n✅ Data inspection complete!")
    return inspection_results


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
