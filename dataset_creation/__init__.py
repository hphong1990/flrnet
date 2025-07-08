"""
Flow Field Dataset Creation Package

This package provides comprehensive tools for creating TensorFlow datasets
from Navier-Stokes flow field data with different sensor placement strategies.

Main Components:
- FlowFieldDatasetCreator: Main class for dataset creation
- Sensor placement strategies: random, circular, edge
- TensorFlow integration: Direct tf.data.Dataset creation
- Visualization tools: Plot sensor layouts and flow fields
- Validation utilities: Data quality checks and validation

Usage:
    from dataset_creation import FlowFieldDatasetCreator
    
    creator = FlowFieldDatasetCreator()
    creator.create_all_datasets()
"""

from .flow_field_dataset import FlowFieldDatasetCreator
from .config import get_config_dict, validate_config, print_config
from .utils import (
    validate_sensor_positions,
    validate_field_data,
    calculate_dataset_statistics,
    plot_sensor_time_series,
    plot_field_snapshots,
    compare_sensor_layouts,
    calculate_reconstruction_metrics,
    create_dataset_summary,
    memory_usage_mb,
    optimize_memory_usage,
    generate_batch_report
)

__version__ = "1.0.0"
__author__ = "Physics-informed Machine Learning Research Group"
__email__ = "your.email@institution.edu"

__all__ = [
    # Main class
    'FlowFieldDatasetCreator',
    
    # Configuration functions
    'get_config_dict',
    'validate_config',
    'print_config',
    
    # Utility functions
    'validate_sensor_positions',
    'validate_field_data',
    'calculate_dataset_statistics',
    'plot_sensor_time_series',
    'plot_field_snapshots',
    'compare_sensor_layouts',
    'calculate_reconstruction_metrics',
    'create_dataset_summary',
    'memory_usage_mb',
    'optimize_memory_usage',
    'generate_batch_report',
]

# Package metadata
__package_info__ = {
    'name': 'flow_field_dataset_creation',
    'version': __version__,
    'description': 'TensorFlow dataset creation for flow field reconstruction',
    'author': __author__,
    'email': __email__,
    'url': 'https://github.com/your-repo/flow_field_recon_parc',
    'license': 'MIT',
    'python_requires': '>=3.7',
    'keywords': ['flow field', 'machine learning', 'tensorflow', 'navier-stokes', 'sensor placement'],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
    ],
}

def get_package_info():
    """
    Get package information.
    
    Returns:
        Dictionary containing package metadata
    """
    return __package_info__

def print_package_info():
    """
    Print package information.
    """
    info = get_package_info()
    print("=" * 60)
    print(f"FLOW FIELD DATASET CREATION PACKAGE v{__version__}")
    print("=" * 60)
    print(f"Description: {info['description']}")
    print(f"Author: {info['author']}")
    print(f"Email: {info['email']}")
    print(f"URL: {info['url']}")
    print(f"License: {info['license']}")
    print(f"Python: {info['python_requires']}")
    print("=" * 60)
    
    # Print available functions
    print("\nAvailable Functions:")
    print("-" * 20)
    for func_name in __all__:
        if func_name == 'FlowFieldDatasetCreator':
            print(f"  {func_name} (Main Class)")
        else:
            print(f"  {func_name}")
    print("=" * 60)

# Version compatibility check
import sys
if sys.version_info < (3, 7):
    raise RuntimeError(
        f"This package requires Python 3.7 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Optional dependency checks
def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'numpy': False,
        'tensorflow': False,
        'matplotlib': False,
        'scipy': False,
    }
    
    # Check numpy
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        dependencies['numpy'] = False
    
    # Check tensorflow
    try:
        import tensorflow
        dependencies['tensorflow'] = tensorflow.__version__
    except ImportError:
        dependencies['tensorflow'] = False
    
    # Check matplotlib
    try:
        import matplotlib
        dependencies['matplotlib'] = matplotlib.__version__
    except ImportError:
        dependencies['matplotlib'] = False
    
    # Check scipy
    try:
        import scipy
        dependencies['scipy'] = scipy.__version__
    except ImportError:
        dependencies['scipy'] = False
    
    return dependencies

def print_dependency_status():
    """
    Print the status of all dependencies.
    """
    deps = check_dependencies()
    
    print("\nDependency Status:")
    print("-" * 20)
    
    for dep_name, status in deps.items():
        if status:
            print(f"  ✓ {dep_name}: {status}")
        else:
            print(f"  ✗ {dep_name}: Not installed")
    
    # Check if all required dependencies are available
    required_deps = ['numpy', 'tensorflow', 'matplotlib']
    missing_deps = [dep for dep in required_deps if not deps[dep]]
    
    if missing_deps:
        print(f"\nMissing required dependencies: {missing_deps}")
        print("Please install using: pip install -r requirements.txt")
    else:
        print("\n✓ All required dependencies are available!")

# Initialization message
def _init_message():
    """Display initialization message when package is imported."""
    print(f"Flow Field Dataset Creation Package v{__version__} loaded successfully!")
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [dep for dep in ['numpy', 'tensorflow', 'matplotlib'] if not deps[dep]]
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {missing_deps}")
        print("Some features may not work correctly.")
    
    # Print quick start info
    print("\nQuick Start:")
    print("  from dataset_creation import FlowFieldDatasetCreator")
    print("  creator = FlowFieldDatasetCreator()")
    print("  creator.create_all_datasets()")
    print("\nFor more info: print_package_info()")

# Call initialization message
_init_message()
