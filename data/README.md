# Flow Field Dataset Creation for TensorFlow

This module provides a comprehensive dataset creation system for flow field reconstruction using different sensor placement strategies. It's designed to work with Navier-Stokes simulation data and create TensorFlow-compatible datasets for machine learning applications.

## Features

- **Three sensor placement strategies**:
  - **Random**: Latin Hypercube Sampling for uniform distribution
  - **Circular**: Sensors placed around the obstacle
  - **Edge**: Sensors placed along horizontal boundaries

- **Multiple sensor configurations**: 8, 16, 32, and 64 sensors

- **TensorFlow integration**: Direct creation of tf.data.Dataset objects

- **Comprehensive visualization**: Sensor layouts and flow field plots

- **Data validation**: Built-in validation for sensor positions and field data

- **Memory optimization**: Efficient data handling and storage

## Directory Structure

```
dataset_creation/
├── flow_field_dataset.py     # Main dataset creation class
├── example_usage.py          # Usage examples and tutorials
├── config.py                 # Configuration parameters
├── utils.py                  # Utility functions
├── README.md                 # This file
├── sensor_layouts/           # Generated sensor position files
│   ├── sensor_layout_random_8.npy
│   ├── sensor_layout_circular_16.npy
│   └── ...
└── datasets/                 # Generated dataset files
    ├── dataset_random_8.npz
    ├── dataset_circular_16.npz
    └── ...
```

## Quick Start

### 1. Installation

Make sure you have the required dependencies:

```bash
pip install numpy tensorflow matplotlib
```

### 2. Basic Usage

```python
from flow_field_dataset import FlowFieldDatasetCreator

# Initialize the creator
creator = FlowFieldDatasetCreator(
    data_path="path/to/your/navier-stokes/data/",
    output_path="./dataset_creation/"
)

# Create all datasets
creator.create_all_datasets()

# Or create specific datasets
dataset = creator.create_dataset_for_layout('random', 32)
creator.save_dataset(dataset, 'random', 32)
```

### 3. Using with TensorFlow

```python
# Load a dataset
dataset = creator.load_dataset('random', 32)

# Create TensorFlow datasets
train_dataset, test_dataset = creator.create_tensorflow_dataset(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    test_split=0.2
)

# Use in model training
model.fit(
    train_dataset.map(lambda x: (x['sensor_data'], x['field_data'])),
    epochs=10,
    validation_data=test_dataset.map(lambda x: (x['sensor_data'], x['field_data']))
)
```

## Configuration

Edit `config.py` to customize:

- **Data paths**: Location of your Navier-Stokes data
- **Domain parameters**: Flow domain shape and time steps
- **Sensor configuration**: Number of sensors and layout types
- **TensorFlow parameters**: Batch size, test split, etc.

## Sensor Placement Strategies

### Random Placement
Uses Latin Hypercube Sampling to ensure uniform distribution across the flow domain.

```python
# Generate random sensor positions
sensor_positions = creator.generate_random_sensor_positions(32)
```

### Circular Placement
Places sensors around the circular obstacle at radius 21 pixels from center (64, 66).

```python
# Generate circular sensor positions
sensor_positions = creator.generate_circular_sensor_positions(32)
```

### Edge Placement
Places sensors along the top and bottom horizontal boundaries of the domain.

```python
# Generate edge sensor positions
sensor_positions = creator.generate_edge_sensor_positions(32)
```

## Dataset Format

Each dataset contains:

```python
dataset = {
    'sensor_data': np.ndarray,      # Shape: (n_re, n_sensors, time_steps)
    'field_data': np.ndarray,       # Shape: (n_re, height, width, time_steps)
    'sensor_positions': np.ndarray, # Shape: (n_sensors, 2)
    'reynolds_numbers': np.ndarray, # Shape: (n_re,)
    'layout_type': str,             # 'random', 'circular', or 'edge'
    'n_sensors': int                # Number of sensors
}
```

## API Reference

### FlowFieldDatasetCreator

Main class for dataset creation.

#### Methods

- `__init__(data_path, output_path, domain_shape, time_steps, reynolds_numbers)`
- `create_dataset_for_layout(layout_type, n_sensors)` - Create dataset for specific configuration
- `create_all_datasets()` - Create all possible dataset combinations
- `save_dataset(dataset, layout_type, n_sensors)` - Save dataset to file
- `load_dataset(layout_type, n_sensors)` - Load previously created dataset
- `create_tensorflow_dataset(dataset, batch_size, shuffle, test_split)` - Create TensorFlow datasets
- `visualize_sensor_layout(layout_type, n_sensors, reynolds_number)` - Visualize sensor placement
- `get_dataset_info()` - Get information about available datasets

#### Sensor Position Generation

- `generate_random_sensor_positions(n_sensors)` - Random placement using LHS
- `generate_circular_sensor_positions(n_sensors)` - Circular placement around obstacle
- `generate_edge_sensor_positions(n_sensors)` - Edge placement along boundaries

## Data Processing Pipeline

1. **Load raw Navier-Stokes data** (`Re_XXX.npy` files)
2. **Extract velocity magnitude** from u and v components
3. **Generate sensor positions** based on layout type
4. **Extract sensor measurements** at specified positions
5. **Create structured dataset** with all Reynolds numbers
6. **Save to compressed format** (.npz files)

## Examples

### Example 1: Create Single Dataset

```python
creator = FlowFieldDatasetCreator()

# Create dataset for random placement with 32 sensors
dataset = creator.create_dataset_for_layout('random', 32)
creator.save_dataset(dataset, 'random', 32)

# Visualize the sensor layout
creator.visualize_sensor_layout('random', 32)
```

### Example 2: Batch Processing

```python
# Create all datasets
creator.create_all_datasets()

# Get information about created datasets
info = creator.get_dataset_info()
print(f"Available datasets: {info['available_datasets']}")
```

### Example 3: TensorFlow Integration

```python
# Load dataset
dataset = creator.load_dataset('circular', 16)

# Create TensorFlow datasets
train_ds, test_ds = creator.create_tensorflow_dataset(
    dataset, 
    batch_size=64, 
    shuffle=True
)

# Example model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(128*256*39, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Training
model.fit(
    train_ds.map(lambda x: (x['sensor_data'], x['field_data'])),
    epochs=50,
    validation_data=test_ds.map(lambda x: (x['sensor_data'], x['field_data']))
)
```

## Validation and Quality Control

The system includes built-in validation:

- **Sensor position validation**: Ensures sensors are within domain bounds
- **Data range validation**: Checks for reasonable velocity magnitudes
- **Shape validation**: Verifies array dimensions
- **NaN/Inf detection**: Identifies invalid data points

## Memory Optimization

For large datasets, the system provides:

- **Compressed storage**: Uses `.npz` format with compression
- **Memory-efficient processing**: Processes one Reynolds number at a time
- **Data type optimization**: Converts to float32 when appropriate

## Troubleshooting

### Common Issues

1. **File not found errors**:
   - Check `data_path` in config.py
   - Ensure Navier-Stokes data files exist

2. **Memory errors**:
   - Reduce batch size
   - Enable memory optimization in config.py

3. **Sensor position errors**:
   - Check domain shape parameters
   - Verify flow region boundaries

### Performance Tips

- Use SSD storage for better I/O performance
- Increase batch size for better GPU utilization
- Enable compression for disk space savings

## Contributing

To extend the system:

1. Add new sensor placement strategies in `flow_field_dataset.py`
2. Implement additional data processing methods
3. Add visualization functions in `utils.py`
4. Update configuration parameters in `config.py`

## License

This project is part of the Physics-informed Machine Learning research and follows the same license as the parent project.

## Citation

If you use this dataset creation system in your research, please cite:

```bibtex
@software{flow_field_dataset_creator,
  title={Flow Field Dataset Creator for TensorFlow},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/flow_field_recon_parc}
}
```

## Support

For questions and issues:
- Check the `example_usage.py` for detailed examples
- Review the configuration in `config.py`
- Use the utility functions in `utils.py` for debugging
