# Flow Field Dataset Generation CLI

This directory contains the command-line interface for generating flow field datasets.

## Files

- `data_generation.py` - Main script for dataset generation
- `README.md` - This file
- `requirements.txt` - Python dependencies

## Usage

### Basic Usage

Generate datasets with default settings:
```bash
python data_generation.py
```

### Advanced Usage

Generate specific configurations:
```bash
python data_generation.py \
    --data-path "E:/Research/Data/NavierStokes/train/" \
    --output-path "./output/" \
    --sensor-counts 8 16 32 \
    --layouts random circular edge \
    --verbose
```

### Validation Only

Validate data and setup without generating datasets:
```bash
python data_generation.py --validate-only
```

## Command Line Options

- `--data-path`: Path to raw Navier-Stokes data files (default: E:/Research/Data/NavierStokes/train/)
- `--output-path`: Path to save generated datasets (default: ./data/)
- `--domain-shape`: Domain shape as height width (default: 128 256)
- `--time-steps`: Number of time steps (default: 39)
- `--sensor-counts`: List of sensor counts (default: 8 16 32)
- `--layouts`: Sensor layout types (choices: random, circular, edge)
- `--reynolds-numbers`: Specific Reynolds numbers to process
- `--obstacle-center`: Obstacle center coordinates (default: 64 128)
- `--obstacle-radius`: Obstacle radius (default: 22)
- `--seed`: Random seed for reproducibility (default: 42)
- `--validate-only`: Only validate, don't generate datasets
- `--verbose`: Enable verbose output

## Output

The script generates:
- Dataset files (.npz format) in the output directory
- Sensor layout files (.npy format) 
- Console logs with progress and statistics

## Examples

### Generate only random layouts with 16 sensors:
```bash
python data_generation.py --layouts random --sensor-counts 16
```

### Generate with custom obstacle parameters:
```bash
python data_generation.py \
    --obstacle-center 60 120 \
    --obstacle-radius 25 \
    --seed 123
```

### Validate data format before generation:
```bash
python data_generation.py --validate-only --verbose
```
