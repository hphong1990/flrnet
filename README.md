# Flow Field Reconstruction with Physics-Informed Machine Learning

This repository implements physics-informed machine learning approaches for reconstructing flow fields from sparse sensor measurements. The project includes multiple neural network architectures optimized for different reconstruction scenarios.

## 🚀 Project Overview

The repository provides comprehensive solutions for flow field reconstruction using:

- **FLRNet (Flow Learning Reconstruction Network)**: A novel architecture combining VAE with sensor mapping for precise flow field reconstruction
- **VAE (Variational Autoencoder)**: For learning compressed representations of flow fields with optional Fourier feature embeddings
- **MLP (Multi-Layer Perceptron)**: Direct sensor-to-field mapping without intermediate representation
- **POD (Proper Orthogonal Decomposition)**: Classical reduced-order modeling approach with neural network enhancement

### Key Features
- 🌊 Multiple sensor layouts (random, edge, circular) with 8, 16, 32 sensors
- 🔄 Fourier feature embeddings for enhanced spatial understanding
- 📊 Perceptual loss integration for improved reconstruction quality
- 🎯 Physics-informed constraints and regularization
- 📈 Comprehensive training pipelines with automated checkpointing
- 🔍 Extensive evaluation metrics and visualization tools

## 📁 Repository Structure

```
├── checkpoints/          # Pre-trained model weights
├── config/              # Configuration files for different experiments
├── data/                # Dataset creation and utilities
├── demo/                # Self-contained training/inference notebooks
├── logs/                # Training logs and metrics
├── metrics/             # Evaluation and analysis tools
├── nn/                  # Neural network model implementations
└── report/              # Documentation and results
```

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- NumPy, Matplotlib, SciPy
- Jupyter Notebook

### Quick Start

1. **Download Data and Checkpoints**
   
   Download the required datasets and pre-trained model checkpoints from the provided link and extract them to:
   ```
   data/datasets/          # Training/testing data
   data/sensor_layouts/    # Sensor layout data

   checkpoints/           # Pre-trained model weights
   ```

2. **Install Dependencies**
   ```bash
   pip install tensorflow numpy matplotlib scipy jupyter
   ```

## 🎯 Usage

### Self-Contained Notebooks

Each model has a dedicated notebook in the `demo/` folder that includes both training and inference capabilities:

| Model | Notebook | Description |
|-------|----------|-------------|
| **FLRNet** | `demo/flrnet_train.ipynb` | Complete FLRNet training and inference pipeline |
| **VAE** | `demo/vae_train.ipynb` | Variational autoencoder training and evaluation |
| **MLP** | `demo/mlp_unified_train.ipynb` | Direct MLP-based reconstruction |
| **POD** | `demo/pod_unified_train.ipynb` | POD-enhanced neural reconstruction |

### Training vs. Inference Mode

Each notebook includes training flags that can be modified for inference-only mode:

```python
# Set to False for inference-only mode
train_vae_model = True      # Skip VAE training
train_flrnet_model = True   # Skip FLRNet training

# For inference only, set all training flags to False
train_vae_model = False
train_flrnet_model = False
```

When training flags are set to `False`, the notebooks will:
- Load pre-trained models from checkpoints
- Skip training phases
- Run inference and evaluation only
- Display visualization results

### Model Configurations

The `config/` directory contains pre-configured YAML files for different experimental setups:

- **Sensor layouts**: `random_8`, `random_16`, `random_32`, `edge_32`, `circular_32`
- **Features**: Standard vs. Fourier-enhanced (`fourier` suffix)
- **Loss functions**: With/without perceptual loss (`percep` suffix)

Example configurations:
```
config/random_32_fourier.yaml      # 32 random sensors with Fourier features
config/edge_32_no_fourier.yaml     # 32 edge sensors, standard features
config/circular_32_standard.yaml   # 32 circular sensors with perceptual loss
```

## 🧠 Model Architectures

### FLRNet
- **Encoder-Decoder**: VAE-based architecture for flow field compression
- **Sensor Mapping**: Neural network mapping sensor readings to latent space
- **Reconstruction**: Decoder transforms latent representation back to flow field
- **Key Features**: Frozen autoencoder training, KL divergence regularization

### VAE (Variational Autoencoder)
- **Probabilistic Encoder**: Maps flow fields to latent distributions
- **Decoder**: Reconstructs flow fields from latent samples
- **Optional Features**: Fourier embeddings, perceptual loss

### MLP (Multi-Layer Perceptron)
- **Direct Mapping**: Sensor readings → Flow field reconstruction
- **Architecture**: Deep feedforward network with batch normalization
- **Advantages**: Simple, fast training, no intermediate representations

### POD (Proper Orthogonal Decomposition)
- **Classical Approach**: Physics-based dimensionality reduction
- **Neural Enhancement**: MLP learns sensor → POD coefficient mapping
- **Reconstruction**: Linear combination of POD modes

## 📊 Pre-trained Models

The `checkpoints/` directory contains pre-trained models for various configurations:

```
checkpoints/
├── fourierTrue_percepTrue_random_32/    # Best performing configuration
├── fourierFalse_percepFalse_edge_32/    # Standard edge sensor setup
├── fourierTrue_percepFalse_circular_32/ # Circular layout with Fourier
└── ...                                  # Additional configurations
```

Each checkpoint directory contains:
- `*_vae_best.index`: Best VAE model weights
- `*_flrnet_best.index`: Best FLRNet model weights
- Training logs and configuration files

## 🔍 Evaluation and Metrics

The repository includes comprehensive evaluation tools:

- **Reconstruction Accuracy**: MSE, MAE, SSIM
- **Physical Constraints**: Divergence-free validation
- **Spectral Analysis**: Frequency domain comparison
- **Visualization**: Flow field plots, error maps, sensor layouts

## 📖 Quick Start Examples

### 1. Run FLRNet Inference
```python
# Open demo/flrnet_train.ipynb
# Set training flags to False for inference only
train_vae_model = False
train_flrnet_model = False
# Execute all cells to load models and run inference
```

### 2. Train New Model
```python
# Open desired model notebook
# Set training flags to True
train_vae_model = True
# Configure parameters as needed
# Execute training cells
```

### 3. Custom Configuration
```python
# Modify config files in config/ directory
# Update notebook parameters
# Run training/inference pipeline
```

## 🤝 Contributing

This repository is part of ongoing research in deep learning for fluid flow field reconstruction. Contributions and improvements are welcome.

## 📄 License

See LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite the associated publication.

---

**Note**: Ensure you have downloaded the required data and checkpoint files before running the notebooks. Each notebook is self-contained and includes both training and inference capabilities - simply modify the training flags to switch between modes.