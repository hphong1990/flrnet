# Flow Field Reconstruction with Physics-Informed Machine Learning

This repository implements **FLRNet**, a novel deep learning method for reconstructing flow fields from sparse sensor measurements, as presented in our paper "[FLRNet: A Deep Learning Method for Regressive Reconstruction of Flow Field From Limited Sensor Measurements](https://arxiv.org/abs/2411.13815)".

## 🚀 Project Overview

Flow field reconstruction from limited sensor data is a fundamental challenge in computational and experimental fluid mechanics. Traditional methods often fail due to the ill-conditioned and non-invertible nature of the measurement operator. This repository provides **FLRNet** and baseline comparison methods to address this challenge.

### 🏆 **Key Contributions**

1. **FLRNet Architecture**: A novel variational autoencoder with Fourier feature layers that learns rich, low-dimensional latent representations of flow fields
2. **Perceptual Loss Integration**: Addresses spectral bias issues that lead to smooth and blurry reconstructed fields
3. **Fourier Features**: Use Fourier feature layers to address spectral bias issue
4. **Comprehensive Evaluation**: Systematic comparison across different Reynolds numbers, sensor configurations, and noise conditions

### 🧠 **Implemented Methods**

- **FLRNet (Fluid Flow Reconstruction Network)**: Our proposed architecture combining VAE with sensor mapping for precise flow field reconstruction
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
- Required packages listed in `requirements.txt`

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data and Checkpoints**
   
   Download the required datasets and pre-trained model checkpoints from the following link:
   
   **📎 Data & Checkpoints Download:** `[INSERT DOWNLOAD LINK HERE]`
   
   Extract the downloaded files to the following directories:
   ```
   data/datasets/          # Training/testing data
   data/sensor_layouts/    # Sensor layout data

   checkpoints/           # Pre-trained model weights
   ```

3. **Verify Installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
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

- **Reconstruction Accuracy**: MSE, MAE
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

This repository is part of ongoing research in physics-informed deep learning for fluid flow field reconstruction. Contributions and improvements are welcome.

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{nguyen2024flrnet,
  title={FLRNet: A Deep Learning Method for Regressive Reconstruction of Flow Field From Limited Sensor Measurements},
  author={Nguyen, Phong C. H. and Choi, Joseph B. and Luu, Quang-Trung},
  journal={arXiv preprint arXiv:2411.13815},
  year={2024},
  url={https://arxiv.org/abs/2411.13815}
}
```

**Paper Abstract:**
> Many applications in computational and experimental fluid mechanics require effective methods for reconstructing flow fields from limited sensor data. However, this task remains a significant challenge because the measurement operator is often ill-conditioned and non-invertible. We introduce FLRNet, a deep learning method for flow field reconstruction from sparse sensor measurements. FLRNet employs a variational autoencoder with Fourier feature layers and incorporates an extra perceptual loss term during training to learn a rich, low-dimensional latent representation of the flow field. Numerical experiments show that FLRNet consistently outperformed other baselines, delivering the most accurate reconstructed flow field and being the most robust to noise.

## 📄 License

See LICENSE file for details.

---

**Note**: Ensure you have downloaded the required data and checkpoint files before running the notebooks. Each notebook is self-contained and includes both training and inference capabilities - simply modify the training flags to switch between modes.