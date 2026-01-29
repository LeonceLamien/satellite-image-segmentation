<div align="center">

# Deep Satellite Segmentation

### A Modular Deep Learning Framework for Semantic Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

*Semantic segmentation of satellite imagery using supervised learning (Part A) and self-supervised learning (Part B) approaches*

**[Features](#features)** • **[Quick Start](#quick-start)** • **[Architecture](#architecture)** • **[Documentation](#documentation)**

</div>

---

## About

**Deep Satellite Segmentation** is a comprehensive deep learning project. This project demonstrates both supervised and self-supervised approaches to semantic segmentation, featuring state-of-the-art architectures and a fully modular codebase.

### What makes this project special?

- **Dual Learning Paradigms**: Supervised segmentation (Part A) and self-supervised learning (Part B)
- **Advanced Architectures**: DeepLabV3+ with ASPP and Attention U-Net implementations
- **Complete Pipeline**: From data loading to evaluation with comprehensive metrics
- **Modular Design**: Shared modules eliminate code duplication and improve maintainability
- **Production-Ready**: Professional code structure with type hints, docstrings, and best practices

## Features

### Part A: Supervised Segmentation

- **DeepLabV3+**: Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
- **Attention U-Net**: Enhanced U-Net with attention gates for better segmentation
- **Advanced Training**: Early stopping, learning rate scheduling, and weight decay
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, IoU, ROC AUC
- **Visualization**: Confusion matrices, ROC curves, and segmentation overlays

### Part B: Self-Supervised Learning

- **Denoising**: Remove Gaussian noise to learn robust features
- **Colorization**: Predict color channels from grayscale images
- **Context Prediction**: Inpainting masked regions for spatial understanding
- **Flexible Architecture**: Encoder-decoder with ASPP bottleneck
- **Transfer Learning Ready**: Pre-trained encoders for downstream tasks

### Core Capabilities

- **Modular Design**: Shared `common/` modules for both parts
- **Configuration Management**: YAML-based configs with validation
- **Data Augmentation**: Extensive augmentations (flips, rotations, color jitter)
- **Checkpointing**: Automatic model saving with best weights tracking
- **Extensible**: Easy to add new models, tasks, or metrics

---

## Repository Structure

```
Deep-Satellite-Segmentation/
├── common/                    # Shared modules between Part A and Part B
│   ├── config.py              # Configuration management
│   ├── data/                  # Data loading and augmentation
│   │   ├── dataset.py         # Unified dataset class
│   │   └── transforms.py      # Albumentations transforms
│   ├── training/              # Training infrastructure
│   │   ├── base_trainer.py    # Base trainer class
│   │   └── callbacks.py       # Early stopping, etc.
│   ├── metrics/               # Evaluation metrics
│   │   ├── evaluation.py      # Metrics calculation
│   │   └── visualization.py   # Plotting functions
│   └── utils/                 # Utility functions
│       └── file_utils.py
│
├── PartA/                     # Supervised Segmentation
│   ├── main.py                # Entry point
│   ├── trainer.py             # Segmentation trainer
│   ├── Networks/
│   │   └── Architectures/
│   │       ├── deepLab.py      # DeepLabV3+ implementation
│   │       └── improvedUNet.py # Enhanced U-Net with attention
│   ├── Dataset/
│   │   ├── images/           # Training images
│   │   ├── annotations/      # Segmentation masks
│   │   ├── train.json        # Training file list
│   │   ├── val.json          # Validation file list
│   │   └── test.json         # Test file list
│   └── Todo_List/            # Configuration files
│       ├── DeepLab_Adam.yaml
│       └── ImprovedUNet_SGD.yaml
│
└── PartB/                        # Self-Supervised Learning
    ├── main.py                   # Entry point
    ├── trainer.py                # SSL trainer
    ├── proxy_tasks/              # Self-supervised tasks
    │   ├── denoising.py          # Image denoising
    │   ├── colorization.py       # Image colorization
    │   └── context_prediction.py # Context prediction (inpainting)
    ├── Dataset/                  # Same structure as Part A
    └── configs/                  # Configuration files
        ├── denoising.yaml
        ├── colorization.yaml
        └── context_pred.yaml
```

---

## Quick Start

### Prerequisites

- **Python 3.8+** (tested with Python 3.10)
- **PyTorch 2.0+** with CUDA support (optional)
- **Git** for cloning the repository

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/mathisdelsart/Deep-Satellite-Segmentation.git
cd Deep-Satellite-Segmentation
```

### Running Part 1: Supervised Learning

```bash
# Navigate to Part A
cd PartA

# Train DeepLabV3+ with Adam optimizer
python main.py -exp DeepLab_Adam

# Train Improved U-Net with SGD
python main.py -exp ImprovedUNet_SGD

# Results will be saved to PartA/Results/{exp_name}/
```

**What happens during training:**
1. Loads configuration from `Todo_List/{exp_name}.yaml`
2. Creates train/val/test datasets with augmentations
3. Trains model with early stopping
4. Evaluates on test set with comprehensive metrics
5. Generates visualizations (confusion matrix, ROC curves)

### Running Part B: Self-Supervised Learning

Train encoders using self-supervised proxy tasks:

```bash
# Navigate to Part B
cd PartB

# Train denoising task (simplest)
python main.py --task denoising --config configs/denoising.yaml --device cuda

# Train colorization task
python main.py --task colorization --config configs/colorization.yaml --device cuda

# Train context prediction (inpainting)
python main.py --task context_prediction --config configs/context_pred.yaml --device cuda
```

Results are saved to `PartB/Results/{task_name}_exp/`

---

## Configuration

### Part A: Supervised Training (YAML)

```yaml
DATASET:
  RESIZE_SHAPE: 64x64          # Image resize dimensions
  USE_AUGMENTATION: true       # Enable data augmentation
  HORIZONTAL_FLIP: true
  VERTICAL_FLIP: true
  ROTATE_LIMIT: 15
  BRIGHTNESS_CONTRAST: true
  BLUR: true
  COLOR_JITTER: true

MODEL:
  ARCHITECTURE: DeepLabV3Plus  # or ImprovedUNet
  BACKBONE_CHANNELS: 64
  DILATION_RATES: [6, 12, 18]  # For DeepLab ASPP
  ASPP_DROPOUT: 0.5

TRAINING:
  OPTIMIZER: Adam              # Adam or SGD
  LEARNING_RATE: 0.001
  EPOCH: 100
  BATCH_SIZE: 16
  DEVICE: cuda
  WEIGHT_DECAY: 0.0
  EARLY_STOPPING: true
  PATIENCE_EARLY_STOPPING: 10
  LR_SCHEDULER: ReduceLROnPlateau
  LR_PATIENCE: 5
  LR_FACTOR: 0.5
```

### Part B: Self-Supervised Training (YAML)

```yaml
DATASET:
  RESIZE_SHAPE: "64x64"

TASK:
  NOISE_STD: 0.12              # For denoising
  # MASK_RATIO: 0.3            # For context prediction
  # PATCH_SIZE: 8              # For context prediction

MODEL:
  BASE_CHANNELS: 64
  DROPOUT: 0.2
  DILATION_RATES: [6, 12, 18]

TRAIN:
  NUM_EPOCHS: 50
  BATCH_SIZE: 16
  LEARNING_RATE: 0.0001
  WEIGHT_DECAY: 0.0001
  USE_SCHEDULER: true
  EARLY_STOP_PATIENCE: 10
  EARLY_STOP_MIN_DELTA: 0.0001
```

---

## Architecture Overview

### Part A: Segmentation Models

#### DeepLabV3+
- Encoder with atrous spatial pyramid pooling (ASPP)
- Multi-scale feature extraction with dilated convolutions
- Decoder with skip connections
- Output: 5-class segmentation masks

#### Improved U-Net
- Encoder-decoder architecture
- Optional attention gates
- Batch normalization and dropout
- Skip connections at each level
- Output: 5-class segmentation masks

### Part B: Self-Supervised Tasks

#### 1. Denoising
- **Goal**: Remove Gaussian noise from images
- **Architecture**: Autoencoder with ASPP bottleneck
- **Loss**: MSE between predicted and clean images

#### 2. Colorization
- **Goal**: Predict color (ab channels) from grayscale (L channel)
- **Architecture**: Autoencoder with Lab color space
- **Loss**: MSE on ab channels

#### 3. Context Prediction
- **Goal**: Inpaint randomly masked image regions
- **Architecture**: Autoencoder with masked input
---

## Evaluation &on masked regions or full image

## Evaluation Metrics

### Segmentation Metrics (Part A)
- **Accuracy**: Overall pixel-wise accuracy
- **Precision, Recall, F1-Score**: Per-class and weighted averages
- **IoU (Jaccard)**: Intersection over Union
- **Confusion Matrix**: Absolute and normalized
- **ROC AUC**: Multi-class ROC curves with micro/macro averaging

### Visualizations
- Learning curves (train/val loss)
- Confusion matrices
- ROC curves for each class

---

## Results & Output Structure

Results for each experiment are saved to:
```
Results/{experiment_name}/
├── _Weights/
│   └── wghts.pkl              # Best model weights
├── _Figures/                   # (Part B)
│   └── training_curves.png
├── Test/                       # Reconstructed segmentations
├── learning_curves.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── roc_curves.png
└── test_metrics.txt           # Numerical metrics
```

---

## Authors

<div align="center">

<table>
  <tr>
    <td width="180" align="left">
      <img src="https://img.shields.io/badge/GitHub-mathisdelsart-black?logo=github" valign="middle"/>
    </td>
    <td align="left">
      <strong>Mathis DELSART</strong>
    </td>
  </tr>
  <tr>
    <td width="180" align="left">
      <img src="https://img.shields.io/badge/GitHub-leolamien-black?logo=github" valign="middle"/>
    </td>
    <td align="left">
      <strong>Siénou LAMIEN</strong>
    </td>
  </tr>
</table>

</div>

## License

This project is developed for academic purposes as part of university coursework.

---

<div align="center">

**Built for LELEC2885 - Image processing and computer vision @ UCLouvain** (Université catholique de Louvain).

</div>