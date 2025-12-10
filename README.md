# Part B - Self-Supervised Learning for Semantic Segmentation

This folder contains the implementation of Part B: self-supervised learning using proxy tasks.

## Structure

```
PartB/
├── README.md                   # This file
├── proxy_tasks/               # Self-supervised proxy task implementations
│   ├── denoising.py          # Task 1: Image denoising (simplest)
│   ├── colorization.py       # Task 2: Grayscale to RGB colorization
│   └── context_prediction.py # Task 3: Masked patch prediction
├── clustering/
│   ├── cluster_features.py   # Clustering on encoder features
│   └── visualize_clusters.py # Visualization and analysis
├── models/
│   ├── ssl_architectures.py  # Adapted encoder-decoder for SSL
│   └── encoder.py            # Shared encoder for feature extraction
├── configs/
│   ├── denoising.yaml        # Config for denoising task
│   ├── colorization.yaml     # Config for colorization task
│   └── context_pred.yaml     # Config for context prediction task
└── main_partB.py             # Main training script for Part B

```

## Three Proxy Tasks

### 1. Denoising (Simplest - Start Here)
- Add Gaussian noise to images
- Train encoder-decoder to reconstruct clean images
- Loss: MSE between predicted and original clean image

### 2. Colorization
- Convert RGB images to grayscale (L channel)
- Train network to predict RGB colors from grayscale
- Loss: MSE or perceptual loss on color channels

### 3. Context Prediction
- Mask random patches in the image
- Train network to predict missing content
- Loss: MSE on masked regions

## Clustering Pipeline

1. Train encoder-decoder on proxy task
2. Extract encoder features from all training tiles → shape (N, C, H', W')
3. Flatten to (N × H' × W', C) feature vectors
4. Apply clustering (KMeans, DBSCAN, etc.)
5. For test tiles:
   - Extract features
   - Assign cluster labels
   - Reshape to (H', W')
   - Upscale to original resolution (H, W)
   - Stitch tiles → full pseudo-segmentation mask
6. Compare with ground truth land-use labels

## Usage

```bash
# Train denoising task
python main_partB.py --task denoising --config configs/denoising.yaml

# Train colorization task
python main_partB.py --task colorization --config configs/colorization.yaml

# Train context prediction task
python main_partB.py --task context_prediction --config configs/context_pred.yaml

# Perform clustering and visualization
python clustering/cluster_features.py --exp denoising_exp --n_clusters 5
python clustering/visualize_clusters.py --exp denoising_exp --n_clusters 5
```

## Key Implementation Notes

- **Padding**: Use "reflect" or "same" padding to avoid border artifacts during clustering
- **Avoid zero-padding**: Zeros at borders should not be treated as a cluster class
- **Code clarity**: All code follows clear structure with detailed comments
- **Separation from Part A**: This folder is independent from Part A work
