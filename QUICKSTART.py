"""
QUICK START GUIDE - PART B
===========================

Complete workflow for self-supervised learning and clustering.

STEP 1: Test the implementations
---------------------------------
cd PartB

# Test denoising task (simplest, start here!)
python proxy_tasks/denoising.py

# Test colorization task
python proxy_tasks/colorization.py

# Test context prediction task
python proxy_tasks/context_prediction.py


STEP 2: Train on proxy task
----------------------------
# Train denoising (recommended to start)
python main_partB.py --task denoising --config configs/denoising.yaml --device cpu

# Train colorization
python main_partB.py --task colorization --config configs/colorization.yaml --device cpu

# Train context prediction
python main_partB.py --task context_prediction --config configs/context_pred.yaml --device cpu

# With GPU (if available)
python main_partB.py --task denoising --config configs/denoising.yaml --device cuda


STEP 3: Extract features and cluster
-------------------------------------
# After training, run clustering on encoder features
python clustering/cluster_features.py \\
    --exp Results/denoising_exp \\
    --task denoising \\
    --n_clusters 5 \\
    --algorithm kmeans \\
    --device cpu

# Try different numbers of clusters
python clustering/cluster_features.py --exp Results/denoising_exp --task denoising --n_clusters 8 --algorithm kmeans

# Try different algorithms
python clustering/cluster_features.py --exp Results/denoising_exp --task denoising --n_clusters 5 --algorithm agglomerative


STEP 4: Visualize and analyze results
--------------------------------------
# Visualize clustering results vs ground truth
python clustering/visualize_clusters.py \\
    --exp Results/denoising_exp \\
    --task denoising \\
    --n_clusters 5

# This will:
# - Reconstruct full pseudo-segmentation mask from tiles
# - Compare with ground truth segmentation
# - Analyze which clusters correspond to which land-use classes
# - Generate visualization figure


EXPECTED OUTPUTS
================

After training (STEP 2):
------------------------
Results/denoising_exp/
├── _Weights/
│   ├── best_model.pth           # Best model weights
│   └── checkpoint_latest.pth    # Latest checkpoint
└── _Figures/
    └── training_curves.png      # Training/validation loss curves


After clustering (STEP 3):
---------------------------
Results/denoising_exp/_Clustering/
├── clustering_kmeans.pkl        # Fitted clustering model
├── train_labels.npy            # Cluster labels for train set
├── train_feature_maps.npy      # Encoder features for train set
├── train_names.txt             # Tile names for train set
├── test_labels.npy             # Cluster labels for test set
├── test_feature_maps.npy       # Encoder features for test set
└── test_names.txt              # Tile names for test set


After visualization (STEP 4):
------------------------------
Results/denoising_exp/_Clustering/
├── clustering_vs_gt.png         # Comparison figure
├── full_cluster_mask.npy        # Full reconstructed pseudo-segmentation
└── full_gt_mask.npy             # Full ground truth mask


TIPS FOR YOUR REPORT
=====================

1. Architecture Adaptations:
   - All models use reflect padding to avoid border artifacts
   - Encoder-decoder U-Net style with skip connections
   - Outputs depend on task (RGB for denoising/colorization, RGB for context pred)

2. Clustering Analysis Questions:
   - Do clusters correspond to ground truth classes?
   - Are there unexpected categories?
   - Which proxy task learns best features?
   - Compare clustering quality across different tasks

3. Key Metrics to Report:
   - Cluster purity (% of dominant GT class in each cluster)
   - Number of discovered clusters vs predefined classes
   - Visual quality of pseudo-segmentation

4. Comparisons:
   - Compare all 3 proxy tasks
   - Compare different clustering algorithms (KMeans, Agglomerative)
   - Compare different numbers of clusters (5, 8, 10, etc.)


TROUBLESHOOTING
===============

Problem: "Config file not found"
Solution: Make sure you're in the PartB/ directory

Problem: "Dataset not found"
Solution: The code expects Dataset/ folder in parent directory (../)
          Make sure PartB/ is inside Image_processing/

Problem: "Out of memory"
Solution: Reduce BATCH_SIZE in config file or use device=cpu

Problem: Border artifacts in clustering
Solution: The code already uses 'reflect' padding, but you can adjust
          BASE_CHANNELS in config to reduce model size


EXPERIMENT IDEAS
================

1. Vary noise level in denoising:
   Edit configs/denoising.yaml → NOISE_STD: 0.05, 0.1, 0.2

2. Vary mask ratio in context prediction:
   Edit configs/context_pred.yaml → MASK_RATIO: 0.2, 0.3, 0.5

3. Train longer or shorter:
   Edit config → NUM_EPOCHS: 30, 50, 100

4. Try different cluster numbers:
   --n_clusters 3, 5, 8, 10, 15

5. Combine features from multiple tasks:
   Extract features from all 3 tasks, concatenate, then cluster
"""

if __name__ == "__main__":
    print(__doc__)
