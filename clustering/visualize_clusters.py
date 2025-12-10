"""
PART B - VISUALIZE CLUSTERING RESULTS
======================================

Visualize pseudo-segmentation masks from clustering and compare with ground truth.

Pipeline:
    1. Load cluster labels for test tiles
    2. Reshape labels to 2D grids (H', W')
    3. Upscale to original tile resolution (H, W)
    4. Stitch tiles together to create full pseudo-segmentation mask
    5. Compare with ground truth segmentation
    6. Analyze: Do clusters correspond to land-use classes?

Usage:
    python clustering/visualize_clusters.py --exp Results/denoising_exp --task denoising
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import os
import re
import argparse


# Ground truth colormap (same as Part A)
GT_COLORMAP = ListedColormap([
    "black",   # 0 = other
    "blue",    # 1 = water
    "red",     # 2 = building
    "yellow",  # 3 = farmland
    "green"    # 4 = forest
])


def load_clustering_results(cluster_dir, set_name="test"):
    """
    Load clustering results.
    
    Args:
        cluster_dir: Path to clustering directory
        set_name: "train" or "test"
    
    Returns:
        labels: Cluster labels (N × H' × W',)
        feature_maps: Feature maps (N, C, H', W')
        tile_names: List of tile names
    """
    labels = np.load(f"{cluster_dir}/{set_name}_labels.npy")
    feature_maps = np.load(f"{cluster_dir}/{set_name}_feature_maps.npy")
    
    with open(f"{cluster_dir}/{set_name}_names.txt", 'r') as f:
        tile_names = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded clustering results:")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Feature maps shape: {feature_maps.shape}")
    print(f"  Number of tiles: {len(tile_names)}")
    
    return labels, feature_maps, tile_names


def reshape_labels_to_tiles(labels, feature_maps_shape):
    """
    Reshape flat labels to per-tile 2D grids.
    
    Args:
        labels: Flat labels (N × H' × W',)
        feature_maps_shape: Shape of feature maps (N, C, H', W')
    
    Returns:
        tile_labels: List of 2D label grids, each (H', W')
    """
    N, C, H, W = feature_maps_shape
    
    # Reshape labels
    tile_labels = labels.reshape(N, H, W)
    
    return tile_labels


def upscale_labels(label_grid, target_height, target_width):
    """
    Upscale label grid to target resolution.
    
    Args:
        label_grid: 2D label grid (H', W')
        target_height, target_width: Target dimensions
    
    Returns:
        upscaled: Upscaled labels (H, W)
    """
    # Use nearest neighbor interpolation to preserve discrete labels
    upscaled = cv2.resize(
        label_grid.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return upscaled.astype(np.int32)


def reconstruct_full_image(tile_labels_list, tile_names, target_size=(64, 64)):
    """
    Stitch upscaled tiles together to create full pseudo-segmentation mask.
    
    Args:
        tile_labels_list: List of upscaled tile labels, each (H, W)
        tile_names: List of tile names (format: "tile_xx_yy.png")
        target_size: Target size for each tile (H, W)
    
    Returns:
        full_mask: Full pseudo-segmentation mask
        coords: Tile coordinates
    """
    # Parse tile coordinates
    coords = []
    for name in tile_names:
        m = re.search(r"tile_(\d+)_(\d+)\.png", name)
        if not m:
            continue
        xx, yy = map(int, m.groups())
        coords.append((xx, yy))
    
    coords = np.array(coords)
    
    if len(coords) == 0:
        print("⚠️  No valid tile coordinates found")
        return None, None
    
    # Determine full image dimensions
    minX, minY = coords.min(axis=0)
    maxX, maxY = coords.max(axis=0)
    
    tileH, tileW = target_size
    fullH = (maxX - minX + 1) * tileH
    fullW = (maxY - minY + 1) * tileW
    
    # Allocate full mask
    full_mask = np.zeros((fullH, fullW), dtype=np.int32)
    
    # Stitch tiles
    for tile_labels, (xx, yy) in zip(tile_labels_list, coords):
        i = xx - minX
        j = yy - minY
        y0, y1 = i * tileH, (i + 1) * tileH
        x0, x1 = j * tileW, (j + 1) * tileW
        
        full_mask[y0:y1, x0:x1] = tile_labels
    
    print(f"✓ Full mask reconstructed: {full_mask.shape}")
    
    return full_mask, coords


def load_ground_truth_mask(mask_dir, tile_names, coords, target_size=(64, 64)):
    """
    Load and reconstruct ground truth mask from tiles.
    
    Args:
        mask_dir: Path to mask directory
        tile_names: List of tile names
        coords: Tile coordinates
        target_size: Target size for each tile
    
    Returns:
        full_gt_mask: Full ground truth mask
    """
    minX, minY = coords.min(axis=0)
    maxX, maxY = coords.max(axis=0)
    
    tileH, tileW = target_size
    fullH = (maxX - minX + 1) * tileH
    fullW = (maxY - minY + 1) * tileW
    
    full_gt_mask = np.zeros((fullH, fullW), dtype=np.int32)
    
    # Load and stitch ground truth tiles
    for tile_name, (xx, yy) in zip(tile_names, coords):
        mask_path = os.path.join(mask_dir, tile_name)
        
        if not os.path.exists(mask_path):
            continue
        
        mask_tile = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_tile = cv2.resize(mask_tile, target_size, interpolation=cv2.INTER_NEAREST)
        
        i = xx - minX
        j = yy - minY
        y0, y1 = i * tileH, (i + 1) * tileH
        x0, x1 = j * tileW, (j + 1) * tileW
        
        full_gt_mask[y0:y1, x0:x1] = mask_tile
    
    print(f"✓ Ground truth mask loaded: {full_gt_mask.shape}")
    
    return full_gt_mask


def analyze_cluster_correspondence(cluster_mask, gt_mask, n_clusters):
    """
    Analyze which clusters correspond to which ground truth classes.
    
    Args:
        cluster_mask: Pseudo-segmentation from clustering (H, W)
        gt_mask: Ground truth segmentation (H, W)
        n_clusters: Number of clusters
    
    Returns:
        correspondence: Dictionary mapping clusters to GT classes
    """
    print("\n" + "=" * 70)
    print("CLUSTER-TO-CLASS CORRESPONDENCE ANALYSIS")
    print("=" * 70)
    
    correspondence = {}
    
    for cluster_id in range(n_clusters):
        # Find pixels belonging to this cluster
        cluster_pixels = (cluster_mask == cluster_id)
        
        if cluster_pixels.sum() == 0:
            continue
        
        # Get ground truth labels for these pixels
        gt_labels_in_cluster = gt_mask[cluster_pixels]
        
        # Count occurrences of each GT class
        unique, counts = np.unique(gt_labels_in_cluster, return_counts=True)
        
        # Find dominant GT class
        dominant_idx = np.argmax(counts)
        dominant_class = unique[dominant_idx]
        dominant_percentage = counts[dominant_idx] / counts.sum() * 100
        
        correspondence[cluster_id] = {
            'dominant_class': dominant_class,
            'percentage': dominant_percentage,
            'distribution': dict(zip(unique.tolist(), counts.tolist()))
        }
        
        # Print analysis
        print(f"\nCluster {cluster_id}:")
        print(f"  Total pixels: {cluster_pixels.sum()}")
        print(f"  Dominant GT class: {dominant_class} ({dominant_percentage:.1f}%)")
        print(f"  Distribution:")
        for cls, count in zip(unique, counts):
            pct = count / counts.sum() * 100
            class_names = ['Other', 'Water', 'Building', 'Farmland', 'Forest']
            print(f"    Class {cls} ({class_names[cls]}): {count} pixels ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    
    return correspondence


def visualize_comparison(cluster_mask, gt_mask, n_clusters, save_path):
    """
    Create side-by-side comparison visualization.
    
    Args:
        cluster_mask: Pseudo-segmentation from clustering
        gt_mask: Ground truth segmentation
        n_clusters: Number of clusters
        save_path: Path to save figure
    """
    # Use same colormap as ground truth for better comparison
    # If n_clusters <= 5, use GT colors directly
    # Otherwise extend with additional colors from tab10
    if n_clusters <= 5:
        cluster_cmap = GT_COLORMAP
    else:
        # Start with GT colors, add more if needed
        gt_colors = np.array([
            [0.0, 0.0, 0.0, 1.0],      # black
            [0.0, 0.0, 1.0, 1.0],      # blue
            [1.0, 0.0, 0.0, 1.0],      # red
            [1.0, 1.0, 0.0, 1.0],      # yellow
            [0.0, 0.5, 0.0, 1.0],      # green
        ])
        extra_colors = plt.cm.tab10(np.linspace(0.5, 1, n_clusters - 5))
        all_colors = np.vstack([gt_colors, extra_colors])
        cluster_cmap = ListedColormap(all_colors)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pseudo-segmentation from clustering
    im1 = axes[0].imshow(cluster_mask, cmap=cluster_cmap, vmin=0, vmax=max(n_clusters-1, 4))
    axes[0].set_title('Pseudo-Segmentation (Clustering)', 
                     fontsize=18, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # Ground truth segmentation
    im2 = axes[1].imshow(gt_mask, cmap=GT_COLORMAP, vmin=0, vmax=4)
    axes[1].set_title('Ground Truth Segmentation', 
                     fontsize=18, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    plt.suptitle('Clustering Results vs Ground Truth', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize clustering results')
    parser.add_argument('--exp', type=str, required=True,
                       help='Experiment directory (e.g., Results/denoising_exp)')
    parser.add_argument('--task', type=str, required=True,
                       choices=['denoising', 'colorization', 'context_prediction'],
                       help='Proxy task name')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters')
    
    args = parser.parse_args()
    
    cluster_dir = f"{args.exp}/_Clustering"
    
    if not os.path.exists(cluster_dir):
        print(f"Clustering directory not found: {cluster_dir}")
        print("   Run cluster_features.py first!")
        return
    
    print("\n" + "=" * 70)
    print(f"VISUALIZING CLUSTERING RESULTS: {args.task.upper()}")
    print("=" * 70)
    
    # Load clustering results
    labels, feature_maps, tile_names = load_clustering_results(cluster_dir, "test")
    
    # Reshape labels to tiles
    tile_labels = reshape_labels_to_tiles(labels, feature_maps.shape)
    
    print(f"\n✓ Reshaped labels to tiles: {len(tile_labels)} tiles of shape {tile_labels[0].shape}")
    
    # Upscale each tile
    target_size = (64, 64)
    upscaled_tiles = [upscale_labels(tile, *target_size) for tile in tile_labels]
    
    print(f"✓ Upscaled tiles to {target_size}")
    
    # Reconstruct full mask
    full_cluster_mask, coords = reconstruct_full_image(upscaled_tiles, tile_names, target_size)
    
    if full_cluster_mask is None:
        print("Failed to reconstruct full mask")
        return
    
    # Load ground truth
    mask_dir = "./Dataset/annotations"
    full_gt_mask = load_ground_truth_mask(mask_dir, tile_names, coords, target_size)
    
    # Analyze correspondence
    correspondence = analyze_cluster_correspondence(
        full_cluster_mask, full_gt_mask, args.n_clusters
    )
    
    # Create _Figures directory
    figures_dir = os.path.join(os.path.dirname(cluster_dir), '_Figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Visualize
    save_path = os.path.join(figures_dir, f"clustering_vs_gt_n{args.n_clusters}.png")
    visualize_comparison(full_cluster_mask, full_gt_mask, args.n_clusters, save_path)
    
    # Save full masks
    np.save(f"{cluster_dir}/full_cluster_mask.npy", full_cluster_mask)
    np.save(f"{cluster_dir}/full_gt_mask.npy", full_gt_mask)
    
    print("\n" + "=" * 70)
    print("Visualization completed!")
    print(f"Results saved in: {cluster_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
