"""
Analyze clustering quality and create remapped visualization.

This script:
1. Loads clustering results and ground truth
2. Computes cluster purity and correspondence
3. Creates remapped pseudo-segmentation where clusters are reordered by their dominant class
4. Computes metrics (purity, ARI, NMI)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Ground truth colormap
GT_COLORMAP = ListedColormap([
    "black",   # 0 = other
    "blue",    # 1 = water
    "red",     # 2 = building
    "yellow",  # 3 = farmland
    "green"    # 4 = forest
])

CLASS_NAMES = ['Other', 'Water', 'Building', 'Farmland', 'Forest']


def load_masks(exp_dir):
    """Load clustering and ground truth masks."""
    cluster_mask = np.load(f"{exp_dir}/_Clustering/full_cluster_mask.npy")
    gt_mask = np.load(f"{exp_dir}/_Clustering/full_gt_mask.npy")
    return cluster_mask, gt_mask


def compute_cluster_to_class_mapping(cluster_mask, gt_mask, n_clusters):
    """
    Find best mapping from clusters to ground truth classes.
    Returns mapping dict: cluster_id -> dominant_gt_class
    """
    mapping = {}
    
    for cluster_id in range(n_clusters):
        cluster_pixels = (cluster_mask == cluster_id)
        gt_in_cluster = gt_mask[cluster_pixels]
        
        if len(gt_in_cluster) == 0:
            mapping[cluster_id] = 0  # Default to "Other"
            continue
        
        unique, counts = np.unique(gt_in_cluster, return_counts=True)
        dominant_class = unique[np.argmax(counts)]
        mapping[cluster_id] = dominant_class
    
    return mapping


def remap_clusters(cluster_mask, mapping):
    """
    Remap cluster IDs to their dominant ground truth class.
    """
    remapped = np.zeros_like(cluster_mask)
    for cluster_id, gt_class in mapping.items():
        remapped[cluster_mask == cluster_id] = gt_class
    return remapped


def compute_metrics(cluster_mask, gt_mask, n_clusters):
    """
    Compute clustering quality metrics.
    
    Returns:
        purity: Overall purity score
        ari: Adjusted Rand Index
        nmi: Normalized Mutual Information
    """
    # Flatten masks
    cluster_flat = cluster_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Purity
    purity = 0
    for cluster_id in range(n_clusters):
        cluster_pixels = (cluster_flat == cluster_id)
        if cluster_pixels.sum() == 0:
            continue
        gt_in_cluster = gt_flat[cluster_pixels]
        unique, counts = np.unique(gt_in_cluster, return_counts=True)
        purity += counts.max()
    purity /= len(cluster_flat)
    
    # ARI and NMI
    ari = adjusted_rand_score(gt_flat, cluster_flat)
    nmi = normalized_mutual_info_score(gt_flat, cluster_flat)
    
    return purity, ari, nmi


def analyze_per_class_recall(cluster_mask, gt_mask, mapping, n_clusters):
    """
    Compute per-class recall: what % of each GT class is captured by clusters.
    """
    recalls = {}
    
    for gt_class in range(5):
        gt_pixels = (gt_mask == gt_class)
        total = gt_pixels.sum()
        
        if total == 0:
            recalls[gt_class] = 0.0
            continue
        
        # Find clusters mapped to this class
        mapped_clusters = [c for c, g in mapping.items() if g == gt_class]
        
        if len(mapped_clusters) == 0:
            recalls[gt_class] = 0.0
            continue
        
        # Count how many GT pixels are in these clusters
        captured = 0
        for cluster_id in mapped_clusters:
            cluster_pixels = (cluster_mask == cluster_id)
            overlap = (gt_pixels & cluster_pixels).sum()
            captured += overlap
        
        recalls[gt_class] = captured / total * 100
    
    return recalls


def visualize_analysis(cluster_mask, gt_mask, remapped_mask, n_clusters, 
                       mapping, purity, ari, nmi, recalls, save_path):
    """
    Create comprehensive visualization with 3 panels + metrics.
    """
    fig = plt.figure(figsize=(24, 8))
    
    # Create colormap for clusters
    if n_clusters <= 5:
        cluster_cmap = GT_COLORMAP
    else:
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
    
    # Panel 1: Original clustering
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(cluster_mask, cmap=cluster_cmap, vmin=0, vmax=max(n_clusters-1, 4))
    ax1.set_title('Original Clustering\n(Raw cluster IDs)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Remapped clustering
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(remapped_mask, cmap=GT_COLORMAP, vmin=0, vmax=4)
    ax2.set_title('Remapped Pseudo-Segmentation\n(Clusters → Dominant class)', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Ground truth
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(gt_mask, cmap=GT_COLORMAP, vmin=0, vmax=4)
    ax3.set_title('Ground Truth Segmentation', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add metrics text
    metrics_text = f"""
    CLUSTERING QUALITY METRICS
    
    Overall Purity: {purity:.3f}
    Adjusted Rand Index: {ari:.3f}
    Normalized Mutual Info: {nmi:.3f}
    
    CLUSTER → CLASS MAPPING:
    """
    for cluster_id in range(n_clusters):
        gt_class = mapping[cluster_id]
        metrics_text += f"\n    Cluster {cluster_id} → {CLASS_NAMES[gt_class]}"
    
    metrics_text += "\n\n    PER-CLASS RECALL:"
    for gt_class in range(5):
        recall = recalls[gt_class]
        metrics_text += f"\n    {CLASS_NAMES[gt_class]}: {recall:.1f}%"
    
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=11, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Clustering Quality Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analysis saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze clustering quality')
    parser.add_argument('--exp', type=str, required=True,
                        help='Experiment directory (e.g., Results/denoising_exp)')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters used')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CLUSTERING QUALITY ANALYSIS")
    print("=" * 70)
    
    # Load masks
    print("\n[1] Loading masks...")
    cluster_mask, gt_mask = load_masks(args.exp)
    print(f"✓ Cluster mask: {cluster_mask.shape}")
    print(f"✓ GT mask: {gt_mask.shape}")
    
    # Compute mapping
    print("\n[2] Computing cluster-to-class mapping...")
    mapping = compute_cluster_to_class_mapping(cluster_mask, gt_mask, args.n_clusters)
    for cluster_id, gt_class in mapping.items():
        print(f"  Cluster {cluster_id} → {CLASS_NAMES[gt_class]}")
    
    # Remap clusters
    print("\n[3] Remapping clusters to dominant classes...")
    remapped_mask = remap_clusters(cluster_mask, mapping)
    
    # Compute metrics
    print("\n[4] Computing quality metrics...")
    purity, ari, nmi = compute_metrics(cluster_mask, gt_mask, args.n_clusters)
    print(f"  Purity: {purity:.3f}")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Normalized Mutual Info: {nmi:.3f}")
    
    # Per-class recall
    print("\n[5] Computing per-class recall...")
    recalls = analyze_per_class_recall(cluster_mask, gt_mask, mapping, args.n_clusters)
    for gt_class in range(5):
        print(f"  {CLASS_NAMES[gt_class]}: {recalls[gt_class]:.1f}%")
    
    # Visualize
    print("\n[6] Creating visualization...")
    figures_dir = os.path.join(args.exp, '_Figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, f'clustering_analysis_n{args.n_clusters}.png')
    
    visualize_analysis(cluster_mask, gt_mask, remapped_mask, args.n_clusters,
                      mapping, purity, ari, nmi, recalls, save_path)
    
    print("\n" + "=" * 70)
    print("✅ Analysis completed!")
    print("=" * 70)
    print(f"\n💡 Interpretation:")
    print(f"   • Purity {purity:.3f}: {'Good' if purity > 0.6 else 'Moderate' if purity > 0.4 else 'Poor'} cluster homogeneity")
    print(f"   • ARI {ari:.3f}: {'Strong' if ari > 0.3 else 'Moderate' if ari > 0.1 else 'Weak'} agreement with GT")
    print(f"   • NMI {nmi:.3f}: {'High' if nmi > 0.4 else 'Moderate' if nmi > 0.2 else 'Low'} mutual information")
    print(f"\n💡 Remember: Self-supervised clustering won't perfectly match supervised segmentation!")
    print(f"   The goal is to learn useful features, not to replicate ground truth.\n")


if __name__ == "__main__":
    main()
