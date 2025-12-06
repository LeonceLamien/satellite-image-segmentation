
#!/usr/bin/env python
"""
Script pour analyser les résultats du clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_cluster_distribution(results_path):
    """Visualiser la distribution des clusters"""
    clustering_path = os.path.join(results_path, "Clustering")
    
    if not os.path.exists(clustering_path):
        print(f"Clustering results not found in {clustering_path}")
        return
    
    cluster_analysis_file = os.path.join(clustering_path, "cluster_analysis.txt")
    if os.path.exists(cluster_analysis_file):
        with open(cluster_analysis_file, 'r') as f:
            content = f.read()
            print("Cluster Analysis Summary:")
            print("=" * 60)
            print(content)
    
    test_analysis_file = os.path.join(clustering_path, "test_cluster_analysis.txt")
    if os.path.exists(test_analysis_file):
        with open(test_analysis_file, 'r') as f:
            lines = f.readlines()
            
        purities = []
        sizes = []
        clusters = []
        
        for line in lines:
            if 'Cluster' in line and 'Purity' in line and 'Size' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        cluster = int(parts[1])
                        purity = float(parts[3])
                        size = int(parts[5])
                        purities.append(purity)
                        sizes.append(size)
                        clusters.append(cluster)
                    except:
                        pass
        
        if purities:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].bar(clusters, purities, color='skyblue')
            axes[0].set_xlabel('Cluster ID')
            axes[0].set_ylabel('Purity')
            axes[0].set_title('Cluster Purity Distribution')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].bar(clusters, sizes, color='lightcoral')
            axes[1].set_xlabel('Cluster ID')
            axes[1].set_ylabel('Size (number of pixels)')
            axes[1].set_title('Cluster Size Distribution')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(clustering_path, "cluster_distributions.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nCluster distributions saved to: {os.path.join(clustering_path, 'cluster_distributions.png')}")

def compare_with_classes(results_path):
    """Comparer les clusters avec les classes réelles"""
    clustering_path = os.path.join(results_path, "Clustering")
    
    maps_path = os.path.join(clustering_path, "Pseudo_Segmentation_Maps")
    if os.path.exists(maps_path):
        npy_files = [f for f in os.listdir(maps_path) if f.endswith('.npy')]
        
        if npy_files:
            print(f"\nFound {len(npy_files)//2} comparison files in {maps_path}")
            
            for i in range(min(3, len(npy_files)//2)):
                pseudo_file = os.path.join(maps_path, f"pseudo_mask_{i}.npy")
                true_file = os.path.join(maps_path, f"true_mask_{i}.npy")
                
                if os.path.exists(pseudo_file) and os.path.exists(true_file):
                    pseudo_mask = np.load(pseudo_file)
                    true_mask = np.load(true_file)
                    
                    print(f"\nSample {i}:")
                    print(f"  Pseudo mask shape: {pseudo_mask.shape}")
                    print(f"  True mask shape: {true_mask.shape}")
                    print(f"  Unique clusters in pseudo: {np.unique(pseudo_mask)}")
                    print(f"  Unique classes in true: {np.unique(true_mask)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--results-dir', type=str, default='Results', help='Results directory')
    
    args = parser.parse_args()
    
    results_path = os.path.join(args.results_dir, args.exp)
    
    if not os.path.exists(results_path):
        print(f"Results directory not found: {results_path}")
        return
    
    print(f"🔍 Analyzing clustering results for experiment: {args.exp}")
    print(f"📁 Results path: {results_path}")
    
    visualize_cluster_distribution(results_path)
    compare_with_classes(results_path)
    
    print("\nAnalysis completed!")

if __name__ == '__main__':
    main()
