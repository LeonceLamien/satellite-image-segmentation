"""
PART B - CLUSTERING ON ENCODER FEATURES
========================================

Extract features from trained encoder and apply clustering algorithm.

Pipeline:
    1. Load trained model from proxy task
    2. Extract encoder features from all training tiles
       - Shape: (N_train, C, H', W') where H' = H/8, W' = W/8
    3. Flatten features: (N_train × H' × W', C)
    4. Apply clustering (KMeans, DBSCAN, etc.)
    5. For test tiles:
       - Extract features
       - Assign cluster labels
       - Reshape to (H', W')
       - Upscale to original resolution (H, W)
       - Stitch tiles together
    6. Visualize pseudo-segmentation mask

Usage:
    python clustering/cluster_features.py --exp denoising_exp --n_clusters 5 --algorithm kmeans
"""

import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import yaml
import os
import argparse
import pickle
import sys

sys.path.append('..')

from proxy_tasks.denoising import DenoisingDataset, DenoisingAutoencoder
from proxy_tasks.colorization import ColorizationDataset, ColorizationAutoencoder
from proxy_tasks.context_prediction import ContextPredictionDataset, ContextPredictionAutoencoder


class FeatureClusterer:
    """
    Extract features from encoder and apply clustering.
    """
    
    def __init__(self, exp_dir, task_name, n_clusters=5, algorithm='kmeans', device='cpu'):
        """
        Args:
            exp_dir: Path to experiment directory (e.g., "Results/denoising_exp")
            task_name: "denoising", "colorization", or "context_prediction"
            n_clusters: Number of clusters
            algorithm: "kmeans", "dbscan", or "agglomerative"
            device: "cpu" or "cuda"
        """
        self.exp_dir = exp_dir
        self.task_name = task_name
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.device = device
        
        # Load config and model
        self._load_config()
        self._load_model()
        
        # Initialize clustering algorithm
        self._init_clustering()
        
        # Create output directory
        self.cluster_dir = f"{exp_dir}/_Clustering"
        os.makedirs(self.cluster_dir, exist_ok=True)
    
    def _load_config(self):
        """Load configuration from experiment directory."""
        config_path = f"{self.exp_dir}/config.yaml"
        
        # If config not saved, try to find it in configs/
        if not os.path.exists(config_path):
            config_path = f"configs/{self.task_name}.yaml"
        
        with open(config_path, 'r') as f:
            self.param = yaml.safe_load(f)
        
        print(f"✓ Config loaded from: {config_path}")
    
    def _load_model(self):
        """Load trained model."""
        weights_path = f"{self.exp_dir}/_Weights/best_model.pth"
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        # Create model
        if self.task_name == "denoising":
            self.model = DenoisingAutoencoder(self.param)
        elif self.task_name == "colorization":
            self.model = ColorizationAutoencoder(self.param)
        elif self.task_name == "context_prediction":
            self.model = ContextPredictionAutoencoder(self.param)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
        
        # Load weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from: {weights_path}")
    
    def _init_clustering(self):
        """Initialize clustering algorithm."""
        if self.algorithm == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.algorithm == 'dbscan':
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        elif self.algorithm == 'agglomerative':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        print(f"✓ Clustering: {self.algorithm} (n_clusters={self.n_clusters})")
    
    def extract_features(self, dataset, set_name="train"):
        """
        Extract encoder features from dataset.
        
        Args:
            dataset: Dataset instance
            set_name: "train", "val", or "test"
        
        Returns:
            features_flat: Flattened features (N × H' × W', C)
            feature_maps: Original feature maps (N, C, H', W')
            tile_names: List of tile names
        """
        print(f"\n[Extracting features from {set_name} set...]")
        
        all_features = []
        all_names = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                # Get image (first element of tuple)
                if self.task_name in ["denoising", "colorization"]:
                    img, _, name = dataset[idx]
                else:  # context_prediction
                    img, _, _, name = dataset[idx]
                
                img = img.unsqueeze(0).to(self.device)
                
                # Extract encoder features
                features = self.model.get_encoder_features(img)  # (1, C, H', W')
                
                all_features.append(features.cpu().numpy())
                all_names.append(name)
                
                if (idx + 1) % 200 == 0:
                    print(f"  Processed {idx+1}/{len(dataset)} images...")
        
        # Stack features: (N, C, H', W')
        feature_maps = np.concatenate(all_features, axis=0)
        
        # Flatten spatial dimensions: (N, C, H', W') -> (N × H' × W', C)
        N, C, H, W = feature_maps.shape
        features_flat = feature_maps.transpose(0, 2, 3, 1).reshape(-1, C)
        
        print(f"✓ Features extracted:")
        print(f"  Feature maps shape: {feature_maps.shape}")
        print(f"  Flattened shape: {features_flat.shape}")
        
        return features_flat, feature_maps, all_names
    
    def fit_clustering(self, features_flat):
        """
        Fit clustering algorithm on features.
        
        Args:
            features_flat: Flattened features (N × H' × W', C)
        
        Returns:
            labels: Cluster labels (N × H' × W',)
        """
        print("\n[Fitting clustering algorithm...]")
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_flat)
        
        # Fit clustering
        labels = self.clusterer.fit_predict(features_scaled)
        
        # Print cluster statistics
        unique_labels = np.unique(labels)
        print(f"✓ Clustering fitted:")
        print(f"  Number of clusters: {len(unique_labels)}")
        for label in unique_labels:
            count = np.sum(labels == label)
            percentage = count / len(labels) * 100
            print(f"    Cluster {label}: {count} pixels ({percentage:.2f}%)")
        
        return labels
    
    def predict_clustering(self, features_flat):
        """
        Predict cluster labels for new features.
        
        Args:
            features_flat: Flattened features (M × H' × W', C)
        
        Returns:
            labels: Cluster labels (M × H' × W',)
        """
        # Standardize features
        features_scaled = self.scaler.transform(features_flat)
        
        # Predict labels
        if self.algorithm == 'kmeans':
            labels = self.clusterer.predict(features_scaled)
        elif self.algorithm in ['dbscan', 'agglomerative']:
            # These don't have predict(), need to refit
            # For simplicity, we'll refit (not ideal for large datasets)
            labels = self.clusterer.fit_predict(features_scaled)
        
        return labels
    
    def save_clustering(self):
        """Save clustering model and scaler."""
        save_path = f"{self.cluster_dir}/clustering_{self.algorithm}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'clusterer': self.clusterer,
                'scaler': self.scaler,
                'n_clusters': self.n_clusters,
                'algorithm': self.algorithm
            }, f)
        
        print(f"✓ Clustering saved to: {save_path}")
    
    def load_clustering(self):
        """Load clustering model and scaler."""
        load_path = f"{self.cluster_dir}/clustering_{self.algorithm}.pkl"
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.clusterer = data['clusterer']
        self.scaler = data['scaler']
        self.n_clusters = data['n_clusters']
        
        print(f"✓ Clustering loaded from: {load_path}")
    
    def run(self):
        """Complete clustering pipeline."""
        print("\n" + "=" * 70)
        print(f"CLUSTERING PIPELINE: {self.task_name.upper()}")
        print("=" * 70)
        
        # Load datasets
        img_dir = "./Dataset/images"
        
        if self.task_name == "denoising":
            train_dataset = DenoisingDataset(img_dir, "train", self.param, 
                                            noise_std=0.1)
            test_dataset = DenoisingDataset(img_dir, "test", self.param,
                                           noise_std=0.1)
        elif self.task_name == "colorization":
            train_dataset = ColorizationDataset(img_dir, "train", self.param)
            test_dataset = ColorizationDataset(img_dir, "test", self.param)
        elif self.task_name == "context_prediction":
            train_dataset = ContextPredictionDataset(img_dir, "train", self.param,
                                                    mask_ratio=0.3, patch_size=8)
            test_dataset = ContextPredictionDataset(img_dir, "test", self.param,
                                                   mask_ratio=0.3, patch_size=8)
        
        # Extract training features
        train_features, train_maps, train_names = self.extract_features(
            train_dataset, "train"
        )
        
        # Fit clustering
        train_labels = self.fit_clustering(train_features)
        
        # Save clustering
        self.save_clustering()
        
        # Save train results
        np.save(f"{self.cluster_dir}/train_labels.npy", train_labels)
        np.save(f"{self.cluster_dir}/train_feature_maps.npy", train_maps)
        with open(f"{self.cluster_dir}/train_names.txt", 'w') as f:
            f.write('\n'.join(train_names))
        
        # Extract test features
        test_features, test_maps, test_names = self.extract_features(
            test_dataset, "test"
        )
        
        # Predict test labels
        test_labels = self.predict_clustering(test_features)
        
        # Save test results
        np.save(f"{self.cluster_dir}/test_labels.npy", test_labels)
        np.save(f"{self.cluster_dir}/test_feature_maps.npy", test_maps)
        with open(f"{self.cluster_dir}/test_names.txt", 'w') as f:
            f.write('\n'.join(test_names))
        
        print("\n" + "=" * 70)
        print("Clustering completed!")
        print(f"Results saved in: {self.cluster_dir}/")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Cluster encoder features')
    parser.add_argument('--exp', type=str, required=True,
                       help='Experiment directory (e.g., Results/denoising_exp)')
    parser.add_argument('--task', type=str, required=True,
                       choices=['denoising', 'colorization', 'context_prediction'],
                       help='Proxy task name')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters')
    parser.add_argument('--algorithm', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'agglomerative'],
                       help='Clustering algorithm')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check if experiment directory exists
    if not os.path.exists(args.exp):
        print(f"❌ Experiment directory not found: {args.exp}")
        return
    
    # Run clustering
    clusterer = FeatureClusterer(
        args.exp, 
        args.task,
        args.n_clusters,
        args.algorithm,
        args.device
    )
    clusterer.run()


if __name__ == "__main__":
    main()
