# from Dataset.dataLoader import *
# from Dataset.makeGraph import *
# from Networks.Architectures.basicNetwork import *
# from Networks.Architectures.deepLab import *
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
        
# import numpy as np
# np.random.seed(2885)
# import os
# import copy
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt
# import seaborn as sns
        
# import torch
# torch.manual_seed(2885)
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# import torch.nn as nn
# import torch.optim


# from Dataset.dataLoader import *
# from Dataset.makeGraph import *
# from Networks.Architectures.basicNetwork import *
# from Networks.Architectures.deepLab import *
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
        
# import numpy as np
# np.random.seed(2885)
# import os
# import copy
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt
# import seaborn as sns
        
# import torch
# torch.manual_seed(2885)
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# import torch.nn as nn
# import torch.optim

# import cv2


# # --------------------------------------------------------------------------------
# # CREATE A FOLDER IF IT DOES NOT EXIST
# # INPUT: 
# #     - desiredPath (str): path to the folder to create
# # --------------------------------------------------------------------------------
# def createFolder(desiredPath): 
#     if not os.path.exists(desiredPath):
#         os.makedirs(desiredPath)


# def rgb_to_lab(rgb_tensor):
#     """Convert RGB tensor to LAB color space"""
#     # Convert tensor to numpy
#     rgb_np = rgb_tensor.permute(0, 2, 3, 1).numpy()
#     lab_images = []
    
#     for img in rgb_np:
#         # Convert to LAB
#         lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#         lab_images.append(lab)
    
#     lab_tensor = torch.tensor(np.array(lab_images), dtype=torch.float32).permute(0, 3, 1, 2)
#     return lab_tensor


# def lab_to_rgb(lab_tensor):
#     """Convert LAB tensor to RGB color space"""
#     lab_np = lab_tensor.permute(0, 2, 3, 1).numpy()
#     rgb_images = []
    
#     for img in lab_np:
#         rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2RGB)
#         rgb_images.append(rgb)
    
#     rgb_tensor = torch.tensor(np.array(rgb_images), dtype=torch.float32).permute(0, 3, 1, 2)
#     return rgb_tensor


# class ColorizationLoss(nn.Module):
#     """Loss function for colorization task"""
#     def __init__(self):
#         super(ColorizationLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()
        
#     def forward(self, pred, target):
#         # MSE loss for color accuracy
#         mse = self.mse_loss(pred, target)
#         # L1 loss for sharper colors
#         l1 = self.l1_loss(pred, target)
#         # Combined loss
#         return 0.7 * mse + 0.3 * l1


# ######################################################################################
# #
# # CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# # An instance of Network_Class has been created in the main.py file
# # 
# ######################################################################################

# class Network_Class: 
#     # --------------------------------------------------------------------------------
#     # INITIALISATION OF THE MODEL
#     # INPUTS: 
#     #     - param (dic): dictionnary containing the parameters defined in the 
#     #                    configuration (yaml) file
#     #     - imgDirectory (str): path to the folder containing the images 
#     #     - maskDirectory (str): path to the folder containing the masks
#     #     - resultsPath (str): path to the folder containing the results of the 
#     #                          experiement
#     # --------------------------------------------------------------------------------
#     def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
#         print("resultsPath", resultsPath)
#         # ----------------
#         # USEFUL VARIABLES 
#         # ----------------
#         self.imgDirectory  = imgDirectory
#         self.maskDirectory = maskDirectory
#         self.resultsPath   = resultsPath
#         self.epoch         = param["TRAINING"]["EPOCH"]
#         self.device        = param["TRAINING"]["DEVICE"]
#         self.lr            = param["TRAINING"]["LEARNING_RATE"]
#         self.batchSize     = param["TRAINING"]["BATCH_SIZE"]
#         self.task          = param["MODEL"].get("TASK", "segmentation")

#         # -----------------------------------
#         # NETWORK ARCHITECTURE INITIALISATION
#         # -----------------------------------
#         self.model = DeepLabV3Plus(param).to(self.device)

#         # -------------------
#         # TRAINING PARAMETERS
#         # -------------------
#         if self.task == "colorization":
#             self.criterion = ColorizationLoss()
#         else:
#             self.criterion = nn.CrossEntropyLoss()
            
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#         # ----------------------------------------------------
#         # DATASET INITIALISATION (from the dataLoader.py file)
#         # ----------------------------------------------------
#         self.dataSetTrain    = LLNDataset(imgDirectory, maskDirectory, "train", param)
#         self.dataSetVal      = LLNDataset(imgDirectory, maskDirectory, "val",   param)
#         self.dataSetTest     = LLNDataset(imgDirectory, maskDirectory, "test",  param)
#         self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
#         self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
#         self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)

#         # Pour suivre les métriques
#         self.train_losses = []
#         self.val_losses = []

#     # ---------------------------------------------------------------------------
#     # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
#     # ---------------------------------------------------------------------------
#     def loadWeights(self): 
#         self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only=True))

#     # -----------------------------------
#     # TRAINING LOOP (implémentation réelle)
#     # -----------------------------------
#     def train(self): 
#         best_loss = float('inf')
        
#         # train for a given number of epochs
#         for epoch in range(self.epoch):
#             # Training phase
#             self.model.train()
#             running_loss = 0.0
            
#             for batch_idx, (images, masks, _, _) in enumerate(self.trainDataLoader):
#                 if self.task == "colorization":
#                     # Pour la colorisation: utiliser l'image comme target
#                     images = images.to(self.device)
#                     targets = images  # Target est l'image couleur originale
                    
#                     # Forward pass
#                     self.optimizer.zero_grad()
#                     outputs = self.model(images)
#                     loss = self.criterion(outputs, targets)
#                 else:
#                     # Pour la segmentation
#                     images = images.to(self.device)
#                     masks = masks.to(self.device).long()
                    
#                     # Forward pass
#                     self.optimizer.zero_grad()
#                     outputs = self.model(images)
#                     loss = self.criterion(outputs, masks)
                
#                 # Backward pass
#                 loss.backward()
#                 self.optimizer.step()
                
#                 running_loss += loss.item()
            
#             avg_train_loss = running_loss / len(self.trainDataLoader)
#             self.train_losses.append(avg_train_loss)
            
#             # Validation phase
#             val_loss = self.validate()
#             self.val_losses.append(val_loss)
            
#             print(f"Epoch [{epoch+1}/{self.epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 modelWts = copy.deepcopy(self.model.state_dict())
        
#         # Print learning curves
#         self.plot_learning_curves()
        
#         # Save the model weights
#         wghtsPath  = self.resultsPath + '/_Weights/'
#         createFolder(wghtsPath)
#         torch.save(modelWts, wghtsPath + '/wghts.pkl')
#         print("Training completed and weights saved!")


from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.deepLab import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
        
import numpy as np
np.random.seed(2885)
import os
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
        
import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim

import cv2

# Import pour le clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


def rgb_to_lab(rgb_tensor):
    """Convert RGB tensor to LAB color space"""
    # Convert tensor to numpy
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).numpy()
    lab_images = []
    
    for img in rgb_np:
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_images.append(lab)
    
    lab_tensor = torch.tensor(np.array(lab_images), dtype=torch.float32).permute(0, 3, 1, 2)
    return lab_tensor


def lab_to_rgb(lab_tensor):
    """Convert LAB tensor to RGB color space"""
    lab_np = lab_tensor.permute(0, 2, 3, 1).numpy()
    rgb_images = []
    
    for img in lab_np:
        rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2RGB)
        rgb_images.append(rgb)
    
    rgb_tensor = torch.tensor(np.array(rgb_images), dtype=torch.float32).permute(0, 3, 1, 2)
    return rgb_tensor


class ColorizationLoss(nn.Module):
    """Loss function for colorization task"""
    def __init__(self):
        super(ColorizationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        return 0.7 * mse + 0.3 * l1


class ClusteringModule:
    """Module pour le clustering des features apprises"""
    def __init__(self, param):
        self.n_clusters = param.get("CLUSTERING", {}).get("N_CLUSTERS", 8)
        self.method = param.get("CLUSTERING", {}).get("METHOD", "kmeans")
        self.random_state = param.get("CLUSTERING", {}).get("RANDOM_STATE", 2885)
        self.cluster_model = None
        self.pca = None
        
    def fit(self, features):
        """Entraîner le modèle de clustering sur les features"""
        print(f"\n🎯 Fitting {self.method} clustering on {len(features)} feature vectors...")
        
        # Réduire la dimensionnalité avec PCA si nécessaire
        if features.shape[1] > 50:
            print(f"Reducing dimensionality from {features.shape[1]} to 50 with PCA...")
            self.pca = PCA(n_components=50, random_state=self.random_state)
            features_reduced = self.pca.fit_transform(features)
        else:
            features_reduced = features
        
        # Appliquer l'algorithme de clustering
        if self.method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        
        self.cluster_model.fit(features_reduced)
        
        # Calculer les métriques de qualité du clustering
        if self.method != "dbscan":
            labels = self.cluster_model.labels_
            n_clusters_found = len(np.unique(labels[labels != -1])) if self.method == "dbscan" else self.n_clusters
            
            if n_clusters_found > 1:
                try:
                    silhouette = silhouette_score(features_reduced, labels)
                    calinski = calinski_harabasz_score(features_reduced, labels)
                    davies = davies_bouldin_score(features_reduced, labels)
                    
                    print(f"Clustering quality metrics:")
                    print(f"  - Number of clusters found: {n_clusters_found}")
                    print(f"  - Silhouette score: {silhouette:.4f}")
                    print(f"  - Calinski-Harabasz score: {calinski:.4f}")
                    print(f"  - Davies-Bouldin score: {davies:.4f}")
                except:
                    print("Could not compute clustering quality metrics")
        
        return self
    
    def predict(self, features):
        """Prédire les clusters pour de nouvelles features"""
        if self.cluster_model is None:
            raise ValueError("Clustering model not fitted yet!")
        
        # Appliquer PCA si nécessaire
        if self.pca is not None:
            features = self.pca.transform(features)
        
        # Prédire les clusters
        if self.method == "gmm":
            labels = self.cluster_model.predict(features)
        else:
            labels = self.cluster_model.fit_predict(features) if self.cluster_model == "dbscan" else self.cluster_model.predict(features)
        
        return labels
    
    def visualize_clusters_2d(self, features, labels, save_path):
        """Visualiser les clusters en 2D avec t-SNE"""
        print("Visualizing clusters with t-SNE...")
        
        # Réduire à 2D avec t-SNE
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        features_2d = tsne.fit_transform(features[:5000]) 
        
        # Créer la visualisation
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=labels[:5000], cmap='tab20', s=5, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f't-SNE Visualization of Feature Clusters ({self.method})', fontsize=16)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder
        plt.savefig(os.path.join(save_path, "clusters_tsne.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster visualization saved to {os.path.join(save_path, 'clusters_tsne.png')}")


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        print("resultsPath", resultsPath)
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]
        self.task          = param["MODEL"].get("TASK", "segmentation")
        
        # Initialiser le module de clustering
        self.clustering_module = ClusteringModule(param)

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = DeepLabV3Plus(param).to(self.device)

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        if self.task == "colorization":
            self.criterion = ColorizationLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = LLNDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = LLNDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = LLNDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)

        # Pour suivre les métriques
        self.train_losses = []
        self.val_losses = []
        
        # Pour stocker les features extraites
        self.train_features = []
        self.train_feature_maps = []

    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only=True))

    # -----------------------------------
    # TRAINING LOOP (implémentation réelle)
    # -----------------------------------
    def train(self): 
        best_loss = float('inf')
        
        # train for a given number of epochs
        for epoch in range(self.epoch):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (images, masks, _, _) in enumerate(self.trainDataLoader):
                if self.task == "colorization":
                    # Pour la colorisation: utiliser l'image comme target
                    images = images.to(self.device)
                    targets = images  # Target est l'image couleur originale
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                else:
                    # Pour la segmentation
                    images = images.to(self.device)
                    masks = masks.to(self.device).long()
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.trainDataLoader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{self.epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                modelWts = copy.deepcopy(self.model.state_dict())
        
        # Print learning curves
        self.plot_learning_curves()
        
        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')
        print("Training completed and weights saved!")
        
        # Si c'est une tâche de colorisation, extraire les features pour le clustering
        if self.task == "colorization":
            print("\n🔍 Extracting features for clustering...")
            self.extract_and_cluster_features()

    def extract_and_cluster_features(self):
        """Extraire les features de l'encodeur et appliquer le clustering"""
        self.model.eval()
        
        all_features = []
        all_feature_maps = []  # Stocker les cartes de features complètes
        all_images = []
        all_masks = []
        all_tile_names = []
        
        print("Extracting features from training set...")
        with torch.no_grad():
            for batch_idx, (images, masks, tile_names, _) in enumerate(self.trainDataLoader):
                images = images.to(self.device)
                
                # Extraire les features de l'encodeur
                features = self.model.get_encoder_features(images)
                
                # Déplacer vers le CPU
                features = features.to('cpu')
                images = images.to('cpu')
                masks = masks.to('cpu')
                
                # Reshape: [B, C, H', W'] -> [B*H'*W', C]
                B, C, H_prime, W_prime = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
                
                all_features.append(features_flat.numpy())
                all_feature_maps.append(features.numpy())
                all_images.append(images.numpy())
                all_masks.append(masks.numpy())
                all_tile_names.extend(tile_names)
                
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{len(self.trainDataLoader)}")
        
        # Concatenate all batches
        all_features = np.concatenate(all_features, axis=0)
        all_feature_maps = np.concatenate(all_feature_maps, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        print(f"Total feature vectors: {all_features.shape}")
        print(f"Feature dimension: {all_features.shape[1]}")
        
        # Sauvegarder pour utilisation ultérieure
        self.train_features = all_features
        self.train_feature_maps = all_feature_maps
        self.train_images = all_images
        self.train_masks = all_masks
        self.train_tile_names = all_tile_names
        
        # Appliquer le clustering
        self.perform_clustering(all_features, all_feature_maps, all_images, all_masks, all_tile_names)
    
    def perform_clustering(self, features, feature_maps, images, masks, tile_names):
        """Effectuer le clustering et analyser les résultats"""
        print("\n🎯 Performing clustering on extracted features...")
        
        # Créer un dossier pour les résultats du clustering
        cluster_path = os.path.join(self.resultsPath, "Clustering")
        createFolder(cluster_path)
        
        # Sous-échantillonner si trop de données pour la performance
        max_samples = 100000
        if len(features) > max_samples:
            print(f"Subsampling to {max_samples} feature vectors for clustering...")
            indices = np.random.choice(len(features), max_samples, replace=False)
            features_sampled = features[indices]
        else:
            features_sampled = features
            indices = np.arange(len(features))
        
        # Entraîner le modèle de clustering
        self.clustering_module.fit(features_sampled)
        
        # Prédire les clusters pour toutes les features
        print("Predicting clusters for all features...")
        all_cluster_labels = self.clustering_module.predict(features)
        
        # Visualiser les clusters en 2D
        self.clustering_module.visualize_clusters_2d(features_sampled, 
                                                    all_cluster_labels[indices], 
                                                    cluster_path)
        
        # Analyser la correspondance avec les classes réelles
        self.analyze_cluster_class_correspondence(all_cluster_labels, masks, cluster_path)
        
        # Visualiser des exemples de clustering
        self.visualize_clustering_results(feature_maps, images, masks, 
                                         all_cluster_labels, tile_names, cluster_path)
        
        # Tester le clustering sur les données de test
        self.test_clustering_on_test_set(cluster_path)
    
    def analyze_cluster_class_correspondence(self, cluster_labels, true_masks, save_path):
        """Analyser la correspondance entre les clusters et les classes réelles"""
        print("\n📊 Analyzing cluster-class correspondence...")
        
        # Reshape true_masks pour correspondre aux features
        # true_masks shape: [N, H, W]
        # Nous devons les downsampler pour correspondre aux features
        B, H, W = true_masks.shape
        H_prime = H // self.model.spatial_downsample
        W_prime = W // self.model.spatial_downsample
        
        # Downsample les masks pour correspondre aux dimensions des features
        true_masks_resized = []
        for i in range(B):
            mask_resized = cv2.resize(true_masks[i], (W_prime, H_prime), 
                                     interpolation=cv2.INTER_NEAREST)
            true_masks_resized.append(mask_resized)
        
        true_masks_resized = np.concatenate([m.flatten() for m in true_masks_resized])
        
        # Comparer avec les clusters
        n_clusters = len(np.unique(cluster_labels))
        n_classes = 5  # Nombre de classes de segmentation
        
        # Créer une matrice de confusion clusters vs classes
        confusion_matrix = np.zeros((n_clusters, n_classes))
        
        for cluster in range(n_clusters):
            for class_id in range(n_classes):
                mask = (cluster_labels == cluster) & (true_masks_resized == class_id)
                confusion_matrix[cluster, class_id] = np.sum(mask)
        
        # Normaliser par ligne (clusters)
        confusion_matrix_norm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        
        # Visualiser la matrice
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f'Class {i}' for i in range(n_classes)],
                   yticklabels=[f'Cluster {i}' for i in range(n_clusters)])
        plt.title('Cluster-Class Correspondence Matrix', fontsize=16)
        plt.xlabel('True Class (Segmentation)')
        plt.ylabel('Discovered Cluster')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, "cluster_class_correspondence.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Trouver la classe dominante pour chaque cluster
        print("\n Dominant class for each cluster:")
        for cluster in range(n_clusters):
            dominant_class = np.argmax(confusion_matrix[cluster])
            proportion = confusion_matrix_norm[cluster, dominant_class]
            print(f"Cluster {cluster}: Dominant class = {dominant_class} ({proportion:.1%})")
        
        # Sauvegarder l'analyse
        analysis_path = os.path.join(save_path, "cluster_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write("Cluster Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of clusters: {n_clusters}\n")
            f.write(f"Number of feature vectors: {len(cluster_labels)}\n\n")
            
            f.write("Dominant class for each cluster:\n")
            for cluster in range(n_clusters):
                dominant_class = np.argmax(confusion_matrix[cluster])
                proportion = confusion_matrix_norm[cluster, dominant_class]
                f.write(f"Cluster {cluster}: Class {dominant_class} ({proportion:.1%})\n")
            
            f.write("\nCluster-Class Correspondence Matrix:\n")
            for i in range(n_clusters):
                f.write(f"Cluster {i}: " + " ".join([f"{val:.3f}" for val in confusion_matrix_norm[i]]) + "\n")
    
    def visualize_clustering_results(self, feature_maps, images, masks, 
                                   cluster_labels, tile_names, save_path):
        """Visualiser les résultats du clustering sur quelques exemples"""
        print("\n🖼️ Visualizing clustering results...")
        
        # Créer un dossier pour les visualisations
        vis_path = os.path.join(save_path, "Visualizations")
        createFolder(vis_path)
        
        # Sélectionner quelques exemples
        n_examples = min(5, len(images))
        indices = np.random.choice(len(images), n_examples, replace=False)
        
        for idx in indices:
            # Obtenir l'image, le mask et la carte de features
            img = images[idx]
            mask = masks[idx]
            feature_map = feature_maps[idx]
            
            # Obtenir les labels de cluster pour cette image
            # Calculer les indices correspondant à cette image dans les labels aplatis
            H_prime, W_prime = feature_map.shape[1], feature_map.shape[2]
            start_idx = idx * H_prime * W_prime
            end_idx = (idx + 1) * H_prime * W_prime
            img_cluster_labels = cluster_labels[start_idx:end_idx].reshape(H_prime, W_prime)
            
            # Upsample les labels de cluster à la résolution originale
            cluster_mask_upsampled = cv2.resize(img_cluster_labels.astype(np.float32), 
                                               (img.shape[2], img.shape[1]), 
                                               interpolation=cv2.INTER_NEAREST)
            
            # Créer la visualisation
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Image originale
            img_rgb = img.transpose(1, 2, 0)
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Image grayscale (input pour la colorisation)
            img_gray = 0.2989 * img_rgb[:, :, 0] + 0.5870 * img_rgb[:, :, 1] + 0.1140 * img_rgb[:, :, 2]
            axes[0, 1].imshow(img_gray, cmap='gray')
            axes[0, 1].set_title('Grayscale Input')
            axes[0, 1].axis('off')
            
            # Mask de segmentation réel
            axes[0, 2].imshow(mask, cmap='tab20')
            axes[0, 2].set_title('True Segmentation')
            axes[0, 2].axis('off')
            
            # Carte de clusters (résolution des features)
            axes[1, 0].imshow(img_cluster_labels, cmap='tab20')
            axes[1, 0].set_title(f'Cluster Map ({H_prime}x{W_prime})')
            axes[1, 0].axis('off')
            
            # Carte de clusters upsamplée
            axes[1, 1].imshow(cluster_mask_upsampled, cmap='tab20')
            axes[1, 1].set_title(f'Upsampled Cluster Map')
            axes[1, 1].axis('off')
            
            # Superposition clusters sur image
            axes[1, 2].imshow(img_rgb)
            axes[1, 2].imshow(cluster_mask_upsampled, cmap='tab20', alpha=0.5)
            axes[1, 2].set_title('Clusters Overlay')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Tile: {tile_names[idx]}', fontsize=14)
            plt.tight_layout()
            
            # Sauvegarder
            save_file = os.path.join(vis_path, f"clustering_example_{idx}.png")
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {n_examples} clustering visualizations to {vis_path}")
    
    def test_clustering_on_test_set(self, save_path):
        """Tester le clustering sur l'ensemble de test"""
        print("\n🧪 Testing clustering on test set...")
        
        self.model.eval()
        
        all_test_features = []
        all_test_images = []
        all_test_masks = []
        all_test_tile_names = []
        
        with torch.no_grad():
            for batch_idx, (images, masks, tile_names, _) in enumerate(self.testDataLoader):
                images = images.to(self.device)
                
                # Extraire les features
                features = self.model.get_encoder_features(images)
                
                # Déplacer vers le CPU
                features = features.to('cpu')
                images = images.to('cpu')
                masks = masks.to('cpu')
                
                # Reshape les features
                B, C, H_prime, W_prime = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
                
                all_test_features.append(features_flat.numpy())
                all_test_images.append(images.numpy())
                all_test_masks.append(masks.numpy())
                all_test_tile_names.extend(tile_names)
        
        # Concatenate
        all_test_features = np.concatenate(all_test_features, axis=0)
        all_test_images = np.concatenate(all_test_images, axis=0)
        all_test_masks = np.concatenate(all_test_masks, axis=0)
        
        # Prédire les clusters
        print("Predicting clusters for test set...")
        test_cluster_labels = self.clustering_module.predict(all_test_features)
        
        # Analyser sur le test set
        self.analyze_test_clustering(test_cluster_labels, all_test_masks, 
                                    all_test_tile_names, save_path)
        
        # Générer des pseudo-masques de segmentation
        self.generate_pseudo_segmentation_maps(test_cluster_labels, all_test_images,
                                              all_test_masks, all_test_tile_names, save_path)
    
    def analyze_test_clustering(self, cluster_labels, true_masks, tile_names, save_path):
        """Analyser les résultats du clustering sur le test set"""
        print("\n📊 Analyzing test set clustering...")
        
        # Reshape true_masks pour correspondre aux features
        B, H, W = true_masks.shape
        H_prime = H // self.model.spatial_downsample
        W_prime = W // self.model.spatial_downsample
        
        # Downsample les masks
        true_masks_resized = []
        for i in range(B):
            mask_resized = cv2.resize(true_masks[i], (W_prime, H_prime), 
                                     interpolation=cv2.INTER_NEAREST)
            true_masks_resized.append(mask_resized)
        
        true_masks_resized = np.concatenate([m.flatten() for m in true_masks_resized])
        
        # Calculer la pureté des clusters
        n_clusters = len(np.unique(cluster_labels))
        n_classes = 5
        
        cluster_purity = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) > 0:
                cluster_classes = true_masks_resized[cluster_indices]
                most_common_class = np.bincount(cluster_classes.astype(int)).argmax()
                purity = np.mean(cluster_classes == most_common_class)
                cluster_purity.append((cluster, purity, len(cluster_indices)))
        
        # Trier par pureté
        cluster_purity.sort(key=lambda x: x[1], reverse=True)
        
        # Afficher les résultats
        print("\n🎯 Cluster purity on test set:")
        print(f"{'Cluster':<10} {'Purity':<10} {'Size':<10} {'Dominant Class':<15}")
        print("-" * 45)
        for cluster, purity, size in cluster_purity:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_classes = true_masks_resized[cluster_indices]
            if len(cluster_classes) > 0:
                dominant_class = np.bincount(cluster_classes.astype(int)).argmax()
                print(f"{cluster:<10} {purity:.4f}     {size:<10} {dominant_class:<15}")
        
        # Sauvegarder l'analyse
        test_analysis_path = os.path.join(save_path, "test_cluster_analysis.txt")
        with open(test_analysis_path, 'w') as f:
            f.write("Test Set Cluster Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total feature vectors: {len(cluster_labels)}\n")
            f.write(f"Number of clusters: {n_clusters}\n\n")
            
            f.write("Cluster purity:\n")
            f.write(f"{'Cluster':<10} {'Purity':<10} {'Size':<10} {'Dominant Class':<15}\n")
            f.write("-" * 45 + "\n")
            for cluster, purity, size in cluster_purity:
                cluster_indices = np.where(cluster_labels == cluster)[0]
                cluster_classes = true_masks_resized[cluster_indices]
                if len(cluster_classes) > 0:
                    dominant_class = np.bincount(cluster_classes.astype(int)).argmax()
                    f.write(f"{cluster:<10} {purity:.4f}     {size:<10} {dominant_class:<15}\n")
    
    def generate_pseudo_segmentation_maps(self, cluster_labels, images, masks, 
                                        tile_names, save_path):
        """Générer et visualiser des pseudo-masques de segmentation"""
        print("\n🎨 Generating pseudo-segmentation maps...")
        
        maps_path = os.path.join(save_path, "Pseudo_Segmentation_Maps")
        createFolder(maps_path)
        
        # Calculer les dimensions
        B = len(images)
        H_prime = images.shape[2] // self.model.spatial_downsample
        W_prime = images.shape[3] // self.model.spatial_downsample
        
        # Sélectionner quelques exemples
        n_examples = min(5, B)
        indices = np.random.choice(B, n_examples, replace=False)
        
        for idx in indices:
            # Extraire les clusters pour cette image
            start_idx = idx * H_prime * W_prime
            end_idx = (idx + 1) * H_prime * W_prime
            img_cluster_labels = cluster_labels[start_idx:end_idx].reshape(H_prime, W_prime)
            
            # Upsample à la résolution originale
            pseudo_mask = cv2.resize(img_cluster_labels.astype(np.float32),
                                   (images.shape[3], images.shape[2]),
                                   interpolation=cv2.INTER_NEAREST)
            
            # Créer la comparaison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Image originale
            img_rgb = images[idx].transpose(1, 2, 0)
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Pseudo-segmentation (clusters)
            axes[1].imshow(pseudo_mask, cmap='tab20')
            axes[1].set_title('Pseudo-Segmentation (Clusters)')
            axes[1].axis('off')
            
            # Segmentation réelle
            axes[2].imshow(masks[idx], cmap='tab20')
            axes[2].set_title('True Segmentation')
            axes[2].axis('off')
            
            plt.suptitle(f'Comparison for {tile_names[idx]}', fontsize=14)
            plt.tight_layout()
            
            # Sauvegarder
            save_file = os.path.join(maps_path, f"pseudo_segmentation_{idx}.png")
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Sauvegarder les données du pseudo-masque
            np.save(os.path.join(maps_path, f"pseudo_mask_{idx}.npy"), pseudo_mask)
            np.save(os.path.join(maps_path, f"true_mask_{idx}.npy"), masks[idx])
        
        print(f"Generated {n_examples} pseudo-segmentation maps in {maps_path}")
        
        # Évaluer la similarité avec les vrais masques
        self.evaluate_pseudo_segmentation_quality(cluster_labels, masks, save_path)
    
    def evaluate_pseudo_segmentation_quality(self, cluster_labels, true_masks, save_path):
        """Évaluer la qualité des pseudo-segmentations"""
        print("\n Evaluating pseudo-segmentation quality...")
        
        B, H, W = true_masks.shape
        H_prime = H // self.model.spatial_downsample
        W_prime = W // self.model.spatial_downsample
        
        # Downsample les vrais masques pour comparaison
        true_masks_downsampled = []
        for i in range(B):
            mask_down = cv2.resize(true_masks[i], (W_prime, H_prime),
                                 interpolation=cv2.INTER_NEAREST)
            true_masks_downsampled.append(mask_down.flatten())
        
        true_masks_flat = np.concatenate(true_masks_downsampled)
        
        # Mapper les clusters aux classes pour maximiser l'accord
        from scipy.optimize import linear_sum_assignment
        
        n_clusters = len(np.unique(cluster_labels))
        n_classes = 5
        
        # Créer la matrice de coût
        cost_matrix = np.zeros((n_clusters, n_classes))
        for i in range(n_clusters):
            for j in range(n_classes):
                cost_matrix[i, j] = -np.sum((cluster_labels == i) & (true_masks_flat == j))
        
        # Trouver l'appariement optimal
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Créer le mapping optimal cluster -> classe
        cluster_to_class = {}
        for cluster, class_id in zip(row_ind, col_ind):
            cluster_to_class[cluster] = class_id
        
        # Appliquer le mapping
        mapped_labels = np.array([cluster_to_class.get(c, -1) for c in cluster_labels])
        
        # Calculer les métriques
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(true_masks_flat, mapped_labels)
        nmi = normalized_mutual_info_score(true_masks_flat, mapped_labels)
        
        print(f"\n🎯 Pseudo-segmentation evaluation:")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        
        # Sauvegarder les résultats
        eval_path = os.path.join(save_path, "pseudo_segmentation_evaluation.txt")
        with open(eval_path, 'w') as f:
            f.write("Pseudo-Segmentation Evaluation\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n\n")
            
            f.write("Optimal cluster-to-class mapping:\n")
            for cluster, class_id in cluster_to_class.items():
                f.write(f"Cluster {cluster} -> Class {class_id}\n")
    
    # Les autres méthodes restent inchangées...
    # (validate, plot_learning_curves, evaluate_colorization, evaluate_segmentation, etc.)

    def validate(self):
        """Validation step"""
        self.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for images, masks, _, _ in self.valDataLoader:
                if self.task == "colorization":
                    images = images.to(self.device)
                    targets = images
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                else:
                    images = images.to(self.device)
                    masks = masks.to(self.device).long()
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                running_val_loss += loss.item()
        
        return running_val_loss / len(self.valDataLoader)

    def plot_learning_curves(self):
        """Plot and save learning curves"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Learning Curves - {self.task}')
        plt.legend()
        plt.grid(True)
        
        curves_path = os.path.join(self.resultsPath, "learning_curves.png")
        print("self.resultsPath", self.resultsPath)
        plt.savefig(curves_path)
        plt.close()
        print(f"📊 Learning curves saved to {curves_path}")

    # -------------------------------------------------
    # EVALUATION PROCEDURE 
    # -------------------------------------------------
    def evaluate(self):
        if self.task == "colorization":
            self.evaluate_colorization()
        else:
            self.evaluate_segmentation()
    
    def evaluate_colorization(self):
        """Evaluation for colorization task"""
        self.model.eval()
        
        all_inputs = []
        all_outputs = []
        all_targets = []
        all_tileNames = []
        
        with torch.no_grad():
            for (images, _, tileNames, _) in self.testDataLoader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Move back to CPU for processing
                images = images.to('cpu')
                outputs = outputs.to('cpu')
                
                all_inputs.append(images)
                all_outputs.append(outputs)
                all_targets.append(images)  # For colorization, target = input
                all_tileNames.extend(tileNames)
        
        # Concatenate all batches
        all_inputs = torch.cat(all_inputs, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        mse_loss = nn.MSELoss()(all_outputs, all_targets)
        psnr = 10 * torch.log10(1.0 / mse_loss)
        
        print(f"\n🎨 Colorization Results:")
        print(f"MSE Loss: {mse_loss.item():.4f}")
        print(f"PSNR: {psnr.item():.2f} dB")
        
        # Save metrics
        metrics_path = os.path.join(self.resultsPath, "colorization_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Colorization Evaluation Results:\n")
            f.write(f"MSE Loss: {mse_loss.item():.4f}\n")
            f.write(f"PSNR: {psnr.item():.2f} dB\n")
        
        # Save sample images
        self.save_colorization_samples(all_inputs, all_outputs, all_tileNames)
    
    def save_colorization_samples(self, inputs, outputs, tileNames, num_samples=5):
        """Save sample colorization results"""
        import matplotlib.pyplot as plt
        
        savePath = os.path.join(self.resultsPath, "Colorization_Samples")
        createFolder(savePath)
        
        num_samples = min(num_samples, len(inputs))
        
        for i in range(num_samples):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Input (grayscale)
            gray_input = 0.2989 * inputs[i, 0] + 0.5870 * inputs[i, 1] + 0.1140 * inputs[i, 2]
            axes[0].imshow(gray_input, cmap='gray')
            axes[0].set_title('Input (Grayscale)')
            axes[0].axis('off')
            
            # Output (colorized)
            output_img = outputs[i].permute(1, 2, 0).numpy()
            output_img = np.clip(output_img, 0, 1)
            axes[1].imshow(output_img)
            axes[1].set_title('Colorized Output')
            axes[1].axis('off')
            
            # Target (original color)
            target_img = inputs[i].permute(1, 2, 0).numpy()
            axes[2].imshow(target_img)
            axes[2].set_title('Original Color')
            axes[2].axis('off')
            
            plt.suptitle(f'Sample {i+1}: {tileNames[i]}', fontsize=12)
            plt.tight_layout()
            
            sample_path = os.path.join(savePath, f"sample_{i+1}_{tileNames[i]}.png")
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {num_samples} colorization samples to {savePath}")
    
    def evaluate_segmentation(self):
        """Evaluation for segmentation task (original code)"""
        self.model.eval()
        
        allMasks, allMasksPreds, allTileNames, allResizedImgs = [], [], [], []
        allOutputs = [] 
        
        with torch.no_grad():
            for (images, masks, tileNames, resizedImgs) in self.testDataLoader:
                images = images.to(self.device)
                outputs = self.model(images)

                # Move back to CPU for processing
                outputs = outputs.to('cpu')
                
                # Store raw outputs for ROC AUC calculation
                allOutputs.extend(outputs.data.numpy())  
                
                # Get predicted masks (class with highest probability)
                masksPreds = torch.argmax(outputs, dim=1)

                allMasks.extend(masks.data.numpy())
                allMasksPreds.extend(masksPreds.data.numpy())
                allResizedImgs.extend(resizedImgs.data.numpy())
                allTileNames.extend(tileNames)
        
        allMasks = np.array(allMasks)
        allMasksPreds = np.array(allMasksPreds)
        allResizedImgs = np.array(allResizedImgs)
        allOutputs = np.array(allOutputs)  
            
        # Qualitative Evaluation
        savePath = os.path.join(self.resultsPath, "Test")
        createFolder(savePath)
        reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, savePath)

        # Quantitative Evaluation
        self.quantitative_evaluation(allMasks, allMasksPreds, allOutputs)

    