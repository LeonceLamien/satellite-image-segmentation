import numpy as np
import torch
import json
import cv2
import os
from tqdm import tqdm

######################################################################################
#
# MODULE FOR COMPUTING CLASS WEIGHTS FOR IMBALANCED SEGMENTATION DATASETS
# This module provides multiple strategies to compute optimal class weights
# 
######################################################################################


def compute_class_weights(imgDirectory, maskDirectory, setName="train", 
                         num_classes=5, method="inverse_frequency", beta=0.999, smoothing=1.0):
    """
    Calcule les poids des classes pour gérer le déséquilibre
    
    INPUTS:
        - imgDirectory (str): path to images folder
        - maskDirectory (str): path to masks folder
        - setName (str): "train", "val", or "test"
        - num_classes (int): number of classes
        - method (str): méthode de calcul des poids
            * "inverse_frequency": 1 / frequency
            * "effective_samples": Class-Balanced Loss (CVPR 2019)
            * "sqrt_inverse": sqrt(1 / frequency) - plus doux
            * "log_inverse": log(total / frequency) - encore plus doux
        - beta (float): paramètre pour effective_samples (0.999 recommandé)
        - smoothing (float): facteur d'adoucissement (0.5-1.0)
            * 1.0 = pas d'adoucissement
            * 0.7 = weights réduits de 30%
            * 0.5 = weights très doux (recommandé si problème)
    
    OUTPUTS:
        - weights (torch.Tensor): poids pour chaque classe
        - class_frequencies (dict): statistiques des classes
    """
    
    print(f"\n{'='*60}")
    print(f"📊 Computing class weights using '{method}' method")
    print(f"{'='*60}\n")
    
    # Load file list
    with open(f"./Dataset/{setName}.json", 'r') as f:
        files = json.load(f)
    
    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    print(f"Analyzing {len(files)} masks from {setName} set...")
    for filename in tqdm(files, desc="Counting pixels"):
        mask_path = os.path.join(maskDirectory, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"⚠️  Warning: Could not read {mask_path}")
            continue
        
        # Count pixels for each class
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(mask == class_id)
    
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    
    # ============================================
    # DISPLAY STATISTICS
    # ============================================
    print(f"\n{'='*60}")
    print(f"📈 Class Distribution Statistics")
    print(f"{'='*60}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"\n{'Class':<8} {'Pixels':<15} {'Frequency':<12} {'Percentage':<12}")
    print("-" * 60)
    
    for i in range(num_classes):
        print(f"{i:<8} {class_counts[i]:<15,} {class_frequencies[i]:<12.6f} {class_frequencies[i]*100:<11.2f}%")
    
    # ============================================
    # COMPUTE WEIGHTS BASED ON METHOD
    # ============================================
    if method == "inverse_frequency":
        # Simple inverse frequency: w_i = 1 / freq_i
        weights = 1.0 / (class_frequencies + 1e-6)
        
    elif method == "effective_samples":
        # Class-Balanced Loss (CVPR 2019): w_i = (1 - beta) / (1 - beta^n_i)
        # Plus efficace pour les classes très déséquilibrées
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-7)
        
        # Debug: afficher les valeurs intermédiaires
        print(f"\n🔍 Debug effective_samples:")
        print(f"   Beta: {beta}")
        for i in range(num_classes):
            print(f"   Class {i}: count={class_counts[i]}, effective_num={effective_num[i]:.6f}, raw_weight={weights[i]:.6f}")
        
    elif method == "sqrt_inverse":
        # Square root of inverse frequency (plus doux)
        weights = np.sqrt(1.0 / (class_frequencies + 1e-6))
        
    elif method == "log_inverse":
        # Logarithmic inverse (encore plus doux)
        weights = np.log(total_pixels / (class_counts + 1e-6))
        
    elif method == "selective":
        # 🆕 SELECTIVE: Ne pondère QUE les classes très rares (< 5%)
        # Les classes majoritaires gardent weight=1.0
        weights = np.ones(num_classes, dtype=np.float32)
        threshold = 0.05  # Classes avec < 5% des pixels
        
        print(f"\n🎯 Selective weighting (threshold={threshold*100}%):")
        for i in range(num_classes):
            if class_frequencies[i] < threshold:
                # Classe rare : appliquer log_inverse
                weights[i] = np.log(total_pixels / (class_counts[i] + 1e-6))
                print(f"   Class {i}: {class_frequencies[i]*100:.2f}% → weighted (w={weights[i]:.3f})")
            else:
                # Classe majoritaire : garder 1.0
                print(f"   Class {i}: {class_frequencies[i]*100:.2f}% → no weighting (w=1.0)")
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: "
                        "'inverse_frequency', 'effective_samples', 'sqrt_inverse', 'log_inverse', 'selective'")
    
    # Apply smoothing factor (brings weights closer to 1.0)
    if smoothing < 1.0:
        weights = 1.0 + (weights - 1.0) * smoothing
        print(f"\n🔧 Smoothing applied: factor={smoothing:.2f}")
    
    # Normalize weights (optionnel mais recommandé)
    weights = weights / weights.sum() * num_classes
    
    # Convert to torch tensor
    weights_tensor = torch.FloatTensor(weights)
    
    # ============================================
    # DISPLAY COMPUTED WEIGHTS
    # ============================================
    print(f"\n{'='*60}")
    print(f"⚖️  Computed Weights (method: {method})")
    print(f"{'='*60}")
    print(f"\n{'Class':<8} {'Weight':<12} {'Normalized':<12}")
    print("-" * 60)
    
    for i in range(num_classes):
        print(f"{i:<8} {weights[i]:<12.4f} {weights[i]/weights.sum():<12.4f}")
    
    print(f"\n✅ Weights computed successfully!")
    print(f"{'='*60}\n")
    
    # Statistics dictionary
    class_info = {
        'class_counts': class_counts.tolist(),
        'class_frequencies': class_frequencies.tolist(),
        'weights': weights.tolist(),
        'method': method,
        'total_pixels': int(total_pixels)
    }
    
    return weights_tensor, class_info


def get_class_weights_from_config(param, imgDirectory, maskDirectory):
    """
    Récupère les class weights depuis la config ou les calcule automatiquement
    
    INPUTS:
        - param (dict): configuration YAML
        - imgDirectory (str): path to images
        - maskDirectory (str): path to masks
    
    OUTPUTS:
        - weights (torch.Tensor): class weights
    """
    
    # Check if weights are manually specified in config
    if "CLASS_WEIGHTS" in param["TRAINING"]:
        weights = torch.FloatTensor(param["TRAINING"]["CLASS_WEIGHTS"])
        print(f"\n✅ Using manual class weights from config: {weights.tolist()}")
        return weights
    
    # Otherwise, compute automatically
    num_classes = param["MODEL"].get("NUM_CLASSES", 5)
    method = param["TRAINING"].get("CLASS_WEIGHT_METHOD", "effective_samples")
    beta = param["TRAINING"].get("CLASS_WEIGHT_BETA", 0.999)
    smoothing = param["TRAINING"].get("CLASS_WEIGHT_SMOOTHING", 1.0)  # 🆕
    
    weights, class_info = compute_class_weights(
        imgDirectory=imgDirectory,
        maskDirectory=maskDirectory,
        setName="train",
        num_classes=num_classes,
        method=method,
        beta=beta,
        smoothing=smoothing  # 🆕
    )
    
    return weights


def compare_weight_methods(imgDirectory, maskDirectory, setName="train", 
                          num_classes=5, beta=0.999):
    """
    Compare différentes méthodes de calcul de poids pour aider au choix
    
    INPUTS:
        - imgDirectory (str): path to images
        - maskDirectory (str): path to masks
        - setName (str): dataset split
        - num_classes (int): number of classes
        - beta (float): parameter for effective_samples
    """
    
    methods = ["inverse_frequency", "effective_samples", "sqrt_inverse", "log_inverse"]
    
    print(f"\n{'='*80}")
    print(f"🔍 COMPARISON OF WEIGHT COMPUTATION METHODS")
    print(f"{'='*80}\n")
    
    all_weights = {}
    
    for method in methods:
        weights, _ = compute_class_weights(
            imgDirectory=imgDirectory,
            maskDirectory=maskDirectory,
            setName=setName,
            num_classes=num_classes,
            method=method,
            beta=beta
        )
        all_weights[method] = weights.numpy()
        print()
    
    # Comparative table
    print(f"\n{'='*80}")
    print(f"📊 COMPARATIVE WEIGHTS TABLE")
    print(f"{'='*80}\n")
    
    header = f"{'Class':<8}"
    for method in methods:
        header += f"{method:<20}"
    print(header)
    print("-" * 80)
    
    for i in range(num_classes):
        row = f"{i:<8}"
        for method in methods:
            row += f"{all_weights[method][i]:<20.4f}"
        print(row)
    
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS:")
    print("   - 'inverse_frequency': Simple, mais peut sur-pondérer les classes rares")
    print("   - 'effective_samples': ⭐ RECOMMANDÉ - Meilleur équilibre (CVPR 2019)")
    print("   - 'sqrt_inverse': Plus doux que inverse_frequency")
    print("   - 'log_inverse': Encore plus doux, bon si légèrement déséquilibré")
    print(f"{'='*80}\n")


######################################################################################
#
# SCRIPT STANDALONE POUR TESTER
# Usage: python Dataset/classWeights.py
#
######################################################################################

if __name__ == "__main__":
    import sys
    import os
    from os.path import dirname, abspath
    
    # Setup paths
    rootDirectory = dirname(dirname(abspath(__file__)))
    datasetDirectory = os.path.join(rootDirectory, "Dataset")
    imgDirectory = os.path.join(datasetDirectory, "images")
    maskDirectory = os.path.join(datasetDirectory, "annotations")
    
    print(f"\n🔬 Testing Class Weights Computation")
    print(f"Images: {imgDirectory}")
    print(f"Masks: {maskDirectory}\n")
    
    # Compare all methods
    compare_weight_methods(
        imgDirectory=imgDirectory,
        maskDirectory=maskDirectory,
        setName="train",
        num_classes=5,
        beta=0.999
    )
