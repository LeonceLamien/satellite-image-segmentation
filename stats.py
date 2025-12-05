#!/usr/bin/env python3
"""
Calcul des statistiques de normalisation (Mean & Std) 
UNIQUEMENT sur le training set (pas de data leakage!)

Usage: python compute_normalization_stats.py
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

print(f"\n{'='*80}")
print(f"📊 CALCUL DES STATISTIQUES DE NORMALISATION (Training Set Only)")
print(f"{'='*80}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).parent
IMAGES_DIR = ROOT_DIR / "Dataset" / "images"
TRAIN_JSON = ROOT_DIR / "Dataset" / "train.json"

print(f"📁 Images directory: {IMAGES_DIR}")
print(f"📁 Train JSON: {TRAIN_JSON}\n")

# ============================================================================
# VÉRIFICATIONS
# ============================================================================
if not IMAGES_DIR.exists():
    print(f"❌ ERROR: Images directory not found: {IMAGES_DIR}")
    exit(1)

if not TRAIN_JSON.exists():
    print(f"❌ ERROR: Train JSON not found: {TRAIN_JSON}")
    exit(1)

# ============================================================================
# CHARGEMENT DE LA LISTE DES TRAINING TILES
# ============================================================================
with open(TRAIN_JSON, 'r') as f:
    train_files = json.load(f)

print(f"✅ Training set loaded: {len(train_files)} images\n")

# ============================================================================
# MÉTHODE 1: CALCUL GLOBAL (STATISTIQUEMENT CORRECT) ⭐ RECOMMANDÉ
# ============================================================================
print(f"{'='*80}")
print(f"🔄 Méthode 1: Calcul global (statistiquement correct)")
print(f"{'='*80}\n")

all_pixels = []
skipped = 0

for filename in tqdm(train_files, desc="Chargement des images"):
    img_path = IMAGES_DIR / filename
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Impossible de lire: {filename}")
        skipped += 1
        continue
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Flatten H×W×3 → N×3
    all_pixels.append(img.reshape(-1, 3))

if skipped > 0:
    print(f"\n⚠️  {skipped} images skipped\n")

# Concatener tous les pixels
all_pixels = np.concatenate(all_pixels, axis=0)  # Shape: (total_pixels, 3)

# Calculer mean et std globaux
mean_global = all_pixels.mean(axis=0)
std_global = all_pixels.std(axis=0)

print(f"\n{'='*80}")
print(f"✅ RÉSULTATS (Méthode Globale - Recommandée)")
print(f"{'='*80}")
print(f"   Total images: {len(train_files) - skipped}")
print(f"   Total pixels: {all_pixels.shape[0]:,}")
print(f"\n   MEAN (R, G, B): [{mean_global[0]:.6f}, {mean_global[1]:.6f}, {mean_global[2]:.6f}]")
print(f"   STD  (R, G, B): [{std_global[0]:.6f}, {std_global[1]:.6f}, {std_global[2]:.6f}]")
print(f"{'='*80}\n")

# Free memory
del all_pixels

# ============================================================================
# MÉTHODE 2: CALCUL INCRÉMENTAL (ÉCONOMISE LA MÉMOIRE)
# ============================================================================
print(f"{'='*80}")
print(f"🔄 Méthode 2: Calcul incrémental (pour gros datasets)")
print(f"{'='*80}\n")

n_pixels = 0
mean_incremental = np.zeros(3, dtype=np.float64)
M2 = np.zeros(3, dtype=np.float64)  # Pour la variance

skipped = 0

for filename in tqdm(train_files, desc="Calcul incrémental"):
    img_path = IMAGES_DIR / filename
    
    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    pixels = img.reshape(-1, 3)  # Shape: (H*W, 3)
    
    # Welford's online algorithm pour mean et variance
    for pixel in pixels:
        n_pixels += 1
        delta = pixel - mean_incremental
        mean_incremental += delta / n_pixels
        delta2 = pixel - mean_incremental
        M2 += delta * delta2

std_incremental = np.sqrt(M2 / n_pixels)

print(f"\n{'='*80}")
print(f"✅ RÉSULTATS (Méthode Incrémentale)")
print(f"{'='*80}")
print(f"   Total images: {len(train_files) - skipped}")
print(f"   Total pixels: {n_pixels:,}")
print(f"\n   MEAN (R, G, B): [{mean_incremental[0]:.6f}, {mean_incremental[1]:.6f}, {mean_incremental[2]:.6f}]")
print(f"   STD  (R, G, B): [{std_incremental[0]:.6f}, {std_incremental[1]:.6f}, {std_incremental[2]:.6f}]")
print(f"{'='*80}\n")

# ============================================================================
# MÉTHODE 3: APPROXIMATION (TON CODE ORIGINAL)
# ============================================================================
print(f"{'='*80}")
print(f"🔄 Méthode 3: Approximation par image (plus rapide, moins précis)")
print(f"{'='*80}\n")

means = []
stds = []
skipped = 0

for filename in tqdm(train_files, desc="Calcul par image"):
    img_path = IMAGES_DIR / filename
    
    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    means.append(img.mean(axis=(0, 1)))
    stds.append(img.std(axis=(0, 1)))

mean_approx = np.mean(means, axis=0)
std_approx = np.mean(stds, axis=0)  # ⚠️ Statistiquement incorrect mais acceptable

print(f"\n{'='*80}")
print(f"✅ RÉSULTATS (Approximation)")
print(f"{'='*80}")
print(f"   Total images: {len(train_files) - skipped}")
print(f"\n   MEAN (R, G, B): [{mean_approx[0]:.6f}, {mean_approx[1]:.6f}, {mean_approx[2]:.6f}]")
print(f"   STD  (R, G, B): [{std_approx[0]:.6f}, {std_approx[1]:.6f}, {std_approx[2]:.6f}]")
print(f"{'='*80}\n")

# ============================================================================
# COMPARAISON DES MÉTHODES
# ============================================================================
print(f"{'='*80}")
print(f"📊 COMPARAISON DES TROIS MÉTHODES")
print(f"{'='*80}")
print(f"{'Stat':<10} {'Canal':<8} {'Globale':<12} {'Incrémentale':<15} {'Approximation':<15}")
print(f"{'-'*80}")

channels = ['R', 'G', 'B']

for i, channel in enumerate(channels):
    print(f"{'Mean':<10} {channel:<8} {mean_global[i]:.6f}   {mean_incremental[i]:.6f}    {mean_approx[i]:.6f}")

print()

for i, channel in enumerate(channels):
    print(f"{'Std':<10} {channel:<8} {std_global[i]:.6f}   {std_incremental[i]:.6f}    {std_approx[i]:.6f}")

print(f"{'='*80}\n")

# ============================================================================
# DIFFÉRENCES
# ============================================================================
print(f"{'='*80}")
print(f"📊 DIFFÉRENCES PAR RAPPORT À LA MÉTHODE GLOBALE")
print(f"{'='*80}")
print(f"{'Méthode':<25} {'Mean Diff (max)':<20} {'Std Diff (max)':<20}")
print(f"{'-'*80}")

diff_mean_inc = np.abs(mean_global - mean_incremental).max()
diff_std_inc = np.abs(std_global - std_incremental).max()
diff_mean_approx = np.abs(mean_global - mean_approx).max()
diff_std_approx = np.abs(std_global - std_approx).max()

print(f"{'Incrémentale':<25} {diff_mean_inc:.8f}         {diff_std_inc:.8f}")
print(f"{'Approximation':<25} {diff_mean_approx:.8f}         {diff_std_approx:.8f}")
print(f"{'='*80}\n")

# ============================================================================
# RECOMMANDATION FINALE
# ============================================================================
print(f"{'='*80}")
print(f"🎯 RECOMMANDATION FINALE")
print(f"{'='*80}")
print(f"\n✅ Utilise la Méthode Globale (statistiquement correcte)")
print(f"\n📋 À copier dans ton fichier YAML (ex: DefaultExp_WithWeights.yaml):\n")
print(f"DATASET:")
print(f"  NORMALIZE_MEAN: [{mean_global[0]:.6f}, {mean_global[1]:.6f}, {mean_global[2]:.6f}]")
print(f"  NORMALIZE_STD:  [{std_global[0]:.6f}, {std_global[1]:.6f}, {std_global[2]:.6f}]")
print(f"\n{'='*80}")

# ============================================================================
# COMPARAISON AVEC IMAGENET (POUR INFO)
# ============================================================================
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

print(f"\n{'='*80}")
print(f"📊 COMPARAISON AVEC IMAGENET (pour info)")
print(f"{'='*80}")
print(f"\nImageNet MEAN: [{imagenet_mean[0]:.6f}, {imagenet_mean[1]:.6f}, {imagenet_mean[2]:.6f}]")
print(f"ImageNet STD:  [{imagenet_std[0]:.6f}, {imagenet_std[1]:.6f}, {imagenet_std[2]:.6f}]")
print(f"\nTon dataset MEAN: [{mean_global[0]:.6f}, {mean_global[1]:.6f}, {mean_global[2]:.6f}]")
print(f"Ton dataset STD:  [{std_global[0]:.6f}, {std_global[1]:.6f}, {std_global[2]:.6f}]")

diff_mean_imagenet = np.abs(mean_global - imagenet_mean)
diff_std_imagenet = np.abs(std_global - imagenet_std)

print(f"\nDifférence Mean: [{diff_mean_imagenet[0]:.6f}, {diff_mean_imagenet[1]:.6f}, {diff_mean_imagenet[2]:.6f}]")
print(f"Différence Std:  [{diff_std_imagenet[0]:.6f}, {diff_std_imagenet[1]:.6f}, {diff_std_imagenet[2]:.6f}]")

if np.all(diff_mean_imagenet < 0.05) and np.all(diff_std_imagenet < 0.05):
    print(f"\n✅ Ton dataset est similaire à ImageNet → OK d'utiliser les stats ImageNet")
else:
    print(f"\n⚠️  Ton dataset diffère d'ImageNet → Utilise tes propres statistiques!")

print(f"{'='*80}\n")

# ============================================================================
# SAUVEGARDE OPTIONNELLE
# ============================================================================
save_choice = input("💾 Voulez-vous sauvegarder les statistiques dans un fichier JSON? (y/n): ")

if save_choice.lower() == 'y':
    output_file = ROOT_DIR / "Dataset" / "normalization_stats.json"
    
    stats = {
        "method": "global",
        "dataset": "training_set_only",
        "num_images": len(train_files) - skipped,
        "num_pixels": int(n_pixels),
        "mean_rgb": mean_global.tolist(),
        "std_rgb": std_global.tolist(),
        "yaml_format": {
            "NORMALIZE_MEAN": [float(mean_global[0]), float(mean_global[1]), float(mean_global[2])],
            "NORMALIZE_STD": [float(std_global[0]), float(std_global[1]), float(std_global[2])]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistiques sauvegardées dans: {output_file}\n")
else:
    print(f"\n❌ Statistiques non sauvegardées\n")

print(f"{'='*80}")
print(f"✅ CALCUL TERMINÉ!")
print(f"{'='*80}\n")