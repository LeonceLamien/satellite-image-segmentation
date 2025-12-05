#!/usr/bin/env python3
"""
Script de Visualisation des Data Augmentations via DataLoader
Génère des images avant/après pour montrer l'impact des augmentations

Usage: python visualize_augmentations.py --num_samples 5
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
import cv2
import copy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Dataset.dataLoader import LLNDataset


# Color map for classes (adaptez selon vos classes)
CLASS_COLORS = {
    0: [255, 0, 0],      # Rouge - Classe 0
    1: [0, 255, 0],      # Vert - Classe 1
    2: [0, 0, 255],      # Bleu - Classe 2
    3: [255, 255, 0],    # Jaune - Classe 3
    4: [255, 0, 255],    # Magenta - Classe 4
}

CLASS_NAMES = {
    0: "Class 0 (36.19%)",
    1: "Class 1 (0.84%)",
    2: "Class 2 (6.36%)",
    3: "Class 3 (12.78%)",
    4: "Class 4 (43.84%)",
}


def tensor_to_numpy_image(tensor_img):
    """Convert normalized tensor to numpy image [0, 255]"""
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor_img.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def mask_to_rgb(mask, class_colors=CLASS_COLORS):
    """Convert mask with class IDs to RGB image"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        rgb_mask[mask == class_id] = color
    
    return rgb_mask


def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay colored mask on image"""
    rgb_mask = mask_to_rgb(mask)
    overlay = (image * (1 - alpha) + rgb_mask * alpha).astype(np.uint8)
    return overlay


def visualize_augmentations(config_path, num_samples=5, seed=42, 
                            output_dir="Results/Augmentation_Viz"):
    """
    Visualize augmentations on sample images using DataLoader
    
    Args:
        config_path: Path to YAML config file
        num_samples: Number of sample images to process
        seed: Random seed for reproducibility
        output_dir: Directory to save visualization images
    """
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        param = yaml.safe_load(f)
    
    # Get paths from config
    root_dir = Path(__file__).parent
    img_dir = root_dir / "Dataset" / "images"
    mask_dir = root_dir / "Dataset" / "annotations"
        
    print(f"\n{'='*80}")
    print(f"🎨 VISUALISATION DES DATA AUGMENTATIONS (via DataLoader)")
    print(f"{'='*80}\n")
    print(f"📁 Configuration: {config_path}")
    print(f"📁 Images: {img_dir}")
    print(f"📁 Masks: {mask_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Number of samples: {num_samples}")
    print(f"🎲 Random seed: {seed}\n")
    
    # ========================================================================
    # UNIQUEMENT LE TRAIN DATASET
    # ========================================================================
    train_dataset = LLNDataset(
        imgDirectory=str(img_dir),
        maskDirectory=str(mask_dir),
        setName="train",
        param=param
    )
    
    print(f"✅ Train dataset loaded: {len(train_dataset)} images")
    
    # ========================================================================
    # SELECT RANDOM SAMPLES FROM TRAIN
    # ========================================================================
    total_samples = min(num_samples, len(train_dataset))
    random_indices = np.random.choice(len(train_dataset), total_samples, replace=False)
    
    print(f"📌 Selected indices: {random_indices.tolist()}\n")
    
    # ========================================================================
    # VISUALIZE EACH SAMPLE
    # ========================================================================
    for sample_idx, dataset_idx in enumerate(random_indices):
        print(f"Processing sample {sample_idx + 1}/{total_samples} (index {dataset_idx})...")
        
        # Get ONE sample from train_dataset
        # Le DataLoader retourne: (augmented_img_tensor, mask, tileName, original_resized_img_tensor)
        img_aug_tensor, mask_aug, tileName, img_original_tensor = train_dataset[dataset_idx]
        
        # Convert to numpy
        img_original = tensor_to_numpy_image(img_original_tensor)  # Original (resized only)
        img_augmented = tensor_to_numpy_image(img_aug_tensor)       # Augmented
        mask_original = mask_aug.cpu().numpy() if isinstance(mask_aug, torch.Tensor) else mask_aug
        
        # Get 3 MORE augmented versions of the SAME tile
        additional_augmented = []
        for aug_idx in range(3):
            img_aug_tensor_extra, mask_aug_extra, _, _ = train_dataset[dataset_idx]
            img_aug_extra = tensor_to_numpy_image(img_aug_tensor_extra)
            mask_aug_extra = mask_aug_extra.cpu().numpy() if isinstance(mask_aug_extra, torch.Tensor) else mask_aug_extra
            additional_augmented.append((img_aug_extra, mask_aug_extra))
        
        # All augmented samples (first one + 3 additional)
        all_augmented = [(img_augmented, mask_original)] + additional_augmented
        
        # ====================================================================
        # CREATE VISUALIZATION
        # ====================================================================
        fig = plt.figure(figsize=(20, 12))
        
        # Row 1: Original image + mask + overlay
        ax1 = plt.subplot(3, 5, 1)
        ax1.imshow(img_original)
        ax1.set_title("Original Image\n(Resized Only)", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 5, 2)
        ax2.imshow(mask_to_rgb(mask_original))
        ax2.set_title("Mask", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 5, 3)
        ax3.imshow(overlay_mask_on_image(img_original, mask_original, alpha=0.4))
        ax3.set_title("Overlay (Original)", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Empty spaces
        plt.subplot(3, 5, 4).axis('off')
        plt.subplot(3, 5, 5).axis('off')
        
        # Row 2 & 3: 4 Augmented versions
        for i, (aug_img, aug_mask) in enumerate(all_augmented):
            # Augmented images (row 2)
            ax = plt.subplot(3, 5, 6 + i)
            ax.imshow(aug_img)
            ax.set_title(f"Augmented #{i+1}", fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Augmented masks (row 3)
            ax = plt.subplot(3, 5, 11 + i)
            ax.imshow(mask_to_rgb(aug_mask))
            ax.set_title(f"Mask #{i+1}", fontsize=11)
            ax.axis('off')
        
        # Overlay for augmented #4
        ax = plt.subplot(3, 5, 15)
        ax.imshow(overlay_mask_on_image(all_augmented[3][0], all_augmented[3][1], alpha=0.4))
        ax.set_title("Overlay (Aug #4)", fontsize=11)
        ax.axis('off')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=np.array(color)/255, label=CLASS_NAMES[class_id])
            for class_id, color in CLASS_COLORS.items()
        ]
        
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                  title="Classes", title_fontsize=11, framealpha=0.9)
        
        plt.suptitle(f"Data Augmentation Visualization - {tileName}\n"
                    f"Sample {sample_idx + 1}/{total_samples} (Training Set Index: {dataset_idx})", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        safe_filename = tileName.replace('.png', '').replace('/', '_')
        output_path = os.path.join(output_dir, f"augmentation_{sample_idx+1:02d}_{safe_filename}.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved to {output_path}")
    
    # ========================================================================
    # CREATE COMPARISON GRID
    # ========================================================================
    create_comparison_grid(train_dataset, random_indices[:3], output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Visualisation terminée!")
    print(f"{'='*80}")
    print(f"\n📊 Fichiers générés dans: {output_dir}")
    print(f"   - augmentation_*.jpg : Visualisations individuelles")
    print(f"   - comparison_grid.jpg : Grille comparative\n")


def create_comparison_grid(train_dataset, indices, output_dir):
    """Create a compact comparison grid using DataLoader"""
    
    print(f"\nCréation de la grille comparative...")
    
    num_samples = len(indices)
    
    fig = plt.figure(figsize=(18, 4 * num_samples))
    
    for row_idx, dataset_idx in enumerate(indices):
        # Get data from train_dataset
        img_aug_tensor, mask_aug, tileName, img_orig_tensor = train_dataset[dataset_idx]
        
        # Convert to numpy
        img_original = tensor_to_numpy_image(img_orig_tensor)
        img_augmented = tensor_to_numpy_image(img_aug_tensor)
        mask = mask_aug.cpu().numpy() if isinstance(mask_aug, torch.Tensor) else mask_aug
        
        # Plot: Original Image, Original Mask, Aug Image, Aug Mask, Overlay
        col_offset = row_idx * 5
        
        # Original image
        ax = plt.subplot(num_samples, 5, col_offset + 1)
        ax.imshow(img_original)
        if row_idx == 0:
            ax.set_title("Original Image\n(Resized)", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{tileName}", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Original mask
        ax = plt.subplot(num_samples, 5, col_offset + 2)
        ax.imshow(mask_to_rgb(mask))
        if row_idx == 0:
            ax.set_title("Mask", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Augmented image
        ax = plt.subplot(num_samples, 5, col_offset + 3)
        ax.imshow(img_augmented)
        if row_idx == 0:
            ax.set_title("Augmented Image", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Augmented mask
        ax = plt.subplot(num_samples, 5, col_offset + 4)
        ax.imshow(mask_to_rgb(mask))
        if row_idx == 0:
            ax.set_title("Augmented Mask", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Overlay
        ax = plt.subplot(num_samples, 5, col_offset + 5)
        ax.imshow(overlay_mask_on_image(img_augmented, mask, alpha=0.4))
        if row_idx == 0:
            ax.set_title("Overlay", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=np.array(color)/255, label=CLASS_NAMES[class_id])
        for class_id, color in CLASS_COLORS.items()
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
              title="Classes", title_fontsize=11, ncol=5, framealpha=0.9)
    
    plt.suptitle("Data Augmentation: Before vs After (Training Set)", 
                fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = os.path.join(output_dir, "comparison_grid.jpg")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Grille comparative sauvegardée: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data augmentations using DataLoader")
    parser.add_argument("--config", type=str, 
                       default="Todo_List/DefaultExp_WithWeights.yaml",
                       help="Path to YAML config file")
    parser.add_argument("--num_samples", type=int, default=15,
                       help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, 
                       default="Results/Augmentation_Viz",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run visualization
    visualize_augmentations(
        config_path=args.config,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"✅ VISUALISATION TERMINÉE!")
    print(f"{'='*80}")
    print(f"\n📊 Pour votre rapport:")
    print(f"   1. Consultez {args.output_dir}/")
    print(f"   2. 'comparison_grid.jpg' montre l'impact général")
    print(f"   3. 'augmentation_*.jpg' montrent des exemples détaillés")
    print(f"\n💡 Usage:")
    print(f"   python visualize_augmentations.py --num_samples 10 --seed 123")
    print(f"\n{'='*80}\n")