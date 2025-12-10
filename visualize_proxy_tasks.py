"""
Visualize examples from each proxy task to verify they work correctly.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys

sys.path.append('.')

from proxy_tasks.denoising import DenoisingDataset
from proxy_tasks.colorization import ColorizationDataset
from proxy_tasks.context_prediction import ContextPredictionDataset


def visualize_denoising():
    """Visualize denoising task examples."""
    print("\n[1] Visualizing DENOISING task...")
    
    param = {"DATASET": {"RESIZE_SHAPE": "64x64"}}
    dataset = DenoisingDataset("./Dataset/images", "train", param, noise_std=0.15)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i in range(3):
        noisy, clean, name = dataset[i * 200]
        
        # Convert to numpy for visualization
        noisy_img = noisy.numpy().transpose(1, 2, 0)
        clean_img = clean.numpy().transpose(1, 2, 0)
        
        # Noisy
        axes[0, i].imshow(noisy_img)
        axes[0, i].set_title(f'Noisy\n{name}', fontsize=10)
        axes[0, i].axis('off')
        
        # Clean
        axes[1, i].imshow(clean_img)
        axes[1, i].set_title('Clean (Target)', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('DENOISING TASK: Add Noise → Predict Clean', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_denoising.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visualization_denoising.png")
    plt.close()


def visualize_colorization():
    """Visualize colorization task examples."""
    print("\n[2] Visualizing COLORIZATION task...")
    
    param = {"DATASET": {"RESIZE_SHAPE": "64x64"}}
    dataset = ColorizationDataset("./Dataset/images", "train", param)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i in range(3):
        gray, rgb, name = dataset[i * 200]
        
        # Convert to numpy for visualization
        gray_img = gray.numpy().squeeze()  # Remove channel dim for grayscale
        rgb_img = rgb.numpy().transpose(1, 2, 0)
        
        # Grayscale input
        axes[0, i].imshow(gray_img, cmap='gray')
        axes[0, i].set_title(f'Grayscale\n{name}', fontsize=10)
        axes[0, i].axis('off')
        
        # RGB target
        axes[1, i].imshow(rgb_img)
        axes[1, i].set_title('RGB (Target)', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('COLORIZATION TASK: Grayscale → Predict RGB Colors', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_colorization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visualization_colorization.png")
    plt.close()


def visualize_context_prediction():
    """Visualize context prediction task examples."""
    print("\n[3] Visualizing CONTEXT PREDICTION task...")
    
    param = {"DATASET": {"RESIZE_SHAPE": "64x64"}}
    dataset = ContextPredictionDataset("./Dataset/images", "train", param, 
                                      mask_ratio=0.3, patch_size=8)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(3):
        masked, original, mask, name = dataset[i * 200]
        
        # Convert to numpy for visualization
        masked_img = masked.numpy().transpose(1, 2, 0)
        original_img = original.numpy().transpose(1, 2, 0)
        mask_img = mask.numpy().squeeze()
        
        # Masked input
        axes[0, i].imshow(masked_img)
        axes[0, i].set_title(f'Masked Input\n{name}', fontsize=10)
        axes[0, i].axis('off')
        
        # Mask visualization
        axes[1, i].imshow(mask_img, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1, i].set_title(f'Mask ({mask_img.mean()*100:.1f}% masked)', fontsize=10)
        axes[1, i].axis('off')
        
        # Original target
        axes[2, i].imshow(original_img)
        axes[2, i].set_title('Original (Target)', fontsize=10)
        axes[2, i].axis('off')
    
    plt.suptitle('CONTEXT PREDICTION TASK: Masked Image → Predict Complete', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_context_prediction.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visualization_context_prediction.png")
    plt.close()


def main():
    print("=" * 70)
    print("VISUALIZING PROXY TASKS")
    print("=" * 70)
    
    visualize_denoising()
    visualize_colorization()
    visualize_context_prediction()
    
    print("\n" + "=" * 70)
    print("✅ All visualizations created!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - visualization_denoising.png")
    print("  - visualization_colorization.png")
    print("  - visualization_context_prediction.png")


if __name__ == "__main__":
    main()
