#!/usr/bin/env python3
"""
Script to visualize segmentation on multiple dataset images
Uses a trained model (wghts.pkl) to make predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import yaml
import os
import argparse
from pathlib import Path

from Dataset.dataLoader import LLNDataset
from Networks.Architectures.improvedUNet import ImprovedUNet
from Networks.Architectures.deepLab import DeepLabV3Plus

CUSTOM_CMAP = ListedColormap([
    "black",   # 0 = "other"
    "blue",    # 1 = "water"
    "red",     # 2 = "building"
    "yellow",  # 3 = "farmland"
    "green"    # 4 = "forest"
])

CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']


def load_model(config_path, weights_path, device='cpu'):
    """
    Load model with trained weights
    
    Args:
        config_path: Path to YAML config file
        weights_path: Path to weights (wghts.pkl)
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Loaded model in eval mode
        param: Parameter dictionary
    """
    # Load config
    with open(config_path, 'r') as f:
        param = yaml.safe_load(f)
    
    # Create model
    model_arch = param["MODEL"].get("ARCHITECTURE", "DeepLabV3Plus")
    if model_arch == "DeepLabV3Plus":
        model = DeepLabV3Plus(param)
    elif model_arch == "ImprovedUNet":
        model = ImprovedUNet(param)
    else:
        raise ValueError(f"Unknown architecture: {model_arch}")
    
    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {model_arch}")
    print(f"✓ Weights loaded: {weights_path}")
    
    return model, param


def denormalize_image(img_tensor):
    """Denormalize an image for visualization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor
    
    # (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    return img


def predict_and_visualize(model, dataset, indices, device='cpu', save_dir='segmentation_results'):
    """
    Make predictions and create visualizations
    
    Args:
        model: Trained model
        dataset: Dataset to use
        indices: List of image indices to process
        device: 'cpu' or 'cuda'
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nGenerating {len(indices)} visualizations...\n")
    
    for idx_num, idx in enumerate(indices):
        # Get image
        img_tensor, mask_gt, tile_name, _ = dataset[idx]
        
        # Prediction
        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
            output = model(img_input)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Denormalize image
        img_display = denormalize_image(img_tensor)
        mask_gt_np = mask_gt.numpy() if isinstance(mask_gt, torch.Tensor) else mask_gt
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=20)
        axes[0].axis('off')
        
        # Ground Truth
        im1 = axes[1].imshow(mask_gt_np, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[1].set_title('Ground Truth', fontsize=16, fontweight='bold', pad=20)
        axes[1].axis('off')
        
        # Prediction
        im2 = axes[2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[2].set_title('Model Prediction', fontsize=16, fontweight='bold', pad=20)
        axes[2].axis('off')
        
        # Calculate pixel-wise accuracy
        accuracy = np.mean(pred_mask == mask_gt_np) * 100
        
        plt.suptitle(f'{tile_name} - Accuracy: {accuracy:.2f}%', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        save_path = os.path.join(save_dir, f'segmentation_{idx_num+1:03d}_{tile_name}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ [{idx_num+1}/{len(indices)}] Saved: {save_path} (Acc: {accuracy:.2f}%)")
    
    print(f"\n{len(indices)} visualizations generated in '{save_dir}/'")
    
    # Also create a grid with multiple examples
    create_grid_visualization(model, dataset, indices[:min(6, len(indices))], device, save_dir)


def create_grid_visualization(model, dataset, indices, device='cpu', save_dir='.'):
    """Create a grid with multiple segmentation examples"""
    
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx_num, idx in enumerate(indices):
        # Get and predict
        img_tensor, mask_gt, tile_name, _ = dataset[idx]
        
        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(device)
            output = model(img_input)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        img_display = denormalize_image(img_tensor)
        mask_gt_np = mask_gt.numpy() if isinstance(mask_gt, torch.Tensor) else mask_gt
        
        # Image
        axes[idx_num, 0].imshow(img_display)
        axes[idx_num, 0].set_title('Original' if idx_num == 0 else '', fontsize=12)
        axes[idx_num, 0].axis('off')
        
        # Ground Truth
        axes[idx_num, 1].imshow(mask_gt_np, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[idx_num, 1].set_title('Ground Truth' if idx_num == 0 else '', fontsize=12)
        axes[idx_num, 1].axis('off')
        
        # Prediction
        axes[idx_num, 2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[idx_num, 2].set_title('Prediction' if idx_num == 0 else '', fontsize=12)
        axes[idx_num, 2].axis('off')
        
        # Accuracy on the side
        accuracy = np.mean(pred_mask == mask_gt_np) * 100
        axes[idx_num, 0].text(-0.1, 0.5, f'{tile_name}\nAcc: {accuracy:.1f}%', 
                             transform=axes[idx_num, 0].transAxes,
                             fontsize=10, rotation=0, va='center', ha='right')
    
    plt.suptitle('Segmentation Grid', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'segmentation_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grid saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('-exp', type=str, required=True,
                       help='Experiment name (e.g., DeepLab_Adam)')
    parser.add_argument('-set', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset to use (train/val/test)')
    parser.add_argument('-n', type=int, default=10,
                       help='Number of images to process')
    parser.add_argument('-start', type=int, default=0,
                       help='Starting index in dataset')
    parser.add_argument('-o', type=str, default=None,
                       help='Output directory (default: segmentation_results_{exp})')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Paths
    config_path = f'Todo_List/{args.exp}.yaml'
    weights_path = f'Results/{args.exp}/_Weights/wghts.pkl'
    
    # Check that files exist
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        print(f"   Tip: Train the model first with: python main.py -exp {args.exp}")
        return
    
    # Output directory
    if args.o is None:
        save_dir = f'segmentation_results_{args.exp}'
    else:
        save_dir = args.o
    
    print(f"Configuration:")
    print(f"   Experiment: {args.exp}")
    print(f"   Config: {config_path}")
    print(f"   Weights: {weights_path}")
    print(f"   Dataset: {args.set}")
    print(f"   Number of images: {args.n}")
    print(f"   Device: {args.device}")
    print(f"   Output: {save_dir}/\n")
    
    # Load model
    model, param = load_model(config_path, weights_path, args.device)
    
    # Load dataset
    imgDirectory = "./Dataset/images"
    maskDirectory = "./Dataset/annotations"
    dataset = LLNDataset(imgDirectory, maskDirectory, args.set, param)
    
    print(f"✓ Dataset loaded: {len(dataset)} images\n")
    
    # Select indices
    max_idx = min(args.start + args.n, len(dataset))
    indices = list(range(args.start, max_idx))
    
    if len(indices) == 0:
        print(f"No images to process (start={args.start}, dataset size={len(dataset)})")
        return
    
    # Generate visualizations
    predict_and_visualize(model, dataset, indices, args.device, save_dir)
    
    print(f"\nDone! Images are in '{save_dir}/'")
    print(f"Use these images for your report!")


if __name__ == "__main__":
    main()
