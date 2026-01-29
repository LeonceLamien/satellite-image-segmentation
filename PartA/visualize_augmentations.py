import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
from Dataset.dataLoader import LLNDataset
import os
from matplotlib.colors import ListedColormap

CLASS_COLORS = [
    'black',  # Class 0
    'blue',  # Class 1
    'red',  # Class 2
    'yellow',  # Class 3
    'green',  # Class 4
]

CUSTOM_CMAP = ListedColormap(CLASS_COLORS)

def visualize_augmentations(config_file, num_samples=5, save_dir="augmentation_examples"):
    """
    Generate plots comparing original and augmented images
    
    Args:
        config_file: Path to YAML config file
        num_samples: Number of samples to visualize
        save_dir: Directory where visualizations will be saved
    """
    # Load config
    with open(config_file, 'r') as f:
        param = yaml.safe_load(f)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Paths
    imgDirectory = "./Dataset/images"
    maskDirectory = "./Dataset/annotations"
    
    # Create two datasets: one without augmentation, one with
    param_no_aug = param.copy()
    param_no_aug["DATASET"] = param["DATASET"].copy()
    param_no_aug["DATASET"]["USE_AUGMENTATION"] = False
    
    # For augmented dataset, force p=1.0 to see transformations
    param_aug = param.copy()
    param_aug["DATASET"] = param["DATASET"].copy()
    param_aug["DATASET"]["USE_AUGMENTATION"] = True
    param_aug["DATASET"]["FORCE_AUGMENTATION"] = True
    
    dataset_original = LLNDataset(imgDirectory, maskDirectory, "train", param_no_aug)
    dataset_augmented = LLNDataset(imgDirectory, maskDirectory, "train", param_aug)
    
    print(f"✓ Original dataset created (USE_AUGMENTATION={param_no_aug['DATASET'].get('USE_AUGMENTATION', False)})")
    print(f"✓ Augmented dataset created (USE_AUGMENTATION={param_aug['DATASET'].get('USE_AUGMENTATION', False)})")
    print(f"  Active augmentations:")
    if param_aug["DATASET"].get("HORIZONTAL_FLIP"):
        print(f"    - Horizontal Flip")
    if param_aug["DATASET"].get("VERTICAL_FLIP"):
        print(f"    - Vertical Flip")
    if param_aug["DATASET"].get("ROTATE_LIMIT", 0) > 0:
        print(f"    - Rotation (±{param_aug['DATASET'].get('ROTATE_LIMIT')}°)")
    if param_aug["DATASET"].get("BRIGHTNESS_CONTRAST"):
        print(f"    - Brightness/Contrast")
    if param_aug["DATASET"].get("BLUR"):
        print(f"    - Blur")
    if param_aug["DATASET"].get("COLOR_JITTER"):
        print(f"    - Color Jitter")
    print()
    
    # Generate visualizations
    for idx in range(min(num_samples, len(dataset_original))):
        # Get original image
        img_orig, mask_orig, tile_name, _ = dataset_original[idx]
        
        # Generate MULTIPLE augmented versions to ensure we see the changes
        _, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        img_orig_np = denormalize_image(img_orig)
        mask_orig_np = mask_orig.numpy() if isinstance(mask_orig, torch.Tensor) else mask_orig
        
        axes[0, 0].imshow(img_orig_np)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(mask_orig_np, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[1, 0].set_title('Original Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Generate 2 different augmented versions
        for aug_idx in range(2):
            img_aug, mask_aug, _, _ = dataset_augmented[idx]
            img_aug_np = denormalize_image(img_aug)
            mask_aug_np = mask_aug.numpy() if isinstance(mask_aug, torch.Tensor) else mask_aug
            
            col = aug_idx + 1
            axes[0, col].imshow(img_aug_np)
            axes[0, col].set_title(f'Augmented Image #{aug_idx+1}', fontsize=14, fontweight='bold')
            axes[0, col].axis('off')
            
            axes[1, col].imshow(mask_aug_np, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
            axes[1, col].set_title(f'Augmented Mask #{aug_idx+1}', fontsize=14, fontweight='bold')
            axes[1, col].axis('off')
        
        plt.suptitle(f'Original vs Augmented Comparison - {tile_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(save_dir, f'augmentation_comparison_{idx+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    print(f"\n{num_samples} visualizations generated in '{save_dir}/'")
    
    # Also create a grid with multiple examples
    create_augmentation_grid(dataset_original, dataset_augmented, 
                            num_samples=min(6, num_samples), 
                            save_dir=save_dir)


def denormalize_image(img_tensor):
    """Denormalize an image tensor for visualization"""
    # ImageNet normalization values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.numpy()
    else:
        img = img_tensor
    
    # img shape: (C, H, W) -> transpose to (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    
    # Denormalize
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    return img


def create_augmentation_grid(dataset_original, dataset_augmented, num_samples=6, save_dir="."):
    """Create a grid with multiple augmentation examples"""
    
    _, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for idx in range(num_samples):
        # Get images
        img_orig, mask_orig, tile_name, _ = dataset_original[idx]
        img_aug, mask_aug, _, _ = dataset_augmented[idx]
        
        # Denormalize
        img_orig_np = denormalize_image(img_orig)
        img_aug_np = denormalize_image(img_aug)
        mask_orig_np = mask_orig.numpy() if isinstance(mask_orig, torch.Tensor) else mask_orig
        mask_aug_np = mask_aug.numpy() if isinstance(mask_aug, torch.Tensor) else mask_aug
        
        # Display
        axes[idx, 0].imshow(img_orig_np)
        axes[idx, 0].set_title('Original' if idx == 0 else '', fontsize=12)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask_orig_np, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[idx, 1].set_title('Original Mask' if idx == 0 else '', fontsize=12)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(img_aug_np)
        axes[idx, 2].set_title('Augmented' if idx == 0 else '', fontsize=12)
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(mask_aug_np, cmap=CUSTOM_CMAP, vmin=0, vmax=4)
        axes[idx, 3].set_title('Augmented Mask' if idx == 0 else '', fontsize=12)
        axes[idx, 3].axis('off')
        
        # Add filename on the left side
        axes[idx, 0].text(-0.1, 0.5, tile_name, 
                         transform=axes[idx, 0].transAxes,
                         fontsize=9, rotation=90, va='center')
    
    plt.suptitle('Comparison Grid: Original vs Augmented', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'augmentation_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grid saved: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize data augmentations')
    parser.add_argument('-exp', type=str, default='DeepLab_Adam_Augmented',
                       help='Config filename (without .yaml)')
    parser.add_argument('-n', type=int, default=5,
                       help='Number of samples to generate')
    parser.add_argument('-o', type=str, default='augmentation_examples',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config_path = f'Todo_List/{args.exp}.yaml'
    
    print(f"Loading config: {config_path}")
    print(f"Generating {args.n} visualizations...")
    print(f"Saving to: {args.o}/\n")
    
    visualize_augmentations(config_path, num_samples=args.n, save_dir=args.o)
    
    print(f"\nDone! Check images in '{args.o}/'")
