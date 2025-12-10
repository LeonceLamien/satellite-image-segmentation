"""
Visualize denoising results from trained model.
Shows: Noisy Input | Model Output | Ground Truth Clean
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

from proxy_tasks.denoising import DenoisingDataset, DenoisingAutoencoder


def visualize_denoising_results(model_path, config_path, n_samples=9, noise_std=0.1):
    """
    Visualize denoising results on validation set.
    
    Args:
        model_path: Path to trained model weights
        config_path: Path to config file
        n_samples: Number of samples to visualize (default 9 for 3x3 grid)
        noise_std: Noise level to test
    """
    
    print("=" * 70)
    print("VISUALIZING DENOISING RESULTS")
    print("=" * 70)
    
    # Load config
    print("\n[1] Loading config...")
    with open(config_path, 'r') as f:
        param = yaml.safe_load(f)
    print(f"✓ Config loaded from: {config_path}")
    
    # Create model
    print("\n[2] Loading model...")
    model = DenoisingAutoencoder(param)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"✓ Model loaded from: {model_path}")
    
    # Load validation dataset
    print("\n[3] Loading validation dataset...")
    val_dataset = DenoisingDataset(
        img_directory="./Dataset/images",
        set_name="val",
        param=param,
        noise_std=noise_std
    )
    print(f"✓ Dataset loaded: {len(val_dataset)} images")
    
    # Compute grid size
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    # Create visualization
    print(f"\n[4] Generating {n_samples} predictions...")
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(n_cols * 9, n_rows * 3))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(n_samples):
            # Get sample (spread across dataset)
            idx = int(i * len(val_dataset) / n_samples)
            noisy, clean, tile_name = val_dataset[idx]
            
            # Predict
            noisy_batch = noisy.unsqueeze(0)
            output = model(noisy_batch).squeeze(0)
            
            # Convert to numpy [0, 1]
            noisy_img = noisy.numpy().transpose(1, 2, 0)
            clean_img = clean.numpy().transpose(1, 2, 0)
            output_img = np.clip(output.numpy().transpose(1, 2, 0), 0, 1)
            
            # Calculate metrics
            mse = np.mean((output_img - clean_img) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            # Get subplot indices
            row = i // n_cols
            col_start = (i % n_cols) * 3
            
            # Noisy input
            axes[row, col_start].imshow(noisy_img)
            axes[row, col_start].set_title(f'Noisy Input\n{tile_name}', fontsize=10)
            axes[row, col_start].axis('off')
            
            # Model output
            axes[row, col_start + 1].imshow(output_img)
            axes[row, col_start + 1].set_title(f'Model Output\nPSNR: {psnr:.1f} dB', 
                                               fontsize=10, color='blue')
            axes[row, col_start + 1].axis('off')
            
            # Ground truth
            axes[row, col_start + 2].imshow(clean_img)
            axes[row, col_start + 2].set_title('Ground Truth\n(Clean)', fontsize=10)
            axes[row, col_start + 2].axis('off')
    
    plt.suptitle(f'Denoising Results (Noise std={noise_std})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    save_path = 'denoising_results_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✅ Visualization completed!")
    print("=" * 70)


def visualize_noise_levels(model_path, config_path):
    """
    Test model on different noise levels.
    """
    
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT NOISE LEVELS")
    print("=" * 70)
    
    # Load config and model
    with open(config_path, 'r') as f:
        param = yaml.safe_load(f)
    
    model = DenoisingAutoencoder(param)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Test on different noise levels
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    fig, axes = plt.subplots(len(noise_levels), 4, figsize=(16, len(noise_levels) * 4))
    
    # Use same image for all noise levels
    val_dataset = DenoisingDataset(
        img_directory="./Dataset/images",
        set_name="val",
        param=param,
        noise_std=0.1  # Just to load dataset
    )
    
    # Get one clean image
    _, clean_base, tile_name = val_dataset[50]
    clean_base_np = clean_base.numpy().transpose(1, 2, 0)
    
    with torch.no_grad():
        for i, noise_std in enumerate(noise_levels):
            # Add noise manually
            noise = np.random.normal(0, noise_std, clean_base_np.shape)
            noisy_np = np.clip(clean_base_np + noise, 0, 1).astype(np.float32)
            
            # Convert to tensor
            noisy = torch.from_numpy(noisy_np.transpose(2, 0, 1))
            
            # Predict
            output = model(noisy.unsqueeze(0)).squeeze(0)
            output_np = np.clip(output.numpy().transpose(1, 2, 0), 0, 1)
            
            # Calculate PSNR
            mse = np.mean((output_np - clean_base_np) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            # Clean reference (only on first row)
            if i == 0:
                axes[i, 0].imshow(clean_base_np)
                axes[i, 0].set_title(f'Clean Original\n{tile_name}', fontsize=11, fontweight='bold')
            else:
                axes[i, 0].imshow(clean_base_np)
                axes[i, 0].set_title('Clean Original', fontsize=11)
            axes[i, 0].axis('off')
            
            # Noisy
            axes[i, 1].imshow(noisy_np)
            axes[i, 1].set_title(f'Noisy (σ={noise_std})', fontsize=11)
            axes[i, 1].axis('off')
            
            # Denoised
            axes[i, 2].imshow(output_np)
            axes[i, 2].set_title(f'Denoised\nPSNR: {psnr:.1f} dB', fontsize=11, color='blue')
            axes[i, 2].axis('off')
            
            # Difference (amplified)
            diff = np.abs(output_np - clean_base_np) * 5  # Amplify for visibility
            axes[i, 3].imshow(diff)
            axes[i, 3].set_title('Error (×5)', fontsize=11)
            axes[i, 3].axis('off')
    
    plt.suptitle('Model Performance on Different Noise Levels', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = 'denoising_noise_levels.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    print("\n✅ Noise level analysis completed!")


def visualize_encoder_features(model_path, config_path):
    """
    Visualize encoder feature maps for clustering analysis.
    """
    
    print("\n" + "=" * 70)
    print("VISUALIZING ENCODER FEATURES")
    print("=" * 70)
    
    # Load config and model
    with open(config_path, 'r') as f:
        param = yaml.safe_load(f)
    
    model = DenoisingAutoencoder(param)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load validation dataset
    val_dataset = DenoisingDataset(
        img_directory="./Dataset/images",
        set_name="val",
        param=param,
        noise_std=0.1
    )
    
    # Extract features for several samples
    n_samples = 6
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, n_samples * 4))
    
    with torch.no_grad():
        for i in range(n_samples):
            idx = int(i * len(val_dataset) / n_samples)
            noisy, clean, tile_name = val_dataset[idx]
            
            # Get encoder features
            features = model.get_encoder_features(noisy.unsqueeze(0)).squeeze(0)  # (512, 8, 8)
            
            # Original image
            axes[i, 0].imshow(clean.numpy().transpose(1, 2, 0))
            axes[i, 0].set_title(f'Original\n{tile_name}', fontsize=9)
            axes[i, 0].axis('off')
            
            # Visualize 4 feature map statistics
            feature_np = features.numpy()
            
            # Mean across channels
            axes[i, 1].imshow(feature_np.mean(axis=0), cmap='viridis')
            axes[i, 1].set_title('Features Mean', fontsize=9)
            axes[i, 1].axis('off')
            
            # Std across channels
            axes[i, 2].imshow(feature_np.std(axis=0), cmap='plasma')
            axes[i, 2].set_title('Features Std', fontsize=9)
            axes[i, 2].axis('off')
            
            # Max across channels
            axes[i, 3].imshow(feature_np.max(axis=0), cmap='inferno')
            axes[i, 3].set_title('Features Max', fontsize=9)
            axes[i, 3].axis('off')
            
            # Random feature channel
            random_channel = np.random.randint(0, 512)
            axes[i, 4].imshow(feature_np[random_channel], cmap='coolwarm')
            axes[i, 4].set_title(f'Channel {random_channel}', fontsize=9)
            axes[i, 4].axis('off')
    
    plt.suptitle('Encoder Feature Maps (512 channels, 8×8 spatial)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = 'denoising_encoder_features.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    print(f"✓ Encoder output shape: (512, 8, 8) per image")
    print(f"✓ These features will be used for clustering")
    print("\n✅ Feature visualization completed!")


def main():
    # Paths
    model_path = 'Results/denoising_exp/_Weights/best_model.pth'
    config_path = 'configs/denoising.yaml'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first: python main_partB.py --task denoising --config configs/denoising.yaml")
        return
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    print("\n🎨 Starting visualizations...\n")
    
    # 1. Basic denoising results
    visualize_denoising_results(model_path, config_path, n_samples=9, noise_std=0.1)
    
    # 2. Encoder features
    visualize_encoder_features(model_path, config_path)
    
    print("\n" + "=" * 70)
    print("🎉 ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. denoising_results_visualization.png  - Basic denoising results")
    print("  2. denoising_encoder_features.png       - Feature maps for clustering")
    print("\nNext step: Run clustering")
    print("  python clustering/cluster_features.py --exp Results/denoising_exp --task denoising --n_clusters 5")


if __name__ == "__main__":
    main()
