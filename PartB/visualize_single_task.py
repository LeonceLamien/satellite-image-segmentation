"""
Visualize results for a single proxy task.

Shows task-specific visualizations:
- Denoising: Noisy vs Denoised vs Clean
- Colorization: Grayscale vs Colorized vs Original
- Context Prediction: Masked vs Predicted vs Original
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import argparse


def visualize_denoising(model, dataset, n_samples=9):
    """Visualize denoising results."""
    from proxy_tasks.denoising import DenoisingAutoencoder
    
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(n_cols * 9, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            idx = int(i * len(dataset) / n_samples)
            noisy, clean, tile_name = dataset[idx]
            
            output = model(noisy.unsqueeze(0)).squeeze(0)
            
            noisy_img = noisy.numpy().transpose(1, 2, 0)
            clean_img = clean.numpy().transpose(1, 2, 0)
            output_img = np.clip(output.numpy().transpose(1, 2, 0), 0, 1)
            
            mse = np.mean((output_img - clean_img) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            row = i // n_cols
            col_start = (i % n_cols) * 3
            
            axes[row, col_start].imshow(noisy_img)
            axes[row, col_start].set_title(f'Noisy\n{tile_name}', fontsize=10)
            axes[row, col_start].axis('off')
            
            axes[row, col_start + 1].imshow(output_img)
            axes[row, col_start + 1].set_title(f'Denoised\nPSNR: {psnr:.1f} dB', fontsize=10, color='blue')
            axes[row, col_start + 1].axis('off')
            
            axes[row, col_start + 2].imshow(clean_img)
            axes[row, col_start + 2].set_title('Clean', fontsize=10)
            axes[row, col_start + 2].axis('off')
    
    plt.suptitle('Denoising Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def visualize_colorization(model, dataset, n_samples=9):
    """Visualize colorization results."""
    from proxy_tasks.colorization import ColorizationAutoencoder
    
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(n_cols * 9, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            idx = int(i * len(dataset) / n_samples)
            gray, rgb, tile_name = dataset[idx]
            
            output = model(gray.unsqueeze(0)).squeeze(0)
            
            gray_img = gray.numpy().squeeze()
            gray_img = np.stack([gray_img] * 3, axis=-1)
            rgb_img = rgb.numpy().transpose(1, 2, 0)
            output_img = np.clip(output.numpy().transpose(1, 2, 0), 0, 1)
            
            mse = np.mean((output_img - rgb_img) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            row = i // n_cols
            col_start = (i % n_cols) * 3
            
            axes[row, col_start].imshow(gray_img, cmap='gray')
            axes[row, col_start].set_title(f'Grayscale\n{tile_name}', fontsize=10)
            axes[row, col_start].axis('off')
            
            axes[row, col_start + 1].imshow(output_img)
            axes[row, col_start + 1].set_title(f'Colorized\nPSNR: {psnr:.1f} dB', fontsize=10, color='blue')
            axes[row, col_start + 1].axis('off')
            
            axes[row, col_start + 2].imshow(rgb_img)
            axes[row, col_start + 2].set_title('Original', fontsize=10)
            axes[row, col_start + 2].axis('off')
    
    plt.suptitle('Colorization Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def visualize_context_prediction(model, dataset, n_samples=9):
    """Visualize context prediction results."""
    from proxy_tasks.context_prediction import ContextPredictionAutoencoder
    
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(n_cols * 9, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            idx = int(i * len(dataset) / n_samples)
            masked, original, mask, tile_name = dataset[idx]  # 4 values: masked, original, mask, tile_name
            
            output = model(masked.unsqueeze(0)).squeeze(0)
            
            masked_img = masked.numpy().transpose(1, 2, 0)
            original_img = original.numpy().transpose(1, 2, 0)
            output_img = np.clip(output.numpy().transpose(1, 2, 0), 0, 1)
            
            mse = np.mean((output_img - original_img) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            row = i // n_cols
            col_start = (i % n_cols) * 3
            
            axes[row, col_start].imshow(masked_img)
            axes[row, col_start].set_title(f'Masked\n{tile_name}', fontsize=10)
            axes[row, col_start].axis('off')
            
            axes[row, col_start + 1].imshow(output_img)
            axes[row, col_start + 1].set_title(f'Predicted\nPSNR: {psnr:.1f} dB', fontsize=10, color='blue')
            axes[row, col_start + 1].axis('off')
            
            axes[row, col_start + 2].imshow(original_img)
            axes[row, col_start + 2].set_title('Original', fontsize=10)
            axes[row, col_start + 2].axis('off')
    
    plt.suptitle('Context Prediction Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def visualize_encoder_features(model, dataset, n_samples=6):
    """Visualize encoder feature maps."""
    
    n_rows = n_samples
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 4))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            idx = int(i * len(dataset) / n_samples)
            
            # Get data based on task
            data = dataset[idx]
            if len(data) == 4:
                # Context prediction returns 4 values
                input_img, target_img, mask, tile_name = data
            elif len(data) == 3:
                # Denoising and colorization return 3 values
                input_img, target_img, tile_name = data
            else:
                raise ValueError(f"Unexpected dataset output length: {len(data)}")
            
            # Get encoder features
            features = model.get_encoder_features(input_img.unsqueeze(0)).squeeze(0)  # (512, 8, 8)
            
            # Original image (use target for display)
            display_img = target_img.numpy()
            if display_img.shape[0] == 1:  # Grayscale
                display_img = np.stack([display_img[0]] * 3, axis=0)
            display_img = display_img.transpose(1, 2, 0)
            
            axes[i, 0].imshow(display_img)
            axes[i, 0].set_title(f'Image\n{tile_name}', fontsize=9)
            axes[i, 0].axis('off')
            
            # Feature statistics
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
            
            # Random channel
            random_channel = np.random.randint(0, feature_np.shape[0])
            axes[i, 4].imshow(feature_np[random_channel], cmap='coolwarm')
            axes[i, 4].set_title(f'Channel {random_channel}', fontsize=9)
            axes[i, 4].axis('off')
    
    plt.suptitle('Encoder Feature Maps (512 channels, 8×8 spatial)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize proxy task results')
    parser.add_argument('--task', type=str, required=True,
                        choices=['denoising', 'colorization', 'context_prediction'],
                        help='Task to visualize')
    parser.add_argument('--exp', type=str, required=True,
                        help='Experiment directory (e.g., Results/denoising_exp)')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path')
    parser.add_argument('--n_samples', type=int, default=9,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(f"VISUALIZING: {args.task.upper()}")
    print("=" * 70)
    
    # Load config
    with open(args.config, 'r') as f:
        param = yaml.safe_load(f)
    
    # Load model
    model_path = os.path.join(args.exp, '_Weights', 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Task-specific initialization
    if args.task == 'denoising':
        from proxy_tasks.denoising import DenoisingDataset, DenoisingAutoencoder
        model = DenoisingAutoencoder(param)
        dataset = DenoisingDataset("./Dataset/images", "val", param)
        
    elif args.task == 'colorization':
        from proxy_tasks.colorization import ColorizationDataset, ColorizationAutoencoder
        model = ColorizationAutoencoder(param)
        dataset = ColorizationDataset("./Dataset/images", "val", param)
        
    elif args.task == 'context_prediction':
        from proxy_tasks.context_prediction import ContextPredictionDataset, ContextPredictionAutoencoder
        model = ContextPredictionAutoencoder(param)
        dataset = ContextPredictionDataset("./Dataset/images", "val", param)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"✓ Model loaded: {model_path}")
    print(f"✓ Dataset loaded: {len(dataset)} images")
    
    # Create _Figures directory
    figures_dir = os.path.join(args.exp, '_Figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate task results visualization
    print(f"\nGenerating task results for {args.n_samples} samples...")
    
    if args.task == 'denoising':
        fig1 = visualize_denoising(model, dataset, args.n_samples)
    elif args.task == 'colorization':
        fig1 = visualize_colorization(model, dataset, args.n_samples)
    elif args.task == 'context_prediction':
        fig1 = visualize_context_prediction(model, dataset, args.n_samples)
    
    # Save task results
    save_path1 = os.path.join(figures_dir, f'{args.task}_results.png')
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path1}")
    plt.close()
    
    # Generate encoder features visualization
    print(f"\nGenerating encoder features visualization...")
    fig2 = visualize_encoder_features(model, dataset, n_samples=6)
    
    # Save encoder features
    save_path2 = os.path.join(figures_dir, f'{args.task}_encoder_features.png')
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path2}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Visualization completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
