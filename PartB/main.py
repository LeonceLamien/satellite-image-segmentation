"""
PART B - MAIN TRAINING SCRIPT
==============================

Train encoder-decoder on self-supervised proxy tasks:
1. Denoising (simplest)
2. Colorization
3. Context Prediction

Usage:
    python main.py --task denoising --config configs/denoising.yaml
    python main.py --task colorization --config configs/colorization.yaml
    python main.py --task context_prediction --config configs/context_pred.yaml
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.config import load_config
from trainer import SSLTrainer

# Import proxy task modules
from proxy_tasks.denoising import (
    DenoisingDataset, 
    DenoisingAutoencoder, 
    denoise_loss
)
from proxy_tasks.colorization import (
    ColorizationDataset,
    ColorizationAutoencoder,
    colorization_loss
)
from proxy_tasks.context_prediction import (
    ContextPredictionDataset,
    ContextPredictionAutoencoder,
    context_prediction_loss
)


def create_datasets(task_name, config):
    """Create datasets based on task."""
    img_dir = "./Dataset/images"
    
    if task_name == "denoising":
        noise_std = config.task.get("NOISE_STD", 0.1)
        train_dataset = DenoisingDataset(img_dir, "train", config.data, noise_std)
        val_dataset = DenoisingDataset(img_dir, "val", config.data, noise_std)
        
    elif task_name == "colorization":
        train_dataset = ColorizationDataset(img_dir, "train", config.data)
        val_dataset = ColorizationDataset(img_dir, "val", config.data)
        
    elif task_name == "context_prediction":
        mask_ratio = config.task.get("MASK_RATIO", 0.3)
        patch_size = config.task.get("PATCH_SIZE", 8)
        train_dataset = ContextPredictionDataset(img_dir, "train", config.data, mask_ratio, patch_size)
        val_dataset = ContextPredictionDataset(img_dir, "val", config.data, mask_ratio, patch_size)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    return train_dataset, val_dataset


def create_model_and_loss(task_name, config):
    """Create model and loss function based on task."""
    if task_name == "denoising":
        model = DenoisingAutoencoder(config.data)
        loss_fn = denoise_loss
        
    elif task_name == "colorization":
        model = ColorizationAutoencoder(config.data)
        loss_fn = colorization_loss
        
    elif task_name == "context_prediction":
        model = ContextPredictionAutoencoder(config.data)
        # Loss function handles mask internally
        loss_fn = lambda pred, target, mask, masked_only=False: context_prediction_loss(
            pred, target, mask, masked_only=masked_only
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    return model, loss_fn


def main(args):
    """Main training procedure."""
    print("\n" + "="*70)
    print(f"PART B - SELF-SUPERVISED LEARNING: {args.task.upper()}")
    print("="*70)
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n✓ Configuration loaded from {args.config}")
    
    # Setup device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Create datasets
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print("="*70)
    
    train_dataset, val_dataset = create_datasets(args.task, config)
    
    print(f"✓ Train set: {len(train_dataset)} images")
    print(f"✓ Val set:   {len(val_dataset)} images")
    
    # Create dataloaders
    batch_size = config.training.get('BATCH_SIZE', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model and loss
    model, loss_fn = create_model_and_loss(args.task, config)
    print(f"✓ Model created: {args.task}")
    
    # Create results directory
    save_dir = Path(f"Results/{args.task}_exp")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = SSLTrainer(
        config=config.data,
        task_name=args.task,
        model=model,
        loss_fn=loss_fn,
        device=device,
        results_path=str(save_dir)
    )
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING")
    print("="*70)
    trainer.train(train_loader, val_loader)
    print(f"\n✓ Training completed!")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part B - Self-Supervised Learning')
    parser.add_argument('--task', type=str, required=True,
                       choices=['denoising', 'colorization', 'context_prediction'],
                       help='Proxy task to train')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Check config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    main(args)
