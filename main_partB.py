"""
PART B - MAIN TRAINING SCRIPT
==============================

Train encoder-decoder on self-supervised proxy tasks:
1. Denoising (simplest)
2. Colorization
3. Context Prediction

Usage:
    python main_partB.py --task denoising --config configs/denoising.yaml
    python main_partB.py --task colorization --config configs/colorization.yaml
    python main_partB.py --task context_prediction --config configs/context_pred.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import from Part A
sys.path.append('..')

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


class SSLTrainer:
    """
    Self-Supervised Learning Trainer for proxy tasks.
    """
    
    def __init__(self, config_path, task_name, device='cpu'):
        """
        Args:
            config_path: Path to YAML config file
            task_name: "denoising", "colorization", or "context_prediction"
            device: "cpu" or "cuda"
        """
        self.task_name = task_name
        self.device = device
        
        # Load config
        with open(config_path, 'r') as f:
            self.param = yaml.safe_load(f)
        
        # Create save directory
        self.save_dir = f"Results/{task_name}_exp"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/_Weights", exist_ok=True)
        os.makedirs(f"{self.save_dir}/_Figures", exist_ok=True)
        
        # Initialize datasets, model, and optimizer
        self._init_datasets()
        self._init_model()
        self._init_optimizer()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def _init_datasets(self):
        """Initialize datasets based on task."""
        img_dir = "../Dataset/images"
        
        if self.task_name == "denoising":
            noise_std = self.param["TASK"].get("NOISE_STD", 0.1)
            self.train_dataset = DenoisingDataset(img_dir, "train", self.param, noise_std)
            self.val_dataset = DenoisingDataset(img_dir, "val", self.param, noise_std)
            
        elif self.task_name == "colorization":
            self.train_dataset = ColorizationDataset(img_dir, "train", self.param)
            self.val_dataset = ColorizationDataset(img_dir, "val", self.param)
            
        elif self.task_name == "context_prediction":
            mask_ratio = self.param["TASK"].get("MASK_RATIO", 0.3)
            patch_size = self.param["TASK"].get("PATCH_SIZE", 8)
            self.train_dataset = ContextPredictionDataset(
                img_dir, "train", self.param, mask_ratio, patch_size
            )
            self.val_dataset = ContextPredictionDataset(
                img_dir, "val", self.param, mask_ratio, patch_size
            )
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
        
        # Create data loaders
        batch_size = self.param["TRAIN"]["BATCH_SIZE"]
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"✓ Datasets loaded:")
        print(f"  Train: {len(self.train_dataset)} images")
        print(f"  Val: {len(self.val_dataset)} images")
    
    def _init_model(self):
        """Initialize model based on task."""
        if self.task_name == "denoising":
            self.model = DenoisingAutoencoder(self.param)
            self.loss_fn = denoise_loss
            
        elif self.task_name == "colorization":
            self.model = ColorizationAutoencoder(self.param)
            self.loss_fn = colorization_loss
            
        elif self.task_name == "context_prediction":
            self.model = ContextPredictionAutoencoder(self.param)
            # For context prediction, we need to pass mask
            self.loss_fn = lambda pred, target, mask: context_prediction_loss(
                pred, target, mask, 
                masked_only=self.param["TASK"].get("MASKED_ONLY_LOSS", False)
            )
        
        self.model.to(self.device)
        print(f"✓ Model created: {self.task_name}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        lr = self.param["TRAIN"]["LEARNING_RATE"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Optional: Learning rate scheduler
        if self.param["TRAIN"].get("USE_SCHEDULER", False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
        
        print(f"✓ Optimizer: Adam (lr={lr})")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Unpack batch based on task
            if self.task_name in ["denoising", "colorization"]:
                input_img, target_img, _ = batch_data
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)
                
                # Forward pass
                output = self.model(input_img)
                loss = self.loss_fn(output, target_img)
                
            else:  # context_prediction
                masked_img, original_img, mask, _ = batch_data
                masked_img = masked_img.to(self.device)
                original_img = original_img.to(self.device)
                mask = mask.to(self.device)
                
                # Forward pass
                output = self.model(masked_img)
                loss = self.loss_fn(output, original_img, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Unpack batch based on task
                if self.task_name in ["denoising", "colorization"]:
                    input_img, target_img, _ = batch_data
                    input_img = input_img.to(self.device)
                    target_img = target_img.to(self.device)
                    
                    output = self.model(input_img)
                    loss = self.loss_fn(output, target_img)
                    
                else:  # context_prediction
                    masked_img, original_img, mask, _ = batch_data
                    masked_img = masked_img.to(self.device)
                    original_img = original_img.to(self.device)
                    mask = mask.to(self.device)
                    
                    output = self.model(masked_img)
                    loss = self.loss_fn(output, original_img, mask)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, f"{self.save_dir}/_Weights/checkpoint_latest.pth")
        
        # Save best checkpoint
        if is_best:
            torch.save(self.model.state_dict(), 
                      f"{self.save_dir}/_Weights/best_model.pth")
            print(f"  ✓ Best model saved (val_loss: {self.best_val_loss:.4f})")
    
    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'{self.task_name.capitalize()} - Training Progress', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/_Figures/training_curves.png", dpi=150)
        plt.close()
    
    def train(self):
        """Main training loop."""
        num_epochs = self.param["TRAIN"]["NUM_EPOCHS"]
        
        print("\n" + "=" * 60)
        print(f"Starting training: {self.task_name.upper()}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch [{epoch}/{num_epochs}]")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best)
            
            # Plot losses
            if epoch % 5 == 0:
                self.plot_losses()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Results saved in: {self.save_dir}/")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train self-supervised proxy tasks')
    parser.add_argument('--task', type=str, required=True,
                       choices=['denoising', 'colorization', 'context_prediction'],
                       help='Proxy task to train')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    # Create trainer and train
    trainer = SSLTrainer(args.config, args.task, args.device)
    trainer.train()


if __name__ == "__main__":
    main()
