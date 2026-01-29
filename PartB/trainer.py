"""
Part B - Self-Supervised Learning Trainer
==========================================

Trainer for self-supervised proxy tasks (denoising, colorization, context prediction).
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common.training import BaseTrainer
from common.metrics import plot_learning_curves


class SSLTrainer(BaseTrainer):
    """Trainer for self-supervised learning proxy tasks (Part B)."""
    
    def __init__(self, config, task_name, model, loss_fn, device='cpu', results_path=None):
        """
        Initialize SSL trainer.
        
        Args:
            config: Configuration dictionary
            task_name: Name of proxy task
            model: Model architecture
            loss_fn: Loss function
            device: Device to use
            results_path: Path to save results
        """
        super().__init__(config, device, results_path)
        
        self.task_name = task_name
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        
        # Setup optimizer
        self._setup_optimizer()
    
    def _build_model(self) -> nn.Module:
        """Model is passed in constructor for Part B."""
        return self.model
    
    def _compute_loss(self, outputs, targets, **kwargs) -> torch.Tensor:
        """Compute loss using the provided loss function."""
        # For context prediction, we need the mask
        if self.task_name == 'context_prediction' and 'mask' in kwargs:
            masked_only = self.config.get('TASK', {}).get('MASKED_ONLY_LOSS', False)
            return self.loss_fn(outputs, targets, kwargs['mask'], masked_only=masked_only)
        else:
            return self.loss_fn(outputs, targets)
    
    def _process_batch(self, batch):
        """Process batch based on task type."""
        if self.task_name in ['denoising', 'colorization']:
            input_img, target_img, _ = batch
            return input_img, target_img, {}
        else:  # context_prediction
            masked_img, original_img, mask, _ = batch
            return masked_img, original_img, {'mask': mask}
    
    def save_results(self):
        """Save model and training curves."""
        if self.results_path is None:
            return
        
        results_path = Path(self.results_path)
        weights_path = results_path / "_Weights"
        figures_path = results_path / "_Figures"
        
        weights_path.mkdir(parents=True, exist_ok=True)
        figures_path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        torch.save(self.best_model_weights, weights_path / "best_model.pth")
        print(f"\n✓ Best model saved to {weights_path / 'best_model.pth'}")
        
        # Save latest checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, weights_path / "checkpoint_latest.pth")
        
        # Save training curves
        curves_path = figures_path / "training_curves.png"
        plot_learning_curves(
            self.train_losses,
            self.val_losses,
            str(curves_path),
            title=f"{self.task_name.capitalize()} - Training Progress"
        )
        print(f"✓ Training curves saved to {curves_path}")
    
    def load_weights(self, weights_path=None):
        """Load model weights."""
        if weights_path is None:
            weights_path = Path(self.results_path) / "_Weights" / "best_model.pth"
        
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"✓ Loaded weights from {weights_path}")
