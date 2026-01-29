"""
Part A - Segmentation Trainer
==============================

Trainer for semantic segmentation with DeepLab and U-Net architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common.training import BaseTrainer
from common.metrics import MetricsCalculator, plot_confusion_matrix, plot_roc_curves
from common.utils import create_folder
from Networks.Architectures.deepLab import DeepLabV3Plus
from Networks.Architectures.improvedUNet import ImprovedUNet
from Dataset.makeGraph import reconstruct_from_tiles


class SegmentationTrainer(BaseTrainer):
    """Trainer for segmentation models (Part A)."""
    
    def __init__(self, config, img_directory, mask_directory, results_path, device='cuda'):
        """
        Initialize segmentation trainer.
        
        Args:
            config: Configuration dictionary
            img_directory: Path to images folder
            mask_directory: Path to masks folder
            results_path: Path to save results
            device: Device to use
        """
        super().__init__(config, device, results_path)
        
        self.img_directory = img_directory
        self.mask_directory = mask_directory
        
        # Build model
        self.model = self._build_model()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _build_model(self) -> nn.Module:
        """Build segmentation model based on config."""
        model_arch = self.config.get('MODEL', {}).get('ARCHITECTURE', 'DeepLabV3Plus')
        
        if model_arch == 'DeepLabV3Plus':
            model = DeepLabV3Plus(self.config)
        elif model_arch == 'ImprovedUNet':
            model = ImprovedUNet(self.config)
        else:
            raise ValueError(f"Unknown architecture: {model_arch}")
        
        return model.to(self.device)
    
    def _compute_loss(self, outputs, targets, **kwargs) -> torch.Tensor:
        """Compute segmentation loss."""
        targets = targets.long()
        return self.criterion(outputs, targets)
    
    def _process_batch(self, batch):
        """Process segmentation batch."""
        images, masks, tile_names, resized_imgs = batch
        return images, masks, {'tile_names': tile_names, 'resized_imgs': resized_imgs}
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
        """
        print("\n" + "="*60)
        print("Starting Evaluation")
        print("="*60)
        
        self.model.eval()
        
        all_masks = []
        all_preds = []
        all_outputs = []
        all_tile_names = []
        all_resized_imgs = []
        
        with torch.no_grad():
            for images, masks, tile_names, resized_imgs in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Move to CPU
                outputs = outputs.cpu()
                
                # Store raw outputs for ROC
                all_outputs.extend(outputs.numpy())
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                all_masks.extend(masks.numpy())
                all_preds.extend(preds.numpy())
                all_resized_imgs.extend(resized_imgs.numpy())
                all_tile_names.extend(tile_names)
        
        # Convert to arrays
        all_masks = np.array(all_masks)
        all_preds = np.array(all_preds)
        all_resized_imgs = np.array(all_resized_imgs)
        all_outputs = np.array(all_outputs)
        
        # Qualitative evaluation
        test_path = Path(self.results_path) / "Test"
        create_folder(test_path)
        reconstruct_from_tiles(all_resized_imgs, all_preds, all_masks, all_tile_names, str(test_path))
        
        # Quantitative evaluation
        self._quantitative_evaluation(all_masks, all_preds, all_outputs)
    
    def _quantitative_evaluation(self, true_masks, pred_masks, outputs):
        """Calculate and save metrics."""
        # Flatten arrays
        true_flat = true_masks.flatten()
        pred_flat = pred_masks.flatten()
        
        # Calculate metrics
        calculator = MetricsCalculator(num_classes=5)
        
        # Basic metrics
        basic_metrics = calculator.calculate_basic_metrics(true_flat, pred_flat)
        print("\n" + "="*60)
        print("Quantitative Evaluation Results")
        print("="*60)
        for metric_name, value in basic_metrics.items():
            print(f"{metric_name.capitalize():12s}: {value:.4f}")
        
        # Per-class metrics
        per_class = calculator.calculate_per_class_metrics(true_flat, pred_flat)
        print("\nPer-Class Metrics:")
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10}")
        print("-" * 50)
        for i in range(5):
            print(f"Class {i}   {per_class['precision'][i]:.4f}     "
                  f"{per_class['recall'][i]:.4f}     {per_class['f1_score'][i]:.4f}     "
                  f"{per_class['iou'][i]:.4f}")
        
        # Confusion matrix
        cm = calculator.calculate_confusion_matrix(true_flat, pred_flat)
        cm_path = Path(self.results_path) / "confusion_matrix.png"
        plot_confusion_matrix(cm, str(cm_path), normalize=False)
        print(f"\n✓ Confusion matrix saved to {cm_path}")
        
        # Normalized confusion matrix
        cm_norm_path = Path(self.results_path) / "confusion_matrix_normalized.png"
        plot_confusion_matrix(cm, str(cm_norm_path), normalize=True, 
                            title='Normalized Confusion Matrix')
        print(f"✓ Normalized confusion matrix saved to {cm_norm_path}")
        
        # ROC AUC
        outputs_flat = outputs.transpose(0, 2, 3, 1).reshape(-1, 5)
        roc_data = calculator.calculate_roc_auc(true_flat, outputs_flat)
        
        print("\nROC AUC Results:")
        for i in range(5):
            print(f"  Class {i} AUC: {roc_data['auc'][i]:.4f}")
        print(f"  Micro-avg AUC: {roc_data['auc']['micro']:.4f}")
        print(f"  Macro-avg AUC: {roc_data['auc']['macro']:.4f}")
        
        roc_path = Path(self.results_path) / "roc_curves.png"
        plot_roc_curves(roc_data['fpr'], roc_data['tpr'], roc_data['auc'], 5, str(roc_path))
        print(f"\n✓ ROC curves saved to {roc_path}")
        
        # Save metrics to file
        metrics_path = Path(self.results_path) / "test_metrics.txt"
        calculator.save_to_file(str(metrics_path))
        print(f"✓ Metrics saved to {metrics_path}")
