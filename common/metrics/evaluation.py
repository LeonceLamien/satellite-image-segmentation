"""
Evaluation Metrics
==================

Calculate various metrics for segmentation and classification tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Optional, Tuple


class MetricsCalculator:
    """Calculate and store evaluation metrics."""
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes for segmentation
        """
        self.num_classes = num_classes
        self.metrics = {}
    
    def calculate_basic_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            true_labels: Ground truth labels (flattened)
            pred_labels: Predicted labels (flattened)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'iou': jaccard_score(true_labels, pred_labels, average='weighted', zero_division=0)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_per_class_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate per-class metrics.
        
        Args:
            true_labels: Ground truth labels (flattened)
            pred_labels: Predicted labels (flattened)
        
        Returns:
            Dictionary of per-class metrics
        """
        per_class = {
            'precision': precision_score(true_labels, pred_labels, average=None, zero_division=0),
            'recall': recall_score(true_labels, pred_labels, average=None, zero_division=0),
            'f1_score': f1_score(true_labels, pred_labels, average=None, zero_division=0),
            'iou': jaccard_score(true_labels, pred_labels, average=None, zero_division=0)
        }
        
        self.metrics['per_class'] = per_class
        return per_class
    
    def calculate_confusion_matrix(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted labels
        
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(true_labels, pred_labels)
        self.metrics['confusion_matrix'] = cm
        return cm
    
    def calculate_roc_auc(
        self,
        true_labels: np.ndarray,
        output_probs: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Calculate ROC AUC for multi-class classification.
        
        Args:
            true_labels: Ground truth labels (flattened)
            output_probs: Output probabilities [N, num_classes]
        
        Returns:
            Dictionary with fpr, tpr, and roc_auc for each class
        """
        # Binarize labels for multi-class ROC
        true_binary = label_binarize(true_labels, classes=range(self.num_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Calculate ROC for each class
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(true_binary[:, i], output_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(true_binary.ravel(), output_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        self.metrics['roc'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    def get_summary(self) -> str:
        """
        Get formatted summary of metrics.
        
        Returns:
            Formatted string with metrics
        """
        summary = "Evaluation Metrics:\n"
        summary += "=" * 50 + "\n"
        
        if 'accuracy' in self.metrics:
            summary += f"Accuracy:  {self.metrics['accuracy']:.4f}\n"
        if 'precision' in self.metrics:
            summary += f"Precision: {self.metrics['precision']:.4f}\n"
        if 'recall' in self.metrics:
            summary += f"Recall:    {self.metrics['recall']:.4f}\n"
        if 'f1_score' in self.metrics:
            summary += f"F1-Score:  {self.metrics['f1_score']:.4f}\n"
        if 'iou' in self.metrics:
            summary += f"IoU:       {self.metrics['iou']:.4f}\n"
        
        if 'roc' in self.metrics and 'auc' in self.metrics['roc']:
            summary += "\nROC AUC:\n"
            roc_auc = self.metrics['roc']['auc']
            for i in range(self.num_classes):
                if i in roc_auc:
                    summary += f"  Class {i}: {roc_auc[i]:.4f}\n"
            if 'micro' in roc_auc:
                summary += f"  Micro-avg: {roc_auc['micro']:.4f}\n"
            if 'macro' in roc_auc:
                summary += f"  Macro-avg: {roc_auc['macro']:.4f}\n"
        
        return summary
    
    def save_to_file(self, filepath: str):
        """
        Save metrics to text file.
        
        Args:
            filepath: Path to save metrics
        """
        with open(filepath, 'w') as f:
            f.write(self.get_summary())
