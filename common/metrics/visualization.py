"""
Visualization Functions
=======================

Plot confusion matrices, ROC curves, learning curves, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix'
):
    """
    Plot and save confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        save_path: Path to save the plot
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        title: Plot title
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    cm = confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    fpr: Dict,
    tpr: Dict,
    roc_auc: Dict,
    num_classes: int,
    save_path: str
):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        fpr: False positive rates dictionary
        tpr: True positive rates dictionary
        roc_auc: ROC AUC scores dictionary
        num_classes: Number of classes
        save_path: Path to save the plot
    """
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot individual classes
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        ax1.plot(
            fpr[i], tpr[i],
            color=color,
            lw=2,
            label=f'Class {i} (AUC = {roc_auc[i]:.2f})'
        )
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Individual Classes')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Plot micro and macro averages
    ax2.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})',
        color='deeppink',
        linestyle=':',
        linewidth=4
    )
    
    ax2.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-avg (AUC = {roc_auc["macro"]:.2f})',
        color='navy',
        linestyle=':',
        linewidth=4
    )
    
    for i in range(num_classes):
        ax2.plot(fpr[i], tpr[i], lw=1, alpha=0.3)
    
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - Averages')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = 'Learning Curves'
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
