"""
Base Trainer
============

Abstract base class for training models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
import copy

from ..utils import create_folder
from ..metrics import plot_learning_curves
from .callbacks import EarlyStopping


class BaseTrainer(ABC):
    """
    Abstract base trainer class with common training functionality.
    
    Subclasses should implement:
        - _build_model()
        - _compute_loss()
        - _process_batch()
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cpu',
        results_path: Optional[str] = None
    ):
        """
        Initialize base trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to use ('cpu' or 'cuda')
            results_path: Path to save results
        """
        self.config = config
        self.device = device
        self.results_path = results_path
        
        # Training parameters from config
        training_cfg = config.get('TRAINING', config.get('TRAIN', {}))
        self.num_epochs = training_cfg.get('EPOCH', training_cfg.get('NUM_EPOCHS', 100))
        self.batch_size = training_cfg.get('BATCH_SIZE', 16)
        self.learning_rate = training_cfg.get('LEARNING_RATE', 0.001)
        self.weight_decay = training_cfg.get('WEIGHT_DECAY', 0.0)
        
        # Early stopping
        use_early_stopping = training_cfg.get('EARLY_STOPPING', False)
        patience = training_cfg.get('PATIENCE_EARLY_STOPPING', 
                                   training_cfg.get('EARLY_STOP_PATIENCE', 10))
        self.early_stopping = EarlyStopping(patience=patience) if use_early_stopping else None
        
        # Model, optimizer, scheduler (to be initialized by subclass)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_weights = None
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build and return the model.
        
        Returns:
            PyTorch model
        """
        pass
    
    @abstractmethod
    def _compute_loss(self, outputs, targets, **kwargs) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            outputs: Model outputs
            targets: Target labels/values
            **kwargs: Additional arguments
        
        Returns:
            Loss tensor
        """
        pass
    
    def _process_batch(self, batch):
        """
        Process a batch of data. Can be overridden by subclasses.
        
        Args:
            batch: Batch from dataloader
        
        Returns:
            Tuple of (inputs, targets, additional_data)
        """
        # Default implementation for simple cases
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            additional = batch[2:] if len(batch) > 2 else {}
            return inputs, targets, additional
        else:
            raise ValueError("Batch must contain at least inputs and targets")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        optimizer_name = self.config.get('TRAINING', {}).get('OPTIMIZER', 'Adam')
        
        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name.lower() in ['sgd', 'sdg']:  # Handle typo in original
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup scheduler if requested
        scheduler_type = self.config.get('TRAINING', {}).get('LR_SCHEDULER')
        if scheduler_type == 'ReduceLROnPlateau':
            patience = self.config.get('TRAINING', {}).get('LR_PATIENCE', 5)
            factor = self.config.get('TRAINING', {}).get('LR_FACTOR', 0.5)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=patience,
                factor=factor,
                verbose=True
            )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets, additional = self._process_batch(batch)
            inputs = inputs.to(self.device)
            
            if targets is not None:
                targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, targets, **additional)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, additional = self._process_batch(batch)
                inputs = inputs.to(self.device)
                
                if targets is not None:
                    targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, **additional)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, epoch):
                    break
        
        # Load best weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save model weights and training curves."""
        if self.results_path is None:
            return
        
        # Save weights
        weights_path = Path(self.results_path) / "_Weights"
        create_folder(weights_path)
        torch.save(self.best_model_weights, weights_path / "wghts.pkl")
        print(f"\n✓ Model weights saved to {weights_path / 'wghts.pkl'}")
        
        # Save learning curves
        curves_path = Path(self.results_path) / "learning_curves.png"
        plot_learning_curves(self.train_losses, self.val_losses, str(curves_path))
        print(f"✓ Learning curves saved to {curves_path}")
    
    def load_weights(self, weights_path: Optional[str] = None):
        """
        Load model weights.
        
        Args:
            weights_path: Path to weights file (optional)
        """
        if weights_path is None:
            weights_path = Path(self.results_path) / "_Weights" / "wghts.pkl"
        
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"✓ Loaded weights from {weights_path}")
