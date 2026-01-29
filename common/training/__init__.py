"""Training modules."""

from .base_trainer import BaseTrainer
from .callbacks import EarlyStopping

__all__ = ['BaseTrainer', 'EarlyStopping']
