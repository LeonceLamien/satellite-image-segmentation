"""
Configuration Management
========================

Unified configuration handler for both Part A and Part B.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration handler with validation and default values."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.data = yaml.safe_load(f)
        elif config_dict:
            self.data = config_dict
        else:
            self.data = {}
        
        self._validate()
    
    def _validate(self):
        """Validate configuration structure."""
        # Ensure required sections exist
        if 'DATASET' not in self.data:
            self.data['DATASET'] = {}
        if 'TRAINING' not in self.data and 'TRAIN' not in self.data:
            self.data['TRAINING'] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (e.g., 'TRAINING.LEARNING_RATE')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data
    
    @property
    def dataset(self) -> Dict:
        """Dataset configuration."""
        return self.data.get('DATASET', {})
    
    @property
    def training(self) -> Dict:
        """Training configuration (handles both TRAINING and TRAIN keys)."""
        return self.data.get('TRAINING', self.data.get('TRAIN', {}))
    
    @property
    def model(self) -> Dict:
        """Model configuration."""
        return self.data.get('MODEL', {})
    
    @property
    def task(self) -> Dict:
        """Task configuration (for Part B)."""
        return self.data.get('TASK', {})


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config object
    """
    return Config(config_path=config_path)
