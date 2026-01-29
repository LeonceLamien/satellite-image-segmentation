"""
Dataset Class
=============

Unified dataset class for loading images and masks.
"""

import cv2
import os
import copy
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import albumentations as alb
import torch
from torch.utils.data import Dataset


class LLNDataset(Dataset):
    """
    Custom dataset for loading images and segmentation masks.
    
    This dataset inherits from torch.utils.data.Dataset and handles
    loading, preprocessing, and augmentation of images and masks.
    """
    
    def __init__(
        self,
        img_directory: str,
        mask_directory: Optional[str],
        set_name: str,
        config: Dict[str, Any],
        transform_train: Optional[alb.Compose] = None,
        transform_val: Optional[alb.Compose] = None
    ):
        """
        Initialize LLNDataset.
        
        Args:
            img_directory: Path to images folder
            mask_directory: Path to masks folder (None for Part B)
            set_name: Dataset split ("train", "val", or "test")
            config: Configuration dictionary
            transform_train: Training transforms (optional)
            transform_val: Validation transforms (optional)
        """
        self.img_directory = img_directory
        self.mask_directory = mask_directory
        self.set_name = set_name
        self.config = config
        
        # Set transforms
        if transform_train is not None:
            self.transform_train = transform_train
        else:
            from .transforms import get_train_transforms
            self.transform_train = get_train_transforms(config)
        
        if transform_val is not None:
            self.transform_val = transform_val
        else:
            from .transforms import get_val_transforms
            self.transform_val = get_val_transforms(config)
        
        # Load file list
        json_path = Path(img_directory).parent / f"{set_name}.json"
        if not json_path.exists():
            # Try alternative location (for PartA/PartB structure)
            json_path = Path(img_directory).parent / "Dataset" / f"{set_name}.json"
        
        with open(json_path, 'r') as f:
            self.files = json.load(f)
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get the idx-th image and mask.
        
        Args:
            idx: Index of image to retrieve
        
        Returns:
            Tuple of (image, mask, tile_name, resized_img)
            For Part B without masks, mask is None
        """
        tile_name = self.files[idx]
        filepath_img = os.path.join(self.img_directory, tile_name)
        
        # Load image
        original_img = cv2.imread(filepath_img)
        if original_img is None:
            raise FileNotFoundError(f"Cannot load image: {filepath_img}")
        
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        copy_img = copy.deepcopy(original_img)
        
        # Load mask if available (Part A)
        mask = None
        if self.mask_directory is not None:
            filepath_mask = os.path.join(self.mask_directory, tile_name)
            if os.path.exists(filepath_mask):
                mask = cv2.imread(filepath_mask, cv2.IMREAD_UNCHANGED)
        
        # Select transform based on dataset split
        if self.set_name == 'train':
            transform = self.transform_train
        else:
            transform = self.transform_val
        
        # Apply transforms
        if transform is not None:
            if mask is not None:
                transformed = transform(image=original_img, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                transformed = transform(image=original_img)
                image = transformed["image"]
            
            # Create resized image for visualization
            resize_transform = alb.Compose([
                t for t in transform 
                if isinstance(t, (alb.Resize, alb.pytorch.transforms.ToTensorV2))
            ])
            resized = resize_transform(image=copy_img)
            resized_img = resized["image"]
        else:
            image = torch.from_numpy(original_img.transpose(2, 0, 1))
            resized_img = image
        
        return image, mask, tile_name, resized_img
