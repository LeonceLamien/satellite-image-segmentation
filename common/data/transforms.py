"""
Data Transformations
====================

Albumentation transforms for images and masks.
"""

import albumentations as alb
import cv2
from typing import Dict, Any


def get_train_transforms(config: Dict[str, Any]) -> alb.Compose:
    """
    Create training data augmentation pipeline.
    
    Args:
        config: Configuration dictionary with DATASET section
    
    Returns:
        Albumentation Compose object
    """
    dataset_config = config.get('DATASET', {})
    
    # Parse resize shape
    resize_shape = dataset_config.get('RESIZE_SHAPE', '64x64')
    height, width = map(int, resize_shape.split('x'))
    
    transforms_list = [
        alb.Resize(height=height, width=width)
    ]
    
    # Augmentation parameters
    use_augmentation = dataset_config.get('USE_AUGMENTATION', False)
    force_augmentation = dataset_config.get('FORCE_AUGMENTATION', False)
    prob = 1.0 if force_augmentation else 0.5
    
    # Add augmentations if enabled
    if use_augmentation:
        if dataset_config.get('HORIZONTAL_FLIP', False):
            transforms_list.append(alb.HorizontalFlip(p=prob))
        
        if dataset_config.get('VERTICAL_FLIP', False):
            transforms_list.append(alb.VerticalFlip(p=prob))
        
        rotate_limit = dataset_config.get('ROTATE_LIMIT', 0)
        if rotate_limit > 0:
            transforms_list.append(alb.Rotate(
                limit=rotate_limit,
                p=prob,
                border_mode=cv2.BORDER_REFLECT_101
            ))
        
        if dataset_config.get('BRIGHTNESS_CONTRAST', False):
            transforms_list.append(alb.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=prob
            ))
        
        if dataset_config.get('BLUR', False):
            blur_prob = 0.5 if force_augmentation else 0.3
            transforms_list.append(alb.OneOf([
                alb.MotionBlur(blur_limit=3, p=1),
                alb.GaussianBlur(blur_limit=3, p=1),
            ], p=blur_prob))
        
        if dataset_config.get('COLOR_JITTER', False):
            transforms_list.append(alb.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=prob
            ))
    
    # Normalization and tensor conversion
    transforms_list.extend([
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        alb.pytorch.transforms.ToTensorV2()
    ])
    
    return alb.Compose(transforms_list)


def get_val_transforms(config: Dict[str, Any]) -> alb.Compose:
    """
    Create validation/test data transformation pipeline.
    
    Args:
        config: Configuration dictionary with DATASET section
    
    Returns:
        Albumentation Compose object
    """
    dataset_config = config.get('DATASET', {})
    
    # Parse resize shape
    resize_shape = dataset_config.get('RESIZE_SHAPE', '64x64')
    height, width = map(int, resize_shape.split('x'))
    
    transforms_list = [
        alb.Resize(height=height, width=width),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        alb.pytorch.transforms.ToTensorV2()
    ]
    
    return alb.Compose(transforms_list)
