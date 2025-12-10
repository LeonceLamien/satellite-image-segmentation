"""
PART B - PROXY TASK 3: CONTEXT PREDICTION
==========================================

Mask random patches in images and train network to predict missing content.

Principle:
    1. Randomly mask patches in the image (set to gray or zero)
    2. Train encoder-decoder to reconstruct the original image
    3. Loss: MSE on masked regions (or full image)

Why this works:
    To predict masked content, the encoder must understand:
    - Spatial context and relationships
    - Object shapes and boundaries
    - Semantic consistency (what objects appear where)
    These features help with segmentation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import json
import random


class ContextPredictionDataset(Dataset):
    """
    Dataset for context prediction proxy task.
    Returns: (masked_image, original_image, mask_positions)
    """
    
    def __init__(self, img_directory, set_name, param, 
                 mask_ratio=0.3, patch_size=8):
        """
        Args:
            img_directory: Path to images folder
            set_name: "train", "val", or "test"
            param: Config dictionary
            mask_ratio: Proportion of image to mask (default 0.3 = 30%)
            patch_size: Size of masked patches (default 8x8)
        """
        self.img_directory = img_directory
        self.set_name = set_name
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.resize_h = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0])
        self.resize_w = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])
        
        # Load file list
        with open(f"./Dataset/{set_name}.json", 'r') as f:
            self.files = json.load(f)
    
    def __len__(self):
        return len(self.files)
    
    def create_random_mask(self, height, width):
        """
        Create random binary mask with patches.
        
        Args:
            height, width: Image dimensions
        
        Returns:
            mask: Binary mask (H, W) where 1 = masked, 0 = visible
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Calculate number of patches to mask
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        total_patches = num_patches_h * num_patches_w
        num_masked = int(total_patches * self.mask_ratio)
        
        # Randomly select patches to mask
        all_patches = [(i, j) for i in range(num_patches_h) 
                       for j in range(num_patches_w)]
        masked_patches = random.sample(all_patches, num_masked)
        
        # Set masked patches to 1
        for i, j in masked_patches:
            y_start = i * self.patch_size
            y_end = min((i + 1) * self.patch_size, height)
            x_start = j * self.patch_size
            x_end = min((j + 1) * self.patch_size, width)
            mask[y_start:y_end, x_start:x_end] = 1.0
        
        return mask
    
    def apply_mask(self, image, mask):
        """
        Apply mask to image.
        Masked regions are filled with gray (0.5) to distinguish from content.
        
        Args:
            image: RGB image (H, W, 3) in [0, 1]
            mask: Binary mask (H, W)
        
        Returns:
            masked_image: Image with masked regions filled with gray
        """
        masked_image = image.copy()
        mask_3ch = mask[:, :, np.newaxis]  # Broadcast to 3 channels
        
        # Fill masked regions with gray (0.5)
        masked_image = masked_image * (1 - mask_3ch) + 0.5 * mask_3ch
        
        return masked_image
    
    def __getitem__(self, idx):
        """
        Returns:
            masked_image: Tensor (3, H, W) with masked patches
            original_image: Tensor (3, H, W) original complete image
            mask: Tensor (1, H, W) binary mask (1=masked, 0=visible)
            tile_name: Filename for reference
        """
        tile_name = self.files[idx]
        filepath = os.path.join(self.img_directory, tile_name)
        
        # Load and resize image
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize_w, self.resize_h))
        
        # Normalize to [0, 1]
        original_image = image.astype(np.float32) / 255.0
        
        # Create and apply mask
        mask = self.create_random_mask(self.resize_h, self.resize_w)
        masked_image = self.apply_mask(original_image, mask)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        original_tensor = torch.from_numpy(original_image.transpose(2, 0, 1))
        masked_tensor = torch.from_numpy(masked_image.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :])  # Add channel dim
        
        return masked_tensor, original_tensor, mask_tensor, tile_name


class ContextPredictionAutoencoder(nn.Module):
    """
    Encoder-Decoder architecture for context prediction task.
    
    Input: Masked image (3 channels)
    Output: Reconstructed complete image (3 channels)
    """
    
    def __init__(self, param):
        super(ContextPredictionAutoencoder, self).__init__()
        
        in_channels = 3  # RGB masked input
        out_channels = 3  # RGB output
        base_channels = param["MODEL"].get("BASE_CHANNELS", 64)
        dropout = param["MODEL"].get("DROPOUT", 0.1)
        
        # ENCODER: Extracts features from masked images
        self.enc1 = self._conv_block(in_channels, base_channels, dropout, padding_mode='reflect')
        self.enc2 = self._conv_block(base_channels, base_channels*2, dropout, padding_mode='reflect')
        self.enc3 = self._conv_block(base_channels*2, base_channels*4, dropout, padding_mode='reflect')
        self.enc4 = self._conv_block(base_channels*4, base_channels*8, dropout, padding_mode='reflect')
        
        # BOTTLENECK
        self.bottleneck = self._conv_block(base_channels*8, base_channels*16, dropout, padding_mode='reflect')
        
        # DECODER: Reconstructs complete image
        self.dec4 = self._upconv_block(base_channels*16, base_channels*8, dropout, padding_mode='reflect')
        self.dec3 = self._upconv_block(base_channels*16, base_channels*4, dropout, padding_mode='reflect')
        self.dec2 = self._upconv_block(base_channels*8, base_channels*2, dropout, padding_mode='reflect')
        self.dec1 = self._upconv_block(base_channels*4, base_channels, dropout, padding_mode='reflect')
        
        # FINAL OUTPUT: Reconstruct RGB image
        self.final = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def _conv_block(self, in_ch, out_ch, dropout, padding_mode='reflect'):
        """Convolutional block with reflect padding."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _upconv_block(self, in_ch, out_ch, dropout, padding_mode='reflect'):
        """Upsampling + Convolutional block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Masked input image (B, 3, H, W)
        
        Returns:
            reconstructed: Complete image (B, 3, H, W)
        """
        # ENCODER
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # BOTTLENECK
        b = self.bottleneck(self.pool(e4))
        
        # DECODER with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        # FINAL OUTPUT
        reconstructed = self.final(d1)
        
        return reconstructed
    
    def get_encoder_features(self, x):
        """
        Extract encoder features for clustering.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            features: Encoder output (B, C, H', W')
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        return e4


def context_prediction_loss(pred, target, mask=None, masked_only=False):
    """
    Enhanced loss function for context prediction task.
    Can focus on masked regions and adds edge-aware component.
    
    Args:
        pred: Predicted image (B, 3, H, W)
        target: Ground truth image (B, 3, H, W)
        mask: Binary mask (B, 1, H, W), 1=masked regions
        masked_only: If True, compute loss only on masked regions
    
    Returns:
        loss: Scalar MSE loss
    """
    if masked_only and mask is not None:
        # Compute loss only on masked regions
        mask_3ch = mask.expand_as(pred)  # Broadcast to 3 channels
        masked_pred = pred * mask_3ch
        masked_target = target * mask_3ch
        mse_loss = F.mse_loss(masked_pred, masked_target)
        
        # Add edge loss for boundary awareness
        def compute_edges(img):
            gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            return gx, gy
        
        pred_gx, pred_gy = compute_edges(pred)
        target_gx, target_gy = compute_edges(target)
        edge_loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        
        loss = mse_loss + 0.2 * edge_loss
    else:
        # Compute loss on full image with edge component
        mse_loss = F.mse_loss(pred, target)
        
        # Add edge awareness
        def compute_edges(img):
            gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            return gx, gy
        
        pred_gx, pred_gy = compute_edges(pred)
        target_gx, target_gy = compute_edges(target)
        edge_loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        
        loss = mse_loss + 0.1 * edge_loss
    
    return loss


if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT PREDICTION PROXY TASK - Test")
    print("=" * 60)
    
    # Test dataset
    print("\n[1] Testing ContextPredictionDataset...")
    param = {
        "DATASET": {"RESIZE_SHAPE": "64x64"}
    }
    
    dataset = ContextPredictionDataset(
        img_directory="./Dataset/images",
        set_name="train",
        param=param,
        mask_ratio=0.3,
        patch_size=8
    )
    
    masked, original, mask, name = dataset[0]
    print(f"✓ Dataset loaded: {len(dataset)} images")
    print(f"  Masked image shape: {masked.shape}")
    print(f"  Original image shape: {original.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Masked ratio: {mask.mean().item():.2%}")
    print(f"  Sample name: {name}")
    
    # Test model
    print("\n[2] Testing ContextPredictionAutoencoder...")
    param["MODEL"] = {"BASE_CHANNELS": 64, "DROPOUT": 0.1}
    
    model = ContextPredictionAutoencoder(param)
    masked_batch = masked.unsqueeze(0)
    
    output = model(masked_batch)
    features = model.get_encoder_features(masked_batch)
    
    print(f"✓ Model created")
    print(f"  Input shape: {masked_batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Encoder features shape: {features.shape}")
    
    # Test loss
    print("\n[3] Testing loss function...")
    loss_full = context_prediction_loss(output, original.unsqueeze(0), masked_only=False)
    loss_masked = context_prediction_loss(output, original.unsqueeze(0), 
                                          mask.unsqueeze(0), masked_only=True)
    print(f"✓ Loss (full image): {loss_full.item():.4f}")
    print(f"✓ Loss (masked only): {loss_masked.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready to train.")
    print("=" * 60)
