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
        Masked regions are filled with BLACK (0.0) to force reconstruction.
        
        Args:
            image: RGB image (H, W, 3) in [0, 1]
            mask: Binary mask (H, W)
        
        Returns:
            masked_image: Image with masked regions set to black
        """
        masked_image = image.copy()
        mask_3ch = mask[:, :, np.newaxis]  # Broadcast to 3 channels
        
        # Fill masked regions with BLACK (0.0) - forces model to reconstruct
        masked_image = masked_image * (1 - mask_3ch)
        
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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels=256, dilation_rates=[6, 12, 18], dropout=0.5):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rates[0], 
                              dilation=dilation_rates[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rates[1], 
                              dilation=dilation_rates[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rates[2], 
                              dilation=dilation_rates[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        
        return x


class ContextPredictionAutoencoder(nn.Module):
    """
    DeepLabV3+ architecture adapted for context prediction task.
    Uses ASPP for multi-scale feature extraction.
    
    Input: Masked RGB image (3 channels)
    Output: Reconstructed RGB image (3 channels)
    """
    
    def __init__(self, param):
        super(ContextPredictionAutoencoder, self).__init__()
        
        backbone_channels = param["MODEL"].get("BASE_CHANNELS", 64)
        dilation_rates = param["MODEL"].get("DILATION_RATES", [6, 12, 18])
        aspp_dropout = param["MODEL"].get("DROPOUT", 0.2)
        
        # ENCODER (simplified backbone - 3 input channels for RGB)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(),
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*2, backbone_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU()
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(backbone_channels*2, backbone_channels*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*4),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*4, backbone_channels*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*4),
            nn.ReLU()
        )
        
        # ASPP module for multi-scale features
        self.aspp = ASPP(backbone_channels*4, backbone_channels*4, dilation_rates, aspp_dropout)
        
        # DECODER
        self.decoder1 = nn.Sequential(
            nn.Conv2d(backbone_channels*4, backbone_channels*2, 1, bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(backbone_channels*2 + backbone_channels*2, backbone_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*2, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU()
        )
        
        # Final output - 3 channels for RGB
        self.final_conv = nn.Sequential(
            nn.Conv2d(backbone_channels, 3, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x):
        """
        Forward pass through DeepLabV3+.
        
        Args:
            x: Masked input image (B, 3, H, W)
        
        Returns:
            reconstructed: Reconstructed RGB image (B, 3, H, W)
        """
        input_size = x.size()[2:]
        
        # ENCODER
        e1 = self.encoder1(x)       # (B, 64, H, W)
        e2 = self.encoder2(e1)      # (B, 128, H/2, W/2)
        e3 = self.encoder3(e2)      # (B, 256, H/4, W/4)
        
        # ASPP
        aspp_out = self.aspp(e3)    # (B, 256, H/4, W/4)
        
        # DECODER with skip connection
        d1 = F.interpolate(aspp_out, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1)      # (B, 128, H/2, W/2)
        
        # Skip connection from encoder2
        d1 = torch.cat([d1, e2], dim=1)  # (B, 256, H/2, W/2)
        d2 = self.decoder2(d1)      # (B, 64, H/2, W/2)
        
        # Final upsampling to original resolution
        out = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
        reconstructed = self.final_conv(out)  # (B, 3, H, W)
        
        return reconstructed
    
    def get_encoder_features(self, x):
        """
        Extract encoder features for clustering (before ASPP).
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            features: Encoder output (B, 256, H/4, W/4) from encoder3
        """
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        return e3


def context_prediction_loss(pred, target, mask=None, masked_only=False):
    """
    Loss function for context prediction task.
    
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
        mask_3ch = mask.expand_as(pred)  # Broadcast to 3 channels (B, 3, H, W)
        
        # Compute MSE only on masked pixels
        diff = (pred - target) ** 2
        masked_diff = diff * mask_3ch
        
        # Normalize by number of masked pixels (not total pixels)
        num_masked = mask_3ch.sum() + 1e-8  # Avoid division by zero
        loss = masked_diff.sum() / num_masked
    else:
        # Compute loss on full image
        loss = F.mse_loss(pred, target)
    
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
