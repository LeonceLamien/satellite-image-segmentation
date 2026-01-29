"""
PART B - PROXY TASK 1: IMAGE DENOISING
=======================================

This is the SIMPLEST proxy task to start with.

Principle:
    1. Take clean images from dataset
    2. Add Gaussian noise to create noisy versions
    3. Train encoder-decoder to predict clean image from noisy input
    4. The encoder learns meaningful features without segmentation labels

Architecture:
    - Encoder: Extracts features from noisy images
    - Decoder: Reconstructs clean images from features
    - Loss: MSE between predicted clean image and original clean image

Why this works:
    To denoise effectively, the encoder must learn to recognize:
    - Edges, textures, structures
    - Meaningful patterns vs random noise
    - Spatial relationships
    These features are useful for segmentation tasks!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import json


class DenoisingDataset(Dataset):
    """
    Dataset for denoising proxy task.
    Returns: (noisy_image, clean_image)
    """
    
    def __init__(self, img_directory, set_name, param, noise_std=0.1):
        """
        Args:
            img_directory: Path to images folder
            set_name: "train", "val", or "test"
            param: Config dictionary
            noise_std: Standard deviation of Gaussian noise (default 0.1)
        """
        self.img_directory = img_directory
        self.set_name = set_name
        self.noise_std = noise_std
        self.resize_h = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0])
        self.resize_w = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])
        
        # Load file list
        with open(f"./Dataset/{set_name}.json", 'r') as f:
            self.files = json.load(f)
    
    def __len__(self):
        return len(self.files)
    
    def add_noise(self, image):
        """
        Add Gaussian noise to image.
        
        Args:
            image: Clean image, numpy array [0, 255] or [0, 1]
        
        Returns:
            noisy_image: Image with added noise
        """
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, image.shape)
        noisy_image = image + noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image.astype(np.float32)
    
    def __getitem__(self, idx):
        """
        Returns:
            noisy_image: Tensor (3, H, W) with added noise
            clean_image: Tensor (3, H, W) original clean image
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
        clean_image = image.astype(np.float32) / 255.0
        
        # Add noise
        noisy_image = self.add_noise(clean_image)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        clean_tensor = torch.from_numpy(clean_image.transpose(2, 0, 1))
        noisy_tensor = torch.from_numpy(noisy_image.transpose(2, 0, 1))
        
        return noisy_tensor, clean_tensor, tile_name


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels=256, dilation_rates=[6, 12, 18], dropout=0.5):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rates[0], 
                              dilation=dilation_rates[0], padding_mode='reflect', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rates[1], 
                              dilation=dilation_rates[1], padding_mode='reflect', bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rates[2], 
                              dilation=dilation_rates[2], padding_mode='reflect', bias=False)
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


class DenoisingAutoencoder(nn.Module):
    """
    DeepLabV3+ architecture adapted for denoising task.
    Uses ASPP for multi-scale feature extraction.
    
    Input: Noisy RGB image (3 channels)
    Output: Clean RGB image (3 channels)
    """
    
    def __init__(self, param):
        super(DenoisingAutoencoder, self).__init__()
        
        backbone_channels = param["MODEL"].get("BASE_CHANNELS", 64)
        dilation_rates = param["MODEL"].get("DILATION_RATES", [6, 12, 18])
        aspp_dropout = param["MODEL"].get("DROPOUT", 0.2)
        
        # ENCODER (simplified backbone - 3 input channels for RGB)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, backbone_channels, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(),
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels*2, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*2, backbone_channels*2, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU()
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(backbone_channels*2, backbone_channels*4, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(backbone_channels*4),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*4, backbone_channels*4, 3, padding=1, padding_mode='reflect', bias=False),
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
            nn.Conv2d(backbone_channels*2 + backbone_channels*2, backbone_channels*2, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*2, backbone_channels, 3, padding=1, padding_mode='reflect', bias=False),
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
            x: Noisy input image (B, 3, H, W)
        
        Returns:
            clean: Reconstructed clean image (B, 3, H, W)
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
        clean = self.final_conv(out)  # (B, 3, H, W)
        
        return clean
    
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


def denoise_loss(pred_clean, target_clean):
    """
    Loss function for denoising task.
    Simple MSE between predicted and target clean images.
    
    Args:
        pred_clean: Predicted clean image (B, 3, H, W)
        target_clean: Ground truth clean image (B, 3, H, W)
    
    Returns:
        loss: Scalar MSE loss
    """
    return F.mse_loss(pred_clean, target_clean)


if __name__ == "__main__":
    print("=" * 60)
    print("DENOISING PROXY TASK - Test")
    print("=" * 60)
    
    # Test dataset
    print("\n[1] Testing DenoisingDataset...")
    param = {
        "DATASET": {"RESIZE_SHAPE": "64x64"}
    }
    
    dataset = DenoisingDataset(
        img_directory="./Dataset/images",
        set_name="train",
        param=param,
        noise_std=0.1
    )
    
    noisy, clean, name = dataset[0]
    print(f"✓ Dataset loaded: {len(dataset)} images")
    print(f"  Noisy image shape: {noisy.shape}")
    print(f"  Clean image shape: {clean.shape}")
    print(f"  Sample name: {name}")
    
    # Test model
    print("\n[2] Testing DenoisingAutoencoder...")
    param["MODEL"] = {"BASE_CHANNELS": 64, "DROPOUT": 0.1}
    
    model = DenoisingAutoencoder(param)
    noisy_batch = noisy.unsqueeze(0)  # Add batch dimension
    
    output = model(noisy_batch)
    features = model.get_encoder_features(noisy_batch)
    
    print(f"✓ Model created")
    print(f"  Input shape: {noisy_batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Encoder features shape: {features.shape}")
    
    # Test loss
    print("\n[3] Testing loss function...")
    loss = denoise_loss(output, clean.unsqueeze(0))
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready to train.")
    print("=" * 60)
