"""
PART B - PROXY TASK 2: IMAGE COLORIZATION
==========================================

Convert RGB images to grayscale, then train network to predict colors.

Principle:
    1. Convert RGB images to grayscale (single channel L)
    2. Train encoder-decoder to predict RGB colors from grayscale
    3. Loss: MSE between predicted colors and original RGB

Why this works:
    To colorize successfully, the encoder must understand:
    - Object types (sky=blue, grass=green, buildings=gray)
    - Semantic context and spatial relationships
    - Texture patterns associated with different materials
    These features are highly relevant for segmentation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import json


class ColorizationDataset(Dataset):
    """
    Dataset for colorization proxy task.
    Returns: (grayscale_image, rgb_image)
    """
    
    def __init__(self, img_directory, set_name, param):
        """
        Args:
            img_directory: Path to images folder
            set_name: "train", "val", or "test"
            param: Config dictionary
        """
        self.img_directory = img_directory
        self.set_name = set_name
        self.resize_h = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0])
        self.resize_w = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])
        
        # Load file list
        with open(f"./Dataset/{set_name}.json", 'r') as f:
            self.files = json.load(f)
    
    def __len__(self):
        return len(self.files)
    
    def rgb_to_grayscale(self, rgb_image):
        """
        Convert RGB to grayscale using standard luminance weights.
        
        Args:
            rgb_image: RGB image (H, W, 3) in [0, 1]
        
        Returns:
            gray_image: Grayscale image (H, W, 1) in [0, 1]
        """
        # Standard luminance weights: 0.299*R + 0.587*G + 0.114*B
        weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(rgb_image, weights)
        return gray[:, :, np.newaxis]  # Add channel dimension
    
    def __getitem__(self, idx):
        """
        Returns:
            gray_image: Tensor (1, H, W) grayscale input
            rgb_image: Tensor (3, H, W) RGB target for colorization
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
        rgb_image = image.astype(np.float32) / 255.0
        
        # Convert to grayscale
        gray_image = self.rgb_to_grayscale(rgb_image)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).float()
        gray_tensor = torch.from_numpy(gray_image.transpose(2, 0, 1)).float()
        
        return gray_tensor, rgb_tensor, tile_name


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


class ColorizationAutoencoder(nn.Module):
    """
    DeepLabV3+ architecture adapted for colorization task.
    Uses ASPP for multi-scale feature extraction.
    
    Input: Grayscale image (1 channel)
    Output: RGB image (3 channels)
    """
    
    def __init__(self, param):
        super(ColorizationAutoencoder, self).__init__()
        
        backbone_channels = param["MODEL"].get("BASE_CHANNELS", 64)
        dilation_rates = param["MODEL"].get("DILATION_RATES", [6, 12, 18])
        aspp_dropout = param["MODEL"].get("DROPOUT", 0.2)
        
        # ENCODER (simplified backbone - 1 input channel for grayscale)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, backbone_channels, 3, padding=1, bias=False),
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
            x: Grayscale input image (B, 1, H, W)
        
        Returns:
            colorized: Reconstructed RGB image (B, 3, H, W)
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
        colorized = self.final_conv(out)  # (B, 3, H, W)
        
        return colorized
    
    def get_encoder_features(self, x):
        """
        Extract encoder features for clustering (before ASPP).
        
        Args:
            x: Input image (B, 1, H, W)
        
        Returns:
            features: Encoder output (B, 256, H/4, W/4) from encoder3
        """
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        return e3


def colorization_loss(pred_rgb, target_rgb):
    """
    Loss function for colorization task.
    Simple MSE between predicted and target RGB images.
    
    Args:
        pred_rgb: Predicted RGB image (B, 3, H, W)
        target_rgb: Ground truth RGB image (B, 3, H, W)
    
    Returns:
        loss: Scalar MSE loss
    """
    return F.mse_loss(pred_rgb, target_rgb)


if __name__ == "__main__":
    print("=" * 60)
    print("COLORIZATION PROXY TASK - Test")
    print("=" * 60)
    
    # Test dataset
    print("\n[1] Testing ColorizationDataset...")
    param = {
        "DATASET": {"RESIZE_SHAPE": "64x64"}
    }
    
    dataset = ColorizationDataset(
        img_directory="./Dataset/images",
        set_name="train",
        param=param
    )
    
    gray, rgb, name = dataset[0]
    print(f"✓ Dataset loaded: {len(dataset)} images")
    print(f"  Grayscale image shape: {gray.shape}")
    print(f"  RGB image shape: {rgb.shape}")
    print(f"  Sample name: {name}")
    
    # Test model
    print("\n[2] Testing ColorizationAutoencoder...")
    param["MODEL"] = {"BASE_CHANNELS": 64, "DROPOUT": 0.1}
    
    model = ColorizationAutoencoder(param)
    gray_batch = gray.unsqueeze(0)
    
    output = model(gray_batch)
    features = model.get_encoder_features(gray_batch)
    
    print(f"✓ Model created")
    print(f"  Input shape: {gray_batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Encoder features shape: {features.shape}")
    
    # Test loss
    print("\n[3] Testing loss function...")
    loss = colorization_loss(output, rgb.unsqueeze(0))
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready to train.")
    print("=" * 60)
