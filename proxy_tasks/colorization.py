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


class ColorizationAutoencoder(nn.Module):
    """
    Encoder-Decoder architecture for colorization task.
    
    Input: Grayscale image (1 channel)
    Output: RGB image (3 channels)
    """
    
    def __init__(self, param):
        super(ColorizationAutoencoder, self).__init__()
        
        in_channels = 1  # Grayscale input
        out_channels = 3  # RGB output
        base_channels = param["MODEL"].get("BASE_CHANNELS", 64)
        dropout = param["MODEL"].get("DROPOUT", 0.1)
        
        # ENCODER: Extracts features from grayscale images
        self.enc1 = self._conv_block(in_channels, base_channels, dropout, padding_mode='reflect')
        self.enc2 = self._conv_block(base_channels, base_channels*2, dropout, padding_mode='reflect')
        self.enc3 = self._conv_block(base_channels*2, base_channels*4, dropout, padding_mode='reflect')
        self.enc4 = self._conv_block(base_channels*4, base_channels*8, dropout, padding_mode='reflect')
        
        # BOTTLENECK
        self.bottleneck = self._conv_block(base_channels*8, base_channels*16, dropout, padding_mode='reflect')
        
        # DECODER: Predicts RGB colors
        self.dec4 = self._upconv_block(base_channels*16, base_channels*8, dropout, padding_mode='reflect')
        self.dec3 = self._upconv_block(base_channels*16, base_channels*4, dropout, padding_mode='reflect')
        self.dec2 = self._upconv_block(base_channels*8, base_channels*2, dropout, padding_mode='reflect')
        self.dec1 = self._upconv_block(base_channels*4, base_channels, dropout, padding_mode='reflect')
        
        # FINAL OUTPUT: Predict RGB colors
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
            x: Grayscale input (B, 1, H, W)
        
        Returns:
            rgb: Predicted RGB image (B, 3, H, W)
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
        rgb = self.final(d1)
        
        return rgb
    
    def get_encoder_features(self, x):
        """
        Extract encoder features for clustering.
        
        Args:
            x: Grayscale input (B, 1, H, W)
        
        Returns:
            features: Encoder output (B, C, H', W')
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        return e4


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
