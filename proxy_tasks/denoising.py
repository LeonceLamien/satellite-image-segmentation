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


class DenoisingAutoencoder(nn.Module):
    """
    Encoder-Decoder architecture for denoising task.
    
    The encoder learns to extract meaningful features from noisy images.
    The decoder reconstructs the clean image from these features.
    """
    
    def __init__(self, param):
        super(DenoisingAutoencoder, self).__init__()
        
        in_channels = 3  # RGB images
        out_channels = 3  # RGB output
        base_channels = param["MODEL"].get("BASE_CHANNELS", 64)
        dropout = param["MODEL"].get("DROPOUT", 0.1)
        
        # ENCODER: Extracts features from noisy images
        # Uses 'reflect' padding to avoid border artifacts
        self.enc1 = self._conv_block(in_channels, base_channels, dropout, padding_mode='reflect')
        self.enc2 = self._conv_block(base_channels, base_channels*2, dropout, padding_mode='reflect')
        self.enc3 = self._conv_block(base_channels*2, base_channels*4, dropout, padding_mode='reflect')
        self.enc4 = self._conv_block(base_channels*4, base_channels*8, dropout, padding_mode='reflect')
        
        # BOTTLENECK: Deepest features
        self.bottleneck = self._conv_block(base_channels*8, base_channels*16, dropout, padding_mode='reflect')
        
        # DECODER: Reconstructs clean image from features
        self.dec4 = self._upconv_block(base_channels*16, base_channels*8, dropout, padding_mode='reflect')
        self.dec3 = self._upconv_block(base_channels*16, base_channels*4, dropout, padding_mode='reflect')  # *16 due to skip connection
        self.dec2 = self._upconv_block(base_channels*8, base_channels*2, dropout, padding_mode='reflect')
        self.dec1 = self._upconv_block(base_channels*4, base_channels, dropout, padding_mode='reflect')
        
        # FINAL OUTPUT: Reconstruct RGB image
        self.final = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def _conv_block(self, in_ch, out_ch, dropout, padding_mode='reflect'):
        """
        Convolutional block with BatchNorm and Dropout.
        Uses reflect padding to avoid zero-padding artifacts.
        """
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
        """
        Upsampling + Convolutional block.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections (U-Net style).
        
        Args:
            x: Noisy input image (B, 3, H, W)
        
        Returns:
            clean: Reconstructed clean image (B, 3, H, W)
        """
        # ENCODER with skip connections
        e1 = self.enc1(x)      # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)
        
        # BOTTLENECK
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)
        
        # DECODER with skip connections
        d4 = self.dec4(b)  # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        
        d3 = self.dec3(d4)  # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)  # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)  # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)
        
        # FINAL OUTPUT
        clean = self.final(d1)  # (B, 3, H, W)
        
        return clean
    
    def get_encoder_features(self, x):
        """
        Extract encoder features for clustering.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            features: Encoder output (B, C, H', W')
                     Typically (B, 512, H/8, W/8) from enc4
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))  # Use this for clustering
        
        return e4


def denoise_loss(pred_clean, target_clean):
    """
    Enhanced loss function for denoising task.
    Combines MSE (pixel-level) with gradient loss (edge preservation).
    
    Args:
        pred_clean: Predicted clean image (B, 3, H, W)
        target_clean: Ground truth clean image (B, 3, H, W)
    
    Returns:
        loss: Scalar loss combining MSE and gradient loss
    """
    # MSE loss (pixel-level reconstruction)
    mse_loss = F.mse_loss(pred_clean, target_clean)
    
    # Gradient loss (edge preservation) - helps learn structural features
    # Compute gradients in x and y directions
    def compute_gradients(img):
        # Sobel-like gradient computation
        gx = img[:, :, :, 1:] - img[:, :, :, :-1]  # Horizontal gradient
        gy = img[:, :, 1:, :] - img[:, :, :-1, :]  # Vertical gradient
        return gx, gy
    
    pred_gx, pred_gy = compute_gradients(pred_clean)
    target_gx, target_gy = compute_gradients(target_clean)
    
    gradient_loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
    
    # Combine losses (MSE dominant for denoising, gradient for structure)
    total_loss = mse_loss + 0.1 * gradient_loss
    
    return total_loss


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
