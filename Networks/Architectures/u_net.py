import torch
import torch.nn as nn
import torch.nn.functional as F

class U_net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        
        self.enc1 = self._conv_block(self.nb_channel, 64) 
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)  # 512*2 car skip connection
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)   # 256*2 car skip connection
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)   # 128*2 car skip connection
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)    # 64*2 car skip connection
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, 5, kernel_size=1)  # 5 classes comme dans votre conv3
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_channels, out_channels):
        """Bloc convolutionnel avec deux convs + batch norm + activation"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1)) # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2)) # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3)) # [B, 512, H/8, W/8]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(e4)) # [B, 1024, H/16, W/16]
        
        # Decoder path with skip connections
        d4 = self.upconv4(bottleneck)               # [B, 512, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)             # [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)                          # [B, 512, H/8, W/8]
        
        d3 = self.upconv3(d4)                       # [B, 256, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)             # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)                          # [B, 256, H/4, W/4]
        
        d2 = self.upconv2(d3)                       # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)             # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)                          # [B, 128, H/2, W/2]
        
        d1 = self.upconv1(d2)                       # [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)             # [B, 128, H, W]
        d1 = self.dec1(d1)                          # [B, 64, H, W]
        
        # Output layer
        out = self.out_conv(d1)                     # [B, 5, H, W]
        
        return out