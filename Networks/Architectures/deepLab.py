import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # Convolution 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Convolutions atrous avec dilations fixes [6, 12, 18]
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Fusion des features
        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Différentes branches ASPP
        x1 = F.relu(self.bn1(self.conv1(x)))  # 1x1 conv
        x2 = F.relu(self.bn2(self.conv2(x)))  # 3x3 conv, dilation=6
        x3 = F.relu(self.bn3(self.conv3(x)))  # 3x3 conv, dilation=12
        x4 = F.relu(self.bn4(self.conv4(x)))  # 3x3 conv, dilation=18
        
        # Global Average Pooling branch
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        # Concaténation de toutes les branches
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        # Fusion et dropout
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        
        return x

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ simplifié - Version gagnante originale"""
    
    def __init__(self, param):
        super(DeepLabV3Plus, self).__init__()
        
        backbone_channels = param["MODEL"].get("BACKBONE_CHANNELS", 64)
        num_classes = param["MODEL"].get("NUM_CLASSES", 5)
        
        # =====================================
        # ENCODER simplifié (comme ton ancien code)
        # =====================================
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
            nn.ReLU()
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(backbone_channels*2, backbone_channels*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*4),
            nn.ReLU()
        )
        
        # =====================================
        # ASPP (avec dilations [6,12,18] fixes)
        # =====================================
        self.aspp = ASPP(backbone_channels*4, backbone_channels*4)
        
        # =====================================
        # DECODER simple
        # =====================================
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
        
        self.final_conv = nn.Conv2d(backbone_channels, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)      # [B, 64, 64, 64]
        e2 = self.encoder2(e1)     # [B, 128, 32, 32]
        e3 = self.encoder3(e2)     # [B, 256, 16, 16]
        
        # ASPP
        aspp_out = self.aspp(e3)   # [B, 256, 16, 16]
        
        # Decoder
        d1 = F.interpolate(aspp_out, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1)     # [B, 128, 32, 32]
        
        # Skip connection
        d1 = torch.cat([d1, e2], dim=1)  # [B, 256, 32, 32]
        d2 = self.decoder2(d1)           # [B, 64, 32, 32]
        
        # Final upsampling
        out = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(out)       # [B, 5, 64, 64]
        
        return out