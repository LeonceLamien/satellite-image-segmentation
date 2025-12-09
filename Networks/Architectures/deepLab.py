import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256, dropout=0.5):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
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

class DeepLabV3Plus(nn.Module):
    def __init__(self, param):
        super(DeepLabV3Plus, self).__init__()
        backbone_channels = param["MODEL"].get("BACKBONE_CHANNELS", 64)
        aspp_output_channels = param["MODEL"].get("ASPP_OUTPUT_CHANNELS", 256)
        aspp_dropout = param["MODEL"].get("ASPP_DROPOUT", 0.5)
        
        # Encoder simplifié (en pratique on utiliserait ResNet)
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
        
        # ASPP with configurable parameters
        self.aspp = ASPP(backbone_channels*4, aspp_output_channels, dropout=aspp_dropout)
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Conv2d(aspp_output_channels, backbone_channels*2, 1, bias=False),
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
        
        self.final_conv = nn.Conv2d(backbone_channels, 5, 1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)      
        e2 = self.encoder2(e1)     
        e3 = self.encoder3(e2)     
        
        # ASPP
        aspp_out = self.aspp(e3)  
        
        # Decoder
        d1 = F.interpolate(aspp_out, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1)    
        
        # Skip connection
        d1 = torch.cat([d1, e2], dim=1)  
        d2 = self.decoder2(d1)    
        
        # Final upsampling
        out = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(out) 
        
        return out