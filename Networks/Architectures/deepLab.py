import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18], dropout=0.5):
        super(ASPP, self).__init__()
        
        # Convolution 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Convolutions atrous avec différentes dilations
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0], 
                               dilation=dilations[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], 
                               dilation=dilations[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], 
                               dilation=dilations[2], bias=False)
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Différentes branches ASPP
        x1 = F.relu(self.bn1(self.conv1(x)))  # 1x1 conv
        x2 = F.relu(self.bn2(self.conv2(x)))  # 3x3 conv, dilation rate 1
        x3 = F.relu(self.bn3(self.conv3(x)))  # 3x3 conv, dilation rate 2
        x4 = F.relu(self.bn4(self.conv4(x)))  # 3x3 conv, dilation rate 3
        
        # Global Average Pooling branch
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        # Concaténation de toutes les branches
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        # Fusion et dropout
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        
        return x

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ Architecture pour la segmentation sémantique"""
    
    def __init__(self, param):
        super(DeepLabV3Plus, self).__init__()
        
        # Récupération des paramètres depuis le fichier YAML
        backbone_channels = param["MODEL"].get("BACKBONE_CHANNELS", 64)
        aspp_out_channels = param["MODEL"].get("ASPP_OUTPUT_CHANNELS", 256)
        aspp_dropout = param["MODEL"].get("ASPP_DROPOUT", 0.5)
        aspp_dilations = param["MODEL"].get("ASPP_DILATIONS", [6, 12, 18])
        num_classes = param["MODEL"].get("NUM_CLASSES", 5)
        
        self.backbone_channels = backbone_channels
        self.num_classes = num_classes
        
        # =====================================
        # ENCODER (Backbone simplifié)
        # =====================================
        self.encoder1 = self._make_encoder_block(3, backbone_channels)
        self.encoder2 = self._make_encoder_block(backbone_channels, backbone_channels*2, stride=2)
        self.encoder3 = self._make_encoder_block(backbone_channels*2, backbone_channels*4, stride=2)
        
        # =====================================
        # ASPP MODULE
        # =====================================
        self.aspp = ASPP(
            in_channels=backbone_channels*4, 
            out_channels=aspp_out_channels,
            dilations=aspp_dilations,
            dropout=aspp_dropout
        )
        
        # =====================================
        # DECODER
        # =====================================
        self.decoder1 = nn.Sequential(
            nn.Conv2d(aspp_out_channels, backbone_channels*2, 1, bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection avec encoder2
        self.decoder2 = nn.Sequential(
            nn.Conv2d(backbone_channels*4, backbone_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels*2, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True)
        )
        
        # Couche de classification finale
        self.final_conv = nn.Conv2d(backbone_channels, num_classes, kernel_size=1)
        
        # Initialisation des poids
        self._init_weights()
    
    def _make_encoder_block(self, in_channels, out_channels, stride=1):
        """Crée un bloc d'encodeur avec convolutions, batch norm et ReLU"""
        if stride == 1:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def _init_weights(self):
        """Initialisation des poids du réseau"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # =====================================
        # ENCODER PATH
        # =====================================
        e1 = self.encoder1(x)       # [B, C, H, W]
        e2 = self.encoder2(e1)      # [B, 2C, H/2, W/2]
        e3 = self.encoder3(e2)      # [B, 4C, H/4, W/4]
        
        # =====================================
        # ASPP MODULE (extraction multi-scale)
        # =====================================
        aspp_out = self.aspp(e3)    # [B, aspp_out_channels, H/4, W/4]
        
        # =====================================
        # DECODER PATH
        # =====================================
        # Upsampling et réduction de channels
        d1 = F.interpolate(aspp_out, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1)      # [B, 2C, H/2, W/2]
        
        # Skip connection avec encoder2
        d1 = torch.cat([d1, e2], dim=1)  # [B, 4C, H/2, W/2]
        d2 = self.decoder2(d1)           # [B, C, H/2, W/2]
        
        # =====================================
        # UPSAMPLING FINAL
        # =====================================
        out = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
        out = self.final_conv(out)  # [B, num_classes, H, W]
        
        return out