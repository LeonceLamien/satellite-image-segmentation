import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - COMPLETELY FIXED VERSION"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.norm4 = nn.InstanceNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        
        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.norm5 = nn.InstanceNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = F.relu(self.norm1(self.conv1(x)))
        x2 = F.relu(self.norm2(self.conv2(x)))
        x3 = F.relu(self.norm3(self.conv3(x)))
        x4 = F.relu(self.norm4(self.conv4(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.relu(self.norm5(self.conv5(x)))
        x = self.dropout(x)
        
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, param):
        super(DeepLabV3Plus, self).__init__()
        backbone_channels = param["MODEL"].get("BACKBONE_CHANNELS", 64)
        self.task = param["MODEL"].get("TASK", "segmentation")
        
        if self.task == "colorization":
            input_channels = 1  
        else:
            input_channels = 3  
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, backbone_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(backbone_channels),
            nn.ReLU(),
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(backbone_channels),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(backbone_channels*2),
            nn.ReLU()
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(backbone_channels*2, backbone_channels*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(backbone_channels*4),
            nn.ReLU()
        )
        
        self.aspp = ASPP(backbone_channels*4, backbone_channels*4)
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(backbone_channels*4, backbone_channels*2, 1, bias=False),
            nn.InstanceNorm2d(backbone_channels*2),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(backbone_channels*2 + backbone_channels*2, backbone_channels*2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(backbone_channels*2),
            nn.ReLU(),
            nn.Conv2d(backbone_channels*2, backbone_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(backbone_channels),
            nn.ReLU()
        )
        
        if self.task == "colorization":
            self.final_conv = nn.Sequential(
                nn.Conv2d(backbone_channels, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 3, 1)  # Sortie RGB
            )
        else:
            self.final_conv = nn.Conv2d(backbone_channels, 5, 1)
        
        self.feature_channels = backbone_channels * 4  
        self.downsample_factor = 4 

    def forward(self, x, return_features=False):
        if self.task == "colorization" and x.shape[1] == 3:
            x_gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
            x = x_gray
        
        e1 = self.encoder1(x)      # [B, C, H, W]
        e2 = self.encoder2(e1)     # [B, 2C, H/2, W/2]
        e3 = self.encoder3(e2)     # [B, 4C, H/4, W/4]
        
        if return_features:
            return e3
        
        aspp_out = self.aspp(e3)   # [B, 4C, H/4, W/4]
        
        d1 = F.interpolate(aspp_out, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1)     # [B, 2C, H/2, W/2]
        
        d1 = torch.cat([d1, e2], dim=1)  # [B, 4C, H/2, W/2]
        d2 = self.decoder2(d1)     # [B, C, H/2, W/2]
        
        out = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        
        if self.task == "colorization":
            out = torch.sigmoid(out)
        
        return out
    
    def get_encoder_features(self, x):
        """Méthode spécifique pour extraire les features de l'encodeur"""
        if self.task == "colorization" and x.shape[1] == 3:
            x_gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
            x = x_gray
        
        e1 = self.encoder1(x)      # [B, C, H, W]
        e2 = self.encoder2(e1)     # [B, 2C, H/2, W/2]
        e3 = self.encoder3(e2)     # [B, 4C, H/4, W/4]
        
        return e3
    
    @property
    def feature_dim(self):
        """Retourne la dimension des features extraites"""
        return self.feature_channels
    
    @property
    def spatial_downsample(self):
        """Retourne le facteur de réduction spatiale"""
        return self.downsample_factor
    


class ContextAttentionModule(nn.Module):
    """Module d'attention pour capturer le contexte spatial"""
    def __init__(self, channels):
        super(ContextAttentionModule, self).__init__()
        
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        out = self.gamma * out + x
        
        return out