import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# BASIC NETWORK ARCHITECTURES FOR SEMANTIC SEGMENTATION
# - Net: Simple baseline model
# - ImprovedU_net: U-Net with optional attention mechanisms
# - AttentionBlock: Attention gate for U-Net
#
######################################################################################


class Net(nn.Module):
    """Simple baseline network for semantic segmentation"""
    
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        num_classes = param["MODEL"].get("NUM_CLASSES", 5)
        
        self.conv1 = nn.Conv2d(3, self.nb_channel, 5, padding='same')
        self.conv2 = nn.Conv2d(self.nb_channel, self.nb_channel, 5, padding='same')
        self.conv3 = nn.Conv2d(self.nb_channel, num_classes, 5, padding='same')

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ImprovedU_net(nn.Module):
    """U-Net architecture with optional attention mechanisms"""
    
    def __init__(self, param):
        super().__init__()
        # Récupération des paramètres depuis le fichier YAML
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        self.dropout_rate = param["MODEL"]["DROPOUT"]
        self.use_attention = param["MODEL"]["USE_ATTENTION"]
        num_classes = param["MODEL"].get("NUM_CLASSES", 5)
        
        # =====================================
        # ENCODER (Contracting Path)
        # =====================================
        self.enc1 = self._conv_block(3, 64) 
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # =====================================
        # BOTTLENECK
        # =====================================
        self.bottleneck = self._conv_block(512, 1024)
        self.dropout = nn.Dropout2d(self.dropout_rate)
        
        # =====================================
        # DECODER (Expansive Path)
        # =====================================
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        if self.use_attention:
            self.attention4 = AttentionBlock(512, 512)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        if self.use_attention:
            self.attention3 = AttentionBlock(256, 256)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        if self.use_attention:
            self.attention2 = AttentionBlock(128, 128)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        if self.use_attention:
            self.attention1 = AttentionBlock(64, 64)
        self.dec1 = self._conv_block(128, 64)
        
        # =====================================
        # OUTPUT LAYER
        # =====================================
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Initialisation des poids
        self._initialize_weights()
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_channels, out_channels):
        """Bloc convolutionnel avec double convolution, batch norm et ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        """Initialisation des poids du réseau"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # =====================================
        # ENCODER PATH
        # =====================================
        e1 = self.enc1(x)               # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))   # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))   # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))   # [B, 512, H/8, W/8]
        
        # =====================================
        # BOTTLENECK
        # =====================================
        bottleneck = self.bottleneck(self.pool(e4))  # [B, 1024, H/16, W/16]
        bottleneck = self.dropout(bottleneck)
        
        # =====================================
        # DECODER PATH avec skip connections
        # =====================================
        d4 = self.upconv4(bottleneck)   # [B, 512, H/8, W/8]
        if self.use_attention:
            e4 = self.attention4(e4, d4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)           # [B, 256, H/4, W/4]
        if self.use_attention:
            e3 = self.attention3(e3, d3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)           # [B, 128, H/2, W/2]
        if self.use_attention:
            e2 = self.attention2(e2, d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)           # [B, 64, H, W]
        if self.use_attention:
            e1 = self.attention1(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # =====================================
        # OUTPUT
        # =====================================
        out = self.out_conv(d1)         # [B, num_classes, H, W]
        
        return out


class AttentionBlock(nn.Module):
    """Attention Gate pour U-Net
    
    Permet de mettre l'accent sur les features pertinentes
    lors de la fusion des skip connections
    """
    
    def __init__(self, F_g, F_l):
        super(AttentionBlock, self).__init__()
        
        # Gate signal (provenant du decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        
        # Skip connection signal (provenant de l'encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        
        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: gate signal du decoder
            x: skip connection de l'encoder
        Returns:
            Weighted features x
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi