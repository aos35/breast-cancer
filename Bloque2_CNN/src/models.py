"""
Módelo architectures para segmentación de lesiones mamarias.

Implementa: U-Net, FCN, DeepLab V3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================================
# U-NET ARCHITECTURE
# ============================================================================

class DoubleConv(nn.Module):
    """Bloque: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling: MaxPool + DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling: Upsample + Concatenate + DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ajustar tamaño si es necesario
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenar skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net para segmentación médica.
    
    Arquitectura:
    - Encoder con 4 niveles (downsampling)
    - Bottleneck
    - Decoder con 4 niveles (upsampling)
    - Skip connections
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Head
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Head
        x = self.outc(x)
        
        return x


# ============================================================================
# FCN - FULLY CONVOLUTIONAL NETWORKS
# ============================================================================

class FCN(nn.Module):
    """
    Fully Convolutional Networks para segmentación.
    
    Backbone: VGG16 o ResNet
    Decoder: Convoluciones transpuestas
    """
    
    def __init__(self, in_channels, out_channels, backbone='vgg16', aux_loss=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_loss = aux_loss
        
        if backbone == 'vgg16':
            self._init_vgg16_backbone()
        elif backbone == 'resnet50':
            self._init_resnet50_backbone()
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Head
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Aux head (si es necesario)
        if aux_loss:
            self.aux_head = nn.Conv2d(256, out_channels, kernel_size=1)
    
    def _init_vgg16_backbone(self):
        """Inicializar backbone VGG16"""
        vgg = models.vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:-1])
    
    def _init_resnet50_backbone(self):
        """Inicializar backbone ResNet50"""
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
    
    def forward(self, x):
        # Backbone
        features = self.encoder(x)
        
        # Decoder
        out = self.decoder(features)
        
        # Head
        out = self.head(out)
        
        # Upsample final para recuperar tamaño original
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if self.aux_loss:
            aux_out = self.aux_head(features)
            aux_out = F.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_out
        
        return out


# ============================================================================
# DEEPLABV3
# ============================================================================

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        
        self.branches = nn.ModuleList()
        
        # 1x1 convolution
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 3x3 convolutions con atrous rates
        for rate in atrous_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         dilation=rate, padding=rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Image pool
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels,
                     kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for branch in self.branches:
            res.append(branch(x))
        
        # Image pool
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pool)
        
        # Concatenar y proyectar
        out = torch.cat(res, dim=1)
        out = self.project(out)
        
        return out


class DeepLabV3(nn.Module):
    """
    DeepLab V3 para segmentación semántica.
    
    Backbone: ResNet50 o ResNet101
    Atrous convolutions + ASPP
    """
    
    def __init__(self, in_channels, out_channels, backbone='resnet50', 
                 atrous_rates=(6, 12, 18), aspp_channels=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            backbone_channels = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            backbone_channels = 2048
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
        
        # Adaptar primer layer si es necesario
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # ASPP
        self.aspp = ASPPModule(backbone_channels, aspp_channels, atrous_rates)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Head
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        
        # ASPP
        aspp_out = self.aspp(features)
        aspp_out = F.interpolate(aspp_out, size=features.shape[2:],
                                  mode='bilinear', align_corners=False)
        
        # Decoder
        out = self.decoder(aspp_out)
        
        # Head
        out = self.head(out)
        
        # Upsample final
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_model(model_name, in_channels, out_channels, **kwargs):
    """
    Factory function para crear modelos.
    
    Args:
        model_name: 'unet', 'fcn', 'deeplabv3'
        in_channels: Canales de entrada (1 para grayscale)
        out_channels: Canales de salida (1 para segmentación binaria)
        **kwargs: Parámetros adicionales según modelo
    
    Returns:
        Modelo instantiado
    """
    if model_name == 'unet':
        return UNet(in_channels, out_channels, **kwargs)
    elif model_name == 'fcn':
        return FCN(in_channels, out_channels, **kwargs)
    elif model_name == 'deeplabv3':
        return DeepLabV3(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")


# Test de uso
if __name__ == "__main__":
    # Test U-Net
    print("Testing U-Net...")
    unet = UNet(1, 1)  # 1 canal entrada, 1 canal salida (máscara binaria)
    x = torch.randn(2, 1, 512, 512)
    y = unet(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Test FCN
    print("\nTesting FCN...")
    fcn = FCN(1, 1, backbone='vgg16')
    y = fcn(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in fcn.parameters()):,}")
    
    # Test DeepLab V3
    print("\nTesting DeepLab V3...")
    deeplab = DeepLabV3(1, 1, backbone='resnet50')
    y = deeplab(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in deeplab.parameters()):,}")
