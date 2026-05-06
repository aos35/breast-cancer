"""
Model Architectures for Bloque2_CNN
Classification (DenseNet121) + Detection (Mask R-CNN) + Segmentation (U-Net, DeepLabV3+)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.nn.functional as F


# ============================================================================
# TASK 1: CLASSIFICATION - DenseNet121
# ============================================================================

class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 for 3-class classification: Benign / Malignant / Negative
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Initialize DenseNet121 classifier
        
        Args:
            num_classes: Number of classes (3 for B/M/N)
            pretrained: Use ImageNet pre-training
        """
        super(DenseNet121Classifier, self).__init__()
        
        # Load pre-trained DenseNet121
        self.backbone = torchvision.models.densenet121(pretrained=pretrained)
        
        # Adapt first layer for 1-channel input (grayscale mammography)
        # Original: 3 channels → 64 channels
        original_conv = self.backbone.features[0]
        self.backbone.features[0] = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new conv layer with weights from first 3 channels averaged
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Get final features dimension (DenseNet121 outputs 1024 channels)
        num_features = self.backbone.classifier.in_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, 512, 512)
        
        Returns:
            logits: (B, 3) classification logits
        """
        features = self.backbone.features(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# TASK 2: DETECTION - Mask R-CNN
# ============================================================================

def get_mask_rcnn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create Mask R-CNN for detection
    
    Args:
        num_classes: Number of classes (2: anomaly + background)
        pretrained: Use COCO pre-training
    
    Returns:
        Mask R-CNN model
    """
    model = maskrcnn_resnet50_fpn(pretrained=pretrained, num_classes=num_classes)
    
    # Adapt for 1-channel input
    original_conv = model.backbone.body.conv1
    model.backbone.body.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    
    # Initialize with averaged weights if pretrained
    if pretrained:
        with torch.no_grad():
            model.backbone.body.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    
    return model


# ============================================================================
# TASK 3: SEGMENTATION - U-Net
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block: Conv + ReLU + BatchNorm"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConvBlock(nn.Module):
    """Upsampling block: Transposed Conv + ReLU + BatchNorm"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNet(nn.Module):
    """
    U-Net for medical image segmentation
    Architecture:
    - Encoder: 4 levels of conv blocks with max pooling
    - Bottleneck: 1024 channels
    - Decoder: 4 levels of transpose conv with skip connections
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        Initialize U-Net
        
        Args:
            in_channels: Input channels (1 for grayscale)
            out_channels: Output channels (1 for binary segmentation)
        """
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder with skip connections
        self.upconv4 = UpConvBlock(1024, 512)
        self.dec4 = ConvBlock(512 + 512, 512)  # 512 from decoder + 512 from skip
        
        self.upconv3 = UpConvBlock(512, 256)
        self.dec3 = ConvBlock(256 + 256, 256)
        
        self.upconv2 = UpConvBlock(256, 128)
        self.dec2 = ConvBlock(128 + 128, 128)
        
        self.upconv1 = UpConvBlock(128, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()  # Binary output [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, 512, 512)
        
        Returns:
            output: (B, 1, 512, 512) segmentation mask probabilities
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Skip connection
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final output
        x = self.final(x)
        
        return x


# ============================================================================
# TASK 3: SEGMENTATION - DeepLabV3+
# ============================================================================

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP)"""
    
    def __init__(self, in_channels: int, out_channels: int, dilations: list = [1, 6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with different dilation rates
        for dilation in dilations[1:]:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Image pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        res = []
        
        for conv in self.convs[:-1]:
            res.append(conv(x))
        
        # Image pooling
        pool = self.convs[-1](x)
        pool = F.interpolate(pool, size=(h, w), mode='bilinear', align_corners=False)
        res.append(pool)
        
        # Concatenate
        x = torch.cat(res, dim=1)
        x = self.project(x)
        
        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for medical image segmentation
    Backbone: ResNet50 with dilated convolutions
    ASPP: Multi-scale feature extraction
    Decoder: Combines low-level and high-level features
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        Initialize DeepLabV3+
        
        Args:
            in_channels: Input channels (1 for grayscale)
            out_channels: Output channels (1 for binary segmentation)
        """
        super(DeepLabV3Plus, self).__init__()
        
        # Load ResNet50 backbone
        resnet = torchvision.models.resnet50(pretrained=True)
        
        # Adapt first layer for 1-channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy ResNet50 layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Dilated ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = self._make_dilated_layer(resnet.layer3, dilation=2)
        self.layer4 = self._make_dilated_layer(resnet.layer4, dilation=4)
        
        # ASPP module
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Sigmoid()  # Binary output
        )
        
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def _make_dilated_layer(self, layer: nn.Module, dilation: int) -> nn.Module:
        """Convert layer to use dilated convolutions"""
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, 512, 512)
        
        Returns:
            output: (B, 1, 512, 512) segmentation mask probabilities
        """
        h, w = x.shape[-2:]
        
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4
        
        x1 = self.layer1(x)  # 1/4, 256 channels
        x2 = self.layer2(x1)  # 1/8, 512 channels
        x3 = self.layer3(x2)  # 1/16, 1024 channels
        x4 = self.layer4(x3)  # 1/32, 2048 channels
        
        # ASPP
        x = self.aspp(x4)  # 1/32, 256 channels
        x = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        
        # Decoder with skip connection from layer1
        x1_proj = self.low_level_proj(x1)  # Project to 256 channels
        x = torch.cat([x, x1_proj], dim=1)  # Concatenate
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        # Final decoder
        x = self.decoder(x)
        
        return x


# ============================================================================
# Factory Functions
# ============================================================================

def get_classification_model(model_name: str = 'densenet121',
                            num_classes: int = 3,
                            pretrained: bool = True) -> nn.Module:
    """
    Get classification model
    
    Args:
        model_name: 'densenet121', 'resnet50', etc.
        num_classes: Number of classes
        pretrained: Use pre-training
    
    Returns:
        Model instance
    """
    if model_name.lower() == 'densenet121':
        return DenseNet121Classifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown classification model: {model_name}")


def get_detection_model(model_name: str = 'mask_rcnn',
                       num_classes: int = 2,
                       pretrained: bool = True) -> nn.Module:
    """
    Get detection model
    
    Args:
        model_name: 'mask_rcnn'
        num_classes: Number of classes
        pretrained: Use pre-training
    
    Returns:
        Model instance
    """
    if model_name.lower() == 'mask_rcnn':
        return get_mask_rcnn(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown detection model: {model_name}")


def get_segmentation_model(model_name: str = 'unet',
                          in_channels: int = 1,
                          out_channels: int = 1) -> nn.Module:
    """
    Get segmentation model
    
    Args:
        model_name: 'unet' or 'deeplabv3'
        in_channels: Input channels
        out_channels: Output channels
    
    Returns:
        Model instance
    """
    if model_name.lower() == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name.lower() == 'deeplabv3':
        return DeepLabV3Plus(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown segmentation model: {model_name}")
