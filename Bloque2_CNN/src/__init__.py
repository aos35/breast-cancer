"""
Bloque2_CNN - CNN para Segmentación de Lesiones Mamarias

Módulos:
- data_loader: Carga de TIFF, Masks, PLA desde DMID_PNG
- models: Arquitecturas (U-Net, FCN, DeepLab V3)
- utils: Funciones auxiliares (loss, metrics, visualization)
- train: Loop de entrenamiento
- evaluate: Evaluación en test set
"""

from .data_loader import DMIDSegmentationDataset, SegmentationDataModule
from .models import UNet, FCN, DeepLabV3, create_model
from .utils import (
    DiceLoss,
    IoULoss,
    CombinedLoss,
    SegmentationMetrics,
    visualize_segmentation
)

__version__ = "1.0.0"
__author__ = "Estudiante Redes Neuronales"

__all__ = [
    'DMIDSegmentationDataset',
    'SegmentationDataModule',
    'UNet',
    'FCN',
    'DeepLabV3',
    'create_model',
    'DiceLoss',
    'IoULoss',
    'CombinedLoss',
    'SegmentationMetrics',
    'visualize_segmentation',
]
