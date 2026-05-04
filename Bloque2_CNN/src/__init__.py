"""Bloque2_CNN - Multi-Task CNN Pipeline for Breast Cancer Detection"""

from .data_loader import MetadataLoader, DMIDMultiTaskDataset, create_data_loaders
from .models import (
    DenseNet121Classifier,
    get_mask_rcnn,
    UNet,
    DeepLabV3Plus,
    get_classification_model,
    get_detection_model,
    get_segmentation_model
)
from .utils import (
    ClassificationLoss,
    DiceLoss,
    BCEWithLogitsLoss,
    CombinedSegmentationLoss,
    classification_metrics,
    segmentation_metrics,
    detection_metrics,
    get_class_weights,
    normalize_image,
    postprocess_segmentation
)

__version__ = "1.0.0"
__author__ = "Proyecto Redes Neuronales 2026"

__all__ = [
    # Data
    'MetadataLoader',
    'DMIDMultiTaskDataset',
    'create_data_loaders',
    
    # Models
    'DenseNet121Classifier',
    'get_mask_rcnn',
    'UNet',
    'DeepLabV3Plus',
    'get_classification_model',
    'get_detection_model',
    'get_segmentation_model',
    
    # Utils
    'ClassificationLoss',
    'DiceLoss',
    'BCEWithLogitsLoss',
    'CombinedSegmentationLoss',
    'classification_metrics',
    'segmentation_metrics',
    'detection_metrics',
    'get_class_weights',
    'normalize_image',
    'postprocess_segmentation'
]
