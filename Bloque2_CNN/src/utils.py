"""
Funciones auxiliares para el proyecto de segmentación.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(torch.nn.Module):
    """Dice Coefficient Loss - Métrica común en segmentación médica"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        predictions = torch.sigmoid(predictions)
        
        intersection = (predictions * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class IoULoss(torch.nn.Module):
    """Intersection over Union Loss"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        predictions = torch.sigmoid(predictions)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(torch.nn.Module):
    """Combinación de Dice + BCE Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class SegmentationMetrics:
    """Calcula métricas de segmentación"""
    
    @staticmethod
    def compute_iou(predictions, targets, threshold=0.5):
        """Intersection over Union"""
        pred_binary = (predictions > threshold).astype(np.int32)
        target_binary = targets.astype(np.int32)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    @staticmethod
    def compute_dice(predictions, targets, threshold=0.5):
        """Dice Coefficient"""
        pred_binary = (predictions > threshold).astype(np.int32)
        target_binary = targets.astype(np.int32)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        dice = 2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-6)
        
        return dice
    
    @staticmethod
    def compute_metrics(predictions, targets, threshold=0.5):
        """Calcula todas las métricas"""
        pred_binary = (predictions > threshold).astype(np.int32).flatten()
        target_binary = targets.astype(np.int32).flatten()
        
        metrics = {
            'iou': jaccard_score(target_binary, pred_binary),
            'dice': f1_score(target_binary, pred_binary),
            'precision': precision_score(target_binary, pred_binary, zero_division=0),
            'recall': recall_score(target_binary, pred_binary, zero_division=0),
            'accuracy': accuracy_score(target_binary, pred_binary),
        }
        
        # F1 score = 2 * (precision * recall) / (precision + recall)
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        return metrics


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def apply_morphological_operations(mask, operation='closing', kernel_size=5):
    """
    Aplica operaciones morfológicas a la máscara.
    
    Args:
        mask: Máscara binaria (0-1)
        operation: 'closing', 'opening', 'dilation', 'erosion'
        kernel_size: Tamaño del kernel
    """
    import cv2
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'closing':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'opening':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'dilation':
        return cv2.dilate(mask, kernel)
    elif operation == 'erosion':
        return cv2.erode(mask, kernel)
    else:
        raise ValueError(f"Operación no reconocida: {operation}")


def post_process_mask(mask, min_area=100):
    """
    Post-procesa máscara predicha.
    
    Args:
        mask: Máscara predicha (0-1)
        min_area: Área mínima para conservar componentes
    """
    import cv2
    
    # Binarizar
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Encontrar componentes conectados
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Filtrar por área
    output = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        if np.sum(component_mask) >= min_area:
            output += component_mask
    
    return (output > 0).astype(np.float32)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_segmentation(image, mask, prediction, save_path=None):
    """
    Visualiza imagen, máscara ground-truth y predicción.
    
    Args:
        image: Imagen original (escala de grises, 0-1)
        mask: Máscara ground-truth (0-1)
        prediction: Predicción (0-1)
        save_path: Ruta para guardar (opcional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Imagen original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicción
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(prediction, cmap='Greens', alpha=0.5)
    axes[2].set_title('Predicción')
    axes[2].axis('off')
    
    # Diferencia
    diff = np.abs(mask - prediction)
    axes[3].imshow(diff, cmap='jet')
    axes[3].set_title('Diferencia (abs)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


# ============================================================================
# UTILITIES
# ============================================================================

def create_directory(path):
    """Crea directorio si no existe"""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(config_path):
    """Carga configuración desde YAML"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Guarda checkpoint del modelo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_path):
    """Carga checkpoint del modelo"""
    checkpoint = torch.load(load_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    return model, optimizer, epoch, metrics


if __name__ == "__main__":
    print("✓ Utilidades de segmentación cargadas")
