"""
Utilities Module for Bloque2_CNN
Loss functions, metrics, and helper functions for training and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import numpy as np
from typing import Dict, Tuple


# ============================================================================
# LOSS FUNCTIONS - CLASSIFICATION
# ============================================================================

class ClassificationLoss(nn.Module):
    """Weighted CrossEntropy loss for handling class imbalance"""
    
    def __init__(self, class_weights: torch.Tensor = None):
        """
        Initialize Classification Loss
        
        Args:
            class_weights: Tensor of shape (num_classes,) for weighted loss
                         If None, uses default uniform weights
        """
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate classification loss
        
        Args:
            predictions: (B, num_classes) logits
            targets: (B,) target class indices
        
        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


# ============================================================================
# LOSS FUNCTIONS - SEGMENTATION
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss: 1 - 2*(TP)/(2*TP+FP+FN)
    Excellent for class imbalance (background >> foreground)
    """
    
    def __init__(self, smooth: float = 1e-5):
        """
        Initialize Dice Loss
        
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss
        
        Args:
            predictions: (B, 1, H, W) probabilities [0, 1]
            targets: (B, 1, H, W) ground truth binary [0, 1]
        
        Returns:
            Scalar loss value
        """
        # Flatten
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Loss (1 - Dice)
        return 1.0 - dice_coeff


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy Loss"""
    
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate BCE loss
        
        Args:
            predictions: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) ground truth [0, 1]
        
        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


class CombinedSegmentationLoss(nn.Module):
    """Combined loss: α*DiceLoss + (1-α)*BCELoss"""
    
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-5):
        """
        Initialize Combined Loss
        
        Args:
            alpha: Weight for Dice loss (1-alpha for BCE)
            smooth: Smoothing constant
        """
        super(CombinedSegmentationLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            predictions: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) ground truth [0, 1]
        
        Returns:
            Scalar loss value
        """
        # Sigmoid for Dice loss
        pred_sigmoid = torch.sigmoid(predictions)
        
        dice = self.dice_loss(pred_sigmoid, targets)
        bce = self.bce_loss(predictions, targets)
        
        return self.alpha * dice + (1 - self.alpha) * bce


# ============================================================================
# METRICS - CLASSIFICATION
# ============================================================================

def classification_metrics(predictions: torch.Tensor,
                          targets: torch.Tensor,
                          class_names: list = None) -> Dict:
    """
    Calculate comprehensive classification metrics
    
    Args:
        predictions: (B, num_classes) logits
        targets: (B,) target indices
        class_names: List of class names for reporting
    
    Returns:
        Dict with metrics
    """
    # Convert to numpy
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Get class predictions
    pred_classes = np.argmax(preds_np, axis=1)
    
    # Get probabilities
    pred_probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()
    
    # Accuracy
    accuracy = accuracy_score(targets_np, pred_classes)
    
    # Per-class metrics
    precision = precision_score(targets_np, pred_classes, average=None, zero_division=0)
    recall = recall_score(targets_np, pred_classes, average=None, zero_division=0)
    f1 = f1_score(targets_np, pred_classes, average=None, zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(targets_np, pred_classes, average='weighted', zero_division=0)
    recall_weighted = recall_score(targets_np, pred_classes, average='weighted', zero_division=0)
    f1_weighted = f1_score(targets_np, pred_classes, average='weighted', zero_division=0)
    
    # AUC-ROC (one-vs-rest for multi-class)
    try:
        auc_roc = roc_auc_score(targets_np, pred_probs, multi_class='ovr', average='weighted')
    except:
        auc_roc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(targets_np, pred_classes)
    
    # Build results dictionary
    results = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auc_roc': auc_roc,
        'confusion_matrix': cm.tolist()
    }
    
    # Per-class metrics
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(precision))]
    
    for i, class_name in enumerate(class_names):
        results[f'{class_name}_precision'] = precision[i]
        results[f'{class_name}_recall'] = recall[i]
        results[f'{class_name}_f1'] = f1[i]
    
    return results


# ============================================================================
# METRICS - SEGMENTATION
# ============================================================================

def segmentation_metrics(predictions: torch.Tensor,
                        targets: torch.Tensor,
                        threshold: float = 0.5) -> Dict:
    """
    Calculate segmentation metrics
    
    Args:
        predictions: (B, 1, H, W) probabilities [0, 1]
        targets: (B, 1, H, W) ground truth binary [0, 1]
        threshold: Probability threshold for binarization
    
    Returns:
        Dict with metrics
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred_np > threshold).astype(np.float32)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_np.flatten()
    
    # Calculate metrics
    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    
    # Dice Score
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    
    # Jaccard/IoU
    iou = tp / (tp + fp + fn + 1e-7)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    
    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn + 1e-7)
    
    # Specificity
    specificity = tn / (tn + fp + 1e-7)
    
    # Precision
    precision = tp / (tp + fp + 1e-7)
    
    # Hausdorff distance (simple approximation)
    hausdorff = compute_hausdorff_distance(pred_binary, target_np)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'hausdorff': hausdorff
    }


def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Hausdorff distance between prediction and target
    
    Args:
        pred: Binary prediction (B, 1, H, W)
        target: Binary target (B, 1, H, W)
    
    Returns:
        Mean Hausdorff distance
    """
    from scipy.spatial.distance import directed_hausdorff
    
    distances = []
    
    for b in range(pred.shape[0]):
        # Get contours
        pred_coords = np.where(pred[b, 0] > 0.5)
        target_coords = np.where(target[b, 0] > 0.5)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            continue
        
        # Stack coordinates
        pred_pts = np.column_stack(pred_coords)
        target_pts = np.column_stack(target_coords)
        
        # Hausdorff distance
        hd = max(
            directed_hausdorff(pred_pts, target_pts)[0],
            directed_hausdorff(target_pts, pred_pts)[0]
        )
        distances.append(hd)
    
    return float(np.mean(distances)) if distances else 0.0


# ============================================================================
# METRICS - DETECTION
# ============================================================================

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU (Intersection over Union) between two boxes
    
    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU value (0-1)
    """
    # Intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0.0
    
    intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-7)


def detection_metrics(pred_boxes: list, pred_scores: list,
                     gt_boxes: list, iou_threshold: float = 0.5) -> Dict:
    """
    Calculate detection metrics
    
    Args:
        pred_boxes: List of predicted boxes (N, 4)
        pred_scores: List of prediction scores (N,)
        gt_boxes: List of ground truth boxes (M, 4)
        iou_threshold: IoU threshold for TP/FP determination
    
    Returns:
        Dict with metrics
    """
    # Sort by confidence
    if len(pred_scores) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'map': 0.0, 'iou_mean': 0.0}
    
    sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
    
    # Compute TP/FP
    tp = 0
    fp = 0
    ious = []
    
    gt_matched = set()
    
    for idx in sorted_indices:
        pred_box = pred_boxes[idx]
        
        # Find best matching GT box
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in gt_matched:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            ious.append(best_iou)
            gt_matched.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(gt_boxes) - tp
    
    # Calculate metrics
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    # mAP (simplified)
    map_score = (precision + recall) / 2.0 if (tp + fp + fn) > 0 else 0.0
    
    iou_mean = float(np.mean(ious)) if ious else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'map': map_score,
        'iou_mean': iou_mean,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_class_weights(class_counts: Dict[int, int], num_classes: int) -> torch.Tensor:
    """
    Calculate balanced class weights for weighted loss
    
    Args:
        class_counts: Dict with class_idx -> count
        num_classes: Total number of classes
    
    Returns:
        Tensor of weights
    """
    total = sum(class_counts.values())
    weights = []
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * max(count, 1))
        weights.append(weight)
    
    # Normalize
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    
    return weights


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1]
    
    Args:
        image: Input image
    
    Returns:
        Normalized image
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
    
    return ((image - img_min) / (img_max - img_min)).astype(np.float32)


def postprocess_segmentation(mask: np.ndarray,
                            threshold: float = 0.5,
                            min_size: int = 50) -> np.ndarray:
    """
    Post-process segmentation mask
    
    Args:
        mask: Raw mask probabilities
        threshold: Binarization threshold
        min_size: Minimum object size
    
    Returns:
        Post-processed binary mask
    """
    from scipy import ndimage
    
    # Binarize
    binary = (mask > threshold).astype(np.uint8)
    
    # Morphological closing
    kernel = ndimage.generate_binary_structure(2, 1)
    binary = ndimage.binary_closing(binary, structure=kernel).astype(np.uint8)
    
    # Remove small objects
    labeled, num_features = ndimage.label(binary)
    
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size < min_size:
            binary[labeled == i] = 0
    
    return binary.astype(np.float32)
