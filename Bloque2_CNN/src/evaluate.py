"""
Evaluation module for multi-task CNN pipeline

Performs comprehensive evaluation including:
- Metrics calculation (AUC-ROC, Accuracy, Precision, Recall, F1, Dice, IoU, etc.)
- Confusion matrices
- ROC curves
- Error analysis
- Result visualization and reporting

Author: Bloque2_CNN Team
Date: May 2026
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import create_data_loaders
from src.models import (
    get_classification_model,
    get_detection_model,
    get_segmentation_model
)
from src.utils import (
    classification_metrics,
    segmentation_metrics,
    detection_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluator for classification task"""
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load trained model from checkpoint"""
        cfg = self.config['classification']
        
        self.model = get_classification_model(
            model_name=cfg['model'],
            num_classes=cfg['num_classes'],
            pretrained=False  # Don't load ImageNet weights
        ).to(self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Classification model loaded from {self.checkpoint_path}")
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        all_preds = []
        all_labels = []
        all_proba = []
        all_losses = []
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['classification'].to(self.device)
                
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                
                all_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_proba.extend(proba)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)
        
        # Calculate metrics
        metrics = classification_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_proba=all_proba
        )
        
        results = {
            'test_loss': np.mean(all_losses),
            'metrics': metrics,
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_proba.tolist(),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return results
    
    def generate_report(self, results, output_dir='results/classification'):
        """Generate evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Classification report
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION EVALUATION REPORT")
        logger.info("="*60)
        logger.info(f"Test Loss: {results['test_loss']:.6f}")
        logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        logger.info(f"AUC-ROC: {results['metrics'].get('auc_roc', 0):.4f}")
        logger.info(f"Precision (weighted): {results['metrics']['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {results['metrics']['recall_weighted']:.4f}")
        logger.info(f"F1-Score (weighted): {results['metrics']['f1_weighted']:.4f}")
        logger.info("\nPer-Class Metrics:")
        for class_name, idx in [('Benign', 0), ('Malignant', 1), ('Negative', 2)]:
            logger.info(f"{class_name}: P={results['metrics'][f'precision_class_{idx}']:.4f}, "
                       f"R={results['metrics'][f'recall_class_{idx}']:.4f}, "
                       f"F1={results['metrics'][f'f1_class_{idx}']:.4f}")
        logger.info("="*60)
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            results_copy = results.copy()
            results_copy['metrics'] = {k: float(v) if isinstance(v, np.floating) else v 
                                      for k, v in results['metrics'].items()}
            json.dump(results_copy, f, indent=2)
        
        # Confusion matrix plot
        cm = np.array(results['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant', 'Negative'],
                   yticklabels=['Benign', 'Malignant', 'Negative'])
        plt.title('Classification - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        logger.info(f"Results saved to {output_dir}")


class DetectionEvaluator:
    """Evaluator for detection task"""
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        self.model = get_detection_model(
            model_name=self.config['detection']['model'],
            num_classes=2
        ).to(self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Detection model loaded from {self.checkpoint_path}")
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        all_tp = []
        all_fp = []
        all_fn = []
        all_ious = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                gt_bboxes = batch['bbox']
                
                # Forward pass (convert to 3-channel for Mask R-CNN)
                outputs = self.model(images.unsqueeze(1).expand(-1, 3, -1, -1))
                
                for pred_output, gt_bbox in zip(outputs, gt_bboxes):
                    pred_boxes = pred_output['boxes'].cpu().numpy()
                    pred_scores = pred_output['scores'].cpu().numpy()
                    
                    if len(pred_boxes) > 0 and gt_bbox is not None:
                        # Calculate best IoU
                        ious = [self.calculate_iou(pred_box, gt_bbox) for pred_box in pred_boxes]
                        best_iou = max(ious)
                        all_ious.append(best_iou)
                        
                        if best_iou >= 0.5:
                            all_tp.append(1)
                            all_fp.append(0)
                        else:
                            all_tp.append(0)
                            all_fp.append(1)
                    elif len(pred_boxes) == 0 and gt_bbox is None:
                        all_tp.append(1)
                    elif len(pred_boxes) > 0 and gt_bbox is None:
                        all_fp.append(1)
                    else:
                        all_fn.append(1)
        
        tp = sum(all_tp) if all_tp else 0
        fp = sum(all_fp) if all_fp else 0
        fn = sum(all_fn) if all_fn else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(all_ious) if all_ious else 0
        
        results = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou,
            'iou_list': all_ious
        }
        
        return results
    
    def generate_report(self, results, output_dir='results/detection'):
        """Generate evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("DETECTION EVALUATION REPORT")
        logger.info("="*60)
        logger.info(f"True Positives: {results['tp']}")
        logger.info(f"False Positives: {results['fp']}")
        logger.info(f"False Negatives: {results['fn']}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-Score: {results['f1']:.4f}")
        logger.info(f"Mean IoU: {results['mean_iou']:.4f}")
        logger.info("="*60)
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


class SegmentationEvaluator:
    """Evaluator for segmentation task"""
    
    def __init__(self, config, checkpoint_path, device='cuda', model_name='unet'):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model_name = model_name
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        self.model = get_segmentation_model(
            model_name=self.model_name,
            in_channels=1,
            out_channels=1
        ).to(self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Segmentation model ({self.model_name}) loaded from {self.checkpoint_path}")
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        all_dice_scores = []
        all_iou_scores = []
        all_accuracy = []
        all_sensitivity = []
        all_specificity = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask']
                
                if masks is None:
                    continue
                
                masks = masks.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                preds = torch.sigmoid(outputs) > 0.5
                preds = preds.float()
                
                # Calculate metrics
                tp = (preds * masks).sum().item()
                tn = ((1 - preds) * (1 - masks)).sum().item()
                fp = (preds * (1 - masks)).sum().item()
                fn = ((1 - preds) * masks).sum().item()
                
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
                iou = tp / (tp + fp + fn + 1e-6)
                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
                sensitivity = tp / (tp + fn + 1e-6)
                specificity = tn / (tn + fp + 1e-6)
                
                all_dice_scores.append(dice)
                all_iou_scores.append(iou)
                all_accuracy.append(accuracy)
                all_sensitivity.append(sensitivity)
                all_specificity.append(specificity)
        
        results = {
            'dice': np.mean(all_dice_scores) if all_dice_scores else 0,
            'iou': np.mean(all_iou_scores) if all_iou_scores else 0,
            'accuracy': np.mean(all_accuracy) if all_accuracy else 0,
            'sensitivity': np.mean(all_sensitivity) if all_sensitivity else 0,
            'specificity': np.mean(all_specificity) if all_specificity else 0,
            'dice_std': np.std(all_dice_scores) if all_dice_scores else 0,
            'iou_std': np.std(all_iou_scores) if all_iou_scores else 0,
        }
        
        return results
    
    def generate_report(self, results, output_dir='results/segmentation'):
        """Generate evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info(f"SEGMENTATION EVALUATION REPORT ({self.model_name.upper()})")
        logger.info("="*60)
        logger.info(f"Dice Score: {results['dice']:.4f} ± {results['dice_std']:.4f}")
        logger.info(f"IoU: {results['iou']:.4f} ± {results['iou_std']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Sensitivity: {results['sensitivity']:.4f}")
        logger.info(f"Specificity: {results['specificity']:.4f}")
        logger.info("="*60)
        
        with open(os.path.join(output_dir, f'results_{self.model_name}.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


def evaluate_classification(config_path='config.yaml', checkpoint_path='models/classification/best.pth'):
    """Evaluate classification model"""
    logger.info("Starting Classification Evaluation...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _, _, test_loader = create_data_loaders(
        config_path=config_path,
        batch_size=config['classification']['batch_size'],
        num_workers=4
    )
    
    evaluator = ClassificationEvaluator(config, checkpoint_path, device=device)
    results = evaluator.evaluate(test_loader)
    evaluator.generate_report(results)
    
    logger.info("Classification evaluation completed!")
    return results


def evaluate_detection(config_path='config.yaml', checkpoint_path='models/detection/best.pth'):
    """Evaluate detection model"""
    logger.info("Starting Detection Evaluation...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _, _, test_loader = create_data_loaders(
        config_path=config_path,
        batch_size=config['detection']['batch_size'],
        num_workers=4
    )
    
    evaluator = DetectionEvaluator(config, checkpoint_path, device=device)
    results = evaluator.evaluate(test_loader)
    evaluator.generate_report(results)
    
    logger.info("Detection evaluation completed!")
    return results


def evaluate_segmentation(config_path='config.yaml', model_name='unet',
                         checkpoint_path=None):
    """Evaluate segmentation model"""
    logger.info(f"Starting Segmentation Evaluation ({model_name})...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if checkpoint_path is None:
        checkpoint_path = f'models/segmentation/best_{model_name}.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _, _, test_loader = create_data_loaders(
        config_path=config_path,
        batch_size=config['segmentation']['batch_size'],
        num_workers=4
    )
    
    evaluator = SegmentationEvaluator(config, checkpoint_path, device=device, model_name=model_name)
    results = evaluator.evaluate(test_loader)
    evaluator.generate_report(results)
    
    logger.info(f"Segmentation evaluation ({model_name}) completed!")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CNN models for breast cancer detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--task', type=str, choices=['classification', 'detection', 'segmentation', 'all'],
                       default='all', help='Which task to evaluate')
    parser.add_argument('--clf-checkpoint', type=str, default='models/classification/best.pth',
                       help='Classification model checkpoint')
    parser.add_argument('--det-checkpoint', type=str, default='models/detection/best.pth',
                       help='Detection model checkpoint')
    parser.add_argument('--seg-checkpoint', type=str, default=None,
                       help='Segmentation model checkpoint')
    parser.add_argument('--seg-model', type=str, choices=['unet', 'deeplabv3plus'],
                       default='unet', help='Segmentation model to evaluate')
    
    args = parser.parse_args()
    
    if args.task in ['classification', 'all']:
        evaluate_classification(config_path=args.config, checkpoint_path=args.clf_checkpoint)
    
    if args.task in ['detection', 'all']:
        evaluate_detection(config_path=args.config, checkpoint_path=args.det_checkpoint)
    
    if args.task in ['segmentation', 'all']:
        evaluate_segmentation(config_path=args.config, model_name=args.seg_model,
                            checkpoint_path=args.seg_checkpoint)
