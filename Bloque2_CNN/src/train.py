"""
Training module for multi-task CNN pipeline (Classification, Detection, Segmentation)

Supports:
- DenseNet121 for Classification
- Mask R-CNN for Detection
- U-Net and DeepLabV3+ for Segmentation
- K-fold cross-validation
- Early stopping and checkpoint saving
- Metric logging

Author: Bloque2_CNN Team
Date: May 2026
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

from src.data_loader import MetadataLoader, create_data_loaders, DMIDMultiTaskDataset
from src.models import (
    get_classification_model,
    get_detection_model,
    get_segmentation_model
)
from src.utils import (
    ClassificationLoss,
    DiceLoss,
    classification_metrics,
    segmentation_metrics,
    detection_metrics,
    get_class_weights
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping mechanism to prevent overfitting"""
    
    def __init__(self, patience=20, verbose=False, delta=0.0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model, optimizer=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, optimizer=None):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


class ClassificationTrainer:
    """Trainer for classification task (DenseNet121)"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.histories = []
        
    def setup_model(self):
        """Initialize model, optimizer, loss, scheduler"""
        cfg = self.config['classification']
        
        # Model
        self.model = get_classification_model(
            model_name=cfg['model'],
            num_classes=cfg['num_classes'],
            pretrained=cfg['pretrained']
        ).to(self.device)
        
        # Loss
        class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32).to(self.device)
        self.loss_fn = ClassificationLoss(class_weights=class_weights)
        
        # Optimizer
        if cfg['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'])
        elif cfg['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=cfg['learning_rate'], momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
        
        # Scheduler
        if cfg['scheduler'].lower() == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=cfg['scheduler_params']['gamma']
            )
        elif cfg['scheduler'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg['scheduler_params']['step_size'],
                gamma=cfg['scheduler_params']['gamma']
            )
        else:
            self.scheduler = None
        
        logger.info(f"Classification model initialized: {cfg['model']}")
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['classification'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        metrics = classification_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_proba=None
        )
        
        return avg_loss, metrics
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_proba = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['classification'].to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_proba.extend(proba)
        
        avg_loss = total_loss / len(val_loader)
        metrics = classification_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_proba=np.array(all_proba)
        )
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, epochs=200, checkpoint_path='models/classification/best.pth'):
        """Full training loop with early stopping"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        early_stopping = EarlyStopping(
            patience=self.config['classification'].get('early_stopping_patience', 20),
            verbose=True,
            path=checkpoint_path
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_auc_roc': []
        }
        
        for epoch in range(epochs):
            logger.info(f'\nEpoch {epoch + 1}/{epochs}')
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_metrics.get('accuracy', 0))
            history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            history['val_auc_roc'].append(val_metrics.get('auc_roc', 0))
            
            logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            logger.info(f'Val Accuracy: {val_metrics["accuracy"]:.4f}, Val AUC-ROC: {val_metrics.get("auc_roc", 0):.4f}')
            
            # Early stopping
            early_stopping(val_loss, self.model, self.optimizer)
            if early_stopping.early_stop:
                logger.info('Early stopping triggered!')
                break
        
        return history


class DetectionTrainer:
    """Trainer for detection task (Mask R-CNN)"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.histories = []
        
    def setup_model(self):
        """Initialize model and optimizer"""
        cfg = self.config['detection']
        
        self.model = get_detection_model(
            model_name=cfg['model'],
            num_classes=2  # foreground + background
        ).to(self.device)
        
        # Optimizer
        if cfg['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'])
        elif cfg['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg['learning_rate'],
                momentum=cfg.get('momentum', 0.9)
            )
        
        logger.info(f"Detection model initialized: {cfg['model']}")
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            bboxes = batch['bbox']  # List or None
            
            # Prepare targets (Mask R-CNN expects list of dicts)
            targets = []
            for i, (img, bbox) in enumerate(zip(images, bboxes)):
                if bbox is not None:
                    target = {
                        'boxes': torch.tensor([bbox], dtype=torch.float32).to(self.device),
                        'labels': torch.tensor([1], dtype=torch.int64).to(self.device)  # 1 for anomaly
                    }
                else:
                    target = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32).to(self.device),
                        'labels': torch.zeros(0, dtype=torch.int64).to(self.device)
                    }
                targets.append(target)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model(images.unsqueeze(1).expand(-1, 3, -1, -1), targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {losses.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                bboxes = batch['bbox']
                
                targets = []
                for i, (img, bbox) in enumerate(zip(images, bboxes)):
                    if bbox is not None:
                        target = {
                            'boxes': torch.tensor([bbox], dtype=torch.float32).to(self.device),
                            'labels': torch.tensor([1], dtype=torch.int64).to(self.device)
                        }
                    else:
                        target = {
                            'boxes': torch.zeros((0, 4), dtype=torch.float32).to(self.device),
                            'labels': torch.zeros(0, dtype=torch.int64).to(self.device)
                        }
                    targets.append(target)
                
                loss_dict = self.model(images.unsqueeze(1).expand(-1, 3, -1, -1), targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=150, checkpoint_path='models/detection/best.pth'):
        """Full training loop"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        best_loss = float('inf')
        patience = self.config['detection'].get('early_stopping_patience', 20)
        patience_counter = 0
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            logger.info(f'\nEpoch {epoch + 1}/{epochs}')
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info('Early stopping triggered!')
                    break
        
        return history


class SegmentationTrainer:
    """Trainer for segmentation task (U-Net / DeepLabV3+)"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
    def setup_model(self, model_name='unet'):
        """Initialize model, optimizer, loss"""
        cfg = self.config['segmentation']
        
        self.model = get_segmentation_model(
            model_name=model_name,
            in_channels=1,
            out_channels=1
        ).to(self.device)
        
        # Loss: Combined Dice + BCE
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Optimizer
        if cfg['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'])
        elif cfg['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=cfg['learning_rate'])
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        logger.info(f"Segmentation model initialized: {model_name}")
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask']
            
            if masks is None:
                continue
            
            masks = masks.to(self.device).float().unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_dice_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask']
                
                if masks is None:
                    continue
                
                masks = masks.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                total_loss += loss.item()
                
                # Calculate Dice score
                preds = torch.sigmoid(outputs) > 0.5
                dice = (2 * (preds & masks).sum()) / (preds.sum() + masks.sum() + 1e-6)
                all_dice_scores.append(dice.item())
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0
        
        return avg_loss, avg_dice
    
    def train(self, train_loader, val_loader, model_name='unet', epochs=200, 
              checkpoint_path='models/segmentation/best.pth'):
        """Full training loop"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        early_stopping = EarlyStopping(
            patience=self.config['segmentation'].get('early_stopping_patience', 30),
            verbose=True,
            path=checkpoint_path
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
        
        for epoch in range(epochs):
            logger.info(f'\nEpoch {epoch + 1}/{epochs}')
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_dice = self.validate(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}')
            
            early_stopping(val_loss, self.model, self.optimizer)
            if early_stopping.early_stop:
                logger.info('Early stopping triggered!')
                break
        
        return history


def train_classification(config_path='config.yaml', fold=None):
    """Train classification model"""
    logger.info("Starting Classification Training...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config_path=config_path,
        batch_size=config['classification']['batch_size'],
        num_workers=4
    )
    
    # Initialize trainer
    trainer = ClassificationTrainer(config, device=device)
    trainer.setup_model()
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config['classification']['epochs'],
        checkpoint_path='models/classification/best.pth'
    )
    
    logger.info("Classification training completed!")
    return history


def train_detection(config_path='config.yaml'):
    """Train detection model"""
    logger.info("Starting Detection Training...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config_path=config_path,
        batch_size=config['detection']['batch_size'],
        num_workers=4
    )
    
    # Initialize trainer
    trainer = DetectionTrainer(config, device=device)
    trainer.setup_model()
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config['detection']['epochs'],
        checkpoint_path='models/detection/best.pth'
    )
    
    logger.info("Detection training completed!")
    return history


def train_segmentation(config_path='config.yaml', model_name='unet'):
    """Train segmentation model"""
    logger.info(f"Starting Segmentation Training ({model_name})...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config_path=config_path,
        batch_size=config['segmentation']['batch_size'],
        num_workers=4
    )
    
    # Initialize trainer
    trainer = SegmentationTrainer(config, device=device)
    trainer.setup_model(model_name=model_name)
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        model_name=model_name,
        epochs=config['segmentation']['epochs'],
        checkpoint_path=f'models/segmentation/best_{model_name}.pth'
    )
    
    logger.info(f"Segmentation training ({model_name}) completed!")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN models for breast cancer detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--task', type=str, choices=['classification', 'detection', 'segmentation', 'all'],
                       default='all', help='Which task to train')
    parser.add_argument('--model', type=str, choices=['unet', 'deeplabv3plus'], 
                       default='unet', help='Segmentation model to train')
    
    args = parser.parse_args()
    
    if args.task in ['classification', 'all']:
        train_classification(config_path=args.config)
    
    if args.task in ['detection', 'all']:
        train_detection(config_path=args.config)
    
    if args.task in ['segmentation', 'all']:
        train_segmentation(config_path=args.config, model_name=args.model)
