#!/usr/bin/env python
"""
FINAL TRAIN MODULE - Bloque2_CNN
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging

from src.data_loader import MetadataLoader, create_data_loaders
from src.models import (
    get_classification_model,
    get_detection_model,
    get_segmentation_model
)
from src.utils import (
    ClassificationLoss,
    classification_metrics
)

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_batch(batch):
    if isinstance(batch, dict):
        return batch

    if isinstance(batch, list):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['classification_label'] for item in batch])
        return {
            'image': images,
            'classification_label': labels,
            'bbox': [item.get('bbox') for item in batch],
            'mask': [item.get('mask') for item in batch],
            'pla': [item.get('pla') for item in batch],
        }

    raise TypeError(f"Unsupported batch type: {type(batch)!r}")

# ===== SEED =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===== EARLY STOPPING =====
class EarlyStopping:
    def __init__(self, patience=20, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, loss, model):
        if loss < self.best:
            self.best = loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ===== CLASSIFICATION =====
class ClassificationTrainer:

    def __init__(self, config, device):
        self.cfg = config['classification']
        self.device = device

        self.model = get_classification_model(
            self.cfg['model'],
            self.cfg['num_classes'],
            self.cfg['pretrained']
        ).to(device)

        weights = torch.tensor(self.cfg['class_weights']).to(device)
        self.loss_fn = ClassificationLoss(weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])

        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        preds_all, labels_all = [], []

        for batch in loader:
            batch = _normalize_batch(batch)
            x = batch['image'].to(self.device)
            y = batch['classification_label'].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = self.model(x)
                loss = self.loss_fn(out, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds_all.extend(torch.argmax(out, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())

        metrics = classification_metrics(
            np.array(labels_all),
            np.array(preds_all),
            None
        )

        return total_loss / len(loader), metrics

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = _normalize_batch(batch)
                x = batch['image'].to(self.device)
                y = batch['classification_label'].to(self.device)

                out = self.model(x)
                loss = self.loss_fn(out, y)
                total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, train_loader, val_loader):
        es = EarlyStopping(self.cfg['early_stopping_patience'],
                           'models/classification/best.pth')

        for epoch in range(self.cfg['epochs']):
            tl, _ = self.train_epoch(train_loader)
            vl = self.validate(val_loader)

            logger.info(f"[CLS] Epoch {epoch} | TL {tl:.4f} | VL {vl:.4f}")

            es(vl, self.model)
            if es.early_stop:
                break


# ===== SEGMENTATION =====
class SegmentationTrainer:

    def __init__(self, config, device):
        self.cfg = config['segmentation']
        self.device = device

        self.model = get_segmentation_model(
            'unet', 1, 1
        ).to(device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])
        self.scaler = torch.cuda.amp.GradScaler()

    def dice(self, preds, targets):
        preds = (preds > 0.5).float()
        intersection = (preds * targets).sum()
        return (2 * intersection) / (preds.sum() + targets.sum() + 1e-6)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        valid = 0

        for batch in loader:
            batch = _normalize_batch(batch)
            x = batch['image'].to(self.device)
            masks = batch['mask']

            if isinstance(masks, list):
                valid_pairs = [
                    (image, mask)
                    for image, mask in zip(x, masks)
                    if mask is not None
                ]
                if not valid_pairs:
                    continue
                x = torch.stack([item[0] for item in valid_pairs]).to(self.device)
                y = torch.stack([item[1] for item in valid_pairs]).to(self.device)
            elif masks is None:
                continue
            else:
                y = masks.to(self.device)

            if y.ndim == 3:
                y = y.unsqueeze(1)

            y = y.float()
            valid += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = self.model(x)
                loss = self.loss_fn(out, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / max(valid, 1)

    def validate(self, loader):
        self.model.eval()
        dice_scores = []

        with torch.no_grad():
            for batch in loader:
                batch = _normalize_batch(batch)
                x = batch['image'].to(self.device)
                masks = batch['mask']

                if isinstance(masks, list):
                    valid_pairs = [
                        (image, mask)
                        for image, mask in zip(x, masks)
                        if mask is not None
                    ]
                    if not valid_pairs:
                        continue
                    x = torch.stack([item[0] for item in valid_pairs]).to(self.device)
                    y = torch.stack([item[1] for item in valid_pairs]).to(self.device)
                elif masks is None:
                    continue
                else:
                    y = masks.to(self.device)

                if y.ndim == 3:
                    y = y.unsqueeze(1)

                y = y.float()
                out = torch.sigmoid(self.model(x))

                dice_scores.append(self.dice(out, y).item())

        return np.mean(dice_scores)

    def train(self, train_loader, val_loader):
        es = EarlyStopping(self.cfg['early_stopping_patience'],
                           'models/segmentation/best.pth')

        for epoch in range(self.cfg['epochs']):
            tl = self.train_epoch(train_loader)
            vd = self.validate(val_loader)

            logger.info(f"[SEG] Epoch {epoch} | Loss {tl:.4f} | Dice {vd:.4f}")

            es(-vd, self.model)
            if es.early_stop:
                break


# ===== MAIN =====
def main(config_path, task):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config['training']['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    metadata_loader = MetadataLoader(config['data']['metadata_path'])

    train_loader, val_loader, _ = create_data_loaders(
        config=config,
        metadata_loader=metadata_loader
    )

    if task in ['classification', 'all']:
        ClassificationTrainer(config, device).train(train_loader, val_loader)

    if task in ['detection', 'all']:
        logger.warning("Detection training is not implemented yet in this build; skipping.")

    if task in ['segmentation', 'all']:
        SegmentationTrainer(config, device).train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--task', default='all')

    args = parser.parse_args()
    main(args.config, args.task)