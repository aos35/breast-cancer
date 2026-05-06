"""
Data Loader Module for Bloque2_CNN
Handles multi-task dataset loading: Classification + Detection + Segmentation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2


# =========================
# METADATA LOADER
# =========================
class MetadataLoader:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.df = pd.read_excel(metadata_path)

        self.class_map = {
            'B': 0,
            'M': 1,
            'N': 2
        }

    def get_image_metadata(self, image_id: str) -> Dict:
        rows = self.df[self.df['Image_Reference'] == image_id]

        if len(rows) == 0:
            return {
                'classification_label': 2,
                'has_anomaly': False,
                'bbox': None
            }

        row = rows.iloc[0]

        class_label = row['Class_Abnormality']
        classification_label = self.class_map.get(class_label, 2)

        has_anomaly = pd.notna(class_label) and class_label != 'N'

        bbox = None
        if pd.notna(row['X_Coordinate']) and pd.notna(row['Y_Coordinate']) and pd.notna(row['Radius_pixels']):
            bbox = (
                float(row['X_Coordinate']),
                float(row['Y_Coordinate']),
                float(row['Radius_pixels'])
            )

        return {
            'classification_label': classification_label,
            'has_anomaly': has_anomaly,
            'bbox': bbox
        }


# =========================
# DATASET
# =========================
class DMIDMultiTaskDataset(Dataset):

    def __init__(
        self,
        tiff_dir: str,
        masks_dir: str,
        pla_dir: str,
        metadata_loader: MetadataLoader,
        split: str = 'train',
        image_ids: Optional[list] = None,
        augmentation: bool = True
    ):
        self.tiff_dir = Path(tiff_dir)
        self.masks_dir = Path(masks_dir)
        self.pla_dir = Path(pla_dir)
        self.metadata_loader = metadata_loader
        self.split = split
        self.augmentation = augmentation

        self.tiff_files = sorted(list(self.tiff_dir.glob('*.tif')))

        if image_ids is not None:
            self.tiff_files = [f for f in self.tiff_files if f.stem in image_ids]

        self._setup_augmentation()

    def _setup_augmentation(self):
        if self.augmentation and self.split == 'train':
            self.transform = A.Compose([
                A.Rotate(limit=15, p=0.5),
                A.Affine(scale=(0.8, 1.2), p=0.5),  # FIX: Zoom no existe en Albumentations
                A.GaussianBlur(p=0.3),
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.tiff_files)

    def __getitem__(self, idx):

        tiff_path = self.tiff_files[idx]
        image_id = tiff_path.stem

        # Load image
        image = cv2.imread(str(tiff_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((512, 512), dtype=np.uint8)

        image = image.astype(np.float32) / 255.0

        metadata = self.metadata_loader.get_image_metadata(image_id)

        # Load mask
        mask = None
        mask_path = self.masks_dir / f"{image_id}.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32)

        # Load PLA
        pla = None
        pla_path = self.pla_dir / f"{image_id}.npy"
        if pla_path.exists():
            pla = np.load(pla_path).astype(np.float32)

        # =====================
        # BBOX NORMALIZATION
        # =====================
        bbox = None
        if metadata['bbox'] is not None:
            x, y, r = metadata['bbox']

            x_min = max(0, (x - r))
            y_min = max(0, (y - r))
            x_max = min(512, (x + r))
            y_max = min(512, (y + r))

            bbox = [x_min, y_min, x_max, y_max]

        # =====================
        # AUGMENTATION
        # =====================
        if self.augmentation and self.split == 'train':
            bboxes = [bbox] if bbox is not None else []

            augmented = self.transform(
                image=image,
                mask=mask if mask is not None else np.zeros_like(image),
                bboxes=bboxes,
                labels=[1] if bboxes else []
            )

            image = augmented['image']
            mask = augmented.get('mask', None)

            if augmented['bboxes']:
                bbox = torch.tensor(augmented['bboxes'][0], dtype=torch.float32)
            else:
                bbox = None
        else:
            image = torch.tensor(image).unsqueeze(0)

            if mask is not None:
                mask = torch.tensor(mask)

            if bbox is not None:
                bbox = torch.tensor(bbox, dtype=torch.float32)

        return {
            'image': image,
            'classification_label': torch.tensor(metadata['classification_label'], dtype=torch.long),
            'bbox': bbox,
            'mask': mask,
            'pla': pla
        }


# =========================
# COLLATE FUNCTION (CRÍTICO PARA DETECTION)
# =========================
def collate_fn(batch):
    return batch


# =========================
# DATALOADERS
# =========================
def create_data_loaders(config, metadata_loader, batch_size=None):

    tiff_dir = config['data']['tiff_dir']
    masks_dir = config['data']['masks_dir']
    pla_dir = config['data']['pla_dir']

    batch_size = batch_size or config['classification']['batch_size']

    full_dataset = DMIDMultiTaskDataset(
        tiff_dir, masks_dir, pla_dir, metadata_loader,
        split='all', augmentation=False
    )

    image_ids = [f.stem for f in full_dataset.tiff_files]

    np.random.seed(config['training']['seed'])
    indices = np.random.permutation(len(image_ids))

    train_ids = [image_ids[i] for i in indices[:int(0.6 * len(indices))]]
    val_ids = [image_ids[i] for i in indices[int(0.6 * len(indices)):int(0.8 * len(indices))]]
    test_ids = [image_ids[i] for i in indices[int(0.8 * len(indices)):]]

    train_dataset = DMIDMultiTaskDataset(tiff_dir, masks_dir, pla_dir, metadata_loader, 'train', train_ids, True)
    val_dataset = DMIDMultiTaskDataset(tiff_dir, masks_dir, pla_dir, metadata_loader, 'val', val_ids, False)
    test_dataset = DMIDMultiTaskDataset(tiff_dir, masks_dir, pla_dir, metadata_loader, 'test', test_ids, False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn  # 🔴 IMPORTANTE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader