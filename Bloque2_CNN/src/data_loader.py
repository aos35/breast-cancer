"""
Data Loader Module for Bloque2_CNN
Handles multi-task dataset loading: Classification + Detection + Segmentation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2


class MetadataLoader:
    """Loads and parses metadata.xlsx for classification labels and bounding boxes"""
    
    def __init__(self, metadata_path: str):
        """
        Initialize MetadataLoader
        
        Args:
            metadata_path: Path to Metadata.xlsx
        """
        self.metadata_path = metadata_path
        self.df = pd.read_excel(metadata_path)
        
        # Map class labels to indices
        self.class_map = {
            'B': 0,           # Benign
            'M': 1,           # Malignant
            'N': 2,           # No Defined
            None: 2,          # Treat None as Negative (class 2)
            np.nan: 2         # NaN is Negative
        }
        
        self.class_names = ['Benign', 'Malignant', 'Negative']
        
    def get_image_metadata(self, image_id: str) -> Dict:
        """
        Get metadata for specific image
        
        Args:
            image_id: Image reference (e.g., 'IMG001')
        
        Returns:
            Dict with keys:
                - classification_label: int (0/1/2)
                - has_anomaly: bool
                - bbox: tuple (x, y, radius) or None
                - anomaly_type: str
                - tissue_type: str
                - view: str
        """
        rows = self.df[self.df['Image_Reference'] == image_id]
        
        if len(rows) == 0:
            return {
                'classification_label': 2,  # Default to Negative
                'has_anomaly': False,
                'bbox': None,
                'anomaly_type': 'NORM',
                'tissue_type': 'G',
                'view': 'UNKNOWN'
            }
        
        row = rows.iloc[0]
        
        # Get classification label
        class_label = row['Class_Abnormality']
        classification_label = self.class_map.get(class_label, 2)
        
        # Check if has anomaly
        has_anomaly = pd.notna(class_label) and class_label != 'N'
        
        # Extract bounding box coordinates
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
            'bbox': bbox,
            'anomaly_type': str(row['Abnormality_Type']),
            'tissue_type': str(row['Background_Tissue']),
            'view': str(row['Mammogram_View'])
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance"""
        counts = self.df['Class_Abnormality'].value_counts()
        total = len(self.df)
        
        # Inverse frequency weighting
        weights = {}
        for cls_name, cls_idx in self.class_map.items():
            if pd.isna(cls_name):
                count = len(self.df[self.df['Class_Abnormality'].isna()])
            else:
                count = len(self.df[self.df['Class_Abnormality'] == cls_name])
            weights[cls_idx] = total / (3 * max(count, 1))
        
        return torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32)


class DMIDMultiTaskDataset(Dataset):
    """
    Multi-task dataset for Classification + Detection + Segmentation
    Loads TIFF images, masks, PLA, and metadata
    """
    
    def __init__(self,
                 tiff_dir: str,
                 masks_dir: str,
                 pla_dir: str,
                 metadata_loader: MetadataLoader,
                 split: str = 'train',
                 image_ids: Optional[list] = None,
                 augmentation: bool = True):
        """
        Initialize dataset
        
        Args:
            tiff_dir: Directory containing TIFF images
            masks_dir: Directory containing mask .npy files
            pla_dir: Directory containing PLA .npy files
            metadata_loader: MetadataLoader instance
            split: 'train', 'val', or 'test'
            image_ids: List of specific image IDs to use
            augmentation: Whether to apply augmentation
        """
        self.tiff_dir = Path(tiff_dir)
        self.masks_dir = Path(masks_dir)
        self.pla_dir = Path(pla_dir)
        self.metadata_loader = metadata_loader
        self.split = split
        self.augmentation = augmentation
        
        # Find all TIFF files
        self.tiff_files = sorted(list(self.tiff_dir.glob('*.tif')))
        
        if image_ids is not None:
            # Filter to specific image IDs
            self.tiff_files = [f for f in self.tiff_files if f.stem in image_ids]
        
        # Setup augmentation pipeline
        self._setup_augmentation()
    
    def _setup_augmentation(self):
        """Setup augmentation pipeline"""
        if self.augmentation and self.split == 'train':
            self.transform = A.Compose([
                A.Rotate(limit=15, p=0.5),
                A.Zoom(scale=(0.8, 1.2), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        """Return number of images"""
        return len(self.tiff_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load single sample
        
        Returns:
            Dict with:
                - image_id: str
                - image: Tensor (1, 512, 512)
                - classification_label: int (0/1/2)
                - has_anomaly: bool
                - bbox: Tensor (4,) normalized or None
                - mask: Tensor (512, 512) or None
                - pla: Tensor (512, 512) or None
                - tissue_type: str
                - view: str
        """
        # Load image
        tiff_path = self.tiff_files[idx]
        image_id = tiff_path.stem
        
        # Read TIFF
        image = cv2.imread(str(tiff_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((512, 512), dtype=np.uint8)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Get metadata
        metadata = self.metadata_loader.get_image_metadata(image_id)
        
        # Load mask if exists
        mask = None
        mask_path = self.masks_dir / f"{image_id}.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32)
        
        # Load PLA if exists
        pla = None
        pla_path = self.pla_dir / f"{image_id}.npy"
        if pla_path.exists():
            pla = np.load(pla_path).astype(np.float32)
        
        # Convert bbox from metadata format to normalized coordinates
        bbox_normalized = None
        if metadata['bbox'] is not None:
            x, y, radius = metadata['bbox']
            # Normalize to [0, 1]
            x_norm = x / 512.0
            y_norm = y / 512.0
            r_norm = radius / 512.0
            
            # Convert to [x_min, y_min, x_max, y_max] format
            x_min = max(0, x_norm - r_norm)
            y_min = max(0, y_norm - r_norm)
            x_max = min(1, x_norm + r_norm)
            y_max = min(1, y_norm + r_norm)
            
            bbox_normalized = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        
        # Apply augmentation
        if self.augmentation and self.split == 'train':
            # Augmentation with bbox support
            bboxes = [bbox_normalized.tolist()] if bbox_normalized is not None else []
            augmented = self.transform(
                image=image,
                mask=mask if mask is not None else np.zeros_like(image),
                bboxes=bboxes,
                class_labels=[0] if bboxes else []
            )
            image = augmented['image']
            mask = augmented.get('mask', mask)
            
            if augmented.get('bboxes'):
                bbox_normalized = torch.tensor(augmented['bboxes'][0], dtype=torch.float32)
        else:
            # Standard normalization
            image = torch.from_numpy(image[np.newaxis, ...]).float()  # Add channel
            if mask is not None:
                mask = torch.from_numpy(mask).float()
            if pla is not None:
                pla = torch.from_numpy(pla).float()
        
        # Ensure image is 3D
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        result = {
            'image_id': image_id,
            'image': image,
            'classification_label': metadata['classification_label'],
            'has_anomaly': metadata['has_anomaly'],
            'bbox': bbox_normalized,
            'mask': mask,
            'pla': pla,
            'tissue_type': metadata['tissue_type'],
            'view': metadata['view']
        }
        
        return result


def create_data_loaders(config: Dict,
                       metadata_loader: MetadataLoader,
                       batch_size: Optional[int] = None) -> Tuple[torch.utils.data.DataLoader, 
                                                                   torch.utils.data.DataLoader,
                                                                   torch.utils.data.DataLoader]:
    """
    Create train/val/test data loaders
    
    Args:
        config: Configuration dict
        metadata_loader: MetadataLoader instance
        batch_size: Override config batch size
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get paths from config
    tiff_dir = config['data']['tiff_dir']
    masks_dir = config['data']['masks_dir']
    pla_dir = config['data']['pla_dir']
    
    batch_size = batch_size or config['classification']['batch_size']
    
    # Create full dataset first to get all image IDs
    full_dataset = DMIDMultiTaskDataset(
        tiff_dir, masks_dir, pla_dir, metadata_loader,
        split='all', augmentation=False
    )
    
    # Get image IDs
    image_ids = [dataset_item['image_id'] for dataset_item in full_dataset]
    
    # Random split
    np.random.seed(config['training']['seed'])
    indices = np.random.permutation(len(image_ids))
    
    train_idx = indices[:int(0.6 * len(indices))]
    val_idx = indices[int(0.6 * len(indices)):int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]
    
    train_ids = [image_ids[i] for i in train_idx]
    val_ids = [image_ids[i] for i in val_idx]
    test_ids = [image_ids[i] for i in test_idx]
    
    # Create datasets
    train_dataset = DMIDMultiTaskDataset(
        tiff_dir, masks_dir, pla_dir, metadata_loader,
        split='train', image_ids=train_ids, augmentation=True
    )
    
    val_dataset = DMIDMultiTaskDataset(
        tiff_dir, masks_dir, pla_dir, metadata_loader,
        split='val', image_ids=val_ids, augmentation=False
    )
    
    test_dataset = DMIDMultiTaskDataset(
        tiff_dir, masks_dir, pla_dir, metadata_loader,
        split='test', image_ids=test_ids, augmentation=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader
