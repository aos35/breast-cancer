"""
Data loading module for DMID_PNG segmentation dataset.

Carga imágenes TIFF, máscaras (Masks) y anotaciones pixel-level (PLA)
desde la carpeta DMID_PNG.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class DMIDSegmentationDataset(Dataset):
    """
    Dataset para segmentación de lesiones mamarias.
    
    Carga TIFF, Masks, y PLA manteniendo alineación por nombre de archivo.
    """
    
    def __init__(
        self,
        tiff_dir: str,
        masks_dir: str,
        pla_dir: str,
        transform=None,
        target_transform=None,
        include_pla=True,
        mode='masks'  # 'masks' para salida binaria, 'pla' para multi-clase
    ):
        """
        Args:
            tiff_dir: Directorio con imágenes TIFF
            masks_dir: Directorio con máscaras binarias
            pla_dir: Directorio con anotaciones pixel-level
            transform: Transformaciones para imágenes
            target_transform: Transformaciones para targets (máscaras)
            include_pla: Si incluir PLA (requiere cálculo adicional)
            mode: 'masks' para salida binaria, 'pla' para multi-clase
        """
        self.tiff_dir = Path(tiff_dir)
        self.masks_dir = Path(masks_dir)
        self.pla_dir = Path(pla_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.include_pla = include_pla
        self.mode = mode
        
        # Buscar imágenes TIFF
        self.tiff_files = sorted(self.tiff_dir.glob('*.png'))
        
        # Buscar máscaras (subconjunto de TIFF)
        self.mask_files = sorted(self.masks_dir.glob('*.png'))
        
        # Crear mapeo de filename -> mask
        self.mask_dict = {f.name: f for f in self.mask_files}
        
        # Si está disponible, mapear PLA también
        if include_pla and pla_dir and Path(pla_dir).exists():
            self.pla_files = sorted(Path(pla_dir).glob('*.png'))
            self.pla_dict = {f.name: f for f in self.pla_files}
        else:
            self.pla_dict = {}
        
        print(f"✓ Dataset inicializado")
        print(f"  - TIFF: {len(self.tiff_files)} imágenes")
        print(f"  - Masks: {len(self.mask_files)} máscaras")
        print(f"  - PLA: {len(self.pla_dict)} anotaciones pixel-level")
    
    def __len__(self):
        return len(self.tiff_files)
    
    def __getitem__(self, idx):
        """
        Retorna: (imagen, máscara, [pla])
        """
        # Cargar imagen TIFF
        tiff_path = self.tiff_files[idx]
        image = cv2.imread(str(tiff_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"No se pudo leer: {tiff_path}")
        
        # Normalizar imagen a [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Aplicar transformaciones a imagen
        if self.transform:
            image = self.transform(image)
        
        # Cargar máscara correspondiente (si existe)
        filename = tiff_path.name
        if filename in self.mask_dict:
            mask_path = self.mask_dict[filename]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                # Si no se puede leer, crear máscara vacía
                mask = np.zeros_like(image, dtype=np.float32)
            else:
                # Binarizar máscara: 0 o 1
                mask = (mask > 127).astype(np.float32)
        else:
            # Si no hay máscara, crear una vacía
            mask = np.zeros(image.shape, dtype=np.float32)
        
        # Aplicar transformaciones a máscara
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # Cargar PLA (anotación pixel-level) si está disponible
        if filename in self.pla_dict and self.include_pla:
            pla_path = self.pla_dict[filename]
            pla = cv2.imread(str(pla_path), cv2.IMREAD_GRAYSCALE)
            
            if pla is None:
                pla = np.zeros_like(image, dtype=np.float32)
            else:
                pla = pla.astype(np.float32) / 255.0
            
            # Retornar (imagen, máscara, pla)
            return {
                'image': image,
                'mask': mask,
                'pla': pla,
                'filename': filename
            }
        else:
            # Retornar solo (imagen, máscara)
            return {
                'image': image,
                'mask': mask,
                'filename': filename
            }


class SegmentationDataModule:
    """
    Módulo para manejar datos de segmentación.
    Divide en train/val/test y crea DataLoaders.
    """
    
    def __init__(
        self,
        tiff_dir: str,
        masks_dir: str,
        pla_dir: str,
        batch_size: int = 16,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
    ):
        """
        Args:
            batch_size: Tamaño de batch
            train_split: Fracción para entrenamiento
            val_split: Fracción para validación
            test_split: Fracción para test
            num_workers: Número de workers para DataLoader
            seed: Random seed para reproducibilidad
        """
        np.random.seed(seed)
        
        self.tiff_dir = tiff_dir
        self.masks_dir = masks_dir
        self.pla_dir = pla_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Obtener lista de imágenes anotadas (que tienen máscara)
        mask_files = set(f.name for f in Path(masks_dir).glob('*.png'))
        tiff_files = [f.name for f in Path(tiff_dir).glob('*.png') 
                      if f.name in mask_files]
        
        # Dividir índices
        n = len(tiff_files)
        train_size = int(n * train_split)
        val_size = int(n * val_split)
        
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        self.train_files = [tiff_files[i] for i in train_idx]
        self.val_files = [tiff_files[i] for i in val_idx]
        self.test_files = [tiff_files[i] for i in test_idx]
        
        print(f"✓ Data split:")
        print(f"  - Train: {len(self.train_files)} ({100*train_split:.1f}%)")
        print(f"  - Val: {len(self.val_files)} ({100*val_split:.1f}%)")
        print(f"  - Test: {len(self.test_files)} ({100*test_split:.1f}%)")
    
    def get_train_loader(self):
        """Retorna DataLoader de entrenamiento"""
        dataset = DMIDSegmentationDataset(
            self.tiff_dir, self.masks_dir, self.pla_dir,
            include_pla=True
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True
        )
        return loader
    
    def get_val_loader(self):
        """Retorna DataLoader de validación"""
        dataset = DMIDSegmentationDataset(
            self.tiff_dir, self.masks_dir, self.pla_dir,
            include_pla=True
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        return loader
    
    def get_test_loader(self):
        """Retorna DataLoader de test"""
        dataset = DMIDSegmentationDataset(
            self.tiff_dir, self.masks_dir, self.pla_dir,
            include_pla=True
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        return loader


# Test de uso
if __name__ == "__main__":
    # Rutas de ejemplo
    tiff_dir = "../../data/raw/DMID_PNG/512x512/TIFF"
    masks_dir = "../../data/raw/DMID_PNG/512x512/Masks"
    pla_dir = "../../data/raw/DMID_PNG/512x512/PLA_PNG"
    
    # Crear dataset
    dataset = DMIDSegmentationDataset(tiff_dir, masks_dir, pla_dir)
    print(f"Dataset creado: {len(dataset)} samples")
    
    # Cargar un sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  - Image shape: {sample['image'].shape}")
    print(f"  - Mask shape: {sample['mask'].shape}")
    if 'pla' in sample:
        print(f"  - PLA shape: {sample['pla'].shape}")
    print(f"  - Filename: {sample['filename']}")
