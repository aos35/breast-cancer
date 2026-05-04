# ÍNDICE - Componentes Técnicos Bloque2_CNN

## 📋 Tabla de Contenidos Rápida

| Componente | Archivo | Descripción | Estado |
|-----------|---------|-------------|--------|
| **README** | README.md | Visión general del proyecto | ✅ |
| **Configuración** | config.yaml | Parámetros centralizados | ✅ |
| **Dependencias** | requirements.txt | Librerías Python | ✅ |
| **Data Loader** | src/data_loader.py | Carga de datos 3 tareas | 🔄 |
| **Modelos** | src/models.py | Arquitecturas CNN | 🔄 |
| **Utilidades** | src/utils.py | Loss, métricas | 🔄 |
| **Training** | src/train.py | Loops de entrenamiento | ❌ |
| **Evaluación** | src/evaluate.py | Métricas finales | ❌ |

---

## 🧩 Componentes por Módulo

### 1️⃣ **Módulo de Datos** (`src/data_loader.py`)

#### Clases Principales

**`MetadataLoader`**
```python
class MetadataLoader:
    def __init__(self, metadata_path: str):
        """Carga metadata.xlsx con etiquetas de clasificación"""
        # Lectura pandas + procesamiento
        
    def get_image_label(self, image_id: str) -> Dict:
        """Retorna: {'class': 'B'|'M'|'N'|None,
                     'bbox': (x, y, radius),
                     'type': 'CIRC'|'CALC'|...}"""
```

**`DMIDMultiTaskDataset`** (clase principal)
```python
class DMIDMultiTaskDataset(Dataset):
    """Dataset para pipeline 3 tareas (Clasificación + Detección + Segmentación)"""
    
    def __init__(self,
                 tiff_dir: str,
                 masks_dir: str,
                 pla_dir: str,
                 metadata_loader: MetadataLoader,
                 split: str = 'train',  # 'train', 'val', 'test'
                 augmentation: bool = True):
        """
        Args:
            tiff_dir: Ruta a imágenes DMID_PNG/*.tif
            masks_dir: Ruta a máscaras *.npy
            pla_dir: Ruta a PLA *.npy
            metadata_loader: Instancia MetadataLoader
            split: Train/val/test (60/20/20)
        """
    
    def __len__(self) -> int:
        """511 imágenes TIFF totales"""
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Retorna diccionario:
        {
            'image_id': 'IMG001',
            'image': Tensor (1, 512, 512) - imagen normalizada
            'classification_label': 0|1|2  # 0=Benign, 1=Malignant, 2=Negative
            'has_anomaly': bool  # si tiene anomalía clasificada
            'bbox': Tensor (4,) si aplica [x, y, w, h]  # coordenadas normalizadas
            'mask': Tensor (512, 512) si aplica  # ground truth segmentación
            'pla': Tensor (512, 512) si aplica  # anotación pixel-level
            'tissue_type': str  # F|G|D (Fatty, Fatty-glandular, Dense)
            'view': str  # CCLT, CCRT, MLOLT, MLORT
        }
        """
```

#### Funciones Auxiliares

```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normaliza a [0, 1] usando min-max"""

def extract_bbox_from_metadata(metadata: Dict) -> Tensor:
    """Convierte (X, Y, radius) de metadata → bbox normalizado (0-1)"""

def apply_augmentation(image: np.ndarray,
                       mask: np.ndarray,
                       augmentation_params: Dict) -> Tuple:
    """Rotación, zoom, intensidad, elastic deformations (usando Albumentations)"""

def create_train_val_test_split(dataset: DMIDMultiTaskDataset,
                                 split_ratio: Tuple = (0.6, 0.2, 0.2),
                                 stratify_by: str = 'classification_label') -> Dict:
    """Divide datos manteniendo proporciones de clases"""
```

---

### 2️⃣ **Módulo de Modelos** (`src/models.py`)

#### TAREA 1: Clasificación

**`DenseNet121Classifier`**
```python
class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        DenseNet121 pre-entrenado en ImageNet
        Input:  (B, 1, 512, 512) - mamografía
        Output: (B, 3) - logits para [Benign, Malignant, Negative]
        """
        self.backbone = torchvision.models.densenet121(pretrained=pretrained)
        # Adaptar entrada a 1 canal (mamografía en escala gris)
        self.backbone.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Returns (B, 3) logits"""
```

---

#### TAREA 2: Detección

**`get_mask_rcnn`** (factory function)
```python
def get_mask_rcnn(num_classes: int = 2,
                  pretrained: bool = True) -> torchvision.models.detection.maskrcnn_resnet50_fpn:
    """
    Mask R-CNN pre-entrenado en COCO
    Input:  (B, 1, 512, 512) o (B, 3, 512, 512)
    Output: Diccionario con:
            - 'boxes': (N, 4) - bounding boxes [x1,y1,x2,y2] normalizados
            - 'scores': (N,) - confidence scores
            - 'labels': (N,) - class predictions
            - 'masks': (N, 512, 512) - instance masks (si aplica)
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained,
        num_classes=num_classes
    )
    # Adaptar para input 1 canal
    model.backbone.body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
    return model
```

---

#### TAREA 3: Segmentación

**`UNet`**
```python
class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        U-Net para segmentación médica
        Input:  (B, 1, 512, 512)
        Output: (B, 1, 512, 512) - probabilidades [0, 1]
        
        Arquitectura:
        - 4 niveles encoder (64 → 512 canales)
        - Bottleneck (1024 canales)
        - 4 niveles decoder con skip connections
        """
        self.encoder = nn.ModuleList([
            self._conv_block(1, 64),      # 512 → 256
            self._conv_block(64, 128),    # 256 → 128
            self._conv_block(128, 256),   # 128 → 64
            self._conv_block(256, 512),   # 64 → 32
        ])
        
        self.bottleneck = self._conv_block(512, 1024)  # 32 → 32
        
        self.decoder = nn.ModuleList([
            self._upconv_block(1024, 512),  # 32 → 64
            self._upconv_block(512, 256),   # 64 → 128
            self._upconv_block(256, 128),   # 128 → 256
            self._upconv_block(128, 64),    # 256 → 512
        ])
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        """Returns (B, 1, 512, 512) probabilidades"""
```

**`DeepLabV3Plus`**
```python
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        DeepLabV3+ para segmentación médica
        Input:  (B, 1, 512, 512)
        Output: (B, 1, 512, 512) - probabilidades [0, 1]
        
        Componentes:
        - Backbone: ResNet50 + atrous convolutions
        - ASPP: Atrous Spatial Pyramid Pooling (captura multi-escala)
        - Decoder: Low-level + high-level features
        """
        self.backbone = ResNet50ASPP(in_channels)
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = DecoderModule(low_dim=256, high_dim=256, out_channels=out_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        """Returns (B, 1, 512, 512) probabilidades"""
```

---

### 3️⃣ **Módulo de Utilidades** (`src/utils.py`)

#### Loss Functions

**Clasificación**
```python
class ClassificationLoss(nn.Module):
    def __init__(self, class_weights: Tensor = None):
        """
        CrossEntropyLoss con pesos opcionales para desbalance de clases
        Pesos: Inverse class frequency
        {Benign: 0.51, Malignant: 0.81, Negative: 0.65}
        """
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: (B, 3) logits
            targets: (B,) class indices 0/1/2
        Returns: scalar loss
        """
```

**Detección**
```python
class DetectionLoss(nn.Module):
    def __init__(self):
        """
        Multi-task loss: SmoothL1 (bbox) + CrossEntropy (classes)
        Usualmente integrada en Mask R-CNN
        """
```

**Segmentación**
```python
class DiceLoss(nn.Module):
    def __init__(self):
        """
        Dice Loss: 1 - 2*(TP)/(2*TP+FP+FN)
        Excelente para desbalance de clases (background >> foreground)
        """
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: (B, 1, H, W) probabilidades [0, 1]
            targets: (B, 1, H, W) ground truth binario [0, 1]
        Returns: scalar loss
        """

class CombinedLoss(nn.Module):
    def __init__(self, lambda_dice: float = 0.5, lambda_bce: float = 0.5):
        """
        Combinación: 0.5*DiceLoss + 0.5*BCELoss
        Da mejor convergencia que Dice sólo
        """
```

#### Métricas

**Clasificación**
```python
def classification_metrics(predictions: Tensor, targets: Tensor) -> Dict:
    """
    Calcula:
    - Accuracy global
    - Precision/Recall/F1 por clase
    - AUC-ROC global y por clase
    - Confusion matrix
    
    Returns: Dict con todas métricas
    """
```

**Detección**
```python
def detection_metrics(predicted_boxes: Tensor,
                     predicted_scores: Tensor,
                     gt_boxes: Tensor) -> Dict:
    """
    Calcula:
    - Precision/Recall a diferentes IoU thresholds
    - mAP (mean Average Precision)
    - IoU distribution
    
    Returns: Dict con métricas mAP
    """
```

**Segmentación**
```python
def segmentation_metrics(predictions: Tensor, targets: Tensor) -> Dict:
    """
    Calcula (sobre píxeles):
    - Dice Score: 2*TP/(2*TP+FP+FN)
    - Jaccard/IoU: TP/(TP+FP+FN)
    - Accuracy: (TP+TN)/(Total)
    - Sensitivity: TP/(TP+FN)
    - Specificity: TN/(TN+FP)
    - Hausdorff Distance: máxima distancia entre contornos
    
    Returns: Dict con todas métricas
    """
```

---

### 4️⃣ **Training Script** (`src/train.py`) [TODO]

```python
def train_classification(config: Dict, device: str):
    """Entrena DenseNet121 para clasificación 3-clases"""

def train_detection(config: Dict, device: str):
    """Entrena Mask R-CNN para detección de bboxes"""

def train_segmentation(config: Dict, device: str, model_type: str = 'unet'):
    """Entrena U-Net o DeepLabV3+ para segmentación"""

def train_multitask_pipeline(config: Dict, device: str):
    """Orquesta entrenamiento secuencial de 3 tareas"""
```

---

### 5️⃣ **Evaluación Script** (`src/evaluate.py`) [TODO]

```python
def evaluate_classification(model, test_loader, device: str) -> Dict:
    """Retorna métricas de clasificación finales"""

def evaluate_detection(model, test_loader, device: str) -> Dict:
    """Retorna métricas de detección (mAP, IoU)"""

def evaluate_segmentation(model, test_loader, device: str) -> Dict:
    """Retorna Dice, IoU, Hausdorff por imagen"""

def compare_segmentation_models(config: Dict, device: str) -> Dict:
    """Compara U-Net vs DeepLabV3+ en test set"""
```

---

## 📊 Configuración (config.yaml)

Ver [config.yaml](config.yaml) para parámetros completos.

**Estructura principal:**
```yaml
project:
  name: "Bloque2_CNN"
  dataset: "DMID"
  
data:
  tiff_dir: "../DMID_PNG/"
  masks_dir: "../DMID_PNG/masks/"
  pla_dir: "../DMID_PNG/pla/"
  metadata_path: "../Metadata.xlsx"
  
classification:
  model: "densenet121"
  num_classes: 3
  batch_size: 32
  epochs: 200
  learning_rate: 0.0001
  
detection:
  model: "mask_rcnn"
  batch_size: 8
  epochs: 150
  learning_rate: 0.0001
  
segmentation:
  models: ["unet", "deeplabv3"]
  batch_size: 16
  epochs: 200
  learning_rate: 0.001
```

---

## 🔄 Flujo de Datos

```
Entrada: DMID_PNG/IMG*.tif
    ↓
[Normalización] → [0, 1]
    ↓
[Augmentation] → Rotate, Zoom, Intensity, Elastic
    ↓
TAREA 1: Clasificación
├─ Input: (1, 512, 512)
├─ Model: DenseNet121
├─ Output: [0.2, 0.7, 0.1] → Clase 1 (Malignant)
└─ Utilizado para: siguiente tarea
    ↓
TAREA 2: Detección (si Maligno o Benigno)
├─ Input: (1, 512, 512)
├─ Model: Mask R-CNN
├─ Output: bbox (x, y, w, h) + score
└─ Utilizado para: crop a ROI
    ↓
TAREA 3: Segmentación (en ROI)
├─ Input: (1, 256, 256) [cropped ROI]
├─ Model: U-Net o DeepLabV3+
├─ Output: (256, 256) mask probabilidades
└─ Post-procesamiento: Thresholding (0.5) → binario
    ↓
Salida Final:
├─ Classification: Benign (clase 0)
├─ Detection: bbox normalizado
├─ Segmentation: Mask binaria
└─ PLA: Anotación pixel-level
```

---

## 📈 Matriz de Métricas

| Tarea | Métrica Primaria | Métrica Secundaria | Interpretación |
|-------|-----------------|------------------|-----------------|
| **Clasificación** | AUC-ROC ≥ 0.95 | Accuracy ≥ 0.90 | ✓ Good generalization |
| **Detección** | mAP @ IoU=0.5 ≥ 0.90 | mAP @ IoU=0.75 ≥ 0.85 | ✓ Precise localization |
| **Segmentación** | Dice ≥ 0.85 | IoU ≥ 0.75 | ✓ Good delineation |

---

## 🎯 Checkpoints y Guardado

```
models/
├── classification/
│   ├── best_model.pth          # Mejor checkpoint (validación)
│   ├── last_model.pth          # Último checkpoint
│   └── metrics.json            # Métricas finales
├── detection/
│   ├── best_model.pth
│   ├── last_model.pth
│   └── metrics.json
└── segmentation/
    ├── unet_best.pth
    ├── deeplabv3_best.pth
    └── comparison.json
```

---

## 📝 Referencia Rápida

**Para usar cada módulo:**

```python
# 1. Data
from src.data_loader import MetadataLoader, DMIDMultiTaskDataset
loader = MetadataLoader('../Metadata.xlsx')
dataset = DMIDMultiTaskDataset(tiff_dir, masks_dir, pla_dir, loader)

# 2. Models
from src.models import DenseNet121Classifier, get_mask_rcnn, UNet, DeepLabV3Plus
clf = DenseNet121Classifier(num_classes=3)
det = get_mask_rcnn(num_classes=2)
seg = UNet(in_channels=1, out_channels=1)

# 3. Losses & Metrics
from src.utils import ClassificationLoss, DiceLoss, classification_metrics
loss_fn = ClassificationLoss(class_weights=...)
metrics = classification_metrics(preds, targets)

# 4. Training
from src.train import train_classification, train_detection, train_segmentation
train_classification(config='config.yaml', device='cuda')
```

---

**Última actualización**: Mayo 2026  
**Status**: Documentación completa - Implementación en progreso
