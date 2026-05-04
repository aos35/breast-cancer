# Bloque 2: CNN para Detección Completa de Cáncer de Mama
## Pipeline: Clasificación → Detección → Segmentación

**Proyecto**: Detección Integral de Cáncer de Mama usando Deep Learning  
**Curso**: Redes Neuronales (3º Año)  
**Dataset**: DMID (Digital Mammography Image Dataset)  
**Metadata**: Metadata.xlsx con etiquetas de clasificación

---

## 🎯 Objetivos del Proyecto

Este proyecto implementa un **pipeline completo de 3 tareas** para análisis automático de mamografías:

### Tarea 1: Clasificación (Multi-clase)
- **Input**: Imagen TIFF (mamografía completa)
- **Output**: Probabilidades para 3 clases:
  - `Benign` (Benigno) - 39.10% del dataset
  - `Malignant` (Maligno) - 24.42% del dataset
  - `Negative` (Sin tumor) - 30.91% del dataset
- **Métrica Clave**: AUC-ROC, Accuracy, Precision, Recall, F1
- **Arquitectura**: DenseNet121 (7M parámetros, 94-96% AUC esperado)

### Tarea 2: Detección (Localización)
- **Input**: Imagen TIFF + clasificación (si aplica)
- **Output**: Bounding boxes de anomalías (si existen)
  - Coordenadas X, Y disponibles en metadata.xlsx
  - Radio en píxeles disponible
- **Métrica Clave**: mAP (mean Average Precision), IoU
- **Arquitectura**: Mask R-CNN (41M parámetros, 92-95% mAP esperado)

### Tarea 3: Segmentación (Delineación)
- **Input**: ROI detectada (bounding box)
- **Output**: 
  - Máscara binaria (Mask)
  - Anotación a nivel de píxel (PLA)
- **Métrica Clave**: Dice Score, IoU, Accuracy
- **Arquitecturas**: 
  - U-Net (1.9M parámetros)
  - DeepLabV3+ (39.5M parámetros)
- **Ground Truth**: 269 imágenes anotadas con Masks + PLA

---

## 📊 Dataset

### Fuente: DMID (Digital Mammography Image Dataset)

**Estructura de Datos:**

| Componente | Cantidad | Descripción |
|-----------|----------|-------------|
| Imágenes TIFF | 511 | Mamografías originales (512×512 píxeles) |
| Máscaras (Masks) | 269 | Ground truth segmentación binaria |
| PLA (Pixel-Level Annotation) | 269 | Anotación detallada por píxel |
| Metadata | metadata.xlsx | Etiquetas clasificación + coordenadas bboxes |

### Metadata.xlsx (Clasificación)

**Columnas Clave:**
- `Image_Reference`: ID único (IMG001-IMG999)
- `Class_Abnormality`: Etiqueta principal (B/M/N/NaN)
  - `B` = Benign (Benigno)
  - `M` = Malignant (Maligno)
  - `N` = No Defined (Sin definir)
  - `NaN` = Normal/No Tumor (Sin tumor)
- `Abnormality_Type`: Tipo de anomalía (CIRC, CALC, SPIC, MISC, NORM)
- `X_Coordinate` / `Y_Coordinate`: Centro de la anomalía
- `Radius_pixels`: Radio aproximado de la región

**Distribución de Clases:**
```
Benign (B):        253 (39.10%)
Malignant (M):     158 (24.42%)
No Defined (N):     36 (5.56%)
Normal (NaN):      200 (30.91%)
─────────────────────────────
Total:             647 instancias
Imágenes únicas:   510
```

**Notas Importantes:**
- ⚠️ **Desbalance de clases**: Benign (39%) > Normal (31%) > Malignant (24%)
- ⚠️ **Valores faltantes**: X/Y/Radius son NaN para imágenes Normal (esperado)
- ✓ Coordenadas disponibles para ~69% del dataset

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRADA: Imagen TIFF                     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
      ┌───────────────────┐
      │  TAREA 1:         │
      │  CLASIFICACIÓN    │  DenseNet121
      │  3 clases         │  Input: (1, 512, 512)
      │  (B/M/N)          │  Output: (3,) logits
      └───────────┬───────┘
                  │
           ┌──────┴──────┐
           │ Resultado:  │
           │ • Benign    │
           │ • Malignant │
           │ • Negative  │
           └──────┬──────┘
                  │
                  ▼
        ┌──────────────────────┐
        │ ¿Anomalía detectada? │
        └──────┬────────────┬──┘
              NO            │ SÍ
              │             ▼
              │      ┌──────────────────┐
              │      │  TAREA 2:        │
              │      │  DETECCIÓN       │  Mask R-CNN
              │      │  Bounding Boxes  │  Input: (1, 512, 512)
              │      │  (X,Y,W,H,conf)  │  Output: boxes, scores
              │      └────────┬─────────┘
              │               │
              │               ▼
              │      ┌──────────────────┐
              │      │  Extraer ROI     │
              │      │  (crop to bbox)  │
              │      └────────┬─────────┘
              │               │
              │               ▼
              │      ┌──────────────────┐
              │      │  TAREA 3:        │
              │      │  SEGMENTACIÓN    │  U-Net / DeepLabV3+
              │      │  Mask + PLA      │  Input: ROI (1, 256, 256)
              │      │  (pixel-level)   │  Output: (1, 256, 256) mask
              │      └────────┬─────────┘
              │               │
              └───────────────┴─────────┐
                                        │
                                        ▼
                        ┌──────────────────────────┐
                        │   SALIDA COMPLETA:       │
                        │ • Clase (B/M/N)         │
                        │ • Bbox (si aplica)      │
                        │ • Mask (si aplica)      │
                        │ • PLA (si aplica)       │
                        └──────────────────────────┘
```

---

## 🛠️ Tecnologías y Dependencias

**Framework Deep Learning:**
- PyTorch 2.0.1+ con CUDA 11.8+
- TorchVision 0.15+ (modelos pre-entrenados)

**Librerías Médicas:**
- PyDICOM (lectura imágenes médicas)
- OpenCV (procesamiento imágenes)
- Pillow (I/O imágenes)
- Albumentations (data augmentation)

**Modelos y Métricas:**
- segmentation-models-pytorch 0.3.0 (segmentación)
- torchmetrics 0.11.4 (métricas de evaluación)
- scikit-learn (métricas clasificación)

**Utilidades:**
- PyYAML (configuración)
- Pandas (manejo datos)
- NumPy (arrays numéricos)
- Matplotlib/Seaborn (visualización)
- Jupyter (notebooks)
- openpyxl (lectura metadata.xlsx)

Ver [requirements.txt](requirements.txt) para lista completa.

---

## 📋 Estructura de Directorios

```
Bloque2_CNN/
├── README.md                           # Este archivo
├── INDICE.md                           # Índice de componentes
├── config.yaml                         # Configuración centralizada
├── requirements.txt                    # Dependencias Python
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Carga de datos (3 tareas)
│   ├── models.py                      # Arquitecturas CNN
│   ├── utils.py                       # Utilidades, loss, métricas
│   ├── train.py                       # Training script
│   └── evaluate.py                    # Evaluación
│
├── notebooks/
│   ├── 00_exploracion.ipynb           # EDA del dataset
│   ├── 01_preprocesamiento.ipynb      # Data pipeline
│   ├── 02_entrenamiento.ipynb         # Training multi-tarea
│   └── 03_evaluacion.ipynb            # Evaluación y análisis
│
└── models/                             # Checkpoints guardados
    ├── classification/
    ├── detection/
    └── segmentation/
```

---

## 🚀 Quick Start

### 1. Instalación de Dependencias
```bash
cd Bloque2_CNN
pip install -r requirements.txt
```

### 2. Preparación de Datos
```python
# Ver notebook: 00_exploracion.ipynb
# Verifica estructura de DMID_PNG, Metadata.xlsx, Masks, PLA
```

### 3. Entrenamiento Clasificación
```python
from src.train import train_classification
from src.models import DenseNet121Classifier

model = DenseNet121Classifier(num_classes=3)
train_classification(config='config.yaml', model=model)
```

### 4. Entrenamiento Detección
```python
from src.train import train_detection
from src.models import get_mask_rcnn

model = get_mask_rcnn(num_classes=2)  # foreground + background
train_detection(config='config.yaml', model=model)
```

### 5. Entrenamiento Segmentación
```python
from src.train import train_segmentation
from src.models import UNet, DeepLabV3Plus

model_unet = UNet(in_channels=1, out_channels=1)
train_segmentation(config='config.yaml', model=model_unet, task='unet')
```

### 6. Evaluación Completa
```bash
python -m src.evaluate --config config.yaml --all-tasks
```

Ver [QUICK_START.md](QUICK_START.md) para guía detallada.

---

## 📊 Arquitecturas Seleccionadas

### Tarea 1: Clasificación

**Modelo: DenseNet121**
```
DenseNet121
├─ Dense Block 1-4 (4×3=12 capas)
├─ Classification Head
│  ├─ Global Average Pool
│  ├─ Fully Connected (1024 → 512)
│  ├─ ReLU
│  ├─ Dropout (0.3)
│  └─ Output (512 → 3) [Benign, Malignant, Negative]
```

**Justificación:**
- ✓ Solo 7M parámetros (eficiente)
- ✓ Excelente gradient flow (conexiones densas)
- ✓ 94-96% AUC reportado en literature
- ✓ Ideal para datasets pequeños con transfer learning
- ✓ Fast inference (~80ms GPU)

**Parámetros Entrenamiento:**
- Optimizer: Adam (lr=0.0001)
- Batch size: 32
- Epochs: 200 (early stopping, patience=20)
- Loss: CrossEntropy con class weights balanceados
- Augmentation: Rotación ±15°, Zoom 0.8-1.2x, Intensidad ±10%

---

### Tarea 2: Detección

**Modelo: Mask R-CNN**
```
Mask R-CNN (ResNet50-FPN)
├─ Backbone: ResNet50 + FPN (Feature Pyramid Network)
├─ RPN (Region Proposal Network)
├─ Head Detección
│  ├─ Box predictor
│  └─ Class predictor
└─ Head Segmentación (no usado, solo bbox)
```

**Justificación:**
- ✓ State-of-the-art para detección
- ✓ 92-95% mAP reportado
- ✓ Genera bounding boxes automáticamente
- ✓ Pre-entrenado en COCO (transfer learning)
- ✓ Maneja múltiples instancias

**Parámetros Entrenamiento:**
- Optimizer: SGD (lr=0.0001, momentum=0.9)
- Batch size: 8 (limita memoria)
- Epochs: 150 (early stopping, patience=20)
- Loss: Multi-task (bbox + classification)
- Augmentation: Horizontal flip, Rotación ±5°

---

### Tarea 3: Segmentación

**Modelo A: U-Net**
```
U-Net
├─ Encoder (4 bloques convolucionales)
│  ├─ Conv 64 → MaxPool
│  ├─ Conv 128 → MaxPool
│  ├─ Conv 256 → MaxPool
│  └─ Conv 512 → MaxPool
├─ Bottleneck (1024 canales)
└─ Decoder (4 bloques transpuestos)
   ├─ Deconv 512 + Skip Concat
   ├─ Deconv 256 + Skip Concat
   ├─ Deconv 128 + Skip Concat
   ├─ Deconv 64 + Skip Concat
   └─ Output (1 canal) → Sigmoid
```

**Especificaciones:**
- Parámetros: 1.9M
- Input: (1, 512, 512)
- Output: (1, 512, 512) [probabilidades 0-1]
- Ventaja: Excelente con datos pequeños

**Modelo B: DeepLabV3+**
```
DeepLabV3+
├─ Backbone: ResNet50 + Atrous Convolutions
├─ Atrous Spatial Pyramid Pooling (ASPP)
├─ Decoder
│  ├─ Low-level features (4x)
│  └─ High-level features (upsampled)
└─ Output (1 canal) → Sigmoid
```

**Especificaciones:**
- Parámetros: 39.5M
- Input: (1, 512, 512)
- Output: (1, 512, 512)
- Ventaja: Captura contexto multi-escala

**Parámetros Entrenamiento:**
- Optimizer: Adam (lr=0.001)
- Batch size: 16
- Epochs: 200 (early stopping, patience=30)
- Loss: Dice Loss (mejor para desbalance pixeles)
- Augmentation: Elastic deformations, CLAHE, Noise

---

## 📈 Métricas de Evaluación

### Clasificación
```
Accuracy:    (TP+TN)/(Total) - proporción correctas
Precision:   TP/(TP+FP) - de predichos positivos, cuántos correctos
Recall:      TP/(TP+FN) - de positivos reales, cuántos detectados
F1-Score:    2*(P*R)/(P+R) - balance P/R
AUC-ROC:     Área bajo curva ROC - métrica estándar industria
Specificidad: TN/(TN+FP) - evitar falsos positivos
```

### Detección
```
IoU (Intersection over Union): Area(pred ∩ truth) / Area(pred ∪ truth)
Precision @ IoU=0.5: % predicciones correctas (IoU > 0.5)
Recall @ IoU=0.5:    % ground truth detectados (IoU > 0.5)
mAP (mean AP):       Average precision promediado sobre clases
```

### Segmentación
```
Dice Score:  2*(TP)/(2*TP+FP+FN) - overlap binario
Jaccard/IoU: TP/(TP+FP+FN) - intersection over union
Accuracy:    (TP+TN)/(Total) - proporción pixeles correctos
Sensitivity: TP/(TP+FN) - detectar pixels lesión
Specificity: TN/(TN+FP) - detectar pixels normales
```

---

## 🔄 Ciclo de Entrenamiento

### Fase 1: Baseline (Semana 1)
- [ ] Entrenar DenseNet121 clasificación (200 epochs)
- [ ] Validación cruzada 5-fold
- [ ] Reportar AUC-ROC, Accuracy

### Fase 2: Optimización (Semana 2-3)
- [ ] Ajustar hiperparámetros
- [ ] Implementar weighted loss para desbalance
- [ ] Validación externa

### Fase 3: Detección (Semana 4)
- [ ] Entrenar Mask R-CNN
- [ ] Extraer bounding boxes de metadata
- [ ] Validar detecciones

### Fase 4: Segmentación (Semana 5)
- [ ] Entrenar U-Net
- [ ] Entrenar DeepLabV3+
- [ ] Comparar Dice scores

### Fase 5: Análisis (Semana 6)
- [ ] GradCAM para interpretabilidad
- [ ] Confusion matrix detallada
- [ ] Análisis de errores

---

## 📚 Referencias

- resumen.txt (6 PDFs compilados sobre CNN para cáncer de mama)
- INDICE.md (detalle técnico de componentes)
- Sección 7.1, 4.2 de resumen.txt (architectures justification)

---

## ✅ Estado del Proyecto

- [x] Estructura de directorios creada
- [x] Configuración YAML
- [ ] Data loader implementado
- [ ] Modelos CNN implementados
- [ ] Training scripts
- [ ] Notebooks exploratorios
- [ ] Evaluación completa

---

**Última actualización**: Mayo 2026  
**Curso**: Redes Neuronales 2025  
**Status**: En desarrollo
