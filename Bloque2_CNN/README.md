================================================================================
              BLOQUE 2: CONVOLUTIONAL NEURAL NETWORKS
        Visión por Computador - Segmentación de Lesiones Mamarias
================================================================================

## Objetivo del Proyecto

Implementar Redes Neuronales Convolucionales (CNN) para:

1. **Clasificación de Imágenes**: Benign vs Malignant
2. **Detección de Objetos**: Localizar lesiones en mamografías
3. **Segmentación Semántica**: 
   - Generar máscaras (Masks) de lesiones
   - Generar anotaciones a nivel de píxel (PLA - Pixel Level Annotation)

## Dataset

**Fuente**: DMID (Digital Mammography Dataset)
**Ubicación**: `../../data/raw/DMID_PNG/`

### Estructura de Datos:

```
DMID_PNG/
├── 512x512/                    # Resolución 512x512 (recomendado)
│   ├── TIFF/                   # 511 imágenes originales
│   │   ├── IMG001.png
│   │   ├── IMG002.png
│   │   └── ... (IMG001-IMG511)
│   ├── Masks/                  # 269 máscaras binarias (subconjunto)
│   │   ├── IMG001.png          # Máscara binaria (0=fondo, 1=lesión)
│   │   ├── IMG008.png
│   │   └── ... (nombres salteados)
│   └── PLA_PNG/                # 269 anotaciones a nivel de píxel
│       ├── IMG001.png          # Multi-clase: 0=fondo, 1,2,3...=tipos lesión
│       ├── IMG008.png
│       └── ... (nombres salteados)
│
├── 1024x1024/                  # Resolución 1024x1024 (alta resolución)
│   ├── TIFF/     (511 imágenes)
│   ├── Masks/    (269 máscaras)
│   └── PLA/      (269 anotaciones)
│
└── Reports/                    # 510 reportes clínicos (.txt)
    ├── Img001.txt
    ├── Img002.txt
    └── ...
```

### Características del Dataset:

| Característica | Valor |
|---|---|
| Total imágenes | 511 TIFF |
| Imágenes anotadas | 269 (Masks + PLA) |
| Cobertura de anotación | 52.6% del dataset |
| Resoluciones disponibles | 512x512, 1024x1024 |
| Formato | PNG (escala de grises) |
| Información clínica | Reports en .txt |

## Estructura del Proyecto

```
Bloque2_CNN/
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias Python
├── config.yaml                 # Configuración del proyecto
│
├── notebooks/                  # Jupyter Notebooks
│   ├── 00_exploracion.ipynb            # EDA y visualización
│   ├── 01_preprocesamiento.ipynb       # Procesamiento de datos
│   ├── 02_segmentacion_baseline.ipynb  # Modelos baseline (U-Net)
│   └── 03_evaluacion.ipynb             # Evaluación de resultados
│
├── src/                        # Código Python
│   ├── __init__.py
│   ├── data_loader.py          # Carga de TIFF, Masks, PLA
│   ├── models.py               # Arquitecturas: U-Net, FCN, DeepLab
│   ├── utils.py                # Funciones auxiliares
│   ├── train.py                # Loop de entrenamiento
│   └── evaluate.py             # Evaluación de modelos
│
├── data/                       # Datos procesados
│   ├── train/                  # Datos de entrenamiento (dividido automático)
│   ├── val/                    # Datos de validación
│   └── test/                   # Datos de test
│
├── models/                     # Modelos entrenados
│   ├── unet_best.pth
│   ├── fcn_best.pth
│   └── training_logs.txt
│
└── results/                    # Resultados y visualizaciones
    ├── predictions/            # Masks predichas
    ├── metrics.json
    ├── roc_curves.png
    └── sample_predictions.png
```

## Tareas por Fase

### Fase 1: Exploración (Semana 1)
- [ ] Cargar TIFF, Masks, PLA de DMID_PNG
- [ ] Análisis exploratorio (distribución, estadísticas)
- [ ] Visualización de imágenes y anotaciones
- [ ] Resumen en `notebooks/00_exploracion.ipynb`

### Fase 2: Preprocesamiento (Semana 1-2)
- [ ] Normalización de imágenes TIFF
- [ ] División train/val/test (60/20/20)
- [ ] Augmentación de datos (rotación, flip, zoom)
- [ ] Alineación de Masks y PLA con TIFF
- [ ] Implementar `src/data_loader.py`

### Fase 3: Modelado (Semana 2-3)
- [ ] Implementar arquitecturas:
  - [ ] U-Net (baseline para segmentación)
  - [ ] FCN (Fully Convolutional Networks)
  - [ ] DeepLab v3 (state-of-the-art)
- [ ] Definir loss functions (Dice, CrossEntropy, Focal Loss)
- [ ] Resumen en `src/models.py`

### Fase 4: Entrenamiento (Semana 3-4)
- [ ] Entrenar modelos con GPU
- [ ] Monitorear pérdida y métricas (Dice, IoU, Accuracy)
- [ ] Early stopping y checkpoint saving
- [ ] Implementar `src/train.py`

### Fase 5: Evaluación (Semana 4-5)
- [ ] Evaluar en test set (métricas: IoU, Dice, Precision, Recall)
- [ ] Generar máscaras predichas
- [ ] Comparar con ground truth
- [ ] Visualizaciones: ROC, confusion matrices, sample predictions
- [ ] Generar reportes finales

## Tareas de Implementación

### Python Scripts Necesarios:

1. **`src/data_loader.py`** - Dataset loading
   ```python
   class DMIDDataset(Dataset):
       def __init__(self, tiff_dir, mask_dir, pla_dir, transform=None):
           # Cargar TIFF, Masks, PLA
           # Alinear por nombre de archivo (IMG001.png, etc)
       
       def __getitem__(self, idx):
           # Retornar: (imagen TIFF, máscara, anotación PLA)
   ```

2. **`src/models.py`** - Arquitecturas CNN
   ```python
   class UNet(nn.Module):
       # Arquitectura U-Net para segmentación
   
   class FCN(nn.Module):
       # FCN 16s/32s para segmentación
   
   class DeepLabV3(nn.Module):
       # DeepLab V3 con atrous convolutions
   ```

3. **`src/train.py`** - Entrenamiento
   ```python
   def train_epoch(model, dataloader, optimizer, criterion):
       # Loop de entrenamiento
   
   def validate(model, dataloader, criterion):
       # Validación y métricas
   ```

4. **`src/evaluate.py`** - Evaluación
   ```python
   def compute_metrics(predictions, ground_truth):
       # IoU, Dice Coefficient, Accuracy
   ```

## Modelos a Implementar

### 1. U-Net (Baseline)
```
Architecture: Encoder-Decoder con Skip Connections
Input: 512x512 (o 1024x1024)
Output: Máscara binaria (0=fondo, 1=lesión)
Parámetros: ~7M
```

### 2. FCN (Fully Convolutional Networks)
```
Architecture: Convoluciones sin fully-connected
Input: Imagen TIFF variable
Output: Máscara segmentada
Parámetros: ~30M (con VGG16 backbone)
```

### 3. DeepLab V3
```
Architecture: Atrous Convolutions + ASPP
Input: 512x512
Output: Máscara + Anotación PLA
Parámetros: ~40M
```

## Métricas de Evaluación

| Métrica | Fórmula | Rango | Interpretación |
|---|---|---|---|
| **IoU (Intersection over Union)** | TP/(TP+FP+FN) | [0,1] | Qué tan bien se superpone predicción vs GT |
| **Dice Coefficient** | 2*TP/(2*TP+FP+FN) | [0,1] | Overlap similarity |
| **Accuracy** | (TP+TN)/(Total) | [0,1] | Precisión general |
| **Precision** | TP/(TP+FP) | [0,1] | De las lesiones detectadas, qué % son correctas |
| **Recall/Sensitivity** | TP/(TP+FN) | [0,1] | De las lesiones reales, qué % se detectaron |
| **F1-Score** | 2*(Precision*Recall)/(Precision+Recall) | [0,1] | Balance entre Precision y Recall |

## Configuración Recomendada

### Hiperparámetros:
```yaml
# Entrenamiento
learning_rate: 0.001
batch_size: 16           # Reducido por segmentación
epochs: 200
patience_early_stopping: 30

# Arquitectura
input_size: 512          # o 1024
output_channels: 1       # Masks (binario)
# Para PLA: output_channels: N (número de clases)

# Augmentación
augmentation:
  rotation: 15           # ±15 grados
  flip: true             # Horizontal y vertical
  zoom: [0.8, 1.2]       # 80% - 120%
  elastic_deform: true   # Deformaciones elásticas (común en médica)
```

## Software y Dependencias

```
Python: 3.8+
PyTorch: 2.0.1
TorchVision: 0.15+
OpenCV: 4.5+
Pandas: 1.3+
NumPy: 1.21+
Matplotlib: 3.4+
Scikit-learn: 0.24+
Jupyter: 1.0+
```

Ver `requirements.txt` para versiones exactas.

## Cómo Ejecutar

### 1. Setup (primera vez)
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Exploración
```bash
jupyter notebook notebooks/00_exploracion.ipynb
```

### 3. Preprocesamiento y Entrenamiento
```bash
jupyter notebook notebooks/02_segmentacion_baseline.ipynb
```

### 4. Evaluación
```bash
jupyter notebook notebooks/03_evaluacion.ipynb
```

## Resultados Esperados

Al finalizar, deberías tener:

✅ **Modelos entrenados** con métricas de validación > 0.85 IoU  
✅ **Máscaras predichas** para todo el test set  
✅ **Anotaciones PLA** generadas automáticamente  
✅ **Reportes** comparando predicciones vs ground truth  
✅ **Visualizaciones** de mejor/peor casos  
✅ **Documento** con resultados y análisis  

## Bibliografía Recomendada

- Ronneberger et al., 2015: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Long et al., 2015: "Fully Convolutional Networks for Semantic Segmentation"
- Chen et al., 2017: "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"
- He et al., 2016: "Deep Residual Learning for Image Recognition" (ResNet backbone)

## Contacto y Ayuda

Para dudas sobre el proyecto:
- Revisar `notebooks/` para ejemplos
- Consultar `src/` para implementación detallada
- Ver `config.yaml` para parámetros

================================================================================
