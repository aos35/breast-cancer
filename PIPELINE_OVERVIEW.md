# PIPELINE OVERVIEW - 3-Task Breast Cancer Detection System

**Fecha de Actualización**: Mayo 2026  
**Status**: ✅ Documentación completa, código implementado, listo para entrenamiento

---

## 🎯 Visión General

Este proyecto implementa un **pipeline integral de 3 tareas** para detección automática de cáncer de mama utilizando Deep Learning:

```
IMAGEN MAMOGRAFÍA (512×512, escala gris)
         ↓
    [TAREA 1] CLASIFICACIÓN
    ├─ Entrada: Imagen completa
    ├─ Modelo: DenseNet121 (7M parámetros)
    ├─ Salida: Clase (Benign/Malignant/Negative)
    └─ Métrica: AUC-ROC ≥ 0.95
         ↓
    ¿Tiene anomalía? → SÍ
         ↓
    [TAREA 2] DETECCIÓN
    ├─ Entrada: Imagen completa
    ├─ Modelo: Mask R-CNN (41M parámetros)
    ├─ Salida: Bounding box (x, y, w, h)
    └─ Métrica: mAP ≥ 0.90
         ↓
    [TAREA 3] SEGMENTACIÓN
    ├─ Entrada: ROI (región detectada)
    ├─ Modelos: U-Net (1.9M) vs DeepLabV3+ (39.5M)
    ├─ Salida: Máscara segmentada + PLA
    └─ Métrica: Dice ≥ 0.85
```

---

## 📊 Dataset (DMID)

### Estadísticas
| Componente | Cantidad | Descripción |
|-----------|----------|-------------|
| Imágenes TIFF | 511 | Mamografías completas (512×512) |
| Máscaras | 269 | Ground truth segmentación |
| PLA | 269 | Anotación pixel-level |
| Registros Metadata | 647 | Etiquetas + coordenadas |
| Imágenes únicas | 510 | Con información de clase |

### Distribución de Clases (Metadata.xlsx)
```
Benign (B):         253 (39.1%) ████████████████████
Malignant (M):      158 (24.4%) ████████████
Negative (N):       200 (30.9%) ███████████████
No Defined:          36 (5.6%) ███
───────────────────────────────
Total:              647
```

### Clase Negativa
- **Definición**: Imágenes sin anomalía clasificada
- **Representación**: 200 de 647 registros (30.9%)
- **Importancia**: Crítica para evitar falsos positivos
- **Manejo**: Incluida como clase #2 en clasificador

---

## 🏗️ Arquitecturas Seleccionadas

### TAREA 1: Clasificación
**Modelo**: DenseNet121  
**Parámetros**: 7M  
**Entrada**: (1, 512, 512)  
**Salida**: (3,) - logits para [Benign, Malignant, Negative]  

**Justificación** (de resumen.txt §7.1):
- ✓ AUC-ROC: 94-96% (mejor que ResNet50)
- ✓ Parámetros: Solo 7M (eficiente)
- ✓ Gradient flow: Excelente (conexiones densas)
- ✓ Transfer learning: Óptimo con datasets pequeños

**Pérdida**: CrossEntropy ponderada  
**Pesos**: Inversamente proporcionales a frecuencia de clase  
**Métricas**: Accuracy, Precision, Recall, F1, AUC-ROC

---

### TAREA 2: Detección
**Modelo**: Mask R-CNN (ResNet50-FPN)  
**Parámetros**: 41M  
**Entrada**: (1, 512, 512)  
**Salida**: Bounding boxes [(x₁,y₁,x₂,y₂), scores, labels]  

**Justificación** (de resumen.txt §4.2):
- ✓ mAP: 92-95% (estado del arte)
- ✓ Manejo múltiples instancias
- ✓ Pre-entrenado en COCO
- ✓ Genera máscaras también (util para siguiente tarea)

**Ground Truth**: Extraído de Metadata.xlsx
- X_Coordinate, Y_Coordinate → centro
- Radius_pixels → radio
- Coordenadas disponibles para 69% del dataset

**Pérdida**: Multi-task (bbox regression + classification)  
**Métricas**: Precision, Recall, mAP, IoU

---

### TAREA 3: Segmentación
**Modelos**: 
1. **U-Net**: 1.9M parámetros (baseline)
2. **DeepLabV3+**: 39.5M parámetros (SOTA)

**Entrada**: (1, 512, 512) o (1, 256, 256) si cropped  
**Salida**: (512, 512) o (256, 256) - probabilidades [0, 1]  

**Justificación** (de resumen.txt §4.2):
- **U-Net**: Excelente con datos pequeños (269 imágenes anotadas)
- **DeepLabV3+**: Multi-escala (ASPP), mejor contexto
- **Ambas**: Mejor que FCN para detalles en límites

**Ground Truth**: 
- Masks: Segmentación binaria
- PLA: Anotación a nivel de píxel

**Pérdida**: Combinada (0.5×Dice + 0.5×BCE)  
**Métricas**: Dice, IoU, Accuracy, Sensitivity, Specificity

---

## 💾 Estructura de Archivos

```
Bloque2_CNN/
├── README.md                    # Documentación principal
├── QUICK_START.md              # Guía de inicio rápido
├── INDICE.md                   # Índice técnico completo
├── PIPELINE_OVERVIEW.md        # Este archivo
├── config.yaml                 # Configuración centralizada
├── requirements.txt            # Dependencias Python
│
├── src/
│   ├── __init__.py            # Exporta módulos principales
│   ├── data_loader.py         # Cargar datos + metadata
│   ├── models.py              # Arquitecturas CNN
│   ├── utils.py               # Loss, métricas, helpers
│   ├── train.py               # [TODO] Training loops
│   └── evaluate.py            # [TODO] Evaluación
│
├── notebooks/
│   ├── 00_exploracion.ipynb      # [TODO] EDA dataset
│   ├── 01_preprocesamiento.ipynb # [TODO] Data pipeline
│   ├── 02_entrenamiento.ipynb    # [TODO] Training
│   └── 03_evaluacion.ipynb       # [TODO] Evaluation
│
└── models/                        # Checkpoints guardados
    ├── classification/
    │   ├── best_model.pth
    │   └── metrics.json
    ├── detection/
    │   ├── best_model.pth
    │   └── metrics.json
    └── segmentation/
        ├── unet_best.pth
        ├── deeplabv3_best.pth
        └── comparison.json
```

---

## 🔄 Flujo de Datos Detallado

### Input Preparation
```python
# Raw TIFF (512×512, 8-bit, escala gris)
image = cv2.imread('IMG001.tif', cv2.IMREAD_GRAYSCALE)

# Normalización a [0, 1]
image = image.astype(np.float32) / 255.0

# Add channel: (1, 512, 512)
image = torch.tensor(image[np.newaxis, ...])
```

### TAREA 1: Classification Forward Pass
```python
image (1, 512, 512)
    ↓ [DenseNet121 backbone]
features (1024, 4, 4)
    ↓ [Global Average Pool]
pooled (1024,)
    ↓ [FC layers: 1024→512→256→3]
logits (3,)
    ↓ [Softmax]
probabilities: [0.05, 0.90, 0.05]
    ↓ [Argmax]
class: 1 (Malignant) ✓
```

### TAREA 2: Detection Forward Pass (si Maligno/Benigno)
```python
image (1, 512, 512)
    ↓ [Mask R-CNN backbone + FPN]
RPN → Proposals
    ↓ [Detection head]
boxes: [[100, 150, 350, 280]]  # (x₁, y₁, x₂, y₂)
scores: [0.92]
labels: [1]
    ↓ [Extract ROI]
roi_crop: (1, 150, 130)  # ~256×256
```

### TAREA 3: Segmentation Forward Pass
```python
roi_crop (1, 256, 256)
    ↓ [U-Net encoder: 4 levels]
    ↓ [Bottleneck]
    ↓ [U-Net decoder: 4 levels with skip]
    ↓ [Final conv + Sigmoid]
mask_prob (1, 256, 256)  # valores [0, 1]
    ↓ [Thresholding @ 0.5]
mask_binary (1, 256, 256)  # valores [0, 1]
```

### Final Output
```python
{
    'classification': {
        'class': 'Malignant',
        'confidence': 0.90,
        'probabilities': [0.05, 0.90, 0.05]
    },
    'detection': {
        'bbox': [100, 150, 350, 280],
        'confidence': 0.92,
        'roi_shape': (256, 256)
    },
    'segmentation': {
        'mask': array (256, 256),
        'dice': 0.87,
        'iou': 0.78
    }
}
```

---

## 📈 Métricas de Referencia

### Benchmarks Esperados (según literatura resumen.txt)

| Tarea | Métrica | Valor | Ref |
|-------|---------|-------|-----|
| **Clasificación** | AUC-ROC | 0.94-0.96 | ResNet50/DenseNet121 |
| | Accuracy | 0.90-0.93 | Transfer learning |
| **Detección** | mAP @ 0.5 | 0.92-0.95 | Mask R-CNN |
| | mAP @ 0.75 | 0.85-0.90 | Stricter threshold |
| **Segmentación** | Dice | 0.85-0.92 | U-Net + DeepLab |
| | IoU | 0.75-0.85 | Pixel-level precision |

### Datos Reales del Proyecto
```
CLASIFICACIÓN:
- Dataset: 510 imágenes (60% benign, 30% negative, 24% malignant)
- Classes: 3 (Benign, Malignant, Negative)
- Desbalance: 39:24:31 (requiere balanced loss)

DETECCIÓN:
- Labeled: ~447 instancias con bbox (69% del dataset)
- Coordinates: X ∈ [28, 4697], Y ∈ [680, 35561], R ∈ [20, 2355]
- Precisión: Metadata.xlsx (ground truth verificado)

SEGMENTACIÓN:
- Annotated: 269 imágenes con Mask + PLA
- Training data: ~161 imágenes (60% de 269)
- Val/Test: ~54 cada uno
```

---

## 🛠️ Configuración Recomendada

### GPU Requirements
```
Minimum:  6GB VRAM (batch_size=8 for all)
Recommended: 12GB VRAM (batch_size=32 classification)
Optimal: 24GB VRAM (all models at full batch)
```

### Training Time Estimates
```
Classification (200 epochs):     4-6 horas
Detection (150 epochs):          6-8 horas
Segmentation (200 epochs):       3-5 horas (U-Net)
                                 5-8 horas (DeepLabV3+)
───────────────────────────────
Total:                          18-27 horas
```

### Hyperparameters (config.yaml)
```yaml
# Optimization
classification:
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adam
  epochs: 200
  early_stopping_patience: 20

detection:
  batch_size: 8           # Smaller for Mask R-CNN
  learning_rate: 0.0001
  epochs: 150
  
segmentation:
  batch_size: 16
  learning_rate: 0.001    # Higher learning rate
  epochs: 200
```

---

## ✅ Implementation Checklist

### Completed ✓
- [x] Dataset analysis (metadata.xlsx)
- [x] Class distribution analysis
- [x] Architecture selection with literature justification
- [x] README.md (comprehensive)
- [x] INDICE.md (technical index)
- [x] config.yaml (complete configuration)
- [x] data_loader.py (metadata + multi-task dataset)
- [x] models.py (all 3 architectures)
- [x] utils.py (losses, metrics, helpers)
- [x] requirements.txt (all dependencies)
- [x] QUICK_START.md (getting started guide)

### TODO 🔄
- [ ] train.py (training loops for 3 tasks)
- [ ] evaluate.py (evaluation and metrics)
- [ ] notebooks/00_exploracion.ipynb (EDA)
- [ ] notebooks/01_preprocesamiento.ipynb (data pipeline)
- [ ] notebooks/02_entrenamiento.ipynb (training)
- [ ] notebooks/03_evaluacion.ipynb (evaluation)
- [ ] Model checkpoints (after training)
- [ ] Results analysis and visualization

---

## 📚 Key Literature Insights

Del resumen.txt compilado:

**Clasificación** (§7.1):
- DenseNet121 vs ResNet50: DenseNet ligeramente mejor
- Transfer learning: 10-20x más rápido
- Data augmentation: Aumenta dataset efectivamente 5-10x

**Detección** (§4.2):
- Mask R-CNN: 92-95% mAP (mejor que Faster R-CNN)
- Handle múltiples lesiones: Crítico para realismo
- Bounding box regression: SmoothL1 loss estándar

**Segmentación** (§4.2, 4.3):
- U-Net: Mejor que FCN para detalles en límites
- SegNet: Alternativa eficiente (29M vs U-Net 1.9M)
- DeepLabV3+: Estado del arte con ASPP
- Dice Loss: Mejor que BCE para desbalance pixeles

**Clase Negativa**:
- 30-40% de datos típicamente "no tumor"
- Critial para especificidad clínica
- Requiere ponderación en loss function

---

## 🚀 Next Steps

### Immediate (Semana 1)
1. Instalar dependencias
2. Verificar dataset
3. Ejecutar EDA (notebooks/00)
4. Preparar datos (notebooks/01)

### Short-term (Semana 2-3)
1. Entrenar clasificación (baseline)
2. Validación cruzada 5-fold
3. Ajustar hiperparámetros

### Medium-term (Semana 4-6)
1. Entrenar detección
2. Entrenar segmentación (U-Net vs DeepLabV3+)
3. Comparar arquitecturas

### Long-term (Semana 7-8)
1. GradCAM para interpretabilidad
2. Análisis de errores
3. Documentación final

---

## 📞 Support & References

- **README.md**: Documentación general y arquitecturas
- **INDICE.md**: Detalles técnicos de cada módulo
- **QUICK_START.md**: Tutorial y ejemplos de código
- **config.yaml**: Todos los parámetros configurables
- **resumen.txt**: Literatura completa con 6 PDFs compilados

---

**Proyecto completado**: Mayo 2026  
**Estado**: ✅ Listo para entrenamiento  
**Última revisión**: Pipeline 3-tareas implementado con manejo de clase negativa
