================================================================================
            ÍNDICE COMPLETO - BLOQUE 2: CNN SEGMENTACIÓN MAMARIA
================================================================================

## 📂 ESTRUCTURA DEL PROYECTO

```
breast-cancer/
└── Bloque2_CNN/                        ← Proyecto para segmentación
    ├── README.md                       (36 KB) Descripción completa
    ├── QUICK_START.md                  (6 KB)  Guía de inicio rápido
    ├── ÍNDICE.md                       ← Este archivo
    │
    ├── requirements.txt                (2 KB)  Dependencias Python
    ├── config.yaml                     (8 KB)  Configuración del proyecto
    │
    ├── 📓 notebooks/                   Jupyter Notebooks interactivos
    │   ├── 00_exploracion.ipynb        EDA - Análisis exploratorio de datos
    │   ├── 01_preprocesamiento.ipynb   Preprocesamiento y augmentación
    │   ├── 02_segmentacion_baseline.ipynb  Entrenamiento de modelos
    │   └── 03_evaluacion.ipynb         Evaluación y visualización
    │
    ├── 🐍 src/                         Código Python reutilizable
    │   ├── __init__.py                 Módulo init
    │   ├── data_loader.py              (4 KB)  Carga de DMID_PNG
    │   │   ├── DMIDSegmentationDataset
    │   │   └── SegmentationDataModule
    │   ├── models.py                   (8 KB)  Arquitecturas CNN
    │   │   ├── UNet
    │   │   ├── FCN
    │   │   ├── DeepLabV3
    │   │   └── create_model()
    │   ├── utils.py                    (6 KB)  Funciones auxiliares
    │   │   ├── DiceLoss
    │   │   ├── IoULoss
    │   │   ├── CombinedLoss
    │   │   ├── SegmentationMetrics
    │   │   └── visualize_segmentation()
    │   ├── train.py                    (TO DO) Loop de entrenamiento
    │   └── evaluate.py                 (TO DO) Evaluación
    │
    ├── 📁 data/                        Datos procesados
    │   ├── train/
    │   │   ├── images/
    │   │   └── masks/
    │   ├── val/
    │   │   ├── images/
    │   │   └── masks/
    │   └── test/
    │       ├── images/
    │       └── masks/
    │
    ├── 🤖 models/                      Modelos entrenados
    │   ├── unet_best.pth               Mejor modelo U-Net
    │   ├── fcn_best.pth                Mejor modelo FCN
    │   ├── deeplabv3_best.pth          Mejor modelo DeepLab V3
    │   └── training_logs.txt           Logs de entrenamiento
    │
    └── 📊 results/                     Resultados finales
        ├── predictions/                Máscaras predichas
        ├── metrics.json                Métricas cuantitativas
        ├── roc_curves.png              Curvas de evaluación
        └── sample_predictions.png      Ejemplos visuales
```

## 📊 DATASET (DMID_PNG)

**Ubicación**: `../../data/raw/DMID_PNG/`

**Estructura**:
```
DMID_PNG/
├── 512x512/
│   ├── TIFF/            511 imágenes originales
│   ├── Masks/           269 máscaras binarias (subconjunto)
│   └── PLA_PNG/         269 anotaciones pixel-level
├── 1024x1024/
│   ├── TIFF/            511 imágenes de alta resolución
│   ├── Masks/           269 máscaras
│   └── PLA/             269 anotaciones
└── Reports/             510 reportes clínicos (.txt)
```

**Estadísticas**:
- Total TIFF: 511 imágenes
- Anotadas: 269 (52.6% cobertura)
- Resoluciones: 512x512 (recomendado), 1024x1024
- Formato: PNG (escala de grises, 8 bits)

## 🚀 INICIO RÁPIDO (5 minutos)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Exploración de datos
jupyter notebook notebooks/00_exploracion.ipynb

# 3. Preprocesamiento
jupyter notebook notebooks/01_preprocesamiento.ipynb

# 4. Entrenamiento
jupyter notebook notebooks/02_segmentacion_baseline.ipynb

# 5. Evaluación
jupyter notebook notebooks/03_evaluacion.ipynb
```

## 📚 DOCUMENTACIÓN DETALLADA

### 1. README.md
Descripción completa del proyecto:
- Objetivos
- Estructura de datos
- Tareas por fase
- Modelos a implementar
- Métricas de evaluación
- Configuración recomendada
- Bibliografía

**Lectura**: 20-30 minutos

### 2. QUICK_START.md
Guía práctica paso-a-paso:
- Setup inicial (15 min)
- Exploración de datos (1-2h)
- Preprocesamiento (1-2h)
- Modelado y entrenamiento (2-3h)
- Evaluación (1-2h)
- Troubleshooting
- Timeline

**Lectura**: 10-15 minutos

### 3. config.yaml
Configuración del proyecto:
- Rutas del dataset
- Parámetros de augmentación
- Arquitectura del modelo
- Hiperparámetros de entrenamiento
- Validación y evaluación
- Logging y monitoreo

**Lectura**: 5-10 minutos

## 🐍 MÓDULOS PYTHON

### data_loader.py (4 KB)
```python
# Carga TIFF, Masks, PLA alineados
dataset = DMIDSegmentationDataset(
    tiff_dir='../../data/raw/DMID_PNG/512x512/TIFF',
    masks_dir='../../data/raw/DMID_PNG/512x512/Masks',
    pla_dir='../../data/raw/DMID_PNG/512x512/PLA_PNG'
)

sample = dataset[0]
# {'image': (1, 512, 512), 'mask': (1, 512, 512), 'pla': (...)}
```

**Clases**:
- `DMIDSegmentationDataset` - Carga individual de datos
- `SegmentationDataModule` - Manejo completo train/val/test

**Métodos principales**:
- `__len__()` - Número de samples
- `__getitem__()` - Cargar sample individual

### models.py (8 KB)
```python
# Arquitecturas disponibles
from src.models import create_model

model = create_model('unet', in_channels=1, out_channels=1)
```

**Modelos**:
1. **U-Net** - 1.9M parámetros
   - Baseline para segmentación médica
   - Encoder-decoder con skip connections
   
2. **FCN** - 29.4M parámetros
   - Fully Convolutional Networks
   - Backbone: VGG16 o ResNet50
   
3. **DeepLab V3** - 39.5M parámetros
   - State-of-the-art en segmentación
   - ASPP + Atrous convolutions

**Uso**:
```python
# Forward pass
y = model(x)  # x: (B, 1, 512, 512) → y: (B, 1, 512, 512)
```

### utils.py (6 KB)
```python
# Funciones auxiliares
from src.utils import DiceLoss, SegmentationMetrics
```

**Loss Functions**:
- `DiceLoss` - Métrica Dice
- `IoULoss` - Intersection over Union
- `CombinedLoss` - Dice + BCE combinado

**Métricas**:
- `SegmentationMetrics.compute_iou()`
- `SegmentationMetrics.compute_dice()`
- `SegmentationMetrics.compute_metrics()` - Todas juntas

**Visualización**:
- `visualize_segmentation()` - Muestra imagen, GT, predicción

## 📓 NOTEBOOKS

### 00_exploracion.ipynb
**Objetivo**: Entender el dataset

**Contenido**:
- Cargar TIFF, Masks, PLA
- Visualizar ejemplos
- Estadísticas de distribución
- Análisis de calidad
- Identificar outliers

**Output**: Visualizaciones exploratorias

**Tiempo**: 1-2 horas

### 01_preprocesamiento.ipynb
**Objetivo**: Preparar datos para entrenamiento

**Contenido**:
- Normalización de imágenes
- Verificación de alineación
- División train/val/test
- Augmentación de datos
- Guardado de datos procesados

**Output**: `data/train/`, `data/val/`, `data/test/`

**Tiempo**: 1-2 horas

### 02_segmentacion_baseline.ipynb
**Objetivo**: Entrenar modelo de segmentación

**Contenido**:
- Cargar datos con DataLoader
- Crear modelo U-Net
- Configurar optimizer + loss
- Training loop
- Monitoreo con TensorBoard
- Guardado de checkpoints

**Output**: `models/unet_best.pth`

**Tiempo**: 3-5 horas (incluyendo entrenamiento)

### 03_evaluacion.ipynb
**Objetivo**: Evaluar modelo entrenado

**Contenido**:
- Cargar modelo
- Evaluar en test set
- Calcular métricas
- Generar predicciones
- Visualizaciones
- Análisis de errores

**Output**: `results/`

**Tiempo**: 1-2 horas

## 📈 FLUJO DE TRABAJO

```
┌──────────────┐
│ 1. SETUP     │ pip install -r requirements.txt
└──────┬───────┘
       ↓
┌──────────────────────┐
│ 2. EXPLORACIÓN       │ 00_exploracion.ipynb
│ (1-2 horas)          │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ 3. PREPROCESAMIENTO  │ 01_preprocesamiento.ipynb
│ (1-2 horas)          │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ 4. MODELADO          │ 02_segmentacion_baseline.ipynb
│ (2-3 horas)          │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ 5. EVALUACIÓN        │ 03_evaluacion.ipynb
│ (1-2 horas)          │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ 6. REPORTE FINAL     │ Documento con resultados
│ (1-2 horas)          │
└──────────────────────┘
```

**Tiempo Total**: ~8-14 horas de trabajo

## 🎯 OBJETIVOS A ALCANZAR

Al finalizar deberías tener:

✅ **Modelos Entrenados**
- U-Net con IoU > 0.75
- FCN con IoU > 0.70
- DeepLab V3 con IoU > 0.80

✅ **Máscaras Predichas**
- Predicciones en todo el test set
- Comparación con ground truth
- Análisis de errores

✅ **Métricas Cuantitativas**
- IoU, Dice, Precision, Recall, F1
- Curvas ROC
- Matrices de confusión

✅ **Visualizaciones**
- Ejemplos de predicciones
- Distribuciones de métricas
- Casos de mejor/peor rendimiento

✅ **Documentación**
- Reporte de resultados
- Conclusiones y discusión
- Posibles mejoras futuras

## 📞 SOPORTE Y AYUDA

### Errores Comunes

| Error | Solución |
|-------|----------|
| CUDA out of memory | Reducir batch_size en config.yaml |
| Dataset no encontrado | Verificar rutas de DMID_PNG |
| Modelo no converge | Ajustar learning rate o loss function |
| ImportError modules | pip install -r requirements.txt |

### Recursos Adicionales

- **README.md** - Descripción completa
- **QUICK_START.md** - Guía práctica
- **src/** - Código documentado
- **config.yaml** - Configuración detallada

## ✅ CHECKLIST FINAL

- [ ] Setup completado (venv + dependencias)
- [ ] Dataset explorado (notebooks/00_exploracion.ipynb)
- [ ] Datos preparados (notebooks/01_preprocesamiento.ipynb)
- [ ] Modelo entrenado (notebooks/02_segmentacion_baseline.ipynb)
- [ ] Evaluación completada (notebooks/03_evaluacion.ipynb)
- [ ] Resultados guardados en results/
- [ ] Reporte final escrito

## 📝 NOTAS IMPORTANTES

1. **Dataset**: Usar resolución 512x512 para training rápido
2. **GPU**: Recomendado NVIDIA GPU (CUDA)
3. **Duración**: El proyecto completo toma 4-5 semanas
4. **Documentación**: Revisar README.md antes de empezar
5. **Configuración**: Personalizar config.yaml según recursos

================================================================================
**Última actualización**: Marzo 2026
**Versión**: 1.0
**Autor**: Estudiante Redes Neuronales
================================================================================
