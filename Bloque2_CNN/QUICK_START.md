================================================================================
                  BLOQUE 2 - QUICK START GUIDE
              CNN para Segmentación de Lesiones Mamarias
================================================================================

## Estructura del Proyecto

```
Bloque2_CNN/
├── README.md               ← Este archivo
├── requirements.txt        ← Dependencias
├── config.yaml             ← Configuración
├── notebooks/              ← Jupyter Notebooks
├── src/                    ← Código Python
├── data/                   ← Datos procesados
├── models/                 ← Modelos entrenados
└── results/                ← Resultados
```

## 1. Setup Inicial (15 minutos)

### 1.1 Crear Entorno Virtual

```bash
cd Bloque2_CNN

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 1.3 Verificar Dataset

```bash
python -c "
from pathlib import Path
tiff_dir = Path('../../data/raw/DMID_PNG/512x512/TIFF')
masks_dir = Path('../../data/raw/DMID_PNG/512x512/Masks')
pla_dir = Path('../../data/raw/DMID_PNG/512x512/PLA_PNG')

print(f'✓ TIFF: {len(list(tiff_dir.glob(\"*.png\")))} imágenes')
print(f'✓ Masks: {len(list(masks_dir.glob(\"*.png\")))} máscaras')
print(f'✓ PLA: {len(list(pla_dir.glob(\"*.png\")))} anotaciones')
"
```

## 2. Exploración de Datos (1-2 horas)

### 2.1 Abrir Notebook

```bash
jupyter notebook notebooks/00_exploracion.ipynb
```

### 2.2 Tareas en el Notebook

- [ ] Cargar dataset DMID_PNG
- [ ] Visualizar ejemplos de TIFF, Masks, PLA
- [ ] Analizar estadísticas (tamaños, distribuciones)
- [ ] Crear visualizaciones (histogramas, etc)

### 2.3 Resumen a Documentar

- Número total de imágenes por tipo
- Rango de valores de píxeles
- Ejemplos visuales de buena/mala anotación

## 3. Preprocesamiento (1-2 horas)

### 3.1 Abrir Notebook

```bash
jupyter notebook notebooks/01_preprocesamiento.ipynb
```

### 3.2 Tareas

- [ ] Normalizar imágenes TIFF (0-1 o 0-255)
- [ ] Verificar alineación TIFF-Masks-PLA
- [ ] Dividir en train/val/test (60/20/20)
- [ ] Aplicar augmentación (rotación, flip, zoom)
- [ ] Guardar datos procesados en `data/`

### 3.3 Salida Esperada

```
data/
├── train/
│   ├── images/ (306 imágenes)
│   └── masks/ (306 máscaras)
├── val/
│   ├── images/ (102 imágenes)
│   └── masks/ (102 máscaras)
└── test/
    ├── images/ (102 imágenes)
    └── masks/ (102 máscaras)
```

## 4. Modelado y Entrenamiento (2-3 horas)

### 4.1 Test de Modelos

```bash
# Probar modelos con input aleatorio
python src/models.py
```

Output esperado:
```
Testing U-Net...
  Input: torch.Size([2, 1, 512, 512])
  Output: torch.Size([2, 1, 512, 512])
  Parameters: 1,940,481

Testing FCN...
  Input: torch.Size([2, 1, 512, 512])
  Output: torch.Size([2, 1, 512, 512])
  Parameters: 29,414,401

Testing DeepLab V3...
  Input: torch.Size([2, 1, 512, 512])
  Output: torch.Size([2, 1, 512, 512])
  Parameters: 39,546,753
```

### 4.2 Notebook de Entrenamiento

```bash
jupyter notebook notebooks/02_segmentacion_baseline.ipynb
```

### 4.3 Tareas

- [ ] Cargar datos con `DMIDSegmentationDataset`
- [ ] Crear modelo U-Net
- [ ] Configurar optimizer y loss function (Dice Loss)
- [ ] Entrenar modelo (50-200 épocas)
- [ ] Monitorear métricas en TensorBoard
- [ ] Guardar best model en `models/`

### 4.4 Comando para TensorBoard

```bash
tensorboard --logdir=./runs/
```

Luego abrir en navegador: `http://localhost:6006`

## 5. Evaluación (1-2 horas)

### 5.1 Notebook de Evaluación

```bash
jupyter notebook notebooks/03_evaluacion.ipynb
```

### 5.2 Tareas

- [ ] Cargar modelo entrenado
- [ ] Evaluar en test set
- [ ] Calcular métricas (IoU, Dice, Precision, Recall, F1)
- [ ] Generar máscaras predichas
- [ ] Visualizar comparaciones
- [ ] Guardar resultados en `results/`

### 5.3 Salida Esperada

```
results/
├── metrics.json          # Métricas cuantitativas
├── predictions/          # Máscaras predichas
├── roc_curves.png        # Curvas de evaluación
└── sample_predictions.png # Ejemplos visuales
```

## 6. Uso de Scripts Python

### 6.1 Data Loader

```python
from src.data_loader import DMIDSegmentationDataset

dataset = DMIDSegmentationDataset(
    tiff_dir='../../data/raw/DMID_PNG/512x512/TIFF',
    masks_dir='../../data/raw/DMID_PNG/512x512/Masks',
    pla_dir='../../data/raw/DMID_PNG/512x512/PLA_PNG'
)

sample = dataset[0]
print(sample['image'].shape)   # Imagen
print(sample['mask'].shape)    # Máscara
print(sample['pla'].shape)     # Anotación
```

### 6.2 Modelos

```python
from src.models import create_model
import torch

# Crear modelo
model = create_model('unet', in_channels=1, out_channels=1)

# Forward pass
x = torch.randn(2, 1, 512, 512)
y = model(x)
print(y.shape)  # (2, 1, 512, 512)
```

### 6.3 Loss Functions

```python
from src.utils import DiceLoss, IoULoss, CombinedLoss

# Dice Loss
dice_loss = DiceLoss()
loss = dice_loss(predictions, targets)

# Combinado
combined_loss = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
loss = combined_loss(predictions, targets)
```

### 6.4 Métricas

```python
from src.utils import SegmentationMetrics
import numpy as np

pred = np.random.rand(512, 512)
target = np.random.randint(0, 2, (512, 512))

metrics = SegmentationMetrics.compute_metrics(pred, target)
print(metrics)  # {'iou': ..., 'dice': ..., 'precision': ..., ...}
```

## 7. Configuración (config.yaml)

Editar `config.yaml` para cambiar:

```yaml
dataset:
  resolution: "512x512"    # O "1024x1024"
  train_split: 0.6
  val_split: 0.2
  test_split: 0.2

training:
  epochs: 200
  batch_size: 16
  learning_rate: 0.001
  loss_fn: "dice"          # O "bce", "focal", "combined"

model:
  type: "unet"             # O "fcn", "deeplabv3"
```

## 8. Troubleshooting

### Problema: "CUDA out of memory"

**Solución**: Reducir batch_size en config.yaml
```yaml
training:
  batch_size: 8  # Reducir de 16 a 8
```

### Problema: "Dataset no encontrado"

**Solución**: Verificar rutas en notebooks
```python
from pathlib import Path
tiff_dir = Path('../../data/raw/DMID_PNG/512x512/TIFF')
print(tiff_dir.exists())  # Debe ser True
```

### Problema: "Modelo no converge"

**Solución**: Probar diferentes configuraciones
- Aumentar learning rate: `0.001` → `0.0001`
- Cambiar loss: `dice` → `combined`
- Aumentar data augmentation

## 9. Timeline Recomendado

| Semana | Tareas | Duración |
|--------|--------|----------|
| 1 | Setup + Exploración + Preprocesamiento | 8 h |
| 2 | Modelado + Entrenamiento (U-Net) | 12 h |
| 3 | Entrenamiento adicional (FCN, DeepLab) | 12 h |
| 4 | Evaluación + Visualización + Reportes | 8 h |

**Total: ~40 horas de trabajo**

## 10. Recursos Adicionales

### Documentación
- README.md - Descripción completa del proyecto
- config.yaml - Configuración detallada
- src/ - Documentación en código

### Papers Recomendados
- Ronneberger et al. (2015): U-Net
- Long et al. (2015): FCN
- Chen et al. (2017): DeepLab v3

### Librerías Clave
- **PyTorch**: Deep learning framework
- **TorchVision**: Pre-trained models
- **OpenCV**: Image processing
- **Scikit-learn**: Métricas de evaluación

## 11. Próximos Pasos

1. **Hoy**: Setup + Exploración
2. **Esta semana**: Preprocesamiento
3. **Próxima semana**: Entrenamiento
4. **Semana 3-4**: Evaluación + Reporte Final

¡Éxito! 🚀

================================================================================
