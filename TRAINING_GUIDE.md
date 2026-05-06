# Training & Evaluation Guide - Bloque2_CNN

Documentación completa para usar los scripts de entrenamiento y evaluación.

---

## 📚 Tabla de Contenidos

1. [Training (train.py)](#training)
2. [Evaluation (evaluate.py)](#evaluation)
3. [Ejemplos de uso](#ejemplos)
4. [Troubleshooting](#troubleshooting)

---

## 🎯 Training

### Script: `src/train.py`

Entrena los modelos para las 3 tareas (clasificación, detección, segmentación).

#### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

#### Uso Básico

**Entrenar todas las tareas:**
```bash
python -m src.train --config config.yaml --task all
```

**Entrenar tarea específica:**
```bash
python -m src.train --config config.yaml --task classification
python -m src.train --config config.yaml --task detection
python -m src.train --config config.yaml --task segmentation --model unet
```

#### Parámetros

```
--config          (default: config.yaml)    Ruta al archivo de configuración
--task            (default: all)             Tarea a entrenar: classification, detection, segmentation, all
--model           (default: unet)            Modelo de segmentación: unet o deeplabv3plus
```

### Configuración en config.yaml

#### Clasificación
```yaml
classification:
  model: "densenet121"
  num_classes: 3
  batch_size: 32
  epochs: 200
  learning_rate: 0.0001
  optimizer: "adam"
  scheduler: "exponential"
  class_weights: [0.51, 0.81, 0.65]
  early_stopping_patience: 20
```

#### Detección
```yaml
detection:
  model: "mask_rcnn"
  batch_size: 8          # Más pequeño por memoria
  epochs: 150
  learning_rate: 0.0001
  optimizer: "sgd"
  momentum: 0.9
  early_stopping_patience: 20
```

#### Segmentación
```yaml
segmentation:
  batch_size: 16
  epochs: 200
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 30
```

### Salidas Generadas

Estructura de directorios después del entrenamiento:

```
models/
├── classification/
│   └── best.pth                  # Mejor checkpoint (clasificación)
├── detection/
│   └── best.pth                  # Mejor checkpoint (detección)
└── segmentation/
    ├── best_unet.pth             # Mejor checkpoint (U-Net)
    └── best_deeplabv3plus.pth    # Mejor checkpoint (DeepLabV3+)
```

### Monitoreo del Entrenamiento

Los logs se imprimen en tiempo real:

```
INFO - Epoch 1/200
INFO - Batch 10/10, Loss: 0.5234
INFO - Train Loss: 0.5230, Val Loss: 0.4856
INFO - Val Accuracy: 0.8500, Val AUC-ROC: 0.9123
```

---

## 📊 Evaluation

### Script: `src/evaluate.py`

Evalúa los modelos entrenados en el test set y genera reportes.

#### Uso Básico

**Evaluar todas las tareas:**
```bash
python -m src.evaluate --config config.yaml --task all
```

**Evaluar tarea específica:**
```bash
python -m src.evaluate --config config.yaml --task classification
python -m src.evaluate --config config.yaml --task detection
python -m src.evaluate --config config.yaml --task segmentation --seg-model unet
```

#### Parámetros

```
--config              Ruta al config.yaml
--task                all, classification, detection, segmentation
--clf-checkpoint      (default: models/classification/best.pth)
--det-checkpoint      (default: models/detection/best.pth)
--seg-checkpoint      (default: None - usa modelo en config.yaml)
--seg-model           unet o deeplabv3plus
```

### Salidas Generadas

#### Clasificación
```
results/classification/
├── results.json              # Métricas en formato JSON
├── confusion_matrix.png      # Matriz de confusión visualizada
```

**Contenido results.json:**
```json
{
  "test_loss": 0.4123,
  "metrics": {
    "accuracy": 0.8500,
    "auc_roc": 0.9234,
    "precision_class_0": 0.87,
    "recall_class_0": 0.85,
    "f1_class_0": 0.86,
    ...
  },
  "confusion_matrix": [[...], [...], [...]]
}
```

#### Detección
```
results/detection/
├── results.json              # TP, FP, FN, Precision, Recall, F1, IoU
```

**Contenido:**
```json
{
  "tp": 45,
  "fp": 8,
  "fn": 5,
  "precision": 0.849,
  "recall": 0.900,
  "f1": 0.874,
  "mean_iou": 0.756
}
```

#### Segmentación
```
results/segmentation/
├── results_unet.json         # Métricas U-Net
└── results_deeplabv3plus.json # Métricas DeepLabV3+
```

**Contenido:**
```json
{
  "dice": 0.8234,
  "iou": 0.7123,
  "accuracy": 0.9145,
  "sensitivity": 0.8900,
  "specificity": 0.9234,
  "dice_std": 0.0345,
  "iou_std": 0.0267
}
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Entrenar Clasificación

```bash
# Entrenar clasificación
python -m src.train --config config.yaml --task classification

# Esperar a que termine (~30 minutos con GPU)
# Resultado en: models/classification/best.pth

# Evaluar
python -m src.evaluate --config config.yaml --task classification

# Resultados en: results/classification/
```

### Ejemplo 2: Entrenar Segmentación (ambos modelos)

```bash
# Entrenar U-Net
python -m src.train --config config.yaml --task segmentation --model unet

# Entrenar DeepLabV3+
python -m src.train --config config.yaml --task segmentation --model deeplabv3plus

# Evaluar ambos
python -m src.evaluate --config config.yaml --task segmentation --seg-model unet
python -m src.evaluate --config config.yaml --task segmentation --seg-model deeplabv3plus

# Comparar resultados
cat results/segmentation/results_unet.json
cat results/segmentation/results_deeplabv3plus.json
```

### Ejemplo 3: Flujo Completo (Clasificación)

```bash
# 1. Entrenar
python -m src.train --config config.yaml --task classification

# 2. Esperar a early stopping (~20-30 minutos)
# Log final: "Early stopping triggered!"

# 3. Evaluar
python -m src.evaluate --config config.yaml --task classification

# 4. Ver resultados
cat results/classification/results.json
```

### Ejemplo 4: Usar Checkpoints Personalizados

```bash
# Entrenar guardando en ruta personalizada
# (Editar checkpoint_path en train.py)

# Evaluar con checkpoint personalizado
python -m src.evaluate \
  --config config.yaml \
  --task classification \
  --clf-checkpoint /path/to/custom/model.pth
```

---

## 🔧 Troubleshooting

### Error: "CUDA out of memory"

**Problema:** La GPU no tiene memoria suficiente

**Soluciones:**
1. Reducir batch_size en config.yaml:
   ```yaml
   classification:
     batch_size: 16  # Reducir de 32
   ```

2. Usar CPU (lento, no recomendado):
   ```python
   device = 'cpu'  # En train.py y evaluate.py
   ```

3. Usar gradient accumulation (modificar train.py)

### Error: "Model checkpoint not found"

**Problema:** No encuentra los pesos del modelo entrenado

**Soluciones:**
1. Verificar que el modelo fue entrenado:
   ```bash
   ls -la models/classification/best.pth
   ```

2. Especificar ruta correcta:
   ```bash
   python -m src.evaluate --clf-checkpoint models/classification/best.pth
   ```

### Error: "Data loader returns None for masks"

**Problema:** Algunas imágenes no tienen máscaras

**Solución:** En train.py y evaluate.py, se saltan automáticamente si mask es None (esperado para 242 imágenes sin máscara).

### Logs muy lento o no aparecen

**Solución:** Usar buffering=1:
```bash
python -u -m src.train --config config.yaml --task classification
```

---

## 📈 Métricas Esperadas (Literatura)

Basadas en revisión de artículos publicados:

### Clasificación (DenseNet121)
- AUC-ROC: 0.94-0.96
- Accuracy: 85-90%
- Sensibilidad: 92-95%
- Especificidad: 88-92%

### Detección (Mask R-CNN)
- mAP: 0.92-0.95
- Precision: 90-93%
- Recall: 91-94%
- Mean IoU: 0.85-0.88

### Segmentación (U-Net)
- Dice Score: 0.88-0.92
- IoU: 0.80-0.86
- Accuracy: 94-97%

### Segmentación (DeepLabV3+)
- Dice Score: 0.90-0.94
- IoU: 0.82-0.88
- Accuracy: 95-98%

---

## 📝 Notas Importantes

1. **Early Stopping**: Se activa automáticamente si la validación no mejora por N epochs (configurable)

2. **K-fold Cross-validation**: Implementado en el data loader (puede adaptarse en train.py)

3. **Seeds**: Fijado a 42 para reproducibilidad

4. **Logging**: Todos los logs se guardan en la salida estándar (redirigible a archivo)

5. **GPU/CPU**: Detecta automáticamente disponibilidad de CUDA

---

## 🚀 Próximos Pasos

1. Ejecutar entrenamiento completo
2. Revisar métricas en resultados
3. Generar confusion matrices y ROC curves
4. Comparar con literatura
5. Ajustar hiperparámetros si es necesario

---

**Última actualización**: May 4, 2026  
**Versión**: 1.0
