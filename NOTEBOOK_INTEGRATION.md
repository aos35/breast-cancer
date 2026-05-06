# 📓 Notebook Integration Guide

**Estado**: ✅ Notebooks alineados con `train.py` y `evaluate.py`

---

## 📚 Estructura de Notebooks

```
notebooks/
├── 00_exploracion.ipynb              (Exploratory Data Analysis)
├── 01_preprocesamiento.ipynb         (Data Preprocessing & Augmentation)
├── 02_entrenamiento.ipynb            (Training Demo)
└── 03_evaluacion.ipynb               (Evaluation & Results)
```

---

## 🔗 Integración con `src/data_loader.py`

### ✅ `01_preprocesamiento.ipynb` - ACTUALIZADO

El notebook ahora:
- ✅ Usa `DMIDMultiTaskDataset` desde `src/data_loader.py`
- ✅ Carga datos con rutas correctas (`../../DMID_PNG/`, `../../Metadata.xlsx`)
- ✅ Crea DataLoaders con `create_data_loaders()` (igual que train.py)
- ✅ Demuestra pipeline de augmentation (transformaciones realistas)
- ✅ Visualiza ejemplos con overlay imagen+máscara
- ✅ Muestra cómo integrar con `ClassificationTrainer`

**Flujo:**
```python
# 1. Importar desde src (centralizado)
from src.data_loader import DMIDMultiTaskDataset, create_data_loaders

# 2. Cargar dataset con nuestras transformaciones
dataset = DMIDMultiTaskDataset(...)

# 3. Crear DataLoaders (igual que train.py)
loaders = create_data_loaders(config_path='config.yaml')

# 4. Entrenar (opcional en notebook o desde terminal)
from src.train import ClassificationTrainer
trainer = ClassificationTrainer(config='config.yaml')
trainer.fit(fold=0)
```

---

## 🎓 Cómo Usar los Notebooks

### **Paso 1: Exploración (00_exploracion.ipynb)**

Objetivo: Entender el dataset

```python
# Qué hace:
# - Carga metadata.xlsx
# - Analiza distribución de clases
# - Visualiza imágenes por clase
# - Muestra estadísticas (shape, min/max, etc.)
# - Detecta imágenes sin máscara

# Ejecutar:
jupyter notebook 00_exploracion.ipynb
```

**Expected Output:**
- Histogramas de clase distribution
- Grid de imágenes de cada clase
- Estadísticas de dimensiones
- Resumen de anotaciones disponibles

---

### **Paso 2: Preprocesamiento (01_preprocesamiento.ipynb)**

Objetivo: Validar el pipeline de datos antes de entrenar

```python
# Qué hace:
# - Define augmentation pipeline
# - Carga dataset con transformaciones
# - Verifica rango de normalización [0, 1]
# - Visualiza ejemplos augmentados
# - Crea train/val/test splits

# Ejecutar:
jupyter notebook 01_preprocesamiento.ipynb
```

**Expected Output:**
```
✓ Dataset cargado:
  - Total imágenes: 511
  - Clase distribution: ...

✓ Estadísticas del sample:
  - Shape: (1, 256, 256)
  - Rango: [0.0000, 1.0000]  ← Normalizado

✓ DataLoaders creados:
  - Train batches: 32
  - Val batches: 8
```

**Transformaciones aplicadas:**
- Resize a 256x256
- Rotación ±15°
- Flips (horizontal + vertical)
- Elastic deformation (realista para tejidos)
- Ruido gaussiano
- Normalización a [0,1]

---

### **Paso 3: Entrenamiento (02_entrenamiento.ipynb)**

Objetivo: Demostrar cómo entrenar modelos interactivamente

```python
# Qué hace:
# - Importa trainers desde src.train
# - Configura modelo (DenseNet121, Mask R-CNN, U-Net)
# - Ejecuta epoch de ejemplo (para validar setup)
# - Muestra pérdida y métricas en tiempo real
# - (Opcional) Continúa con full training

# Ejecutar:
jupyter notebook 02_entrenamiento.ipynb

# O entrenar desde terminal (recomendado para full training):
python quick_start.py train --task classification
```

**Ventaja del notebook:**
- Ver losses/métricas en tiempo real
- Plotear gráficos de evolución
- Inspeccionar activaciones de modelo
- Parar/resumir en cualquier momento

---

### **Paso 4: Evaluación (03_evaluacion.ipynb)**

Objetivo: Analizar resultados después del entrenamiento

```python
# Qué hace:
# - Carga modelos entrenados desde checkpoints
# - Calcula métricas (AUC, Dice, mAP, etc.)
# - Genera confusion matrices
# - Plottea ROC/PR curves
# - Analiza ejemplos correctos/incorrectos
# - Genera reporte HTML

# Ejecutar:
jupyter notebook 03_evaluacion.ipynb

# O evaluar desde terminal:
python quick_start.py eval --task all
```

**Salidas:**
- Confusion matrices (heatmap)
- ROC curves (per-class)
- PR curves (detection)
- Ejemplos correctos vs incorrectos
- Error analysis por clase

---

## 🔄 Flujo Completo (Recomendado)

### **Opción A: Solo Notebooks (Exploración + Demo)**

```bash
cd notebooks

# 1. Exploración
jupyter notebook 00_exploracion.ipynb
# Revisa distribución de datos

# 2. Preprocesamiento
jupyter notebook 01_preprocesamiento.ipynb
# Valida pipeline de datos

# 3. Demo entrenamiento (1 epoch)
jupyter notebook 02_entrenamiento.ipynb
# Verifica que todo funciona

# 4. Resultados
jupyter notebook 03_evaluacion.ipynb
# Carga resultados previos
```

### **Opción B: Terminal + Notebooks (Recomendado para Producción)**

```bash
# 1. Exploración (Notebook)
jupyter notebook 00_exploracion.ipynb

# 2. Preprocesamiento (Notebook)
jupyter notebook 01_preprocesamiento.ipynb

# 3. ENTRENAR (Terminal - MUCHO MÁS RÁPIDO)
cd ..
python quick_start.py train --task all
# ⏱️ Mientras entrena en terminal, puedes hacer otras cosas

# 4. EVALUAR (Terminal)
python quick_start.py eval --task all

# 5. ANALIZAR (Notebook)
cd notebooks
jupyter notebook 03_evaluacion.ipynb
# Carga resultados y visualiza
```

---

## 📊 Consistencia Garantizada

Todos los notebooks usan **las mismas** funciones que train.py:

| Componente | Ubicación | Usado por |
|-----------|-----------|----------|
| `DMIDMultiTaskDataset` | `src/data_loader.py` | Notebooks + train.py |
| `create_data_loaders` | `src/data_loader.py` | Notebooks + train.py |
| Transformaciones | `config.yaml` | Notebooks + train.py |
| Modelos | `src/models.py` | Notebooks + train.py + evaluate.py |
| Métricas | `src/utils.py` | Notebooks + evaluate.py |

✅ **Resultado**: No hay código duplicado, todo centralizado en `src/`

---

## 🔧 Personalización: Agregar Transformaciones Propias

Si quieres modificar el pipeline de augmentation:

**En `config.yaml`:**
```yaml
augmentation:
  train:
    - type: "Rotate"
      limit: 20
      p: 0.5
    - type: "HorizontalFlip"
      p: 0.5
    - type: "ElasticTransform"
      alpha: 1
      sigma: 50
      p: 0.3
  val: null  # Sin augmentation en validación
```

O **directamente en el notebook** (para experimentos rápidos):

```python
# Crear transformación personalizada
custom_transform = A.Compose([
    A.Resize(256, 256),
    A.Rotate(limit=20, p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ToTensorV2()
])

# Crear dataset con transformación personalizada
dataset = DMIDMultiTaskDataset(
    image_dir=str(DMID_PNG_DIR),
    metadata_path=str(METADATA_PATH),
    transform=custom_transform
)
```

---

## 📝 Rutas y Configuración

### **Rutas Correctas**

Todos los notebooks usan rutas relativas **correctas** desde `notebooks/`:

```python
PROJECT_ROOT = Path('..').resolve()  # Sube un nivel a Bloque2_CNN

DMID_PNG_DIR = PROJECT_ROOT / 'DMID_PNG'        # ✅ ../../DMID_PNG/
METADATA_PATH = PROJECT_ROOT / 'Metadata.xlsx'  # ✅ ../../Metadata.xlsx
CONFIG_PATH = PROJECT_ROOT / 'config.yaml'      # ✅ ../../config.yaml
```

### **Validación Automática**

El notebook verifica que las rutas existan:

```python
print(f"✓ DMID_PNG: {DMID_PNG_DIR} (exists: {DMID_PNG_DIR.exists()})")
print(f"✓ Metadata: {METADATA_PATH} (exists: {METADATA_PATH.exists()})")
```

---

## ⚡ Tips & Tricks

### **Para Desarrollo Rápido (Notebook)**

```python
# Usar solo 100 imágenes para debugging
small_dataset = DMIDMultiTaskDataset(..., sample_fraction=0.2)

# Entrenar por solo 1 epoch para test
trainer.fit(fold=0, max_epochs=1)

# Usar CPU si no tienes GPU disponible
torch.cuda.is_available()  # Verifica
device = 'cpu'  # Force CPU
```

### **Para Producción (Terminal)**

```bash
# Entrenar con todas las configuraciones de config.yaml
python quick_start.py train --task all

# Ver progreso en tiempo real
# Los logs se imprimen en la terminal
```

### **Debugging de DataLoaders**

```python
# Ver un batch completo
train_loader = loaders['train']
batch = next(iter(train_loader))

print(f"Batch shapes:")
print(f"  - Images: {batch['image'].shape}")
print(f"  - Labels: {batch['label'].shape if 'label' in batch else 'N/A'}")
print(f"  - Masks: {batch['mask'].shape if 'mask' in batch else 'N/A'}")

# Verificar normalización
print(f"Image range: [{batch['image'].min()}, {batch['image'].max()}]")
# Debería estar en [0, 1]
```

---

## 📚 Referencia Rápida

| Tarea | Notebook | Terminal |
|------|----------|----------|
| Explorar datos | `00_exploracion.ipynb` | N/A |
| Validar pipeline | `01_preprocesamiento.ipynb` | N/A |
| Demo training | `02_entrenamiento.ipynb` | N/A |
| Full training | N/A | `python quick_start.py train --task all` |
| Evaluar modelos | `03_evaluacion.ipynb` | `python quick_start.py eval --task all` |
| Debug datos | `01_preprocesamiento.ipynb` | N/A |
| Debug modelos | `02_entrenamiento.ipynb` | N/A |

---

## ✅ Checklist: Antes de Ejecutar

- [ ] DMID_PNG existe en `../DMID_PNG/` (desde notebooks/)
- [ ] Metadata.xlsx existe en `../Metadata.xlsx`
- [ ] PyTorch instalado: `pip install -r requirements.txt`
- [ ] Kernel de Jupyter activado
- [ ] Rutas validadas al iniciar notebook

---

## 🆘 Troubleshooting

### **Error: ModuleNotFoundError: No module named 'src'**

Solución:
```python
# Agregar al inicio del notebook
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve().parent))
```

### **Error: FileNotFoundError - DMID_PNG no encontrado**

Solución:
```python
from pathlib import Path
PROJECT_ROOT = Path('..').resolve()
DMID_PNG_DIR = PROJECT_ROOT / 'DMID_PNG'
print(f"Buscando en: {DMID_PNG_DIR}")
print(f"Existe: {DMID_PNG_DIR.exists()}")
# Navega al directorio correcto
```

### **Error: CUDA out of memory**

Solución:
```python
# Reducir batch size en config.yaml o en el notebook
loaders = create_data_loaders(..., batch_size=8)
```

---

## 📞 Contacto & Apoyo

Si tienes problemas:

1. Revisa [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Troubleshooting
2. Revisa [EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md) - Outputs esperados
3. Comprueba que DMID_PNG y Metadata.xlsx existen
4. Verifica las rutas con `Path.resolve().absolute()`

---

**Última actualización**: May 4, 2026  
**Versión**: 1.0 - ✅ Notebooks + Code Centralizados
