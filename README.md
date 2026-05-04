# Breast Cancer Detection - CNN Pipeline

**Repositorio**: Detección automática de cáncer de mama usando Redes Neuronales Convolucionales  
**Enfoque**: Pipeline multi-tarea (Clasificación → Detección → Segmentación)  
**Dataset**: DMID (Digital Mammography Image Dataset)

---

## 📁 Estructura del Repositorio

```
breast-cancer/
├── .git/                 # Control de versiones Git
├── Bloque2_CNN/          # ⭐ Proyecto principal
│   ├── README.md         # Documentación completa
│   ├── QUICK_START.md    # Guía de inicio rápido
│   ├── INDICE.md         # Índice técnico
│   ├── PIPELINE_OVERVIEW.md  # Visión general del pipeline
│   ├── config.yaml       # Configuración centralizada
│   ├── requirements.txt  # Dependencias Python
│   ├── src/              # Código fuente
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Carga de datos multi-tarea
│   │   ├── models.py         # Arquitecturas CNN
│   │   └── utils.py          # Loss functions, métricas
│   ├── notebooks/        # Jupyter notebooks (por implementar)
│   ├── models/           # Checkpoints guardados (vacío inicialmente)
│   └── .gitignore        # Archivos a ignorar en Git
│
└── README.md             # Este archivo
```

---

## 🚀 Quick Start

### 1. Clonar repositorio
```bash
git clone https://github.com/user/breast-cancer.git
cd breast-cancer
```

### 2. Ir a Bloque2_CNN
```bash
cd Bloque2_CNN
```

### 3. Ver documentación
```bash
# Visión general (START HERE)
cat README.md

# Guía de instalación y ejecución
cat QUICK_START.md

# Detalles técnicos
cat INDICE.md

# Arquitectura del pipeline
cat PIPELINE_OVERVIEW.md
```

---

## 📚 Documentación

| Documento | Contenido |
|-----------|----------|
| **README.md** | Descripción completa del proyecto, arquitecturas, dataset |
| **QUICK_START.md** | Tutorial paso a paso con ejemplos de código |
| **INDICE.md** | Índice técnico de módulos, clases y funciones |
| **PIPELINE_OVERVIEW.md** | Flujo de datos, benchmarks, checklist |
| **config.yaml** | Parámetros configurables para entrenamiento |

---

## 🎯 Pipeline de 3 Tareas

```
IMAGEN MAMOGRAFÍA (512×512)
    ↓
[TAREA 1] CLASIFICACIÓN (DenseNet121)
Benign / Malignant / Negative
    ↓
[TAREA 2] DETECCIÓN (Mask R-CNN) [si tiene anomalía]
Bounding box (x, y, w, h)
    ↓
[TAREA 3] SEGMENTACIÓN (U-Net / DeepLabV3+)
Máscara + PLA
```

---

## 📊 Dataset (DMID)

```
📦 DMID_PNG/
├── 511 imágenes TIFF (512×512)
├── masks/ → 269 máscaras segmentación
├── pla/ → 269 anotaciones pixel-level
└── Metadata.xlsx → Etiquetas clasificación
   ├── Benign: 253 (39.1%)
   ├── Malignant: 158 (24.4%)
   ├── Negative: 200 (30.9%)
   └── No Defined: 36 (5.6%)
```

---

## 🛠️ Instalación

```bash
# Crear virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r Bloque2_CNN/requirements.txt
```

---

## 📝 Requisitos

- Python 3.9+
- GPU NVIDIA con CUDA 11.8+ (recomendado)
- 8GB+ VRAM
- 20GB disk space

---

## 🔗 Enlaces

- **Bloque2_CNN/README.md** - Documentación principal
- **Bloque2_CNN/QUICK_START.md** - Tutorial completo
- **Bloque2_CNN/config.yaml** - Configuración

---

## 📄 Licencia

Proyecto académico - Curso Redes Neuronales 2026

---

**Para empezar**: Lee [Bloque2_CNN/README.md](Bloque2_CNN/README.md)
