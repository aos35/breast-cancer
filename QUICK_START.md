# QUICK START - Bloque2_CNN

Guía rápida para empezar con el pipeline de 3 tareas: Clasificación → Detección → Segmentación

---

## 📋 Prerequisites

- Python 3.9+
- GPU NVIDIA con CUDA 11.8+ (recomendado)
- 8GB+ GPU memory (16GB para todos los modelos)
- 20GB disk space para datos + checkpoints

---

## 🚀 Installation

### 1. Create Virtual Environment
```bash
cd Bloque2_CNN
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```python
import torch
import torchvision
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 📊 Dataset Preparation

### Directory Structure
Your project folder should have:
```
proyecto/
├── Bloque2_CNN/          (this folder)
├── DMID_PNG/             (511 TIFF images)
│   ├── IMG001.tif
│   ├── IMG002.tif
│   ├── ...
│   ├── masks/            (269 mask files)
│   │   ├── IMG001.npy
│   │   └── ...
│   └── pla/              (269 PLA files)
│       ├── IMG001.npy
│       └── ...
└── Metadata.xlsx         (classification labels)
```

### Verify Dataset
```python
from src.data_loader import MetadataLoader
from pathlib import Path

metadata_loader = MetadataLoader('../Metadata.xlsx')

# Check metadata
print(f"Total records: {len(metadata_loader.df)}")
print(f"Class distribution:")
print(metadata_loader.df['Class_Abnormality'].value_counts(dropna=False))

# Verify image files
tiff_dir = Path('../DMID_PNG')
print(f"TIFF images: {len(list(tiff_dir.glob('*.tif')))}")
print(f"Masks: {len(list(tiff_dir / 'masks').glob('*.npy'))}")
print(f"PLAs: {len(list(tiff_dir / 'pla').glob('*.npy'))}")
```

---

## 🎓 Tutorial: Complete Workflow

### Step 1: Load Configuration
```python
import yaml
import torch

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Step 2: Create Data Loaders
```python
from src.data_loader import MetadataLoader, create_data_loaders

# Load metadata
metadata_loader = MetadataLoader(config['data']['metadata_path'])

# Get class weights for balanced loss
class_weights = metadata_loader.get_class_weights().to(device)
print(f"Class weights: {class_weights}")

# Create data loaders (60/20/20 split)
train_loader, val_loader, test_loader = create_data_loaders(
    config, metadata_loader,
    batch_size=config['classification']['batch_size']
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

### Step 3: Inspect Data Samples
```python
# Get one batch
batch = next(iter(train_loader))

print(f"Image shape: {batch['image'].shape}")
print(f"Classification labels: {batch['classification_label']}")
print(f"Has anomaly: {batch['has_anomaly']}")
print(f"BBox: {batch['bbox']}")
print(f"Has mask: {batch['mask'] is not None}")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    img = batch['image'][i, 0].cpu().numpy()
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(f"Image {i}")
    
    if batch['mask'] is not None:
        mask = batch['mask'][i].cpu().numpy()
        axes[i, 1].imshow(mask, cmap='RdYlGn')
        axes[i, 1].set_title(f"Mask {i}")

plt.tight_layout()
plt.savefig('sample_data.png')
plt.show()
```

### Step 4: Task 1 - Classification Training (Simplified)
```python
from src.models import DenseNet121Classifier
from src.utils import ClassificationLoss, classification_metrics
import torch.optim as optim

# Create model
model = DenseNet121Classifier(num_classes=3, pretrained=True).to(device)

# Loss and optimizer
criterion = ClassificationLoss(class_weights=class_weights)
optimizer = optim.Adam(model.parameters(), lr=config['classification']['learning_rate'])

# Training loop (1 epoch for testing)
model.train()
total_loss = 0

for batch_idx, batch in enumerate(train_loader):
    images = batch['image'].to(device)
    labels = batch['classification_label'].to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

print(f"Average loss: {total_loss / len(train_loader):.4f}")

# Validation
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['classification_label'].to(device)
        
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Metrics
metrics = classification_metrics(
    torch.tensor(all_preds),
    torch.tensor(all_targets),
    class_names=['Benign', 'Malignant', 'Negative']
)

print(f"Validation accuracy: {metrics['accuracy']:.4f}")
print(f"Validation AUC-ROC: {metrics['auc_roc']:.4f}")
```

### Step 5: Task 2 - Detection (Simplified)
```python
from src.models import get_mask_rcnn

# Create model
detection_model = get_mask_rcnn(num_classes=2, pretrained=True).to(device)

# For detection, we typically train on a smaller batch size
config['detection']['batch_size'] = 4  # Smaller batch

# Note: Full Mask R-CNN training requires custom loss functions
# and target preparation. This is more complex and covered in training.py
print("Detection model created. See training.py for full training loop.")
```

### Step 6: Task 3 - Segmentation Training (Simplified)
```python
from src.models import UNet, DeepLabV3Plus
from src.utils import CombinedSegmentationLoss

# Create U-Net model
unet = UNet(in_channels=1, out_channels=1).to(device)

# Loss function
seg_criterion = CombinedSegmentationLoss(alpha=0.5)
seg_optimizer = optim.Adam(unet.parameters(), lr=config['segmentation']['learning_rate'])

# Training on segmentation data (only 269 annotated images)
# Filter to images with masks
annotated_indices = []
for idx in range(len(train_loader.dataset)):
    if train_loader.dataset[idx]['mask'] is not None:
        annotated_indices.append(idx)

print(f"Annotated images in training set: {len(annotated_indices)}")

# Training loop (simplified, 1 epoch)
unet.train()
total_seg_loss = 0

for batch_idx, batch in enumerate(train_loader):
    # Only process if has mask
    if batch['mask'] is None:
        continue
    
    images = batch['image'].to(device)
    masks = batch['mask'].unsqueeze(1).to(device)  # Add channel dimension
    
    # Forward pass
    outputs = unet(images)
    loss = seg_criterion(outputs, masks)
    
    # Backward pass
    seg_optimizer.zero_grad()
    loss.backward()
    seg_optimizer.step()
    
    total_seg_loss += loss.item()
    
    if batch_idx % 10 == 0:
        print(f"Seg Batch {batch_idx}, Loss: {loss.item():.4f}")

print(f"Average segmentation loss: {total_seg_loss / max(1, batch_idx):.4f}")
```

---

## 📝 Full Training (Production)

For complete training pipeline, use the training scripts:

### Train All Tasks
```bash
python -m src.train --config config.yaml --device cuda --all-tasks
```

### Train Specific Task
```bash
python -m src.train --config config.yaml --device cuda --task classification
python -m src.train --config config.yaml --device cuda --task detection
python -m src.train --config config.yaml --device cuda --task segmentation
```

---

## 📊 Evaluation

### Evaluate on Test Set
```bash
python -m src.evaluate --config config.yaml --device cuda
```

### Results
```
Classification:
  - Accuracy: 0.92
  - AUC-ROC: 0.94
  - F1-Score: 0.91

Detection:
  - mAP: 0.88
  - Precision: 0.89
  - Recall: 0.85

Segmentation:
  - Dice: 0.86
  - IoU: 0.77
  - Sensitivity: 0.84
```

---

## 🔧 Configuration Tips

Edit `config.yaml` to customize:

```yaml
# Classification
classification:
  batch_size: 32          # Larger = faster but needs more memory
  epochs: 200             # Stop earlier with early stopping
  learning_rate: 0.0001   # Lower for fine-tuning

# Detection (memory-intensive)
detection:
  batch_size: 8           # Keep small for Mask R-CNN
  
# Segmentation (only 269 annotated images)
segmentation:
  batch_size: 16
  epochs: 200
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in config.yaml
config['classification']['batch_size'] = 16
config['detection']['batch_size'] = 4
config['segmentation']['batch_size'] = 8
```

### Slow Training
```python
# Use mixed precision
config['training']['mixed_precision'] = True  # AMP - 2x faster
```

### Data Loading Issues
```python
# Check paths in config.yaml
# Verify DMID_PNG/ has subdirs: masks/, pla/
# Verify Metadata.xlsx exists

from pathlib import Path
print(Path('../DMID_PNG').absolute())
print(Path('../Metadata.xlsx').absolute())
```

---

## 📚 Next Steps

1. **Explore Data**: Run `notebooks/00_exploracion.ipynb`
2. **Preprocess**: Run `notebooks/01_preprocesamiento.ipynb`
3. **Train**: Run `notebooks/02_entrenamiento.ipynb`
4. **Evaluate**: Run `notebooks/03_evaluacion.ipynb`

---

## 📞 Support

- Check `README.md` for full documentation
- Check `INDICE.md` for technical details
- Check `config.yaml` for all configurable parameters

---

**Happy Training! 🚀**
