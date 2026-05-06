# Expected Outputs & Metrics Reference

## 📊 Salidas Esperadas Después del Entrenamiento

### 1. Logs del Entrenamiento (Terminal)

#### Clasificación (DenseNet121)
```
INFO - Training Classification Model (DenseNet121)
INFO - Configuration loaded from config.yaml
INFO - Device: CUDA (GPU Name: NVIDIA GeForce RTX 4090)
INFO - Starting 5-Fold Cross-Validation

=== FOLD 1/5 ===
INFO - Fold 1 - Epoch 1/200
INFO - Train - Batch 10/100, Loss: 0.9234
INFO - Train - Epoch Loss: 0.8923, Acc: 0.6234
INFO - Val - Epoch Loss: 0.7234, Acc: 0.7234, AUC: 0.7856
INFO - Fold 1 - Epoch 2/200
... [continues for 200 epochs or until early stopping]
INFO - Early stopping triggered at epoch 35
INFO - Best model saved: models/classification/best.pth
INFO - Best metrics - Acc: 0.8500, AUC: 0.9123

=== FOLD 2/5 ===
... [repeats for 5 folds]

SUMMARY:
  Mean Accuracy: 0.8367 ± 0.0156
  Mean AUC-ROC: 0.9034 ± 0.0089
  Training completed in 42.5 minutes
```

#### Detección (Mask R-CNN)
```
INFO - Training Detection Model (Mask R-CNN)
INFO - Configuration loaded from config.yaml
INFO - Device: CUDA (GPU Name: NVIDIA GeForce RTX 4090)
INFO - Starting Training

INFO - Epoch 1/150
INFO - Batch 5/8, Loss: 2.3456
  - Classifier Loss: 0.8234
  - Bbox Loss: 0.5123
  - Objectness Loss: 0.6234
  - RPN Loss: 0.3865
INFO - Train - Epoch Loss: 2.1234
INFO - Val - Mean IoU: 0.6234, Precision: 0.7123, Recall: 0.6856

INFO - Epoch 2/150
... [continues]

SUMMARY:
  Best Mean IoU: 0.7856
  Best Precision: 0.8234
  Best Recall: 0.8123
  Training completed in 124.3 minutes
```

#### Segmentación (U-Net)
```
INFO - Training Segmentation Model (U-Net)
INFO - Configuration loaded from config.yaml
INFO - Device: CUDA (GPU Name: NVIDIA GeForce RTX 4090)
INFO - Starting 5-Fold Cross-Validation

=== FOLD 1/5 ===
INFO - Fold 1 - Epoch 1/200
INFO - Train - Epoch Loss: 0.4567
INFO - Val - Dice: 0.7234, IoU: 0.5678, Accuracy: 0.8923
INFO - Best Val Dice: 0.7234

... [continues]

SUMMARY (5-Fold):
  Mean Dice: 0.8134 ± 0.0267
  Mean IoU: 0.6923 ± 0.0345
  Training completed in 65.2 minutes
```

---

### 2. Archivos Generados (models/)

#### Estructura
```
models/
├── classification/
│   ├── best.pth                    (7.2 MB) - Mejor checkpoint
│   ├── checkpoint_10.pth
│   ├── checkpoint_20.pth
│   └── checkpoint_30.pth
├── detection/
│   ├── best.pth                    (162 MB) - Mejor checkpoint
│   ├── checkpoint_10.pth
│   ├── checkpoint_20.pth
│   └── checkpoint_30.pth
└── segmentation/
    ├── best_unet.pth               (7.8 MB)
    ├── best_deeplabv3plus.pth      (152 MB)
    ├── checkpoints_unet/
    └── checkpoints_deeplabv3plus/
```

#### Tamaño Total
- Classification: ~30-50 MB
- Detection: ~400-500 MB
- Segmentation: ~300-350 MB
- **Total**: ~750 MB aprox.

---

### 3. Métricas JSON (results/)

#### Classification Metrics
```json
{
  "fold_results": [
    {
      "fold": 0,
      "test_loss": 0.3456,
      "metrics": {
        "accuracy": 0.8600,
        "auc_roc": 0.9234,
        "precision_macro": 0.8450,
        "recall_macro": 0.8500,
        "f1_macro": 0.8475,
        "specificity_macro": 0.9120,
        
        "precision_benign": 0.8700,
        "recall_benign": 0.8600,
        "f1_benign": 0.8650,
        
        "precision_malignant": 0.8200,
        "recall_malignant": 0.8400,
        "f1_malignant": 0.8300,
        
        "precision_negative": 0.8400,
        "recall_negative": 0.8500,
        "f1_negative": 0.8450,
        
        "auc_roc_benign": 0.9345,
        "auc_roc_malignant": 0.9123,
        "auc_roc_negative": 0.9156
      }
    }
  ],
  
  "aggregate_metrics": {
    "mean_accuracy": 0.8567,
    "std_accuracy": 0.0145,
    "mean_auc_roc": 0.9231,
    "std_auc_roc": 0.0089
  },
  
  "confusion_matrices": {
    "fold_0": [
      [215, 18, 12],
      [15, 105, 8],
      [10, 5, 98]
    ]
  }
}
```

#### Detection Metrics
```json
{
  "test_loss": 2.1234,
  "metrics": {
    "tp": 287,
    "fp": 45,
    "fn": 23,
    "precision": 0.8643,
    "recall": 0.9259,
    "f1": 0.8936,
    "mean_iou": 0.7856,
    "mean_iou_std": 0.0456,
    "detections_per_image": {
      "mean": 1.23,
      "std": 0.456,
      "min": 0,
      "max": 3
    },
    "confidence_statistics": {
      "mean_confidence": 0.8756,
      "min_confidence": 0.5234,
      "max_confidence": 0.9987
    }
  }
}
```

#### Segmentation Metrics
```json
{
  "fold_results": [
    {
      "fold": 0,
      "test_loss": 0.2345,
      "metrics": {
        "dice": 0.8234,
        "dice_std": 0.0289,
        "iou": 0.7012,
        "iou_std": 0.0345,
        "accuracy": 0.9456,
        "accuracy_std": 0.0156,
        "sensitivity": 0.8567,
        "specificity": 0.9834,
        "hausdorff_mean": 3.2156,
        "hausdorff_std": 1.2345
      }
    }
  ],
  
  "aggregate_metrics": {
    "mean_dice": 0.8156,
    "std_dice": 0.0267,
    "mean_iou": 0.6923,
    "std_iou": 0.0312
  }
}
```

---

### 4. Visualizaciones (results/)

#### Classification
```
results/classification/
├── confusion_matrix.png          # Heatmap 3x3 (B/M/N)
├── roc_curve_benign.png          # ROC para clase Benign
├── roc_curve_malignant.png       # ROC para clase Malignant
├── roc_curve_negative.png        # ROC para clase Negative
├── roc_curve_combined.png        # One-vs-Rest ROC
├── prediction_distribution.png   # Histogram de confianzas
├── correct_predictions.png       # Grid de aciertos
├── incorrect_predictions.png     # Grid de errores
└── metrics_summary.png           # Resumen tabular
```

#### Detection
```
results/detection/
├── pr_curve_benign.png           # Precision-Recall curve
├── pr_curve_malignant.png        # Precision-Recall curve
├── pr_curve_combined.png         # Curva combinada
├── iou_distribution.png          # Histogram de IoU
├── detection_examples.png        # Bounding boxes visualizados
├── error_examples.png            # False positives/negatives
├── confidence_distribution.png   # Histogram de confidencias
└── metrics_summary.png           # Resumen tabular
```

#### Segmentation
```
results/segmentation/
├── dice_curve.png                # Dice score distribution
├── iou_curve.png                 # IoU distribution
├── accuracy_curve.png            # Pixel accuracy
├── segmentation_correct.png      # Ejemplos correctos
├── segmentation_incorrect.png    # Ejemplos incorrectos
├── error_maps.png                # Mapas de error
├── overlap_visualization.png     # Pred vs GT overlay
└── metrics_summary.png           # Resumen tabular
```

---

## 📈 Métricas Esperadas (Baseline Literature)

### Classification (DenseNet121)
Based on similar studies with breast cancer detection:

| Métrica | Esperado | Mínimo | Máximo |
|---------|----------|--------|--------|
| Accuracy | 85-90% | 82% | 92% |
| AUC-ROC | 0.94-0.96 | 0.90 | 0.97 |
| Sensitivity | 92-95% | 90% | 97% |
| Specificity | 88-92% | 85% | 94% |
| Precision (Malignant) | 80-88% | 75% | 90% |
| F1 (Malignant) | 85-91% | 80% | 93% |

### Detection (Mask R-CNN)
| Métrica | Esperado | Mínimo | Máximo |
|---------|----------|--------|--------|
| mAP (IoU=0.5) | 0.92-0.95 | 0.88 | 0.96 |
| Precision | 90-93% | 87% | 95% |
| Recall | 91-94% | 88% | 96% |
| Mean IoU | 0.85-0.88 | 0.82 | 0.90 |

### Segmentation (U-Net)
| Métrica | Esperado | Mínimo | Máximo |
|---------|----------|--------|--------|
| Dice Score | 0.88-0.92 | 0.85 | 0.93 |
| IoU | 0.80-0.86 | 0.77 | 0.88 |
| Accuracy | 94-97% | 92% | 98% |
| Sensitivity | 88-92% | 85% | 94% |
| Specificity | 95-97% | 93% | 98% |

### Segmentation (DeepLabV3+)
| Métrica | Esperado | Mínimo | Máximo |
|---------|----------|--------|--------|
| Dice Score | 0.90-0.94 | 0.88 | 0.95 |
| IoU | 0.82-0.88 | 0.80 | 0.90 |
| Accuracy | 95-98% | 94% | 99% |
| Sensitivity | 90-94% | 88% | 95% |
| Specificity | 96-98% | 95% | 99% |

---

## 🔍 Interpretación de Resultados

### Señales de Buen Entrenamiento

✅ **Clasificación:**
- Loss disminuye monotónicamente en validación
- AUC-ROC > 0.90
- Sin overfitting (train loss ≈ val loss)
- Early stopping activado entre epoch 20-40

✅ **Detección:**
- mAP crece durante el entrenamiento
- IoU > 0.75
- Detecciones consistentes por imagen
- Pocos false positives

✅ **Segmentación:**
- Dice score > 0.85
- IoU > 0.75
- Máscara predicha alineada con ground truth
- Sin artefactos visuales

### Señales de Problemas

⚠️ **Underfitting:**
- Loss no disminuye (permanece alto)
- Métricas mejoran lentamente
- Train y Val loss similares pero altos

⚠️ **Overfitting:**
- Train loss << Val loss
- Val metrics empeoran después de cierto punto
- Early stopping se activa temprano

⚠️ **Datos deficientes:**
- Métricas inestables entre folds
- Algunos folds mucho peor que otros
- Predictions sesgadas a una clase

---

## 📝 Logging Esperado (Ejemplo Completo)

```
=================================================================
        Bloque2_CNN - Multi-Task Training Pipeline
=================================================================

Configuration:
  - Data Path: ../../DMID_PNG/
  - Metadata: ../../Metadata.xlsx
  - Device: CUDA (NVIDIA GeForce RTX 4090, 24GB)
  - Precision: Mixed (fp32 + fp16)
  - Seed: 42

=================================================================
TASK 1: CLASSIFICATION (DenseNet121)
=================================================================

Loading data...
  - Total images: 511
  - Annotated images: 269
  - Classes: Benign (105, 39.1%), Malignant (66, 24.4%), Negative (98, 30.9%)
  - Train/Val/Test split: 0.70/0.15/0.15

Fold 1/5:
  Epoch 1/200 [=========>                    ] - Loss: 0.8923, Acc: 0.6234
  Epoch 2/200 [=========>                    ] - Loss: 0.7234, Acc: 0.7234
  ... [continues]
  Epoch 35/200 [=========>                    ] - Loss: 0.2345, Acc: 0.9123
  ✓ Early stopping triggered
  Best checkpoint saved: models/classification/best.pth

Fold 2/5:
  ... [similar output]

CLASSIFICATION SUMMARY:
  Mean Accuracy: 0.8567 ± 0.0145
  Mean AUC-ROC: 0.9231 ± 0.0089
  Results saved: results/classification/results.json
  Training time: 42.5 minutes

=================================================================
TASK 2: DETECTION (Mask R-CNN)
=================================================================

Loading data...
  - Total images: 269 (with masks)
  - Train/Val/Test split: 0.70/0.15/0.15

Training:
  Epoch 1/150 [>                             ] - Loss: 2.3456
  Epoch 2/150 [>                             ] - Loss: 2.1234
  ... [continues]
  Epoch 45/150 [==================>          ] - Loss: 0.5678, mAP: 0.7856
  ✓ Early stopping triggered
  Best checkpoint saved: models/detection/best.pth

DETECTION SUMMARY:
  Mean IoU: 0.7856
  Precision: 0.8643
  Recall: 0.9259
  F1 Score: 0.8936
  Results saved: results/detection/results.json
  Training time: 124.3 minutes

=================================================================
TASK 3: SEGMENTATION (U-Net + DeepLabV3+)
=================================================================

Model 1: U-Net (1.9M parameters)

Loading data...
  - Total images with masks: 269

Fold 1/5:
  Epoch 1/200 - Loss: 0.4567, Dice: 0.7234
  ... [continues]
  ✓ Early stopping triggered

Model 2: DeepLabV3+ (39.5M parameters)

Fold 1/5:
  Epoch 1/200 - Loss: 0.3456, Dice: 0.7856
  ... [continues]
  ✓ Early stopping triggered

SEGMENTATION SUMMARY:
  U-Net Mean Dice: 0.8156 ± 0.0267
  DeepLabV3+ Mean Dice: 0.8456 ± 0.0189
  Results saved: results/segmentation/results_*.json
  Training time: 189.7 minutes (both models, 5-fold)

=================================================================
PIPELINE COMPLETED
=================================================================

Total Models Trained: 5 (1 classification, 1 detection, 2 segmentation × 5 folds)
Total Training Time: 356.5 minutes (~6 hours)
Total Checkpoints: 15
Total Results: 3 JSON files + visualizations

Next Steps:
  1. Review results: cat results/*/results.json
  2. Generate LaTeX tables with metrics
  3. Include visualizations in final report
  4. Complete Resultados section in proyectofinal.tex

=================================================================
```

---

## 💾 Storage Requirements

| Componente | Tamaño | Duración |
|------------|--------|----------|
| Dataset (DMID_PNG) | ~500 MB | Permanente |
| Metadata (xlsx) | ~2 MB | Permanente |
| Model Checkpoints | ~750 MB | Temporal (después usar best.pth) |
| Results (JSON + PNG) | ~50-100 MB | Permanente (para reporte) |
| Logs (si se guardan) | ~10-20 MB | Temporal |
| **Total** | **~1.3 GB** | **~1 GB permanente** |

---

## 📞 FAQ

**P: ¿Cuánto tarda el entrenamiento?**  
R: ~6 horas total (GPU RTX 4090). En GPU inferior: 10-15 horas. En CPU: 48+ horas.

**P: ¿Puedo interrumpir el entrenamiento?**  
R: Sí, usa Ctrl+C. El último checkpoint se guardará. Puedes continuar desde ahí.

**P: ¿Qué significa "Early stopping triggered"?**  
R: La validación no mejoró en los últimos N epochs. El modelo se consideró completado.

**P: ¿Por qué algunos folds son peor que otros?**  
R: Normal en k-fold CV. Algunos folds pueden tener distribuciones diferentes.

**P: ¿Necesito GPU?**  
R: Altamente recomendado. Sin GPU tomará 48+ horas.

**P: ¿Puedo usar los checkpoints después?**  
R: Sí. Usa `torch.load()` para cargar best.pth en inference.

---

**Última actualización**: May 4, 2026  
**Versión**: 1.0
