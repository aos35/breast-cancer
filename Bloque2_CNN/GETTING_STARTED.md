# 🚀 GETTING STARTED - 5 Minute Quick Start

**Goal**: Get training running in < 5 minutes  
**Platform**: Windows, macOS, Linux  
**Prerequisites**: Python 3.8+, pip

---

## Step 1: Navigate to Project (30 seconds)

```bash
cd c:\Users\aleja\Desktop\3ANYO\redesNeuronales\proyecto\breast-cancer\Bloque2_CNN
```

---

## Step 2: Check Environment (1 minute)

```bash
# Check Python
python --version

# Check CUDA (optional - CPU works too, just slower)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```
Python 3.x.x
CUDA available: True   (or False - both OK)
```

---

## Step 3: Install Dependencies (2 minutes)

```bash
python quick_start.py install
```

**What it does:**
- Installs PyTorch, torchvision, segmentation-models, scikit-learn, etc.
- Takes ~2-3 minutes depending on connection

**Tip:** If installation fails, run:
```bash
pip install -r requirements.txt
```

---

## Step 4: Verify Setup (1 minute)

```bash
python quick_start.py setup
```

**Expected output:**
```
✓ Python found: Python 3.x.x
✓ PyTorch found: 2.x.x
✓ CUDA available - GPU: NVIDIA GeForce RTX 4090
✓ Environment ready for training
```

---

## Step 5: Start Training! (< 30 seconds)

### Option A: Train Classification (Fast, ~30 min)
```bash
python quick_start.py train --task classification
```

### Option B: Train Detection (Medium, ~2 hours)
```bash
python quick_start.py train --task detection
```

### Option C: Train Segmentation (Medium-Fast, ~1.5 hours)
```bash
python quick_start.py train --task segmentation --model unet
```

### Option D: Train Everything (Full Pipeline, ~6 hours)
```bash
python quick_start.py train --task all
```

---

## 📊 During Training - What to Expect

**Classification Training (should look like this):**
```
INFO - Training Classification Model (DenseNet121)
INFO - Device: CUDA (GPU Name: NVIDIA GeForce RTX 4090)
INFO - Starting 5-Fold Cross-Validation

=== FOLD 1/5 ===
INFO - Fold 1 - Epoch 1/200
INFO - Train - Batch 10/100, Loss: 0.9234
INFO - Train - Epoch Loss: 0.8923, Acc: 0.6234
INFO - Val - Epoch Loss: 0.7234, Acc: 0.7234, AUC: 0.7856
... [repeats for 200 epochs or until early stopping]
```

**Signs it's working:**
- ✅ Loss decreases over time
- ✅ Accuracy/AUC increases
- ✅ No error messages
- ✅ GPU memory usage stable

**If you see errors:**
- ❌ CUDA out of memory → Reduce batch_size in config.yaml
- ❌ Data not found → Check ../../DMID_PNG/ path
- ❌ Module not found → Run `python quick_start.py install` again

---

## ✅ After Training Completes

**1. Check Results:**
```bash
# See metrics
cat results/classification/results.json

# See visualizations
ls -la results/classification/
```

**2. Run Evaluation:**
```bash
python quick_start.py eval --task classification
```

**3. View Output:**
- JSON metrics in `results/classification/results.json`
- Figures in `results/classification/`.png files
- Best model in `models/classification/best.pth`

---

## 📚 Learn More

| Want to... | Read... |
|-----------|---------|
| Understand what train.py does | TRAINING_GUIDE.md (section "Training") |
| See all configuration options | config.yaml |
| Learn about expected metrics | EXPECTED_OUTPUTS.md |
| Understand the pipeline | IMPLEMENTATION_SUMMARY.md |
| Troubleshoot problems | TRAINING_GUIDE.md (section "Troubleshooting") |
| See code structure | README.md or INDICE.md |

---

## 🎯 Common Commands

```bash
# Setup & Install
python quick_start.py setup                 # Check environment
python quick_start.py install               # Install dependencies

# Training
python quick_start.py train --task all                           # Everything
python quick_start.py train --task classification               # Just classification
python quick_start.py train --task detection                    # Just detection
python quick_start.py train --task segmentation --model unet    # Just U-Net

# Evaluation
python quick_start.py eval --task all       # Evaluate everything
python quick_start.py eval --task classification  # Evaluate classification

# Full Pipeline
python quick_start.py full --task classification   # Train + Evaluate
```

---

## 💾 What Gets Saved?

**During training:**
- Checkpoints in `models/[task]/` (every 10 epochs)
- Best model in `models/[task]/best.pth`
- Logs printed to terminal (can redirect: `> train.log`)

**During evaluation:**
- Metrics JSON in `results/[task]/results.json`
- Figures in `results/[task]/`.png files
- CSV exports in `results/[task]/`.csv files

**You can delete:**
- `models/[task]/checkpoint_*.pth` (keep best.pth only)
- Logs (they're just text)

**Keep for LaTeX:**
- Metrics JSON (for tables)
- Figures (PNG files for report)

---

## ⏱️ Time Estimates

| Task | Time (GPU) | Time (CPU) | Output Size |
|------|-----------|-----------|------------|
| Classification | 30-45 min | 4-6 hours | 7 MB |
| Detection | 2-3 hours | 24 hours | 160 MB |
| U-Net Segmentation | 1-2 hours | 12-16 hours | 8 MB |
| DeepLabV3+ Segmentation | 2-3 hours | 20-30 hours | 150 MB |
| **All (5-fold)** | **6-8 hours** | **60-100 hours** | **400 MB** |

---

## 🆘 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `python quick_start.py install` |
| `CUDA out of memory` | Edit config.yaml, reduce batch_size |
| `Data not found` | Check `../../DMID_PNG/` exists |
| Training very slow | Check if using CPU (install CUDA) |
| Process hangs | Ctrl+C to stop, last checkpoint saved |
| Can't find results | Check `results/` directory |

---

## 🎓 What Happens Inside?

When you run `python quick_start.py train --task classification`:

1. **Data Loading** (30 sec)
   - Reads metadata.xlsx
   - Maps class labels (Benign→0, Malignant→1, Negative→2)
   - Splits into 5 folds with stratification
   - Creates data loaders with augmentation

2. **Model Initialization** (10 sec)
   - Creates DenseNet121 architecture
   - Loads from PyTorch pretrained weights
   - Moves to GPU if available
   - Sets up optimizer (Adam) and scheduler

3. **K-Fold Training Loop** (varies)
   - For each of 5 folds:
     - Train for up to 200 epochs
     - After each epoch: validate and check if improved
     - If no improvement for 20 epochs: stop (early stopping)
     - Save best model

4. **Evaluation & Metrics** (varies)
   - Load best model
   - Run on test set
   - Calculate AUC-ROC, accuracy, precision, recall, F1
   - Generate confusion matrix
   - Save results to JSON

---

## 📞 Quick Reference Card

**Current Directory:**
```
breast-cancer/Bloque2_CNN/  ← You should be here
```

**Main Scripts:**
```
quick_start.py              ← Use this to train & evaluate
src/train.py                ← Advanced: direct access
src/evaluate.py             ← Advanced: direct access
```

**Configuration:**
```
config.yaml                 ← Change hyperparameters here
```

**After Training:**
```
models/                     ← Checkpoints saved here
results/                    ← Metrics & visualizations here
```

---

## ✨ Pro Tips

1. **Save terminal output:**
   ```bash
   python quick_start.py train --task classification > train.log 2>&1
   ```

2. **Train in background:**
   ```bash
   nohup python quick_start.py train --task all &
   ```

3. **Check GPU usage during training:**
   ```bash
   # In another terminal
   nvidia-smi                 # Watch GPU memory
   ```

4. **Reduce batch size if memory issues:**
   ```bash
   # Edit config.yaml
   classification:
     batch_size: 16          # Reduce from 32
   ```

5. **Speed up first run:**
   ```bash
   # Train classification first (fast) to test setup
   python quick_start.py train --task classification
   ```

---

## 🎯 Success Checklist

After completing these steps, you should have:

- [ ] Environment set up (`python quick_start.py setup` ✓)
- [ ] Dependencies installed (`python quick_start.py install` ✓)
- [ ] Training started (`python quick_start.py train --task [X]` ✓)
- [ ] Logs appearing in terminal (loss/accuracy showing)
- [ ] Checkpoints being saved (check `models/` directory)
- [ ] Training completes without errors
- [ ] Results generated (check `results/` directory)
- [ ] JSON metrics created (check `results/[task]/results.json`)
- [ ] Visualizations generated (check `.png` files)

---

## 🚀 Next: What to Do With Results

Once you have results (JSON + PNG files):

1. **Create LaTeX tables** from JSON metrics
2. **Insert figures** (PNG files) into proyectofinal.tex
3. **Write Resultados section** with findings
4. **Discuss in Discusión** section

See: [EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md) for example metrics and interpretations.

---

## 📚 Full Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training & eval guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical architecture
- **[EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md)** - Example outputs & metrics
- **[PHASE2_COMPLETION.md](PHASE2_COMPLETION.md)** - Completion checklist
- **[README.md](README.md)** - Project overview

---

**Ready?** Run this now:
```bash
python quick_start.py setup
```

Then:
```bash
python quick_start.py install
```

Finally:
```bash
python quick_start.py train --task classification
```

Go! 🚀

---

**Last Updated**: May 4, 2026  
**Version**: 1.0
