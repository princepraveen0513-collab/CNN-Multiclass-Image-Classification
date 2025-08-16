# CNN Multiclass Image Classification with PyTorch — 3‑Part Project (Step‑by‑Step)

This repo contains a 3‑stage workflow for building and improving a **Convolutional Neural Network (CNN)** image classifier using **PyTorch**. Follow **Steps 1 → 2 → 3** to reproduce results and iterate on model quality.

> **Dataset link:**
> **`(https://www.kaggle.com/datasets/ryanbadai/clothes-dataset)`**

---

## 📁 Repository Layout
```
.
├── Part1_DeepLearning_CNN.ipynb     # Step 1: Baseline CNN
├── Part2_DeepLearning_CNN.ipynb     # Step 2: Data imports & preprocessing
├── Part3_DeepLearning_CNN.ipynb     # Step 3: Optimization & training strategy
└── README.md                         # This combined guide
```

---

## 🧰 Environment & Setup

**Dependencies:** `torch`, `torchvision`, `pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install:
```bash
pip install -U torch torchvision torchaudio pillow numpy pandas scikit-learn matplotlib
```

(Optional) Set seeds for reproducibility in each notebook (Python/NumPy/PyTorch).

**Expected data layout (typical):**
```
data/
├── train/
│   ├── class_0/  ├── class_1/  ├── ... ├── class_N/
├── val/
│   ├── class_0/  ├── class_1/  ├── ... ├── class_N/
└── test/
    ├── class_0/  ├── class_1/  ├── ... ├── class_N/
```
> If your notebooks use different path variables (e.g., `data_dir`, `train_dir`, `valid_dir`, `test_dir`), update those first.

**Common transforms (from the notebooks):**
- `Resize`, `ToTensor` (+ `Normalize` if configured)
- Light augmentation: `RandomHorizontalFlip`, `RandomRotation`

---

## ✅ Step 1 — Baseline CNN (Part 1)

**Notebook:** `Part1_DeepLearning_CNN.ipynb`  
Goal: Build a working baseline with a custom **`CNN_Model`** and confirm the training loop, dataloaders, and evaluation are correct.

**What happens here**
- Load data and apply basic transforms.
- Define **`CNN_Model`** (custom `nn.Module` with `Conv2d`, `ReLU`, `MaxPool2d`, optional `BatchNorm`/`Dropout`, and a linear head).
- Train for a few epochs; print training/validation metrics.
- Sanity‑check overfitting/underfitting and save notes for improvements.

**Run:**
```bash
jupyter notebook "Part1_DeepLearning_CNN.ipynb"
```

---

## ⚙️ Step 2 — Data Imports & Pre‑Processing (Part 2)

**Notebook:** `Part2_DeepLearning_CNN.ipynb`  
Goal: Stabilize the **data pipeline** and reuse the best‑performing baseline. Keep transforms consistent and verify splits.

**What happens here**
- Re‑load data with the same (or slightly refined) transforms.
- Ensure consistent **train/val/test** splits and class mapping.
- Reuse or slightly tweak **`CNN_Model`** from Step 1.
- Confirm metrics reproducibility; record best baseline as a reference.

**Run:**
```bash
jupyter notebook "Part2_DeepLearning_CNN.ipynb"
```

---

## 🚀 Step 3 — Optimization & Training Strategy (Part 3)

**Notebook:** `Part3_DeepLearning_CNN.ipynb`  
Goal: Improve performance/robustness with **training strategy** changes.

**Potential changes explored**
- Optimizer/lr updates (`SGD`/`Adam`, tune **lr**).
- Regularization: `Dropout`, **weight decay**.
- (Optional) LR schedules: `StepLR`, `ReduceLROnPlateau`, `CosineAnnealingLR`, `OneCycleLR`.
- Early stopping/checkpointing if included.
- Track best epoch and compare to Step 1/2.

**Run:**
```bash
jupyter notebook "Part3_DeepLearning_CNN.ipynb"
```

---

## 🧪 Metrics & Reporting

Each notebook prints metrics to the console (accuracy, and scaffolding for precision/recall/F1).  


**Tips**
- Save a **confusion matrix** and `classification_report` for per‑class insight.
- Log loss/accuracy curves per epoch and keep them in an `images/` folder.

---

## 🔧 Implementation Notes

- **Custom model:** notebooks define a `CNN_Model` (custom `nn.Module`); variants appear across steps.
- **Transforms observed:** `Resize`, `ToTensor`, with optional `Normalize`; light augments: `RandomHorizontalFlip`, `RandomRotation`.
- **Device:** CUDA if available, otherwise CPU.
- **Loss/Optimizers:** typical setup is `CrossEntropyLoss` with `SGD` or `Adam`.

---

## 🔗 Dataset Link (if provided)
https://www.kaggle.com/datasets/ryanbadai/clothes-dataset

---

## 🧭 Next Steps

- Add richer augmentation (`ColorJitter`, `RandomResizedCrop`), then re‑benchmark.
- Try **weight decay** and **LR schedules**; compare validation curves.
- Consider **transfer learning** (e.g., ResNet18/MobileNetV2) as a strong baseline.
- Export best checkpoints to `checkpoints/` and embed plots in `images/` within this README.
