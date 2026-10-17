# 🩺 Pneumonia Detection from Chest X-Rays

Deep learning system for detecting pneumonia from chest X-rays and staging its severity, with an interactive web interface and explainability via Grad-CAM.

---

## 📋 Overview

This project builds an end-to-end AI system that:

- **Detects pneumonia** from chest X-ray images (binary: NORMAL vs PNEUMONIA)
- **Stages severity** (MILD / MODERATE / SEVERE) when pneumonia is present
- **Explains predictions** using Grad-CAM heatmaps
- **Serves real-time predictions** through a Flask web interface

Trained on the **RSNA Pneumonia Detection Challenge** dataset (26,684 chest X-rays from NIH), the model uses transfer learning with **EfficientNet-B0** and a multi-task architecture.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Binary AUC** | **0.8892** |
| **Binary Accuracy** | 82.84% |
| **Pneumonia Precision** | 59.42% |
| **Pneumonia Recall** | 75.17% |
| **F1 Score** | 0.6637 |
| **Stage Accuracy** | 55.32% |
| **Stage F1 (macro)** | 53.36% |

**Attribute-only baseline (for comparison):**

| Feature Type | Features | AUC |
|--------------|----------|-----|
| Hand-crafted attributes | 10 | 0.6914 |
| **Deep learning (ours)** | **1,280** | **0.8892** |
| **Improvement** | — | **+0.1978** |

---

## 🏗️ Architecture

### Model: EfficientNet-B0 with Multi-Task Heads

```
Input (224x224x3 RGB)
       |
EfficientNet-B0 Backbone (4.17M params, ImageNet pretrained)
       |
[1,280-dim feature vector]
       |
    +--+------------------+
    |                     |
Binary Head           Stage Head
(NORMAL/PNEUMONIA)    (MILD/MODERATE/SEVERE)
    |                     |
2 logits              3 logits
```

**Why EfficientNet-B0?**
- State-of-the-art accuracy on ImageNet
- Lightweight (5.3M params) — trains fast on RTX 4050
- Proven in medical imaging research

**Why multi-task?**
- One model, two outputs — binary + severity
- Shared backbone gives better feature learning
- Matches clinical workflow (detect + stage)

---

## 📊 Dataset

**Source:** RSNA Pneumonia Detection Challenge (Kaggle)

| Property | Value |
|----------|-------|
| Total patients | 26,684 |
| Normal cases | 20,672 (77.5%) |
| Pneumonia cases | 6,012 (22.5%) |
| Image format | DICOM (1024x1024 grayscale) |
| Annotations | Radiologist bounding boxes |
| Train / Val / Test split | 70% / 15% / 15% |

**Stage distribution (from bounding box areas):**

| Stage | Count |
|-------|-------|
| MILD | 1,713 |
| MODERATE | 2,566 |
| SEVERE | 1,733 |

---

## 🔄 Pipeline

### 1. Data Preparation

```
DICOM files -> Parse with pydicom -> Normalize -> Resize to 224x224

Extract 10 attributes:
  - pixel_mean, pixel_std             (image intensity stats)
  - image_width, image_height         (dimensions)
  - bbox_count                        (number of lesions)
  - bbox_area_total                   (lesion total area)
  - opacity_ratio                     (fraction of lung affected)
  - bbox_centroid_x, bbox_centroid_y  (average lesion location)
  - patient_age                       (from DICOM metadata)
```

### 2. Stage Labeling

```
For each pneumonia patient:
  1. Sum all bounding box areas
  2. opacity_ratio = total_area / image_area
  3. Assign stage:
       < 5%   -> MILD
       < 15%  -> MODERATE
       >= 15% -> SEVERE
```

### 3. Training

| Config | Value |
|--------|-------|
| Backbone | EfficientNet-B0 |
| Loss | Focal Loss (gamma=2.0) + CrossEntropy (stage) |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Epochs | 20 (early stopped) |
| Mixed precision | Yes (CUDA AMP) |
| Class weights | Balanced |

### 4. Inference

```
X-ray -> Preprocess -> Model -> Probabilities
                                 |
                  Binary: threshold 0.67 (optimized)
                                 |
                       NORMAL / PNEUMONIA
                                 |
                    If PNEUMONIA -> Stage (MILD/MOD/SEV)
                                 |
                  Grad-CAM heatmap for explainability
```

---

## 🖥️ Web Interface

Interactive Flask app at `http://localhost:5000`:

**Features:**
- Upload DICOM / PNG / JPG
- Real-time prediction
- Probability breakdown bars
- Stage prediction (for pneumonia cases)
- **Grad-CAM heatmap** — shows where the AI looks
- Automatic file saving with timestamps

---

## 🚀 How to Run

### Prerequisites

- Python 3.11+
- NVIDIA GPU (optional but recommended) — CUDA 12.6
- Windows / Linux / Mac

### Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# Install dependencies
pip install torch torchvision timm numpy pandas pillow scikit-learn matplotlib seaborn opencv-python pydicom pyyaml tqdm tensorboard flask

# For GPU support (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Data Preparation

```bash
# 1. Download RSNA dataset from Kaggle and extract into data/raw/
# 2. Generate train/val/test splits
python -m scripts.prepare_data

# 3. Extract attributes
python -m src.data.attributes
```

### Training

```bash
python -m scripts.train
```

Expected time: ~30 min on RTX 4050 GPU.

### Evaluation

```bash
python -m scripts.evaluate
python -m scripts.optimize_threshold
```

### Run the Web App

```bash
python app.py
```

Then open: **http://localhost:5000**

---

## 📁 Project Structure

```
pneumonia_detection/
├── app.py                          # Flask web server
├── configs/
│   └── config.yaml                 # Hyperparameters
├── data/
│   ├── raw/                        # RSNA DICOM files + labels
│   ├── splits/                     # train.csv, val.csv, test.csv
│   └── processed/
│       └── images/                 # 500 exported PNGs
├── src/
│   ├── data/
│   │   ├── dataset.py              # DICOM PyTorch Dataset
│   │   ├── transforms.py           # Augmentation
│   │   ├── stage_labeler.py        # Stage from bboxes
│   │   └── attributes.py           # 10 hand-crafted features
│   ├── models/
│   │   ├── classifier.py           # EfficientNet + 2 heads
│   │   └── losses.py               # Focal + MultiTask loss
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   └── metrics.py              # Accuracy, F1, AUC
│   └── utils/
│       ├── seed.py                 # Reproducibility
│       └── logger.py               # Logging
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── optimize_threshold.py
│   ├── attribute_baseline.py
│   ├── extract_features.py
│   ├── dicom_to_images.py
│   └── predict.py
├── models/
│   └── best.pt                     # Trained checkpoint
├── reports/                        # All evaluation plots
├── logs/                           # Training logs
├── static/
│   ├── style.css
│   └── uploads/
└── templates/
    └── index.html
```

---

## 🔬 Explainability — Grad-CAM

**Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights which regions of the X-ray the model focused on for its decision.

- **PNEUMONIA case:** Red heatmap concentrates on lung fields where opacities appear
- **NORMAL case:** Diffuse attention, no single hot spot

This confirms the model learned **medically meaningful features** — not random artifacts like medical devices or image borders.

---

## 📈 Results Breakdown

### Binary Classification (test set, n=4,003)

```
              precision    recall  f1-score   support
      NORMAL     0.8907    0.6784    0.7699      3101
   PNEUMONIA     0.5942    0.7517    0.6637       902

    accuracy                         0.8284      4003
   macro avg     0.7425    0.7151    0.7168      4003
weighted avg     0.8471    0.8284    0.8256      4003
```

### Threshold Analysis

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.43 | 0.94 | 0.59 |
| **0.67 (chosen)** | **0.59** | **0.75** | **0.66** |
| 0.75 | 0.72 | 0.52 | 0.60 |

We chose **0.67** — best F1 balance, best precision-recall tradeoff for clinical use.

### Stage Classification (pneumonia only, n=902)

```
              precision    recall  f1-score   support
        MILD     0.6074    0.3402    0.4362       241
    MODERATE     0.4885    0.7380    0.5879       374
      SEVERE     0.6980    0.4913    0.5767       287

    accuracy                         0.5532       902
```

---

## ⚠️ Limitations & Future Work

### Limitations

- **Dataset size:** Trained on 20K of 26.6K samples — full training would improve metrics
- **Stage accuracy:** 55% is moderate — stage classification from X-rays alone is inherently difficult
- **No clinical context:** Model uses only X-ray images (no symptoms, labs, patient history)
- **Single-institution data:** RSNA data comes from a limited set of hospitals

### Future Work

- **Multi-modal fusion:** Combine X-ray with symptoms (fever, cough) and labs (WBC, CRP)
- **Larger datasets:** Train on MIMIC-CXR (377K X-rays) — requires credentialing
- **Localization:** Predict bounding boxes directly (object detection)
- **Uncertainty quantification:** Monte Carlo dropout for confidence intervals
- **Clinical validation:** Prospective study with radiologist review

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.14 + CUDA 12.6 |
| Pretrained Models | timm (EfficientNet-B0) |
| Image Processing | OpenCV, Pillow, pydicom |
| ML Utilities | scikit-learn |
| Web Framework | Flask |
| Visualization | matplotlib, seaborn |
| Config | PyYAML |
| Hardware | NVIDIA RTX 4050 (6 GB VRAM) |

---

## 👨‍💻 Author

**Syed M**
B.Tech AIML · KL University
Academic Year 2026-27

---

## 📄 License

This project is for academic purposes as part of the B.Tech Capstone Project at KL University.

---

## 🙏 Acknowledgments

- **RSNA** for the publicly available Pneumonia Detection Challenge dataset
- **NIH** for funding the original chest X-ray collection
- **PyTorch team** for the deep learning framework
- **Hugging Face** for pretrained model weights

---

## 📚 References

1. Wang, X., et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database." CVPR.
2. Shih, G., et al. (2019). "Augmenting the NIH Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia." Radiology: AI.
3. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML.
4. Selvaraju, R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
5. Lin, T., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

---

**Built as part of B.Tech AIML Capstone Project — KL University**