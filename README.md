# Mango Leaf Disease Classification & Explainability Prototype

An interactive, thesis-grade deep learning prototype comparing **GourNet V1 (Baseline)** against **Enhanced GourNet V2** for automated mango leaf disease classification with **Grad-CAM Explainable AI (XAI)** and botanical input validation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Mango (*Mangifera indica*) is one of the most economically critical fruit crops globally. Foliar diseases can severely devastate orchard productivity, crop yield, and tree vitality. 

This research investigates the architectural constraints of the recently proposed **GourNet** architecture (*Alam et al., 2026*), which reproduces an 8-class classification model using a traditional Convolutional Neural Network (CNN) with a 589,888-parameter dense bottleneck. 

We propose **Enhanced GourNet V2**, an architectural redesign that:
- **Reduces total parameters by 85.6%** (from 683,656 down to 98,280 parameters).
- **Eliminates the dense bottleneck** by replacing `Flatten()` with `GlobalAveragePooling2D()`.
- **Stabilizes feature distributions** via `GroupNormalization(groups=8)` to avoid batch-size dependency.
- **Enforces spatial regularization** using `SpatialDropout(0.2)` and `Head Dropout(0.4)`.
- **Improves test accuracy to 97.12%** (exceeding the paper's reported 97.00% benchmark).
- **Integrates Grad-CAM visual heatmaps** to provide explainable, localized lesion attention.
- **Implements a Botanical Foliage Gatekeeper** to prevent Out-of-Distribution (OOD) false predictions on non-leaf images.

---

## Architectural Comparison (V1 vs V2)

| Architectural Component | GourNet V1 (Baseline) | GourNet V2 (Enhanced) | Impact & Rationale |
|---|---|---|---|
| **Base Architecture** | Alam et al. (2026) | Thesis Improvement | Scalable edge-friendly redesign |
| **Total Parameters** | **683,656** | **98,280** | **-85.6% footprint reduction** (~400 KB weights) |
| **Transition to Head** | `Flatten()` (9,216 features) | `GlobalAveragePooling2D()` (64 features) | Eliminates 589k-parameter dense bottleneck & overfitting |
| **Normalization** | None | `GroupNormalization(groups=8)` | Prevents batch-statistic collapse on small batches |
| **Regularization** | Data Augmentation only | SpatialDropout(0.2) + Head Dropout(0.4) | Prevents co-adaptation and logit saturation |
| **Target Conv Layer (Grad-CAM)** | `Convolution-4` | `Block4_Conv` | Clear, lesion-focused class activation maps |
| **Test Accuracy** | ~97.00% (Paper reported) | **97.12%** (5-seed: 96.01% ± 1.95%) | Matches/exceeds accuracy with ~85% fewer weights |

---

## Target Dataset & Disease Classes

The models were evaluated on the **MangoLeafBD** benchmark dataset (4,000 leaf images across 8 balanced classes, resized to $224 \times 224 \times 3$):

1. **Anthracnose** (*Colletotrichum gloeosporioides*)
2. **Bacterial Canker** (*Xanthomonas citri pv. mangiferaeindicae*)
3. **Cutting Weevil** (*Deporaus marginatus*)
4. **Die Back** (*Lasiodiplodia theobromae*)
5. **Gall Midge** (*Procontarinia matteiana*)
6. **Healthy** (Healthy Foliage)
7. **Powdery Mildew** (*Oidium mangiferae*)
8. **Sooty Mould** (*Meliola mangiferae*)

---

## Key Prototype Features

- **Side-by-Side Comparative Diagnosis**: Simultaneously runs inference on both GourNet V1 (Baseline) and GourNet V2 (Enhanced) for direct performance comparison.
- **Explainable AI (Grad-CAM)**: Generates high-resolution class activation heatmaps showing where each model directs its visual attention, with configurable opacity and colormaps (`jet`, `viridis`, `plasma`, `inferno`, `turbo`).
- **Leaf-Only Validation Guard**: Multi-stage computer vision gatekeeper analyzing chlorophyll absorption ($G > 0.9R$ and $G > 1.1B$), central foliage coverage, and texture variation to reject non-leaf uploads (e.g. photos of cars, faces, furniture, or outdoor backgrounds).
- **Strictness Control**: Slider in the sidebar (`Permissive`, `Balanced`, `Strict`) to adjust input validation thresholds.
- **Model Calibration & OOD Warning**: Flags low-confidence predictions ($< 40\%$) to alert users of potential out-of-distribution or atypical specimens.
- **Dynamic Sample Loading**: Drop any `.jpg`, `.jpeg`, `.png`, or `.webp` file into the `samples/` directory, and it instantly appears in the demo selector dropdown.

---

## Project Structure

```text
├── Gournet_v1/
│   ├── gournet_model/             # Saved Keras 3 model directory (683k params)
│   │   ├── config.json
│   │   ├── metadata.json
│   │   └── model.weights.h5
│   └── gournet_v1.ipynb           # Baseline training and reproduction notebook
├── Gournet_v2/
│   ├── gournet_v2_best/           # Enhanced GourNet v2 checkpoint (98k params)
│   │   ├── config.json
│   │   ├── metadata.json
│   │   └── model.weights.h5
│   ├── gournet_v2_model/          # Final epoch model checkpoint
│   └── enhanced-gournetv2-12-mseed.ipynb # Enhanced training & multi-seed notebook
├── samples/                       # Pre-loaded demo test specimens
│   ├── anthracnose_sample.jpg
│   ├── bacterial_canker_sample.jpg
│   ├── healthy_sample.jpg
│   └── powdery_mildew_sample.jpg
├── app.py                         # Streamlit interactive web application
├── model_utils.py                 # Preprocessing, inference, Grad-CAM & validation
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Comprehensive project documentation
```

---

## Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/90377Sednaaa/mango-leaf-disease-classification.git
cd mango-leaf-disease-classification
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Deployment Guide (Streamlit Community Cloud)

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account (`https://github.com/<your-username>/<repo-name>`).
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Select this repository and specify:
   - **Main file path**: `app.py`
5. Click **"Deploy!"**. 
   Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, and host your live application with a public URL.

---

## Citations & References

1. **Alam, M. A., et al. (2026)**. *GourNet: A CNN-Based Model for Mango Leaf Disease Detection*. Proceedings of the International Conference on Advanced Computing and Systems (AdComSys 2025). arXiv:2604.27764.
2. **Selvaraju, R. R., et al. (2017)**. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE International Conference on Computer Vision (ICCV), pp. 618-626.
3. **Lin, M., Chen, Q., & Yan, S. (2013)**. *Network In Network*. arXiv:1312.4400.
4. **Guo, C., et al. (2017)**. *On Calibration of Modern Neural Networks*. International Conference on Machine Learning (ICML), PMLR 70:1321-1330.
