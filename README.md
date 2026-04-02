# 🐶 Dog Breed Classification Using CNN

<p align="center">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%2FKeras-FF6F00?style=flat-square&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Big%20Data-PySpark-E25A1C?style=flat-square&logo=apache-spark" />
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Dataset-Stanford%20Dogs-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Classes-120%20Breeds-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Test%20Accuracy-82.58%25-brightgreen?style=flat-square" />
</p>

> **A scalable deep learning pipeline classifying 120 dog breeds across 20,580 images using a custom 5-block CNN built with TensorFlow/Keras and a PySpark-powered preprocessing pipeline — achieving 82.58% test accuracy.**

---

## 📌 Problem Statement

Manual dog breed identification is error-prone and inconsistent, leading to inadequate veterinary care and adoption mismatches. This project builds a reliable, scalable image classification system that automatically identifies dog breeds from images, leveraging CNNs for accuracy and PySpark for big-data-scale preprocessing.

---

## ⚡ Key Highlights

| | |
|---|---|
| 🐕 **Classes** | 120 dog breeds |
| 🖼️ **Dataset** | 20,580 images (Stanford Dogs Dataset) |
| 🎯 **Test Accuracy** | **82.58%** (100 epochs) |
| 📉 **Test Loss** | **0.6461** |
| 🔥 **Architecture** | 5-block CNN · 19 layers · 16→256 progressive filters |
| ⚙️ **Big Data** | PySpark DataFrames + custom UDFs for distributed preprocessing |
| 🧪 **Split** | 14,406 train / 6,174 test (70/30 stratified) |

---

## 🛠️ Tech Stack

```
Deep Learning   : TensorFlow / Keras (Sequential CNN)
Big Data        : PySpark 3.5.1 (SparkSession, DataFrames, UDFs, VectorAssembler)
Image Processing: OpenCV (cv2), PIL (Pillow)
Data Science    : NumPy, Pandas, scikit-learn, tqdm
Visualization   : Matplotlib, Seaborn
Dataset         : Stanford Dogs Dataset (120 breeds, ~20,500 images)
Environment     : Google Colab / Jupyter Notebook
```

---

## 📊 Dataset

**Source:** [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

| Property | Detail |
|----------|--------|
| Total images | 20,580 |
| Number of breeds | 120 |
| Images per breed | ~148 – 252 |
| Average image size | ~360 × 236 px |
| Dominant aspect ratio | ~0.655 |
| Format | JPG, organized by breed subfolder |

**Top breeds by image count (sample):**

| Breed | Count |
|-------|-------|
| Maltese dog | 252 |
| Afghan hound | 239 |
| Scottish deerhound | 232 |
| Pomeranian | 219 |
| Irish wolfhound | 218 |

---

## 🔄 Pipeline

```
Stanford Dogs Dataset (URL)
        │
        ▼
┌───────────────────────┐
│  Data Collection      │  requests + tarfile extraction
│                       │  20,580 JPG images across 120 breed folders
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  PySpark Exploration  │  SparkSession · createDataFrame
│                       │  regexp_extract breed labels from file paths
│                       │  Custom UDFs → height, width, aspect_ratio columns
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Preprocessing        │  Filter: height ≥ 128, width ≥ 128
│                       │  Aspect ratio filter: 0.6 – 1.34
│                       │  Grayscale (PIL LANCZOS) → resize 128×128
│                       │  Normalize pixels to [0, 1] (÷ 255)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Train / Test Split   │  70/30 stratified split (scikit-learn)
│                       │  14,406 train · 6,174 test
│                       │  Further 80/20 train/val split for training
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  CNN Training         │  5-block CNN · Adam · CategoricalCrossentropy
│                       │  Batch size: 128 · Epochs: 100
│                       │  Early stopping + LR scheduling + checkpointing
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Evaluation           │  Test accuracy: 82.58% · Test loss: 0.6461
│                       │  Prediction visualisation (green/red labels)
└───────────────────────┘
```

---

## 🧠 Model Architecture

Custom 5-block CNN with progressively doubling filters (16 → 256):

```
Input: (128, 128, 1)  ← Grayscale
│
├── Conv2D(16,  5×5, relu) → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(32,  5×5, relu) → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(64,  3×3, relu) → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(128, 3×3, relu) → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(256, 3×3, relu) → MaxPool(2×2) → Dropout(0.25)
│
├── Dense(128, relu)
├── Flatten
├── Dropout(0.25)
└── Dense(120, softmax)   ← 120-class output
```

| Parameter | Value |
|-----------|-------|
| Total layers | 19 |
| Activation (hidden) | ReLU |
| Activation (output) | Softmax |
| Loss function | Categorical Crossentropy |
| Optimizer | Adam |
| Regularization | Dropout (0.25) after every Conv block |
| Input shape | (128, 128, 1) |
| Output classes | 120 |

---

## 📈 Results

### Training (100 Epochs)

| Metric | Value |
|--------|-------|
| Final training accuracy | 79.26% |
| Final validation accuracy | 78.80% |
| **Test accuracy** | **82.58%** |
| **Test loss** | **0.6461** |

### Sample Prediction (10 test images)

| Outcome | Count |
|---------|-------|
| ✅ Correct | 7 |
| ❌ Incorrect | 3 |

> **Observation:** Accuracy improved consistently with larger training data. Validation accuracy fluctuated during training, indicating potential for improvement via data augmentation and hyperparameter tuning.

---

## 📁 Repository Structure

```
Dog-Breed-Classification-CNN/
├── 603_Dog_Breed_Classification_notebook.ipynb   # Full pipeline notebook
├── 603 - Dog Breed Classification.pdf            # Technical report
├── 603 - Dog Breed Classification - ppt.pdf      # Presentation slides
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sreenandha/Dog-Breed-Classification-CNN.git
cd Dog-Breed-Classification-CNN
```

### 2. Install dependencies
```bash
pip install pyspark tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow tqdm
```

### 3. Run the notebook
```bash
# Open in Jupyter or Google Colab (GPU recommended)
jupyter notebook 603_Dog_Breed_Classification_notebook.ipynb
```

> **Note:** The dataset (~800 MB) is downloaded automatically inside the notebook from the Stanford URL. A GPU is strongly recommended for training (100 epochs on 14,406 images).

---

## 🔮 Future Work

- [ ] Data augmentation (horizontal flip, rotation, zoom) to improve generalisation
- [ ] Transfer learning with pretrained models (ResNet50, EfficientNet, VGG16)
- [ ] Colour (RGB) inputs instead of grayscale for richer feature learning
- [ ] Hyperparameter tuning (learning rate, batch size, dropout rate)
- [ ] Grad-CAM visualisations to interpret CNN attention regions
- [ ] REST API deployment via FastAPI for real-time breed prediction

---
